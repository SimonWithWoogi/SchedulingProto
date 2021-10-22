import argparse
import gym
import os

import ray
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy, DQNTorchPolicy
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy, PPOTorchPolicy
from ray.rllib.agents.a3c import A3CTrainer, a3c_tf_policy, a3c_torch_policy
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

class Trainer:
    def __init__(self, Env):
        parser = argparse.ArgumentParser()
        # Use torch for both policies.
        parser.add_argument(
            "--framework",
            choices=["tf", "tf2", "tfe", "torch"],
            default="tf",
            help="The DL framework specifier.")
        parser.add_argument(
            "--as-test",
            action="store_true",
            help="Whether this script should be run as a test: --stop-reward must "
                 "be achieved within --stop-timesteps AND --stop-iters.")
        parser.add_argument(
            "--stop-iters",
            type=int,
            default=40,
            help="Number of iterations to train.")
        parser.add_argument(
            "--stop-timesteps",
            type=int,
            default=100000,
            help="Number of timesteps to train.")
        parser.add_argument(
            "--stop-reward",
            type=float,
            default=50.0,
            help="Reward at which we stop training.")

        self.Env = Env
        self.__args = parser.parse_args()
        self.__obs_space = self.Env.observation_space
        self.__act_space = self.Env.action_space
        self.Trainers = dict()

    def __getPolicies(self, algorithms):
        torch = {"ppo": PPOTorchPolicy, "dqn": DQNTorchPolicy, "a3c": a3c_torch_policy.A3CTorchPolicy}
        tf = {"ppo": PPOTFPolicy, "dqn": DQNTFPolicy, "a3c": a3c_tf_policy.A3CTFPolicy}
        policies = dict()
        for algo in algorithms:
            policy = (torch[algo] if self.__args.framework == "torch" else
                      tf[algo], self.__obs_space, self.__act_space, {})
            policies[algo+'_policy'] = policy
        return policies

    def Instance(self, algorithms, env = None):
        if env is not None:
            self.Env = env
        Policies = self.__getPolicies(algorithms)
        trainers = {"ppo": 'PPOTrainer', "dqn": 'DQNTrainer', "a3c": 'A3CTrainer'}
        configs = {
            "ppo":
            {"multiagent": {
                "policies": Policies,
                "policy_mapping_fn": self.policy_mapping_fn,
                "policies_to_train": ["ppo_policy"],
            },
                "model": {
                    "vf_share_layers": True,
                },
                "num_sgd_iter": 6,
                "vf_loss_coeff": 0.01,
                # disable filters, otherwise we would need to synchronize those
                # as well to the DQN agent
                "observation_filter": "MeanStdFilter",
                # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                "framework": self.__args.framework,
                "render_env": True},
            "dqn":
                {"multiagent": {
                    "policies": Policies,
                    "policy_mapping_fn": self.policy_mapping_fn,
                    "policies_to_train": ["dqn_policy"],
                },
                    "model": {
                        "vf_share_layers": True,
                    },
                    "gamma": 0.95,
                    "n_step": 3,
                    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "framework": self.__args.framework,
                    "render_env": True,
                },
            "a3c":
                {"multiagent": {
                    "policies": Policies,
                    "policy_mapping_fn": self.policy_mapping_fn,
                    "policies_to_train": ["a3c_policy"],
                },
                    "model": {
                        "vf_share_layers": True,
                    },
                    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "framework": self.__args.framework,
                    "render_env": True}
        }
        for algo in algorithms:
            # config = configs[algo]
            # if algo == 'ppo':
            #     ppo_trainer = PPOTrainer(env=self.Env, config=config)
            #     self.Trainers[algo] = ppo_trainer
            # elif algo is 'dqn':
            #     dqn_trainer = DQNTrainer(env=self.Env, config=config)
            #     self.Trainers[algo] = dqn_trainer
            # elif algo is 'a3c':
            #     self.Trainers[algo] = A3CTrainer(env=self.Env, config=config)
            cmd = trainers[algo] + '(env = self.Env, config = configs[algo])'
            trainer = eval(cmd)
            self.Trainers[algo] = trainer


    def policy_mapping_fn(self, agent_id, episode, **kwargs):
        if agent_id % 2 == 0:
            return "ppo_policy"
        else:
            return "dqn_policy"

    def Train(self):
        for i in range(self.__args.stop_iters):
            if len(self.Trainers) == 0:
                return True
            print("== Iteration", i, "==")
            for algo, trainer in self.Trainers.items():
                print("--[" + algo + "]--")
                result = trainer.train()
                print(pretty_print(result))

                # Test passed gracefully.
                if self.__args.as_test and result["episode_reward_mean"] > self.__args.stop_reward:
                    print("test passed (" + algo + ")")
                    self.Trainers.pop(algo)
                # swap weights to synchronize
                trainer.set_weights(trainer.get_weights([algo + "_policy"]))

        # Desired reward not reached.
        if self.__args.as_test:
            raise ValueError("Desired reward ({}) not reached!".format(
                self.__args.stop_reward))

if __name__ == "__main__":

    ray.init()

    # Simple environment with 4 independent cartpole entities
    register_env("multi_agent_cartpole",
                 lambda _: MultiAgentCartPole({"num_agents": 4}))
    single_dummy_env = gym.make("CartPole-v0")

    algorithms = ['ppo', 'dqn']
    Runner = Trainer(single_dummy_env)
    Runner.Instance(algorithms=algorithms, env="multi_agent_cartpole")
    Runner.Train()