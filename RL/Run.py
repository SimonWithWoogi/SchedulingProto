from Environment import Photolithography, PhotolithographyV2

import argparse
import gym
import numpy as np
import os
import random

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents import a3c
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="A3C",
    help="The RLlib-registered algorithm to use.")
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
    default=3000,
    help="Number of iterations to train.")
# parser.add_argument(
#     "--stop-timesteps",
#     type=int,
#     default=100000,
#     help="Number of timesteps to train.")
# parser.add_argument(
#     "--stop-reward",
#     type=float,
#     default=10000,
#     help="Reward at which we stop training.")
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.")
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.")

if __name__ == '__main__':

    # ParamLots = [300, 500, 1000]
    # ParamMachine = [5, 10, 20]
    # ParamRecipe = [10, 20]
    ParamLots = [300]
    ParamMachine = [10]
    ParamRecipe = [20]
    checkpoint_path = None

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode)

    stop = {
        "training_iteration": args.stop_iters,
    }

    for LotsNum in ParamLots:
        for MachineNum in ParamMachine:
            for RecipeNum in ParamRecipe:
                config = {
                    "env": PhotolithographyV2,
                    "env_config": {"Number of Machines": MachineNum,
                                   "Number of Kinds": RecipeNum,
                                   "Lots Volume": LotsNum},
                    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
                    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
                    "num_workers": 1,  # parallelism
                    "framework": args.framework,
                    "render_env": False
                }

                a3c_config = a3c.DEFAULT_CONFIG.copy()
                a3c_config.update(config)
                trainer = a3c.A3CTrainer(config=a3c_config, env=PhotolithographyV2)

                for _ in range(args.stop_iters):
                    result = trainer.train()
                    print(pretty_print(result))


    ray.shutdown()