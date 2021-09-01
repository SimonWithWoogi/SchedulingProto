import pandas as pd
import numpy as np
import pickle
import random
import math
import time

from JobSchedule import Env
from DQNAgent import Agent

EPIOSDE = 3000
ACTION_CHANCE = 1
def LoadEpisode(Path):
    DataSet = pd.read_csv(Path)
    return DataSet

def RemainingObservation(test):
    # 각 머신 별로 남아있는 시간을 돌려준다.
    return None

def main():
    with open('Params.p', 'rb') as file:
        Params = pickle.load(file)

    ScheduleTable = Env()

    agent = Agent(action_size=Params.MachinesNumber(),
                  state_size=(Params.MachinesNumber(), Params.MachinesNumber()))
    # State - 누적 시간, 마지막 제품명(알파벳을 스칼라로), start time,

    rtnDict = {'Demand Id': demandid, 'Machine Id': number, 'Type': '',
               'Processing Time': Demand, 'Start Time': stime, 'Complete Time': ctime,
               'Due date': DueDate, 'Set-Up': bool(Setup), 'Violation Time': violation}

    state = np.zeros((Params.MachinesNumber(), 2))
    history = np.stack((state, state), axis=1)
    history = np.reshape([history],
                         (1, Params.MachinesNumber(), Params.MachinesNumber))

    start_time = time.time()
    action_time = time.time()
    global_steps = 0

    print('Beginning Episode')
    run = True
    for epi in range(EPIOSDE):
        step = 0
        action_change_chance = ACTION_CHANCE
        while run:
            end_time = time.time()

            # Action select
            global_steps += 1
            step += 1
            action = 0
            reward = None
            is_new_block = None

            if action_change_chance > 0:
                action = agent.get_action(history)
                reward, is_new_block = tetris.step(action)
                action_change_chance -= 1
if __name__ == '__main__':
    main()