import pandas as pd
import numpy as np
import pickle
import random
import math
import time

import JobSchedule
from JobSchedule import Env
from DQNAgent import Agent

EPIOSDE = 3000
ACTION_CHANCE = 1
TIME_BOUNDARY = 4

def pre_processing(Row, Features):
    State = []
    for n in range(Row):
        FeatureSet = []
        for m in range(Features):
            FeatureSet.append(0)
        State.append(FeatureSet)
    return State

def main():
    with open('Params.p', 'rb') as file:
        Params = pickle.load(file)
    State_name = {'Facility time', 'Last type', 'Start Time', 'Demand Time',
                  'Demand DueDate', 'Demand Type', 'SetUp Warning'}
    ScheduleTable = Env()
    dummy = pre_processing(Params.MachinesNumber(), len(State_name))
    history = np.stack((dummy, dummy, dummy, dummy), axis=2)
    history = np.reshape([history],
                         (1, Params.MachinesNumber(), len(State_name), TIME_BOUNDARY))
    agent = Agent(action_size=Params.MachinesNumber(),
                  state_size=(Params.MachinesNumber(), len(State_name), TIME_BOUNDARY))

    start_time = time.time()
    action_time = time.time()
    global_steps = 0

    print('Beginning Episode')
    run = True
    for epi in range(EPIOSDE):
        episode = ScheduleTable.Reset('./DemandSet/DemandStatement' + str(epi + 1) + '.csv')
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
                reward, is_new_block = ScheduleTable.step(action)
                action_change_chance -= 1


if __name__ == '__main__':
    main()