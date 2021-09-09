import pandas as pd
import numpy as np
import pickle
import random
import math
import time

import JobSchedule
from JobSchedule import Env
from DQNAgent import Agent

EPIOSDE = 1000000
TIME_BOUNDARY = 4

def pre_processing(oldState, type = 0, quantity = 0, duedate = 0):
    State = []
    for single in oldState:
        Facility_time = int(single[0])
        Last_type = int(single[1])
        Start_time = int(single[2])
        Demand_Quantity = quantity
        Demand_DueDate = duedate
        Demand_Type = type
        Violation = int(single[7])
        Machine_id = int(single[8])
        if Last_type != type & Last_type != 0:
            SetUp_Warning = 1
        else:
            SetUp_Warning = 0
        State.append([Facility_time, Last_type, Start_time, Demand_Quantity,
                      Demand_DueDate, Demand_Type, SetUp_Warning, Violation, Machine_id])
    return State

def main():
    with open('Params.p', 'rb') as file:
        Params = pickle.load(file)
    State_name = ['Facility time', 'Last type', 'Start Time', 'Demand Quantity',
                  'Demand DueDate', 'Demand Type', 'SetUp Warning', 'Violation Time', 'Machine Id']
    ScheduleTable = Env()
    agent = Agent(action_size=Params.MachinesNumber(),
                  state_size=(168, 168, TIME_BOUNDARY))
                  #state_size=(Params.MachinesNumber(), len(State_name), TIME_BOUNDARY))

    start_time = time.time()
    action_time = time.time()
    global_steps = 0

    print('Beginning Episode')
    run = True
    for epi in range(EPIOSDE):
        allreward = 0
        #episode = ScheduleTable.Reset(Path='./DemandSet/DemandStatement' + str(epi + 1) + '.csv')
        episode = ScheduleTable.Reset(Path='./DemandSet/DemandStatement' + str(0 + 1) + '.csv')

        OutputTable = pd.DataFrame(columns=['Demand Id', 'Machine Id', 'Type',
                                            'Processing Time', 'Start Time', 'Complete Time',
                                            'Due date', 'Set-Up', 'Violation Time'])
        state = ScheduleTable.getStateImage((168, 168))
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 168, 168, 4))
        statelog = pre_processing(np.zeros((Params.MachinesNumber(), len(State_name))))
        #history = np.stack((state, state, state, state), axis=2)
        #history = np.reshape([history],
        #                     (1, Params.MachinesNumber(), len(State_name), TIME_BOUNDARY))
        for index, step in episode.iterrows():
            end_time = time.time()

            # Action select
            global_steps += 1
            action = 0
            reward = None
            is_new_block = None

            # Demand pop
            type = ord(step.Type) - 64
            quantity = step.Quantity
            duedate = step.DueDate

            statelog = pre_processing(statelog, type=type,
                                   quantity=math.ceil(quantity / Params.MachineCapa()), duedate=duedate)
            ScheduleTable.ScheduleView.UpdateDemand(time=math.ceil(quantity / Params.MachineCapa()),
                                                    duedate=duedate, type=type)

            action = agent.get_action(history=history, statelog=statelog)
            # 환경에 적용 후 다음 state로 적용
            statelog, reward, OutputTable = ScheduleTable.step(action=action, state=statelog, table=OutputTable)
            OutputTable.iloc[index, 2] = step.Type
            next_state = ScheduleTable.getStateImage((168, 168))
            next_state = np.reshape([next_state], (1, 168, 168, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            #agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / history.max(axis=1)))[0])
            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])
            agent.append_sample(history, action, reward, next_history)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_steps % agent.update_target_rate == 0:
                agent.update_target_model()

            history = next_history
            action_time = time.time()

        print('episode:{}, score:{}, epsilon:{}, global step:{}, avg_qmax:{}, memory:{}'.
              format(epi, ScheduleTable.score, agent.epsilon, global_steps,
                     agent.avg_q_max / float(index), len(agent.memory)))
        agent.avg_q_max, agent.avg_loss = 0, 0
        agent.model.save_weights("./save_model/breakout_dqn_1.h5")
        OutputTable.to_csv('./DataDigestTables/ScheduleTable' + str(epi + 1) + '(' +
                           str(math.ceil(ScheduleTable.score)) + ').csv', index=False)
        start_time = time.time()

if __name__ == '__main__':
    main()