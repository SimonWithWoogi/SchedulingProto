import pandas as pd
import numpy as np
import pickle
import random
import math

def LoadEpisode(Path):
    DataSet = pd.read_csv(Path)
    return DataSet

def RemainingObservation(test):
    # 각 머신 별로 남아있는 시간을 돌려준다.
    return None

def AssignMachine(number, Setup, Demand, DueDate, ScheduleTable): # 액션에 해당
    # 머신 넘버와 함께 demand 는 배열 넘어온다.
    # Schedule table에 반영 후 리턴
    # Setup 여부와 Violation time도 함께 리턴

    AssignTable = ScheduleTable.loc[ScheduleTable['Machine Id'] == number]

    oldctime = 0
    if not AssignTable.shape[0] == 0:
        target = AssignTable.tail(1)
        oldctime = np.array(target['Complete Time'])
        oldctime = oldctime[0]

    demandid = ScheduleTable.shape[0] + 1
    proctime = Demand
    stime = oldctime + Setup
    ctime = stime + proctime
    violation = ctime - DueDate
    if violation < 0:
        violation = 0

    rtnDict = {'Demand Id': demandid, 'Machine Id': number, 'Type': '',
                'Processing Time': Demand, 'Start Time': stime, 'Complete Time': ctime,
                'Due date': DueDate, 'Set-Up': bool(Setup), 'Violation Time': violation}

    return ScheduleTable.append(rtnDict , ignore_index=True), violation

def RewardFunction(setup, violation):
    Reward = 1 - (5 * violation)
    if setup:
        Reward = Reward - 2
    return Reward

def getActionRand(N):
    return random.randrange(0, N)
def getActionUniform(idx, N):
    idx = idx + 1
    if idx >= N:
        idx = 0
    return idx
def getActionMinMachine(state):
    idx = 0
    rtn = idx
    lowvalue = math.inf
    for machine, dummy in state:
        if lowvalue > machine:
            lowvalue = machine
            rtn = idx
        idx = idx + 1
    return rtn

def main():
    with open('Params.p', 'rb') as file:
        Params = pickle.load(file)

    Episodes = 3000
    SJFTotalReword = []
    FIFOTotalReword = []
    RandTotalReword = []
    for week in range(0, Episodes):
        AllReward = 0
        # State 용 Utilization
        State = []
        for machine in range(Params.MachinesNumber()):
            # 누적 시간, 마지막 제품명(알파벳을 스칼라로)
            State.append([0, 0])
        episode = LoadEpisode('./DataSet/DemandStatement' + str(week + 1) + '.csv')
        # ScheduleTable 생성 - 이게 곧 환경이 된다.
        #index = range(0, episode.shape[0]),
        ScheduleTable = pd.DataFrame(columns=['Demand Id', 'Machine Id', 'Type',
                                              'Processing Time', 'Start Time', 'Complete Time',
                                              'Due date', 'Set-Up', 'Violation Time'])
        Filenames = {0: 'SJFTable', 1: 'FIFOTable', 2: 'RandTable'}
        for key in range(3):
            tempidx = Params.MachinesNumber()
            # --------- 여기서부터 전부 temp
            AllReward = 0
            # State 용 Utilization
            State = []
            for machine in range(Params.MachinesNumber()):
                # 누적 시간, 마지막 제품명(알파벳을 스칼라로)
                State.append([0, 0])
            episode = LoadEpisode('./DataSet/DemandStatement' + str(week + 1) + '.csv')
            # ScheduleTable 생성 - 이게 곧 환경이 된다.
            # index = range(0, episode.shape[0]),
            ScheduleTable = pd.DataFrame(columns=['Demand Id', 'Machine Id', 'Type',
                                                  'Processing Time', 'Start Time', 'Complete Time',
                                                  'Due date', 'Set-Up', 'Violation Time'])
            # --------- 여기까지 전부 temp
            for index, step in episode.iterrows():
                type = ord(step.Type) - 64
                quantity = step.Quantity

                # 모델에 해당 두 벡터를 넣고 state를 확인하면 action(어떤 머신에 넣을 것인지)를 알 수 있다.
                # 그리고 보상 그러나 모듈 테스트 먼저

                # baseline action
                if key == 0:
                    action = getActionMinMachine(State)
                elif key == 1:
                    tempidx = getActionUniform(tempidx, Params.MachinesNumber())
                    action = tempidx
                else:
                    action = getActionRand(Params.MachinesNumber())

                # 환경에 반영
                setup = 0
                Mstate = State[action]
                if Mstate[1] != type & Mstate[1] != 0:
                    setup = Params.SetUpTime()

                ScheduleTable, Violation = AssignMachine(action, setup,
                                                         math.ceil(quantity / Params.MachineCapa()),
                                                         step.DueDate, ScheduleTable)
                ScheduleTable.iloc[index, 2] = step.Type
                Mstate = [Mstate[0] + math.ceil(quantity / Params.MachineCapa()), type]
                State[action] = Mstate
                AllReward = AllReward + RewardFunction(bool(setup),Violation)

            if key == 0:
                SJFTotalReword.append(AllReward)
            elif key == 1:
                FIFOTotalReword.append(AllReward)
            else:
                RandTotalReword.append(AllReward)
            ScheduleTable.to_csv('./ScheduleTables/' + str(week+1) + Filenames[key] + str(AllReward) + '.csv', index=False)
    # Save
    A = pd.Series(SJFTotalReword, name='SJF reward')
    B = pd.Series(FIFOTotalReword, name='FIFO reward')
    C = pd.Series(RandTotalReword, name='Rand reward')
    SaveData = pd.concat([A, B, C], axis=1)
    SaveData.to_csv('./Reward/RewardSet.csv', index=False)




if __name__ == '__main__':
    main()