import pickle
import pandas as pd
import numpy as np
from Renderer import Draw
class Env():
    def __init__(self):
        with open('Params.p', 'rb') as file:
            self.Params = pickle.load(file)
        super(Env, self).__init__()
        self.score = 0
        self.ScheduleView = Draw()

    def Reset(self, Path):
        DataSet = pd.read_csv(Path)
        self.score = 0
        self.ScheduleView.ResetBackboard()
        return DataSet

    def step(self, action, state, table):
        State_name = ['Facility time', 'Last type', 'Start Time', 'Demand Quantity',
                      'Demand DueDate', 'Demand Type', 'SetUp Warning', 'Violation Time', 'Machine Id']
        Mstate = state[action]
        self.ScheduleView.render()
        table, violation, Mstate = self.AssignMachine(number=action, state=Mstate, Output=table)
        reward = self.RewardFunction(setup=bool(Mstate[6]), violation=violation, workingtime=Mstate[0])
        state[action] = Mstate
        self.score += reward
        return state, reward, table

    def AssignMachine(self, number, state, Output):  # 액션에 해당
        # 머신 넘버와 함께 demand 는 배열 넘어온다.
        # Schedule table에 반영 후 리턴
        # Setup 여부와 Violation time도 함께 리턴

        AssignTable = Output.loc[Output['Machine Id'] == number]

        oldctime = 0
        if not AssignTable.shape[0] == 0:
            target = AssignTable.tail(1)
            oldctime = np.array(target['Complete Time'])
            oldctime = oldctime[0]

        demandid = Output.shape[0] + 1
        proctime = state[3]
        stime = oldctime + state[6]
        ctime = stime + proctime
        violation = ctime - state[4]
        if violation < 0:
            violation = 0

        rtnDict = {'Demand Id': demandid, 'Machine Id': number, 'Type': '',
                   'Processing Time': proctime, 'Start Time': stime, 'Complete Time': ctime,
                   'Due date': state[4], 'Set-Up': bool(state[6]), 'Violation Time': violation}
        state[0] += proctime
        state[1] = state[5]
        state[2] = stime
        state[7] += violation
        state[8] = number

        self.ScheduleView.UpdateSchedule(machine=number, time=proctime, setup=bool(state[6]),
                                         violation=violation, type=state[5])
        self.ScheduleView.render()
        return Output.append(rtnDict, ignore_index=True), violation, state
    def RewardFunction(self, setup, violation, workingtime):
        alltime = self.Params.LimitationTime()
        Reward = ((alltime - workingtime) / alltime) * 100 - (50 * violation)
        if setup:
            Reward = Reward - 30
        return Reward

    def getStateImage(self, size):
        return self.ScheduleView.getBackBoardImage(size)