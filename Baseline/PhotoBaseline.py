import pandas as pd
from Environment.Renderer import GanttChart
from Environment import SimulationEngine as Sim

import pickle
import math

def PrintLog(str, Ignore=False):
    Logmode = False
    #Logmode = False

    if Logmode | Ignore:
        print(str)

class EMRFEngine: # Existing Method in Real Fab
    def __init__(self, Env):
        self.Env = Env
        self.Arrival = self.Env.Lots.sort_values(by='Arrival Time', axis=0, ascending=True)
        self.Departure = pd.DataFrame(columns=['Departure Time', 'Number'])
        self.Machine = []
        for i in range(self.Env.MachineAttributes.shape[0]):
            self.Machine.append(Sim.Engine())

        self.ObjectValue = 0
        # self.Renderer = GanttChart((len(self.Machine), self.Env.Lots.shape[0] * 2), len(self.Machine),
        #                            len(self.Env.Stocker))
        # self.Renderer.SetTitle('Photo line GanttChart')
    def StartEnd(self):
        head = self.Arrival.head(1)
        start = head['Arrival Time'].tolist()
        tail = self.Arrival.tail(1)
        end = tail['Arrival Time'].tolist()
        return start, end
    def run(self, post=False):
        # Dashboard 출력용
        # PrintLog('===============Dash Board================')
        # PrintLog('Masks in Stocker')
        # PrintLog(self.Env.Stocker)
        # PrintLog('Masks in Machines')
        # PrintLog(self.Env.MachineAttributes['Masks'])
        # PrintLog('========================================')
        # 초기화
        self.Departure.sort_values(by='Departure Time', axis=0, ascending=True)

        # 동시간 스케줄셋 추출
        head = self.Arrival.head(1)
        val = head['Arrival Time'].tolist()
        if post:
            val = self.Departure['Departure Time'].head(1).tolist()
            PrintLog('==================Departure Time:' + str(val) + '====================')
        else:
            PrintLog('==================Arrival Time:' + str(val) + '====================')
        # Departure event
        Deplist = self.Departure[self.Departure['Departure Time'] <= val[0]]
        delidx = self.Departure[self.Departure['Departure Time'] <= val[0]].index
        self.Departure = self.Departure.drop(delidx)
        for timenum in Deplist.iterrows():
            self.Machine[timenum[1]['Number']].Departure(timenum[1]['Departure Time'])
            PrintLog('[Departure Event] Machine:' + str(timenum[1]['Number']))

        AssignSet = self.Arrival[self.Arrival['Arrival Time'] == val[0]]
        delidx = self.Arrival[self.Arrival['Arrival Time'] == val[0]].index
        self.Arrival = self.Arrival.drop(delidx)
        proctime = val[0]
        # 스케줄셋 우선순위 정렬
        AssignSet = AssignSet.sort_values(by='Priority', axis=0, ascending=True)
        for index, lot in AssignSet.iterrows():
            time = lot['Arrival Time']
            recipe = lot['Recipe']
            priority = lot['Priority']

            # 가능한 머신 탐색
            AcceptanceMachine = self.Env.Constraints['Acceptance Machine']
            number = 0
            for acceptance in AcceptanceMachine:
                # 가능한 공정조건의 설비 정의
                if recipe in acceptance:
                    # 유휴한 설비 정의
                    if not self.Machine[number].isBusy():
                        # 마스크를 가지고 있으면
                        if recipe in self.Env.MachineAttributes.loc[number, 'Masks']:
                            # 전 공정조건과 현 공정조건이 동일한 설비 정의
                            if self.Machine[number].WorkType == recipe:
                                self.ArrivalEvent('Normal', number, recipe, time, priority)
                                break
                            # 다른 모델을 처리했으나, 마스크를 가지고 있으면 Step 4-2
                            else:
                                self.ArrivalEvent('Setting', number, recipe, time, priority)
                                break
                        else:
                            # 마스크는 안가지고 있는데, 보관소에 그 마스크가 있으면 Step 4-3
                            if not self.Env.Stocker[recipe - 1] == 0:
                                # Extract mask from Stocker
                                self.Env.Stocker[recipe - 1] -= 1
                                self.ArrivalEvent('Stocker', number, recipe, time, priority)
                                break
                            else:
                                # 마스크가 타설비에 쓰고있는데, 타설비가 쉬고있으면 Step 4-4
                                Masklist = self.Env.MachineAttributes['Masks']
                                for m in range(Masklist.size):
                                    if not self.Machine[m].isBusy():
                                        if recipe in Masklist[m]:
                                            # Extract mask from other Machine
                                            Masklist[m].remove(recipe)
                                            Masklist[m].append(0)
                                            self.ArrivalEvent('Machine', number, recipe, time, priority)
                                            break
                number += 1
                if number == len(self.Machine):
                    # 모든 머신에도 들어오지 않았을 경우 다음 Arrival time으로 예약한다
                    if self.Arrival.shape[0] == 0:
                        self.Arrival.loc[0] = [recipe, time + 1, priority]
                    else:
                        head = self.Arrival.head(1)
                        val = head['Arrival Time'].tolist()
                        time = val[0]
                        newrow = {'Recipe': recipe, 'Arrival Time': time, 'Priority': priority}
                        temp1 = self.Arrival[self.Arrival['Arrival Time'] < time]
                        temp2 = self.Arrival[self.Arrival['Arrival Time'] >= time]
                        self.Arrival = temp1.append(newrow, ignore_index=True).append(temp2, ignore_index=True)
                    # PrintLog('[Assign Failed]' + 'Recipe Type: ' + str(recipe))

        return proctime
    def ArrivalEvent(self, mode, number, recipe, time, priority):
        perform = self.Env.MachineAttributes.loc[number, 'Performance']
        perform = perform[recipe-1]
        Masklist = self.Env.MachineAttributes['Masks']
        setup = 0
        toS = 0
        toM = 0
        ToStocker = 0

        if mode == 'Setting':
            setup = 5
        elif mode == 'Stocker':
            # Install mask to Machine
            temp = self.Env.MachineAttributes.loc[number, 'Masks']
            ToStocker = temp.pop(0)
            temp.append(recipe)
            self.Env.MachineAttributes.loc[number, 'Masks'] = temp

            toS = self.Env.Constraints.loc[number, 'MaskTime from S']
        elif mode == 'Machine':
            # Install mask to Machine
            ToStocker = Masklist[number].pop(0)
            Masklist[number].append(recipe)
            self.Env.MachineAttributes['Masks'] = Masklist

            toM = self.Env.Constraints.loc[number, 'MaskTime from M']

        # Store other mask to Stocker
        if not ToStocker == 0:
            self.Env.Stocker[ToStocker - 1] += 1

        nexttime = self.Machine[number].Arrival(time,
                                                [recipe,
                                                 perform + setup + toS + toM])
        NewDeparture = {'Departure Time': nexttime, 'Number': number}
        self.Departure = self.Departure.append(NewDeparture, ignore_index=True)

        self.ObjectValue += (priority * (perform + setup + toS + toM))

        PrintLog('[Arrival Event] Assign Machine:' + str(number) + ', Recipe Type: ' + str(recipe)
                 + ', SetUp: ' + str(setup) + ', Moving Time: ' + str(toS + toM)
                 + ', Departure: ' + str(nexttime))

class MTWFEngine: # Minimizing Total Weighted Flowtime
    def __init__(self, Env):
        self.Env = Env
        self.Arrival = self.Env.Lots.sort_values(by='Arrival Time', axis=0, ascending=True)
        self.Departure = pd.DataFrame(columns=['Departure Time', 'Number'])
        self.Machine = []
        for i in range(self.Env.MachineAttributes.shape[0]):
            self.Machine.append(Sim.Engine())

        self.ObjectValue = 0
        # self.Renderer = GanttChart((len(self.Machine), self.Env.Lots.shape[0] * 2), len(self.Machine),
        #                            len(self.Env.Stocker))
        # self.Renderer.SetTitle('Photo line GanttChart')

    def StartEnd(self):
        head = self.Arrival.head(1)
        start = head['Arrival Time'].tolist()
        tail = self.Arrival.tail(1)
        end = tail['Arrival Time'].tolist()
        return start, end

    def run(self, post=False):
        # 초기화
        self.Departure = self.Departure.sort_values(by='Departure Time', axis=0, ascending=True)

        # 동시간 스케줄셋 추출
        head = self.Arrival.head(1)
        val = head['Arrival Time'].tolist()
        if post:
            val = self.Departure['Departure Time'].head(1).tolist()
            PrintLog('==================Departure Time:' + str(val) + '====================')
        else:
            PrintLog('==================Arrival Time:' + str(val) + '====================')

        # Departure event
        Deplist = self.Departure[self.Departure['Departure Time'] <= val[0]]
        delidx = self.Departure[self.Departure['Departure Time'] <= val[0]].index
        self.Departure = self.Departure.drop(delidx)
        for timenum in Deplist.iterrows():
            self.Machine[timenum[1]['Number']].Departure(timenum[1]['Departure Time'])
            PrintLog('[Departure Event] Machine:' + str(timenum[1]['Number']))

        # Assign Work
        AssignSet = self.Arrival[self.Arrival['Arrival Time'] == val[0]]
        delidx = self.Arrival[self.Arrival['Arrival Time'] == val[0]].index
        self.Arrival = self.Arrival.drop(delidx)
        proctime = val[0]
        # 스케줄셋 우선순위 정렬
        AssignSet = AssignSet.sort_values(by='Priority', axis=0, ascending=True)

        # 남은 로트를 처리할 수 있는 Idle한 머신 탐색
        IdleMachine, RemainRecipe = self.__GetIdleMachine(AssignSet=AssignSet)
        # Idle한 설비중에 처리할 수 있는 Lot가 없다면 종료
        if len(IdleMachine) == 0:
            # 현재 스케줄셋을 가장 빠른 다음으로 미룬다.
            self.__ShiftSchedule(AssignSet, proctime)
            return proctime

        # Recipe Selection module RPR5(Recipe selection Rule) 적용
        RPR = [None] * len(self.Machine) # Machine마다 최적의 Recipe 결정
        for i in IdleMachine:
            RPR[i] = self.Env.Constraints.loc[i, 'Acceptance Machine'].copy()
            if self.Machine[i].WorkType in self.Env.MachineAttributes.loc[i, 'Masks']:
                if self.Machine[i].WorkType in RemainRecipe:
                    RPR[i] = [ self.Machine[i].WorkType ]
                else:
                    RPR[i].remove(self.Machine[i].WorkType)

        for i in IdleMachine:
            minTimeRecipe = [math.inf, 0]
            for recipe in RPR[i]:
                # 여러개의 레시피(한 개일수도 있음) 중에서 제일 적은 공정시간을 가지는 경우 RPR[i]를 하나의 스칼라로 추림
                performance = self.Env.MachineAttributes['Performance']
                if minTimeRecipe[0] > performance[i][recipe - 1]:
                    if not AssignSet[AssignSet['Recipe'] == recipe].shape[0] == 0:
                        minTimeRecipe = [performance[i][recipe - 1], recipe]

            # 해당 레시피에서 높은 우선순위의 로트를 추려낸다.
            if not minTimeRecipe[1] == 0:
                TargetLots = AssignSet[AssignSet['Recipe'] == minTimeRecipe[1]]
                TargetLots = TargetLots.sort_values(by='Priority', axis=0, ascending=True)
                AssignLot = TargetLots.head(1)
                idx = AssignLot.index
                AssignSet = AssignSet.drop(idx)

                # 추려낸 로트를 할당한다.
                AssignTime = AssignLot['Arrival Time'].item()
                if self.Machine[i].Tnow > AssignTime:
                    AssignTime = self.Machine[i].Tnow
                self.__ArrivalEvent(i, minTimeRecipe[1], AssignTime, AssignLot['Priority'])

        if not AssignSet.shape[0] == 0:
            # 현재 스케줄셋을 가장 빠른 다음으로 미룬다.
            self.__ShiftSchedule(AssignSet, proctime)
        return proctime

    def __ArrivalEvent(self, number, recipe, time, priority):
        perform = self.Env.MachineAttributes.loc[number, 'Performance']
        perform = perform[recipe-1]
        Masklist = self.Env.MachineAttributes['Masks']
        setup = 0
        toS = 0
        toM = 0
        ToStocker = 0

        # setup 여부
        if not self.Machine[number].WorkType == recipe:
            setup = 5

        # Mask 소지 여부
        if not recipe in Masklist[number]:
            # Stocker 소지 여부
            if not self.Env.Stocker[recipe - 1] == 0:
                self.Env.Stocker[recipe - 1] -= 1
                temp = self.Env.MachineAttributes.loc[number, 'Masks']
                ToStocker = temp.pop(0)
                temp.append(recipe)
                self.Env.MachineAttributes.loc[number, 'Masks'] = temp

                toS = self.Env.Constraints.loc[number, 'MaskTime from S']

            # Other Machine 소지 여부
            else:
                Idle = []
                Running = []
                for j in range(len(Masklist)):
                    if recipe in Masklist[j]:
                        if not self.Machine[j].isBusy():
                            Idle.append(j)
                        else:
                            Running.append(j)
                if not len(Idle) == 0:
                    # Extract mask from other Machine
                    Masklist[Idle[0]].remove(recipe)
                    Masklist[Idle[0]].append(0)
                    # Target Machine pop mask
                    ToStocker = Masklist[number].pop(0)
                    Masklist[number].append(recipe)
                    self.Env.MachineAttributes['Masks'] = Masklist

                    toM = self.Env.Constraints.loc[number, 'MaskTime from M']
                else:
                    # 가동중인 다른 머신을 기다렸다가 할당해야하는 경우
                    if not len(Running) == 0:
                        # Extract mask from other Machine
                        Masklist[Running[0]].remove(recipe)
                        Masklist[Running[0]].append(0)
                        # Target Machine pop mask
                        ToStocker = Masklist[number].pop(0)
                        Masklist[number].append(recipe)
                        self.Env.MachineAttributes['Masks'] = Masklist

                        toM = self.Env.Constraints.loc[number, 'MaskTime from M']

                        targetmachine = self.Departure[self.Departure['Number'] == Running[0]]
                        if targetmachine.shape[0] == 0:
                            Bug = True
                        if targetmachine.shape[0] > 1:
                            Bug = True
                        completetime = targetmachine['Departure Time']

                        toM = toM + (completetime.item() - time)
                    else:
                        WhatisThis = 1

        # Store other mask to Stocker
        if not ToStocker == 0:
            self.Env.Stocker[ToStocker - 1] += 1

        nexttime = self.Machine[number].Arrival(time,
                                                [recipe,
                                                 perform + setup + toS + toM])
        NewDeparture = {'Departure Time': nexttime, 'Number': number}
        self.Departure = self.Departure.append(NewDeparture, ignore_index=True)

        self.ObjectValue += (priority.item() * (perform + setup + toS + toM))

        PrintLog('[Arrival Event] Assign Machine:' + str(number) + ', Recipe Type: ' + str(recipe)
                 + ', SetUp: ' + str(setup) + ', Moving Time: ' + str(toS + toM)
                 + ', Departure: ' + str(nexttime))

    def __GetIdleMachine(self, AssignSet):
        IdleMachine = []
        RemainRecipe = AssignSet['Recipe'].tolist()
        for i in range(len(self.Machine)):
            if not self.Machine[i].isBusy():
                acceptance = self.Env.Constraints.loc[i, 'Acceptance Machine']
                for model in acceptance:
                    if model in RemainRecipe:
                        IdleMachine.append(i)
                        break
        return IdleMachine, RemainRecipe

    def __ShiftSchedule(self, batch, time):
        dep = 0
        time = time + 1
        if not self.Departure.shape[0] == 0:
            self.Departure = self.Departure.sort_values(by='Departure Time', axis=0, ascending=True)
            val = self.Departure['Departure Time'].head(1).tolist()
            dep = val[0]
            time = dep
        if not self.Arrival.shape[0] == 0:
            val = self.Arrival['Arrival Time'].head(1).tolist()
            arr = val[0]
            if arr <= dep:
                time = arr

        for _, term in batch.iterrows():
            term['Arrival Time'] = time
            temp1 = self.Arrival[self.Arrival['Arrival Time'] < time]
            temp2 = self.Arrival[self.Arrival['Arrival Time'] >= time]
            self.Arrival = temp1.append(term, ignore_index=True).append(temp2, ignore_index=True)

def EMRFRun(EMRF):
    endtime = 0
    while EMRF.Arrival.shape[0] != 0:
        EMRF.run()

    while EMRF.Departure.shape[0] != 0:
        endtime = EMRF.run(post=True)

    PrintLog('Simulation End[EMRF] [end time, object value]= ['
             + str(endtime) + ', ' + str(EMRF.ObjectValue) + ']', Ignore=True)
    for i in range(len(EMRF.Machine)):
        PrintLog('-----------------Machine[' + str(i) + ']-----------------')
        PrintLog('Acceptance model in Machine')
        PrintLog(EMRF.Env.Constraints.loc[i, 'Acceptance Machine'])
        PrintLog('Performance of Machine')
        PrintLog(EMRF.Env.MachineAttributes.loc[i, 'Performance'])
        PrintLog('Masks in Machines')
        PrintLog(EMRF.Env.MachineAttributes.loc[i, 'Masks'])
        PrintLog('[Summary]')
        # EMRF.Machine[i].Summary()
        PrintLog('---------------------------------------------------------')

    return endtime, EMRF.ObjectValue

def MTWFRun(MTWF):
    endtime = 0
    while MTWF.Arrival.shape[0] != 0:
        MTWF.run()

    while MTWF.Departure.shape[0] != 0:
        endtime = MTWF.run(post=True)

    PrintLog('Simulation End[MTWF] [end time, object value]= ['
             + str(endtime) + ', ' + str(MTWF.ObjectValue) + ']', Ignore=True)
    for i in range(len(MTWF.Machine)):
        PrintLog('-----------------Machine[' + str(i) + ']-----------------')
        PrintLog('Acceptance model in Machine')
        PrintLog(MTWF.Env.Constraints.loc[i, 'Acceptance Machine'])
        PrintLog('Performance of Machine')
        PrintLog(MTWF.Env.MachineAttributes.loc[i, 'Performance'])
        PrintLog('Masks in Machines')
        PrintLog(MTWF.Env.MachineAttributes.loc[i, 'Masks'])
        PrintLog('[Summary]')
        # MTWF.Machine[i].Summary()
        PrintLog('---------------------------------------------------------')

    return endtime, MTWF.ObjectValue

def main():
    ParamLots = [300, 500, 1000]
    ParamMachine = [5, 10, 20]
    ParamRecipe = [10, 20]
    Iteration = 10
    num = 0
    GenFlag = False
    if GenFlag:
        for LotsNum in ParamLots:
            for MachineNum in ParamMachine:
                for RecipeNum in ParamRecipe:
                    for i in range(Iteration):
                        Photo = Sim.PhotoLine(Number_Of_Machines=MachineNum,
                                              Number_Of_Kinds=RecipeNum,
                                              LotsVolume=LotsNum)
                        Photo.generateDemandSet()
                        with open('./PhotoData/ReservedEnvironment' + str(num) + '.pickle', 'wb') as f:
                            pickle.dump(Photo, f)
                        num += 1

    boundary = len(ParamMachine) * len(ParamMachine) * len(ParamRecipe) * Iteration
    ResultData = pd.DataFrame(index=range(0, boundary),
                              columns=['Number', 'Schedule Time',
                                       'EMRF Finish', 'MTWF Finish', 'EMRF Object Value', 'MTWF Object Value'])
    for i in range(boundary):
        ResultData.at[i, 'Number'] = i + 1
        with open('./PhotoData/ReservedEnvironment' + str(i) + '.pickle', 'rb') as f:
            Photo = pickle.load(f)

        # EMRF 테스트
        EMRF = EMRFEngine(Photo)
        start, end = EMRF.StartEnd()
        ResultData.at[i, 'Schedule Time'] = end
        PrintLog(str(i) + 'th Simulation Start[EMRF] [start, end]= [' + str(start) + str(end) + ']', Ignore=True)
        finish, value = EMRFRun(EMRF)

        ResultData.at[i, 'EMRF Finish'] = finish
        ResultData.at[i, 'EMRF Object Value'] = value

        # MTWF 테스트
        MTWF = MTWFEngine(Photo)
        start, end = MTWF.StartEnd()
        PrintLog(str(i) + 'th Simulation Start[MTWF] [start, end]= [' + str(start) + str(end) + ']', Ignore=True)
        finish, value = MTWFRun(MTWF)

        ResultData.at[i, 'MTWF Finish'] = finish
        ResultData.at[i, 'MTWF Object Value'] = value

    ResultData.to_csv('./ResultData.csv')
    return None

if __name__ == '__main__':
    main()
