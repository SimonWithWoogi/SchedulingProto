import numpy as np
import pandas as pd
from collections import deque


class PhotoLine:
    # 포토설비 대수, 공정조건 개수, 스케줄할 로트 수
    def __init__(self, Number_Of_Machines, Number_Of_Kinds, LotsVolume=None):
        self.__M = Number_Of_Machines
        self.__P = Number_Of_Kinds
        if LotsVolume is None:
            self.__V = 1000
        self.__V = LotsVolume
        self.Stocker = []
        self.MachineAttributes = pd.DataFrame(index=range(self.__M),
                                              columns=['Performance', 'Masks', 'MaskCapa'])
        self.Lots = pd.DataFrame(index=range(self.__V),
                                 columns=['Recipe', 'Arrival Time', 'Priority'])
        # Machine            #Stocker
        self.Constraints = pd.DataFrame(index=range(self.__M),
                                        columns=['Acceptance Machine', 'MaskTime from M', 'MaskTime from S'])

    def generateDemandSet(self, Number_Of_Machines=None, Number_Of_Kinds=None, LotsVolume=None):
        if Number_Of_Machines is not None:
            self.__M = Number_Of_Machines
        if Number_Of_Kinds is not None:
            self.__P = Number_Of_Kinds
        if LotsVolume is not None:
            self.__V = LotsVolume

        # Machine Attribute Setting
        Performance = self.__SetMachinePerformance(20, 40+1)
        MachineMaskCapa = self.__UniformRandomInt(2, 3+1, self.__M)
        MachineHasMasks = []
        for i in range(self.__M):
            MachineHasMasks.append([])
            for j in range(MachineMaskCapa[i]):
                MachineHasMasks[i].append(0)
        MachineHasMasks = np.array(MachineHasMasks)
        npstack = np.vstack((Performance, MachineHasMasks, MachineMaskCapa))
        nptr = np.transpose(npstack)
        self.MachineAttributes[:][:] = nptr

        self.Stocker = self.__UniformRandomInt(1, 2+1, self.__P)


        # Demand Setting
        Recipe = self.__SetRecipeIntoLot()
        ArrivalTime = self.__SetArrivalTime(Performance)
        Priority = self.__SetLotPriority(1, 10+1)

        npstack = np.vstack((Recipe, ArrivalTime, Priority))
        nptr = np.transpose(npstack)
        self.Lots[:][:] = nptr

        # Constraints Setting
        Acceptance_Machine = self.__SetMachineAcceptance()
        MasktimeM = self.__SetMovingTime(10, 40+1)
        MasktimeS = self.__SetMovingTime(5, 20+1)

        npstack = np.vstack((Acceptance_Machine, MasktimeM, MasktimeS))
        nptr = np.transpose(npstack)
        self.Constraints[:][:] = nptr

    def __UniformRandomInt(self, LSL, USL, size):
        distribution = 'uniform'
        value = getRandomValue(distribution=distribution,
                               param='low=' + str(LSL) + ',' +
                                     'high=' + str(USL),
                               size=size)
        value = value.astype(np.int32)
        return value
    # 공정시간 설정
    def __SetMachinePerformance(self, LSL, USL):
        MachinePerformance = []
        distribution = 'uniform'
        for i in range(self.__M):
            value = getRandomValue(distribution=distribution,
                                   param='low=' + str(LSL) + ',' +
                                         'high=' + str(USL),
                                   size=1)
            value = value.astype(np.int32)
            MachinePerformance.append(value.item(0))
        return np.array(MachinePerformance)
    # 공정조건 설정(recipe or model type)
    def __SetRecipeIntoLot(self):
        distribution = 'uniform'
        Recipe = getRandomValue(distribution=distribution,
                                param='low=' + str(1) + ',' +
                                      'high=' + str(self.__P),
                                size=self.__V)
        return Recipe.astype(np.int32)

    # 각 로트의 도착시간
    def __SetArrivalTime(self, Performance):
        distribution = 'uniform'
        MinTime = Performance.tolist().copy()
        MinTime.sort()
        USL = (MinTime[0] * self.__V) / self.__M
        ArrivalTime = getRandomValue(distribution=distribution,
                                     param='low=' + str(0) + ',' +
                                           'high=' + str(USL),
                                     size=self.__V)
        return ArrivalTime.astype(np.int32)

    # 각 로트의 중요도
    def __SetLotPriority(self, LSL, USL):
        distribution = 'uniform'
        Priority = getRandomValue(distribution=distribution,
                                  param='low=' + str(LSL) + ',' +
                                        'high=' + str(USL),
                                  size=self.__V)
        return Priority.astype(np.int32)

    # 각 머신의 공정조건
    def __SetMachineAcceptance(self):
        distribution = 'uniform'
        MachineAcceptance = []
        for i in range(self.__M):
            MachineAcceptance.append([])

        for i in range(1, self.__P + 1):
            Many = getRandomValue(distribution='uniform',
                                  param='low=' + str(1) + ',' +
                                        'high=' + str(self.__M),
                                  size=1)
            Many = Many.astype(np.int32)
            Acceptance = np.random.choice(range(0, self.__M), Many.item(0), replace=False)
            Acceptance = Acceptance.tolist()
            for j in Acceptance:
                MachineAcceptance[j].append(i)
        return np.array(MachineAcceptance)

    # 각 머신의 mask 이동시간
    def __SetMovingTime(self, LSL, USL):
        distribution = 'uniform'
        MovingTime = getRandomValue(distribution=distribution,
                                    param='low=' + str(LSL) + ',' +
                                          'high=' + str(USL),
                                    size=self.__M)
        return np.around(MovingTime).astype(np.int32)


class Engine:
    def __init__(self):
        self.Q = deque()
        self.Tnow = 0
        self.WorkType = -1
        self.TotalProduction = 0
        self.AverageWaitingTime = 0
        self.MaximumWaitingTime = 0
        self.TimeAverageNumberOfPartsInQ = 0
        self.MaximumNumberOfPartsInQ = 0
        self.CycleTime = 0
        self.Utilization = 0

        self.__Bt = 0
        self.__AreaUnderQt = 0
        self.__QTime = deque()
        self.__TotalWaitingTime = 0
        self.__NextDeparture = 0
        self.__WorkType = 0
        self.__TotalSystemTime = 0

    def Arrival(self, time, state):
        self.__AreaUnderQt += len(self.Q) * (time - self.Tnow)
        self.Tnow = time
        self.__QTime.append(self.Tnow)
        self.Q.append(state)
        if len(self.Q) > self.MaximumNumberOfPartsInQ:
            self.MaximumNumberOfPartsInQ = len(self.Q)
        return self.Process()

    def Process(self):
        if self.__Bt == 0:
            self.__Bt = 1
            WaitingTime = self.Tnow - self.__QTime.pop()
            self.__TotalWaitingTime += WaitingTime

            if self.MaximumWaitingTime < WaitingTime:
                self.MaximumWaitingTime = WaitingTime

            state = self.Q.pop()
            self.WorkType = state[0]
            ServiceTime = state[1]
            self.__NextDeparture = self.Tnow + ServiceTime
            self.Utilization += ServiceTime
            self.__TotalSystemTime += (WaitingTime + ServiceTime)
        return self.__NextDeparture

    def Departure(self, time):
        self.__AreaUnderQt += len(self.Q) * (time - self.Tnow)
        #self.Utilization += time - self.Tnow
        self.Tnow = time
        self.TotalProduction += 1
        self.__Bt = 0
        if len(self.Q) > 0:
            return self.Process()
        return 0
    def Summary(self):
        self.AverageWaitingTime = self.__TotalWaitingTime / self.TotalProduction
        self.TimeAverageNumberOfPartsInQ = self.__AreaUnderQt / self.Tnow
        self.CycleTime = self.__TotalSystemTime / self.TotalProduction
        self.Utilization = self.Utilization / self.Tnow
        print('Average Waiting Time: ' + str(self.AverageWaitingTime))
        print('Time Average Number Of Parts in Q: ' + str(self.TimeAverageNumberOfPartsInQ))
        print('Cycle Time: ' + str(self.CycleTime))
        print('Utilization: ' + str(self.Utilization))
        print('Total Production: ' + str(self.TotalProduction))

    def isBusy(self):
        return self.__Bt

def getRandomValue(distribution, param, size):
    value = eval('np.random.' + distribution +
                 '(' + param + ', size=' + str(size) + ')')
    np.random.uniform()
    return value


def main():
    Sim = PhotoLine(Number_Of_Machines=20, Number_Of_Kinds=20, LotsVolume=500)
    Sim.generateDemandSet()


if __name__ == '__main__':
    main()
