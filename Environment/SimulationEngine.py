import numpy as np
import pandas as pd
class PhotoLine:
    # 포토설비 대수, 공정조건 개수, 스케줄할 로트 수
    def __init__(self, Number_Of_Machines, Number_Of_Kinds, LotsVolume=None):
        self.__M = Number_Of_Machines
        self.__P = Number_Of_Kinds
        if LotsVolume is None:
            self.__V = 1000
        self.__V = LotsVolume
        self.MachinePerformance = []
        self.Lots = pd.DataFrame(index=range(self.__V), columns=['Recipe', 'Arrival Time', 'Priority'])
                                                                                                #Machine            #Stocker
        self.Constraints = pd.DataFrame(index=range(self.__M), columns=['Acceptance Recipe', 'MaskTime from M', 'MaskTime from S'])

    def generateDemandSet(self, Number_Of_Machines = None, Number_Of_Kinds = None, LotsVolume = None):
        if Number_Of_Machines is not None:
            self.__M = Number_Of_Machines
        if Number_Of_Kinds is not None:
            self.__P = Number_Of_Kinds
        if LotsVolume is not None:
            self.__V = LotsVolume

        # Machine Attribute Setting
        self.__SetMachinePerformance(20, 40)
        # Demand Setting
        Recipe = self.__SetRecipeIntoLot()
        ArrivalTime = self.__SetArrivalTime()
        Priority = self.__SetLotPriority(1, 10)

        npstack = np.vstack((Recipe, ArrivalTime, Priority))
        nptr = np.transpose(npstack)
        self.Lots[:][:] = nptr

        # Constraints Setting
        Acceptance_Recipe = self.__SetMachineAcceptance()
        MasktimeM = self.__SetMovingTime(10, 40)
        MasktimeS = self.__SetMovingTime(5, 20)

        npstack = np.vstack((Acceptance_Recipe, MasktimeM, MasktimeS))
        nptr = np.transpose(npstack)
        self.Constraints[:][:] = nptr

        ['Acceptance Recipe', 'MaskTime from M', 'MaskTime from S']
    # 공정시간 설정
    def __SetMachinePerformance(self, LSL, USL):
        self.MachinePerformance = []
        distribution = 'uniform'
        for i in range(self.__M):
            value = getRandomValue(distribution=distribution,
                                   param='low=' + str(LSL) + ',' +
                                         'high=' + str(USL),
                                   size=1)
            self.MachinePerformance.append(np.around(value))
    # 공정조건 설정(recipe or model type)
    def __SetRecipeIntoLot(self):
        distribution = 'uniform'
        Recipe = getRandomValue(distribution=distribution,
                                param='low=' + str(1) + ',' +
                                      'high=' + str(self.__P),
                                size=self.__V)
        return np.around(Recipe)
    # 각 로트의 도착시간
    def __SetArrivalTime(self):
        distribution = 'uniform'
        MinTime = self.MachinePerformance.copy()
        MinTime.sort()
        USL = (MinTime[0] * self.__V) / self.__M
        ArrivalTime = getRandomValue(distribution=distribution,
                                     param='low=' + str(0) + ',' +
                                           'high=' + str(USL),
                                     size=self.__V)
        return np.around(ArrivalTime)
    # 각 로트의 중요도
    def __SetLotPriority(self, LSL, USL):
        distribution = 'uniform'
        Priority = getRandomValue(distribution=distribution,
                                  param='low=' + str(LSL) + ',' +
                                        'high=' + str(USL),
                                  size=self.__V)
        return np.around(Priority)
    # 각 머신의 공정조건
    def __SetMachineAcceptance(self):
        distribution = 'uniform'
        MachineAcceptance = []
        Checker = range(1, self.__P)
        while
        for i in range(self.__M):
            Many = getRandomValue(distribution='uniform',
                                  param='low=' + str(1) + ',' +
                                        'high=' + str(self.__P),
                                  size=1)
            Many = Many.astype(np.int32)
            Acceptance = np.random.choice(range(1, self.__P + 1), Many.item(0), replace=False)
            Acceptance = Acceptance.tolist()
            MachineAcceptance.append(Acceptance)
        return np.array(MachineAcceptance)
    # 각 머신의 mask 이동시간
    def __SetMovingTime(self, LSL, USL):
        distribution = 'uniform'
        MovingTime = getRandomValue(distribution=distribution,
                                    param='low=' + str(LSL) + ',' +
                                          'high=' + str(USL),
                                    size=self.__M)
        return np.around(MovingTime)

def getRandomValue(distribution, param, size):
    value = eval('np.random.' + distribution +
                 '(' + param + ', size='+str(size) + ')')
    np.random.uniform()
    return value

def main():
    Sim = PhotoLine(Number_Of_Machines=20, Number_Of_Kinds=20, LotsVolume=500)
    Sim.generateDemandSet()
if __name__ == '__main__':
    main()