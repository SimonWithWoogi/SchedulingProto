import RendererV3
from Parameters import PreSetInfo

import numpy as np
import pandas as pd
import math
import csv
from collections import deque, OrderedDict
import copy
import gym
from gym.spaces import Discrete, Box, Dict
from ray.rllib.env.env_context import EnvContext

INF = 30000


def getParams(M=None, P=None, maxQ=None, minQ=None, maxT=None,
              setuptime=None, maxalltime=None, capacity=None,
              distribution=None):
    # Number of Machines in single line         =   M
    # Number of Kinds about Products            =   P
    # Max order quantity in a requirement       =   MaxQ
    # Min order quantity in a requirement       =   MinQ
    # Max due date about a requirement          =   MaxT
    # Set-up time(model change)                 =   setuptime
    # Max time of a single demand statements    =   maxalltime
    # A single machine capacity                 =   capacity

    if M is None:
        M = 17
    if P is None:
        P = 12
    if maxQ is None:
        maxQ = 8000
    if minQ is None:
        minQ = 100
    if maxT is None:
        maxT = 12
    if setuptime is None:
        setuptime = 1
    if maxalltime is None:
        maxalltime = 168
    if capacity is None:
        capacity = 1000
    if distribution is None:
        distribution = "Uniform"

    PreSet = PreSetInfo(M, P, maxQ, minQ, maxT, setuptime, maxalltime, capacity, distribution)
    return PreSet


class Photolithography(gym.Env):
    def __init__(self, config: EnvContext):
        params = getParams()
        self.Params = params
        self.View = RendererV3.Draw(Params=params)
        self.File = pd.DataFrame(columns=['Demand Id', 'Machine Id', 'Type',
                                          'Processing Time', 'Start Time', 'Complete Time',
                                          'Due date', 'Set-Up', 'Violation Time'])

        self.Demand = None
        self.obs = self.getRawObservation()
        self.score = 0
        self.action_space = Discrete(params.MachinesNumber())
        self.observation_space = self.getObservationSpace()
        # self.seed(config.worker_index * config.num_workers)

    def reset(self):
        self.View.ResetBackboard()
        self.File = pd.DataFrame(columns=['Demand Id', 'Machine Id', 'Type',
                                          'Processing Time', 'Start Time', 'Complete Time',
                                          'Due date', 'Set-Up', 'Violation Time'])
        self.score = 0
        path = './../DemandSet/DemandStatement' + str(0 + 1) + '.csv'
        self.Demand = pd.read_csv(path)
        self.obs = self.getRawObservation()
        self.ApplyDemand()

        return self.DictToList()

    def step(self, action):
        reward = self.AssignMachine(number=action)
        self.score += reward
        # Demand가 다 떨어지면 됨
        done = False
        if self.Demand.shape[0] == 0:
            done = True
        else:
            self.ApplyDemand()
        return self.DictToList(), reward, done, {}

    def saveFile(self, name):
        self.File.to_csv('./DataDigestTables/ScheduleTable' + name + '(' +
                         str(math.ceil(self.score)) + ').csv', index=False)
        return self.File

    def RewardFunction(self, action):

        state = self.obs['Machine_' + str(action)]

        alltime = self.Params.LimitationTime()
        reward = ((alltime - state['Facility Time']) / alltime) * 10 \
                 - (5 * state['Violation Time'])
        if bool(state['SetUp Warning']):
            reward = reward - 3
        return reward

    def AssignMachine(self, number):  # 액션에 해당
        AssignTable = self.File.loc[self.File['Machine Id'] == number]

        demand = self.obs['Demand']
        state = self.obs['Machine_' + str(number)]
        oldctime = 0
        if not AssignTable.shape[0] == 0:
            target = AssignTable.tail(1)
            oldctime = np.array(target['Complete Time'])
            oldctime = oldctime[0]
            oldtype = target.Type
        else:
            oldtype = demand['Type']

        demandid = self.File.shape[0] + 1
        proctime = demand['Quantity']

        if demand['Type'] != oldtype:
            state['SetUp Warning'] = 1
        else:
            state['SetUp Warning'] = 0

        stime = oldctime + (state['SetUp Warning'] * self.Params.SetUpTime())
        ctime = stime + proctime
        violation = ctime - demand['DueDate']
        if violation < 0:
            violation = 0

        rtnDict = {'Demand Id': demandid, 'Machine Id': number, 'Type': chr(demand['Type'] + 64),
                   'Processing Time': proctime, 'Start Time': stime, 'Complete Time': ctime,
                   'Due date': demand['DueDate'], 'Set-Up': bool(state['SetUp Warning']),
                   'Violation Time': violation}

        state['Facility Time'] += proctime
        state['Last Type'] = demand['Type']
        state['Start Time'] = stime
        state['Violation Time'] += violation
        state['Machine Id'] = number

        self.View.UpdateSchedule(machine=number, time=proctime, setup=bool(state['SetUp Warning']),
                                 violation=violation, type=state['Last Type'])
        self.File.append(rtnDict, ignore_index=True)

        reward = self.RewardFunction(number)
        return reward

    def render(self):
        self.View.render()

    def ApplyDemand(self):
        temp = self.Demand.head(1).to_numpy()
        demand = self.obs['Demand']
        demand['Quantity'] = math.ceil(temp[0][2] / self.Params.MachineCapa())
        demand['DueDate'] = temp[0][3]
        demand['Type'] = ord(temp[0][1]) - 64
        self.View.UpdateDemand(math.ceil(demand['Quantity'] / self.Params.MachineCapa()),
                               demand['DueDate'], demand['Type'])
        self.Demand = self.Demand.drop(index=temp[0][0], axis=1)

    # def getRawObservation(self):
    #     Observation = dict()
    #     for i in range(self.Params.MachinesNumber()):
    #         key = 'Machine_' + str(i)
    #         value = {'Facility Time': np.array(0), 'Last Type': np.array(0), 'Start Time': np.array(0),
    #                  'SetUp Warning': np.array(0), 'Violation Time': np.array(0), 'Machine Id': np.array(i + 1)}
    #         Observation[key] = value
    #     key = 'Demand'
    #     value = {'Quantity': np.array(0), 'DueDate': np.array(0), 'Type': np.array(0)}
    #     Observation[key] = value
    #
    #     return Observation

    def getRawObservation(self):
        Observation = dict()
        for i in range(self.Params.MachinesNumber()):
            key = 'Machine_' + str(i)
            value = {'Facility Time': 0, 'Last Type': 0, 'Start Time': 0,
                     'SetUp Warning': 0, 'Violation Time': 0, 'Machine Id': i + 1}
            Observation[key] = value
        key = 'Demand'
        value = {'Quantity': 0, 'DueDate': 0, 'Type': 0}
        Observation[key] = value

        return Observation

    # def getObservationSpace(self):
    #     Observation = dict()
    #     for i in range(self.Params.MachinesNumber()):
    #         key = 'Machine_' + str(i)
    #         value = {'Facility Time': Box(low=0, high=self.Params.LimitationTime(), shape=(1,)),
    #                  'Last Type': Box(low=0, high=self.Params.ProductKinds(), shape=(1,)),
    #                  'Start Time': Box(low=0, high=self.Params.LimitationTime(), shape=(1,)),
    #                  'SetUp Warning': Box(low=0, high=1, shape=(1,)),
    #                  'Violation Time': Box(low=0, high=self.Params.LimitationTime(), shape=(1,)),
    #                  'Machine Id': Box(low=0, high=self.Params.MachinesNumber(), shape=(1,))}
    #         Observation[key] = Dict(value)
    #     key = 'Demand'
    #     value = {'Quantity': Box(low=0, high=self.Params.LimitationTime(), shape=(1,)),
    #              'DueDate': Box(low=0, high=self.Params.LimitationTime(), shape=(1,)),
    #              'Type': Box(low=0, high=self.Params.ProductKinds(), shape=(1,))}
    #     Observation[key] = Dict(value)
    #
    #     return Dict(Observation)

    def getObservationSpace(self):
        temp = self.DictToList()
        low = [0] * len(temp)
        high = [self.Params.LimitationTime() * 3] * len(temp)

        return Box(low=np.array(low), high=np.array(high), shape=(len(temp),), dtype=np.int64)

    def DictToList(self):
        rtn = []
        for i in range(self.Params.MachinesNumber()):
            key = 'Machine_' + str(i)
            state = self.obs[key]
            for _, value in state.items():
                rtn.append(value)
        demand = self.obs['Demand']
        for _, value in demand.items():
            rtn.append(value)

        return np.array(rtn)


class PhotolithographyV2(gym.Env):
    def __init__(self, config: EnvContext):
        self.__M = config["Number of Machines"]
        self.__P = config["Number of Kinds"]
        self.__V = config["Lots Volume"]
        self.__T = 0
        self.__MaxT = 0
        self.__Score = 0
        self.MachineAttributes = pd.DataFrame(index=range(self.__M),
                                              columns=['Performance', 'Masks', 'MaskCapa'])
        self.Lots = pd.DataFrame(index=range(self.__V),
                                 columns=['Recipe', 'Arrival Time', 'Priority'])
        # Machine            # Stocker
        self.Constraints = pd.DataFrame(index=range(self.__M),
                                        columns=['Acceptance Machine', 'MaskTime from M', 'MaskTime from S'])

        self.File = pd.DataFrame(index=range(self.__V),
                                 columns=['Lot Index', 'Recipe', 'Arrival Time', 'Priority',
                                          'Machine Id', 'Start Time', 'Complete Time', 'Last Type', 'Mask Storage',
                                          'Performance', 'Stocker', 'Reward', 'Event'])
        self.pivot = 0
        self.resetcount = 0
        # Define Observation, Space
        self.obs = self.__getObservation()
        self.score = 0
        self.action_space = Discrete(self.__M)
        self.observation_space = self.__getObservationSpace()

        self.__SetAttributes()

    def reset(self):
        self.MachineAttributes = pd.DataFrame(index=range(self.__M),
                                              columns=['Performance', 'Masks', 'MaskCapa'])
        self.Lots = pd.DataFrame(index=range(self.__V),
                                 columns=['Recipe', 'Arrival Time', 'Priority'])
        # Machine            # Stocker
        self.Constraints = pd.DataFrame(index=range(self.__M),
                                        columns=['Acceptance Machine', 'MaskTime from M', 'MaskTime from S'])

        self.obs = self.__getObservation()
        self.__SetAttributes()
        self.ApplyDemand()

        self.__Score = 0

        self.File = pd.DataFrame(index=range(self.__V),
                                 columns=['Recipe', 'Arrival Time', 'Priority',
                                          'Machine Id', 'Start Time', 'Complete Time', 'Last Type', 'Mask Storage',
                                          'Performance', 'Stocker', 'Reward', 'Event'])
        self.pivot = 0
        self.resetcount += 1
        return self.DictToList()

    def step(self, action):
        # self.__t보다 작은 Departure를 가진 머신들중에 Busy인 애들 Depareture 이벤트
        done = False
        reward = 0
        demand = self.obs['Demand']
        event = ""
        # Departure process
        for i in range(self.__M):
            machine = self.obs['Machine_' + str(action)]
            if self.__T >= machine['Complete Time'] and machine['Busy'] == 1:
                machine['Busy'] = 0

        machine = self.obs['Machine_' + str(action)]
        arrange = machine['Arrange']
        busy = machine['Busy']
        accept = arrange[demand['Recipe'] - 1]
        if busy == 0 and accept == 1:
            # Demand Recipe를 지원하지않으면 종료 (디맨드 갱신없음)
            stocker = self.obs['Stocker']

            setuptime = machine['SetUp Warning'] * 5
            if setuptime != 0:
                event = event + "Setup/"
            movingtime = 0
            # machine 의 마스크 확인
            storage = machine['Mask Storage']
            if not demand['Recipe'] in storage:
                # Stocker에 있는 지 확인하고
                if stocker[demand['Recipe'] - 1] > 0:
                    stocker[demand['Recipe'] - 1] -= 1
                    storage.append(demand['Recipe'])
                    if -1 in storage:
                        storage.remove(-1)
                        storage.append(-1)
                    mask = storage.pop(0)
                    if mask != 0:
                        stocker[mask - 1] += 1
                    movingtime = self.Constraints.loc[action, 'MaskTime from S']
                    event = event + "Stocker pop"
                else:
                    # Other Machine 중에 Idle한 머신을 가져가고
                    for i in range(self.__M):
                        if i == action:
                            continue
                        other = self.obs['Machine_' + str(i)]
                        if other['Busy'] == 0:
                            other_storage = other['Mask Storage']
                            if demand['Recipe'] in other_storage:
                                other_storage.remove(demand['Recipe'])
                                other_storage.append(0)
                                if -1 in other_storage:
                                    other_storage.remove(-1)
                                    other_storage.append(-1)

                                storage.append(demand['Recipe'])
                                if -1 in storage:
                                    storage.remove(-1)
                                    storage.append(-1)
                                mask = storage.pop(0)
                                if mask != 0:
                                    stocker[mask - 1] += 1
                                movingtime = self.Constraints.loc[action, 'MaskTime from M']
                                event = event + "Other Machine" + str(other['Machine Id']) + " pop"
                                break
                    # 그것도 아니라면 Arrival Time + Complete Time 추가해서 Lots에 넣기
                    if movingtime == 0:
                        candidate = []
                        for i in range(self.__M):
                            if i == action:
                                continue
                            other = self.obs['Machine_' + str(i)]
                            if other['Busy'] == 1:
                                other_storage = other['Mask Storage']
                                if demand['Recipe'] in other_storage:
                                    candidate.append(other['Complete Time'])

                        demand['Arrival Time'] = min(candidate)
                        temp1 = self.Lots[self.Lots['Arrival Time'] < demand['Arrival Time']]
                        temp2 = self.Lots[self.Lots['Arrival Time'] >= demand['Arrival Time']]
                        self.Lots = temp1.append(demand, ignore_index=True).append(temp2, ignore_index=True)
                        self.ApplyDemand()
                        return self.DictToList(), 0, False, {}

            machine['Busy'] = 1
            machine['Facility Time'] += machine['Performance']
            machine['Start Time'] = machine['Complete Time'] + setuptime + movingtime
            machine['Complete Time'] = machine['Start Time'] + machine['Performance']

            self.File.at[self.pivot, 'Recipe'] = demand['Recipe']
            self.File.at[self.pivot, 'Arrival Time'] = demand['Arrival Time']
            self.File.at[self.pivot, 'Priority'] = demand['Priority']

            self.File.at[self.pivot, 'Machine Id'] = machine['Machine Id']
            self.File.at[self.pivot, 'Start Time'] = machine['Start Time']
            self.File.at[self.pivot, 'Complete Time'] = machine['Complete Time']
            self.File.at[self.pivot, 'Last Type'] = machine['Last Type']
            self.File.at[self.pivot, 'Mask Storage'] = machine['Mask Storage'].copy()
            self.File.at[self.pivot, 'Performance'] = machine['Performance']
            self.File.at[self.pivot, 'Stocker'] = stocker.copy()
            self.File.at[self.pivot, 'Reward'] = self.score
            self.File.at[self.pivot, 'Event'] = event
            self.pivot += 1

            machine['Last Type'] = demand['Recipe']

            if movingtime == 0:
                reward += 0.010
            if setuptime == 0:
                reward += 0.005
            done = self.ApplyDemand()
            if done:
                # Departure process
                for i in range(self.__M):
                    machine = self.obs['Machine_' + str(action)]
                    if machine['Busy'] == 1:
                        if self.__T <= machine['Complete Time']:
                            self.__T = machine['Complete Time']
                        machine['Busy'] = 0

                reward += (self.__MaxT / self.__T * 100)
                self.__Score += reward

                f = open('./Result/Performance.csv', 'a', newline='')
                wr = csv.writer(f)
                wr.writerow([self.__M, self.__P, self.__V, self.__MaxT, self.__T, math.ceil(self.__Score)])
                f.close()

                self.File.to_csv('./Demand/M-' + str(self.__M) + 'P-' + str(self.__P) + 'V-'+str(self.__V)
                                 + ']itr-' + str(self.resetcount) + '.csv', index=False)
                print('Save result file')

        else:
            reward = -0.001
        self.__Score += reward
        return self.DictToList(), reward, done, {}

    def render(self):
        test = 1

    def ApplyDemand(self):
        feasible = False
        candidates = []
        demand = self.obs['Demand']
        if self.Lots.shape[0] == 0:
            return True
        else:
            temp = self.Lots.head(1)

            demand['Recipe'] = temp['Recipe'].item()
            demand['Arrival Time'] = temp['Arrival Time'].item()
            demand['Priority'] = temp['Priority'].item()

            for num in range(self.__M):
                machine = self.obs['Machine_' + str(num)]
                performance = self.MachineAttributes.loc[num, 'Performance']
                machine['Performance'] = performance[temp['Recipe'].item() - 1]
                if machine['Complete Time'] <= demand['Arrival Time'] and machine['Performance'] != INF:
                    feasible = True
                else:
                    if machine['Performance'] != INF:
                        candidates.append(machine['Complete Time'])

                if machine['Last Type'] is not 0 & machine['Last Type'] != demand['Recipe']:
                    machine['SetUp Warning'] = 1
                else:
                    machine['SetUp Warning'] = 0
            if not feasible:
                demand['Arrival Time'] = min(candidates)

            self.__T = demand['Arrival Time']
            self.Lots = self.Lots.drop(index=temp.index, axis=1)
            # print("Left Lots volume:"+str(self.Lots.shape[0]+1))
            # test = self.DictToList()
            # test2 = self.observation_space.sample()
            # print(self.observation_space.contains(test))
            # test3 = 1
            return False

    def DictToList(self):
        lim = ['Machine Id', 'Busy', 'Last Type', 'SetUp Warning']

        temp = copy.deepcopy(self.obs)
        demand = temp['Demand']
        demand['Arrival Time'] = np.array([demand['Arrival Time']])
        temp['Stocker'] = np.array(temp['Stocker'])

        for i in range(self.__M):
            machine = temp['Machine_' + str(i)]
            for key, value in machine.items():
                if key in lim:
                    continue
                if not isinstance(value, list):
                    convert = [value]
                else:
                    convert = value
                machine[key] = np.array(convert)
        #print(self.observation_space.contains(temp))
        return temp

    def __getObservation(self):
        observation = OrderedDict()
        # Demand
        observation['Demand'] = OrderedDict({'Arrival Time': 0, 'Priority': 0, 'Recipe': 0})
        for num in range(self.__M):
            sub_modular = OrderedDict({'Arrange': [0] * self.__P, 'Busy': 0,
                                       'Complete Time': 0, 'Facility Time': 0, 'Last Type': 0,
                                       'Machine Id': num + 1, 'Mask Storage': [0, 0, 0],
                                       'MaskFromM': 0, 'MaskFromS': 0, 'Performance': 0,
                                       'SetUp Warning': 0, 'Start Time': 0, 'Violation Time': 0})
            observation['Machine_' + str(num)] = sub_modular
        observation['Stocker'] = [0] * self.__P
        return observation

    def __getObservationSpace(self):
        # observation = dict()
        # # Demand
        # observation['Demand'] = Dict({'Recipe': Discrete(self.__P),
        #                               'Arrival Time': Discrete(INF),
        #                               'Priority': Discrete(11)})
        # observation['Stocker'] = Box(low=np.array([0] * self.__P), high=np.array([1] * self.__P),
        #                              shape=(self.__P, ), dtype=np.int64)
        # for num in range(self.__M):
        #     sub_modular = Dict({'Machine Id': Discrete(self.__M + 1), 'Facility Time': Discrete(INF),
        #                         'Start Time': Discrete(INF),
        #                         'Complete Time': Discrete(INF),
        #                         'Busy': Discrete(1),
        #                         'Last Type': Discrete(self.__P + 1),
        #                         'Performance': Discrete(INF),
        #                         'Arrange': Box(low=np.array([0] * self.__P), high=np.array([1] * self.__P),
        #                                        shape=(self.__P, ), dtype=np.int64),
        #                         'SetUp Warning': Discrete(1), 'Violation Time': Discrete(INF),
        #                         'Mask Storage': Box(low=np.array([0, 0, 0]), high=np.array([self.__P + 1] * 3),
        #                                             shape=(3, ), dtype=np.int64)})
        #     observation['Machine_' + str(num)] = sub_modular
        observation = dict()
        # Demand
        observation['Demand'] = Dict({'Recipe': Discrete(self.__P + 1),
                                      'Arrival Time': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                      'Priority': Discrete(11)})
        observation['Stocker'] = Box(low=np.array([0] * self.__P), high=np.array([5] * self.__P),
                                     shape=(self.__P,), dtype=np.int32)
        for num in range(self.__M):
            sub_modular = Dict({'Machine Id': Discrete(self.__M + 1),
                                'Facility Time': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                'Start Time': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                'Complete Time': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                'Busy': Discrete(2),
                                'Last Type': Discrete(self.__P + 1),
                                'Performance': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                'Arrange': Box(low=np.array([0] * self.__P), high=np.array([1] * self.__P),
                                               shape=(self.__P,), dtype=np.int32),
                                'SetUp Warning': Discrete(2),
                                'Violation Time': Box(low=0, high=INF, shape=(1,), dtype=np.int32),
                                'Mask Storage': Box(low=np.array([-1, -1, -1]), high=np.array([self.__P + 1] * 3),
                                                    shape=(3,), dtype=np.int32),
                                'MaskFromM': Box(low=0, high=50, shape=(1,), dtype=np.int32),
                                'MaskFromS': Box(low=0, high=50, shape=(1,), dtype=np.int32)})
            observation['Machine_' + str(num)] = sub_modular

        return Dict(observation)

    def __SetAttributes(self):
        # Constraints Setting
        Acceptance_Machine = self.__SetMachineAcceptance()
        MasktimeM = self.__SetMovingTime(10, 40 + 1)
        MasktimeS = self.__SetMovingTime(5, 20 + 1)

        npstack = np.vstack((Acceptance_Machine, MasktimeM, MasktimeS))
        nptr = np.transpose(npstack)
        self.Constraints[:][:] = nptr

        # Machine Attribute Setting
        Performance, RecipeMinTime = self.__SetMachinePerformance(20, 40 + 1, Acceptance_Machine)
        MachineMaskCapa = self.__UniformRandomInt(2, 3 + 1, self.__M)
        MachineHasMasks = []

        # Observation Update
        for i in range(self.__M):
            MachineHasMasks.append([0] * MachineMaskCapa[i])
            machine = self.obs['Machine_' + str(i)]
            arrange = machine['Arrange']
            storage = machine['Mask Storage']
            machine['MaskFromM'] = MasktimeM[i].item()
            machine['MaskFromS'] = MasktimeS[i].item()

            for j in Acceptance_Machine[i]:
                arrange[j - 1] = 1

            if MachineMaskCapa[i] > 2:
                storage[2] = -1

        MachineMaskCapa = MachineMaskCapa.tolist()
        self.MachineAttributes[:]['Performance'] = Performance
        self.MachineAttributes[:]['Masks'] = MachineHasMasks
        self.MachineAttributes[:]['MaskCapa'] = MachineMaskCapa

        stock = self.__UniformRandomInt(1, 2 + 1, self.__P)
        for i in range(self.__P):
            stocker = self.obs['Stocker']
            stocker[i] = stock[i]

        # Demand Setting
        Recipe = self.__SetRecipeIntoLot()
        ArrivalTime = self.__SetArrivalTime(Recipe, RecipeMinTime)
        Priority = self.__SetLotPriority(1, 10 + 1)

        npstack = np.vstack((Recipe, ArrivalTime, Priority))
        nptr = np.transpose(npstack)
        self.Lots[:][:] = nptr
        self.Lots = self.Lots.sort_values(by='Arrival Time', axis=0, ascending=True)
        tail = self.Lots.tail(1)
        self.__MaxT = tail['Arrival Time'].item()

    def __UniformRandomInt(self, LSL, USL, size):
        distribution = 'uniform'
        value = getRandomValue(distribution=distribution,
                               param='low=' + str(LSL) + ',' +
                                     'high=' + str(USL),
                               size=size)
        value = value.astype(np.int32)
        return value

    # 공정시간 설정
    def __SetMachinePerformance(self, LSL, USL, Acceptance):
        MachinePerformance = []
        RecipeMinTime = [INF] * self.__P
        distribution = 'uniform'
        for i in range(self.__M):
            MachinePerformance.append([])
            for j in range(1, self.__P + 1):
                value = INF
                if j in Acceptance[i]:
                    value = getRandomValue(distribution=distribution,
                                           param='low=' + str(LSL) + ',' +
                                                 'high=' + str(USL),
                                           size=1)
                    value = np.asscalar(value.astype(np.int32))
                    if RecipeMinTime[j - 1] > value:
                        RecipeMinTime[j - 1] = value
                MachinePerformance[i].append(value)
        return MachinePerformance, RecipeMinTime

    # 공정조건 설정(recipe or model type)
    def __SetRecipeIntoLot(self):
        distribution = 'uniform'
        Recipe = getRandomValue(distribution=distribution,
                                param='low=' + str(1) + ',' +
                                      'high=' + str(self.__P + 1),
                                size=self.__V)
        return Recipe.astype(np.int32)

    # 각 로트의 도착시간
    def __SetArrivalTime(self, Recipe, RecipeMinTime):
        distribution = 'uniform'
        ArrivalTime = []
        for model in Recipe:
            time = RecipeMinTime[model - 1]
            USL = (time * self.__V) / self.__M
            value = getRandomValue(distribution=distribution,
                                   param='low=' + str(0) + ',' +
                                         'high=' + str(USL),
                                   size=1)
            value = np.asscalar(value.astype(np.int32))
            ArrivalTime.append(value)
        return np.array(ArrivalTime)

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
        data = []
        for i in range(self.__M):
            data.append([])

        for i in range(1, self.__P + 1):
            Many = getRandomValue(distribution='uniform',
                                  param='low=' + str(1) + ',' +
                                        'high=' + str(self.__M + 1),
                                  size=1)
            Many = Many.astype(np.int32)
            Acceptance = np.random.choice(range(0, self.__M), Many.item(0), replace=False)
            Acceptance = Acceptance.tolist()
            for j in Acceptance:
                data[j].append(i)

        MachineAcceptance = np.empty((self.__M,), dtype=object)
        for i in range(self.__M):
            MachineAcceptance[i] = data[i]

        return MachineAcceptance

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
        self.NextDeparture = 0
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
            self.NextDeparture = self.Tnow + ServiceTime
            self.Utilization += ServiceTime
            self.__TotalSystemTime += (WaitingTime + ServiceTime)
        return self.NextDeparture

    def Departure(self, time):
        self.__AreaUnderQt += len(self.Q) * (time - self.Tnow)
        # self.Utilization += time - self.Tnow
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
    Env = PhotolithographyV2({"Number of Machines": 4, "Number of Kinds": 8, "Lots Volume": 10})

    Env.reset()
    score = 0
    for i in range(50):
        m = int(input('Machine: '))
        obs, reward, done, info = Env.step(m)
        value = Env.observation_space.contains(obs)
        score += reward
        print('Total score: ' + str(score) + ' reward: ' + str(reward) + ' done: ' + str(done))
        if done:
            return


if __name__ == '__main__':
    main()
