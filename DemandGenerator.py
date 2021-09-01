from Parameters import PreSetInfo
import math
import pandas as pd
import pickle

def Initialize(M=None, P=None, maxQ=None, minQ=None, maxT=None,
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
        maxalltime = 148
    if capacity is None:
        capacity = 1000
    if distribution is None:
        distribution = "Uniform"

    PreSet = PreSetInfo(M, P, maxQ, minQ, maxT, setuptime, maxalltime, capacity, distribution)
    return PreSet

def GenerateDemand(Param, Num):
    import random
    baseunit = 100
    offset = 0
    demandid = 0

    #Trim 30%를 구현해야 하나, 지금은 Max 30%로
    nowMax = 0
    cumDudate = 0
    Quantity = []
    DueDate = []
    Type = []
    DemandId = []
    print(math.ceil((Param.LimitationTime() * Param.MachinesNumber() - Param.SetUpTime()) * 0.7))
    for week in range(0, Num):
        while cumDudate < math.ceil((Param.LimitationTime() * Param.MachinesNumber() - Param.SetUpTime()) * 0.7):
            for i in range(0, Param.MachinesNumber()):
                quantity = random.randrange(Param.MinOrderQ(), Param.MaxOrderQ()+baseunit)
                processtime = math.ceil(quantity / Param.MachineCapa())
                duedate = random.randrange(processtime, Param.MaxDueDate())
                duedate = duedate + offset
                type = chr(65 + random.randrange(0, Param.ProductKinds()))
                # 리스트에 넣어주기
                Quantity.append(quantity)
                DueDate.append(duedate)
                Type.append(type)
                DemandId.append(demandid)

                if nowMax < (duedate - offset):
                    nowMax = duedate
                demandid = demandid + 1
                cumDudate = cumDudate + (duedate - offset)
            offset = offset + round(nowMax * 0.7)
        # 엑셀파일로 저장
        A = pd.Series(DemandId, name='Demand Id')
        B = pd.Series(Type, name='Type')
        C = pd.Series(Quantity, name='Quantity')
        D = pd.Series(DueDate, name='DueDate')
        SaveData = pd.concat([A, B, C, D], axis=1)
        SaveData.to_csv('./DemandSet/DemandStatement' + str(week+1) + '.csv', index=False)

def main():
    Params = Initialize()
    print(Params.MachinesNumber())
    print(Params.ProductKinds())
    print(Params.MaxOrderQ())
    print(Params.MinOrderQ())
    print(Params.MaxDueDate())
    print(Params.SetUpTime())
    print(Params.LimitationTime())
    print(Params.MachineCapa())
    print(Params.Distribution())
    temp = GenerateDemand(Params, 100000)
    with open('Params.p', 'wb') as file:
        pickle.dump(Params, file)

if __name__ == '__main__':
    main()