import pickle

class Env():
    def __init__(self):
        with open('Params.p', 'rb') as file:
            Params = pickle.load(file)
        super(Env, self).__init__()
        self.score = 0
        self.score_weight = []
        self.counter = 0
def step(action):

    return reward, setup, violation

def RewardFunction(setup, violation):
    Reward = 1 - (5 * violation)
    if setup:
        Reward = Reward - 2
    return Reward

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