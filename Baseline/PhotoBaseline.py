import pandas as pd
from Environment.Renderer import GanttChart
from Environment import SimulationEngine as Sim


class EMRFEngine:
    def __init__(self, Env):
        self.Env = Env
        self.Arrival = self.Env.Lots.sort_values(by='Arrival Time', axis=0, ascending=True)
        self.Departure = pd.DataFrame(columns=['Departure Time', 'Number'])
        self.Machine = []
        for i in range(self.Env.MachineAttributes.shape[0]):
            self.Machine.append(Sim.Engine())

        self.Renderer = GanttChart((len(self.Machine), self.Env.Lots.shape[0] * 2), len(self.Machine),
                                   len(self.Env.Stocker))
        self.Renderer.SetTitle('Photo line GanttChart')

    def run(self, post=False):
        # Dashboard 출력용
        # print('===============Dash Board================')
        # print('Masks in Stocker')
        # print(self.Env.Stocker)
        # print('Masks in Machines')
        # print(self.Env.MachineAttributes['Masks'])
        # print('========================================')
        # 초기화
        nexttime = 0
        self.Departure.sort_values(by='Departure Time', axis=0, ascending=True)

        # 동시간 스케줄셋 추출
        head = self.Arrival.head(1)
        val = head['Arrival Time'].tolist()
        if post:
            val = self.Departure['Departure Time'].head(1).tolist()
            print('==================Departure Time:' + str(val) + '====================')
        else:
            print('==================Arrival Time:' + str(val) + '====================')
        # Departure event
        Deplist = self.Departure[self.Departure['Departure Time'] <= val[0]]
        delidx = self.Departure[self.Departure['Departure Time'] <= val[0]].index
        self.Departure = self.Departure.drop(delidx)
        for timenum in Deplist.iterrows():
            self.Machine[timenum[1]['Number']].Departure(timenum[1]['Departure Time'])
            print('[Departure Event] Machine:' + str(timenum[1]['Number']))

        AssignSet = self.Arrival[self.Arrival['Arrival Time'] == val[0]]
        delidx = self.Arrival[self.Arrival['Arrival Time'] == val[0]].index
        self.Arrival = self.Arrival.drop(delidx)

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
                                self.ArrivalEvent('Normal', number, recipe, time)
                                break
                            # 다른 모델을 처리했으나, 마스크를 가지고 있으면 Step 4-2
                            else:
                                self.ArrivalEvent('Setting', number, recipe, time)
                                break
                        else:
                            # 마스크는 안가지고 있는데, 보관소에 그 마스크가 있으면 Step 4-3
                            if not self.Env.Stocker[recipe - 1] == 0:
                                # Extract mask from Stocker
                                self.Env.Stocker[recipe - 1] -= 1
                                self.ArrivalEvent('Stocker', number, recipe, time)
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
                                            self.ArrivalEvent('Machine', number, recipe, time)
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
                    # print('[Assign Failed]' + 'Recipe Type: ' + str(recipe))

    def ArrivalEvent(self, mode, number, recipe, time):
        perform = self.Env.MachineAttributes.loc[number, 'Performance']
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

        print('[Arrival Event] Assign Machine:' + str(number) + ', Recipe Type: ' + str(recipe)
              + ', SetUp: ' + str(setup) + ', Moving Time: ' + str(toS + toM)
              + ', Departure: ' + str(nexttime))


def main():
    Photo = Sim.PhotoLine(Number_Of_Machines=20, Number_Of_Kinds=20, LotsVolume=500)
    Photo.generateDemandSet()
    EMRF = EMRFEngine(Photo)

    while EMRF.Arrival.shape[0] != 0:
        EMRF.run()

    while EMRF.Departure.shape[0] != 0:
        EMRF.run(post=True)

    for i in range(len(EMRF.Machine)):
        print('-----------------Machine[' + str(i) + ']-----------------')
        print('Acceptance model in Machine')
        print(EMRF.Env.Constraints.loc[i, 'Acceptance Machine'])
        print('Performance of Machine')
        print(EMRF.Env.MachineAttributes.loc[i, 'Performance'])
        print('Masks in Machines')
        print(EMRF.Env.MachineAttributes.loc[i, 'Masks'])
        print('[Summary]')
        EMRF.Machine[i].Summary()
        print('---------------------------------------------------------')

    return None


if __name__ == '__main__':
    main()
