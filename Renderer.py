# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pickle
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import time
class Draw:
    def __init__(self):
        with open('Params.p', 'rb') as file:
            self.Params = pickle.load(file)
        self.MachinePos = 0
        self.MachineUnit = 12  # 한 머신당 height 4 pixel 잡아먹음
        self.WorkUnit = 8  # 두 작업당 width 1 pixel 잡아먹음
        # 여기서부터 Demand 표현
        self.DemandPos = self.MachineUnit * self.Params.MachinesNumber()
        self.DemandUnit = 12
        self.Backboard = np.zeros((self.DemandUnit + self.MachineUnit
                                  * self.Params.MachinesNumber(),
                                  self.WorkUnit * self.Params.LimitationTime()))
        self.oldduedate = 0
        self.oldbackboard = self.Backboard
        self.typegap = 255 / self.Params.ProductKinds()
        self.win = tk.Tk()  # 인스턴스 생성
    def ResetBackboard(self):
        self.Backboard = np.zeros((self.DemandUnit + self.MachineUnit
                                  * self.Params.MachinesNumber(),
                                  self.WorkUnit * self.Params.LimitationTime()))
        self.oldduedate = 0
        self.oldbackboard = self.Backboard
    def UpdateDemand(self, time, duedate, type):

        DemandSection = self.Backboard[self.DemandPos:, :] # It's pointer
        DemandSection[:, :] = 0
        DemandSection[:self.DemandUnit, :time * self.WorkUnit] = 255

        self.Backboard[:, self.oldduedate * self.WorkUnit] = self.oldbackboard[:, self.oldduedate * self.WorkUnit]
        self.oldduedate = duedate
        self.oldbackboard = self.Backboard.copy()

        self.Backboard[:, duedate * self.WorkUnit] = 180
        DemandSection[:self.DemandUnit, -4 * self.WorkUnit:] = (type+1) * self.typegap

    def UpdateSchedule(self, machine, time, setup, violation, type):
        self.Backboard[:, self.oldduedate * self.WorkUnit] = self.oldbackboard[:, self.oldduedate * self.WorkUnit].copy()

        base = machine * self.MachineUnit + self.MachinePos
        ScheduleSection = self.Backboard[base:base+self.MachineUnit, :] # It's pointer

        idx = np.where(ScheduleSection[0, :] == 0)
        EmptySection = ScheduleSection[0, idx]
        if setup:
            setuptime = self.Params.SetUpTime()
        else:
            setuptime = 0
        jobtime = time + setuptime
        if violation > time + setuptime:
            violation = jobtime

        EmptySection[0, :jobtime * self.WorkUnit] = 255
        if violation > 0:
            EmptySection[0, (jobtime - violation) * self.WorkUnit:jobtime * self.WorkUnit] = 128
            retouch_board = self.oldbackboard[base:base + self.MachineUnit, :]
            retouch_section = retouch_board[0, np.where(retouch_board[0, :] == 0)]
            retouch_section[:, (jobtime - violation) * self.WorkUnit:jobtime * self.WorkUnit - 1] = 128
            retouch_board[:, np.where(retouch_board[0, :] == 0)] = retouch_section

        ScheduleSection[:, idx] = EmptySection
        ScheduleSection[:, -4 * self.WorkUnit:] = (type + 1) * self.typegap

        self.Backboard[:, self.oldduedate * self.WorkUnit] = 180
    def render(self):
        pil_image = Image.fromarray(self.Backboard)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        label = tk.Label(self.win, image=imgtk)
        label.pack(side="top")
        self.win.update()
        label.destroy()
        return None

    def getBackBoardImage(self, size):
        pil_image = Image.fromarray(self.Backboard)
        resize_image = pil_image.resize((168, 168))
        return np.array(resize_image)