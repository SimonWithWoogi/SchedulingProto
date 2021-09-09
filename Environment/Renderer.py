import numpy as np
import tkinter as tk
import matplotlib.colors as mcolors
import pyscreenshot
from PIL import Image
from PIL import ImageTk

ColorList = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class GanttChart:
    def __init__(self, Size, Number_Of_Machines, Number_Of_Kinds):
        if len(ColorList) < Number_Of_Kinds:
            raise Exception('Over range: Please set parameter(Number_Of_Kinds) less than ' + str(len(ColorList)+1))
        self.M = Number_Of_Machines
        self.P = Number_Of_Kinds
        self.BoardSize = Size
        self.BackBoard = np.zeros(self.BoardSize)

        # GUI 설정
        self.win = tk.Tk()
        # GanttChart board
        pil_image = Image.fromarray(self.BackBoard)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.GUIBackBoard = tk.Label(self.win, image=imgtk)
        self.GUIBackBoard.place(x=0, y=0)
        self.GUIBackBoard.pack()
        # GanttChart Machines
        self.GUIMachine = []
        for i in range(Number_Of_Machines):
            self.GUIMachine.append(tk.Button(self.win, text='M'+str(i+1)))
            self.GUIMachine[i].grid(row=i, column=0, sticky=tk.N+tk.S)

    def Reset(self):
        self.BackBoard = np.zeros(self.BoardSize)
    def Render(self):
        pil_image = Image.fromarray(self.Backboard)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.GUIBackBoard.config(image=imgtk)
        self.win.update()

        pil_image.close()
        pil_image.__exit__()
        imgtk.__del__()
    def Capture(self, Save, Name):
        im = pyscreenshot.grab(bbox=(10, 10, 510, 510))  # X1,Y1,X2,Y2
        if Save:
            im.save('./ScreenShot/'+Name+'.png')
        return im
    def SetTitle(self, Text):
        self.win.title = Text

def main():
    Chart = GanttChart((76, 224), 17, 10)
    Chart.win.mainloop()
if __name__ == '__main__':
    main()

