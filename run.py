import wx
from src.app import MainFrame

""" Run this to see the program in action!. """

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()
