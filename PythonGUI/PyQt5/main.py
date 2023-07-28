# # -*- coding: UTF-8 -*-
# """
# ==================================================
# @path   ：inference_script -> main.py
# @IDE    ：PyCharm
# @Author ：
# @Email  ：2458543125@qq.com
# @Date   ：2023/7/27 11:37
# @Version:
# @License:
# @Reference:
# @History:
# - 2023/7/27 11:37:
# ==================================================
# """
# __author__ = 'zxx'
#
import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QLabel, QFileDialog
from crown_generate import Ui_CrownGenerate
from PyQt5 import QtCore
# 加载背景图片的重要依据，首先把.qrc文件转成py文件，然后导包调用
from bcg import *
from tkinter import Label, Button, Toplevel
import tkinter as tk


# 注意：ui界面文件是个对话框，那么MyApp就必须继承 QDialog
# 类似的，若ui界面文件是个MainWindow，那么MyApp就必须继承 QMainWindow
class MyWindow(QMainWindow, Ui_CrownGenerate):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)  # 设置界面
        """
        设置全局变量toothNumber和generateModel
        """
        self.toothNumber = None
        self.generateModel = None
        self.folder_path = None
        # 绑定点击信号和信号槽
        self.btn_lower.clicked.connect(self.btn_lower_click)
        self.tooth_4.clicked.connect(self.tooth_4_click)
        self.tooth_5.clicked.connect(self.tooth_5_click)
        self.tooth_6.clicked.connect(self.tooth_6_click)
        self.tooth_7.clicked.connect(self.tooth_7_click)

    def btn_upper_click(self):
        sender = self.sender()
        self.generateModel = sender.text()
        print(self.generateModel)

        def show_my_info():
            info_window = Toplevel()
            info_window.title("提示信息")
            info_window.geometry('250x150')
            label = Label(info_window, text=f"你选择了{self.toothNumber}号牙的{self.generateModel}模型!")
            button = Button(info_window, text="确定", command=handle_button_click)
            label.place(relx=0.1, rely=0.1)
            button.place(relx=0.2, rely=0.45, height=50, width=100)

        def handle_button_click():
            root.destroy()
            folder_dialog = QFileDialog()
            current_dir = os.path.dirname(__file__)  # 获取当前文件所在的目录
            self.folder_path = folder_dialog.getExistingDirectory(self, current_dir)
            if self.folder_path:
                print("选择的文件夹路径为:", self.folder_path)
                widget.hide()
                # self.main()

        root = tk.Tk()
        root.withdraw()
        show_my_info()
        root.mainloop()

    def btn_lower_click(self):
        sender = self.sender()
        self.generateModel = sender.text()
        print(self.generateModel)

        def show_my_info():
            info_window = Toplevel()
            info_window.title("提示信息")
            info_window.geometry('250x150')
            label = Label(info_window, text=f"你选择了{self.toothNumber}号牙的{self.generateModel}模型!")
            button = Button(info_window, text="确定", command=handle_button_click)
            label.place(relx=0.1, rely=0.1)
            button.place(relx=0.2, rely=0.45, height=50, width=100)

        def handle_button_click():
            app.exec_()
            root.destroy()
            folder_dialog = QFileDialog()
            current_dir = os.path.dirname(__file__)  # 获取当前文件所在的目录
            self.folder_path = folder_dialog.getExistingDirectory(self, current_dir)
            if self.folder_path:
                print("选择的文件夹路径为:", self.folder_path)
                widget.hide()
                # self.main()

        root = tk.Tk()
        # root.geometry('667x500')
        root.withdraw()
        show_my_info()
        root.mainloop()

    def tooth_4_click(self):
        sender = self.sender()
        self.toothNumber = sender.text()
        print(self.toothNumber)

    def tooth_5_click(self):
        sender = self.sender()
        self.toothNumber = sender.text()
        print(self.toothNumber)

    def tooth_6_click(self):
        sender = self.sender()
        self.toothNumber = sender.text()
        print(self.toothNumber)

    def tooth_7_click(self):
        sender = self.sender()
        self.toothNumber = sender.text()
        print(self.toothNumber)

    def fun(self):
        return self.folder_path


if __name__ == "__main__":
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication([])
    widget = MyWindow()
    widget.show()
    sys.exit(app.exec_())
