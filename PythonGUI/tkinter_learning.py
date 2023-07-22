# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> tkinter_learning
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/22 14:02
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/22 14:02:
==================================================  
"""
__author__ = 'zxx'

from tkinter import Tk
from tkinter.simpledialog import askinteger, askfloat, askstring
from tkinter.filedialog import askopenfilename, askopenfilenames, asksaveasfilename, askdirectory
from tkinter.messagebox import showinfo, showwarning, showerror

"""
原文链接：https://blog.csdn.net/ScienceRui/article/details/93889827
"""

if __name__ == "__main__":
    app = Tk()  # 初始化GUI程序
    app.withdraw()  # 仅显示对话框，隐藏主窗口

    showinfo(title="提示",
             message="这是一个提示信息对话框!")
    showwarning(title="警告",
                message="这是一个警告信息对话框!")
    showerror(title="错误",
              message="这是一个错误信息对话框!")
    #
    showinfo(title="提示",
             message="程序程序即将开始运行!")

    # 整型数输入对话框
    var_int = askinteger(title="请输入一个整数", prompt="整型变量x:")

    # 浮点型输入对话框
    var_float = askfloat(title="请输入一个浮点数", prompt="浮点型变量x:")

    # 字符或字符串输入对话框
    var_string = askstring(title='请输入一个字符或字符串', prompt='字符型变量x:')

    open_file_path = askopenfilename(title="请选择一个要打开的Excel文件",
                                     filetypes=[("Microsoft Excel文件", "*.xlsx"),
                                                ("Microsoft Excel 97-20003 文件", "*.xls")])
    open_file_path_list = askopenfilenames(title="请选择一个或多个要打开的Excel文件",
                                           filetypes=[("Microsoft Excel文件", "*.xlsx"),
                                                      ("Microsoft Excel 97-20003 文件", "*.xls")])
    save_file_path = asksaveasfilename(title="请创建或者选择一个保存数据的Excel文件",
                                       filetypes=[("Microsoft Excel文件", "*.xlsx"),
                                                  ("Microsoft Excel 97-20003 文件", "*.xls")],
                                       defaultextension=".xlsx")
    select_directory = askdirectory(title="请选择一个文件夹")

    #
    print("var_int:", var_int)
    print("var_float:", var_float)
    print("var_string:", var_string)
    print(open_file_path)
    print(open_file_path_list)
    print(save_file_path)
    print(select_directory)
    #
    showinfo(title="提示",
             message="程序已运行结束!")
    #
    app.destroy()  # 关闭GUI窗口，释放资源
