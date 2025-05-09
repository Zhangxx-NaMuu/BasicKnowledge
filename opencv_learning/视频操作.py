# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 视频操作
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/5/8 13:15
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/5/8 13:15:
==================================================  
"""
__author__ = 'zxx'

import numpy as np
import cv2

# cap = cv2.VideoCapture('videos/dog.mp4')
# while True:
#     # 获取一帧
#     ret, frame = cap.read()
#     # 对读入的帧图进行处理
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(50) & 0xFF == ord('q'):
#         break
#
# # 当一切完成时，释放捕获
# cap.release()
# cv2.destroyAllWindows()
#
# """
# 保存视频：创建一个VideoWrite对象，确定输出文件名，指定FourCC编码，播放频率和帧的大小，最后是isColor标签，True为彩色。FourCC是一个4字节码，用来确定视频的编码格式。
# * In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2 XVID是最好的，MJPG是高尺寸视频，X264得到小尺寸视频
# * In Windows: DIVX
# 设置FOURCC格式时，原文里采用了cv2.VidelWriter_fourcc()这个函数，若运行程序时显示这个函数可能不存在，可以改用了cv2.cv.CV_FOURCC这个函数
# """
# cap = cv2.VideoCapture(0)
#
# # 定义编解码器并创建VideoWriter对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./videos/output.avi', fourcc, 20.0, (640, 480))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret == True:
#         #图片处理（翻转）
#         frame = cv2.flip(frame, 0)
#         #写出当前帧
#         out.write(frame)
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# # 释放设备，io
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# 高级操作
import io
import base64
from IPython.display import display, HTML, clear_output
import numpy as np
import cv2
import matplotlib.pyplot as plt


def open_video_controller(fpath, width=480, height=360):
    display(HTML(data='''<video alt="test" width="''' + str(width) + '''" height="''' + str(height) + '''" controls>
            <source src="''' + fpath + '''" type="video/mp4" />
            </video>'''))


def open_video_controller2(fpath, width=480, height=360):
    video = io.open(fpath, 'r+b').read()
    encoded = base64.b64encode(video)
    display(HTML(data='''<video alt="test" width="''' + str(width) + '''" height="''' + str(height) + '''" controls>
            <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))


open_video_controller("images/vtest.mp4")
