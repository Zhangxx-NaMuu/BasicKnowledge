# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 人脸检测
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/5/9 10:54
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/5/9 10:54:
==================================================  
"""
__author__ = 'zxx'
"""
通过使用opencv实现人脸检测喝其他部位的检测。我们将用到CascadeClassifier分类器，利用opencv分类器CascadeClassifier，
对数据进行实时检测，并把结果显示到屏幕上，CascadeClassifier不仅可以检测人脸，还能检测眼睛，身体，嘴巴。
通过加载一个想要的.XML分类器文件就可以。然后使用detectMultiScale()函数，可以检测出图片上的所有人脸，并将人脸用vector保存各个人脸的坐标、大小，用举行表示，函数由分类器对象调用
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# 读取级联分类器
cascPath = r'detect/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# 读取图片
img = cv2.imread('images/player.jpg')
plt.imshow(img)
plt.show()
"""
detectMultiScale(image, object, scaleFactor, minNeighbors, minSize, maxSize)
image:要检测的输入图像,
object：检测到的人脸目标序列,
scaleFactor：表示前后两次相继的扫描中，搜索窗口的比例系数，默认为1.1即每次搜索窗口依次扩大10%,
minNeighbors：表示构成检测目标的相邻矩形的最小个数，默认3,
minSize：目标的最小尺寸, 
maxSize：目标的最大尺寸
"""


def detect_faces_show(fpath):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 检测图像中的人脸
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    print("发现%d个人脸" % len(faces))
    # 在这些脸周围画一个矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    plt.imshow(img)
    plt.show()


detect_faces_show('images/player.jpg')

# 检测人脸和眼睛
face_cascade = cv2.CascadeClassifier('detect/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('detect/haarcascade_eye.xml')

img = cv2.imread('images/zms.png')
plt.imshow(img)
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi_gray = gray[y: y + h, x: x + w]
    roi_color = img[y: y + h, x: x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=3)

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

plt.imshow(img)
plt.show()
