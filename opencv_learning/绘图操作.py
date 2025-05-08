# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 绘图操作
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/4/30 15:27
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/4/30 15:27:
==================================================  
"""
__author__ = 'zxx'

import numpy as np

"""
基础绘图函数
* cv2.line(image, startPoint, endPoint, rgb, thinkness)
* cv2.rectangle(image, topLeft, bottomRight, rgb, thinkness)
* cv2.circle(image, center, radius, rgb, thinkness)
* cv2.ellipse(image, center, axes, angle, startAngle, rgb, thinkness)
需要设置的参数：
* image:要绘图所在的图像
* color(rgb)：形状的颜色，以RGB为例，需要传入的元组，对于灰度图只需要传入灰度值
* thinkness：线条的粗细，如果给一个闭合图形设置为-1，那么这个图形就会被填充，默认设置为1
* linetype：线条的类型，8连接，抗锯齿等，cv2.LINE_AA为抗锯齿
"""
import cv2
import matplotlib.pyplot as plt

# 创建一个黑色图案
img2 = np.zeros((512, 521, 3), np.uint8)
plt.imshow(img2)
plt.show()
# 用cv2在上面绘制的黑色图像上画一条线，cv2.line(image, startPoint, endPoint, rgb, thinkness)
cv2.line(img2, (0, 0), (511, 511), (255, 0, 0), 5)
# 传入参数，这条线的起点，终点，颜色为红色，厚度为5px

# 绘制矩形cv2.rectangle(image, topLeft, bottomRight, rgb, thinkness)
cv2.rectangle(img2, (384, 0), (510, 128), (0, 255, 0), 3)
# 传入参数为：左上角顶点和右下角顶点的坐标，颜色为绿色，厚度为3px

# 画一个圆cv2.circle(image, center, radius, rgb, thinkness)
cv2.circle(img2, (477, 63), 63, (0, 0, 255), -1)

# 画一个椭圆cv2.ellipse(image, center, axes, angle, startAngle, rgb, thinkness)
cv2.ellipse(img2, (256, 256), (100, 50), -45, 0, 180, (255, 0, 0), -1)
# 传入参数为：中心点的位置坐标，长轴短轴，旋转角度（顺时针方向），绘制的起始角度（顺时针方向）
# 绘制的终止角度（例如，绘制整个椭圆是0,360，绘制下半椭圆就是0,180），颜色为红色，线条粗细（默认值=1）
plt.imshow(img2)
plt.show()

pts = np.array([[10, 10], [150, 200], [300, 150], [200, 50]], np.int32)
pts = pts.reshape((-1, 1, 2))

# 画一个青色封闭的四边形
cv2.polylines(img2, [pts], True, (0, 255, 255), 3)
plt.imshow(img2)
plt.show()

# 写一些文字cv2.putText(image, text, bottomLeft, fontType, fontScale, rgb, thinkness, lineType)
# 需要设置，文字内容，绘制的位置，字体类型、大小、颜色、粗细、类型等，这里推荐linetype=cv2.LINE_AA

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img2, 'OpenCV', (10, 500), font, 4, (255, 255, 255), 10, cv2.LINE_AA)

plt.imshow(img2)
plt.show()
