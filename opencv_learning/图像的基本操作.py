# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 图像的基本操作
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/5/8 09:50
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/5/8 09:50:
==================================================  
"""
__author__ = 'zxx'
"""
读取图像，根据像素的行和列的坐标获取它的像素值，对于TGB图像而言，返回的RGB的值
"""
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("images/messi.jpg")
print(img)
blue = img[100, 100, 0]  # 截取下标为100， 100， 0的像素点
print("像素值：", blue)
plt.imshow(img)
plt.show()
# 修改某区域的值
for i in range(20):
    for j in range(20):
        img[50 + i, 235 + j] = (0, 255, 0)

plt.imshow(img)
plt.show()
"""获取图像属性"""
print(img.shape)
print(img.size)
print(img.dtype)
"""图像ROI：对于图像特定区域操作，ROI是使用numpy索引来获得的"""
ball = img[300:340, 350:390]  # 选择300：340，350：390区域作为截取对象
plt.imshow(ball)
plt.show()
# 将选定区域赋值给指定区域
img[100:140, 150:190] = ball
plt.imshow(img)
plt.show()
"""拆分并合并图像通道：有时需要把RGB拆分为单个分别操作，有时也需要把独立的通道图像合成一个RGB"""
(B, G, R) = cv2.split(img)
cv2.imshow('g', G)
cv2.waitKey()
cv2.destroyAllWindows()
"""为什么分离出来的图像颜色都是灰色的呢？因为CV2.split函数分离出来的B,G,R是单通道图像"""
# 合并通道
img = cv2.merge([R, G, B])
plt.imshow(img)
plt.show()
"""
结果说明，原图（彩色图像）是三通道图像，经过cv2.split()之后，每个通道时单通道图像。所以结果为灰色图像
其实，我们最开始想象的cv2.split()图像如下的样子：
* 蓝色通道（其它通道值为0）
* 绿色通道（其它通道值为0）
* 红色通道（其它通道值为0）
分离通道之后保留三通道，只是其他通道为0，为不是单通道
"""
img[:, :, 2] = 0  # 把所有B通道的值都为0，不必拆分复制，使用numpy索引
cv2.imshow('B', img)
cv2.waitKey()
cv2.destroyAllWindows()