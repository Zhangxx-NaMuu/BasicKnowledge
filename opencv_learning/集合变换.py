# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 集合变换
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/5/8 10:20
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/5/8 10:20:
==================================================  
"""
__author__ = 'zxx'
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/messi.jpg")
"""
扩展缩放：cv2.resize()可以实现这个功能。在缩放时推荐cv2.INTER_AREA，在扩展时推荐cv2.INTER_CUBIC(慢)和cv2.INTER_LINEAR，默认情况下所有改变图像尺寸的操作使用的插值法都是cv2.INTER_LINEAR
"""""
height, width = img.shape[:2]
res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.show()
"""
平移:如果想要沿(x,y)方向移动，移动的距离为(tx,ty)，使用numpy数组构建矩阵，数据类型是np.float32，然后传给函数cv2.warpAffine()，函数cv2.warpAffine()的第三个参数的是输出图像的大小
它的格式应该是图像（宽，高），应该记住的是图像的宽对应的是列数，高对应的是行数
"""
img = cv2.imread("images/messi.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

M = np.float32([[1, 0, 100], [0, 1, 50]])  # 图像沿x轴移动100， 沿y轴移动50
dst = cv2.warpAffine(img, M, (cols, rows))

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.subplot(1, 2, 2)
plt.imshow(dst)
plt.show()

# 旋转
img = cv2.imread('images/messi.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape

M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
print(M)
dst1 = cv2.warpAffine(img, M, (cols, rows))
dst2 = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_WRAP)  # 表示以平铺方式进行边缘处理
dst3 = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)  # 表示已镜像方式进行边缘处理

plt.figure(figsize=(16, 3))
plt.subplot(1, 4, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 4, 2)
plt.imshow(dst1, cmap='gray')
plt.subplot(1, 4, 3)
plt.imshow(dst2, cmap='gray')
plt.subplot(1, 4, 4)
plt.imshow(dst3, cmap='gray')
plt.show()

"""
仿射变换：又称仿射映射，是指在几何中，一个向量空间进行一次线性变换并接上上一个平移，变换为另一个向量空间。
在仿射变换中，原图中所有平行线在结果图像中同样平行。为创建这个矩阵，需要从原图像中找到三个点以及他们在输出图像中的位置，然后使用cv2.getAffineTransForm()
会创建一个2*3的矩阵，最后这个矩阵会被传给函数cv2.warpAffine()
"""
img = cv2.imread("images/drawing.png")
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
print(M)

dst = cv2.warpAffine(img, M, (cols, rows))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(dst, cmap='gray')
plt.show()
"""
透视变换：是指利用透视中心、像点、目标点三点共线的条件，按透视旋转定律使承影面（透视面）绕迹线（透视轴）旋转某一角度，破坏原有的
投影光线束，仍能保持承影面上投影几何图形不变的变换。
对于视角变换，我们需要一个3*3变幻矩阵，在变换前后直线还是直线。需要在原图上找到4个点，以及他们在输出图上对应的位置，这四个点中任意三个都不能共线，
可以有函数cv2.getPerspectiveTransform()构建，然后这个矩阵传给函数cv2.warpPerspective()
"""
img1 = cv2.imread("images/messi.jpg")
img2 = cv2.imread("images/messi.jpg")
rows1, cols1, ch1 = img1.shape
rows2, cols2, ch2 = img2.shape

# 透视面的坐标
pts1 = np.float32([0, 0], (cols1-1, 0), (cols2-1, rows2-1), (0, rows2-1))
pts2 = np.float32([(671,314),(1084,546),(689,663),(386,361)])

M = cv2.getPerspectiveTransform(pts1,pts2)
print(M)
img3 = np.copy(img1)
cv2.warpPerspective(img2,M,(cols1,rows1),img3,borderMode=cv2.BORDER_TRANSPARENT)

plt.figure(figsize=(14,3))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
plt.show()

