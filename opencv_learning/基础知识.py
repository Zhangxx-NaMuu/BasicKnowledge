# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> 基础知识
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/4/30 14:30
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/4/30 14:30:
==================================================  
"""
__author__ = 'zxx'
import cv2

img = cv2.imread('images/messi.jpg')
h, w = img.shape[:2]
print('图像高{}，图像宽{}'.format(h, w))

# 展示图像
cv2.imshow('image', img)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyWindow()

# 保存图像
cv2.imwrite('result.png', img)
"""
在opencv中，图像颜色通道是BGR的顺序（即RGB的倒序），不是传统的RGB颜色通道，但是可以通过cvColor进行颜色空间的转换。
OpenCV颜色转换代码中，最有用的一些转换代码如下：
* cv2.COLOR_BGR2GRAY：将BGR彩色图像转换为灰度图像
* cv2.COLOR_BGR2RGB：将BGR彩色图像转化为RGB图像
* cv2.COLOR_GRAY2BGR：将灰度图像转化为BGR彩色图像
上面每个转换代码中，转换后的图像颜色通道与对应的转换代码相匹配，比如对于灰度图只有一个通道，对于RGB和BGR图像则有三个通道。
最后的cv2.COLOR_GRAY2BGR将灰度图转换成BGR彩色图像；如果你想在图像上绘制或覆盖有彩色的对象，cv2.COLOR_GAY2BGR是非常有用的，我们会在后面的例子中用到它。
"""
import matplotlib.pyplot as plt
plt.imshow(img)
plt.show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


