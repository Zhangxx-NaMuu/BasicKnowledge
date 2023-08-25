# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> numpy数组切片
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/2 15:16
@Version: 
@License: 
@Reference: 
@History:
- 2023/8/2 15:16:
==================================================  
"""
__author__ = 'zxx'

import numpy as np

a = [1, 2, 3, 4, 5]

# 一个参数
print(a[2])  # 取下表为2的元素[3]
print(a[-1])  # 取最后一个元素[5]

# 两个参数
print(a[1:3])  # 取下表从1， 2的元素怒[2, 3]

# 三个参数 a[i:j:s],s代表步长
print(a[0:3:1])  # 取步长为1，下标0，1，2的元素[1, 2, 3]
print(a[::-1])  # 步长为-1时代表翻转读取[5,4,3,2,1]
print(a[2::-1])  # [3, 2, 1]

