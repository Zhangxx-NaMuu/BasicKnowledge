# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> numpy矩阵连接
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/25 14:26
@Version: 
@License: 
@Reference: 
@History:
- 2023/8/25 14:26:
==================================================  
"""
__author__ = 'zxx'

import numpy as np

"""
numpy.c_() and numpy.r_()的用法
np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
c表示columns，按列拼接，应该是使列的数量增加；r表示rows，按行拼接，使行数增加。

np.hstack([a,b])，np.vstack([a,b])和np.c_，p.r_的用法区别：
1、np.hstack([a,b]) 和 np.vstack([a,b])是函数，有小括号
   np.r_[a, b] 和np.c_[a, b] 没用小括号，说明它俩不是函数
   
2、 np.r_[a, b] 和np.c_[a, b]一维数组时，作用机制应该是先把这两个一维数组转置成列向量，然后拼接。
"""

if __name__ == '__main__':
    x = np.arange(12).reshape(3, 4)
    print('x:', x, x.shape)
    y = np.arange(10, 22).reshape(3, 4)
    print('y:', y, y.shape)

    z = np.c_[x, y]
    print('z:', z, z.shape)

    q = np.r_[x, y]
    print('z:', q, q.shape)
