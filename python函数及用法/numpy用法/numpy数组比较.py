# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> numpy数组比较
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/24 15:56
@Version: 
@License: 
@Reference: 
@History:
- 2023/8/24 15:56:
==================================================  
"""
__author__ = 'zxx'

import numpy as np

if __name__ == '__main__':
    array1 = np.array([1, 2, 3])
    array2 = np.array([1, 2, 3])

    """
    使用numpy.array_equal()函数比较两个数组是否相同。将要比较的两个数组作为参数传递给该函数。
    numpy.array_equal()
    函数将返回一个布尔值，指示两个数组是否相同。如果相同，则返回True；如果不同，则返回False。
    """
    result = np.array_equal(array1, array2)

    '''
    使用numpy.not_equal()函数比较两个数组的对应元素是否相同。该函数将返回一个布尔值的数组，表示两个数组中每个元素是否不同。
    '''
    not_equal_elements = np.not_equal(array1, array2)

    '''
    使用numpy.sum()函数计算不同元素的数量。将布尔值的数组作为参数传递给numpy.sum()函数，它会将True视为1，False视为0，并计算总和
    使用numpy.abs()函数获取两个数组中不同元素的绝对差值：
    使用numpy.max()函数获取绝对差值的最大值，即两个数组中最大差距的大小：
    '''
    different_count = np.sum(not_equal_elements)
    absolute_diff = np.abs(array1 - array2)
    max_diff = np.max(absolute_diff)

    '''
    要比较NumPy数组中的整数部分，您可以使用numpy.floor函数来获得每个元素的下限（即最大整数）
    要比较NumPy数组中的浮点数部分，您可以使用numpy.modf函数来获得每个元素的小数部分
    '''
    import numpy as np

    # 创建一个带有浮点数的NumPy数组
    arr = np.array([1.5, 2.7, 3.9, 4.2, 5.8])

    # 获取整数部分
    integer_part = np.floor(arr)

    # 获取浮点数部分
    fractional_part, _ = np.modf(arr)

