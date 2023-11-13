# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> eval函数
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2023/11/13 13:58
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2023/11/13 13:58:
==================================================  
"""
__author__ = 'zxx'
"""
eval()函数是一个内置函数，用于执行存储在字符串中的Python代码。
它接受一个字符串作为参数，并将该字符串解释为有效的Python表达式或语句。
然后，eval()函数会计算并返回该表达式的结果
"""


result = eval("2 + 3")
print(result)

is_true = eval("5 > 3")
print(is_true)  # 输出：True

my_list = [1, 2, 3, 4, 5]
length = eval("len(my_list)")
print(length)  # 输出：5

