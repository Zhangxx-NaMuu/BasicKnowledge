# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> re_test
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2024/4/8 17:17
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 正则表达式
@History:
- 2024/4/8 17:17:
==================================================  
"""
__author__ = 'zxx'

import re

line = "Cats are smarter than dogs"
matchObj = re.match(r'(.*) are (.*?) .*', line, re.M | re.I)
if matchObj:
    print("matchObj.group() : ", matchObj.group())
    print("matchObj.group(1) : ", matchObj.group(1))
    print("matchObj.group(2) : ", matchObj.group(2))
else:
    print("No match!!")
print(matchObj)
