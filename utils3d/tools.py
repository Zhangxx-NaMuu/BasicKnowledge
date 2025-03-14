# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> tools
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/14 13:43
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/14 13:43:
==================================================  
"""
__author__ = 'zxx'
class NpEncoder(json.JSONEncoder):
    """
    Notes:
        将numpy类型编码成json格式
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)