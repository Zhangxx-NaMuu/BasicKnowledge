# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> test
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/28 10:16
@Version: 
@License: 
@Reference: 
@History:
- 2023/7/28 10:16:
==================================================  
"""
__author__ = 'zxx'

import numpy as np


def crown_generate(in_vec, in_face_id, generateModel, toothNumber):
    in_vec = np.array(in_vec).reshape(-1, 3)
    in_face_id = np.array(in_face_id).reshape(-1, 3)

    # 归一化
    if generateModel == 'True':
        model_path = toothNumber + generateModel
    else:
        model_path = toothNumber + generateModel + 1

    out_vec = in_vec.tolist()
    out_face_id = in_face_id.tolist()
    return {"points": out_vec, "faces": out_face_id}


if __name__ == '__main__':

    print(1)