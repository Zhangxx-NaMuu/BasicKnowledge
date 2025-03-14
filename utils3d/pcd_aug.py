# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> pcd_aug
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/14 14:56
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/14 14:56:
==================================================  
"""
__author__ = 'zxx'
import numpy as np

def dataAugment(xyz, jitter=False, flip=False, rot=False, jitter_std=0.1):
    """
    对三维点云数据进行数据增强，支持任意旋转轴和噪声控制

    变换包含（启用时按顺序执行）：
    1.抖动：向变换矩阵添加可控幅度的随机噪声
    2.翻转：随机反转X轴方向
    3.旋转：随机旋转任意轴

    注意：
    - 变换组合顺序为：抖动->翻转->旋转
    - 矩阵惩罚顺序为右乘，即最终变换矩阵 m = m_jitter @ m_flip @ m_rot
    - 点云坐标采用矩阵右乘方式变换： xyz' = xyz @ m
    Args:
        xyz: (np.ndarray) 输入点云数据，shape=(N, 3)
        jitter: (bool) 是否启用抖动， 默认False
        flip: (bool) 是否随机X轴翻转， 默认False
        rot: (bool) 是否随机旋转， 默认False
        jitter_std: (float) 抖动幅度，噪声标准差 默认0.1
    Returns:
        np.ndarray: 增强后的点云数据，shape=(N, 3)

    Example:
        >>> aug_xyz = dataAugment(xyz, jitter=True, rot=True, jitter_std=0.2)
    """
    # 初始化单位变换矩阵
    m = np.eye(4)

    # 添加可控幅度噪声
    if jitter:
        # 生成正太分布噪声矩阵，标准差为jitter_std
        m += np.random.randn(3, 3) * jitter_std

    # X轴随机反射变换
    if flip:
        # 生成伯努利分布的反转因子(-1或1)
        flip_factor = np.random.choice([-1, 1], p=[0.5, 0.5])
        # 仅影响X轴方向
        m[0, 0] *= flip_factor

    # 任意轴随机旋转
    if rot:
        # 随机选择旋转轴(三维均匀分布)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)  # 单位化

        # 生成随即旋转角度[0, 2pi]
        angle = np.random.uniform(0, 2 * np.pi)

        # 通过罗德里格斯公式生成旋转矩阵
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        t = 1 - cos_t

        # 旋转矩阵分量计算
        rot_m = np.array([
            [cos_t + t * axis ** 2,
             t * axis * axis - sin_t * axis,
             t * axis * axis + sin_t * axis],
            [t * axis * axis + sin_t * axis,
             cos_t + t * axis ** 2,
             t * axis[1] * axis - sin_t * axis],
            [t * axis * axis - sin_t * axis,
             t * axis * axis + sin_t * axis,
             cos_t + t * axis ** 2]
        ])

        # 组合旋转变换
        m = m @ rot_m

    # 应用复合变换矩阵
    return xyz @ m