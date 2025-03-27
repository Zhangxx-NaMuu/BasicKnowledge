import numpy as np
import math


def dataAugment(self, xyz, jitter=False, flip=False, rot=False, jitter_std=0.1):
    """对三维点云数据进行数据增强，支持任意旋转轴和噪声控制

    变换包含（启用时按顺序执行）：
    1. 抖动：向变换矩阵添加可控幅度的随机噪声
    2. 翻转：随机翻转x轴方向
    3. 旋转：绕任意轴进行随机角度旋转

    注意：
    - 变换组合顺序为：抖动 -> 翻转 -> 旋转
    - 矩阵乘法顺序为右乘，即最终变换矩阵 m = m_jitter @ m_flip @ m_rot
    - 点云坐标采用矩阵右乘方式变换：xyz' = xyz @ m

    Args:
        xyz (np.ndarray): 输入点云 (N,3)
        jitter (bool): 是否添加矩阵抖动 (默认False)
        flip (bool): 是否随机x轴翻转 (默认False)
        rot (bool): 是否任意轴旋转 (默认False)
        jitter_std (float): 噪声标准差 (默认0.1)

    Returns:
        np.ndarray: 增强后的点云 (N,3)

    Example:
        >>> aug_xyz = dataAugment(xyz, jitter=True, rot=True, jitter_std=0.2)
    """
    # 初始化单位变换矩阵
    m = np.eye(3)

    # 添加可控幅度噪声
    if jitter:
        # 生成正态分布噪声矩阵，标准差为jitter_std
        m += np.random.randn(3, 3) * jitter_std

    # X轴随机反射变换
    if flip:
        # 生成伯努利分布的翻转因子（-1或1）
        flip_factor = np.random.choice([-1, 1], p=[0.5, 0.5])
        # 仅影响X轴方向
        m[0, 0] *= flip_factor

    # 任意轴随机旋转
    if rot:
        # 生成随机旋转轴（三维均匀分布）
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)  # 单位化

        # 生成随机旋转角度[0, 2π)
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