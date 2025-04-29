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
"""
1. 如果加入了法线作为特征输入，注意下旋转平移增强；

原因：
1. 如果增强变换矩阵为正交矩阵，法线直接跟随顶点一起点积即可。
2. 如果增强变换矩阵为不是正交矩阵，法线必须跟对矩阵求逆再转置再点击再归一化处理。

表现：
1. 网络突然效果很差，训练很难拟合

解决方案：
1. 外部增强，先增强，并基于网格重新计算法线，再制作成数据集即可；
2. 训练同时增强，对随机变换矩阵进行正交化（np.linalg.qr(M) )，或者检测abs(行列式值)=1,
3. 法线特征转换成RGB法线贴图；
建议执行：
1. 外部增强。代价就多费点硬盘空间；
2. 不要用随机变换矩阵，用轴角/四元数/欧拉角生成旋转矩阵，平移，缩放，旋转分开进行，不要组合成4*4矩阵！！！！


我们的输入类型不存在非刚性变换，只有刚性变换，各自检测下代码，是否存在随机生成矩阵对法线做变换。

还有一点，不要用第三方库的apply_tansformer(M) 进行；因为内部实现无法统一；

应用变换统一为：
正向变换顺序：缩放 → 旋转 → 平移，
matmul( 输入*缩放，旋转)+平移

还原
逆平移 → 逆旋转 → 逆缩放。
matmul( 输入+平移的逆，旋转的逆)*缩放的逆

统一用matmul  ，因为我们至少3维度及以上，    
用@及dot效果是一样的（一维度有差异）
"""
import numpy as np
import torch


def angle_axis_np(angle, axis):
    """
    计算绕给定轴旋转指定角度的旋转矩阵。

    Args:
        angle (float): 旋转角度（弧度）。
        axis (np.ndarray): 旋转轴，形状为 (3,) 的 numpy 数组。

    Returns:
        np.array: 3x3 的旋转矩阵，数据类型为 np.float32。
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                               [u[2], 0.0, -u[0]],
                               [-u[1], u[0], 0.0]])
    R = cosval * np.eye(3) + sinval * cross_prod_mat + (1.0 - cosval) * np.outer(u, u)
    return R


class Flip_np:
    def __init__(self, axis_x=True, axis_y=True, axis_z=True):
        """
        用于随机翻转点云数据。

        Args:
            axis_x (bool): 是否在 x 轴上进行翻转，默认为 True。
            axis_y (bool): 是否在 y 轴上进行翻转，默认为 True。
            axis_z (bool): 是否在 z 轴上进行翻转，默认为 True。
        """
        self.axis_x, self.axis_y, self.axis_z = axis_x, axis_y, axis_z

    def __call__(self, points):
        # 初始化翻转因子，默认为1（不翻转）
        flip_factors = np.ones(3)

        # 根据指定的轴生成伯努利分布的翻转因子（-1或1）
        if self.axis_x:
            flip_factors[0] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_y:
            flip_factors[1] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_z:
            flip_factors[2] = np.random.choice([-1, 1], p=[0.5, 0.5])

        # 仅影响指定轴方向
        normals = points.shape[1] > 3
        points[:, 0:3] *= flip_factors
        if normals:
            points[:, 3:6] *= flip_factors

        return points


class Scale_np:
    def __init__(self, lo=0.8, hi=1.25):
        """
        初始化 Scale 类，用于随机缩放点云数据。

        Args:
            lo (float): 缩放因子的下限，默认为 0.8。
            hi (float): 缩放因子的上限，默认为 1.25。
        """
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class RotateAxis_np:
    def __init__(self, axis=[0.0, 0.0, 0.0]):
        """
        初始化 RotateAxis 类，用于绕指定轴随机旋转点云数据。

        Args:
            axis (np.ndarray): 旋转轴，形状为 (3,),默认为 [0.0, 0.0, 0.0]（z 轴）。
        """
        self.axis = np.array(axis)

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis_np(rotation_angle, self.axis)

        normals = points.shape[1] > 3
        if not normals:
            return np.matmul(points, rotation_matrix.T)
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:6]
            points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
            points[:, 3:6] = np.matmul(pc_normals, rotation_matrix.T)

            return points


class RotateXYZ_np(object):
    def __init__(self, angle_sigma=0.05, angle_clip=0.15):
        """
        用于在三个轴上随机微扰旋转点云数据。

        Args:
            angle_sigma (float): 旋转角度的高斯分布标准差，默认为 0.05。
            angle_clip (float): 旋转角度的裁剪范围，默认为 0.15。
        """
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def __call__(self, points):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )
        Rx = angle_axis_np(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis_np(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis_np(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)

        normals = points.shape[1] > 3
        if not normals:
            return np.matmul(points, rotation_matrix.T)
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:6]
            points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
            points[:, 3:6] = np.matmul(pc_normals, rotation_matrix.T)

            return points


class Jitter_np(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        用于给点云数据添加随机抖动。

        Args:
            std (float): 抖动的高斯分布标准差，默认为 0.01。
            clip (float): 抖动的裁剪范围，默认为 0.05。
        """
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = np.random.normal(loc=0.0, scale=self.std, size=(points.shape[0], 3))
        jittered_data = np.clip(jittered_data, -self.clip, self.clip)
        points[:, 0:3] += jittered_data
        return points


class Translate_np(object):
    def __init__(self, translate_range=0.1):
        """
        用于随机平移点云数据。

        Args:
            translate_range (float): 平移的范围，默认为 0.1。
        """
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class RandomDropout_np(object):
    def __init__(self, max_dropout_ratio=0.5, return_idx=False):
        """
        用于随机丢弃点云数据中的点。

        Args:
            max_dropout_ratio (float): 最大丢弃比例，范围为 [0, 1)，默认为 0.5。
        """
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.return_idx = return_idx

    def __call__(self, pc, ):
        dropout_ratio = np.random.random() * self.max_dropout_ratio
        ran = np.random.random(pc.shape[0])
        # 找出需要保留的点的索引
        keep_idx = np.where(ran > dropout_ratio)[0]
        if self.return_idx:
            return keep_idx
        else:
            return pc[keep_idx]


class Normalize_np:
    def __init__(self, method="std", v_range=[0, 1]):
        self.method = method
        self.v_range = v_range
        if len(self.v_range) != 2 or not (self.v_range[0] < self.v_range[1]):
            raise ValueError("v_range 需为包含两个元素的列表/元组，且 min_val < max_val")

    def __call__(self, points):
        vertices = points[:, 0:3]
        if self.method == "std":
            centroid = np.mean(vertices, axis=0)
            vertices = vertices - centroid
            m = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
            vertices = vertices / m
            return vertices
        else:
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            vertices = (vertices - vmin) / (vmax - vmin).max()
            # 再从 [0, 1] 区间缩放到 [min_val, max_val] 区间
            vertices = vertices * (self.v_range[-1] - self.v_range[0]) + self.v_range[0]

        points[:, 0:3] = vertices
        return points


def angle_axis(angle, axis):
    """
    计算绕给定轴旋转指定角度的旋转矩阵（PyTorch版本）。

    Args:
        angle (float): 旋转角度（弧度）。
        axis (torch.Tensor): 旋转轴，形状为 (3,) 的Tensor。

    Returns:
        torch.Tensor: 3x3 的旋转矩阵，数据类型为 torch.float32。
    """
    u = axis / torch.norm(axis)
    cosval = torch.cos(angle)
    sinval = torch.sin(angle)

    # 构建叉乘矩阵
    cross_prod_mat = torch.tensor([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0]
    ], device=axis.device, dtype=torch.float32)

    R = (
            cosval * torch.eye(3, device=axis.device) +
            sinval * cross_prod_mat +
            (1.0 - cosval) * torch.outer(u, u)
    )
    return R


class Flip:
    """对点云数据进行随机翻转增强。

    Attributes:
        axis_x (bool): 是否启用X轴翻转，默认为True
        axis_y (bool): 是否启用Y轴翻转，默认为True
        axis_z (bool): 是否启用Z轴翻转，默认为True
    """

    def __init__(self, axis_x=True, axis_y=True, axis_z=True):
        """
        初始化翻转增强器。

        Args:
            axis_x (bool, optional): 是否沿X轴翻转. Defaults to True.
            axis_y (bool, optional): 是否沿Y轴翻转. Defaults to True.
            axis_z (bool, optional): 是否沿Z轴翻转. Defaults to True.
        """
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z

    def __call__(self, points):
        """
        对输入点云应用随机翻转变换。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)（包含法线）

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        flip_factors = torch.ones(3, device=points.device)

        if self.axis_x:
            flip_factors[0] = 1 if torch.rand(1, device=points.device) < 0.5 else -1
        if self.axis_y:
            flip_factors[1] = 1 if torch.rand(1, device=points.device) < 0.5 else -1
        if self.axis_z:
            flip_factors[2] = 1 if torch.rand(1, device=points.device) < 0.5 else -1

        normals = points.size(1) > 3
        points[:, 0:3] *= flip_factors
        if normals:
            points[:, 3:6] *= flip_factors
        return points


class Scale:
    """对点云数据进行随机缩放增强。"""

    def __init__(self, lo=0.8, hi=1.25):
        """
        初始化缩放增强器。

        Args:
            lo (float, optional): 缩放下限. Defaults to 0.8.
            hi (float, optional): 缩放上限. Defaults to 1.25.
        """
        self.lo = lo
        self.hi = hi

    def __call__(self, points):
        """
        对输入点云应用随机缩放变换。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        scaler = torch.empty(1, device=points.device).uniform_(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class RotateAxis:
    """绕指定轴随机旋转点云。"""

    def __init__(self, axis=[0.0, 0.0, 1.0]):
        """
        初始化旋转增强器。

        Args:
            axis (torch.Tensor, optional): 旋转轴向量，形状为 (3,). Defaults to Z轴.
        """
        self.axis = torch.tensor(axis)

    def __call__(self, points):
        """
        应用绕轴随机旋转变换。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        rotation_angle = torch.empty(1).uniform_(0, 2 * torch.pi).to(points.device)
        rotation_matrix = angle_axis(rotation_angle, self.axis.to(dtype=torch.float32, device=points.device))

        normals = points.size(1) > 3
        points[:, 0:3] = torch.mm(points[:, 0:3], rotation_matrix.t())
        if normals:
            points[:, 3:6] = torch.mm(points[:, 3:6], rotation_matrix.t())
        return points


class RotateXYZ:
    """绕XYZ轴应用随机欧拉角旋转。"""

    def __init__(self, angle_sigma=0.05, angle_clip=0.15):
        """
        初始化旋转增强器。

        Args:
            angle_sigma (float, optional): 旋转角度的标准差. Defaults to 0.05.
            angle_clip (float, optional): 旋转角度的截断范围. Defaults to 0.15.
        """
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, points):
        """
        应用三维随机旋转变换。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3, device=points.device),
            -self.angle_clip, self.angle_clip
        )

        Rx = angle_axis(angles[0], torch.tensor([1.0, 0.0, 0.0], device=points.device))
        Ry = angle_axis(angles[1], torch.tensor([0.0, 1.0, 0.0], device=points.device))
        Rz = angle_axis(angles[2], torch.tensor([0.0, 0.0, 1.0], device=points.device))

        rotation_matrix = torch.mm(torch.mm(Rz, Ry), Rx)

        normals = points.size(1) > 3
        points[:, 0:3] = torch.mm(points[:, 0:3], rotation_matrix.t())
        if normals:
            points[:, 3:6] = torch.mm(points[:, 3:6], rotation_matrix.t())
        return points


class Jitter:
    """对点云坐标添加随机噪声。"""

    def __init__(self, std=0.01, clip=0.05):
        """
        初始化噪声增强器。

        Args:
            std (float, optional): 噪声标准差. Defaults to 0.01.
            clip (float, optional): 噪声截断范围. Defaults to 0.05.
        """
        self.std = std
        self.clip = clip

    def __call__(self, points):
        """
        应用噪声扰动。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        jitter = torch.clamp(
            torch.randn(points[:, 0:3].shape, device=points.device) * self.std,
            -self.clip, self.clip
        )
        points[:, 0:3] += jitter
        return points


class Translate:
    """对点云应用随机平移变换。"""

    def __init__(self, translate_range=0.1):
        """
        初始化平移增强器。

        Args:
            translate_range (float, optional): 平移范围. Defaults to 0.1.
        """
        self.translate_range = translate_range

    def __call__(self, points):
        """
        应用平移变换。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, 3) 或 (N, 6)

        Returns:
            torch.Tensor: 变换后的点云数据
        """
        translation = torch.empty(3, device=points.device).uniform_(
            -self.translate_range, self.translate_range
        )
        points[:, 0:3] += translation
        return points


class RandomDropout:
    """随机丢弃部分点云数据。"""

    def __init__(self, max_dropout_ratio=0.2):
        """
        初始化丢弃增强器。

        Args:
            max_dropout_ratio (float, optional): 最大丢弃比例. Defaults to 0.2.
        """
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        """
        应用随机丢弃。

        Args:
            pc (torch.Tensor): 输入点云数据，形状为 (N, C)

        Returns:
            torch.Tensor: 丢弃后的点云数据
        """
        dropout_ratio = torch.rand(1).item() * self.max_dropout_ratio
        mask = torch.rand(pc.size(0), device=pc.device) > dropout_ratio

        if mask.any():
            pc = pc[mask]
        return pc


class Normalize:
    """对点云进行归一化处理。"""

    def __init__(self, method="std", v_range=[0, 1]):
        """
        初始化归一化处理器。

        Args:
            method (str, optional): 归一化方法，可选"std"（标准化）或其他（最大最小归一化）.
                                    Defaults to "std".
            v_range (list, optional): 当使用最大最小归一化时的目标范围. Defaults to [0, 1].
        """
        self.method = method
        self.v_range = v_range

    def __call__(self, points):
        """
        应用归一化处理。

        Args:
            points (torch.Tensor): 输入点云数据，形状为 (N, C)

        Returns:
            torch.Tensor: 归一化后的点云数据
        """
        vertices = points[:, 0:3]
        if self.method == "std":
            centroid = torch.mean(vertices, dim=0)
            vertices -= centroid
            m = torch.max(torch.norm(vertices, dim=1))
            vertices /= m
        else:
            vmin = torch.min(vertices, dim=0)[0]
            vmax = torch.max(vertices, dim=0)[0]
            vertices = (vertices - vmin) / (vmax - vmin).max()
            vertices = vertices * (self.v_range[1] - self.v_range[0]) + self.v_range[0]
        points[:, 0:3] = vertices
        return points


class ToTensor:
    """将输入数据转换为torch.Tensor格式。"""

    def __init__(self, device="cpu"):
        """
        初始化转换器。

        Args:
            device (str, optional): 目标设备. Defaults to "cpu".
        """
        self.device = device

    def __call__(self, points):
        """
        执行数据类型转换和设备转移。

        Args:
            points (numpy.ndarray | torch.Tensor): 输入点云数据

        Returns:
            torch.Tensor: 转换后的张量
        """
        if "numpy" in str(type(points)):
            return torch.from_numpy(points).to(dtype=torch.float32, device=self.device)
        else:
            return points.to(dtype=torch.float32, device=self.device)


class RandomCrop:
    def __init__(self, radius=0.15):
        """
        随机移除一个点周围指定半径内的所有点

        Args:
            radius (float): 移除半径，默认为0.15
        """
        assert radius >= 0, "Radius must be non-negative"
        self.radius = radius

    def __call__(self, inputs):
        from scipy.spatial import KDTree
        # 提取点云坐标（保留所有通道）
        points = inputs[..., :3]
        # 随机选择一个中心点
        center_idx = np.random.randint(len(points))
        center = points[center_idx]
        # 构建KDTree加速搜索
        tree = KDTree(points)
        # 查询半径范围内的点索引
        remove_indices = tree.query_ball_point(center, self.radius)
        # 生成保留掩码（排除要删除的点）
        mask = np.ones(len(points), dtype=bool)
        mask[remove_indices] = False

        return inputs[mask]


if __name__ == "__main__":
    from torchvision import transforms

    transforms_torch = transforms.Compose(
        [
            ToTensor(device="cuda:0"),
            Normalize(method="MaxMix", v_range=[0, 1]),
            RotateAxis(axis=[0, 1, 0]),
            RotateXYZ(angle_sigma=0.05, angle_clip=0.15),
            Scale(lo=0.8, hi=1.25),
            Translate(translate_range=0.1),
            Jitter(std=0.01, clip=0.05),
            RandomDropout(max_dropout_ratio=0.2),
            Flip(axis_x=False, axis_y=False, axis_z=True),
        ]
    )

    transforms_np = transforms.Compose(
        [

            Normalize_np(method="MaxMix", v_range=[0, 1]),
            RotateAxis_np(axis=[0, 1, 0]),
            RotateXYZ_np(angle_sigma=0.05, angle_clip=0.15),
            Scale_np(lo=0.8, hi=1.25),
            Translate_np(translate_range=0.1),
            Jitter_np(std=0.01, clip=0.05),
            RandomDropout_np(max_dropout_ratio=0.2),
            Flip_np(axis_x=False, axis_y=False, axis_z=True),
            ToTensor(device="cuda:0"),
        ]
    )

    # 示例数据
    points = np.random.randn(1024, 6)
    points[:, 3:6] = np.random.rand(1024, 3)
    import time

    s = time.time()
    for i in range(50):
        transformed_points = transforms_torch(points)
    e1 = time.time()
    for i in range(50):
        transformed_points_np = transforms_np(points)
    e2 = time.time()

    print(e1 - s, transformed_points.shape, transformed_points.max(), transformed_points.min())
    print(e2 - e1, transformed_points_np.shape, transformed_points_np.max(), transformed_points_np.min())

    # 0.39881253242492676 torch.Size([910, 6]) tensor(1.3718, device='cuda:0') tensor(-0.5744, device='cuda:0')
    # 0.031032800674438477 torch.Size([856, 6]) tensor(1.3002, device='cuda:0') tensor(-1.2248, device='cuda:0')