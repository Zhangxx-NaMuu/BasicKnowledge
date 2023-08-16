# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> 几种加噪声的方式
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/16 15:27
@Version: 
@License: 
@Reference: 
@History:
- 2023/8/16 15:27:
==================================================  
"""
__author__ = 'zxx'

import trimesh
import vedo
from vedo import *
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial import cKDTree


def create_grid_points_from_bounds(minimum, maximum, res):
    """基于包围盒生成网格点的坐标

    Args:
        minimum (float): 包围盒的最小值
        maximum (float): 包围盒的最大值
        res (int): 分辨率

    Returns:
        np.array: 体素化后的网格点坐标，形状：((res,)*3, 3)
    """
    axis_range = np.linspace(minimum, maximum, res)
    x_axis, y_axis, z_axis = np.meshgrid(axis_range, axis_range, axis_range, indexing='ij')
    x_axis = x_axis.reshape((np.prod(x_axis.shape),))
    y_axis = y_axis.reshape((np.prod(y_axis.shape),))
    z_axis = z_axis.reshape((np.prod(z_axis.shape),))

    points_list = np.column_stack((x_axis, y_axis, z_axis))
    del x_axis, y_axis, z_axis, axis_range
    return points_list


def normalize_meshes(mesh):
    """
  对数据进行归一化
  :param mesh: 待归一化的网格
  :return: 归一化之后的mesh
  """
    # 归一化到[-0.5, 0.5]
    bounds = mesh.bounds
    # print(bounds)
    scale = (bounds[1] - bounds[0]).max()
    # print(scale)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1 / scale)
    return mesh


if __name__ == '__main__':
    target = 'test_gum.ply'
    target_mesh = trimesh.load(target)
    # 归一化数据到（-0.5， 0.5）之间
    target_mesh = normalize_meshes(target_mesh)
    # mesh采样50000个点
    target_mesh_sample = target_mesh.sample(50000)
    # 建立立方体，范围在(-0.5,0.5)之间
    grid_coords = create_grid_points_from_bounds(-0.5, 0.5, 256)
    # 以立方体作为目标建立KDTree
    kdtree = KDTree(grid_coords)
    # 将立方体中的所有点都设为0
    occupancies = np.zeros(len(grid_coords), dtype=np.int8)

    # 在立方体中查询目标mesh的坐标，并将值设为1
    _, idx = kdtree.query(target_mesh_sample)
    occupancies[idx] = 1
    # 分别获取占用值为1和0的索引
    in_idx = np.where(occupancies == 1)
    out_idx = np.where(occupancies == 0)
    # TODO：方法一，利用KD树查找最近点
    # 利用KD树找牙龈临近点
    tree = cKDTree(grid_coords[out_idx])
    # 指定查询的最近邻数量
    k = 10  # 只查询最近的一个点，你可以根据需要调整这个值

    # 查询每个点云1中的最近邻点
    distances, indices = tree.query(grid_coords[in_idx], k=k)
    # 从点云2中提取附近的点
    nearby_points = grid_coords[out_idx][indices]
    nearby_points = nearby_points.reshape(-1, nearby_points.shape[-1])
    p1 = trimesh.PointCloud(nearby_points)

    # TODO:方法二，立方体均匀噪声
    # 添加均匀噪声--->立方体
    noise_points = np.random.uniform(-0.5, 0.5, (20000, 3))
    p2 = trimesh.PointCloud(noise_points)

    # TODO:方法三，利用长方体添加均匀噪声，更贴合mesh的形状
    min_bound = np.min(grid_coords[in_idx], axis=0)  # 点云的最小坐标值
    max_bound = np.max(grid_coords[in_idx], axis=0)  # 点云的最大坐标值
    # 定义边界盒子的中心点和边长
    boundary_center = (min_bound + max_bound) / 2
    boundary_length = np.max(max_bound - min_bound)
    noise_points = np.random.uniform(low=min_bound, high=max_bound, size=(10000, 3))
    p3 = trimesh.PointCloud(noise_points)

    # TODO:方法四，高斯噪声
    mean = (min_bound + max_bound) / 2  # 均值为边界范围的中心点
    print(min_bound)
    print(max_bound)
    print(mean)
    cov = np.diag((max_bound - min_bound) ** 2)  # 协方差矩阵，各维度上的方差为边界范围的平方
    # 生成高斯分布噪声点
    noise_points = np.random.multivariate_normal(mean, cov, 20000)
    # 对高斯噪声进行缩放，使其在(-1, 1)之间，目的是将噪声点全部集中在mesh附近
    x_range = np.max(noise_points[:, 0]) - np.min(noise_points[:, 0])
    y_range = np.min(noise_points[:, 1]) - np.max(noise_points[:, 1])
    z_range = np.min(noise_points[:, 2]) - np.max(noise_points[:, 2])
    x_scale = 2 / x_range
    y_scale = 2 / y_range
    z_scale = 2 / z_range

    scaled_point_cloud = noise_points.copy()
    scaled_point_cloud[:, 0] = scaled_point_cloud[:, 0] * x_scale
    scaled_point_cloud[:, 1] = scaled_point_cloud[:, 1] * y_scale
    scaled_point_cloud[:, 2] = scaled_point_cloud[:, 2] * z_scale
    # print(scaled_mean)
    # noise_points = np.random.multivariate_normal(scaled_mean, cov, 20000)
    p4 = trimesh.PointCloud(scaled_point_cloud)

    p = trimesh.PointCloud(grid_coords[in_idx])
    # p1 = trimesh.PointCloud(noise_points)
    p.visual.vertex_colors = [255, 0, 0]
    p1.visual.vertex_colors = [0, 255, 0]
    p2.visual.vertex_colors = [0, 0, 255]
    p3.visual.vertex_colors = [255, 255, 0]
    p4.visual.vertex_colors = [0, 255, 255]
    # pts = trimesh.PointCloud(input_pts)
    trimesh.Scene([p,  p4]).show()
