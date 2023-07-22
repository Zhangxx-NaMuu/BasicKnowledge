# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> trimesh_learning
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/7/19 16:48
@Version: 
@License: 
@Reference: trimesh库的相关函数的学习笔记
@History:
- 2023/7/19 16:48:
==================================================  
"""
__author__ = 'zxx'

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.cluster import DBSCAN
import numpy as np


def compute_edges_unique():
    """
    获取mesh的边缘线
    :return:
    """
    mesh = trimesh.load(r"D:\data\crown_data\4\crop_data\lower\opposite\J10124454234_44.stl")

    # 获取mesh的边界轮廓线
    boundary_edges = mesh.edges_unique
    print(boundary_edges)
    # 提取边界轮廓线的顶点坐标
    lines = mesh.vertices[boundary_edges]
    print(lines)
    # 绘制边界轮廓线
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line_collection = Line3DCollection(lines, colors='r', linewidths=2.0)
    ax.add_collection(line_collection)
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    plt.show()

    # points = mesh.vertices
    # # 使用DBSCAN算法进行点云聚类
    # clustering = DBSCAN(eps=0.01, min_samples=5000).fit(points)
    # # 获取位于聚类边界的点
    # boundary_points = points[clustering.labels_ == -1]
    # 获取mesh的边界点
    boundary_points = []
    for vertex in mesh.vertices:
        is_boundary = False
        for face in mesh.faces:
            if np.sum(face == vertex) > 1:
                is_boundary = True
                break
        if is_boundary:
            boundary_points.append(vertex)
    #
    # # 转换为NumPy数组
    boundary_points = np.array(boundary_points)
    # 绘制边界点
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], c='r')
    plt.show()


if __name__ == '__main__':
    compute_edges_unique()
