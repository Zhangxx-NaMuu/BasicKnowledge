# -*- coding: UTF-8 -*-
"""
==================================================
@path   ：BasicKnowledge -> normalize_mesh
@IDE    ：PyCharm
@Author ：
@Email  ：2458543125@qq.com
@Date   ：2023/8/22 14:54
@Version: 
@License: 
@Reference: 用open3d和trimesh两种方式做归一化处理
@History:
- 2023/8/22 14:54:
==================================================  
"""
__author__ = 'zxx'

import trimesh
import open3d as o3d
import numpy as np


def normalize_meshes_trimesh(mesh):
    """
    针对加入对颌数据之后做归一化，保留位移信息和缩放尺寸，便于对中间牙操作
    对数据进行归一化
    :param mesh: 待归一化的网格
    :return: 归一化之后的mesh
    """
    # 归一化到[-0.5,0.5]
    bounds = mesh.bounds
    scale = (bounds[1] - bounds[0]).max()
    displacement = -mesh.bounding_box.centroid
    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_scale(1 / scale)
    return mesh, scale, displacement


def normalize_meshes_open3d(mesh):
    """
        针对加入对颌数据之后做归一化，保留位移信息和缩放尺寸，便于对中间牙操作
        对数据进行归一化
        :param mesh: 待归一化的网格
        :return: 归一化之后的mesh
        """
    # 归一化到[-0.45,0.45]
    # 获取边界框
    bounds = mesh.get_axis_aligned_bounding_box()

    # 计算缩放因子和位移向量
    scale = (bounds.get_max_bound() - bounds.get_min_bound()).max()
    displacement = -bounds.get_center()

    # 平移网格
    mesh.translate(displacement)

    # 缩放网格
    mesh.scale(1 / scale, [0, 0, 0])

    return mesh, scale, displacement


if __name__ == '__main__':
    # path = 'test_gum.ply'
    # mesh1 = trimesh.load(path)
    # mesh2 = o3d.io.read_triangle_mesh(path)
    # # 测试open3d 和 trimesh归一化的差别
    # mesh1, scale1, displacement1 = normalize_meshes_trimesh(mesh1)
    # mesh2, scale2, displacement2 = normalize_meshes_open3d(mesh2)
    # print(scale1, displacement1)
    # print(scale2, displacement2)
    #
    # mesh1.export(r'C:\Users\dell\Desktop\1.ply')
    # o3d.io.write_triangle_mesh(r'C:\Users\dell\Desktop\2.ply', mesh2)
    #
    # # 还原归一化后的模型
    # mesh1.apply_scale(float(scale1))
    # mesh1.apply_translation(-displacement1)
    # mesh2.scale(scale2, center=[0, 0, 0])
    # mesh2.translate(-displacement2)
    # mesh1.export(r'C:\Users\dell\Desktop\3.ply')
    # o3d.io.write_triangle_mesh(r'C:\Users\dell\Desktop\4.ply', mesh2)

    import vedo

    mesh_path = 'test_gum.ply'
    mesh = vedo.load(mesh_path)
    vertices = mesh.points()  # 顶点
    faces = mesh.faces()  # 面片索引

    # 已知顶点和面片，组成mesh
    mesh1 = vedo.Mesh([vertices, faces])

    # 可视化mesh
    vp = vedo.Plotter(N=2)
    vp.at(0).show([mesh])
    vp.at(1).show([mesh1])
    vp.interactive().close()