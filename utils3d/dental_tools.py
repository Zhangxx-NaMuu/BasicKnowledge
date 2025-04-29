# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> dental_tools
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/27 15:52
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/27 15:52:
==================================================  
"""
__author__ = 'zxx'
"""


专注于牙颌mesh的特殊实现

"""
__author__ = 'sindre'

import vedo
import numpy as np
from typing import *
from sklearn.decomposition import PCA
from sindre.utils3d.algorithm import apply_transform, cut_mesh_point_loop, subdivide_loop_by_trimesh


def convert_fdi2idx(labels):
    """

    将口腔牙列的fid (11-18,21-28,31-38,41-48) 转换成1-18;

    """

    if labels.max() > 30:
        labels -= 20
    labels[labels // 10 == 1] %= 10
    labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
    labels[labels < 0] = 0
    return labels


def convert_labels2color(data: Union[np.array, list]) -> list:
    """
        将牙齿标签转换成RGBA颜色

    Notes:
        只支持以下标签类型：

            upper_dict = [0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]

            lower_dict = [0, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

    Args:
        data: 属性

    Returns:
        colors: 对应属性的RGBA类型颜色

    """

    colormap_hex = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
                    '#dcbeff',
                    '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075', '#a9a9a9', '#ffffff', '#000000'
                    ]
    hex2rgb = lambda h: list(int(h.lstrip("#")[i: i + 2], 16) for i in (0, 2, 4))
    colormap = [hex2rgb(h) for h in colormap_hex]
    upper_dict = [0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    lower_dict = [0, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

    if max(data) in upper_dict:
        colors = [colormap[upper_dict.index(data[i])] for i in range(len(data))]
    else:
        colors = [colormap[lower_dict.index(data[i])] for i in range(len(data))]
    return colors


def transform_crown(near_mesh: vedo.Mesh, jaw_mesh: vedo.Mesh) -> vedo.Mesh:
    """
        调整单冠的轴向

    Tip:
        1.通过连通域分割两个邻牙;

        2.以邻牙质心为确定x轴；

        3.通过找对颌最近的点确定z轴方向;如果z轴方向上有mesh，则保持原样，否则将z轴取反向;

        4.输出调整后的牙冠


    Args:
        near_mesh: 两个邻牙组成的mesh
        jaw_mesh: 两个邻牙的对颌

    Returns:
        变换后的单冠mesh

    """
    vertices = near_mesh.points()
    # 通过左右邻牙中心指定x轴
    m_list = near_mesh.split()
    center_vec = m_list[0].center_of_mass() - m_list[1].center_of_mass()
    user_xaxis = center_vec / np.linalg.norm(center_vec)

    # 通过找对颌最近的点确定z轴方向
    jaw_mesh = jaw_mesh.split()[0]
    jaw_near_point = jaw_mesh.closest_point(vertices.mean(0))
    jaw_vec = jaw_near_point - vertices.mean(0)
    user_zaxis = jaw_vec / np.linalg.norm(jaw_vec)

    components = PCA(n_components=3).fit(vertices).components_
    xaxis, yaxis, zaxis = components

    # debug
    # arrow_user_zaxis = vedo.Arrow(vertices.mean(0), user_zaxis*5+vertices.mean(0), c="blue")
    # arrow_zaxis = vedo.Arrow(vertices.mean(0), zaxis*5+vertices.mean(0), c="red")
    # arrow_xaxis = vedo.Arrow(vertices.mean(0), user_xaxis*5+vertices.mean(0), c="green")
    # vedo.show([arrow_user_zaxis,arrow_zaxis,arrow_xaxis,jaw_mesh.split()[0], vedo.Point(jaw_near_point,r=12,c="black"),vedo.Point(vertices.mean(0),r=20,c="red5"),vedo.Point(m_list[0].center_of_mass(),r=24,c="green"),vedo.Point(m_list[1].center_of_mass(),r=24,c="green"),near_mesh], axes=3)
    # print(np.dot(user_zaxis, zaxis))

    if np.dot(user_zaxis, zaxis) < 0:
        # 如果z轴方向上有mesh，则保持原样，否则将z轴取反向
        zaxis = -zaxis
    yaxis = np.cross(user_xaxis, zaxis)
    components = np.stack([user_xaxis, yaxis, zaxis], axis=0)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = components
    transform[:3, 3] = - components @ vertices.mean(0)

    # 渲染
    new_m = vedo.Mesh([apply_transform(near_mesh.points(), transform), near_mesh.faces()])
    return new_m


def cut_mesh_point_loop_crow(mesh, pts, error_show=True, invert=True):
    """

    实现的基于线的牙齿冠分割;

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points/Line): 切割线
        error_show(bool, optional): 裁剪失败是否进行渲染. Defaults to True.
        invert(bool): 是否取反；

    Returns:
        _type_: 切割后的网格
    """
    if len(pts.vertices) > 30:
        s = int(len(pts.vertices) / 30)
        pts = vedo.Points(pts.vertices[::s])

    # 计算各区域到曲线的最近距离,去除不相关的联通体
    def batch_closest_dist(vertices, curve_pts):
        curve_matrix = np.array(curve_pts)
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.sum(dist_matrix, axis=1)

    regions = mesh.split()
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh = regions[np.argmin(min_dists)]

    c1 = cut_mesh_point_loop(mesh, pts, invert=not invert)
    c2 = cut_mesh_point_loop(mesh, pts, invert=invert)

    # 可能存在网格错误造成的洞,默认执行补洞
    c1_num = len(c1.fill_holes().split()[0].boundaries().split())
    c2_num = len(c2.fill_holes().split()[0].boundaries().split())

    # 牙冠只能有一个开口
    if c1_num == 1:
        cut_mesh = c1
    elif c2_num == 1:
        cut_mesh = c2
    else:
        print("裁剪失败,请检查分割线,尝试pts[::3]进行采样输入")
        if error_show:
            print(f"边界1:{c1_num},边界2：{c2_num}")
            vedo.show([(c1), (c2)], N=2).close()
        return None

    return cut_mesh


def subdivide_with_pts(v, f, line_pts, r=0.15, iterations=3, method="mid"):
    """
    对给定的网格和线点集进行局部细分。

    Args:
        v (array-like): 输入网格的顶点数组。
        f (array-like): 输入网格的面数组。
        line_pts (array-like): 线的点集数组。
        r (float, optional): 查找线点附近顶点的半径，默认为 0.15.
        method (str, optional): 细分方法，可选值为 "mid"（中点细分）或其他值（对应 ls3_loop 细分），默认为 "mid"。

    Returns:
        - new_vertices (np.ndarray): 细分后的顶点数组;

        - new_face (np.ndarray): 细分后的面数组;

    Notes:
        ```python
        # 闭合线可能在曲面上，曲面内，曲面外
        line = Line(pts)
        mesh = isotropic_remeshing_by_acvd(mesh)
        v, f = np.array(mesh.vertices), np.array(mesh.cells)
        new_vertices, new_face = subdivide_with_pts(v, f, pts)

        show([(Mesh([new_vertices, new_face]).c("green"), Line(pts, lw = 2, c = "red")),
             (Mesh([v, f]).c("pink"), Line(pts, lw = 2, c = "red"))], N = 2).close()
        ```

    """

    from scipy.spatial import cKDTree
    import pymeshlab
    v, f, pts = np.array(v), np.array(f), np.array(line_pts)
    # 获取每个点的最近表面点及对应面
    face_indices = set()
    kdtree = cKDTree(v)
    vertex_indices = kdtree.query_ball_point(pts, r=r)

    #合并所有邻近顶点并去重
    all_vertex_indices = np.unique(np.hstack(vertex_indices)).astype(np.int32)
    face_mask = np.any(np.isin(f, all_vertex_indices), axis=1)

    # 局部细分
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f, f_scalar_array=face_mask))
    ms.compute_selection_by_condition_per_face(condselect="fq == 1")
    if method == "mid":
        ms.meshing_surface_subdivision_midpoint(
            iterations=iterations,
            threshold=pymeshlab.PercentageValue(1e-4),
            selected=True
        )
    else:
        ms.meshing_surface_subdivision_ls3_loop(
            iterations=iterations,
            threshold=pymeshlab.PercentageValue(1e-4),
            selected=True
        )
    current_mesh = ms.current_mesh()
    new_vertices = current_mesh.vertex_matrix()
    new_faces = current_mesh.face_matrix()
    return new_vertices, new_faces
