# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> algorithm
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/27 15:50
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference:
@History:
- 2025/3/27 15:50:
==================================================
"""
__author__ = 'zxx'

import json
import vedo
import numpy as np
from typing import *
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import vtk
import os


def labels2colors(labels: np.array):
    """
    将labels转换成颜色标签
    Args:
        labels: numpy类型,形状(N)对应顶点的标签；

    Returns:
        RGBA颜色标签;
    """
    labels = labels.reshape(-1)
    from colorsys import hsv_to_rgb
    unique_labels = np.unique(labels)
    num_unique = len(unique_labels)

    if num_unique == 0:
        return np.zeros((len(labels), 4), dtype=np.uint8)

    # 生成均匀分布的色相（0-360度），饱和度和亮度固定为较高值
    hues = np.linspace(0, 360, num_unique, endpoint=False)
    s = 0.8  # 饱和度
    v = 0.9  # 亮度

    colors = []
    for h in hues:
        # 转换HSV到RGB
        r, g, b = hsv_to_rgb(h / 360.0, s, v)
        # 转换为0-255的整数并添加Alpha通道
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])

    colors = np.array(colors, dtype=np.uint8)

    # 创建颜色映射字典
    color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}

    # 生成结果数组
    color_labels = np.zeros((len(labels), 4), dtype=np.uint8)
    for label in unique_labels:
        mask = (labels == label)
        color_labels[mask] = color_dict[label]

    return color_labels


def vertex_labels_to_face_labels(faces: Union[np.array, list], vertex_labels: Union[np.array, list]) -> np.array:
    """
        将三角网格的顶点标签转换成面片标签，存在一个面片，多个属性，则获取出现最多的属性。

    Args:
        faces: 三角网格面片索引
        vertex_labels: 顶点标签

    Returns:
        面片属性

    """

    # 获取三角网格的面片标签
    face_labels = np.zeros(len(faces))
    for i in range(len(face_labels)):
        face_label = []
        for face_id in faces[i]:
            face_label.append(vertex_labels[face_id])

        # 存在一个面片，多个属性，则获取出现最多的属性
        maxlabel = max(face_label, key=face_label.count)
        face_labels[i] = maxlabel

    return face_labels.astype(np.int32)


def face_labels_to_vertex_labels(vertices: Union[np.array, list], faces: Union[np.array, list],
                                 face_labels: np.array) -> np.array:
    """
        将三角网格的面片标签转换成顶点标签

    Args:
        vertices: 牙颌三角网格
        faces: 面片标签
        face_labels: 顶点标签

    Returns:
        顶点属性

    """

    # 获取三角网格的顶点标签
    vertex_labels = np.zeros(len(vertices))
    for i in range(len(faces)):
        for vertex_id in faces[i]:
            vertex_labels[vertex_id] = face_labels[i]

    return vertex_labels.astype(np.int32)


def get_axis_rotation(axis: list, angle: float) -> np.array:
    """
        绕着指定轴获取3*3旋转矩阵

    Args:
        axis: 轴向,[0,0,1]
        angle: 旋转角度,90.0

    Returns:
        3*3旋转矩阵

    """

    ang = np.radians(angle)
    R = np.zeros((3, 3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang) + ux * ux * (1 - cos(ang))
    R[0][1] = ux * uy * (1 - cos(ang)) - uz * sin(ang)
    R[0][2] = ux * uz * (1 - cos(ang)) + uy * sin(ang)
    R[1][0] = uy * ux * (1 - cos(ang)) + uz * sin(ang)
    R[1][1] = cos(ang) + uy * uy * (1 - cos(ang))
    R[1][2] = uy * uz * (1 - cos(ang)) - ux * sin(ang)
    R[2][0] = uz * ux * (1 - cos(ang)) - uy * sin(ang)
    R[2][1] = uz * uy * (1 - cos(ang)) + ux * sin(ang)
    R[2][2] = cos(ang) + uz * uz * (1 - cos(ang))
    return R


def get_pca_rotation(vertices: np.array) -> np.array:
    """
        通过pca分析顶点，获取3*3旋转矩阵，并应用到顶点；

    Args:
        vertices: 三维顶点

    Returns:
        应用旋转矩阵后的顶点
    """

    pca_axis = PCA(n_components=3).fit(vertices).components_
    rotation_mat = pca_axis
    vertices = (rotation_mat @ vertices[:, :3].T).T
    return vertices


def get_pca_transform(mesh: vedo.Mesh) -> np.array:
    """
        将输入的顶点数据根据曲率及PCA分析得到的主成分向量，
        并转换成4*4变换矩阵。

    Notes:
        必须为底部非封闭的网格

    Args:
        mesh: vedo网格对象

    Returns:
        4*4 变换矩阵


    """
    """

    :param mesh: 
    :return: 
    """
    vedo_mesh = mesh.clone().decimate(n=5000).clean()
    vertices = vedo_mesh.points()

    vedo_mesh.compute_curvature(method=1)
    data = vedo_mesh.pointdata['Mean_Curvature']
    verticesn_curvature = vertices[data < 0]

    xaxis, yaxis, zaxis = PCA(n_components=3).fit(verticesn_curvature).components_

    # 通过找边缘最近的点确定z轴方向
    near_point = vedo_mesh.boundaries().center_of_mass()
    vec = near_point - vertices.mean(0)
    user_zaxis = vec / np.linalg.norm(vec)
    if np.dot(user_zaxis, zaxis) > 0:
        # 如果z轴方向与朝向边缘方向相似，那么取反
        zaxis = -zaxis

    """
    plane = vedo.fit_plane(verticesn_curvature)
    m=vedo_mesh.cut_with_plane(plane.center,zaxis).split()[0]
    #m.show()
    vertices = m.points()


    # 将点投影到z轴，重新计算x,y轴
    projected_vertices_xy = vertices - np.dot(vertices, zaxis)[:, None] * zaxis

    # 使用PCA分析投影后的顶点数据
    #xaxis, yaxis = PCA(n_components=2).fit(projected_vertices_xy).components_

    # y = vedo.Arrow(vertices.mean(0), yaxis*5+vertices.mean(0), c="green")
    # x = vedo.Arrow(vertices.mean(0), xaxis*5+vertices.mean(0), c="red")
    # p = vedo.Points(projected_vertices_xy)
    # vedo.show([y,x,p])
    """

    components = np.stack([xaxis, yaxis, zaxis], axis=0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = components
    transform[:3, 3] = - components @ vertices.mean(0)

    return transform


def apply_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        对4*4矩阵进行应用

    Args:
        vertices: 顶点
        transform: 4*4 矩阵

    Returns:
        变换后的顶点

    """

    # 在每个顶点的末尾添加一个维度为1的数组，以便进行齐次坐标转换
    vertices = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1)
    vertices = vertices @ transform.T
    # 移除结果中多余的维度，只保留前3列，即三维坐标
    vertices = vertices[..., :3]

    return vertices


def restore_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        根据提供的顶点及矩阵，进行逆变换(还原应用矩阵之前的状态）

    Args:
        vertices: 顶点
        transform: 4*4变换矩阵

    Returns:
        还原后的顶点坐标

    """
    # 得到转换矩阵的逆矩阵
    inv_transform = np.linalg.inv(transform.T)

    # 将经过转换后的顶点坐标乘以逆矩阵
    vertices_restored = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1) @ inv_transform

    # 去除齐次坐标
    vertices_restored = vertices_restored[:, :3]

    # 最终得到还原后的顶点坐标 vertices_restored
    return vertices_restored


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


def save_np_json(output_path: str, obj) -> None:
    """
    保存np形式的json

    Args:
        output_path: 保存路径
        obj: 保存对象


    """

    with open(output_path, 'w') as fp:
        json.dump(obj, fp, cls=NpEncoder)


def get_obb_box(x_pts: np.array, z_pts: np.array, vertices: np.array) -> Tuple[list, list, np.array]:
    """
    给定任意2个轴向交点及顶点，返回定向包围框mesh
    Args:
        x_pts: x轴交点
        z_pts: z轴交点
        vertices: 所有顶点

    Returns:
        包围框的顶点， 面片索引，3*3旋转矩阵

    """

    # 计算中心
    center = np.mean(vertices, axis=0)
    print(center)

    # 定义三个射线
    x_axis = np.array(x_pts - center).reshape(3)
    z_axis = np.array(z_pts - center).reshape(3)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis).reshape(3)

    # 计算AABB
    x_project = np.dot(vertices, x_axis)
    y_project = np.dot(vertices, y_axis)
    z_project = np.dot(vertices, z_axis)
    z_max_pts = vertices[np.argmax(z_project)]
    z_min_pts = vertices[np.argmin(z_project)]
    x_max_pts = vertices[np.argmax(x_project)]
    x_min_pts = vertices[np.argmin(x_project)]
    y_max_pts = vertices[np.argmax(y_project)]
    y_min_pts = vertices[np.argmin(y_project)]

    # 计算最大边界
    z_max = np.dot(z_max_pts - center, z_axis)
    z_min = np.dot(z_min_pts - center, z_axis)
    x_max = np.dot(x_max_pts - center, x_axis)
    x_min = np.dot(x_min_pts - center, x_axis)
    y_max = np.dot(y_max_pts - center, y_axis)
    y_min = np.dot(y_min_pts - center, y_axis)

    # 计算最大边界位移
    inv_x = x_min * x_axis
    inv_y = y_min * y_axis
    inv_z = z_min * z_axis
    x = x_max * x_axis
    y = y_max * y_axis
    z = z_max * z_axis

    # 绘制OBB
    verts = [
        center + x + y + z,
        center + inv_x + inv_y + inv_z,

        center + inv_x + inv_y + z,
        center + x + inv_y + inv_z,
        center + inv_x + y + inv_z,

        center + x + y + inv_z,
        center + x + inv_y + z,
        center + inv_x + y + z,

    ]

    faces = [
        [0, 6, 7],
        [6, 7, 2],
        [0, 6, 3],
        [0, 5, 3],
        [0, 7, 5],
        [4, 7, 5],
        [4, 7, 2],
        [1, 2, 4],
        [1, 2, 3],
        [2, 3, 6],
        [3, 5, 4],
        [1, 3, 4]

    ]
    R = np.vstack([x_axis, y_axis, z_axis]).T
    return verts, faces, R


def get_obb_box_max_min(x_pts: np.array,
                        z_pts: np.array,
                        z_max_pts: np.array,
                        z_min_pts: np.array,
                        x_max_pts: np.array,
                        x_min_pts: np.array,
                        y_max_pts: np.array,
                        y_min_pts: np.array,
                        center: np.array) -> Tuple[list, list, np.array]:
    """
     给定任意2个轴向交点及最大/最小点，返回定向包围框mesh

    Args:
        x_pts: x轴交点
        z_pts: z轴交点
        z_max_pts: 最大z顶点
        z_min_pts:最小z顶点
        x_max_pts:最大x顶点
        x_min_pts:最小x顶点
        y_max_pts:最大y顶点
        y_min_pts:最小y顶点
        center: 中心点

    Returns:
        包围框的顶点， 面片索引，3*3旋转矩阵

    """

    # 定义三个射线
    x_axis = np.array(x_pts - center).reshape(3)
    z_axis = np.array(z_pts - center).reshape(3)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis).reshape(3)

    # 计算最大边界
    z_max = np.dot(z_max_pts - center, z_axis)
    z_min = np.dot(z_min_pts - center, z_axis)
    x_max = np.dot(x_max_pts - center, x_axis)
    x_min = np.dot(x_min_pts - center, x_axis)
    y_max = np.dot(y_max_pts - center, y_axis)
    y_min = np.dot(y_min_pts - center, y_axis)

    # 计算最大边界位移
    inv_x = x_min * x_axis
    inv_y = y_min * y_axis
    inv_z = z_min * z_axis
    x = x_max * x_axis
    y = y_max * y_axis
    z = z_max * z_axis

    # 绘制OBB
    verts = [
        center + x + y + z,
        center + inv_x + inv_y + inv_z,

        center + inv_x + inv_y + z,
        center + x + inv_y + inv_z,
        center + inv_x + y + inv_z,

        center + x + y + inv_z,
        center + x + inv_y + z,
        center + inv_x + y + z,

    ]

    faces = [
        [0, 6, 7],
        [6, 7, 2],
        [0, 6, 3],
        [0, 5, 3],
        [0, 7, 5],
        [4, 7, 5],
        [4, 7, 2],
        [1, 2, 4],
        [1, 2, 3],
        [2, 3, 6],
        [3, 5, 4],
        [1, 3, 4]

    ]
    R = np.vstack([x_axis, y_axis, z_axis]).T
    return verts, faces, R


def create_voxels(vertices, resolution: int = 256):
    """
        通过顶点创建阵列方格体素
    Args:
        vertices: 顶点
        resolution:  分辨率

    Returns:
        返回 res**3 的顶点 , mc重建需要的缩放及位移

    Notes:
        v, f = mcubes.marching_cubes(data.reshape(256, 256, 256), 0)

        m=vedo.Mesh([v*scale+translation, f])


    """
    vertices = np.array(vertices)
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    # 使用np.mgrid生成网格索引
    i, j, k = np.mgrid[0:resolution, 0:resolution, 0:resolution]

    # 计算步长（即网格单元的大小）
    dx = (x_max - x_min) / resolution
    dy = (y_max - y_min) / resolution
    dz = (z_max - z_min) / resolution
    scale = np.array([dx, dy, dz])

    # 将索引转换为坐标值
    x = x_min + i * dx
    y = y_min + j * dy
    z = z_min + k * dz
    translation = np.array([x_min, y_min, z_min])

    verts = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)
    # print(verts.shape)
    # vedo.show(vedo.Points(verts[::30]),self.crown).close()
    return verts, scale, translation


def compute_face_normals(vertices, faces):
    """
    计算三角形网格中每个面的法线
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3)，每个面由三个顶点索引组成
    Returns:
        面法线数组，形状为 (M, 3)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # 处理退化面（法线长度为0的情况）
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    eps = 1e-8
    norms = np.where(norms < eps, 1.0, norms)  # 避免除以零
    face_normals = face_normals / norms

    return face_normals


def compute_vertex_normals(vertices, faces):
    """
    计算三角形网格中每个顶点的法线
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3)，每个面由三个顶点索引组成
    Returns:
        顶点法线数组，形状为 (N, 3)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0

    # 计算未归一化的面法线（叉积的模长为两倍三角形面积）
    face_normals = np.cross(edge1, edge2)

    vertex_normals = np.zeros(vertices.shape)
    # 累加面法线到对应的顶点
    np.add.at(vertex_normals, faces.flatten(), np.repeat(face_normals, 3, axis=0))

    # 归一化顶点法线并处理零向量
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    eps = 1e-8
    lengths = np.where(lengths < eps, 1.0, lengths)  # 避免除以零
    vertex_normals = vertex_normals / lengths

    return vertex_normals


def cut_mesh_point_loop(mesh, pts: vedo.Points, invert=False):
    """

    基于vtk+dijkstra实现的基于线的分割;

    线支持在网格上或者网格外；

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points): 切割线
        invert (bool, optional): 选择保留外部. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """

    # 强制关闭Can't follow edge错误弹窗
    vtk.vtkObject.GlobalWarningDisplayOff()
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOn()
    selector.Update()
    if selector.GetOutput().GetNumberOfPoints() == 0:
        # Can't follow edge
        selector.SetEdgeSearchModeToDijkstra()
        selector.Update()

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(selector.GetOutput())
    clipper.SetInsideOut(not invert)
    clipper.SetValue(0.0)
    clipper.Update()

    cut_mesh = vedo.Mesh(clipper.GetOutput())
    vtk.vtkObject.GlobalWarningDisplayOn()
    return cut_mesh


def reduce_face_by_meshlab(vertices, faces, max_facenum: int = 30000) -> vedo.Mesh:
    """通过二次边折叠算法减少网格中的面数，简化模型。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。
        max_facenum (int, optional): 简化后的目标最大面数，默认为 200000。

    Returns:
        pymeshlab.MeshSet: 简化后的网格模型。
    """
    import pymeshlab

    mesh = pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return vedo.Mesh(mesh.current_mesh())


def remove_floater_by_meshlab(vertices, faces, nbfaceratio=0.1, nonclosedonly=False) -> vedo.Mesh:
    """移除网格中的浮动小组件（小面积不连通部分）。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。
        nbfaceratio (float): 面积比率阈值，小于该比率的部分将被移除。
        nonclosedonly (bool): 是否仅移除非封闭部分。

    Returns:
        pymeshlab.MeshSet: 移除浮动小组件后的网格模型。
    """
    import pymeshlab

    mesh = pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=nbfaceratio, nonclosedonly=nonclosedonly)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return vedo.Mesh(mesh.current_mesh())


def isotropic_remeshing_pymeshlab(vertices, faces, target_edge_length=0.5, iterations=1) -> vedo.Mesh:
    """
    使用 PyMeshLab 实现网格均匀化。

    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。
        target_edge_length: 目标边长比例 %。
        iterations: 迭代次数，默认为 1。

    Returns:
        均匀化后的网格对象。
    """

    import pymeshlab
    mesh = pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    # 应用 Isotropic Remeshing 过滤器
    mesh.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        targetlen=pymeshlab.PercentageValue(target_edge_length),
        iterations=iterations,
    )

    # 返回处理后的网格
    return vedo.Mesh(mesh.current_mesh())


def clean_redundant(ms):
    """
    处理冗余元素，如合并临近顶点、移除重复面和顶点等。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_merge_close_vertices")
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    return ms


def clean_invalid(ms):
    """
    清理无效的几何结构，如折叠面、零面积面和未引用的顶点。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_folded_faces")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    return ms


def clean_low_qualitys(ms):
    """
    移除低质量的组件，如小的连通分量。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_connected_component_by_diameter")
    ms.apply_filter("meshing_remove_connected_component_by_face_number", mincomponentsize=10)
    return ms


def repair_topology(ms):
    """
    修复拓扑问题，如 T 型顶点、非流形边和非流形顶点，并对齐不匹配的边界。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_t_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_snap_mismatched_borders")
    return ms


def labels_mapping(old_vertices, old_faces, new_vertices, old_labels, fast=True):
    """
    将原始网格的标签属性精确映射到新网格

    参数:
        old_mesh(vedo) : 原始网格对象
        new_mesh(vedo): 重网格化后的新网格对象
        old_labels (np.ndarray): 原始顶点标签数组，形状为 (N,)

    返回:
        new_labels (np.ndarray): 映射后的新顶点标签数组，形状为 (M,)
    """
    if len(old_labels) != len(old_vertices):
        raise ValueError(f"标签数量 ({len(old_labels)}) 必须与原始顶点数 ({len(old_vertices)}) 一致")

    if fast:
        tree = KDTree(old_vertices)
        _, idx = tree.query(new_vertices, workers=-1)
        return old_labels[idx]

    else:
        import trimesh
        old_mesh = trimesh.Trimesh(old_vertices, old_faces)
        # 步骤1: 查询每个新顶点在原始网格上的最近面片信息
        closest_points, distances, tri_ids = trimesh.proximity.closest_point(old_mesh, new_vertices)
        # 步骤2: 计算每个投影点的重心坐标
        tri_vertices = old_mesh.faces[tri_ids]
        tri_points = old_mesh.vertices[tri_vertices]
        # 计算重心坐标 (M,3)
        bary_coords = trimesh.triangles.points_to_barycentric(
            triangles=tri_points,
            points=closest_points
        )
        # 步骤3: 确定最大重心坐标对应的顶点
        max_indices = np.argmax(bary_coords, axis=1)
        # 根据最大分量索引选择顶点编号
        nearest_vertex_indices = tri_vertices[np.arange(len(max_indices)), max_indices]
        # 步骤4: 映射标签
        new_labels = np.array(old_labels)[nearest_vertex_indices]
        return new_labels


class BestKFinder:
    def __init__(self, points, labels):
        """
        初始化类，接收点云网格数据和对应的标签

        Args:
            points (np.ndarray): 点云数据，形状为 (N, 3)
            labels (np.ndarray): 点云标签，形状为 (N,)
        """
        self.points = np.array(points)
        self.labels = np.array(labels).reshape(-1)

    def calculate_boundary_points(self, k):
        """
        计算边界点
        :param k: 最近邻点的数量
        :return: 边界点的标签数组
        """
        points = self.points
        tree = KDTree(points)
        _, near_points = tree.query(points, k=k, workers=-1)
        # 确保 near_points 是整数类型
        near_points = near_points.astype(int)
        labels_arr = self.labels[near_points]
        # 将 labels_arr 转换为整数类型
        labels_arr = labels_arr.astype(int)
        label_counts = np.apply_along_axis(lambda x: np.bincount(x).max(), 1, labels_arr)
        label_ratio = label_counts / k
        bdl_ratio = 0.8  # 假设的边界点比例阈值
        bd_labels = np.zeros(len(points))
        bd_labels[label_ratio < bdl_ratio] = 1
        return bd_labels

    def evaluate_boundary_points(self, bd_labels):
        """
        评估边界点的分布合理性
        这里简单使用边界点的数量占比作为评估指标
        :param bd_labels: 边界点的标签数组
        :return: 评估得分
        """
        boundary_ratio = np.sum(bd_labels) / len(bd_labels)
        # 假设理想的边界点比例在 0.1 - 0.2 之间
        ideal_ratio = 0.15
        score = 1 - np.abs(boundary_ratio - ideal_ratio)
        return score

    def find_best_k(self, k_values):
        """
        找出最佳的最近邻点大小

        :param k_values: 待测试的最近邻点大小列表
        :return: 最佳的最近邻点大小
        """
        best_score = -1
        best_k = None
        for k in k_values:
            bd_labels = self.calculate_boundary_points(k)
            score = self.evaluate_boundary_points(bd_labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k


class GraphCutRefiner:
    def __init__(self, vertices, faces, vertex_labels, smooth_factor=None, temperature=None, keep_label=True):
        """
        基于顶点的图切优化器

        Args:
            vertices (array-like): 顶点坐标数组，形状为 (n_vertices, 3)。
            faces (array-like): 面片索引数组，形状为 (n_faces, 3)。
            vertex_labels (array-like): 顶点初始标签数组，形状为 (n_vertices,)。
            smooth_factor (float, optional): 平滑强度系数，越大边界越平滑。默认值为 None，此时会自动计算。范围通常在 0.1 到 0.6 之间。
            temperature (float, optional): 温度参数，越大标签越平滑，处理速度越快。默认值为 None，此时会自动计算。典型值范围在 50 到 500 之间，会随网格复杂度自动调整。
            keep_label (bool, optional): 是否保持优化前后标签类别一致性，默认值为 True。
        """
        import trimesh
        self.mesh = trimesh.Trimesh(vertices, faces)
        self._precompute_geometry()
        self.smooth_factor = smooth_factor
        self.keep_label = keep_label
        vertex_labels = vertex_labels.reshape(-1)

        # 处理标签映射
        self.unique_labels, mapped_labels = np.unique(vertex_labels, return_inverse=True)
        if temperature is None:
            self.temperature = self._compute_temperature(mapped_labels)
        else:
            self.temperature = temperature
        self.prob_matrix = self._labels_to_prob(mapped_labels, self.unique_labels.size)
        print(self.prob_matrix.shape)

    def _precompute_geometry(self):
        """预计算顶点几何特征"""
        self.mesh.fix_normals()
        self.vertex_normals = self.mesh.vertex_normals.copy()  # 顶点法线
        self.vertex_positions = self.mesh.vertices.copy()  # 顶点坐标
        self.adjacency = self._compute_adjacency()  # 顶点邻接关系

    def _compute_temperature(self, labels):
        """根据邻域标签一致性计算温度参数"""
        n = len(labels)
        total_inconsistency = 0.0

        for i in range(n):
            neighbors = self.adjacency[i]
            if not neighbors.size:
                continue
            # 计算邻域标签不一致性
            same_count = np.sum(labels[neighbors] == labels[i])
            inconsistency = 1.0 - same_count / len(neighbors)
            total_inconsistency += inconsistency

        avg_inconsistency = total_inconsistency / n
        # 温度公式: 基础0.1 + 平均不一致性系数
        return 0.1 + avg_inconsistency * 0.5

    def _compute_adjacency(self):
        """计算顶点邻接关系"""
        # 使用trimesh内置的顶点邻接查询
        return [np.array(list(adj)) for adj in self.mesh.vertex_neighbors]

    def refine_labels(self):
        """
        执行标签优化
        :return: 优化后的顶点标签数组 (n_vertices,)
        """
        from pygco import cut_from_graph
        # 数值稳定性处理
        prob = np.clip(self.prob_matrix, 1e-6, 1.0)
        prob /= prob.sum(axis=1, keepdims=True)

        # 计算unary potential
        unaries = (-100 * np.log10(prob)).astype(np.int32)

        # 自适应计算smooth_factor
        if self.smooth_factor is None:
            edges_raw = self._compute_edge_weights(scale=1.0)
            weights_raw = edges_raw[:, 2]
            unary_median = np.median(np.abs(unaries))
            weight_median = np.median(weights_raw) if weights_raw.size else 1.0
            self.smooth_factor = unary_median / max(weight_median, 1e-6) * 4  # *0.8* 4 #经验值

        # print(self.smooth_factor)
        # 构造pairwise potential
        n_classes = self.prob_matrix.shape[-1]
        pairwise = (1 - np.eye(n_classes, dtype=np.int32))

        # 执行图切优化
        try:
            """
            edges1 = np.array([[0,1,100], [1,2,100], [2,3,100], 
                 [0,2,200], [1,3,200]], dtype=np.int32)
            unaries1 = np.array([[5, 0], [0, 5], [5, 0], [5, 0]], dtype=np.int32)
            pairwise1 = np.array([[0,1],[1,0]], dtype=np.int32)

            optimized = cut_from_graph(edges1, unaries1, pairwise1)
            print("应输出[0,1,0,0]，实际输出:", optimized)
            print("调用图切函数前参数检查:")
            print("edges shape:", edges.shape, "dtype:", edges.dtype,edges,len(self.vertex_positions))
            print("unaries shape:", unaries.shape, "dtype:", unaries.dtype,unaries)
            print("pairwise shape:", pairwise.shape, "dtype:", pairwise.dtype,pairwise)
            assert edges[:, :2].max() < len(self.vertex_positions), "边包含非法顶点索引"
            assert not np.isinf(edges[:,2]).any(), "边权重包含无穷值"
            assert (np.abs(edges[:,2]) < 2**30).all(), "边权重超过int32范围"

            """

            if self.keep_label:
                optimized_labels = None
                for i in range(10):
                    # 计算边权重
                    edges = self._compute_edge_weights(self.smooth_factor)
                    optimized_labels_it = cut_from_graph(edges, unaries, pairwise)
                    if len(np.unique(optimized_labels_it)) == n_classes:
                        optimized_labels = optimized_labels_it
                        self.smooth_factor *= 1.5
                        print(f"当前smooth_factor={self.smooth_factor},优化中({i + 1}/10)....")
                    else:
                        break

            else:
                edges = self._compute_edge_weights(self.smooth_factor)
                optimized_labels = cut_from_graph(edges, unaries, pairwise)


        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"图切优化失败: {str(e)}") from e

        return self.unique_labels[optimized_labels]

    def _compute_edge_weights(self, scale):
        """计算边权重（基于顶点几何特征）"""
        edges = []

        for i in range(len(self.adjacency)):
            for j in self.adjacency[i]:
                if j <= i:  # 避免重复计算边
                    continue

                # 计算法线夹角
                ni, nj = self.vertex_normals[i], self.vertex_normals[j]
                cos_theta = np.dot(ni, nj)
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                theta = np.maximum(theta, 1e-8)  # 防止θ为0导致对数溢出

                # 计算空间距离
                pi, pj = self.vertex_positions[i], self.vertex_positions[j]
                distance = np.linalg.norm(pi - pj)

                # 计算自适应权重
                if theta > np.pi / 2:
                    weight = -np.log(theta / np.pi) * distance
                else:
                    weight = -10 * np.log(theta / np.pi) * distance  # 加强平滑区域约束

                edges.append([i, j, int(weight * scale)])

        return np.array(edges, dtype=np.int32)

    def _labels_to_prob(self, labels, n_classes):
        """将标签转换为概率矩阵"""
        one_hot = np.eye(n_classes)[labels]
        prob = np.exp(one_hot / self.temperature)
        return prob / prob.sum(axis=1, keepdims=True)


def load_all(path):
    """
    读取各种格式的文件

    Returns:
        data: 读取的数据，失败返回None；
    """
    try:
        if path.endswith(".json"):
            with open(path, 'r', encoding="utf-8") as f:
                data = json.load(f)


        elif path.endswith((".npy", "npz")):
            data = np.load(path, allow_pickle=True)


        elif path.endswith((".pkl", ".pickle")):
            import pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)

        elif path.endswith(".txt"):
            with open(path, 'r') as f:
                data = f.read()

        elif path.endswith((".db", ".lmdb", "mdb", ".yx")) or os.path.isdir(path):
            import sindre
            data = sindre.lmdb.Reader(path, True)
            print("使用完成请关闭 data.close()")

        elif path.endswith((".pt", ".pth")):
            import torch
            # 使用 map_location='cpu' 避免CUDA设备不可用时的错误
            data = torch.load(path, map_location='cpu')

        else:
            data = vedo.load(path)
        return data
    except Exception as e:
        print("读取失败", e)
        return None


def farthest_point_sampling(arr, n_sample, start_idx=None):
    """
    无需计算所有点对之间的距离，进行最远点采样。

    Args:

        arr : numpy array
            形状为 (n_points, n_dim) 的位置数组，其中 n_points 是点的数量，n_dim 是每个点的维度。
        n_sample : int
            需要采样的点的数量。
        start_idx : int, 可选
            如果给定，指定起始点的索引；否则，随机选择一个点作为起始点。（默认值: None）

    Return:

        numpy array of shape (n_sample,)
            采样得到的点的索引数组。

    Example:

        >>> import numpy as np
        >>> data = np.random.rand(100, 1024)
        >>> point_idx = farthest_point_sampling(data, 3)
        >>> print(point_idx)
            array([80, 79, 27])

        >>> point_idx = farthest_point_sampling(data, 5, 60)
        >>> print(point_idx)
            array([60, 39, 59, 21, 73])
    """
    n_points, n_dim = arr.shape

    if (start_idx is None) or (start_idx < 0):
        start_idx = np.random.randint(0, n_points)

    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)

    for _ in range(n_sample - 1):
        current_point = arr[sampled_indices[-1]]
        dist_to_current_point = np.linalg.norm(arr - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return np.array(sampled_indices)


def add_base(vd_mesh, value_z=-20, close_base=True, return_strips=False):
    """给网格边界z方向添加底座

    Args:
        vd_mesh (_type_):vedo.mesh
        value_z (int, optional): 底座长度. Defaults to -20.
        close_base (bool, optional): 底座是否闭合. Defaults to True.
        return_strips (bool, optional): 是否返回添加的网格. Defaults to False.

    Returns:
        _type_: 添加底座的网格
    """

    # 开始边界
    boundarie_start = vd_mesh.clone().boundaries()
    boundarie_start = boundarie_start.generate_delaunay2d(mode="fit").boundaries()
    # TODO:补充边界损失
    # 底座边界
    boundarie_end = boundarie_start.copy()
    boundarie_end.vertices[..., 2:] = value_z
    strips = boundarie_start.join_with_strips(boundarie_end)
    merge_list = [vd_mesh, strips]
    if return_strips:
        return strips
    if close_base:
        merge_list.append(boundarie_end.generate_delaunay2d(mode="fit"))
    out_mesh = vedo.merge(merge_list).clean()
    return out_mesh


def equidistant_mesh(mesh, d=-0.01, merge=True):
    """

    此函数用于创建一个与输入网格等距的新网格，可选择将新网格与原网格合并。


    Args:
        mesh (vedo.Mesh): 输入的三维网格对象。
        d (float, 可选): 顶点偏移的距离，默认为 -0.01。负值表示向内偏移，正值表示向外偏移。
        merge (bool, 可选): 是否将原网格和偏移后的网格合并，默认为 True。

    Returns:
        vedo.Mesh 或 vedo.Assembly: 如果 merge 为 True，则返回合并后的网格；否则返回偏移后的网格。
    """
    mesh.compute_normals().clean()
    cells = np.asarray(mesh.cells)
    original_vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    pts_id = mesh.boundaries(return_point_ids=True)

    # 创建边界掩码
    boundary_mask = np.zeros(len(original_vertices), dtype=bool)
    boundary_mask[pts_id] = True

    # 仅对非边界顶点应用偏移
    pts = original_vertices.copy()
    pts[~boundary_mask] += vertex_normals[~boundary_mask] * d

    # 构建新网格
    offset_mesh = vedo.Mesh([pts, cells]).clean()
    if merge:
        return vedo.merge([mesh, offset_mesh])
    else:
        return offset_mesh


def voxel2array(grid_index_array, voxel_size=32):
    """
    将 voxel_grid_index 数组转换为固定大小的三维数组。

    该函数接收一个形状为 (N, 3) 的 voxel_grid_index 数组，
    并将其转换为形状为 (voxel_size, voxel_size, voxel_size) 的三维数组。
    其中，原 voxel_grid_index 数组中每个元素代表三维空间中的一个网格索引，
    在转换后的三维数组中对应位置的值会被设为 1，其余位置为 0。

    Args:
        grid_index_array (numpy.ndarray): 形状为 (N, 3) 的数组，
            通常从 open3d 的 o3d.voxel_grid.get_voxels() 方法获取，
            表示三维空间中每个体素的网格索引。
        voxel_size (int, optional): 转换后三维数组的边长，默认为 32。

    Returns:
        numpy.ndarray: 形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，
            其中原 voxel_grid_index 数组对应的网格索引位置值为 1，其余为 0。

    Example:
        ```python
        # 获取 grid_index_array
        voxel_list = voxel_grid.get_voxels()
        grid_index_array = list(map(lambda x: x.grid_index, voxel_list))
        grid_index_array = np.array(grid_index_array)
        voxel_grid_array = voxel2array(grid_index_array, voxel_size=32)
        grid_index_array = array2voxel(voxel_grid_array)
        pointcloud_array = grid_index_array  # 0.03125 是体素大小
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud_array)
        o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
        o3d.visualization.draw_geometries([pcd, cc, o3d_voxel])
        ```
    """
    array_voxel = np.zeros((voxel_size, voxel_size, voxel_size))
    array_voxel[grid_index_array[:, 0], grid_index_array[:, 1], grid_index_array[:, 2]] = 1
    return array_voxel


def array2voxel(voxel_array):
    """
        将固定大小的三维数组转换为 voxel_grid_index 数组。
        该函数接收一个形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，
        找出其中值为 1 的元素的索引，将这些索引组合成一个形状为 (N, 3) 的数组，
        类似于从 open3d 的 o3d.voxel_grid.get_voxels () 方法获取的结果。

    Args:
        voxel_array (numpy.ndarray): 形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，数组中值为 1 的位置代表对应的体素网格索引。

    Returns:

        numpy.ndarray: 形状为 (N, 3) 的数组，表示三维空间中每个体素的网格索引，类似于从 o3d.voxel_grid.get_voxels () 方法获取的结果。

    Example:

        ```python

        # 获取 grid_index_array
        voxel_list = voxel_grid.get_voxels()
        grid_index_array = list(map(lambda x: x.grid_index, voxel_list))
        grid_index_array = np.array(grid_index_array)
        voxel_grid_array = voxel2array(grid_index_array, voxel_size=32)
        grid_index_array = array2voxel(voxel_grid_array)
        pointcloud_array = grid_index_array  # 0.03125 是体素大小
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud_array)
        o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
        o3d.visualization.draw_geometries([pcd, cc, o3d_voxel])


        ```

    """
    x, y, z = np.where(voxel_array == 1)
    index_voxel = np.vstack((x, y, z))
    grid_index_array = index_voxel.T
    return grid_index_array


def homogenizing_mesh(vedo_mesh, target_num=10000):
    """
    对给定的 vedo 网格进行均质化处理，使其达到指定的目标面数。

    该函数使用 pyacvd 库中的 Clustering 类对输入的 vedo 网格进行处理。
    如果网格的顶点数小于等于目标面数，会先对网格进行细分，然后进行聚类操作，
    最终生成一个面数接近目标面数的均质化网格。

    Args:
        vedo_mesh (vedo.Mesh): 输入的 vedo 网格对象，需要进行均质化处理的网格。
        target_num (int, optional): 目标面数，即经过处理后网格的面数接近该值。
            默认为 10000。

    Returns:
        vedo.Mesh: 经过均质化处理后的 vedo 网格对象，其面数接近目标面数。

    Notes:
        该函数依赖于 pyacvd 和 pyvista 库，使用前请确保这些库已正确安装。

    """
    from pyacvd import Clustering
    from pyvista import wrap
    print(" Clustering target_num:{}".format(target_num))
    clus = Clustering(wrap(vedo_mesh.dataset))
    if vedo_mesh.npoints <= target_num:
        clus.subdivide(3)
    clus.cluster(target_num, maxiter=100, iso_try=10, debug=False)
    return vedo.Mesh(clus.create_mesh())


def fill_hole_with_center(mesh, boundaries, return_vf=False):
    """
        用中心点方式强制补洞

    Args:
        mesh (_type_): vedo.Mesh
        boundaries:vedo.boundaries
        return_vf: 是否返回补洞的mesh


    """
    vertices = mesh.vertices.copy()
    cells = mesh.cells

    # 获取孔洞边界的顶点坐标
    boundaries = boundaries.join(reset=True)
    if not boundaries:
        return mesh  # 没有孔洞
    pts_coords = boundaries.vertices

    # 将孔洞顶点坐标转换为原始顶点的索引
    hole_indices = []
    for pt in pts_coords:
        distances = np.linalg.norm(vertices - pt, axis=1)
        idx = np.argmin(distances)
        if distances[idx] < 1e-6:
            hole_indices.append(idx)
        else:
            raise ValueError("顶点坐标未找到")

    n = len(hole_indices)
    if n < 3:
        return mesh  # 无法形成面片

    # 计算中心点并添加到顶点
    center = np.mean(pts_coords, axis=0)
    new_vertices = np.vstack([vertices, center])
    center_idx = len(vertices)

    # 生成新的三角形面片
    new_faces = []
    for i in range(n):
        v1 = hole_indices[i]
        v2 = hole_indices[(i + 1) % n]
        new_faces.append([v1, v2, center_idx])

    if return_vf:
        return vedo.Mesh([new_vertices, new_faces]).clean().compute_normals()
    # 合并面片并创建新网格
    updated_cells = np.vstack([cells, new_faces])
    new_mesh = vedo.Mesh([new_vertices, updated_cells])
    return new_mesh.clean().compute_normals()


def collision_depth(mesh1, mesh2) -> float:
    """计算两个网格间的碰撞深度或最小间隔距离。

    使用VTK的带符号距离算法检测碰撞状态：
    - 正值：两网格分离，返回值为最近距离
    - 零值：表面恰好接触
    - 负值：发生穿透，返回值为最大穿透深度（绝对值）

    Args:
        mesh1 (vedo.Mesh): 第一个网格对象，需包含顶点数据
        mesh2 (vedo.Mesh): 第二个网格对象，需包含顶点数据

    Returns:
        float: 带符号的距离值，符号表示碰撞状态，绝对值表示距离量级

    Raises:
        RuntimeError: 当VTK计算管道出现错误时抛出

    Notes:
        1. 当输入网格顶点数>1000时会产生性能警告
        2. 返回float('inf')表示计算异常或无限远距离

    """
    # 性能优化提示
    if mesh1.npoints > 1000 or mesh2.npoints > 1000:
        print("[性能警告] 检测到高精度网格(顶点数>1000)，建议执行 mesh.decimate(n=500) 进行降采样")

    try:
        # 初始化VTK距离计算器
        distance_filter = vtk.vtkDistancePolyDataFilter()
        distance_filter.SetInputData(0, mesh1.dataset)
        distance_filter.SetInputData(1, mesh2.dataset)
        distance_filter.SignedDistanceOn()
        distance_filter.Update()

        # 提取距离数据
        distance_array = distance_filter.GetOutput().GetPointData().GetScalars("Distance")
        if not distance_array:
            return float('inf')

        return distance_array.GetRange()[0]

    except Exception as e:
        raise RuntimeError(f"VTK距离计算失败: {str(e)}") from e


def compute_curvature_by_meshlab(ms):
    """
    使用 MeshLab 计算网格的曲率和顶点颜色。

    该函数接收一个顶点矩阵和一个面矩阵作为输入，创建一个 MeshLab 的 MeshSet 对象，
    并将输入的顶点和面添加到 MeshSet 中。然后，计算每个顶点的主曲率方向，
    最后获取顶点颜色矩阵和顶点曲率数组。

    Args:
        ms: pymeshlab格式mesh;

    Returns:
        - vertex_colors (numpy.ndarray): 顶点颜色矩阵，形状为 (n, 3)，其中 n 是顶点的数量。
            每个元素的范围是 [0, 255]，表示顶点的颜色。
        - vertex_curvature (numpy.ndarray): 顶点曲率数组，形状为 (n,)，其中 n 是顶点的数量。
            每个元素表示对应顶点的曲率。
        - mesh: pymeshlab格式ms

    """
    ms.compute_curvature_principal_directions_per_vertex()
    curr_ms = ms.current_mesh()
    vertex_colors = curr_ms.vertex_color_matrix() * 255
    vertex_curvature = curr_ms.vertex_scalar_array()
    return vertex_colors, vertex_curvature, ms


def compute_curvature_by_igl(v, f):
    """
    用igl计算平均曲率并归一化

    Args:
        v: 顶点;
        f: 面片:

    Returns:
        - vertex_curvature (numpy.ndarray): 顶点曲率数组，形状为 (n,)，其中 n 是顶点的数量。
            每个元素表示对应顶点的曲率。


    """
    try:
        import igl
    except ImportError:
        print("请安装igl, pip install libigl")
    _, _, K, _ = igl.principal_curvature(v, f)
    K_normalized = (K - K.min()) / (K.max() - K.min())
    return K_normalized


def harmonic_by_igl(v, f, map_vertices_to_circle=True):
    """
    谐波参数化后的2D网格

    Args:
        v (_type_): 顶点
        f (_type_): 面片
        map_vertices_to_circle: 是否映射到圆形（正方形)

    Returns:
        uv,v_p: 创建参数化后的2D网格,3D坐标

    Note:

        ```

        # 创建空间索引
        uv_kdtree = KDTree(uv)

        # 初始化可视化系统
        plt = Plotter(shape=(1, 2), axes=False, title="Interactive Parametrization")

        # 创建网格对象
        mesh_3d = Mesh([v, f]).cmap("jet", calculate_curvature(v, f)).lighting("glossy")
        mesh_2d = Mesh([v_p, f]).wireframe(True).cmap("jet", calculate_curvature(v, f))

        # 存储选中标记
        markers_3d = []
        markers_2d = []

        def on_click(event):
            if not event.actor or event.actor not in [mesh_2d, None]:
                return
            if not hasattr(event, 'picked3d') or event.picked3d is None:
                return

            try:
                # 获取点击坐标
                uv_click = np.array(event.picked3d[:2])

                # 查找最近顶点
                _, idx = uv_kdtree.query(uv_click)
                v3d = v[idx]
                uv_point = uv[idx]  # 获取对应2D坐标


                # 创建3D标记（使用球体）
                marker_3d = Sphere(v3d, r=0.1, c='cyan', res=12)
                markers_3d.append(marker_3d)

                # 创建2D标记（使用大号点）
                marker_2d = Point(uv_point, c='magenta', r=10, alpha=0.8)
                markers_2d.append(marker_2d)

                # 更新视图
                plt.at(0).add(marker_3d)
                plt.at(1).add(marker_2d)
                plt.render()

            except Exception as e:
                print(f"Error processing click: {str(e)}")

        plt.at(0).show(mesh_3d, "3D Visualization", viewup="z")
        plt.at(1).show(mesh_2d, "2D Parametrization").add_callback('mouse_click', on_click)
        plt.interactive().close()


        ```

    """
    try:
        import igl
    except ImportError:
        print("请安装igl, pip install libigl")

    # 正方形边界映射）
    def map_to_square(bnd):
        n = len(bnd)
        quarter = n // 4
        uv = np.zeros((n, 2))
        for i in range(n):
            idx = i % quarter
            side = i // quarter
            t = idx / (quarter - 1)
            if side == 0:
                uv[i] = [1, t]
            elif side == 1:
                uv[i] = [1 - t, 1]
            elif side == 2:
                uv[i] = [0, 1 - t]
            else:
                uv[i] = [t, 0]
        return uv

    try:
        # 参数化
        bnd = igl.boundary_loop(f)
        if map_vertices_to_circle:
            bnd_uv = igl.map_vertices_to_circle(v, bnd)  # 圆形参数化
        else:
            bnd_uv = map_to_square(bnd)  # 正方形参数化
        uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    except Exception as e:
        print(f"生成错误，请检测连通体数量，{e}")
    # 创建参数化后的2D网格（3D坐标）
    v_p = np.hstack([uv, np.zeros((uv.shape[0], 1))])

    return uv, v_p


def hole_filling_by_Radial(boundary_coords):
    """
    参考

    [https://www.cnblogs.com/shushen/p/5759679.html]

    实现的最小角度法补洞法；

    Args:
        boundary_coords (_type_): 有序边界顶点

    Returns:
        v,f: 修补后的曲面


    Note:
        ```python

        # 创建带孔洞的简单网格
        s = vedo.load(r"J10166160052_16.obj")
        # 假设边界点即网格边界点
        boundary =vedo.Spline((s.boundaries().join(reset=True).vertices),res=100)
        # 通过边界点进行补洞
        filled_mesh =vedo.Mesh(hole_filling(boundary.vertices))
        # 渲染补洞后的曲面
        vedo.show([filled_mesh,boundary,s.alpha(0.8)], bg='white').close()

        ```

    """
    # 初始化顶点列表和边界索引
    vertex_list = np.array(boundary_coords.copy())
    boundary = list(range(len(vertex_list)))  # 存储顶点在vertex_list中的索引
    face_list = []

    while len(boundary) >= 3:
        # 1. 计算平均边长
        avg_length = 0.0
        n_edges = len(boundary)
        for i in range(n_edges):
            curr_idx = boundary[i]
            next_idx = boundary[(i + 1) % n_edges]
            avg_length += np.linalg.norm(vertex_list[next_idx] - vertex_list[curr_idx])
        avg_length /= n_edges

        # 2. 寻找最小内角顶点在边界列表中的位置
        min_angle = float('inf')
        min_idx = 0  # 默认取第一个顶点
        for i in range(len(boundary)):
            prev_idx = boundary[(i - 1) % len(boundary)]
            curr_idx = boundary[i]
            next_idx = boundary[(i + 1) % len(boundary)]

            v1 = vertex_list[prev_idx] - vertex_list[curr_idx]
            v2 = vertex_list[next_idx] - vertex_list[curr_idx]
            # 检查向量长度避免除以零
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm == 0 or v2_norm == 0:
                continue  # 跳过无效顶点
            cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
            angle = np.arccos(np.clip(cos_theta, -1, 1))
            if angle < min_angle:
                min_angle = angle
                min_idx = i  # 记录边界列表中的位置

        # 3. 获取当前处理的三个顶点索引
        curr_pos = min_idx
        prev_pos = (curr_pos - 1) % len(boundary)
        next_pos = (curr_pos + 1) % len(boundary)

        prev_idx = boundary[prev_pos]
        curr_idx = boundary[curr_pos]
        next_idx = boundary[next_pos]

        # 计算前驱和后继顶点的距离
        dist = np.linalg.norm(vertex_list[next_idx] - vertex_list[prev_idx])

        # 4. 根据距离决定添加三角形的方式
        if dist < 2 * avg_length:
            # 添加单个三角形
            face_list.append([prev_idx, curr_idx, next_idx])
            # 从边界移除当前顶点
            boundary.pop(curr_pos)
        else:
            # 创建新顶点并添加到顶点列表
            new_vertex = (vertex_list[prev_idx] + vertex_list[next_idx]) / 2
            vertex_list = np.vstack([vertex_list, new_vertex])
            new_idx = len(vertex_list) - 1

            # 添加两个三角形
            face_list.append([prev_idx, curr_idx, new_idx])
            face_list.append([curr_idx, next_idx, new_idx])

            # 更新边界：替换当前顶点为新顶点
            boundary.pop(curr_pos)
            boundary.insert(curr_pos, new_idx)

    return vertex_list, face_list


class A_Star:
    def __init__(self, vertices, faces):
        """
        使用A*算法在三维三角网格中寻找最短路径

        参数：
        vertices: numpy数组，形状为(N,3)，表示顶点坐标
        faces: numpy数组，形状为(M,3)，表示三角形面的顶点索引

        """
        self.adj = self.build_adjacency(faces)
        self.vertices = vertices

    def build_adjacency(self, faces):
        """构建顶点的邻接表"""
        from collections import defaultdict
        adj = defaultdict(set)
        for face in faces:
            for i in range(3):
                u = face[i]
                v = face[(i + 1) % 3]
                adj[u].add(v)
                adj[v].add(u)
        return {k: list(v) for k, v in adj.items()}

    def run(self, start_idx, end_idx, vertex_weights=None):
        """
        使用A*算法在三维三角网格中寻找最短路径

        参数：
        start_idx: 起始顶点的索引
        end_idx: 目标顶点的索引
        vertex_weights: 可选，形状为(N,)，顶点权重数组，默认为None

        返回：
        path: 列表，表示从起点到终点的顶点索引路径，若不可达返回None
        """
        import heapq
        end_coord = self.vertices[end_idx]

        # 启发式函数（当前顶点到终点的欧氏距离）
        def heuristic(idx):
            return np.linalg.norm(self.vertices[idx] - end_coord)

        # 优先队列：(f, g, current_idx)
        open_heap = []
        heapq.heappush(open_heap, (heuristic(start_idx), 0, start_idx))

        # 记录各顶点的g值和父节点
        g_scores = {start_idx: 0}
        parents = {}
        closed_set = set()

        while open_heap:
            current_f, current_g, current_idx = heapq.heappop(open_heap)

            # 若当前节点已处理且有更优路径，跳过
            if current_idx in closed_set:
                if current_g > g_scores.get(current_idx, np.inf):
                    continue
            # 找到终点，回溯路径
            if current_idx == end_idx:
                path = []
                while current_idx is not None:
                    path.append(current_idx)
                    current_idx = parents.get(current_idx)
                return path[::-1]

            closed_set.add(current_idx)

            # 遍历邻接顶点
            for neighbor in self.adj.get(current_idx, []):
                if neighbor in closed_set:
                    continue

                # 计算移动代价
                distance = np.linalg.norm(self.vertices[current_idx] - self.vertices[neighbor])
                if vertex_weights is not None:
                    cost = distance * vertex_weights[neighbor]
                else:
                    cost = distance

                tentative_g = current_g + cost

                # 更新邻接顶点的g值和父节点
                if tentative_g < g_scores.get(neighbor, np.inf):
                    parents[neighbor] = current_idx
                    g_scores[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        # 开放队列空，无路径
        return None


class MeshRandomWalks:
    def __init__(self, vertices, faces, face_normals=None):
        """
        随机游走分割网格

        参考：https://www.cnblogs.com/shushen/p/5144823.html

        Args:
            vertices: 顶点坐标数组，形状为(N, 3)
            faces: 面片索引数组，形状为(M, 3)
            face_normals: 可选的面法线数组，形状为(M, 3)


        Note:

            ```python

                # 加载并预处理网格
                mesh = vedo.load(r"upper_jaws.ply")
                mesh.compute_normals()

                # 创建分割器实例
                segmenter = MeshRandomWalks(
                    vertices=mesh.points,
                    faces=mesh.faces(),
                    face_normals=mesh.celldata["Normals"]
                )

                head = [1063,3571,1501,8143]
                tail = [7293,3940,8021]

                # 执行分割
                labels, unmarked = segmenter.segment(
                    foreground_seeds=head,
                    background_seeds=tail
                )
                p1 = vedo.Points(mesh.points[head],r=20,c="red")
                p2 = vedo.Points(mesh.points[tail],r=20,c="blue")
                # 可视化结果
                mesh.pointdata["labels"] = labels
                mesh.cmap("jet", "labels")
                vedo.show([mesh,p1,p2], axes=1, viewup='z').close()
            ```
        """
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces, dtype=int)
        self.face_normals = face_normals

        # 自动计算面法线（如果未提供）
        if self.face_normals is None:
            self.face_normals = self._compute_face_normals()

        # 初始化其他属性
        self.edge_faces = None
        self.edge_weights = None
        self.W = None  # 邻接矩阵
        self.D = None  # 度矩阵
        self.L = None  # 拉普拉斯矩阵
        self.labels = None  # 顶点标签
        self.marked = None  # 标记点掩码

    def _compute_face_normals(self):
        """计算每个面片的单位法向量"""
        normals = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            vec1 = v1 - v0
            vec2 = v2 - v0
            normal = np.cross(vec1, vec2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            else:
                normal = np.zeros(3)  # 处理退化面片
            normals.append(normal)
        return np.array(normals)

    def _compute_edge_face_map(self):
        """构建边到面片的映射关系"""

        from collections import defaultdict
        edge_map = defaultdict(list)
        for fid, face in enumerate(self.faces):
            for i in range(3):
                v0, v1 = sorted([face[i], face[(i + 1) % 3]])
                edge_map[(v0, v1)].append(fid)
        self.edge_faces = edge_map

    def _compute_edge_weights(self):
        """基于面片法线计算边权重"""
        self._compute_edge_face_map()
        self.edge_weights = {}

        for edge, fids in self.edge_faces.items():
            if len(fids) != 2:
                continue  # 只处理内部边

            # 获取相邻面法线
            n1, n2 = self.face_normals[fids[0]], self.face_normals[fids[1]]

            # 计算角度差异权重
            cos_theta = np.dot(n1, n2)
            eta = 1.0 if cos_theta < 0 else 0.2
            d = eta * (1 - abs(cos_theta))
            self.edge_weights[edge] = np.exp(-d)

    def _build_adjacency_matrix(self):
        """构建顶点邻接权重矩阵"""
        from scipy.sparse import csr_matrix, lil_matrix

        n = len(self.vertices)
        self.W = lil_matrix((n, n))

        for (v0, v1), w in self.edge_weights.items():
            self.W[v0, v1] = w
            self.W[v1, v0] = w

        self.W = self.W.tocsr()  # 转换为压缩格式提高效率

    def _build_laplacian_matrix(self):
        """构建拉普拉斯矩阵 L = D - W"""
        from scipy.sparse import csr_matrix
        degrees = self.W.sum(axis=1).A.ravel()
        self.D = csr_matrix((degrees, (range(len(degrees)), range(len(degrees)))))
        self.L = self.D - self.W

    def segment(self, foreground_seeds, background_seeds, vertex_weights=None):
        """
        执行网格分割

        参数:
            foreground_seeds: 前景种子点索引列表
            background_seeds: 背景种子点索引列表
            vertex_weights: 可选的顶点权重矩阵（稀疏矩阵）

        返回:
            labels: 顶点标签数组 (0: 背景，1: 前景)
            unmarked: 未标记顶点的布尔掩码
        """
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix
        # 初始化标签数组
        n = len(self.vertices)
        self.labels = np.full(n, -1, dtype=np.float64)
        self.labels[foreground_seeds] = 1.0
        self.labels[background_seeds] = 0.0

        # 处理权重矩阵
        if vertex_weights is not None:
            self.W = vertex_weights
        else:
            if not self.edge_weights:
                self._compute_edge_weights()
            if self.W is None:
                self._build_adjacency_matrix()

        # 构建拉普拉斯矩阵
        self._build_laplacian_matrix()

        # 分割问题求解
        self.marked = self.labels != -1
        L_uu = self.L[~self.marked, :][:, ~self.marked]
        L_ul = self.L[~self.marked, :][:, self.marked]
        rhs = -L_ul @ self.labels[self.marked]

        # 求解并应用阈值
        L_uu_reg = L_uu + 1e-9 * csr_matrix(np.eye(L_uu.shape[0]))  # 防止用户选择过少造成奇异值矩阵
        try:
            x = spsolve(L_uu_reg, rhs)
        except:
            # 使用最小二乘法作为备选方案
            x = np.linalg.lstsq(L_uu_reg.toarray(), rhs, rcond=None)[0]
        self.labels[~self.marked] = (x > 0.5).astype(int)

        return self.labels.astype(int), ~self.marked















