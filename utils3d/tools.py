import json
import vedo
import numpy as np
from typing import *
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
from scipy.linalg import eigh
import vtk
import trimesh
import os


def fdi2idx(labels):
    """

    将口腔牙列的fid (11-18,21-28,31-38,41-48) 转换成1-18;

    """

    if labels.max() > 30:
        labels -= 20
    labels[labels // 10 == 1] %= 10
    labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
    labels[labels < 0] = 0
    return labels


def labels2colors(labels: np.array):
    """
    将labels转换成颜色标签
    Args:
        labels: numpy类型,形状(N)对应顶点的标签；

    Returns:
        RGBA颜色标签;
    """
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


def tooth_labels_to_color(data: Union[np.array, list]) -> list:
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


def apply_pac_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        对pca获得4*4矩阵进行应用

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


def restore_pca_transform(vertices: np.array, transform: np.array) -> np.array:
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


def rotation_crown(near_mesh: vedo.Mesh, jaw_mesh: vedo.Mesh) -> vedo.Mesh:
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
    new_m = vedo.Mesh([apply_pac_transform(near_mesh.points(), transform), near_mesh.faces()])
    return new_m


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
        invert (bool, optional): 选择保留最大/最小模式. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """

    # 去除不相关的联通体
    regions = mesh.split()

    def batch_closest_dist(vertices, curve_pts):
        # 将曲线点集转为矩阵（n×3）
        curve_matrix = np.array(curve_pts)
        # 计算顶点到曲线点的所有距离（矩阵运算）
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.min(dist_matrix, axis=1)

    # 计算各区域到曲线的最近距离
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh = regions[np.argmin(min_dists)]

    # 切割网格并设置EdgeSearchMode
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)  # 直接获取VTK数据
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOff()
    selector.SetEdgeSearchModeToDijkstra()  # 设置搜索模式
    if invert:
        selector.SetSelectionModeToLargestRegion()
    selector.SetSelectionModeToSmallestRegion()
    selector.Update()

    cut_mesh = vedo.Mesh(selector.GetOutput())
    return cut_mesh


def cut_mesh_point_loop_crow(mesh, pts):
    """

    基于vtk+dijkstra实现的基于线的牙齿冠分割;

    线支持在网格上或者网格外；

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points): 切割线
        invert (bool, optional): 选择保留最大/最小模式. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """
    # 去除不相关的联通体
    regions = mesh.split()

    def batch_closest_dist(vertices, curve_pts):
        # 将曲线点集转为矩阵（n×3）
        curve_matrix = np.array(curve_pts)
        # 计算顶点到曲线点的所有距离（矩阵运算）
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.min(dist_matrix, axis=1)

    # 计算各区域到曲线的最近距离
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh = regions[np.argmin(min_dists)]

    # 切割网格并设置EdgeSearchMode
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOff()
    selector.SetEdgeSearchModeToDijkstra()  # 设置搜索模式
    if np.min(min_dists) < 0.1:
        print("mesh已经被裁剪")
        selector.SetSelectionModeToClosestPointRegion()
    else:
        selector.SetSelectionModeToSmallestRegion()
    selector.Update()
    cut_mesh = vedo.vedo.Mesh(selector.GetOutput()).clean()
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


def isotropic_remeshing_pymeshlab(vertices, faces, target_edge_length=0.5, iterations=2) -> vedo.Mesh:
    """
    使用 PyMeshLab 实现网格均匀化。

    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。
        target_edge_length: 目标边长比例 %。
        iterations: 迭代次数，默认为 10。

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


def optimize_mesh_by_meshlab(vertices, faces) -> vedo.Mesh:
    """
    使用 PyMeshLab 实现一键优化网格。

    ```

    Merge Close Vertices：合并临近顶点
    Merge Wedge Texture Coord：合并楔形纹理坐标
    Remove Duplicate Faces：移除重复面
    Remove Duplicate Vertices：移除重复顶点
    Remove Isolated Folded Faces by Edge Flip：通过边翻转移除孤立的折叠面
    Remove Isolated pieces (wrt diameter)：移除孤立部分（相对于直径）
    Remove Isolated pieces (wrt Face Num.)：移除孤立部分（相对于面数）
    Remove T-Vertices：移除 T 型顶点
    Remove Unreferenced Vertices：移除未引用的顶点
    Remove Vertices wrt Quality：根据质量移除顶点
    Remove Zero Area Faces：移除零面积面
    Repair non Manifold Edges：修复非流形边
    Repair non Manifold Vertices by splitting：通过拆分修复非流形顶点
    Snap Mismatched Borders ：对齐不匹配的边界


    ```





    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。

    Returns:
        优化后的网格对象。
    """
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    # 1. 合并临近顶点
    ms.apply_filter("meshing_merge_close_vertices")

    # 2. 合并楔形纹理坐标
    ms.apply_filter("apply_texcoord_merge_per_wedge")

    # 3. 移除重复面
    ms.apply_filter("meshing_remove_duplicate_faces")

    # 4. 移除重复顶点
    ms.apply_filter("meshing_remove_duplicate_vertices")

    # 5. 通过边翻转移除孤立的折叠面
    ms.apply_filter("meshing_remove_folded_faces")

    # 6. 移除孤立部分（基于直径）
    ms.apply_filter("meshing_remove_connected_component_by_diameter")

    # 7. 移除孤立部分（基于面数）
    ms.apply_filter("meshing_remove_connected_component_by_face_number", mincomponentsize=10)

    # 8. 移除 T 型顶点（文档中无直接对应过滤器）
    ms.apply_filter("meshing_remove_t_vertices")

    # 9. 移除未引用的顶点
    ms.apply_filter("meshing_remove_unreferenced_vertices")

    # 10. 根据质量移除顶点（需自定义质量阈值）
    ms.apply_filter("meshing_remove_vertices_by_scalar", maxqualitythr=pymeshlab.PercentageValue(10))

    # 11. 移除零面积面
    ms.apply_filter("meshing_remove_null_faces")

    # 12. 修复非流形边
    ms.apply_filter("meshing_repair_non_manifold_edges")

    # 13. 通过拆分修复非流形顶点
    ms.apply_filter("meshing_repair_non_manifold_vertices")

    # 15. 对齐不匹配的边界
    ms.apply_filter("meshing_snap_mismatched_borders")

    return vedo.Mesh(ms.current_mesh())


def labels_mapping(old_mesh, new_mesh, old_labels):
    """
    将原始网格的标签属性精确映射到新网格

    参数:
        old_mesh(vedo) : 原始网格对象
        new_mesh(vedo): 重网格化后的新网格对象
        old_labels (np.ndarray): 原始顶点标签数组，形状为 (N,)

    返回:
        new_labels (np.ndarray): 映射后的新顶点标签数组，形状为 (M,)
    """
    import trimesh

    old_mesh = trimesh.Trimesh(old_mesh.vertices, old_mesh.cells)
    if len(old_labels) != len(old_mesh.vertices):
        raise ValueError(f"标签数量 ({len(old_labels)}) 必须与原始顶点数 ({len(old_mesh.vertices)}) 一致")

    new_vertices = new_mesh.vertices

    # 步骤1: 查询每个新顶点在原始网格上的最近面片信息
    # closest_points: 新顶点在原始网格上的投影坐标 (M,3)
    # distances: 新顶点到投影点的距离 (M,)
    # tri_ids: 最近面片的索引 (M,)
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
        _, near_points = tree.query(points, k=k)
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
        self.mesh = trimesh.Trimesh(vertices, faces)
        self._precompute_geometry()
        self.smooth_factor = smooth_factor
        self.keep_label = keep_label

        # 处理标签映射
        self.unique_labels, mapped_labels = np.unique(vertex_labels, return_inverse=True)
        if temperature is None:
            self.temperature = self._compute_temperature(mapped_labels)
        else:
            self.temperature = temperature
        self.prob_matrix = self._labels_to_prob(mapped_labels, self.unique_labels.size)

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
