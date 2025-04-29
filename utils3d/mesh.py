from functools import cached_property, lru_cache
import numpy as np
import json
from sindre.utils3d.algorithm import *


class SindreMesh:
    """三维网格中转类，假设都是三角面片 """

    def __init__(self, any_mesh=None) -> None:
        # 检查传入的参数

        if (isinstance(any_mesh, str)
                or "vtk" in str(type(any_mesh))
                or (isinstance(any_mesh, list) and len(any_mesh) == 2)
                or "meshlib" in str(type(any_mesh))
                or "meshio" in str(type(any_mesh))
        ):
            # 交由vedo处理
            import vedo
            self.any_mesh = vedo.Mesh(any_mesh)
        else:
            self.any_mesh = any_mesh

        self.vertices = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.vertex_curvature = None
        self.vertex_labels = None
        self.vertex_kdtree = None
        self.face_normals = None
        self.faces = None
        self._update()

    def set_vertex_labels(self, vertex_labels):
        """设置顶点labels,并自动渲染颜色"""
        self.vertex_labels = np.array(vertex_labels).reshape(-1, 1)
        self.vertex_colors = labels2colors(self.vertex_labels)[..., :3]

    def compute_normals(self, force=False):
        """计算顶点法线及面片法线.force代表是否强制重新计算"""
        if force or self.vertex_normals is None:
            self.vertex_normals = compute_vertex_normals(self.vertices, self.faces)
        if force or self.face_normals is None:
            self.face_normals = compute_face_normals(self.vertices, self.faces)

    def apply_transform_normals(self, mat):
        """处理顶点法线的变换（支持非均匀缩放和反射变换）"""
        # 提取3x3线性变换部分
        linear_mat = mat[:3, :3] if mat.shape == (4, 4) else mat
        # 计算法线变换矩阵（逆转置矩阵）(正交的逆转置是本身)
        try:
            inv_transpose = np.linalg.inv(linear_mat).T
        except np.linalg.LinAlgError:
            inv_transpose = np.eye(3)  # 退化情况处理

        # 应用变换并归一化
        self.vertex_normals = np.dot(self.vertex_normals, inv_transpose)
        norms = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6  # 防止除零
        self.vertex_normals /= norms

        # 将面片法线重新计算
        self.face_normals = None
        self.compute_normals()

    def apply_transform(self, mat):
        """对顶点应用4x4/3x3变换矩阵(支持非正交矩阵)"""
        if mat.shape[0] == 4:
            # 齐次坐标变换
            homogeneous = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
            self.vertices = (homogeneous @ mat.T)[:, :3]
        else:
            """对顶点应用3*3旋转矩阵"""
            self.vertices = np.dot(self.vertices, mat)

        # 计算法线
        self.apply_transform_normals(mat)

    def apply_inv_transform(self, mat):
        """对顶点应用4x4/3x3变换矩阵进行逆变换(支持非正交矩阵)"""
        mat = np.linalg.inv(mat)
        self.vertices = self.apply_transform(self.vertices, mat)

    def shift_xyz(self, dxdydz):
        """平移xyz指定量,支持输入3个向量和1个向量"""
        dxdydz = np.asarray(dxdydz, dtype=np.float64)  # 统一转换为数组
        if dxdydz.size == 1:
            delta = np.full(3, dxdydz.item())  # 标量扩展为三维
        elif dxdydz.size == 3:
            delta = dxdydz.reshape(3)  # 确保形状正确
        else:
            raise ValueError("dxdydz 应为标量或3元素数组")

        self.vertices += delta

    def scale_xyz(self, dxdydz):
        """缩放xyz指定量,支持输入3个向量和1个向量"""
        dxdydz = np.asarray(dxdydz, dtype=np.float64)
        if dxdydz.size == 1:
            scale = np.full(3, dxdydz.item())
        elif dxdydz.size == 3:
            scale = dxdydz.reshape(3)
        else:
            raise ValueError("dxdydz 应为标量或3元素数组")
        self.vertices *= scale

    def rotate_xyz(self, angles_xyz, return_mat=False):
        """按照给定xyz角度列表进行xyz对应旋转"""
        Rx = angle_axis_np(angles_xyz[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis_np(angles_xyz[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis_np(angles_xyz[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)
        if return_mat:
            return rotation_matrix
        else:
            self.apply_transform(rotation_matrix)

    def _update(self):
        try:
            self._convert()
        except Exception as e:
            raise RuntimeError(f"转换错误:{e}")
        # 给定默认颜色
        if self.vertex_colors is None:
            self.vertex_colors = np.ones_like(self.vertices) * np.array([255, 0, 0]).astype(np.uint8)
        # 给定默认标签
        if self.vertex_labels is None:
            self.vertex_labels = np.ones(len(self.vertices))
        # 给定默认曲率
        if self.vertex_curvature is None:
            self.vertex_curvature = np.zeros(len(self.vertices))

        if len(self.vertex_labels) != len(self.vertices):
            print(f"顶点发生改变，标签重新映射{len(self.vertex_labels), len(self.vertices)} ")
            self.vertex_labels = self.vertex_labels[self.get_near_idx(self.vertices)]

        if len(self.vertex_colors) != len(self.vertices):
            print(f"顶点发生改变，颜色重新映射{len(self.vertex_colors), len(self.vertices)} ")
            self.vertex_colors = self.vertex_colors[self.get_near_idx(self.vertices)]

        if len(self.vertex_curvature) != len(self.vertices):
            print(f"顶点发生改变，曲率重新映射 {len(self.vertex_curvature), len(self.vertices)}")
            self.vertex_curvature = self.vertex_curvature[self.get_near_idx(self.vertices)]

            # 重置kdtree
        self.vertex_kdtree = None

    def _convert(self):
        """将模型转换到类中"""
        inputobj_type = str(type(self.any_mesh))

        # Trimesh 转换
        if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.faces, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.face_normals, dtype=np.float64)

            if self.any_mesh.visual.kind == "face":
                self.vertex_colors = np.asarray(self.any_mesh.visual.face_colors, dtype=np.uint8)
            else:
                self.vertex_colors = np.asarray(self.any_mesh.visual.to_color().vertex_colors, dtype=np.uint8)

        # MeshLab 转换
        elif "MeshSet" in inputobj_type:
            mmesh = self.any_mesh.current_mesh()
            self.vertices = np.asarray(mmesh.vertex_matrix(), dtype=np.float64)
            self.faces = np.asarray(mmesh.face_matrix(), dtype=np.int32)
            self.vertex_normals = np.asarray(mmesh.vertex_normal_matrix(), dtype=np.float64)
            self.face_normals = np.asarray(mmesh.face_normal_matrix(), dtype=np.float64)
            if mmesh.has_vertex_color():
                self.vertex_colors = (np.asarray(mmesh.vertex_color_matrix())[..., :3] * 255).astype(np.uint8)



        # Open3D 转换
        elif "open3d" in inputobj_type:
            import open3d as o3d
            self.any_mesh.compute_vertex_normals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.triangles, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.triangle_normals, dtype=np.float64)

            if self.any_mesh.has_vertex_colors():
                self.vertex_colors = (np.asarray(self.any_mesh.vertex_colors)[..., :3] * 255).astype(np.uint8)

        # Vedo 转换
        elif "vedo" in inputobj_type:
            self.any_mesh.compute_normals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.cells, dtype=np.int32)
            self.vertex_normals = self.any_mesh.vertex_normals
            self.face_normals = self.any_mesh.cell_normals
            if self.any_mesh.pointdata["PointsRGBA"] is not None:
                self.vertex_colors = np.asarray(self.any_mesh.pointdata["PointsRGBA"][..., :3], dtype=np.uint8)


        # pytorch3d 转换
        elif "pytorch3d.structures.meshes.Meshes" in inputobj_type:
            self.any_mesh._compute_vertex_normals(True)
            self.vertices = np.asarray(self.any_mesh.verts_padded().cpu().numpy()[0], dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.faces_padded().cpu().numpy()[0], dtype=np.int32)
            self.vertex_normals = self.any_mesh.verts_normals_padded().cpu().numpy()[0]
            self.face_normals = self.any_mesh.faces_normals_padded().cpu().numpy()[0]
            if self.any_mesh.textures is not None:
                self.vertex_colors = np.asarray(self.any_mesh.textures.verts_features_padded().cpu().numpy()[0] * 255,
                                                dtype=np.uint8)


        elif "OCC" in inputobj_type:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods
            from OCC.Core.BRep import BRep_Tool

            BRepMesh_IncrementalMesh(self.any_mesh, 0.1).Perform()
            vertices = []
            faces = []
            vertex_index_map = {}
            current_index = 0
            explorer = TopExp_Explorer(self.any_mesh, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)

                if triangulation:
                    nb_nodes = triangulation.NbNodes()
                    for i in range(1, nb_nodes + 1):
                        pnt = triangulation.Node(i)
                        vertex = (pnt.X(), pnt.Y(), pnt.Z())
                        if vertex not in vertex_index_map:
                            vertex_index_map[vertex] = current_index
                            vertices.append(vertex)
                            current_index += 1
                    triangles = triangulation.Triangles()
                    for i in range(1, triangles.Length() + 1):
                        triangle = triangles.Value(i)
                        n1, n2, n3 = triangle.Get()
                        face_indices = [
                            vertex_index_map[
                                (triangulation.Node(n1).X(), triangulation.Node(n1).Y(), triangulation.Node(n1).Z())],
                            vertex_index_map[
                                (triangulation.Node(n2).X(), triangulation.Node(n2).Y(), triangulation.Node(n2).Z())],
                            vertex_index_map[
                                (triangulation.Node(n3).X(), triangulation.Node(n3).Y(), triangulation.Node(n3).Z())]
                        ]
                        faces.append(face_indices)
                explorer.Next()
            self.vertices = np.array(vertices, dtype=np.float64)
            self.faces = np.array(faces, dtype=np.int64)
        else:
            raise RuntimeError(f"不支持类型：{inputobj_type}")

    @property
    def to_occ(self):
        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakePolygon,
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_Sewing,
                BRepBuilderAPI_MakeSolid,
            )
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopAbs import TopAbs_SHELL
            from OCC.Core.TopoDS import topods
        except ImportError:
            raise f"请安装 ：conda install -c conda-forge pythonocc-core=7.8.1.1"
        vertices = self.vertices
        faces = self.faces
        sewing = BRepBuilderAPI_Sewing(0.1)
        for face_indices in faces:
            polygon = BRepBuilderAPI_MakePolygon()
            for idx in face_indices:
                x = float(vertices[idx][0])
                y = float(vertices[idx][1])
                z = float(vertices[idx][2])
                polygon.Add(gp_Pnt(x, y, z))  #
            polygon.Close()
            wire = polygon.Wire()
            face_maker = BRepBuilderAPI_MakeFace(wire)
            if face_maker.IsDone():
                sewing.Add(face_maker.Face())
            else:
                raise ValueError("无法从顶点创建面")
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        if sewed_shape.ShapeType() == TopAbs_SHELL:
            shell = topods.Shell(sewed_shape)
            solid_maker = BRepBuilderAPI_MakeSolid(shell)
            if solid_maker.IsDone():
                solid = solid_maker.Solid()
                # 网格化确保几何质量
                BRepMesh_IncrementalMesh(solid, 0.1).Perform()
                return solid
            else:
                print("警告：Shell无法生成Solid，返回Shell")
                return shell
        else:
            print("返回原始缝合结果（如Compound）")
            return sewed_shape

    @property
    def to_trimesh(self):
        """转换成trimesh"""
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals
        )
        mesh.visual.vertex_colors = self.vertex_colors
        return mesh

    @property
    def to_meshlab(self):
        """转换成meshlab"""
        import pymeshlab
        ms = pymeshlab.MeshSet()
        v_color_matrix = np.hstack([self.vertex_colors / 255, np.ones((len(self.vertices), 1), dtype=np.float64)])
        mesh = pymeshlab.Mesh(
            vertex_matrix=np.asarray(self.vertices, dtype=np.float64),
            face_matrix=np.asarray(self.faces, dtype=np.int32),
            v_normals_matrix=np.asarray(self.vertex_normals, dtype=np.float64),
            v_color_matrix=v_color_matrix,
        )
        ms.add_mesh(mesh)
        return ms

    @property
    def to_vedo(self):
        """转换成vedo"""
        from vedo import Mesh
        vedo_mesh = Mesh([self.vertices, self.faces])
        vedo_mesh.pointdata["Normals"] = self.vertex_normals
        vedo_mesh.pointdata["labels"] = self.vertex_labels
        vedo_mesh.pointdata["curvature"] = self.vertex_curvature
        vedo_mesh.celldata["Normals"] = self.face_normals
        vedo_mesh.pointcolors = self.vertex_colors
        return vedo_mesh

    @property
    def to_open3d(self):
        """转换成open3d"""
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors[..., :3] / 255.0)
        return mesh

    @property
    def to_open3d_t(self, device="CPU:0"):
        """转换成open3d_t"""
        import open3d as o3d
        device = o3d.core.Device(device)
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions = o3d.core.Tensor(self.vertices, dtype=dtype_f, device=device)
        mesh.triangle.indices = o3d.core.Tensor(self.faces, dtype=dtype_i, device=device)
        mesh.vertex.normals = o3d.core.Tensor(self.vertex_normals, dtype=dtype_f, device=device)
        mesh.vertex.colors = o3d.core.Tensor(self.vertex_colors[..., :3] / 255.0, dtype=dtype_f, device=device)
        mesh.vertex.labels = o3d.core.Tensor(self.vertex_labels, dtype=dtype_f, device=device)
        return mesh

    @property
    def to_dict(self):
        """将属性转换成python字典"""
        return {
            'vertices': self.vertices,
            'vertex_colors': self.vertex_colors,
            'vertex_normals': self.vertex_normals,
            'vertex_curvature': self.vertex_curvature,
            'vertex_labels': self.vertex_labels,
            'faces': self.faces,
        }

    @property
    def to_json(self):
        """转换成json"""
        return json.dumps(self.to_dict, cls=NpEncoder)

    def to_torch(self, device="cpu"):
        """将顶点&面片转换成torch形式

        Returns:
            vertices,faces,vertex_normals,vertex_colors: 顶点，面片,法线，颜色（没有则为None)
        """
        import torch
        vertices = torch.from_numpy(self.vertices).to(device, dtype=torch.float32)
        faces = torch.from_numpy(self.faces).to(device, dtype=torch.float32)

        vertex_normals = torch.from_numpy(self.vertex_normals).to(device, dtype=torch.float32)
        if self.vertex_colors is not None:
            vertex_colors = torch.from_numpy(self.vertex_colors).to(device, dtype=torch.int8)
        else:
            vertex_colors = None
        return vertices, faces, vertex_normals, vertex_colors

    def to_pytorch3d(self, device="cpu"):
        """转换成pytorch3d形式

        Returns:
            mesh : pytorch3d类型mesh
        """
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import TexturesVertex
        vertices = torch.from_numpy(self.vertices).to(device, dtype=torch.float32)
        faces = torch.from_numpy(self.faces).to(device, dtype=torch.float32)
        if self.vertex_colors is not None:
            verts_rgb = torch.from_numpy(self.vertex_colors) / 255
        else:
            verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb[None].to(device))
        mesh = Meshes(verts=vertices[None], faces=faces[None], textures=textures)
        return mesh

    def show(self, show_append=[], labels=None, exclude_list=[0], create_axes=True, return_vedo_obj=False):
        """
        渲染展示网格数据，并根据标签添加标记和坐标轴。

        Args:
            show_append (list) : 需要一起渲染的vedo属性
            labels (numpy.ndarray, optional): 网格顶点的标签数组，默认为None。如果提供，将根据标签为顶点着色，并为每个非排除标签添加标记。
            exclude_list (list, optional): 要排除的标签列表，默认为[0]。列表中的标签对应的标记不会被显示。
            create_axes: 是否强制绘制世界坐标系。
            return_vedo_obj: 是否返回vedo显示对象列表；

        Returns:
            None: 该方法没有返回值，直接进行渲染展示。
        """
        import vedo
        from sindre.utils3d.algorithm import labels2colors
        mesh_vd = self.to_vedo
        show_list = [] + show_append
        if labels is not None:
            labels = labels.reshape(-1)
            fss = self._labels_flag(mesh_vd, labels, exclude_list=exclude_list)
            show_list = show_list + fss
            colors = labels2colors(labels)
            mesh_vd.pointcolors = colors
            self.vertex_colors = colors

        show_list.append(mesh_vd)
        if create_axes:
            show_list.append(self._create_vedo_axes(mesh_vd))

        if return_vedo_obj:
            return show_list
        # 渲染
        vp = vedo.Plotter(N=1, title="SindreMesh", bg2="black", axes=3)
        vp.at(0).show(show_list)
        vp.close()

    def _create_vedo_axes(self, mesh_vd):
        """
        创建vedo的坐标轴对象。

        Args:
            mesh_vd (vedo.Mesh): vedo的网格对象，用于确定坐标轴的长度。

        Returns:
            vedo.Arrows: 表示坐标轴的vedo箭头对象。
        """
        import vedo
        bounds = mesh_vd.bounds()
        max_length = max([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        start_points = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        end_points = [(max_length, 0, 0), (0, max_length, 0), (0, 0, max_length)]
        colors = ['red', 'green', 'blue']
        arrows = vedo.Arrows(start_points, end_points, c=colors, shaft_radius=0.005, head_radius=0.01, head_length=0.02)
        return arrows

    def _labels_flag(self, mesh_vd, labels, exclude_list=[0]):
        """
        根据标签为网格的每个非排除类别创建标记。

        Args:
            mesh_vd (vedo.Mesh): vedo的网格对象。
            labels (numpy.ndarray): 网格顶点的标签数组。
            exclude_list (list, optional): 要排除的标签列表，默认为[0]。列表中的标签对应的标记不会被创建。

        Returns:
            list: 包含所有标记对象的列表。
        """
        fss = []
        for i in np.unique(labels):
            if int(i) not in exclude_list:
                v_i = mesh_vd.vertices[labels == i]
                cent = np.mean(v_i, axis=0)
                fs = mesh_vd.flagpost(f"{i}", cent)
                fss.append(fs)
        return fss

    def _count_duplicate_vertices(self):
        """统计重复顶点"""
        return len(self.vertices) - len(np.unique(self.vertices, axis=0))

    def _count_degenerate_faces(self):
        """统计退化面片"""
        areas = np.linalg.norm(self.face_normals, axis=1) / 2
        return np.sum(areas < 1e-8)

    def _count_connected_components(self):
        """计算连通体数量"""
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(self.get_adj_matrix, directed=False)
        return n_components, labels

    def _count_unused_vertices(self):
        """统计未使用顶点"""
        used = np.unique(self.faces)
        return len(self.vertices) - len(used)

    def _is_watertight(self):
        """判断是否闭合"""
        unique_edges = np.unique(np.sort(self.get_edges, axis=1), axis=0)
        return len(self.get_edges) == 2 * len(unique_edges)

    def get_color_mapping(self, value):
        """将向量映射为颜色，遵从vcg映射标准"""
        import matplotlib.colors as mcolors
        colors = [
            (1.0, 0.0, 0.0, 1.0),  # 红
            (1.0, 1.0, 0.0, 1.0),  # 黄
            (0.0, 1.0, 0.0, 1.0),  # 绿
            (0.0, 1.0, 1.0, 1.0),  # 青
            (0.0, 0.0, 1.0, 1.0)  # 蓝
        ]
        cmap = mcolors.LinearSegmentedColormap.from_list("VCG", colors)
        norm = mcolors.Normalize(vmin=-1, vmax=1)
        value = norm(np.asarray(value))
        rgba = cmap(value)
        return (rgba * 255).astype(np.uint8)

    def subdivison(self, face_mask, iterations=3, method="mid"):
        """局部细分"""

        assert len(face_mask) == len(self.faces), "face_mask长度不匹配:要求每个面片均有对应索引"
        import pymeshlab
        if int(face_mask).max() != 1:
            # # 索引值转bool值
            face_mask = np.any(np.isin(self.faces, face_mask), axis=1)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.vertices, face_matrix=self.faces, f_scalar_array=face_mask))
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
        self.any_mesh = ms
        self._convert()

    def texture2colors(self, image_path="texture_uv.png", uv=None):
        """将纹理贴图转换成顶点颜色"""
        from PIL import Image
        from scipy.ndimage import map_coordinates
        if uv is None:
            uv = self.get_uv()
        texture = np.array(Image.open(image_path))

        # 转换UV到图像坐标（考虑翻转V轴）
        h, w = texture.shape[:2]
        u_coords = uv[:, 0] * (w - 1)
        v_coords = (1 - uv[:, 1]) * (h - 1)
        coords = np.vstack([v_coords, u_coords])  # scipy的坐标格式为(rows, cols)
        channels = []
        for c in range(3):
            sampled = map_coordinates(texture[:, :, c], coords, order=1, mode='nearest')
            channels.append(sampled)

        self.vertex_colors = np.stack(channels, axis=1).astype(np.uint8)

    def get_texture(self, write_path="texture_uv.png", image_size=(512, 512), uv=None):
        """将颜色转换为纹理贴图,  Mesh([v, f]).texture(write_path,uv)"""
        from PIL import Image
        from scipy.interpolate import griddata
        if uv is None:
            uv = self.get_uv()

        def compute_interpolation_map(shape, tcoords, values):
            points = (tcoords * np.asarray(shape)[None, :]).astype(np.int32)
            x = np.arange(shape[0])
            y = np.flip(np.arange(shape[1]))
            X, Y = np.meshgrid(x, y)
            res = griddata(points, values, (X, Y), method='nearest')
            res[np.isnan(res)] = 0
            return res

        texture_map = compute_interpolation_map(image_size, uv, self.vertex_colors)
        Image.fromarray(texture_map.astype(np.uint8)).save(write_path)

    def sample(self, density=1, num_samples=None):
        """
        网格表面上进行点云重采样
        Args:
            density (float, 可选): 每单位面积的采样点数，默认为1
            num_samples (int, 可选): 指定总采样点数N，若提供则忽略density参数

        Returns:
            numpy.ndarray: 重采样后的点云数组，形状为(N, 3)，N为总采样点数

        """

        return resample_mesh(vertices=self.vertices, faces=self.faces, density=density, num_samples=num_samples)

    def decimate(self, n=10000):
        """将网格下采样到指定点数，采用面塌陷"""
        vd_ms = self.to_vedo.decimate(n=n)
        self.any_mesh = vd_ms
        self._update()

    def homogenize(self, n=10000):
        """ 均匀化网格到指定点数，采用聚类"""
        self.any_mesh = isotropic_remeshing_by_acvd(self.to_vedo, target_num=n)
        self._update()

    def check(self):
        """检测数据完整性,正常返回True"""
        checks = [
            self.vertices is not None,
            self.faces is not None,
            not np.isnan(self.vertices).any() if self.vertices is not None else False,
            not np.isinf(self.vertices).any() if self.vertices is not None else False,
            not np.isnan(self.vertex_normals).any() if self.vertex_normals is not None else False
        ]
        return all(checks)

    def get_curvature(self):
        """获取曲率"""
        vd_ms = self.to_vedo.compute_curvature(method=1)
        self.vertex_curvature = vd_ms.pointdata["Mean_Curvature"]
        self.vertex_colors = self.get_color_mapping(self.vertex_curvature)

    def get_curvature_advanced(self):
        """获取更加精确曲率,但要求网格质量"""
        try:
            # 限制太多，舍弃
            assert self.npoints < 100000, "顶点必须小于10W"
            assert len(self.get_non_manifold_edges) == 0, "存在非流形"
            assert self._count_connected_components()[0] == 1, "连通体数量应为1"
            ms = self.to_meshlab
            ms.compute_curvature_principal_directions_per_vertex(autoclean=False)
            mmesh = ms.current_mesh()
            self.vertex_colors = (mmesh.vertex_color_matrix() * 255)[..., :3]
            self.vertex_curvature = mmesh.vertex_scalar_array()
        except Exception as e:
            print(f"无法使用使用meshlab计算主曲率,{e}")
            self.vertex_curvature = compute_curvature_by_igl(self.vertices, self.faces, False)
            self.vertex_colors = self.get_color_mapping(self.vertex_curvature)

    def get_near_idx(self, query_vertices):
        """获取最近索引"""
        if self.vertex_kdtree is None:
            self.vertex_kdtree = KDTree(self.vertices)
        _, idx = self.vertex_kdtree.query(query_vertices, workers=-1)
        return idx

    def remesh(self):
        ms = self.to_meshlab
        # 去除较小连通体
        fix_component_by_meshlab(ms)
        # 先修复非流形
        fix_topology_by_meshlab(ms)
        # 清理无效结构
        fix_invalid_by_meshlab(ms)
        # 更新信息
        self.any_mesh = ms
        self._update()

    @lru_cache(maxsize=None)
    def get_uv(self, return_circle=False):
        """ 获取uv映射 与顶点一致(npoinst,2) """
        uv, _ = harmonic_by_igl(self.vertices, self.faces, map_vertices_to_circle=return_circle)
        return uv

    @cached_property
    def npoints(self):
        """获取顶点数量"""
        return len(self.vertices)

    @cached_property
    def nfaces(self):
        """获取顶点数量"""
        return len(self.faces)

    @cached_property
    def faces_vertices(self):
        """将面片索引用顶点来表示"""
        return self.vertices[self.faces]

    @cached_property
    def faces_area(self):
        """
        计算每个三角形面片的面积。

        Notes:
            使用叉乘公式计算面积：
            面积 = 0.5 * ||(v1 - v0) × (v2 - v0)||
        """
        tri_vertices = self.faces_vertices
        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        area = 0.5 * np.linalg.norm(np.cross((v1 - v0), (v2 - v0)), axis=1)
        return area

    @cached_property
    def faces_center(self):
        """每个三角形的中心（重心 [1/3,1/3,1/3]）"""
        return self.faces_vertices.mean(axis=1)

    @cached_property
    def center(self) -> np.ndarray:
        """计算网格的加权质心（基于面片面积加权）。

        Returns:
            np.ndarray: 加权质心坐标，形状为 (3,)。

        Notes:
            使用三角形面片面积作为权重，对三角形质心坐标进行加权平均。
            该结果等价于网格的几何中心。
        """
        return np.average(self.faces_center, weights=self.faces_area, axis=0)

    @cached_property
    def get_adj_matrix(self):
        """基于去重边构建邻接矩阵"""
        from scipy.sparse import csr_matrix
        n = len(self.vertices)
        edges = np.unique(self.get_edges, axis=0)
        data = np.ones(edges.shape[0] * 2)  # 两条边（无向图）
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    @cached_property
    def get_adj_list(self):
        """邻接表属性"""
        edges = np.unique(self.get_edges, axis=0)  # 去重
        adj = [[] for _ in range(len(self.vertices))]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    @cached_property
    def get_edges(self):
        """未去重边缘属性 """
        edges = np.concatenate([self.faces[:, [0, 1]],
                                self.faces[:, [1, 2]],
                                self.faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=1)  # 确保边无序性
        return edges

    @cached_property
    def get_non_manifold_edges(self):
        # 提取有效边并排序
        edges = self.get_edges
        valid_edges = edges[edges[:, 0] != edges[:, 1]]
        edges_sorted = np.sort(valid_edges, axis=1)
        unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        # 返回非流形边
        return unique_edges[counts >= 3]

    def __repr__(self):
        """网格质量检测"""

        stats = [
            "\033[91m\t网格质量检测(numpy): \033[0m",
            f"\033[94m顶点数:             {len(self.vertices)} \033[0m",
            f"\033[94m面片数:             {len(self.faces)}\033[0m",
            f"\033[94m网格水密(闭合):     {self._is_watertight()}\033[0m",
            f"\033[94m连通体数量：        {self._count_connected_components()[0]}\033[0m",
            f"\033[94m未使用顶点:         {self._count_unused_vertices()}\033[0m",
            f"\033[94m重复顶点:           {self._count_duplicate_vertices()}\033[0m",
            f"\033[94m网格退化:           {self._count_degenerate_faces()}\033[0m",
            f"\033[94m法线异常:           {np.isnan(self.vertex_normals).any()}\033[0m",
            f"\033[94m边流形:             {len(self.get_non_manifold_edges) == 0}\033[0m",
        ]

        return "\n".join(stats)

    def print_o3d(self):
        """使用open3d网格质量检测"""
        mesh = self.to_open3d
        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        orientable = mesh.is_orientable()

        stats = [
            "\033[91m\t网格质量检测(open3d): \033[0m",
            f"\033[94m顶点数:             {len(self.vertices) if self.vertices is not None else 0}\033[0m",
            f"\033[94m面片数:             {len(self.faces) if self.faces is not None else 0}\033[0m",
            f"\033[94m网格水密(闭合):     {self._is_watertight()}\033[0m",
            f"\033[94m连通体数量：        {self._count_connected_components()[0]}\033[0m",
            f"\033[94m未使用顶点:         {self._count_unused_vertices()}\033[0m",
            f"\033[94m重复顶点:           {self._count_duplicate_vertices()}\033[0m",
            f"\033[94m网格退化:           {self._count_degenerate_faces()}\033[0m",
            f"\033[94m法线异常:           {np.isnan(self.vertex_normals).any() if self.vertex_normals is not None else True}\033[0m",
            f"\033[94m边为流形:           {edge_manifold}\033[0m",
            f"\033[94m边的边界为流形:     {edge_manifold_boundary}\033[0m",
            f"\033[94m顶点为流形:         {vertex_manifold}\033[0m",
            f"\033[94m可定向:             {orientable}\033[0m",
        ]

        print("\n".join(stats))




