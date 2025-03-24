# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> mesh_tools
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/14 13:39
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/14 13:39:
==================================================  
"""
__author__ = 'zxx'
from functools import cached_property
import numpy as np
import json
from tools import NpEncoder

class Mesh:
    def __init__(self, any_mesh) -> None:
        self.any_mesh = any_mesh
        self.vertices = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.face_normals = None
        self.faces = None
        try:
            self._convert()
        except Exception as e:
            raise RuntimeError(f"Error in converting mesh: {e}")

    def _convert(self):
        """
        Convert the mesh to a standard format.
        将模型转换到类中
        """
        inputobj_type = str(type(self.any_mesh))  # 获取mesh的类型
        # trimesh转换
        if "Trimesh" in inputobj_type or "primitives" in inputobj_type:
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.faces, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.face_normals, dtype=np.float64)

            if self.any_mesh.visual.kind == "face":
                self.vertex_colors = np.asarray(self.any_mesh.visual.face_colors, dtype=np.uint8)
            else:
                self.vertex_colors = np.asarray(self.any_mesh.visual.to_color().vertex_colors, dtype=np.uint8)

        # MeshLab转换
        elif "MeshSet" in inputobj_type:
            import pymeshlab
            mmesh = self.any_mesh.current_mesh()
            self.vertices = np.asarray(mmesh.vertex_matrix(), dtype=np.float64)
            self.faces = np.asarray(mmesh.face_matrix(), dtype=np.int32)

        # Open3D转换
        elif "open3d" in inputobj_type:
            import open3d as o3d
            self.any_mesh.compute_vertex_normals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.triangles, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.triangle_normals, dtype=np.float64)

            if self.any_mesh.has_vertex_colors():
                self.vertex_colors = (np.asarray(self.any_mesh.vertex_colors) * 255).astype(np.uint8)

        # vedo/vtk转换
        elif "vedo" in inputobj_type or "vtk" in inputobj_type:
            self.any_mesh.computeNormals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.cells, dtype=np.int32)
            self.vertex_normals = self.any_mesh.vertex_ormals
            self.face_normals = self.any_mesh.cell_normals
            self.vertex_colors = self.any_mesh.pointdata["RGBA"]


    def to_trimesh(self):
        """
        Convert the mesh to a trimesh object.
        将模型转换为trimesh对象
        """
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals
        )
        if self.vertex_colors is not None:
            mesh.visual.vertex_colors = self.vertex_colors
        return mesh

    def to_meshlab(self):
        """
        Convert the mesh to a MeshLab object.
        将模型转换为MeshLab对象
        """
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            vertex_matrix=self.vertices,
            face_matrix=self.faces,
        )
        return ms

    def to_vedo(self):
        """
        Convert the mesh to a vedo object.
        将模型转换为vedo对象
        """
        from vedo import Mesh
        vedo_mesh = Mesh([self.vertices, self.faces])
        if self.vertex_colors is not None:
            vedo_mesh.pointcolors(self.vertex_colors)
        return vedo_mesh

    def to_open3d(self):
        """
        Convert the mesh to an Open3D object.
        将模型转换为Open3D对象
        """
        import open3d as o3d
        open3d_mesh = o3d.geometry.TriangleMesh()
        open3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        open3d_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        if self.vertex_normals is not None:
            open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertex_colors is not None:
            open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors / 255.0)

        return open3d_mesh

    def to_dict(self):
        """
        Convert the mesh to a python dictionary.
        将属性转换为python字典
        """
        return {
            "vertices": self.vertices if self.vertices is not None else [],
            "faces": self.faces if self.faces is not None else [],
            "vertex_colors": self.vertex_colors if self.vertex_colors is not None else [],
            "vertex_normals": self.vertex_normals if self.vertex_normals is not None else [],
        }

    def to_json(self):
        """
        Convert the mesh to a JSON string.
        将属性转换为JSON字符串
        """
        return json.dumps(self.to_dict(), cls=NpEncoder)

    def to_torch(self):
        """
        Convert the mesh to a PyTorch tensor.
        将顶点&面片转换成torch形式
        return:
            vertices: (N, 3)
            faces: (M, 3)
        """
        import torch
        vertices = torch.from_numpy(self.vertices).float()
        faces = torch.from_numpy(self.faces)
        return vertices, faces

    def to_pytorch3d(self):
        """
        Convert the mesh to a PyTorch3D tensor.
        转换成pytroch3d的格式
        return:
            mesh: pytorch3d 类型mesh
        """
        from pytorch3d.structures import Meshes
        v, f = self.to_torch()
        mesh = Meshes(verts=v[None], faces=f[None])
        return mesh

    def _count_duplicate_vertices(self):
        """
        Count the number of duplicate vertices in the mesh.
        计算模型中重复顶点的数量
        """
        return len(self.vertices) - len(np.unique(self.vertices, axis=0)) if self.vertices is not None else 0

    def _count_degenerate_faces(self):
        """
        Count the number of degenerate faces in the mesh.
        计算模型中退化面片的数量
        """
        if self.faces is None:
            return 0
        areas = np.linalg.norm(self.face_normals, axis=1) / 2
        return np.sum(areas < 1e-8)


    def _count_connected_components(self):
        """
        Count the number of connected components in the mesh.
        计算模型中连通分量的数量
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        if self.vertices is None:
            return 0
        n = len(self.vertices)
        data = np.ones(len(self.faces) * 3)
        rows = self.faces.flatten()
        cols = np.roll(self.faces, shift=1, axis=1).flatten()
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        return connected_components(adj, directed=False)

    def _count_unused_vertices(self):
        """统计未使用顶点"""
        if self.vertices is None or self.faces is None:
            return 0
        used = np.unique(self.faces)
        return len(self.vertices) - len(used)

    def _is_watertight(self):
        """
        Check if the mesh is watertight.
        检查模型是否为水密模型
        """
        if self.faces is None:
            return False
        edges = np.concatenate([self.faces[:, :2], self.faces[:, 1:], self.faces[:, [2, 0]]])
        unique_edges = np.unique(np.sort(edges, axis=1), axis=0)
        return 2 *len(unique_edges) == len(edges)

    def to_texture(self):
        """
        将颜色转换为纹理贴图
        """
        if self.vertex_colors is not None:
            return self.vertex_colors.reshape(-1, 3)
        return None

    def check(self):
        """
        检测数据完整性，正常返回True，异常返回False
        """
        checks = [
            self.vertices is not None,
            self.faces is not None,
            not np.isnan(self.vertices).any() if self.vertices is not None else False,
            not np.isinf(self.vertices).any() if self.vertices is not None else False,
            not np.isnan(self.vertex_normals).any() if self.vertex_normals is not None else False
        ]
        return all(checks)

    def __repr__(self):
        return self.get_quality()

    def get_quality(self):
        """
        获取模型质量
        """
        mesh = self.to_open3d()
        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        orientable = mesh.is_orientable()

        stats = [
            "\033[91m\t网格质量检测: \033[0m",
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

        return "\n".join(stats)

def sdf2mesh_by_diso(sdf,diffdmc=None ,deform=None,return_quads=False, normalize=True,isovalue=0 ,invert=True):
    """
    exampel:
    occ = torch.tensor(occ.reshape((256,)*3)).to(device)
    vertices, triangles = sdf2mesh_by_diso(sdf=occ, normalize=False)
    vertices, triangles = vertices.cpu().numpy(), triangles.cpu().numpy()
    """
    try:
        from diso import DiffDMC
    except ImportError:
        print("请安装 pip install diso")
    if diffdmc is None:
        diffdmc =DiffDMC(dtype=torch.float32).cuda()
    if invert:
        sdf*=-1
    v, f = diffdmc(sdf, deform, return_quads=return_quads, normalize=normalize, isovalue=isovalue)
    return v,f




