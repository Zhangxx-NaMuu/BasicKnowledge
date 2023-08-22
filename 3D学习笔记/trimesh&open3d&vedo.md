---
title: BasicKnowledge trimesh&open3d&vedo
date: 2023/8/22 15:03
tags:
    - cv
categories: 
    - 框架
    - opencv
---

## <font face="微软雅黑" color=green size=5>文件IO与save</font>


### <font face="微软雅黑" color=green size=5>open3d </font>
```python
mesh_path = 'test_gum.ply'
import numpy as np
import open3d as o3d
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()  # 计算顶点法向量
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 设置mesh颜色
vertices = np.asarray(mesh.vertices)  # 顶点
faces = np.asarray(mesh.triangles)  # 面片索引

# 已知顶点和面片，组成mesh
mesh1 = o3d.geometry.TriangleMesh()
mesh1.vertices = o3d.utility.Vector3dVector(vertices)
mesh1.triangles = o3d.utility.Vector3iVector(faces)

# 可视化mesh 
o3d.visualization.draw_geometries([mesh, mesh1])

# 输出mesh
o3d.io.write_triangle_mesh('test_gum.ply', mesh)
```

### <font face="微软雅黑" color=green size=5>trimesh</font>

```python
import trimesh
mesh_path = 'test_gum.ply'
mesh = trimesh.load(mesh_path)
vertices = mesh.vertices  # 顶点
faces = mesh.faces # 面片索引

# 已知顶点和面片，组成mesh
mesh1 = trimesh.Trimesh(vertices, faces)

# 可视化mesh
trimesh.Scene([mesh, mesh1]).show()
```

### <font face="微软雅黑" color=green size=5>vedo</font>

```python
import vedo
mesh_path = 'test_gum.ply'
mesh = vedo.load(mesh_path)
vertices = mesh.points()  # 顶点
faces = mesh.faces() # 面片索引

# 已知顶点和面片，组成mesh
mesh1 = vedo.Mesh([vertices, faces])

# 可视化mesh
vp = vedo.Plotter(N=2)
vp.at(0).show([mesh])
vp.at(1).show([mesh1])
vp.interactive().close()
```

## <font face="微软雅黑" color=green size=5>trimesh&open3d采样</font>

```
# trimesh
mesh_co_pcd = inputs.sample(pcd_samples_num)  
# open3d
mesh_co_pcd = inputs.sample_points_uniformly(number_of_points=pcd_samples_num)
mesh_co_pcd = np.asarray(mesh_co_pcd.points)
```
