
### <font face="微软雅黑" color=green size=5>vedo与trimesh互转问题</font>
```
# trimesh 转vedo
mesh = trimesh2vedo(mesh)
# vedo转trimesh
mesh = vedo.vedo2trimesh(mesh)
```


```
# trimesh 转open3d
mesh = mesh.as_open3d
# open3d转trimesh
把open3d的顶点和面片给trimesh
```

```
# 利用边界点生成平面
mesh = vedo.delaunay2d(edge_point)
```

```
# 提取边界点
mesh = vedo.load(path)
pids = mesh.boundaries(return_point_ids=True)

bpts = mesh.points()
# 创建边界线对象
pts = vedo.Points(bpts[pids], r=10, c='red5')
vedo.show(pts, __doc__, zoom=2).close()  
```