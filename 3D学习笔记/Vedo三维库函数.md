
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
