# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> split_mesh_with_line
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/3/13 15:44
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/3/13 15:44:
==================================================  
"""
__author__ = 'zxx'

from vedo import *
import sindre
import vtk

ms = Mesh(r"C:\Users\dell\Desktop\ai_crown\583\upper_jaw.ply")
pts = load(r"C:\Users\dell\Desktop\ai_crown\583\11\margin.ply")

lines = Spline(pts, closed=True, res=100).c('red').join(reset=True)

#cms = ms.clone().cut_with_point_loop(lines,False)
cms = sindre.utils3d.cut_mesh_point_loop_crow(ms.clone(), lines)
show(ms, cms, lines, axes=8, bg='blackboard').close()
