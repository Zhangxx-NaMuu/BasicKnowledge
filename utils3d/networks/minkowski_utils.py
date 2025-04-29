# -*- coding: UTF-8 -*-
"""
==================================================
@path   :BasicKnowledge -> minkowski_utils
@IDE    :PyCharm
@Author :NaMuu
@Email  :2458543125@qq.com
@Date   :2025/4/29 11:33
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2025/4/29 11:33:
==================================================  
"""
__author__ = 'zxx'
"""
闵可夫斯基引擎常用工具
"""
import torch.nn as nn
import torch
import MinkowskiEngine as ME
import numpy as np

@torch.no_grad()
def ravel_multi_index(coords, step):
    coords = coords.long()
    step = step.long()
    coords_sum = coords[:, 0] \
            + coords[:, 1]*step \
            + coords[:, 2]*step*step \
            + coords[:, 3]*step*step*step
    return coords_sum


@torch.no_grad()
def get_target_by_sp_tensor( out, coords_T):
    step = max(out.C.cpu().max(), coords_T.max()) + 1
    out_sp_tensor_coords_1d = ravel_multi_index(out.C.cpu(), step)
    target_coords_1d = ravel_multi_index(coords_T, step)
    # test whether each element of a 1-D array is also present in a second array.
    target = np.in1d(out_sp_tensor_coords_1d, target_coords_1d)
    return torch.Tensor(target).bool()





@torch.no_grad()
def choose_keep( out, feats,coords_T, device):
#   feats = torch.from_numpy(np.expand_dims(np.ones(coords_T.shape[0]), 1))
    x = ME.SparseTensor(features=feats.to(device), coordinates=coords_T.to(device))
    coords_nums = [len(coords) for coords in x.decomposed_coordinates]
    _,row_indices_per_batch = out.coordinate_manager.origin_map(out.coordinate_map_key)
    keep = torch.zeros(len(out), dtype=torch.bool)
    for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
        coords_num = min(len(row_indices), ori_coords_num)# select top k points.
#     			print(f"coords_num: {coords_num}")
#     			print(f"out.F shape: {out.F.shape}")
#     			print(f"row_indices shape: {row_indices.shape}")
#     			print(f"out.F[row_indices] shape: {out.F[row_indices].shape}")
        values, indices = torch.topk(out.F[row_indices], int(coords_num), dim=0)
        keep[row_indices[indices]]=True
    return keep


@torch.no_grad()
def get_target(out, target_key, kernel_size=1):
    target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
    cm = out.coordinate_manager
    strided_target_key = cm.stride(
        target_key, out.tensor_stride[0],
    )
    kernel_map = cm.kernel_map(
        out.coordinate_map_key,
        strided_target_key,
        kernel_size=kernel_size,
        region_type=1,
    )
    for k, curr_in in kernel_map.items():
        target[curr_in[0].long()] = 1
    return target

def valid_batch_map(batch_map):
    for b in batch_map:
        if len(b) == 0:
            return False
    return True


