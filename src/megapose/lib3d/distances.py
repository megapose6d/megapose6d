"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""



# Third Party
import torch

# MegaPose
from megapose.lib3d.transform_ops import transform_pts


def dists_add(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points - TXO_pred_points
    return dists


def dists_add_symmetries(TXO_pred, TXO_gt_possible, points):
    bsz = TXO_pred.shape[0]
    TXO_pred_points = transform_pts(TXO_pred, points).unsqueeze(1)
    TXO_gt_points = transform_pts(TXO_gt_possible, points)
    dists = TXO_gt_points - TXO_pred_points
    dists_norm = torch.norm(dists, dim=-1, p=2).mean(-1)
    min_dist, min_id = dists_norm.min(dim=1)
    dists = dists[torch.arange(bsz), min_id]
    return dists


def dists_add_symmetric(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)
    dists_norm_squared = (dists**2).sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    dists = dists[ids_row, assign, ids_col]
    return dists
