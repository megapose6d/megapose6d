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

# Local Folder
from .transform_ops import transform_pts


def compute_ADDS_loss(TCO_gt, TCO_pred, points):
    assert TCO_gt.dim() == 3 and TCO_gt.shape[-2:] == (4, 4)
    assert TCO_pred.shape[-2:] == (4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    TXO_gt_points = transform_pts(TCO_gt, points)
    TXO_pred_points = transform_pts(TCO_pred, points)
    dists_squared = (TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)) ** 2
    dists = dists_squared
    dists_norm_squared = dists_squared.sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    losses = dists_squared[ids_row, assign, ids_col].mean(dim=(-1, -2))
    return losses


def compute_ADD_L1_loss(TCO_gt, TCO_pred, points):
    bsz = len(TCO_gt)
    assert TCO_pred.shape == (bsz, 4, 4) and TCO_gt.shape == (bsz, 4, 4)
    assert points.dim() == 3 and points.shape[-1] == 3
    dists = (
        (transform_pts(TCO_gt, points) - transform_pts(TCO_pred, points)).abs().mean(dim=(-1, -2))
    )
    return dists


def compute_ADD_symmetric_L1_loss(TXO_pred, TXO_gt, points):
    TXO_pred_points = transform_pts(TXO_pred, points)
    TXO_gt_points = transform_pts(TXO_gt, points)
    dists = TXO_gt_points.unsqueeze(1) - TXO_pred_points.unsqueeze(2)
    dists_norm_squared = dists.abs().sum(dim=-1)
    assign = dists_norm_squared.argmin(dim=1)
    ids_row = torch.arange(dists.shape[0]).unsqueeze(1).repeat(1, dists.shape[1])
    ids_col = torch.arange(dists.shape[1]).unsqueeze(0).repeat(dists.shape[0], 1)
    dists = dists[ids_row, assign, ids_col]
    return dists
