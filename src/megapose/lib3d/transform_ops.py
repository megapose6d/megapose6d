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



# Standard Library
from typing import Tuple

# Third Party
import numpy as np
import torch
import transforms3d

# Local Folder
from .rotations import compute_rotation_matrix_from_ortho6d


def transform_pts(T: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """

    Args:
        T (torch.Tensor): (bsz, 4, 4) or (bsz, dim2, 4, 4)
        pts (torch.Tensor): (bsz, n_pts, 3)

    Raises:
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    bsz = T.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bsz, n_pts, 3)
    if T.dim() == 4:
        pts = pts.unsqueeze(1)
        assert T.shape[-2:] == (4, 4)
    elif T.dim() == 3:
        assert T.shape == (bsz, 4, 4)
    else:
        raise ValueError("Unsupported shape for T", T.shape)
    pts = pts.unsqueeze(-1)
    T = T.unsqueeze(-3)
    pts_transformed = T[..., :3, :3] @ pts + T[..., :3, [-1]]
    return pts_transformed.squeeze(-1)


def invert_transform_matrices(T: torch.Tensor) -> torch.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, [-1]]
    R_inv = R.transpose(-2, -1)
    t_inv = -R_inv @ t
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_inv.to(T_inv.dtype)
    T_inv[..., :3, [-1]] = t_inv.to(T_inv.dtype)
    return T_inv


def add_noise(
    TCO: torch.Tensor,
    euler_deg_std: Tuple[float, float, float] = (15, 15, 15),
    trans_std: Tuple[float, float, float] = (0.01, 0.01, 0.05),
) -> torch.Tensor:
    TCO_out = TCO.clone()
    device = TCO_out.device
    bsz = TCO.shape[0]
    euler_noise_deg = np.concatenate(
        [
            np.random.normal(loc=0, scale=euler_deg_std_i, size=bsz)[:, None]
            for euler_deg_std_i in euler_deg_std
        ],
        axis=1,
    )
    euler_noise_rad = euler_noise_deg * np.pi / 180
    R_noise = (
        torch.tensor(np.stack([transforms3d.euler.euler2mat(*xyz) for xyz in euler_noise_rad]))
        .float()
        .to(device)
    )

    trans_noise = np.concatenate(
        [
            np.random.normal(loc=0, scale=trans_std_i, size=bsz)[:, None]
            for trans_std_i in trans_std
        ],
        axis=1,
    )
    trans_noise = torch.tensor(trans_noise).float().to(device)
    TCO_out[:, :3, :3] = TCO_out[:, :3, :3] @ R_noise
    TCO_out[:, :3, 3] += trans_noise
    return TCO_out


def compute_transform_from_pose9d(pose9d: torch.Tensor) -> torch.Tensor:
    assert pose9d.shape[-1] == 9
    R = compute_rotation_matrix_from_ortho6d(pose9d[..., :6])
    trans = pose9d[..., 6:]
    T = torch.zeros(*pose9d.shape[:-1], 4, 4, dtype=pose9d.dtype, device=pose9d.device)
    T[..., 0:3, 0:3] = R
    T[..., 0:3, 3] = trans
    T[..., 3, 3] = 1
    return T


def normalize_T(T: torch.Tensor) -> torch.Tensor:
    pose_9d = torch.cat([T[..., :3, 0], T[..., :3, 1], T[..., :3, -1]], dim=-1)
    return compute_transform_from_pose9d(pose_9d)
