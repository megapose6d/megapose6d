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
import roma
import torch

# MegaPose
import megapose
from megapose.config import PROJECT_DIR


def load_SO3_grid(resolution):
    """
    The data.qua files were generated with the following code
    http://lavalle.pl/software/so3/so3.html

    They are in (x,y,z,w) ordering

    Returns:
        rotmats: [N,3,3]
    """
    data_fname = PROJECT_DIR / f"src/megapose/data/data_{resolution}.qua"

    assert data_fname.is_file(), f"File {data_fname} not found"

    quats = []
    with open(data_fname) as fp:
        lines = fp.readlines()
        for line in lines:
            x, y, z, w = [float(i) for i in line.split()]
            quats.append([x, y, z, w])

    quats = torch.tensor(quats)
    rotmats = roma.unitquat_to_rotmat(quats)
    return rotmats


def compute_geodesic_distance(query, target):
    """

    Computes distance, in radians from query to target
    Args:
        query: [N,3,3]
        target: [M,3,3]
    """

    N = query.shape[0]
    M = target.shape[0]
    query_exp = query.unsqueeze(1).expand([-1, M, -1, -1]).flatten(0, 1)
    target_exp = target.unsqueeze(0).expand([N, -1, -1, -1]).flatten(0, 1)

    # [N*M]

    dist = roma.rotmat_geodesic_distance(query_exp, target_exp)
    dist = dist.reshape([N, M])

    min_val, min_idx = torch.min(dist, dim=-1)

    return {
        "distance": min_val,
        "rotmat": target[min_idx],
        "distance_all": dist,
    }
