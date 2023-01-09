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
import numpy as np
import open3d as o3d
import transforms3d as t3d


def numpy_to_open3d(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def compute_masks(mask_type, depth_rendered, depth_measured, depth_delta_thresh=0.1):
    """
    Function for computing masks

    Args:
        mask_type: str
        depth_rendered: [H,W]
        depth_measured: [H,W]
        depth_delta_thresh: 0.1

    """

    mask_rendered = depth_rendered > 0
    mask_measured = np.logical_and(depth_measured > 0, depth_rendered > 0)

    if mask_type == "simple":
        pass
    elif mask_type == "threshold":
        depth_delta = np.abs(depth_measured - depth_rendered)
        mask_measured[depth_delta > depth_delta_thresh] = 0
    else:
        raise ValueError(f"Unknown mask type {mask_type}")

    # Most conservative
    mask_rendered = mask_measured

    return mask_rendered, mask_measured
