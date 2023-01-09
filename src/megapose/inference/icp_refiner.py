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
from typing import List, Optional, Tuple

# Third Party
import cv2
import numpy as np
import torch
from scipy import ndimage

# MegaPose
from megapose.config import DEBUG_DATA_DIR
from megapose.inference.depth_refiner import DepthRefiner
from megapose.inference.refiner_utils import compute_masks
from megapose.inference.types import PoseEstimatesType
from megapose.lib3d.rigid_mesh_database import BatchedMeshes
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.panda3d_renderer.types import Panda3dLightData


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, bbox=np.array([0]), refine=True):
    # Copied from https://github.com/kirumang/Pix2Pose/blob/master/pix2pose_util/common_util.py
    """
    fast normal computation
    """
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]
    centerX = cx
    centerY = cy
    constant_x = 1 / fx
    constant_y = 1 / fy

    if refine:
        depth_refine = np.nan_to_num(depth_refine)
        mask = np.zeros_like(depth_refine).astype(np.uint8)
        mask[depth_refine == 0] = 1
        depth_refine = depth_refine.astype(np.float32)
        depth_refine = cv2.inpaint(depth_refine, mask, 2, cv2.INPAINT_NS)
        depth_refine = depth_refine.astype(np.float32)
        depth_refine = ndimage.gaussian_filter(depth_refine, 2)

    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)
    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)

    if bbox.shape[0] == 4:
        uv_table = uv_table[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        v_x = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
        v_y = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
        normals = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
        depth_refine = depth_refine[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    else:
        v_x = np.zeros((res_y, res_x, 3))
        v_y = np.zeros((res_y, res_x, 3))
        normals = np.zeros((res_y, res_x, 3))

    uv_table_sign = np.copy(uv_table)
    uv_table = np.abs(np.copy(uv_table))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant_x * dig[0]
    v_y[:, :, 1] = depth_refine * constant_y + (uv_table_sign[:, :, 0] * constant_y) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant_x + uv_table_sign[:, :, 1] * constant_x * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant_y * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    norm[norm == 0] = 1
    cross = cross / norm
    if bbox.shape[0] == 4:
        cross = cross.reshape((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))
    else:
        cross = cross.reshape(res_y, res_x, 3)
    cross = np.nan_to_num(cross)
    return cross


def getXYZ(depth, fx, fy, cx, cy, bbox=np.array([0])):
    # Copied from https://github.com/kirumang/Pix2Pose/blob/master/pix2pose_util/common_util.py
    uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
    column = np.arange(0, depth.shape[0])
    uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - cx  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - cy  # y-c_y (v)

    if bbox.shape[0] == 1:
        xyz = np.zeros((depth.shape[0], depth.shape[1], 3))  # x,y,z
        xyz[:, :, 0] = uv_table[:, :, 1] * depth * 1 / fx
        xyz[:, :, 1] = uv_table[:, :, 0] * depth * 1 / fy
        xyz[:, :, 2] = depth
    else:  # when boundry region is given
        xyz = np.zeros((bbox[2] - bbox[0], bbox[3] - bbox[1], 3))  # x,y,z
        xyz[:, :, 0] = (
            uv_table[bbox[0] : bbox[2], bbox[1] : bbox[3], 1]
            * depth[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            * 1
            / fx
        )
        xyz[:, :, 1] = (
            uv_table[bbox[0] : bbox[2], bbox[1] : bbox[3], 0]
            * depth[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            * 1
            / fy
        )
        xyz[:, :, 2] = depth[bbox[0] : bbox[2], bbox[1] : bbox[3]]
    return xyz


def icp_refinement(
    depth_measured, depth_rendered, object_mask_measured, cam_K, TCO_pred, n_min_points=1000
):
    # Inspired from https://github.com/kirumang/Pix2Pose/blob/843effe0097e9982f4b07dd90b04ede2b9ee9294/tools/5_evaluation_bop_icp3d.py#L57

    points_tgt = np.zeros((depth_measured.shape[0], depth_measured.shape[1], 6), np.float32)
    points_tgt[:, :, :3] = getXYZ(
        depth_measured, fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2]
    )
    points_tgt[:, :, 3:] = get_normal(
        depth_measured, fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], refine=True
    )
    depth_valid = np.logical_and(depth_measured > 0.2, depth_measured < 5)
    depth_valid = np.logical_and(depth_valid, object_mask_measured)
    points_tgt = points_tgt[depth_valid]

    points_src = np.zeros((depth_measured.shape[0], depth_measured.shape[1], 6), np.float32)
    points_src[:, :, :3] = getXYZ(
        depth_rendered, cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]
    )
    points_src[:, :, 3:] = get_normal(
        depth_rendered, fx=cam_K[0, 0], fy=cam_K[1, 1], cx=cam_K[0, 2], cy=cam_K[1, 2], refine=True
    )
    points_src = points_src[np.logical_and(depth_valid, depth_rendered > 0)]

    if len(points_tgt) < n_min_points or len(points_src) < n_min_points:
        return np.eye(4) * float("nan"), -1

    TCO_pred_refined = TCO_pred.copy()

    # adjust the initial translation using centroids of visible points
    centroid_src = np.mean(points_src[:, :3], axis=0)
    centroid_tgt = np.mean(points_tgt[:, :3], axis=0)
    dists_centroid = centroid_tgt - centroid_src
    TCO_pred_refined[:3, -1] += dists_centroid.reshape(-1)
    points_src[:, :3] += dists_centroid[None]

    tolerence = 0.05
    icp_fnc = cv2.ppf_match_3d_ICP(100, tolerence=tolerence, numLevels=4)
    retval, residual, pose = icp_fnc.registerModelToScene(
        points_src.reshape(-1, 6), points_tgt.reshape(-1, 6)
    )
    TCO_pred_refined = pose @ TCO_pred_refined
    TCO_pred_refined = torch.tensor(TCO_pred_refined, dtype=torch.float32).cuda()

    if residual > tolerence or residual < 0:
        retval = -1
    return TCO_pred_refined, retval


def icp_refinement_multiproc(queue, idx, args):
    out = icp_refinement(args)
    queue.put({"idx": idx, "out": out})


class ICPRefiner(DepthRefiner):
    def __init__(
        self,
        mesh_db: BatchedMeshes,
        renderer: Panda3dBatchRenderer,
    ) -> None:
        self.mesh_db = mesh_db
        self.renderer = renderer

        # default light_datas for rendering
        self.light_datas = [Panda3dLightData("ambient")]

    def refine_poses(
        self,
        predictions: PoseEstimatesType,
        masks: Optional[torch.tensor] = None,
        depth: Optional[torch.tensor] = None,
        K: Optional[torch.tensor] = None,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs icp refinement. See superclass DepthRefiner for full documentation."""

        assert depth is not None
        assert K is not None

        predictions_refined = predictions.clone()
        resolution = depth.shape[-2:]

        df = predictions.infos
        labels = df.label.tolist()
        batch_im_ids = df.batch_im_id.tolist()

        # N = len(predictions)
        N = len(predictions)
        TCO_ = predictions.poses  # [N,4,4]
        K_ = K[batch_im_ids]  # [N,4,4]

        render_output = self.renderer.render(
            labels,
            TCO=TCO_,
            K=K_,
            light_datas=[self.light_datas] * N,
            resolution=resolution,
            render_depth=True,
        )

        # [N,H,W]
        all_depth_rendered = render_output.depths

        for n in range(len(predictions)):
            view_id = predictions.infos.loc[n, "batch_im_id"]
            TCO_pred = predictions.poses[n].cpu().numpy()

            # [H,W]
            depth_measured = depth[view_id].squeeze().cpu().numpy()
            cam_K = K_[n].cpu().numpy()
            depth_rendered = all_depth_rendered[n].squeeze().cpu().numpy()

            if masks is None:
                mask_rendered, mask_measured = compute_masks(
                    mask_type="threshold",
                    depth_rendered=depth_rendered,
                    depth_measured=depth_measured,
                    depth_delta_thresh=0.1,
                )

                mask = mask_measured
            else:
                mask = masks[view_id].squeeze().cpu().numpy()

            TCO_refined, retval = icp_refinement(
                depth_measured, depth_rendered, mask, cam_K, TCO_pred, n_min_points=1000
            )

            # Assign poses to predictions refined
            predictions_refined.poses_input[n] = predictions.poses[n].clone()
            if retval != -1:
                predictions_refined.poses[n] = TCO_refined

        extra_data = dict()
        return (predictions_refined, extra_data)
