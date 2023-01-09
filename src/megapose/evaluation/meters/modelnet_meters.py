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
import torch
import xarray as xr

# MegaPose
from megapose.lib3d.camera_geometry import project_points
from megapose.lib3d.distances import dists_add
from megapose.lib3d.transform import Transform
from megapose.lib3d.transform_ops import transform_pts

# Local Folder
from .base import Meter
from .lf_utils import angular_distance
from .utils import one_to_one_matching


class ModelNetErrorMeter(Meter):
    def __init__(self, mesh_db, sample_n_points=None):
        self.reset()
        self.mesh_db = mesh_db.batched(resample_n_points=sample_n_points).cuda().float()

    def is_data_valid(self, data):
        valid = False
        if not valid and hasattr(data, "K"):
            valid = True
        return valid

    def add(self, pred_data, gt_data):
        pred_data = pred_data.float()
        gt_data = gt_data.float()

        matches = one_to_one_matching(
            pred_data.infos, gt_data.infos, keys=("scene_id", "view_id"), allow_pred_missing=False
        )

        pred_data = pred_data[matches.pred_id]
        gt_data = gt_data[matches.gt_id]

        TXO_gt = gt_data.poses[0]
        TXO_pred = pred_data.poses[0]

        trans_dist = torch.norm((TXO_gt[:3, -1] - TXO_pred[:3, -1]), dim=-1)
        quat_gt = torch.tensor(Transform(TXO_gt.cpu().numpy()).quaternion.coeffs())
        quat_pred = torch.tensor(Transform(TXO_pred.cpu().numpy()).quaternion.coeffs())
        angular_dist = angular_distance(quat_gt, quat_pred) * 180 / np.pi

        labels = [pred_data.infos["label"].item()]
        n_points = self.mesh_db.infos[labels[0]]["n_points"]
        meshes = self.mesh_db.select(labels)
        points = meshes.points[:, :n_points]
        dist = dists_add(TXO_pred.unsqueeze(0), TXO_gt.unsqueeze(0), points)[0]
        dist_add = torch.norm(dist, dim=-1).mean(dim=-1)

        extent = (points.max(1)[0] - points.min(1)[0]).cpu().numpy()[0]
        diameter_1 = np.linalg.norm(extent).item()

        K = gt_data.K[0]
        uv_pred = project_points(points, K.unsqueeze(0), TXO_pred.unsqueeze(0))[0]
        uv_gt = project_points(points, K.unsqueeze(0), TXO_gt.unsqueeze(0))[0]
        uv_dists = torch.norm(uv_pred - uv_gt, dim=-1)
        uv_avg = uv_dists.mean()

        df = xr.Dataset(matches).rename(dict(dim_0="match_id"))
        df["add"] = "match_id", np.array([dist_add.item()])
        df["diameter"] = "match_id", np.array([diameter_1])
        df["proj_error"] = "match_id", np.array([uv_avg.item()])
        df["angular_dist"] = "match_id", np.array([angular_dist.item()])
        df["trans_dist"] = "match_id", np.array([trans_dist.item()])
        self.datas["df"].append(df)

    def summary(self):
        df = xr.concat(self.datas["df"], dim="match_id")
        add = df["add"].values < 0.1 * df["diameter"].values
        trans_dist = df["trans_dist"].values
        angular_dist = df["angular_dist"].values
        proj_dist = df["proj_error"].values
        proj_2d = proj_dist < 5
        rot_trans = np.logical_and(trans_dist < 0.05, angular_dist < 5)

        summary = {
            "add0.1d": add.mean(),
            "5deg_5cm": rot_trans.mean(),
            "proj2d_5px": proj_2d.mean(),
        }
        return summary, df
