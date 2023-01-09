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
import time
from typing import Optional, Tuple

# Third Party
import numpy as np
import numpy.typing as npt
import teaserpp_python
import torch

# MegaPose
from megapose.inference.depth_refiner import DepthRefiner
from megapose.inference.refiner_utils import compute_masks, numpy_to_open3d
from megapose.inference.types import PoseEstimatesType
from megapose.lib3d.rigid_mesh_database import BatchedMeshes
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.panda3d_renderer.types import Panda3dLightData
from megapose.visualization.meshcat_utils import get_pointcloud


def get_solver_params(noise_bound: float = 0.01):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    return solver_params


def compute_teaserpp_refinement(
    depth_src: npt.ArrayLike,
    depth_tgt: npt.ArrayLike,
    cam_K: npt.ArrayLike,
    mask: npt.ArrayLike,
    solver_params=None,
    max_num_points=None,
    normals_src=None,
    use_farthest_point_sampling: bool = True,
    **solver_params_kwargs
) -> dict:
    """Compute registration using Teaser++

    Follows the example of https://github.com/MIT-SPARK/TEASER-plusplus#minimal-python-3-example

    Args:
        depth_src: [H,W,3]
        depth_tgt: [H,W, 3]
        cam_K: [3,3] intrinsics matrix
        mask: [M,] mask to apply to src/tgt pointclouds
        max_num_points: N, number of points to downsample to.
        normals_src: (optional) normals for the src pointcloud

    Returns:
        A dict.

        - 'T_tgt_src': The rigid transform that aligns src to tgt.

    """

    if solver_params is None:
        solver_params = get_solver_params(**solver_params_kwargs)

    pc_src_raw = get_pointcloud(depth_src, cam_K)
    pc_tgt_raw = get_pointcloud(depth_tgt, cam_K)

    pc_src_mask = pc_src_raw[mask]
    pc_tgt_mask = pc_tgt_raw[mask]

    if max_num_points is not None:
        N = pc_src_mask.shape[0]
        M = min(max_num_points, N)

        if normals_src is None:
            x = torch.tensor(pc_src_mask).cuda()
        else:
            a = torch.tensor(pc_src_mask).cuda()
            b = torch.tensor(normals_src[mask]).cuda()
            x = torch.cat((a, b), dim=-1)

        # Do furthest point sampling with pytorch3d
        # [1, N, 3]
        x = x.unsqueeze(0)
        lengths = torch.tensor([x.shape[1]]).cuda()
        K = max_num_points

        idx = None
        if use_farthest_point_sampling:
            # Third Party
            import pytorch3d.ops

            _, idx = pytorch3d.ops.sample_farthest_points(x, lengths, K)

            # [M, 3]
            idx = idx.squeeze().cpu().numpy()
        else:
            idx = np.random.choice(np.arange(N), size=M, replace=False)

        # Random sampling

        pc_src = pc_src_mask[idx]
        pc_tgt = pc_tgt_mask[idx]
    else:
        pc_src = pc_src_mask
        pc_tgt = pc_src_mask

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()

    # teaserpp wants [3,N] pointclouds
    solver.solve(pc_src.transpose(), pc_tgt.transpose())
    end = time.time()

    solution = solver.getSolution()

    T = np.eye(4)
    T[:3, :3] = solution.rotation
    T[:3, 3] = solution.translation
    T_tgt_src = T

    # check number of inliers
    pc_src_o3d = numpy_to_open3d(pc_src)
    pc_src_o3d_refined = pc_src_o3d.transform(T_tgt_src)
    pc_src_refined = np.array(pc_src_o3d_refined.points)

    diff = np.linalg.norm(pc_src_refined - pc_tgt, axis=-1)
    inliers = np.count_nonzero(diff < solver_params.noise_bound)

    return {
        "solution": solution,
        "pc_src_raw": pc_src_raw,
        "pc_tgt_raw": pc_tgt_raw,
        "pc_src": pc_src,
        "pc_tgt": pc_tgt,
        "pc_src_mask": pc_src_mask,
        "pc_tgt_mask": pc_tgt_mask,
        "T": T,
        "T_tgt_src": T,  # transform that aligns src to tgt
        "num_inliers": inliers,
    }


class TeaserppRefiner(DepthRefiner):
    def __init__(
        self,
        mesh_db: BatchedMeshes,
        renderer: Panda3dBatchRenderer,
        mask_type: str = "simple",
        depth_delta_thresh: float = 0.1,
        n_min_points: int = 100,
        n_points: int = 1000,
        noise_bound: float = 0.01,
        min_num_inliers: int = 50,
        use_farthest_point_sampling: bool = True,
    ) -> None:
        self.mesh_db = mesh_db
        self.renderer = renderer
        self.mask_type = mask_type
        self.depth_delta_thresh = depth_delta_thresh
        self.n_min_points = n_min_points
        self.n_points = n_points
        self.noise_bound = noise_bound
        self.min_num_inliers = min_num_inliers
        self.use_farthest_point_sampling = use_farthest_point_sampling

        # default light_datas for rendering
        self.light_datas = [Panda3dLightData("ambient")]

        self.debug = {}

    def refine_poses(
        self,
        predictions: PoseEstimatesType,
        masks: Optional[torch.tensor] = None,
        depth: Optional[torch.tensor] = None,
        K: Optional[torch.tensor] = None,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs Teaserpp refiner. See superclass DepthRefiner for full documentation.

        To generate correspondences for Teaser++ we use the following approach.
        1. Render depth image depth_rendered at the estimated pose from predictions.
        2. Generate 3D --> 3D correspondences across rendered and observed depth images.
            by assuming that pose is correctly aligned in rgb space. So depth_rendered[u,v]
            corresponds to depth_observed[u,v].
        3. Estimate a mask to filter out some outliers in our generated correspondences.

        Args:
            predictions: PandasTensorCollection
                Index into depth, K with batch_im_id
            depth: [B, H, W]
            masks: not needed, for backward compatibility
            K: [B,3,3]

        """

        assert depth is not None
        assert K is not None

        predictions_refined = predictions.clone()
        resolution = depth.shape[-2:]

        df = predictions.infos
        labels = df.label.tolist()
        batch_im_ids = df.batch_im_id.tolist()

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

            mask_rendered, mask_measured = compute_masks(
                self.mask_type,
                depth_rendered=depth_rendered,
                depth_measured=depth_measured,
                depth_delta_thresh=self.depth_delta_thresh,
            )

            # If insufficient number of points then just use the current prediction
            if (np.count_nonzero(mask_rendered) < self.n_min_points) or (
                np.count_nonzero(mask_measured) < self.n_min_points
            ):
                continue
            else:
                out = compute_teaserpp_refinement(
                    depth_src=depth_rendered,
                    depth_tgt=depth_measured,
                    mask=mask_measured,
                    cam_K=cam_K,
                    max_num_points=self.n_points,
                    noise_bound=self.noise_bound,
                    use_farthest_point_sampling=self.use_farthest_point_sampling,
                )

                # Only update the pose if we are "confident" about the solution,
                # i.e. the num of inliers is above a threshold
                if out["num_inliers"] >= self.min_num_inliers:
                    # T_measured_rendered
                    T_tgt_src = out["T_tgt_src"]
                    TCO_refined = T_tgt_src @ TCO_pred
                    device = predictions_refined.poses_input[n].device
                    predictions_refined.poses_input[n] = predictions.poses[n].clone()
                    predictions_refined.poses[n] = torch.tensor(TCO_refined, device=device)

                self.debug = out

        extra_data = self.debug
        return predictions_refined, extra_data
