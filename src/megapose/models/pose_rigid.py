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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
from torch import nn

# MegaPose
from megapose.datasets.scene_dataset import Resolution
from megapose.lib3d.camera_geometry import boxes_from_uv, get_K_crop_resize
from megapose.lib3d.camera_geometry import (
    project_points_robust as project_points_robust,
)
from megapose.lib3d.cosypose_ops import pose_update_with_reference_point
from megapose.lib3d.cropping import deepim_crops_robust as deepim_crops_robust
from megapose.lib3d.multiview import make_TCO_multiview
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.lib3d.rotations import compute_rotation_matrix_from_ortho6d
from megapose.lib3d.transform_ops import normalize_T
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.panda3d_renderer.panda3d_scene_renderer import make_scene_lights
from megapose.training.utils import CudaTimer
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PosePredictorOutput:
    TCO_output: torch.Tensor
    TCO_input: torch.Tensor
    renders: torch.Tensor
    images_crop: torch.Tensor
    TCV_O_input: torch.Tensor
    KV_crop: torch.Tensor
    tCR: torch.Tensor
    labels: List[str]
    K: torch.Tensor
    K_crop: torch.Tensor
    network_outputs: Dict[str, torch.Tensor]
    boxes_rend: torch.Tensor
    boxes_crop: torch.Tensor
    renderings_logits: torch.Tensor
    timing_dict: Dict[str, float]


@dataclass
class PosePredictorDebugData:
    """Filled when debug=True."""

    output: Optional[PosePredictorOutput] = None
    images: Optional[torch.Tensor] = None
    origin_uv: Optional[torch.Tensor] = None
    ref_point_uv: Optional[torch.Tensor] = None
    origin_uv_crop: Optional[torch.Tensor] = None
    pose_predictor_outputs: Optional[torch.Tensor] = None


class PosePredictor(nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        renderer: Panda3dBatchRenderer,
        mesh_db: MeshDataBase,
        render_size: Resolution = (240, 320),
        multiview_type: str = "front_3views",
        views_inplane_rotations: bool = False,
        remove_TCO_rendering: bool = False,
        predict_pose_update: bool = True,
        predict_rendered_views_logits: bool = False,
        render_normals: bool = True,
        n_rendered_views: int = 1,
        input_depth: bool = False,
        render_depth: bool = False,
        depth_normalization_type: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.renderer = renderer
        self.render_size = render_size
        self.n_rendered_views = n_rendered_views
        self.input_depth = input_depth
        self.multiview_type = multiview_type
        self.views_inplane_rotations = views_inplane_rotations
        self.render_normals = render_normals
        self.render_depth = render_depth
        self.depth_normalization_type = depth_normalization_type
        self.predict_rendered_views_logits = predict_rendered_views_logits
        self.remove_TCO_rendering = remove_TCO_rendering
        self.predict_pose_update = predict_pose_update
        self.mesh_db = mesh_db

        n_features = backbone.n_features
        assert isinstance(n_features, int)

        # TODO: Change to torch ModuleDict
        self.heads: Dict[str, Union[torch.nn.Linear, Callable]] = dict()
        self.predict_pose_update = predict_pose_update
        if self.predict_pose_update:
            self._pose_dim = 9
            self.pose_fc = nn.Linear(n_features, self._pose_dim, bias=True)
            self.heads["pose"] = self.pose_fc

        self.predict_rendered_views_logits = predict_rendered_views_logits
        if self.predict_rendered_views_logits:
            self.views_logits_head = nn.Linear(n_features, self.n_rendered_views, bias=True)
            self.heads["renderings_logits"] = self.views_logits_head

        # Dimensions for indexing into input and rendered images
        self._input_rgb_dims = [0, 1, 2]
        if self.input_depth:
            self._input_depth_dims = [3]
        else:
            self._input_depth_dims = []

        self._render_rgb_dims = [0, 1, 2]
        if self.render_normals:
            self._render_normal_dims = [3, 4, 5]
        else:
            self._render_normal_dims = []

        if self.render_depth:
            start_dim = len(self._render_rgb_dims) + len(self._render_normal_dims)
            self._render_depth_dims = [start_dim]
        else:
            self._render_depth_dims = []

        # rgb only --> 3
        # rgb + normal --> 6
        # rgb + normal + depth --> 7
        self._n_single_render_channels = (
            len(self._render_rgb_dims)
            + len(self._render_normal_dims)
            + len(self._render_depth_dims)
        )

        self.debug = False
        self.timing_dict: Dict[str, float] = defaultdict(float)
        self.debug_data = PosePredictorDebugData()

    @property
    def input_rgb_dims(self) -> List[int]:
        return self._input_rgb_dims

    @property
    def input_depth_dims(self) -> List[int]:
        return self._input_depth_dims

    @property
    def render_rgb_dims(self) -> List[int]:
        return self._render_rgb_dims

    @property
    def render_depth_dims(self) -> List[int]:
        return self._render_depth_dims

    def crop_inputs(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        TCO: torch.Tensor,
        tCR: torch.Tensor,
        labels: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Crop input images.

        The image is cropped using the reprojection of the object points in the input pose (TCO).
        The reference point reprojects to the center of the cropped image.
        Please note that the unlike DeepIm, we do not explicitly use the input bounding
        box for cropping.

        Args:
            images (torch.Tensor): (bsz, ndims, h, w) where ndims is 3 or 4.
            K (torch.Tensor): (bsz, 3, 3), intrinsics of input images
            TCO (torch.Tensor): (bsz, 4, 4)
            tCR (torch.Tensor): (bsz, 3) Position of the reference point wrt camera.
            labels (List[str]): Object labels

        Returns:
            images_cropped: Images cropped and resized to self.render_size
            K_crop: Intrinsics of the fictive cropped camera.
            boxes_rend: smallest bounding box defined by the reprojection of object points in
                pose TCO.
            boxes_crop: bounding box used to crop the input image.
        """

        bsz = images.shape[0]
        assert K.shape == (bsz, 3, 3)
        assert tCR.shape == (bsz, 3)
        assert TCO.shape == (bsz, 4, 4)
        assert len(labels) == bsz
        meshes = self.mesh_db.select(labels)
        points = meshes.sample_points(2000, deterministic=True)

        uv = project_points_robust(points, K, TCO)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, images_cropped = deepim_crops_robust(
            images=images,
            obs_boxes=boxes_rend,
            K=K,
            TCO_pred=TCO,
            tCR_in=tCR,
            O_vertices=points,
            output_size=self.render_size,
            lamb=1.4,
        )

        K_crop = get_K_crop_resize(
            K=K.clone(), boxes=boxes_crop, orig_size=images.shape[-2:], crop_resize=self.render_size
        ).detach()

        if self.debug:
            TCR = TCO.clone()
            TCR[:, :3, -1] = tCR
            self.debug_data.ref_point_uv = project_points_robust(
                torch.zeros(bsz, 1, 3).to(K.device), K, TCR
            )
            self.debug_data.origin_uv = project_points_robust(
                torch.zeros(bsz, 1, 3).to(K.device), K, TCO
            )
            self.debug_data.origin_uv_crop = project_points_robust(
                torch.zeros(bsz, 1, 3).to(K.device), K_crop, TCO
            )
        return images_cropped, K_crop, boxes_rend, boxes_crop

    def compute_crops_multiview(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        TCV_O: torch.Tensor,
        tCR: torch.Tensor,
        labels: List[str],
    ) -> torch.Tensor:
        """Computes the intrinsics of the fictive camera used to
            render the additional viewpoints.

        Args:
            images (torch.Tensor): _description_
            K (torch.Tensor): _description_
            TCV_O (torch.Tensor): _description_
            tCR (torch.Tensor): _description_
            labels (List[str]): _description_

        Returns:
            K_crop
        """

        labels_mv = []
        bsz = len(labels)
        n_views = TCV_O.shape[1]
        assert tCR.shape == (bsz, n_views, 3)
        assert TCV_O.shape == (bsz, n_views, 4, 4)
        assert K.shape == (bsz, 3, 3)
        TCV_O = TCV_O.flatten(0, 1)
        tCR = tCR.flatten(0, 1)
        K = K.unsqueeze(1).repeat(1, n_views, 1, 1).flatten(0, 1)
        for n in range(bsz):
            for _ in range(n_views):
                labels_mv.append(labels[n])

        meshes = self.mesh_db.select(labels_mv)
        points = meshes.sample_points(200, deterministic=True)
        uv = project_points_robust(points, K, TCV_O)
        boxes_rend = boxes_from_uv(uv)
        boxes_crop, _ = deepim_crops_robust(
            images=images,
            obs_boxes=boxes_rend,
            K=K,
            TCO_pred=TCV_O,
            tCR_in=tCR,
            O_vertices=points,
            output_size=self.render_size,
            lamb=1.4,
            return_crops=False,
        )
        K_crop = get_K_crop_resize(
            K=K.clone(), boxes=boxes_crop, orig_size=images.shape[-2:], crop_resize=self.render_size
        )
        K_crop = K_crop.view(bsz, n_views, 3, 3)
        return K_crop

    def update_pose(
        self, TCO: torch.Tensor, K_crop: torch.Tensor, pose_outputs: torch.Tensor, tCR: torch.Tensor
    ) -> torch.Tensor:
        assert pose_outputs.shape[-1] == 9
        dR = compute_rotation_matrix_from_ortho6d(pose_outputs[:, 0:6])
        vxvyvz = pose_outputs[:, 6:9]
        TCO_updated = pose_update_with_reference_point(TCO, K_crop, vxvyvz, dR, tCR)
        return TCO_updated

    def net_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): input tensor (images + renderings)

        Returns:
            Dict[str, torch.Tensor]: Output of each network head.
        """
        x = self.backbone(x)
        if x.dim() == 2:
            pass
        elif x.dim() == 4:
            # Average pooling
            x = x.flatten(2).mean(dim=-1)
        else:
            raise ValueError
        outputs = dict()
        for k, head in self.heads.items():
            outputs[k] = head(x)
        return outputs

    def render_images_multiview(
        self,
        labels: List[str],
        TCV_O: torch.Tensor,
        KV: torch.Tensor,
        random_ambient_light: bool = False,
    ) -> torch.Tensor:
        """Render multiple images.

        Args:
            labels: list[str] with length bsz
            TCV_O: [bsz, n_views, 4, 4] pose of the cameras defining each view
            KV: [bsz, n_views, 4, 4] intrinsics of the associated cameras
            random_ambient_light: Whether to use randomize ambient light parameter.

        Returns
            renders: [bsz, n_views*n_channels, H, W]
        """

        labels_mv = []
        bsz = len(labels)
        n_views = TCV_O.shape[1]
        for n in range(bsz):
            for _ in range(n_views):
                labels_mv.append(labels[n])

        if random_ambient_light:
            light_datas = []
            for _ in range(len(labels_mv)):
                intensity = np.random.uniform(0.7, 1.0)
                lights = [
                    Panda3dLightData(
                        light_type="ambient",
                        color=(intensity, intensity, intensity, 1.0),
                    )
                ]
                light_datas.append(lights)
        else:
            if self.render_normals:
                ambient_light = Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
                light_datas = [[ambient_light] for _ in range(len(labels_mv))]
            else:
                light_datas = [make_scene_lights() for _ in range(len(labels_mv))]

        assert isinstance(self.renderer, Panda3dBatchRenderer)

        render_mask = False

        render_data = self.renderer.render(
            labels=labels_mv,
            TCO=TCV_O.flatten(0, 1),
            K=KV.flatten(0, 1),
            render_mask=render_mask,
            resolution=self.render_size,
            render_normals=self.render_normals,
            render_depth=self.render_depth,
            light_datas=light_datas,
        )

        cat_list = []
        cat_list.append(render_data.rgbs)

        if self.render_normals:
            cat_list.append(render_data.normals)

        if self.render_depth:
            cat_list.append(render_data.depths)

        renders = torch.cat(cat_list, dim=1)
        n_channels = renders.shape[1]

        renders = renders.view(bsz, n_views, n_channels, *renders.shape[-2:]).flatten(1, 2)
        return renders  # [bsz, n_views*n_channels, H, W]

    def normalize_images(
        self,
        images: torch.Tensor,
        renders: torch.Tensor,
        tCR: torch.Tensor,
        images_inplace: bool = False,
        renders_inplace: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize the depth images by the distance from the camera

        If we are using depth then this involves inplace ops so to be
        safe we will make copies of the tensors

        Args:
            images: [bsz, C, H, W]
            renders: [bsz, n_view*n_render_channels, H, W]
            tCR: [bsz, 3] anchor point for rendering
        """

        # NOTE (lmanuelli): Avoid errors with inplace ops as the same
        # input might be used in multiple iterations. Since we re-crop
        # on each iteration this might not be a problem but err'ing on
        # the side of caution
        if not images_inplace:
            images = images.clone()

        if not renders_inplace:
            renders = renders.clone()

        if self.input_depth:

            if not images_inplace:
                images = images.clone()
            C = images.shape[1]
            assert C == 4, "images must have C=4 channels if input_depth=True"

            # Add some trailing dimensions to make the broadcasting work
            # [B,1,1,1]
            depth = images[:, self._input_depth_dims]
            depth_norm = self.normalize_depth(depth, tCR)
            images[:, self._input_depth_dims] = depth_norm

        if self.render_depth:

            # Need to index into the right channels, assuming no normals
            # 1-view --> depth_dims = [3]
            # 2-view --> depth_dims = [3,7]
            depth_dims = self._render_depth_dims[0] + self._n_single_render_channels * torch.arange(
                0, self.n_rendered_views
            )

            depth = renders[:, depth_dims]
            renders[:, depth_dims] = self.normalize_depth(depth, tCR)

        return images, renders

    def normalize_depth(self, depth: torch.Tensor, tCR: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [B,-1,1,H,W]
            tCR: [B,3]

        Returns:
            depth_norm: same shape as depth
        """

        # [B,]
        z_norm = tCR[:, 2]

        # [B,-1,1,1,1], broadcastable to depth
        z_norm_unsqz = z_norm[(...,) + (None,) * (depth.ndim - 1)]
        if self.depth_normalization_type == "tCR_scale":
            depth_norm = depth / z_norm_unsqz
        elif self.depth_normalization_type == "tCR_scale_clamp_center":
            # values are always in [-1, 1]
            depth_norm = depth / z_norm_unsqz
            depth_norm = torch.clamp(depth_norm, 0, 2) - 1
        elif self.depth_normalization_type == "tCR_center_clamp":
            depth_norm = torch.clamp(depth - z_norm_unsqz, -2, 2)
        elif self.depth_normalization_type == "tCR_center_obj_diam":
            raise NotImplementedError("Not yet implemented")
        elif self.depth_normalization_type == "none":
            depth_norm = depth
        else:
            raise ValueError(f"Unknown depth_normalization_type = {self.depth_normalization_type}")

        return depth_norm

    def forward(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        labels: List[str],
        TCO: torch.Tensor,
        n_iterations: int = 1,
        random_ambient_light: bool = False,
    ) -> Dict[str, PosePredictorOutput]:

        timing_dict: Dict[str, float] = defaultdict(float)

        if not self.input_depth:
            # Remove the depth dimension if it is not used
            images = images[:, self.input_rgb_dims]

        bsz = images.shape[0]
        assert TCO.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)
        assert len(labels) == bsz
        dtype = TCO.dtype
        device = TCO.device

        outputs = dict()
        TCO_input = TCO
        for n in range(n_iterations):
            TCO_input = normalize_T(TCO_input).detach()

            # Anchor / reference point
            tOR = torch.zeros(bsz, 3, device=device, dtype=dtype)
            tCR = TCO_input[..., :3, [-1]] + TCO_input[..., :3, :3] @ tOR.unsqueeze(-1)
            tCR = tCR.squeeze(-1)

            TCV_O_input = make_TCO_multiview(
                TCO=TCO_input,
                tCR=tCR,
                multiview_type=self.multiview_type,
                n_views=self.n_rendered_views,
                remove_TCO_rendering=self.remove_TCO_rendering,
            )
            TCV_O_input_flatten = TCV_O_input.flatten(0, 1)

            n_views = TCV_O_input.shape[1]
            tCV_R = TCV_O_input_flatten[..., :3, [-1]] + TCV_O_input_flatten[
                ..., :3, :3
            ] @ tOR.unsqueeze(1).repeat(1, n_views, 1).flatten(0, 1).unsqueeze(-1)
            tCV_R = tCV_R.squeeze(-1).view(bsz, TCV_O_input.shape[1], 3)

            images_crop, K_crop, boxes_rend, boxes_crop = self.crop_inputs(
                images, K, TCO_input, tCR, labels
            )

            KV_crop = self.compute_crops_multiview(images, K, TCV_O_input, tCV_R, labels)
            if not self.remove_TCO_rendering:
                KV_crop[:, 0] = K_crop

            t = time.time()
            renders = self.render_images_multiview(
                labels, TCV_O_input, KV_crop, random_ambient_light=random_ambient_light
            )
            render_time = time.time() - t
            timing_dict["render"] = render_time

            # Need to normalize the depth in images/renders here
            images_crop, renders = self.normalize_images(
                images_crop,
                renders,
                tCR,
            )
            x = torch.cat((images_crop, renders), dim=1)

            # would expect this to error out
            network_outputs = self.net_forward(x)
            if self.predict_pose_update:
                TCO_output = self.update_pose(TCO_input, K_crop, network_outputs["pose"], tCR)
            else:
                TCO_output = TCO_input.detach().clone()

            if self.predict_rendered_views_logits:
                renderings_logits = network_outputs["renderings_logits"]
                assert not self.predict_pose_update
            else:
                renderings_logits = torch.empty(
                    bsz, self.n_rendered_views, dtype=dtype, device=device
                )

            outputs[f"iteration={n+1}"] = PosePredictorOutput(
                renders=renders,
                images_crop=images_crop,
                TCO_input=TCO_input,
                TCO_output=TCO_output,
                TCV_O_input=TCV_O_input,
                tCR=tCR,
                labels=labels,
                K=K,
                K_crop=K_crop,
                KV_crop=KV_crop,
                network_outputs=network_outputs,
                boxes_rend=boxes_rend,
                boxes_crop=boxes_crop,
                renderings_logits=renderings_logits,
                timing_dict=timing_dict,
            )
            if self.debug:
                self.debug_data.output = outputs[f"iteration={n+1}"]
            TCO_input = TCO_output
        return outputs

    def forward_coarse_tensor(
        self, x: torch.Tensor, cuda_timer: bool = False
    ) -> Dict[str, Union[torch.Tensor, float]]:

        """Forward pass on coarse model given an input tensor.

        The input already contains the concatenated input + rendered images and has
        been appropriately normalized.

        Args:
            x: [B,C,H,W] where C=9 typically. This is the concatenated
                input + rendered image

        """
        assert (
            self.predict_rendered_views_logits
        ), "Method only valid if coarse classification model"

        timer = CudaTimer(enabled=cuda_timer)
        timer.start()

        logits = self.net_forward(x)["renderings_logits"]
        scores = torch.sigmoid(logits)

        timer.end()

        return {"logits": logits, "scores": scores, "time": timer.elapsed()}

    def forward_coarse(
        self,
        images: torch.Tensor,
        K: torch.Tensor,
        labels: List[str],
        TCO_input: torch.Tensor,
        cuda_timer: bool = False,
        return_debug_data: bool = False,
    ) -> Dict[str, Any]:
        # TODO: Is this still necessary ?
        """Run the coarse model given images + poses

        Only valid if we are using the coarse model.


        Args:
            images: [B,C,H,W] torch tensor, should already be normalized to
                [0,255] --> [0,1]
            K: [B,3,3] camera intrinsics
            labels: list(str) of len(B)
            TCO: [B,4,4] object poses


        Returns:
            dict:
                - logits: tensor [B,]
                - scores tensor [B,]


        """

        assert (
            self.predict_rendered_views_logits
        ), "Method only valid if coarse classification model"

        if not self.input_depth:
            # Remove the depth dimension if it is not used
            images = images[:, self.input_rgb_dims]

        bsz, nchannels, h, w = images.shape
        assert TCO_input.shape == (bsz, 4, 4)
        assert K.shape == (bsz, 3, 3)
        assert len(labels) == bsz

        TCO_input = normalize_T(TCO_input).detach()
        tCR = TCO_input[..., :3, -1]
        images_crop, K_crop, boxes_rend, boxes_crop = self.crop_inputs(
            images, K, TCO_input, tCR, labels
        )

        # [B,1,4,4], hack to use the multi-view function
        TCO_V_input = TCO_input.unsqueeze(1)
        KV_crop = K_crop.unsqueeze(1)

        render_start = time.time()
        renders = self.render_images_multiview(
            labels,
            TCO_V_input,
            KV_crop,
        )
        render_time = time.time() - render_start

        images_crop, renders = self.normalize_images(images_crop, renders, tCR)
        x = torch.cat((images_crop, renders), dim=1)

        out = self.forward_coarse_tensor(x, cuda_timer=cuda_timer)

        out["render_time"] = render_time
        out["model_time"] = out["time"]

        if return_debug_data:
            out["images_crop"] = images_crop
            out["renders"] = renders

        return out
