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


from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Tuple

# Third Party
import numpy as np
import torch

# MegaPose
from megapose.utils.tensor_collection import PandasTensorCollection

# Type Aliases

"""
infos: pd.DataFrame with (at least) fields
    - 'label': str, the object label
    - 'batch_im_id': int, image in batch corresponding to this pose
        estimate
    - 'instance_id': int, instance_id of this detection. This
        serves to identify which instance in the scene it belongs to
        in case of multiple instances of a specific object in a scene.
    - 'hypothesis_id': int, the hypothesis of this estimate. We can operate
        in a mode where we refine multiple hypotheses for one detection.
    - 'pose_score': Optional[float], the score as evaluated by the coarse
        scoring module
    - 'coarse_logit' Optional[float], s
    - 'refiner_batch_idx', Optional[int], used to index into the
        correct batch for the refiner outputs
    - 'refiner_instance_idx', Optional[int], used to index into
        refiner outputs such as "image_crop", "render_crop", etc.
    - 'scene_id', Optional[str] used to identify predictions on a dataset.
    - 'view_id', Optional[str] used to identify predictions on a dataset.
Tensors:
    - poses: [B,4,4] float32
"""
PoseEstimatesType = PandasTensorCollection


"""
infos: pd.DataFrame with fields
    - 'label': str, the object label
    - 'batch_im_id': int, image in batch corresponding to this pose
        estimate
    - 'instance_id': int, instance_id of this detection. This
        serves to identify which instance in the scene it belongs to
        in case of multiple instances of a specific object in a scene.
    - 'score': Optional[float], the detection score
    - 'scene_id', Optional[str] used to identify predictions on a dataset.
    - 'view_id', Optional[str] used to identify predictions on a dataset.

tensors:
    - bboxes: [B,4] int

"""
DetectionsType = PandasTensorCollection


def assert_detections_valid(detections: DetectionsType) -> None:
    """Checks if detections contains the required fields."""
    df = detections.infos

    fields = ["batch_im_id", "label", "instance_id"]

    for f in fields:
        assert f in df, f"detections.infos missing column {f}"

    assert "bboxes" in detections.tensors, "detections missing tensor bboxes."


@dataclass
class InferenceConfig:
    # TODO: move detection_type outside of here
    detection_type: str = "detector"  # ['detector', 'gt']
    coarse_estimation_type: str = "SO3_grid"
    SO3_grid_size: int = 576
    n_refiner_iterations: int = 5
    n_pose_hypotheses: int = 5
    run_depth_refiner: bool = False
    depth_refiner: Optional[str] = None  # ['icp', 'teaserpp']
    bsz_objects: int = 16  # How many parallel refiners to run
    bsz_images: int = 576  # How many images to push through coarse model


@dataclass
class ObservationTensor:
    """

    images: [B,C,H,W] with C=3 (rgb) or C=4 (rgbd). RGB dimensions should already
        be normalized to be in [0,1] by diving the uint8 values by 255

    K: [B,3,3] camera intrinsics
    """

    images: torch.Tensor  # [B,C,H,W]
    K: Optional[torch.Tensor] = None  # [B,3,3]

    def cuda(self) -> ObservationTensor:
        self.images = self.images.cuda()
        if self.K is not None:
            self.K = self.K.cuda()
        return self

    @property
    def batch_size(self) -> int:
        """Returns the batch size."""

        return self.images.shape[0]

    @property
    def depth(self) -> torch.tensor:
        """Returns depth tensor.

        Returns:
            torch.tensor with shape [B,H,W]
        """
        assert self.channel_dim == 4
        return self.images[:, 3]

    @property
    def channel_dim(self) -> int:
        """Returns the channel size."""
        return self.images.shape[1]

    def is_valid(self) -> bool:

        if not self.images.ndim == 4:
            return False

        B = self.batch_size
        C = self.channel_dim

        if C not in [3, 4]:
            return False

        if self.K is not None:
            if not self.K.shape == torch.Size([B, 3, 3]):
                return False

        if not self.images.dtype == torch.float:
            return False

        # rgb values should be already be converted to [0,1] as floats
        # rather than being in [0,255]
        rgb_max = torch.max(self.images[:, :3])
        if rgb_max > 1:
            return False

        return True

    @staticmethod
    def from_numpy(
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
    ) -> ObservationTensor:
        """Create an ObservationData type from numpy data.

        Args:
            rgb: [H,W,3] np.uint8
            depth: [H,W] np.float
            K: [3,3] np.float

        """

        assert rgb.dtype == np.uint8
        rgb_tensor = torch.as_tensor(rgb).float() / 255

        # convert it to [C,H,W]
        if rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)

        # [C,H,W]
        if depth is not None:
            depth_tensor = torch.as_tensor(depth).unsqueeze(0)
            img_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)
        else:
            img_tensor = rgb_tensor

        # img_tensor is [C,H,W] where C=3 (rgb) or C=4 (rgbd)
        K_tensor = torch.as_tensor(K).float()
        return ObservationTensor(img_tensor.unsqueeze(0), K_tensor.unsqueeze(0))

    @staticmethod
    def from_torch_batched(
        rgb: torch.Tensor, depth: torch.Tensor, K: torch.Tensor
    ) -> ObservationTensor:
        """

        Args:
            rgb: [B,3,H,W] torch.uint8
            depth: [B,1,H,W] torch.float
            K: [B,3,3] torch.float

        """

        assert rgb.dtype == torch.uint8

        # [B,3,H,W]
        rgb = torch.as_tensor(rgb).float() / 255

        B, _, H, W = rgb.shape

        # [C,H,W]
        if depth is not None:

            if depth.ndim == 3:
                depth.unsqueeze(1)

            # Now depth is [B,1,H,W]
            img_tensor = torch.cat((rgb, depth), dim=1)
        else:
            img_tensor = rgb

        # img_tensor is [C,H,W] where C=3 (rgb) or C=4 (rgbd)
        K_tensor = torch.as_tensor(K).float()
        return ObservationTensor(img_tensor, K_tensor)
