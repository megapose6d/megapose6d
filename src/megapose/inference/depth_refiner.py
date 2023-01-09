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
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# Third Party
import torch

# MegaPose
from megapose.inference.types import PoseEstimatesType


class DepthRefiner(ABC):
    @abstractmethod
    def refine_poses(
        self,
        predictions: PoseEstimatesType,
        masks: Optional[torch.tensor] = None,
        depth: Optional[torch.tensor] = None,
        K: Optional[torch.tensor] = None,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Run the depth refinement.

        Args:
            predictions: len(predictions) = N, index into depth, masks, K using
                the batch_im_id field.
            depth: [B, H, W]
            masks: [B, H, W]
            K: [B,3,3]

        Returns: Tuple(refined_preds, extra_data)
            refined_preds: Replaces `poses` tensor with the refined poses.
            extra_data: dict with additional information

        """
