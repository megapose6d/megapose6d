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
from typing import Any, Optional

# Third Party
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

# MegaPose
import megapose
import megapose.utils.tensor_collection as tc
from megapose.inference.types import DetectionsType, ObservationTensor


class Detector(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.eval()
        self.config = model.config
        self.category_id_to_label = {v: k for k, v in self.config.label_to_category_id.items()}

    def image_tensor_from_numpy(
        self,
        rgb: npt.NDArray[np.uint8],
    ) -> torch.tensor:
        """Convert numpy image to torch tensor.


        Args:
            rgb: [H,W,3]

        Returns:
            rgb_tensor: [3,H,W] torch.tensor with dtype torch.float
        """
        assert rgb.dtype == np.uint8
        rgb_tensor = torch.as_tensor(rgb).float() / 255

        # convert it to [C,H,W]
        if rgb_tensor.shape[-1] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)

        return rgb_tensor

    @torch.no_grad()
    def get_detections(
        self,
        observation: ObservationTensor,
        detection_th: Optional[float] = None,
        output_masks: bool = False,
        mask_th: float = 0.8,
        one_instance_per_class: bool = False,
    ) -> DetectionsType:
        """Runs the detector on the given images.

        Args:
            detection_th: If specified only keep detections above this
                threshold.
            mask_th: Threshold to use when computing masks
            one_instance_per_class: If True, keep only the highest scoring
                detection within each class.


        """

        # [B,3,H,W]
        RGB_DIMS = [0, 1, 2]
        images = observation.images[:, RGB_DIMS]

        # TODO (lmanuelli): Why are we splitting this up into a list of tensors?
        outputs_ = self.model([image_n for image_n in images])

        infos = []
        bboxes = []
        masks = []
        for n, outputs_n in enumerate(outputs_):
            outputs_n["labels"] = [
                self.category_id_to_label[category_id.item()] for category_id in outputs_n["labels"]
            ]
            for obj_id in range(len(outputs_n["boxes"])):
                bbox = outputs_n["boxes"][obj_id]
                info = dict(
                    batch_im_id=n,
                    label=outputs_n["labels"][obj_id],
                    score=outputs_n["scores"][obj_id].item(),
                )
                mask = outputs_n["masks"][obj_id, 0] > mask_th
                bboxes.append(torch.as_tensor(bbox))
                masks.append(torch.as_tensor(mask))
                infos.append(info)

        if len(bboxes) > 0:
            bboxes = torch.stack(bboxes).cuda().float()
            masks = torch.stack(masks).cuda()
        else:
            infos = dict(score=[], label=[], batch_im_id=[])
            bboxes = torch.empty(0, 4).cuda().float()
            masks = torch.empty(0, images.shape[1], images.shape[2], dtype=torch.bool).cuda()

        outputs = tc.PandasTensorCollection(
            infos=pd.DataFrame(infos),
            bboxes=bboxes,
        )
        if output_masks:
            outputs.register_tensor("masks", masks)
        if detection_th is not None:
            keep = np.where(outputs.infos["score"] > detection_th)[0]
            outputs = outputs[keep]

        # Keep only the top-detection for each class label
        if one_instance_per_class:
            outputs = megapose.inference.utils.filter_detections(
                outputs, one_instance_per_class=True
            )

        # Add instance_id column to dataframe
        # Each detection is now associated with an `instance_id` that
        # identifies multiple instances of the same object
        outputs = megapose.inference.utils.add_instance_id(outputs)
        return outputs

    def __call__(self, *args: Any, **kwargs: Any) -> DetectionsType:
        return self.get_detections(*args, **kwargs)
