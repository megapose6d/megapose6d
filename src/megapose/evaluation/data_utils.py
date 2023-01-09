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
from typing import List, Optional

# Third Party
import numpy as np
import pandas as pd
import torch

# MegaPose
import megapose.utils.tensor_collection as tc
from megapose.datasets.scene_dataset import SceneObservation
from megapose.utils.tensor_collection import PandasTensorCollection


def parse_obs_data(
    obs: SceneObservation,
    object_labels: Optional[List[str]] = None,
) -> PandasTensorCollection:
    """Parses object data into PandasTensorCollection.

    Args:
        obs: The scene observation.
        object_labels: If specified will only parse information for these
            object labels.

    Returns:
        PandasTensorCollection
            infos: pd.DataFrame with fields ['label',
                'scene_id', 'view_id', 'visib_fract']
            tensors:
                K: [B,3,3] camera intrinsics
                poses: [B,4,4] object to camera transform
                TCO: same as poses
                bboxes: [B,4] bounding boxes for objects
                masks: (optional)

    """

    raise ValueError("This function is deprecated.")
    infos = []
    TWO = []
    bboxes = []
    masks = []
    TWC = torch.as_tensor(obs.camera_data.TWC.matrix).float()
    for n, obj_data in enumerate(obs.object_datas):
        if object_labels is not None and obj_data.label not in object_labels:
            continue

        info = dict(
            label=obj_data.label,
            scene_id=obs.infos.scene_id,
            view_id=obs.infos.view_id,
            visib_fract=getattr(obj_data, "visib_fract", 1),
        )
        infos.append(info)
        TWO.append(torch.tensor(obj_data.TWO.matrix).float())
        bboxes.append(torch.tensor(obj_data.bbox_modal).float())

        if obs.binary_masks is not None:
            binary_mask = torch.tensor(obs.binary_masks[obj_data.unique_id]).float()
            masks.append(binary_mask)

        if obs.segmentation is not None:
            binary_mask = np.zeros_like(obs.segmentation)
            binary_mask[obs.segmentation == obj_data.unique_id] = 1
            binary_mask = torch.as_tensor(binary_mask).float()
            masks.append(binary_mask)

    TWO = torch.stack(TWO)
    bboxes = torch.stack(bboxes)
    infos = pd.DataFrame(infos)
    if len(masks) > 0:
        masks = torch.stack(masks)
    B = len(infos)

    TCW = torch.linalg.inv(TWC)  # [4,4]

    # [B,4,4]
    TCO = TCW.unsqueeze(0) @ TWO
    K = torch.tensor(obs.camera_data.K).unsqueeze(0).expand([B, -1, -1])

    data = tc.PandasTensorCollection(
        infos=infos,
        TCO=TCO,
        bboxes=bboxes,
        poses=TCO,
        K=K,
    )

    if masks is not None:
        data.register_tensor("masks", masks)

    return data
