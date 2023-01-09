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
from typing import Dict, List

# Third Party
import numpy as np


def make_detections_from_segmentation(
    segmentations: np.ndarray,
) -> List[Dict[int, np.ndarray]]:
    """
    segmentations: (n, h, w) int np.ndarray
    """
    assert segmentations.ndim == 3
    detections = []
    for segmentation_n in segmentations:
        dets_n = dict()
        for unique_id in np.unique(segmentation_n):
            ids = np.where(segmentation_n == unique_id)
            x1, y1, x2, y2 = np.min(ids[1]), np.min(ids[0]), np.max(ids[1]), np.max(ids[0])
            dets_n[int(unique_id)] = np.array([x1, y1, x2, y2])
        detections.append(dets_n)
    return detections
