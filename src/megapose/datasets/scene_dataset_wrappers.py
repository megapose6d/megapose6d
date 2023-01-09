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
import dataclasses

# Third Party
import numpy as np

# Local Folder
from .scene_dataset import SceneDataset, SceneObservation


class SceneDatasetWrapper(SceneDataset):
    def __init__(self, scene_ds: SceneDataset):
        self.scene_ds = scene_ds
        self.iterator = None
        super().__init__(
            frame_index=scene_ds.frame_index,
            load_depth=scene_ds.load_depth,
            load_segmentation=scene_ds.load_segmentation,
        )

    @property
    def unwrapped(self) -> SceneDataset:
        if isinstance(self, SceneDatasetWrapper):
            return self.scene_ds
        else:
            return self

    def __len__(self) -> int:
        return len(self.scene_ds)

    def __getitem__(self, idx: int) -> SceneObservation:
        data = self.scene_ds[idx]
        return self.process_observation(data)

    @staticmethod
    def process_observation(obs: SceneObservation) -> SceneObservation:
        raise NotImplementedError


def remove_invisible_objects(obs: SceneObservation) -> SceneObservation:
    """Remove objects that do not appear in the segmentation."""
    assert obs.segmentation is not None
    assert obs.object_datas is not None
    ids_in_segm = np.unique(obs.segmentation)
    ids_visible = set(ids_in_segm[ids_in_segm > 0])
    visib_object_datas = [
        object_data for object_data in obs.object_datas if object_data.unique_id in ids_visible
    ]
    new_obs = dataclasses.replace(obs, object_datas=visib_object_datas)
    return new_obs


class VisibilityWrapper(SceneDatasetWrapper):
    @staticmethod
    def process_observation(obs: SceneObservation) -> SceneObservation:
        return remove_invisible_objects(obs)
