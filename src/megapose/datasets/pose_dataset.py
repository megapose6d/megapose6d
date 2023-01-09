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
import random
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Set, Union

# Third Party
import numpy as np
import torch

# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.augmentations import (
    CropResizeToAspectTransform,
    DepthBackgroundDropoutTransform,
    DepthBlurTransform,
    DepthCorrelatedGaussianNoiseTransform,
    DepthDropoutTransform,
    DepthEllipseDropoutTransform,
    DepthEllipseNoiseTransform,
    DepthGaussianNoiseTransform,
    DepthMissingTransform,
    PillowBlur,
    PillowBrightness,
    PillowColor,
    PillowContrast,
    PillowSharpness,
)
from megapose.datasets.augmentations import SceneObservationAugmentation as SceneObsAug
from megapose.datasets.augmentations import VOCBackgroundAugmentation
from megapose.datasets.scene_dataset import (
    IterableSceneDataset,
    ObjectData,
    SceneDataset,
    SceneObservation,
)
from megapose.datasets.scene_dataset_wrappers import remove_invisible_objects
from megapose.utils.types import Resolution


@dataclass
class PoseData:
    """
    rgb: (h, w, 3) uint8
    depth: (bsz, h, w) float32
    bbox: (4, ) int
    K: (3, 3) float32
    TCO: (4, 4) float32
    """

    rgb: np.ndarray
    bbox: np.ndarray
    TCO: np.ndarray
    K: np.ndarray
    depth: Optional[np.ndarray]
    object_data: ObjectData


@dataclass
class BatchPoseData:
    """
    rgbs: (bsz, 3, h, w) uint8
    depths: (bsz, h, w) float32
    bboxes: (bsz, 4) int
    TCO: (bsz, 4, 4) float32
    K: (bsz, 3, 3) float32
    """

    rgbs: torch.Tensor
    object_datas: List[ObjectData]
    bboxes: torch.Tensor
    TCO: torch.Tensor
    K: torch.Tensor
    depths: Optional[torch.Tensor] = None

    def pin_memory(self) -> "BatchPoseData":
        self.rgbs = self.rgbs.pin_memory()
        self.bboxes = self.bboxes.pin_memory()
        self.TCO = self.TCO.pin_memory()
        self.K = self.K.pin_memory()
        if self.depths is not None:
            self.depths = self.depths.pin_memory()
        return self


class NoObjectError(Exception):
    pass


class PoseDataset(torch.utils.data.IterableDataset):
    RGB_DIMS = [0, 1, 2]
    DEPTH_DIMS = [3]

    def __init__(
        self,
        scene_ds: Union[SceneDataset, IterableSceneDataset],
        resize: Resolution = (480, 640),
        min_area: Optional[float] = None,
        apply_rgb_augmentation: bool = True,
        apply_depth_augmentation: bool = False,
        apply_background_augmentation: bool = False,
        return_first_object: bool = False,
        keep_labels_set: Optional[Set[str]] = None,
        depth_augmentation_level: int = 1,
    ):

        self.scene_ds = scene_ds
        self.resize_transform = CropResizeToAspectTransform(resize=resize)
        self.min_area = min_area

        self.background_augmentations = []
        if apply_background_augmentation:
            self.background_augmentations += [
                (SceneObsAug(VOCBackgroundAugmentation(LOCAL_DATA_DIR / "VOC2012"), p=0.3))
            ]

        self.rgb_augmentations = []
        if apply_rgb_augmentation:
            self.rgb_augmentations += [
                SceneObsAug(
                    [
                        SceneObsAug(PillowBlur(factor_interval=(1, 3)), p=0.4),
                        SceneObsAug(PillowSharpness(factor_interval=(0.0, 50.0)), p=0.3),
                        SceneObsAug(PillowContrast(factor_interval=(0.2, 50.0)), p=0.3),
                        SceneObsAug(PillowBrightness(factor_interval=(0.1, 6.0)), p=0.5),
                        SceneObsAug(PillowColor(factor_interval=(0.0, 20.0)), p=0.3),
                    ],
                    p=0.8,
                )
            ]

        self.depth_augmentations = []
        if apply_depth_augmentation:
            # original augmentations
            if depth_augmentation_level == 0:
                self.depth_augmentations += [
                    SceneObsAug(DepthBlurTransform(), p=0.3),
                    SceneObsAug(DepthEllipseDropoutTransform(), p=0.3),
                    SceneObsAug(DepthGaussianNoiseTransform(std_dev=0.01), p=0.3),
                    SceneObsAug(DepthMissingTransform(max_missing_fraction=0.2), p=0.3),
                ]

            # medium augmentation
            elif depth_augmentation_level in {1, 2}:
                # medium augmentation
                self.depth_augmentations += [
                    SceneObsAug(DepthBlurTransform(), p=0.3),
                    SceneObsAug(
                        DepthCorrelatedGaussianNoiseTransform(
                            gp_rescale_factor_min=15.0, gp_rescale_factor_max=40.0, std_dev=0.01
                        ),
                        p=0.3,
                    ),
                    SceneObsAug(
                        DepthEllipseDropoutTransform(
                            ellipse_dropout_mean=175.0,
                            ellipse_gamma_shape=5.0,
                            ellipse_gamma_scale=2.0,
                        ),
                        p=0.5,
                    ),
                    SceneObsAug(
                        DepthEllipseNoiseTransform(
                            ellipse_dropout_mean=175.0,
                            ellipse_gamma_shape=5.0,
                            ellipse_gamma_scale=2.0,
                            std_dev=0.01,
                        ),
                        p=0.5,
                    ),
                    SceneObsAug(DepthGaussianNoiseTransform(std_dev=0.01), p=0.1),
                    SceneObsAug(DepthMissingTransform(max_missing_fraction=0.9), p=0.3),
                ]

                # Set the depth image to zero occasionally.
                if depth_augmentation_level == 2:
                    self.depth_augmentations.append(SceneObsAug(DepthDropoutTransform(), p=0.3))
                    self.depth_augmentations.append(
                        SceneObsAug(DepthBackgroundDropoutTransform(), p=0.2)
                    )
                self.depth_augmentations = [SceneObsAug(self.depth_augmentations, p=0.8)]
            else:
                raise ValueError(f"Unknown depth augmentation type {depth_augmentation_level}")

        self.return_first_object = return_first_object

        self.keep_labels_set = None
        if keep_labels_set is not None:
            self.keep_labels_set = keep_labels_set

    def collate_fn(self, list_data: List[PoseData]) -> BatchPoseData:
        batch_data = BatchPoseData(
            rgbs=torch.from_numpy(np.stack([d.rgb for d in list_data])).permute(0, 3, 1, 2),
            bboxes=torch.from_numpy(np.stack([d.bbox for d in list_data])),
            K=torch.from_numpy(np.stack([d.K for d in list_data])),
            TCO=torch.from_numpy(np.stack([d.TCO for d in list_data])),
            object_datas=[d.object_data for d in list_data],
        )

        has_depth = [d.depth is not None for d in list_data]
        if all(has_depth):
            batch_data.depths = torch.from_numpy(np.stack([d.depth for d in list_data]))  # type: ignore # noqa
        return batch_data

    def make_data_from_obs(self, obs: SceneObservation) -> Union[PoseData, None]:
        """Construct a PoseData for a object random of the scene_ds[idx] observation.
        The object satisfies the constraints:
            1. The visible 2D area is superior or equal to min_area
            2. if `keep_objects_set` isn't None, the object must belong to this set
        If there are no objects that satisfy this condition in the observation, returns None.
        """

        obs = remove_invisible_objects(obs)

        start = time.time()
        timings = dict()

        s = time.time()
        obs = self.resize_transform(obs)
        timings["resize_augmentation"] = time.time() - s

        s = time.time()
        for aug in self.background_augmentations:
            obs = aug(obs)
        timings["background_augmentation"] = time.time() - s

        s = time.time()
        for aug in self.rgb_augmentations:
            obs = aug(obs)
        timings["rgb_augmentation"] = time.time() - s

        s = time.time()
        for aug in self.depth_augmentations:
            obs = aug(obs)
        timings["depth_augmentation"] = time.time() - s

        s = time.time()
        unique_ids_visible = set(np.unique(obs.segmentation))
        valid_objects = []

        assert obs.object_datas is not None
        assert obs.rgb is not None
        assert obs.camera_data is not None
        for obj in obs.object_datas:
            assert obj.bbox_modal is not None
            valid = False
            if obj.unique_id in unique_ids_visible and np.all(obj.bbox_modal) >= 0:
                valid = True

            if valid and self.min_area is not None:
                bbox = obj.bbox_modal
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area >= self.min_area:
                    valid = True
                else:
                    valid = False

            if valid and self.keep_labels_set is not None:
                valid = obj.label in self.keep_labels_set

            if valid:
                valid_objects.append(obj)

        if len(valid_objects) == 0:
            return None

        if self.return_first_object:
            object_data = valid_objects[0]
        else:
            object_data = random.sample(valid_objects, k=1)[0]
        assert object_data.bbox_modal is not None

        timings["other"] = time.time() - s
        timings["total"] = time.time() - start

        for k, v in timings.items():
            timings[k] = v * 1000

        self.timings = timings

        assert obs.camera_data.K is not None
        assert obs.camera_data.TWC is not None
        assert object_data.TWO is not None
        # Add depth to PoseData
        data = PoseData(
            rgb=obs.rgb,
            depth=obs.depth if obs.depth is not None else None,
            bbox=object_data.bbox_modal,
            K=obs.camera_data.K,
            TCO=(obs.camera_data.TWC.inverse() * object_data.TWO).matrix,
            object_data=object_data,
        )
        return data

    def __getitem__(self, index: int) -> Union[PoseData, None]:
        assert isinstance(self.scene_ds, SceneDataset)
        obs = self.scene_ds[index]
        return self.make_data_from_obs(obs)

    def find_valid_data(self, iterator: Iterator[SceneObservation]) -> PoseData:
        n_attempts = 0
        while True:
            obs = next(iterator)
            data = self.make_data_from_obs(obs)
            if data is not None:
                return data
            n_attempts += 1
            if n_attempts > 200:
                raise ValueError("Cannot find valid image in the dataset")

    def __iter__(self) -> Iterator[PoseData]:
        assert isinstance(self.scene_ds, IterableSceneDataset)
        iterator = iter(self.scene_ds)
        while True:
            yield self.find_valid_data(iterator)
