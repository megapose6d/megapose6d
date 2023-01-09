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
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

# Third Party
import numpy as np
import pandas as pd
import torch
import webdataset as wds

# MegaPose
import megapose.utils.tensor_collection as tc
from megapose.lib3d.transform import Transform
from megapose.utils.random import make_seed
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.utils.types import Resolution

ListBbox = List[int]
ListPose = List[List[float]]

"""
infos: pd.DataFrame with fields
    - 'label': str, the object label
    - 'scene_id': int
    - 'view_id': int
    - 'visib_fract': float

tensors:
    K: [B,3,3] camera intrinsics
    poses: [B,4,4] object to camera transform
    poses_init [Optional]: [B,4,4] object to camera transform. Used if the dataset has initial estimates (ModelNet)
    TCO: same as poses
    bboxes: [B,4] bounding boxes for objects
    masks: (optional)

"""
SceneObservationTensorCollection = PandasTensorCollection

SingleDataJsonType = Union[str, float, ListPose, int, ListBbox, Any]
DataJsonType = Union[Dict[str, SingleDataJsonType], List[SingleDataJsonType]]


def transform_to_list(T: Transform) -> ListPose:
    return [T.quaternion.coeffs().tolist(), T.translation.tolist()]


@dataclass
class ObjectData:
    # NOTE (Yann): bbox_amodal, bbox_modal, visib_fract should be moved to SceneObservation
    label: str
    TWO: Optional[Transform] = None
    unique_id: Optional[int] = None
    bbox_amodal: Optional[np.ndarray] = None  # (4, ) array [xmin, ymin, xmax, ymax]
    bbox_modal: Optional[np.ndarray] = None  # (4, ) array [xmin, ymin, xmax, ymax]
    visib_fract: Optional[float] = None
    TWO_init: Optional[
        Transform
    ] = None  # Some pose estimation datasets (ModelNet) provide an initial pose estimate
    #  NOTE: This should be loaded externally

    def to_json(self) -> Dict[str, SingleDataJsonType]:
        d: Dict[str, SingleDataJsonType] = dict(label=self.label)
        for k in ("TWO", "TWO_init"):
            if getattr(self, k) is not None:
                d[k] = transform_to_list(getattr(self, k))
        for k in ("bbox_amodal", "bbox_modal"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k).tolist()
        for k in ("visib_fract", "unique_id"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    @staticmethod
    def from_json(d: DataJsonType) -> "ObjectData":
        assert isinstance(d, dict)
        label = d["label"]
        assert isinstance(label, str)
        data = ObjectData(label=label)
        for k in ("TWO", "TWO_init"):
            if k in d:
                item = d[k]
                assert isinstance(item, list)
                quat_list, trans_list = item
                assert isinstance(quat_list, list)
                assert isinstance(trans_list, list)
                quat = tuple(quat_list)
                trans = tuple(trans_list)
                setattr(data, k, Transform(quat, trans))
        for k in ("unique_id", "visib_fract"):
            if k in d:
                setattr(data, k, d[k])
        for k in ("bbox_amodal", "bbox_modal"):
            if k in d:
                setattr(data, k, np.array(d[k]))
        return data


@dataclass
class CameraData:
    K: Optional[np.ndarray] = None
    resolution: Optional[Resolution] = None
    TWC: Optional[Transform] = None
    camera_id: Optional[str] = None
    TWC_init: Optional[
        Transform
    ] = None  # Some pose estimation datasets (ModelNet) provide an initial pose estimate
    #  NOTE: This should be loaded externally

    def to_json(self) -> str:
        d: Dict[str, SingleDataJsonType] = dict()
        for k in ("TWC", "TWC_init"):
            if getattr(self, k) is not None:
                d[k] = transform_to_list(getattr(self, k))
        for k in ("K",):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k).tolist()
        for k in ("camera_id", "resolution"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return json.dumps(d)

    @staticmethod
    def from_json(data_str: str) -> "CameraData":
        d: DataJsonType = json.loads(data_str)
        assert isinstance(d, dict)
        data = CameraData()
        for k in ("TWC", "TWC_init"):
            if k in d:
                item = d[k]
                assert isinstance(item, list)
                quat_list, trans_list = item
                assert isinstance(quat_list, list)
                assert isinstance(trans_list, list)
                quat = tuple(quat_list)
                trans = tuple(trans_list)
                setattr(data, k, Transform(tuple(quat), tuple(trans)))
        for k in ("camera_id",):
            if k in d:
                setattr(data, k, d[k])
        for k in ("K",):
            if k in d:
                setattr(data, k, np.array(d[k]))
        if "resolution" in d:
            assert isinstance(d["resolution"], list)
            h, w = d["resolution"]
            assert isinstance(h, int)
            assert isinstance(w, int)
            data.resolution = (h, w)
        return data


@dataclass
class ObservationInfos:
    scene_id: str
    view_id: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(data_str: str) -> "ObservationInfos":
        d = json.loads(data_str)
        assert "scene_id" in d
        assert "view_id" in d
        return ObservationInfos(scene_id=d["scene_id"], view_id=d["view_id"])


@dataclass
class SceneObservation:
    rgb: Optional[np.ndarray] = None  # (h,w,3) uint8 numpy array
    depth: Optional[np.ndarray] = None  # (h, w), np.float32
    segmentation: Optional[np.ndarray] = None  # (h, w), np.uint32 (important);
    # contains objects unique ids. int64 are not handled and can be dangerous when used with PIL
    infos: Optional[ObservationInfos] = None
    object_datas: Optional[List[ObjectData]] = None
    camera_data: Optional[CameraData] = None
    binary_masks: Optional[
        Dict[int, np.ndarray]
    ] = None  # dict mapping unique id to (h, w) np.bool_

    @staticmethod
    def collate_fn(
        batch: List[SceneObservation], object_labels: Optional[List[str]] = None
    ) -> Dict[Any, Any]:
        """Collate a batch of SceneObservation objects.

        Args:
            object_labels: If passed in parse only those object labels.

        Returns:
            A dict with fields
                cameras: PandasTensorCollection
                rgb: torch.tensor [B,3,H,W] torch.uint8
                depth: torch.tensor [B,1,H,W]
                im_infos: List[dict]
                gt_detections: SceneObservationTensorCollection
                gt_data: SceneObservationTensorCollection


        """
        if object_labels is not None:
            object_labels = set(object_labels)

        cam_infos, K = [], []
        im_infos = []
        gt_data = []
        gt_detections = []
        initial_data = []
        batch_im_id = -1
        rgb_images = []
        depth_images = []

        for n, data in enumerate(batch):
            # data is of type SceneObservation
            batch_im_id += 1
            im_info = dict(
                scene_id=data.infos.scene_id,
                view_id=data.infos.view_id,
                batch_im_id=batch_im_id,
            )
            im_infos.append(im_info)

            K.append(data.camera_data.K)
            cam_info = dict(
                TWC=data.camera_data.TWC,
                resolution=data.camera_data.resolution,
            )
            cam_infos.append(cam_info)

            # [3,H,W]
            rgb = torch.as_tensor(data.rgb).permute(2, 0, 1).to(torch.uint8)
            rgb_images.append(rgb)
            if data.depth is not None:
                depth = np.expand_dims(data.depth, axis=0)
            else:
                depth = np.array([])

            depth_images.append(depth)

            gt_data_ = data.as_pandas_tensor_collection(object_labels=object_labels)
            gt_data_.infos["batch_im_id"] = batch_im_id  # Add batch_im_id
            gt_data.append(gt_data_)

            initial_data_ = None
            if hasattr(gt_data_, "poses_init"):
                initial_data_ = copy.deepcopy(gt_data_)
                initial_data_.poses = initial_data_.poses_init
                initial_data.append(initial_data_)

            # Emulate detection data
            gt_detections_ = copy.deepcopy(gt_data_)
            gt_detections_.infos["score"] = 1.0  # Add score field
            gt_detections.append(gt_detections_)

        gt_data = tc.concatenate(gt_data)
        gt_detections = tc.concatenate(gt_detections)
        if initial_data:
            initial_data = tc.concatenate(initial_data)
        else:
            initial_data = None

        cameras = tc.PandasTensorCollection(
            infos=pd.DataFrame(cam_infos),
            K=torch.as_tensor(np.stack(K)),
        )
        return dict(
            cameras=cameras,
            rgb=torch.stack(rgb_images),  # [B,3,H,W]
            depth=torch.as_tensor(np.stack(depth_images)),  # [B,1,H,W] or [B,0]
            im_infos=im_infos,
            gt_detections=gt_detections,
            gt_data=gt_data,
            initial_data=initial_data,
        )

    def as_pandas_tensor_collection(
        self,
        object_labels: Optional[List[str]] = None,
    ) -> SceneObservationTensorCollection:
        """Convert SceneData to a PandasTensorCollection representation."""
        obs = self

        assert obs.camera_data is not None
        assert obs.object_datas is not None

        infos = []
        TWO = []
        bboxes = []
        masks = []
        TWC = torch.as_tensor(obs.camera_data.TWC.matrix).float()

        TWO_init = []
        TWC_init = None
        if obs.camera_data.TWC_init is not None:
            TWC_init = torch.as_tensor(obs.camera_data.TWC_init.matrix).float()

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
                binary_mask = np.zeros_like(obs.segmentation, dtype=np.bool_)
                binary_mask[obs.segmentation == obj_data.unique_id] = 1
                binary_mask = torch.as_tensor(binary_mask).float()
                masks.append(binary_mask)

            if obj_data.TWO_init:
                TWO_init.append(torch.tensor(obj_data.TWO_init.matrix).float())

        TWO = torch.stack(TWO)
        bboxes = torch.stack(bboxes)
        infos = pd.DataFrame(infos)
        if len(masks) > 0:
            masks = torch.stack(masks)
        else:
            masks = None

        B = len(infos)

        TCW = torch.linalg.inv(TWC)  # [4,4]

        # [B,4,4]
        TCO = TCW.unsqueeze(0) @ TWO
        TCO_init = None
        if len(TWO_init):
            TCO_init = torch.linalg.inv(TWC_init).unsqueeze(0) @ torch.stack(TWO_init)
        K = torch.tensor(obs.camera_data.K).unsqueeze(0).expand([B, -1, -1])

        data = tc.PandasTensorCollection(
            infos=infos,
            TCO=TCO,
            bboxes=bboxes,
            poses=TCO,
            K=K,
        )

        # Only register the mask tensor if it is not None
        if masks is not None:
            data.register_tensor("masks", masks)
        if TCO_init is not None:
            data.register_tensor("TCO_init", TCO_init)
            data.register_tensor("poses_init", TCO_init)
        return data


class SceneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        frame_index: Optional[pd.DataFrame],
        load_depth: bool = False,
        load_segmentation: bool = True,
    ):
        """Scene dataset.
        Can be an IterableDataset or a map-style Dataset.

        Args:
            frame_index (pd.DataFrame): Must contain the following columns: scene_id, view_id
            load_depth (bool, optional): Whether to load depth images. Defaults to False.
            load_segmentation (bool, optional): Whether to load image segmentation.
            Defaults to True.
            Defaults to f'{label}'.
        """

        self.frame_index = frame_index
        self.load_depth = load_depth
        self.load_segmentation = load_segmentation

    def _load_scene_observation(self, image_infos: ObservationInfos) -> SceneObservation:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SceneObservation:
        assert self.frame_index is not None
        row = self.frame_index.iloc[idx]
        infos = ObservationInfos(scene_id=row.scene_id, view_id=row.view_id)
        return self._load_scene_observation(infos)

    def __len__(self) -> int:
        assert self.frame_index is not None
        return len(self.frame_index)

    def __iter__(self) -> Iterator[SceneObservation]:
        raise NotImplementedError


class IterableSceneDataset:
    def __iter__(self) -> Iterator[SceneObservation]:
        """Returns an infinite iterator over SceneObservation samples."""
        raise NotImplementedError


class RandomIterableSceneDataset(IterableSceneDataset):
    """RandomIterableSceneDataset.

    Generates an infinite iterator over SceneObservation by
    randomly sampling from a SceneDataset.
    """

    def __init__(
        self,
        scene_ds: SceneDataset,
        deterministic: bool = False,
    ):
        self.scene_ds = scene_ds
        self.deterministic = deterministic
        self.worker_seed_fn = wds.utils.pytorch_worker_seed

    def __iter__(self) -> Iterator[SceneObservation]:
        if self.deterministic:
            seed = make_seed(self.worker_seed_fn())
        else:
            seed = make_seed(
                self.worker_seed_fn(),
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        self.rng = random.Random(seed)
        while True:
            idx = self.rng.randint(0, len(self.scene_ds) - 1)
            yield self.scene_ds[idx]


class IterableMultiSceneDataset(IterableSceneDataset):
    def __init__(
        self,
        list_iterable_scene_ds: List[IterableSceneDataset],
        deterministic: bool = False,
    ):
        self.list_iterable_scene_ds = list_iterable_scene_ds
        self.deterministic = deterministic
        self.worker_seed_fn = wds.utils.pytorch_worker_seed

    def __iter__(self) -> Iterator[SceneObservation]:
        if self.deterministic:
            seed = make_seed(self.worker_seed_fn())
        else:
            seed = make_seed(
                self.worker_seed_fn(),
                os.getpid(),
                time.time_ns(),
                os.urandom(4),
            )
        self.rng = random.Random(seed)
        self.iterators = [iter(ds) for ds in self.list_iterable_scene_ds]
        while True:
            idx = self.rng.randint(0, len(self.iterators) - 1)
            yield next(self.iterators[idx])
