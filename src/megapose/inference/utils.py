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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

# MegaPose
import megapose
import megapose.utils.tensor_collection as tc
from megapose.config import EXP_DIR
from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.detector import Detector
from megapose.inference.types import DetectionsType, PoseEstimatesType
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.models.pose_rigid import PosePredictor
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector,
)
from megapose.training.detector_models_cfg import create_model_detector
from megapose.training.pose_models_cfg import (
    check_update_config as check_update_config_pose,
)
from megapose.training.pose_models_cfg import create_model_pose
from megapose.training.training_config import TrainingConfig
from megapose.utils.logging import get_logger
from megapose.utils.models_compat import change_keys_of_older_models
from megapose.utils.tensor_collection import PandasTensorCollection

logger = get_logger(__name__)


def load_detector(run_id: str) -> torch.nn.Module:
    run_dir = EXP_DIR / run_id
    cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.UnsafeLoader)
    cfg = check_update_config_detector(cfg)
    label_to_category_id = cfg.label_to_category_id
    model = create_model_detector(cfg, len(label_to_category_id))
    ckpt = torch.load(run_dir / "checkpoint.pth.tar")
    ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt)
    model = model.cuda().eval()
    model.cfg = cfg
    model.config = cfg
    model = Detector(model)
    return model


def load_cfg(path: Union[str, Path]) -> OmegaConf:
    cfg = yaml.load(Path(path).read_text(), Loader=yaml.UnsafeLoader)
    if isinstance(cfg, dict):
        cfg = OmegaConf.load(path)
    return cfg


def load_pose_models(
    coarse_run_id: str,
    refiner_run_id: str,
    object_dataset: RigidObjectDataset,
    force_panda3d_renderer: bool = False,
    renderer_kwargs: Optional[dict] = None,
    models_root: Path = EXP_DIR,
) -> Tuple[torch.nn.Module, torch.nn.Module, megapose.lib3d.rigid_mesh_database.BatchedMeshes]:

    coarse_run_dir = models_root / coarse_run_id
    coarse_cfg: TrainingConfig = load_cfg(coarse_run_dir / "config.yaml")
    coarse_cfg = check_update_config_pose(coarse_cfg)

    refiner_run_dir = models_root / refiner_run_id
    refiner_cfg: TrainingConfig = load_cfg(refiner_run_dir / "config.yaml")
    refiner_cfg = check_update_config_pose(refiner_cfg)

    # TODO: Handle loading older cosypose models with bullet renderer.
    assert force_panda3d_renderer

    logger.debug("Creating MeshDatabase")
    mesh_db = MeshDataBase.from_object_ds(object_dataset)
    logger.debug("Done creating MeshDatabase")

    def make_renderer(renderer_type: str) -> Panda3dBatchRenderer:
        logger.debug("renderer_kwargs", renderer_kwargs)
        if renderer_kwargs is None:
            renderer_kwargs_ = dict()
        else:
            renderer_kwargs_ = renderer_kwargs

        renderer_kwargs_.setdefault("split_objects", True)
        renderer_kwargs_.setdefault("preload_cache", False)
        renderer_kwargs_.setdefault("n_workers", 4)

        if renderer_type == "panda3d" or force_panda3d_renderer:
            renderer = Panda3dBatchRenderer(object_dataset=object_dataset, **renderer_kwargs_)
        else:
            raise ValueError(renderer_type)
        return renderer

    coarse_renderer = make_renderer(coarse_cfg.renderer)
    if refiner_cfg.renderer == coarse_cfg.renderer:
        refiner_renderer = coarse_renderer
    else:
        refiner_renderer = make_renderer(refiner_cfg.renderer)

    mesh_db_batched = mesh_db.batched().cuda()

    def load_model(run_id: str, renderer: Panda3dBatchRenderer) -> PosePredictor:
        if run_id is None:
            return
        run_dir = models_root / run_id
        cfg: TrainingConfig = load_cfg(run_dir / "config.yaml")
        cfg = check_update_config_pose(cfg)
        model = create_model_pose(cfg, renderer=renderer, mesh_db=mesh_db_batched)
        ckpt = torch.load(run_dir / "checkpoint.pth.tar")
        ckpt = ckpt["state_dict"]
        ckpt = change_keys_of_older_models(ckpt)
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        return model

    coarse_model = load_model(coarse_run_id, coarse_renderer)
    refiner_model = load_model(refiner_run_id, refiner_renderer)

    return coarse_model, refiner_model, mesh_db


def add_instance_id(
    inputs: Union[PoseEstimatesType, DetectionsType]
) -> Union[PoseEstimatesType, DetectionsType]:
    """Adds a column with instance_id to the provided detections.

    Instance_id uniquely identifies multiple occurences of the same object
    within a given image (specified by batch_im_id).
    """
    if "instance_id" in inputs.infos:
        return inputs

    def create_instance_id(df: pd.DataFrame) -> pd.DataFrame:
        df["instance_id"] = np.arange(len(df))
        return df

    df = inputs.infos
    df = df.groupby(["batch_im_id", "label"], group_keys=False).apply(
        lambda df: create_instance_id(df)
    )
    inputs.infos = df
    return inputs


def filter_detections(
    detections: DetectionsType,
    labels: Optional[List[str]] = None,
    one_instance_per_class: bool = False,
) -> DetectionsType:
    """Filter detections based on kwargs."""

    if labels is not None:
        df = detections.infos
        df = df[df.label.isin(labels)]
        detections = detections[df.index.tolist()]

    if one_instance_per_class:
        group_cols = ["batch_im_id", "label"]
        filter_field = "score"
        df = detections.infos
        df = df.sort_values(filter_field, ascending=False).groupby(group_cols).head(1)

        detections = detections[df.index.tolist()]

    return detections


def make_cameras(camera_data: List[CameraData]) -> PandasTensorCollection:
    """Creates a PandasTensorCollection from list of camera data.

    Returns:
        PandasTensorCollection.
            infos: pd.DataFrame with columns ['batch_im_id', 'resolution']
            tensor: K with shape [B,3,3] of camera intrinsics matrices.
    """
    infos = []
    K = []
    for n, cam_data in enumerate(camera_data):
        K.append(torch.tensor(cam_data.K))
        infos.append(dict(batch_im_id=n, resolution=cam_data.resolution))

    return tc.PandasTensorCollection(infos=pd.DataFrame(infos), K=torch.stack(K))


def make_detections_from_object_data(object_data: List[ObjectData]) -> DetectionsType:
    infos = pd.DataFrame(
        dict(
            label=[data.label for data in object_data],
            batch_im_id=0,
            instance_id=np.arange(len(object_data)),
        )
    )
    bboxes = torch.as_tensor(
        np.stack([data.bbox_modal for data in object_data]),
    )
    return PandasTensorCollection(infos=infos, bboxes=bboxes)
