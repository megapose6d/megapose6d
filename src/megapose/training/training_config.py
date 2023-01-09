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
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import omegaconf
import torch

# MegaPose
from megapose.utils.types import Resolution


@dataclass
class DatasetConfig:
    ds_name: str
    mesh_obj_ds_name: str
    renderer_obj_ds_name: str
    n_repeats: int = 1


@dataclass
class HardwareConfig:
    n_cpus: int = 8
    n_gpus: int = 1


@dataclass
class TrainingConfig(omegaconf.dictconfig.DictConfig):
    """Training config.
    Two options for creating a training configuration:
    1. Create it manually, and set `run_id`.
    2. If `run_id` is None, then use `config_id`, `run_comment`and
    `run_postfix` to create a `run_id`

    In 2., the parameters of the config are set-up using the function `update_cfg_with_config_id`.
    """

    # Datasets
    train_datasets: List[DatasetConfig] = field(default_factory=lambda: [])
    input_resize: Resolution = (540, 720)
    val_datasets: List[DatasetConfig] = field(default_factory=lambda: [])
    val_epoch_interval: int = 10
    split_objects_across_gpus: bool = True
    n_max_objects: Optional[int] = None

    # Meshes
    n_symmetries_batch: int = 32
    resample_n_points: Optional[int] = None

    # Data augmentation
    rgb_augmentation: bool = True
    background_augmentation: bool = True
    depth_augmentation: bool = False
    depth_augmentation_level: int = 2
    min_area: Optional[float] = None

    # Run management
    run_id: Optional[str] = None
    resume_run_id: Optional[str] = None
    run_id_pretrain: Optional[str] = None
    save_dir: Optional[str] = None
    run_comment: str = ""
    run_postfix: str = str(np.random.randint(int(1e6)))
    batch_size: int = 16
    epoch_size: int = 115200
    val_size: int = 1280
    n_epochs: int = 700
    save_epoch_interval: int = 100
    n_dataloader_workers: int = 8
    n_rendering_workers: int = 8
    sample_buffer_size: int = 200
    renderer: str = "panda3d"
    logging_style: str = "manyprints"

    # Network
    backbone_str: str = "vanilla_resnet34"
    multiview_type: str = "TCO"
    views_inplane_rotations: bool = False
    n_rendered_views: int = 1
    remove_TCO_rendering: bool = False
    predict_rendered_views_logits: bool = False
    predict_pose_update: bool = True
    render_normals: bool = True
    input_depth: bool = False
    depth_normalization_type: str = "tCR_scale_clamp_center"
    render_depth: bool = False

    # Training
    # Hypotheses
    hypotheses_init_method: str = "refiner_gt+noise"
    n_hypotheses: int = 1
    init_euler_deg_std: Tuple[float, float, float] = (15, 15, 15)
    init_trans_std: Tuple[float, float, float] = (0.01, 0.01, 0.05)

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = 1000.0
    lr: float = 3e-4
    n_epochs_warmup: int = 50
    lr_epoch_decay: int = 500

    # Misc
    n_iterations: int = 3
    add_iteration_epoch_interval: int = 100
    random_ambient_light: bool = True

    # Loss
    n_points_loss: int = 2000
    loss_alpha_pose: float = 1.0
    renderings_logits_temperature: float = 1.0
    loss_alpha_renderings_confidence: float = 0.0

    # Visualization
    do_visualization: bool = False
    vis_epoch_interval: int = 100
    vis_batch_size: int = 64
    vis_save_only_last: bool = False

    # Misc
    pytorch_version: str = str(torch.__version__)
    sync_batchnorm: bool = False
    overfit: bool = False
    cuda_timing: bool = False

    # Infos
    global_batch_size: Optional[int] = None
    hardware: HardwareConfig = HardwareConfig()
