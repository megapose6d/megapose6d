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


# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE

# Standard Library
import os
from typing import List, Optional

# Third Party
import numpy as np
from colorama import Fore, Style
from omegaconf import OmegaConf

# MegaPose
from megapose.bop_config import BOP_CONFIG
from megapose.config import EXP_DIR
from megapose.training.train_megapose import DatasetConfig, train_megapose
from megapose.training.training_config import HardwareConfig, TrainingConfig
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)


def train_on_bop_pbr_datasets(cfg: TrainingConfig, use_webdataset: bool = True) -> TrainingConfig:
    bop_names = ["lm", "tless", "itodd", "hb", "ycbv", "icbin", "tudl"]
    for bop_name in bop_names:
        bop_cfg = BOP_CONFIG[bop_name]
        ds_cfg = DatasetConfig(
            ds_name=str(bop_cfg["train_pbr_ds_name"][0]),
            mesh_obj_ds_name=str(bop_cfg["obj_ds_name"]),
            renderer_obj_ds_name=str(bop_cfg["obj_ds_name"]) + ".panda3d",
        )
        if use_webdataset:
            ds_cfg.ds_name = "webdataset." + ds_cfg.ds_name
        cfg.train_datasets.append(ds_cfg)
    cfg.input_resize = (540, 720)
    cfg.n_symmetries_batch = 32
    return cfg


def train_on_shapenet(
    cfg: TrainingConfig,
    ds_name: str = "shapenet_1M",
    obj_filters: List[str] = [
        "10mb_20k",
    ],
    remove_modelnet: bool = False,
) -> TrainingConfig:

    if remove_modelnet:
        obj_filters.append("remove_modelnet")

    cfg.input_resize = (540, 720)
    filters_str = ",".join(obj_filters)
    obj_ds_name = f"shapenet.filters={filters_str}"
    cfg.train_datasets.append(
        DatasetConfig(
            ds_name="webdataset." + ds_name,
            mesh_obj_ds_name=f"{obj_ds_name}.pointcloud",
            renderer_obj_ds_name=f"{obj_ds_name}.panda3d_bam",
        )
    )
    cfg.n_symmetries_batch = 1
    return cfg


def train_on_gso(
    cfg: TrainingConfig,
    ds_name: str = "gso_1M",
    n_objects: int = 940,
) -> TrainingConfig:

    cfg.input_resize = (540, 720)
    obj_ds_name = f"gso.nobjects={n_objects}"
    cfg.train_datasets.append(
        DatasetConfig(
            ds_name="webdataset." + ds_name,
            mesh_obj_ds_name=f"{obj_ds_name}.pointcloud",
            renderer_obj_ds_name=f"{obj_ds_name}.normalized",
        )
    )
    cfg.n_symmetries_batch = 1
    return cfg


def make_refiner_cfg(cfg: TrainingConfig) -> TrainingConfig:
    cfg.hypotheses_init_method = "refiner_gt+noise"
    cfg.n_hypotheses = 1
    cfg.predict_pose_update = True
    cfg.loss_alpha_pose = 1.0
    cfg.predict_rendered_views_logits = False
    cfg.multiview_type = "TCO+front_3views"
    cfg.n_rendered_views = 4
    cfg.n_iterations = 3
    return cfg


def make_coarse_cfg(cfg: TrainingConfig) -> TrainingConfig:
    cfg.hypotheses_init_method = "coarse_classif_multiview_paper"
    cfg.n_iterations = 1
    cfg.n_hypotheses = 6
    cfg.n_rendered_views = 1
    cfg.predict_rendered_views_logits = True
    cfg.multiview_type = "1view_TCO"
    cfg.renderings_logits_temperature = 1.0
    cfg.loss_alpha_renderings_confidence = 1.0
    cfg.remove_TCO_rendering = True
    cfg.predict_pose_update = False
    return cfg


def enable_depth_in_cfg(cfg: TrainingConfig) -> TrainingConfig:
    """Adds flags for input depth + render depth to cfg"""
    cfg.depth_normalization_type = "tCR_scale_clamp_center"
    cfg.input_depth = True
    cfg.render_depth = True
    cfg.sync_batchnorm = True
    cfg.depth_augmentation = True
    return cfg


def update_cfg_with_config_id(cfg: TrainingConfig, config_id: str) -> TrainingConfig:
    def train_on_gso_and_shapenet(
        cfg: TrainingConfig,
        shapenet_obj_ds_name: Optional[str] = "shapenet_1M",
        shapenet_obj_filters: List[str] = ["10mb_20k"],
        gso_obj_ds_name: Optional[str] = "gso_1M",
        gso_n_objects: int = 940,
        remove_modelnet: bool = False,
    ) -> TrainingConfig:
        cfg.train_datasets = []
        if shapenet_obj_ds_name is not None:
            cfg = train_on_shapenet(
                cfg,
                ds_name="shapenet_1M",
                obj_filters=shapenet_obj_filters,
                remove_modelnet=remove_modelnet,
            )
        if gso_obj_ds_name is not None:
            cfg = train_on_gso(cfg, ds_name="gso_1M", n_objects=gso_n_objects)
        return cfg

    #######
    #######

    # Views ablation
    if config_id == "refiner-gso_shapenet-1view-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
        cfg.n_rendered_views = 1
        cfg.multiview_type = "TCO"
    elif config_id == "refiner-gso_shapenet-2views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
        cfg.multiview_type = "TCO+front_1view"
        cfg.n_rendered_views = 2
    elif config_id == "refiner-gso_shapenet-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
    elif config_id == "refiner-gso_shapenet-4views-normals-depth":
        cfg = make_refiner_cfg(cfg)
        cfg = enable_depth_in_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
    elif config_id == "refiner-gso_shapenet-4views-no_normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
        cfg.render_normals = False

    # Data ablation
    elif config_id == "refiner-gso_shapenet-4views-normals-objects50p":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg, shapenet_obj_ds_name="shapenet_10mb_10k", gso_obj_ds_name="gso_500"
        )
    elif config_id == "refiner-gso_shapenet-4views-normals-objects25p":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg, shapenet_obj_ds_name="shapenet_10mb_2k", gso_obj_ds_name="gso_250"
        )
    elif config_id == "refiner-gso_shapenet-4views-normals-objects10p":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg, shapenet_obj_ds_name="shapenet_10mb_1k", gso_obj_ds_name="gso_100"
        )
    elif config_id == "refiner-gso_shapenet-4views-normals-objects1p":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg, shapenet_obj_ds_name="shapenet_10mb_100", gso_obj_ds_name="gso_10"
        )

    elif config_id == "refiner-gso-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg, shapenet_obj_ds_name=None, gso_obj_ds_name="gso_940")
    elif config_id == "refiner-shapenet-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg, shapenet_obj_ds_name="shapenet_10mb_20k", gso_obj_ds_name=None
        )
    elif config_id == "refiner-gso_shapenet_nomodelnet-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(
            cfg,
            shapenet_obj_ds_name="shapenet_10mb_20k",
            gso_obj_ds_name=None,
            remove_modelnet=True,
        )

    elif config_id == "refiner-all_bop-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_bop_pbr_datasets(cfg, use_webdataset=False)

    elif config_id == "refiner-all_bop-4views-normals-depth":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_bop_pbr_datasets(cfg, use_webdataset=False)
        cfg = enable_depth_in_cfg(cfg)

    # Modelnet config
    elif config_id == "refiner-gso_shapenet_nomodelnet-4views-normals":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)
    elif config_id == "refiner-gso_shapenet_nomodelnet-4views-normals-depth":
        cfg = make_refiner_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg, remove_modelnet=True)
        cfg = enable_depth_in_cfg(cfg)

    # Coarse
    elif config_id == "coarse-gso_shapenet-6hypothesis":
        cfg = make_coarse_cfg(cfg)
        cfg = train_on_gso_and_shapenet(cfg)

    else:
        raise ValueError("Unknown config")

    if cfg.run_id is None:
        cfg.run_postfix = str(np.random.randint(int(1e6)))
        cfg.run_id = f"{config_id}-{cfg.run_comment}-{cfg.run_postfix}"

    return cfg


def update_cfg_debug(cfg: TrainingConfig) -> TrainingConfig:
    assert cfg.run_id is not None
    cfg.n_epochs = 4
    cfg.val_epoch_interval = 1
    cfg.batch_size = 4
    cfg.epoch_size = 5 * cfg.batch_size * cfg.hardware.n_gpus
    cfg.run_id = "debug-" + cfg.run_id
    cfg.add_iteration_epoch_interval = 2
    cfg.train_datasets = cfg.train_datasets
    cfg.n_max_objects = 500
    cfg.sample_buffer_size = 1
    return cfg


def update_cfg_overfit(cfg: TrainingConfig) -> TrainingConfig:
    cfg.background_augmentation = False
    cfg.rgb_augmentation = False
    cfg.n_epochs_warmup = 1
    cfg.epoch_size = 10 * cfg.batch_size
    cfg.val_size = 10 * cfg.batch_size
    return cfg


if __name__ == "__main__":
    set_logging_level("debug")

    cli_cfg = OmegaConf.from_cli()
    logger.info(f"CLI config: \n {OmegaConf.to_yaml(cli_cfg)}")

    cfg: TrainingConfig = OmegaConf.structured(TrainingConfig)
    cfg.hardware = HardwareConfig(
        n_cpus=int(os.environ.get("N_CPUS", 10)),
        n_gpus=int(os.environ.get("WORLD_SIZE", 1)),
    )
    if "config_id" in cli_cfg:
        assert "resume_run_id" not in cli_cfg
        config_id = cli_cfg.config_id
        logger.info(f"{Fore.GREEN}Training with config: {config_id} {Style.RESET_ALL}")
        cfg = update_cfg_with_config_id(cfg, config_id)
        del cli_cfg.config_id
    elif "resume_run_id" in cli_cfg:
        logger.info(f"{Fore.RED}Resuming {cfg.resume_run_id} {Style.RESET_ALL}")
        resume_cfg = OmegaConf.load(EXP_DIR / cli_cfg.resume_run_id / "config.yaml")
        resume_cfg.hardware = cfg.hardware
        cfg = resume_cfg  # type: ignore
        del cli_cfg.resume_run_id
    else:
        assert "run_id" in cli_cfg
        del cli_cfg.run_id

    if cli_cfg.get("overfit", False):
        cfg = update_cfg_overfit(cfg)
        del cli_cfg.overfit

    if cli_cfg.get("debug", False):
        cfg = update_cfg_debug(cfg)
        del cli_cfg.debug

    cfg = OmegaConf.merge(
        cfg,
        cli_cfg,
    )

    assert cfg.run_id is not None
    logger.info(f"{Fore.GREEN}Running training of {cfg.run_id} ... {Style.RESET_ALL}")
    cfg.save_dir = str(EXP_DIR / cfg.run_id)
    train_megapose(cfg)
