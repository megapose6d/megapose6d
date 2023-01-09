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
from typing import Union

# MegaPose
# Backbones
import megapose.models.torchvision_resnet as models
from megapose.lib3d.rigid_mesh_database import BatchedMeshes

# Pose models
from megapose.models.pose_rigid import PosePredictor
from megapose.models.wide_resnet import WideResNet18, WideResNet34
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.training.training_config import TrainingConfig
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def check_update_config(cfg: TrainingConfig) -> TrainingConfig:
    """Useful for loading models previously trained with different configurations."""

    cfg.is_coarse_compat = False
    # Detect old coarse model definition
    if hasattr(cfg, "input_strategy") and cfg.input_strategy == "input=obs+one_render":
        cfg.is_coarse_compat = True
        cfg.n_rendered_views = 1
        cfg.multiview_type = "1view_TCO"
        cfg.predict_rendered_views_logits = True
        cfg.remove_TCO_rendering = True
        cfg.predict_pose_update = False

    if cfg.multiview_type == "front_3views":
        cfg.multiview_type = "TCO+front_3views"
    if cfg.multiview_type == "front_5views":
        cfg.multiview_type = "TCO+front_5views"
    if cfg.multiview_type == "front_1view":
        cfg.multiview_type = "TCO+front_1view"

    if not hasattr(cfg, "predict_pose_update"):
        cfg.predict_pose_update = True

    if not hasattr(cfg, "remove_TCO_rendering"):
        cfg.remove_TCO_rendering = False
    if not hasattr(cfg, "predict_rendered_views_logits"):
        cfg.predict_rendered_views_logits = False
    if "n_rendered_views" not in cfg:
        if "n_views" in cfg:
            cfg.n_rendered_views = cfg.n_views
            del cfg.n_views
        else:
            cfg.n_rendered_views = 1

    if "render_normals" not in cfg:
        cfg.render_normals = False
    if "render_depth" not in cfg:
        cfg.render_depth = False
    if "input_depth" not in cfg:
        cfg.input_depth = False
    if "multiview_type" not in cfg:
        cfg.multiview_type = "TCO"
        assert not cfg.remove_TCO_rendering
    if not "views_inplane_rotation" not in cfg:
        cfg.views_inplane_rotations = False
    if "depth_augmentation" not in cfg:
        cfg.depth_normalization_type = "tCR_scale"

    if "renderer" not in cfg:
        logger.info("Renderer is now Panda3D by default.")
        cfg.renderer = "panda3d"
    return cfg


def create_model_pose(
    cfg: TrainingConfig,
    renderer: Panda3dBatchRenderer,
    mesh_db: BatchedMeshes,
) -> PosePredictor:
    n_channels = 3
    n_normals_channels = 3 if cfg.render_normals else 0
    n_rendered_depth_channels = 1 if cfg.render_depth else 0
    n_depth_channels = 1 if cfg.input_depth else 0
    # Assumes that if you are rendering depth you are also
    # inputting it from the model
    n_inputs = (n_channels + n_depth_channels) + (
        (n_channels + n_normals_channels + n_rendered_depth_channels) * cfg.n_rendered_views
    )
    backbone_str = cfg.backbone_str
    render_size = (240, 320)
    if "vanilla_resnet34" == backbone_str:
        n_features = 512
        backbone = models.__dict__["resnet34"](num_classes=n_features, n_input_channels=n_inputs)
        backbone.n_features = n_features
    elif "resnet34" == backbone_str:
        backbone = WideResNet34(n_inputs=n_inputs)
    elif "resnet18" == backbone_str:
        backbone = WideResNet18(n_inputs=n_inputs)
    elif "resnet34_width=" in backbone_str:
        width = int(backbone_str.split("resnet34_width=")[1])
        backbone = WideResNet34(n_inputs=n_inputs, width=width)
    else:
        raise ValueError("Unknown backbone", backbone_str)

    logger.debug(f"Backbone: {backbone_str}")
    backbone.n_inputs = n_inputs
    model = PosePredictor(
        backbone=backbone,
        renderer=renderer,
        mesh_db=mesh_db,
        render_size=render_size,
        n_rendered_views=cfg.n_rendered_views,
        views_inplane_rotations=cfg.views_inplane_rotations,
        multiview_type=cfg.multiview_type,
        render_normals=cfg.render_normals,
        render_depth=cfg.render_depth,
        input_depth=cfg.input_depth,
        predict_rendered_views_logits=cfg.predict_rendered_views_logits,
        remove_TCO_rendering=cfg.remove_TCO_rendering,
        predict_pose_update=cfg.predict_pose_update,
        depth_normalization_type=cfg.depth_normalization_type,
    )
    return model
