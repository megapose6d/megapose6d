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


# MegaPose
from megapose.models.mask_rcnn import DetectorMaskRCNN
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def check_update_config(cfg):
    obj_prefix = cfg.train_ds_names[0][0].split(".")[0]
    cfg.label_to_category_id = {f"{obj_prefix}-{k}": v for k, v in cfg.label_to_category_id.items()}
    return cfg


def create_model_detector(cfg, n_classes):
    model = DetectorMaskRCNN(
        input_resize=cfg.input_resize,
        n_classes=n_classes,
        backbone_str=cfg.backbone_str,
        anchor_sizes=cfg.anchor_sizes,
    )
    return model
