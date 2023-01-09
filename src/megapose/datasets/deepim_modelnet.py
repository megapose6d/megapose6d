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

# Third Party
import numpy as np
import pandas as pd
import torch
from PIL import Image

# MegaPose
from megapose.datasets.utils import make_detections_from_segmentation
from megapose.lib3d.transform import Transform

# Local Folder
from .scene_dataset import (
    CameraData,
    ObjectData,
    ObservationInfos,
    SceneDataset,
    SceneObservation,
)


def parse_pose(pose_str: str) -> np.ndarray:
    pose_str_split = pose_str.split("\n")[-3:]
    pose = np.eye(4, dtype=float)
    for r in range(3):
        pose[r, :] = np.array(list(map(float, pose_str_split[r].split(" "))))
    return pose


class DeepImModelNetDataset(SceneDataset):
    def __init__(
        self,
        modelnet_dir: Path,
        category: str,
        split: str = "test",
        label_format: str = "{label}",
        n_objects: int = 70,
        n_images_per_object: int = 50,
        load_depth: bool = False,
    ):

        self.test_template_im = (
            modelnet_dir
            / "modelnet_render_v1/data/real/{category}/{split}/{obj_id}_{im_id:04d}-color.png"
        )
        self.test_template_depth = (
            modelnet_dir
            / "modelnet_render_v1/data/real/{category}/{split}/{obj_id}_{im_id:04d}-depth.png"
        )
        self.test_template_label = (
            modelnet_dir
            / "modelnet_render_v1/data/real/{category}/{split}/{obj_id}_{im_id:04d}-label.png"
        )
        self.test_template_pose = (
            modelnet_dir
            / "modelnet_render_v1/data/real/{category}/{split}/{obj_id}_{im_id:04d}-pose.txt"
        )

        self.init_template_im = (
            modelnet_dir
            / "modelnet_render_v1/data/rendered/{category}/{split}/{obj_id}_{im_id:04d}_0-color.png"
        )
        self.init_template_depth = (
            modelnet_dir
            / "modelnet_render_v1/data/rendered/{category}/{split}/{obj_id}_{im_id:04d}_0-depth.png"
        )
        self.init_template_label = (
            modelnet_dir
            / "modelnet_render_v1/data/rendered/{category}/{split}/{obj_id}_{im_id:04d}_0-label.png"
        )
        self.init_template_pose = (
            modelnet_dir
            / "modelnet_render_v1/data/rendered/{category}/{split}/{obj_id}_{im_id:04d}_0-pose.txt"
        )

        object_ids = (
            Path(modelnet_dir / "model_set" / f"{category}_{split}.txt")
            .read_text()
            .splitlines()[:n_objects]
        )

        self.category = category
        self.split = split

        scene_ids, view_ids = [], []
        for object_id in object_ids:
            for im_id in range(n_images_per_object):
                scene_ids.append(object_id)
                view_ids.append(im_id)
        frame_index = pd.DataFrame({"scene_id": scene_ids, "view_id": view_ids})
        self.ds_dir = modelnet_dir
        self.depth_im_scale = 1000.0
        self.label_format = label_format
        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
        )

    def _load_scene_observation(self, image_infos: ObservationInfos) -> SceneObservation:
        infos_dict = dict(
            category=self.category,
            split=self.split,
            obj_id=image_infos.scene_id,
            im_id=image_infos.view_id,
        )
        obj_label = image_infos.scene_id

        rgb = np.array(Image.open(str(self.test_template_im).format(**infos_dict)))

        if self.load_depth:
            depth = np.array(Image.open(str(self.test_template_depth).format(**infos_dict)))
            depth = torch.as_tensor(depth) / self.depth_im_scale
        else:
            depth = None

        segmentation = np.array(
            Image.open(str(self.test_template_label).format(**infos_dict)), dtype=np.int_
        )
        pose_str = Path(str(self.test_template_pose).format(**infos_dict)).read_text()
        pose = Transform(parse_pose(pose_str))
        init_pose_str = Path(str(self.init_template_pose).format(**infos_dict)).read_text()
        init_pose = Transform(parse_pose(init_pose_str))

        obj_label = self.label_format.format(label=obj_label)
        TWO = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        TWO_init = Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        dets = make_detections_from_segmentation(segmentation[None])[0]

        object_datas = [
            ObjectData(
                label=obj_label,
                TWO=TWO,
                TWO_init=TWO_init,
                visib_fract=1.0,
                unique_id=1,
                bbox_modal=dets[1],
            )
        ]

        K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
        camera_data = CameraData(
            TWC=pose.inverse(),
            TWC_init=init_pose.inverse(),
            K=K,
            resolution=rgb.shape[:2],
        )

        observation = SceneObservation(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_data=camera_data,
            infos=image_infos,
            object_datas=object_datas,
        )
        return observation
