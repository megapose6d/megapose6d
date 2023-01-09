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
import json
import pickle
import sys
from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# MegaPose
from megapose.config import BOP_TOOLKIT_DIR, MEMORY
from megapose.lib3d.transform import Transform
from megapose.utils.logging import get_logger

# Local Folder
from .scene_dataset import (
    CameraData,
    ObjectData,
    ObservationInfos,
    SceneDataset,
    SceneObservation,
)

sys.path.append(str(BOP_TOOLKIT_DIR))
# Third Party
from bop_toolkit_lib import inout  # noqa

sys.path = sys.path[:-1]


logger = get_logger(__name__)


def remap_bop_targets(targets):
    targets = targets.rename(columns={"im_id": "view_id"})
    targets["label"] = targets["obj_id"].apply(lambda x: f"obj_{x:06d}")
    return targets


def build_index_and_annotations(
    ds_dir: Path,
    split: str,
    save_file_frame_index=None,
    save_file_annotations=None,
    make_per_view_annotations=True,
):

    scene_ids, view_ids = [], []

    annotations = dict()
    base_dir = ds_dir / split

    for scene_dir in tqdm(base_dir.iterdir()):
        scene_id = scene_dir.name
        annotations_scene = dict()
        for f in ("scene_camera.json", "scene_gt_info.json", "scene_gt.json"):
            path = scene_dir / f
            if path.exists():
                annotations_scene[f.split(".")[0]] = json.loads(path.read_text())

        if not make_per_view_annotations:
            annotations[scene_id] = annotations_scene

        scene_annotation = annotations_scene
        for view_id in scene_annotation["scene_camera"].keys():
            if make_per_view_annotations:
                this_annotation = dict()
                this_annotation["camera"] = scene_annotation["scene_camera"][str(view_id)]
                if "scene_gt_info" in scene_annotation:
                    this_annotation["gt"] = scene_annotation["scene_gt"][str(view_id)]
                    this_annotation["gt_info"] = scene_annotation["scene_gt_info"][str(view_id)]
                annotation_dir = base_dir / scene_id / "per_view_annotations"
                annotation_dir.mkdir(exist_ok=True)
                (annotation_dir / f"view={view_id}.json").write_text(json.dumps(this_annotation))
            scene_ids.append(int(scene_id))
            view_ids.append(int(view_id))

    frame_index = pd.DataFrame({"scene_id": scene_ids, "view_id": view_ids})
    if save_file_frame_index:
        frame_index.to_feather(save_file_frame_index)
        frame_index = None
    if len(annotations) > 0 and save_file_annotations:
        save_file_annotations.write_bytes(pickle.dumps(annotations))
        save_file_annotations = None
    return frame_index, annotations


class BOPDataset(SceneDataset):
    """Read a dataset in the BOP format.
    See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md

    # TODO: Document whats happening with the per-view annotations.
    # TODO: Remove per-view annotations, recommend using WebDataset for performance ?
    """

    def __init__(
        self,
        ds_dir: Path,
        label_format: str,
        split: str = "train",
        load_depth: bool = False,
        use_raw_object_id: bool = False,
        allow_cache: bool = False,
        per_view_annotations: bool = False,
    ):

        self.ds_dir = ds_dir
        assert ds_dir.exists(), "Dataset does not exists."

        self.split = split
        self.base_dir = ds_dir / split

        logger.info("Loading/making index and annotations...")
        if allow_cache:
            save_file_index = self.ds_dir / f"index_{split}.feather"
            save_file_annotations = self.ds_dir / f"annotations_{split}.pkl"
            fn = MEMORY.cache(build_index_and_annotations)
            fn(
                ds_dir=ds_dir,
                save_file_frame_index=save_file_index,
                save_file_annotations=save_file_annotations,
                split=split,
                make_per_view_annotations=per_view_annotations,
            )
            frame_index = pd.read_feather(save_file_index).reset_index(drop=True)
            if not per_view_annotations:
                self.annotations = pickle.loads(save_file_annotations.read_bytes())
        else:
            frame_index, self.annotations = build_index_and_annotations(
                ds_dir, split, make_per_view_annotations=per_view_annotations
            )

        self.use_raw_object_id = use_raw_object_id
        self.label_format = label_format

        super().__init__(
            frame_index=frame_index,
            load_depth=load_depth,
            load_segmentation=True,
        )

    def _load_scene_observation(self, image_infos: ObservationInfos) -> SceneObservation:
        scene_id, view_id = image_infos.scene_id, image_infos.view_id
        view_id = int(view_id)
        view_id_str = f"{view_id:06d}"
        scene_id_str = f"{int(scene_id):06d}"
        scene_dir = self.base_dir / scene_id_str

        # All stored in self.annotations (basic, problem with shared memory)
        # TODO: Also change the pandas numpy arrays to np.string_ instead of np.object
        # See https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        this_annotation_path = scene_dir / "per_view_annotations" / f"view={str(view_id)}.json"
        if this_annotation_path.exists():
            this_annotation = json.loads(this_annotation_path.read_text())
            this_gt = this_annotation.get("gt")
            this_gt_info = this_annotation.get("gt_info")
            this_cam_info = this_annotation.get("camera")
        else:
            this_annotation = self.annotations[scene_id_str]
            this_scene_gt = this_annotation.get("scene_gt")
            this_cam_info = this_annotation["scene_camera"][str(view_id)]
            if this_scene_gt is not None and str(view_id) in this_scene_gt:
                this_gt = this_scene_gt[str(view_id)]
                this_scene_gt_info = this_annotation.get("scene_gt_info")
                this_gt_info = this_scene_gt_info[str(view_id)]
            else:
                this_gt = None
                this_gt_info = None

        rgb_dir = scene_dir / "rgb"
        if not rgb_dir.exists():
            rgb_dir = scene_dir / "gray"
        rgb_path = rgb_dir / f"{view_id_str}.png"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".jpg")
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".tif")

        rgb = np.array(Image.open(rgb_path))
        if rgb.ndim == 2:
            # NOTE: This is for ITODD's gray images
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]

        cam_annotation = this_cam_info
        if "cam_R_w2c" in cam_annotation:
            RCW = np.array(cam_annotation["cam_R_w2c"]).reshape(3, 3)
            tCW = np.array(cam_annotation["cam_t_w2c"]) * 0.001
            TCW = Transform(RCW, tCW)
        else:
            TCW = Transform(np.eye(3), np.zeros(3))
        K = np.array(cam_annotation["cam_K"]).reshape(3, 3)
        TWC = TCW.inverse()

        camera_data = CameraData(TWC=TWC, K=K, resolution=rgb.shape[:2])

        TWC = TCW.inverse()

        object_datas = []
        segmentation = np.zeros((h, w), dtype=np.uint32)
        if this_gt_info is not None:
            annotation = this_gt
            n_objects = len(annotation)
            visib = this_gt_info
            for n in range(n_objects):
                RCO = np.array(annotation[n]["cam_R_m2c"]).reshape(3, 3)
                tCO = np.array(annotation[n]["cam_t_m2c"]) * 0.001
                TCO = Transform(RCO, tCO)
                TWO = TWC * TCO
                if self.use_raw_object_id:
                    name = str(annotation[n]["obj_id"])
                else:
                    obj_id = annotation[n]["obj_id"]
                    name = f"obj_{int(obj_id):06d}"

                bbox_visib = np.array(visib[n]["bbox_visib"]).tolist()
                x, y, w, h = bbox_visib
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                bbox_visib = [x1, y1, x2, y2]

                bbox_obj = np.array(visib[n]["bbox_obj"]).tolist()
                x, y, w, h = bbox_obj
                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h
                bbox_obj = [x1, y1, x2, y2]

                label = self.label_format.format(label=name)
                object_data = ObjectData(
                    label=label,
                    TWO=TWO,
                    visib_fract=visib[n]["visib_fract"],
                    unique_id=n + 1,
                    bbox_modal=bbox_visib,
                    bbox_amodal=bbox_obj,
                )
                object_datas.append(object_data)

            mask_path = scene_dir / "mask_visib" / f"{view_id_str}_all.png"
            if mask_path.exists():
                segmentation = np.array(Image.open(mask_path), dtype=np.uint32)
            else:
                for n in range(n_objects):
                    binary_mask_n = np.array(
                        Image.open(scene_dir / "mask_visib" / f"{view_id_str}_{n:06d}.png")
                    )
                    segmentation[binary_mask_n == 255] = n + 1

        depth = None
        if self.load_depth:
            depth_path = scene_dir / "depth" / f"{view_id_str}.png"
            if not depth_path.exists():
                depth_path = depth_path.with_suffix(".tif")
            depth = np.array(inout.load_depth(depth_path))
            depth *= cam_annotation["depth_scale"] / 1000

        observation = SceneObservation(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            camera_data=camera_data,
            infos=image_infos,
            object_datas=object_datas,
        )
        return observation
