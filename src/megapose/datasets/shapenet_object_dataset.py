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
from pathlib import Path

# MegaPose
from megapose.config import MEMORY

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


class ShapeNetSynset:
    def __init__(self, synset_id, name):
        self.synset_id = synset_id
        self.name = name
        self.parents = []
        self.children = []
        self.models = []
        self.models_descendants = []


class ShapeNetModel:
    def __init__(self, synset_id, source_id):
        self.synset_id = synset_id
        self.source_id = source_id


@MEMORY.cache
def make_shapenet_infos(shapenet_dir, model_name):
    # TODO: This probably has issues / is poorly implemented and very slow
    shapenet_dir = Path(shapenet_dir)
    taxonomy_path = shapenet_dir / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text())

    synset_id_to_synset = dict()

    def get_synset(synset_id):
        if synset_id not in synset_id_to_synset:
            synset = ShapeNetSynset(synset_id, synset_dict["name"])
            synset_id_to_synset[synset_id] = synset
        else:
            synset = synset_id_to_synset[synset_id]
        return synset

    for synset_dict in taxonomy:
        synset_id = synset_dict["synsetId"]
        synset = get_synset(synset_id)
        for child_synset_id in synset_dict["children"]:
            child_synset = get_synset(child_synset_id)
            child_synset.parents.append(synset)

    def model_exists(model_dir):
        model_dir_ = model_dir / "models"
        return (model_dir_ / model_name).exists()

    for synset in synset_id_to_synset.values():
        synset_dir = shapenet_dir / synset.synset_id
        if synset_dir.exists():
            model_dirs = list(synset_dir.iterdir())
        else:
            model_dirs = []
        model_names = [model_dir.name for model_dir in model_dirs if model_exists(model_dir)]
        synset.models = model_names

    def get_descendants(synset):
        if len(synset.children) == 0:
            return synset.models
        else:
            return sum([get_descendants(child) for child in children])

    for synset in synset_id_to_synset.values():
        synset.models_descendants = get_descendants(synset)
    return list(synset_id_to_synset.values())


class ShapeNetObjectDataset(RigidObjectDataset):
    def __init__(
        self,
        shapenet_root: Path,
        split: str = "orig",
    ):
        self.shapenet_dir = shapenet_root / f"models_{split}"

        if split == "orig":
            model_name = "model_normalized.obj"
            ypr_offset_deg = (0.0, 0.0, 0.0)
        elif split == "panda3d_bam":
            model_name = "model_normalized_binormals.bam"
            ypr_offset_deg = (0.0, -90.0, 0.0)
        elif split == "pointcloud":
            model_name = "model_normalized_pointcloud.obj"
            ypr_offset_deg = (0.0, 0.0, 0.0)
        else:
            raise ValueError("split")

        synsets = make_shapenet_infos(self.shapenet_dir, model_name)
        main_synsets = [
            synset
            for synset in synsets
            if len(synset.parents) == 0 and len(synset.models_descendants) > 0
        ]
        objects = []

        for synset in main_synsets:

            for source_id in synset.models_descendants:
                model_path = (
                    self.shapenet_dir / synset.synset_id / source_id / "models" / model_name
                )
                label = f"shapenet_{synset.synset_id}_{source_id}"
                category = synset.name
                obj = RigidObject(
                    label=label,
                    category=category,
                    mesh_path=model_path,
                    scaling_factor=0.1,
                    ypr_offset_deg=ypr_offset_deg,
                )
                objects.append(obj)
        super().__init__(objects)
