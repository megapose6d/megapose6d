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
from typing import List

# MegaPose
from megapose.config import MEMORY

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


@MEMORY.cache
def make_gso_infos(gso_dir: Path, model_name: str = "model.obj") -> List[str]:
    gso_dir = Path(gso_dir)
    models_dir = gso_dir.iterdir()
    invalid_ids = set(json.loads((gso_dir.parent / "invalid_meshes.json").read_text()))
    object_ids = []
    for model_dir in models_dir:
        if (model_dir / "meshes" / model_name).exists():
            object_id = model_dir.name
            if object_id not in invalid_ids:
                object_ids.append(object_id)
    object_ids.sort()
    return object_ids


class GoogleScannedObjectDataset(RigidObjectDataset):
    def __init__(self, gso_root: Path, split: str = "orig"):
        self.gso_dir = gso_root / f"models_{split}"

        if split == "orig":
            scaling_factor = 1.0
        elif split in {"normalized", "pointcloud"}:
            scaling_factor = 0.1

        object_ids = make_gso_infos(self.gso_dir)
        objects = []
        for object_id in object_ids:
            model_path = self.gso_dir / object_id / "meshes" / "model.obj"
            label = f"gso_{object_id}"
            obj = RigidObject(
                label=label,
                mesh_path=model_path,
                scaling_factor=scaling_factor,
            )
            objects.append(obj)
        super().__init__(objects)
