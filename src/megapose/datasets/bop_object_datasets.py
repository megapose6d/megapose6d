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

# Third Party
import numpy as np

# MegaPose
from megapose.lib3d.symmetries import ContinuousSymmetry, DiscreteSymmetry

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


class BOPObjectDataset(RigidObjectDataset):
    def __init__(self, ds_dir: Path, label_format: str = "{label}"):
        infos_file = ds_dir / "models_info.json"
        infos = json.loads(infos_file.read_text())
        objects = []
        for obj_id, bop_info in infos.items():
            obj_id = int(obj_id)
            obj_label = f"obj_{obj_id:06d}"
            mesh_path = (ds_dir / obj_label).with_suffix(".ply").as_posix()
            symmetries_discrete = [
                DiscreteSymmetry(pose=np.array(x).reshape((4, 4)))
                for x in bop_info.get("symmetries_discrete", [])
            ]
            symmetries_continuous = [
                ContinuousSymmetry(offset=d["offset"], axis=d["axis"])
                for d in bop_info.get("symmetries_continuous", [])
            ]
            obj = RigidObject(
                label=label_format.format(label=obj_label),
                mesh_path=Path(mesh_path),
                mesh_units="mm",
                symmetries_discrete=symmetries_discrete,
                symmetries_continuous=symmetries_continuous,
                mesh_diameter=bop_info["diameter"],
            )
            objects.append(obj)

        self.ds_dir = ds_dir
        super().__init__(objects)
