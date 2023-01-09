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

# MegaPose
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset


class UrdfDataset(RigidObjectDataset):
    def __init__(self, ds_dir: Path, mesh_units: str = "m", label_format: str = "{label}"):
        objects = []
        for urdf_dir in ds_dir.iterdir():
            urdf_paths = list(urdf_dir.glob("*.urdf"))
            if len(urdf_paths) == 1:
                urdf_path = urdf_paths[0]
                label = urdf_dir.name
                label = label_format.format(label=label)
                objects.append(
                    RigidObject(label=label, mesh_path=urdf_path, mesh_units=mesh_units)
                )
        super().__init__(objects)
