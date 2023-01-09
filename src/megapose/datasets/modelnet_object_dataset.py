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

# Local Folder
from .object_dataset import RigidObject, RigidObjectDataset


class ModelNetObjectDataset(RigidObjectDataset):
    def __init__(
        self,
        modelnet_dir: Path,
        category: str,
        split: str = "test",
        rescaled: bool = True,
        n_objects: int = 30,
    ):

        object_ids = (
            Path(modelnet_dir / "model_set" / f"{category}_{split}.txt")
            .read_text()
            .splitlines()[:n_objects]
        )

        objects = []
        for object_id in object_ids:
            if rescaled:
                mesh_path = (
                    modelnet_dir / "ModelNet40" / category / split / f"{object_id}_rescaled.obj"
                )
            else:
                mesh_path = modelnet_dir / "ModelNet40" / category / split / f"{object_id}.obj"
            obj = RigidObject(
                label=object_id,
                category=category,
                mesh_path=mesh_path,
                scaling_factor=0.1,
            )
            objects.append(obj)
        super().__init__(objects=objects)
