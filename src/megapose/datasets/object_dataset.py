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
import itertools
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Third Party
import numpy as np

# MegaPose
from megapose.lib3d.symmetries import (
    ContinuousSymmetry,
    DiscreteSymmetry,
    make_symmetries_poses,
)


class RigidObject:
    def __init__(
        self,
        label: str,
        mesh_path: Path,
        category: Optional[str] = None,
        mesh_diameter: Optional[float] = None,
        mesh_units: str = "m",
        symmetries_discrete: List[DiscreteSymmetry] = [],
        symmetries_continuous: List[ContinuousSymmetry] = [],
        ypr_offset_deg: Tuple[float, float, float] = (0., 0., 0.),
        scaling_factor: float = 1.0,
        scaling_factor_mesh_units_to_meters: Optional[float] = None,
    ):
        """
        Args:
            label (str): A unique label to identify an object.
            mesh_path (Path): Path to a mesh. Multiple object types are supported.
                Please refer to downstream usage of this class for the supported formats.
                For example, when a `RigidObjectDataset`is passed to a `Panda3dSceneRenderer`,
                the user must ensure that the mesh can be loaded correctly.
            category (Optional[str], optional): Can be used to identify the object
                as one of a known category,  e.g. mug or shoes.  In the general case, an
                object does not need to belong to a category. The notion of category can also
                ambiguous. In this codebase, this is only used to parse the categories of the
                ShapeNet dataset in order to remove the instances that overlap with the test
                categories of the ModelNet dataset.
            mesh_diameter (Optional[float], optional): Diameter of the object, expressed
                the in unit of the meshes.
                This is useful for computing error some metrics like ADD<0.1d or ADD-S<0.1d.
            mesh_units (str, optional): Units in which the vertex positions are expressed.
                Can be `m`or `mm`, defaults to `m`. In the operations of this codebase,
                all mesh coordinates and poses must be expressed in meters.
                When an object is loaded, a scaling will be applied to the mesh
                to ensure its coordinates are in meters when in memory.
            symmetries_discrete (List[ContinuousSymmetry], optional):
                See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
            symmetries_continuous (List[DiscreteSymmetry], optional):
                See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
            ypr_offset_deg (np.ndarray, optional): A rotation offset applied to the mesh
                **only when loaded in Panda3D**. This can be useful to correct
                some mesh conventions where axes are flipped.
                Defaults to np.zeros(3, dtype=float).
            scaling_factor (float, optional): An extra scaling factor that can
                be applied to the mesh to rescale it. Defaults to 1.0. Please note
                that this is applied on top applying the scale factor to resize the
                mesh to meters.
                For example, if you have a mesh with coordinates expressed in `mm`
                which you want to resize to 10% of its size,
                you should pass `mesh_units=mm`and `scaling_factor=0.1`.
                Note that `mesh_units=m` and `scaling_factor=100` would be strictly equivalent.
            scaling_factor_mesh_units_to_meters (float, optional): Can be used
                instead of the mesh_units argument. This is the scale that converts
                mesh units to meters.
        """

        self.label = label
        self.category = category
        self.mesh_path = mesh_path
        self.mesh_units = mesh_units

        if scaling_factor_mesh_units_to_meters is not None:
            self.scaling_factor_mesh_units_to_meters = scaling_factor_mesh_units_to_meters
        else:
            self.scaling_factor_mesh_units_to_meters = {
                "m": 1.0,
                "mm": 0.001,
            }[self.mesh_units]
        self.scaling_factor = scaling_factor

        self._mesh_diameter = None
        self.diameter_meters = None

        if self._mesh_diameter is not None:
            self.mesh_diameter = mesh_diameter
            self.diameter_meters = mesh_diameter * self.scaling_factor_mesh_units_to_meters

        self.symmetries_discrete = symmetries_discrete
        self.symmetries_continuous = symmetries_continuous
        self.ypr_offset_deg = ypr_offset_deg

    @property
    def is_symmetric(self) -> bool:
        return len(self.symmetries_discrete) > 0 or len(self.symmetries_continuous) > 0

    @property
    def scale(self) -> float:
        """Returns the scale factor that converts the mesh to desired units."""
        return self.scaling_factor_mesh_units_to_meters * self.scaling_factor

    def make_symmetry_poses(
        self, n_symmetries_continuous: int = 64) -> np.ndarray:
        """Generates the set of object symmetries.

        Returns:
            (num_symmetries, 4, 4) array
        """
        return make_symmetries_poses(
            self.symmetries_discrete,
            self.symmetries_continuous,
            n_symmetries_continuous=n_symmetries_continuous,
            scale=self.scale,
        )


class RigidObjectDataset:
    def __init__(
        self,
        objects: List[RigidObject],
    ):
        self.list_objects = objects
        self.label_to_objects = {obj.label: obj for obj in objects}
        if len(self.list_objects) != len(self.label_to_objects):
            raise RuntimeError("There are objects with duplicate labels")

    def __getitem__(self, idx: int) -> RigidObject:
        return self.list_objects[idx]

    def get_object_by_label(self, label: str) -> RigidObject:
        return self.label_to_objects[label]

    def __len__(self) -> int:
        return len(self.list_objects)

    @property
    def objects(self) -> List[RigidObject]:
        """Returns a list of objects in this dataset."""
        return self.list_objects

    def filter_objects(self, keep_labels: Set[str]) -> "RigidObjectDataset":
        list_objects = [obj for obj in self.list_objects if obj.label in keep_labels]
        return RigidObjectDataset(list_objects)


def append_dataset_name_to_object_labels(
    ds_name: str, object_dataset: RigidObjectDataset
) -> RigidObjectDataset:
    for obj in object_dataset.list_objects:
        obj.label = f"ds_name={ds_name}_{obj.label}"
    return object_dataset


def concat_object_datasets(datasets: List[RigidObjectDataset]) -> RigidObjectDataset:
    objects = list(itertools.chain.from_iterable([ds.list_objects for ds in datasets]))
    return RigidObjectDataset(objects)
