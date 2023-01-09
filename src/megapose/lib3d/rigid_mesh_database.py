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
from copy import deepcopy
from typing import List

# Third Party
import numpy as np
import torch
import trimesh

# MegaPose
from megapose.datasets.object_dataset import RigidObject
from megapose.lib3d.mesh_ops import get_meshes_bounding_boxes, sample_points
from megapose.lib3d.symmetries import make_symmetries_poses
from megapose.utils.tensor_collection import TensorCollection


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        mesh = scene_or_mesh
    return mesh


class MeshDataBase:
    def __init__(self, obj_list: List[RigidObject]):
        self.obj_dict = {obj.label: obj for obj in obj_list}
        self.obj_list = obj_list
        self.infos = {obj.label: dict() for obj in obj_list}
        self.meshes = {
            l: as_mesh(
                trimesh.load(
                    obj.mesh_path,
                    group_material=False,
                    process=False,
                    skip_materials=True,
                    maintain_order=True,
                )
            )
            for l, obj in self.obj_dict.items()
        }

        for label, obj in self.obj_dict.items():
            if obj.diameter_meters is None:

                mesh = self.meshes[label]
                points = np.array(mesh.vertices) * obj.scale
                extent = points.max(0) - points.min(0)
                diameter = np.linalg.norm(extent)

                obj.diameter_meters = diameter

    @staticmethod
    def from_object_ds(object_ds):
        obj_list = [object_ds[n] for n in range(len(object_ds))]
        return MeshDataBase(obj_list)

    def batched(self, aabb=False, resample_n_points=None, n_sym=64):
        if aabb:
            assert resample_n_points is None

        labels, points, symmetries = [], [], []
        new_infos = deepcopy(self.infos)
        for label, mesh in self.meshes.items():
            if aabb:
                points_n = get_meshes_bounding_boxes(torch.as_tensor(mesh.vertices).unsqueeze(0))[0]
            elif resample_n_points:
                if isinstance(mesh, trimesh.PointCloud):
                    points_n = sample_points(
                        torch.as_tensor(mesh.vertices).unsqueeze(0),
                        resample_n_points,
                        deterministic=True,
                    )[0]
                else:
                    points_n = torch.tensor(
                        trimesh.sample.sample_surface(mesh, resample_n_points)[0]
                    )
            else:
                points_n = torch.tensor(mesh.vertices)

            mesh_obj: RigidObject = self.obj_dict[label]
            points_n = points_n.clone()
            points_n *= mesh_obj.scale

            symmetries_n = mesh_obj.make_symmetry_poses(n_symmetries_continuous=n_sym)

            # QUESTION (lmanuelli): Is this used anywhere?
            new_infos[label]["n_points"] = points_n.shape[0]
            new_infos[label]["n_sym"] = symmetries_n.shape[0]

            symmetries.append(torch.as_tensor(symmetries_n))
            points.append(torch.as_tensor(points_n))
            labels.append(label)

        labels = np.array(labels)
        points = pad_stack_tensors(points, fill="select_random", deterministic=True)
        symmetries = pad_stack_tensors(symmetries, fill=torch.eye(4), deterministic=True)
        return BatchedMeshes(new_infos, labels, points, symmetries).float()


class BatchedMeshes(TensorCollection):
    def __init__(self, infos, labels, points, symmetries):
        # QUESTION (lmanuelli): What is `infos` supposed to be?
        super().__init__()
        self.infos = infos
        self.label_to_id = {label: n for n, label in enumerate(labels)}
        self.labels = np.asarray(labels)
        self.register_tensor("points", points)
        self.register_tensor("symmetries", symmetries)

    @property
    def n_sym_mapping(self):
        return {label: obj["n_sym"] for label, obj in self.infos.items()}

    def select(self, labels):
        ids = [self.label_to_id[l] for l in labels]
        return Meshes(
            infos=[self.infos[l] for l in labels],
            labels=self.labels[ids],
            points=self.points[ids],
            symmetries=self.symmetries[ids],
        )


class Meshes(TensorCollection):
    def __init__(self, infos, labels, points, symmetries):
        super().__init__()
        self.infos = infos
        self.labels = np.asarray(labels)
        self.register_tensor("points", points)
        self.register_tensor("symmetries", symmetries)

    def select_labels(self, labels):
        raise NotImplementedError

    def sample_points(self, n_points, deterministic=False):
        return sample_points(self.points, n_points, deterministic=deterministic)


def pad_stack_tensors(tensor_list, fill="select_random", deterministic=True):
    n_max = max([t.shape[0] for t in tensor_list])
    if deterministic:
        np_random = np.random.RandomState(0)
    else:
        np_random = np.random
    tensor_list_padded = []
    for tensor_n in tensor_list:
        n_pad = n_max - len(tensor_n)

        if n_pad > 0:
            if isinstance(fill, torch.Tensor):
                assert isinstance(fill, torch.Tensor)
                assert fill.shape == tensor_n.shape[1:]
                pad = (
                    fill.unsqueeze(0)
                    .repeat(n_pad, *[1 for _ in fill.shape])
                    .to(tensor_n.device)
                    .to(tensor_n.dtype)
                )
            else:
                assert fill == "select_random"
                ids_pad = np_random.choice(np.arange(len(tensor_n)), size=n_pad)
                pad = tensor_n[ids_pad]
            tensor_n_padded = torch.cat((tensor_n, pad), dim=0)
        else:
            tensor_n_padded = tensor_n
        tensor_list_padded.append(tensor_n_padded)
    return torch.stack(tensor_list_padded)
