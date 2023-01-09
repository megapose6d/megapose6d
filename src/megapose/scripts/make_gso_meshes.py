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
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from multiprocessing import Process, Queue
from pathlib import Path

# Third Party
import numpy as np
import trimesh
from tqdm import tqdm

# MegaPose
from megapose.config import (
    GSO_DIR,
    GSO_NORMALIZED_DIR,
    GSO_ORIG_DIR,
    GSO_POINTCLOUD_DIR,
    GSO_SCALED_DIR,
)
from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.lib3d.rigid_mesh_database import as_mesh

SCALE = 0.1
SCALED_MODELS_DIR_TEMPLATE = "models_bop-renderer_scale={scale}"
NORMALIZED_MODELS_DIR = "models_normalized"
PANDA3D_MODELS_DIR = "panda3d"


def rescale_mesh(mesh_path):
    text = Path(mesh_path).read_text()
    lines = text.splitlines()

    elements = defaultdict(list)

    for line in lines:
        line_split = line.split(" ")
        elements[line_split[0]].append(line)

    vertices = []
    for line in elements["v"]:
        vertices.append(list(map(float, line.split(" ")[1:])))
    vertices = np.array(vertices)

    normals = []
    for line in elements["vn"]:
        normals.append(list(map(float, line.split(" ")[1:])))
    normals = np.array(normals)
    n_normals = len(normals)

    face_vertex_ids = []
    new_faces = []
    for face in elements["f"]:
        face_line = face.split(" ")
        new_face_line = deepcopy(face_line)
        for n_idx in (1, 2, 3):
            face_elements = new_face_line[n_idx].split("/")
            face_elements[1] = str(int(face_elements[1]) + n_normals)
            face_vertex_ids.append(int(face_elements[0]))
            new_face_line[n_idx] = "/".join(face_elements)
        new_faces.append(" ".join(new_face_line))
    face_vertex_ids = np.array(face_vertex_ids) - 1

    positions = vertices[face_vertex_ids]
    xmin, xmax = float(positions[:, 0].min()), float(positions[:, 0].max())
    ymin, ymax = float(positions[:, 1].min()), float(positions[:, 1].max())
    zmin, zmax = float(positions[:, 2].min()), float(positions[:, 2].max())
    scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin) / 2.0

    vertices[:, 0] -= (xmax + xmin) / 2.0
    vertices[:, 1] -= (ymax + ymin) / 2.0
    vertices[:, 2] -= (zmax + zmin) / 2.0
    vertices[:, :3] /= scale

    out = elements["mtllib"][0]


    faces = elements["faces"]

    text = elements["mtllib"][0]
    text += "\n\n"
    for vertex_line in vertices.tolist():
        line = ["v"] + list(map(str, vertex_line))
        text += " ".join(line)
        text += "\n"

    text += "\n"
    for normal_line in normals.tolist():
        line = ["vn"] + list(map(str, normal_line))
        text += " ".join(line)
        text += "\n"

    text += "\n\n"
    text += elements["usemtl"][0]
    text += "\n"

    for vt_line in elements["vt"]:
        text += vt_line
        text += "\n"

    for f_line in elements["f"]:
        text += f_line
        text += "\n"
    return text


def make_obj_normalized(obj_id):
    mesh_dir = Path(GSO_ORIG_DIR) / obj_id / "meshes"
    new_mesh_dir = Path(GSO_NORMALIZED_DIR) / obj_id / "meshes"
    new_mesh_dir.mkdir(exist_ok=True, parents=True)
    for f in ("model.mtl", "texture.png"):
        if (mesh_dir / f).exists():
            shutil.copy(mesh_dir / f, new_mesh_dir / f)
    mesh = rescale_mesh(mesh_dir / "model.obj")
    (new_mesh_dir / "model.obj").write_text(mesh)


def make_ply_scaled(obj_id, scale=SCALE):
    mesh_dir = Path(GSO_NORMALIZED_DIR) / obj_id / "meshes"
    new_mesh_dir = Path(GSO_SCALED_DIR) / obj_id / "meshes"
    new_mesh_path = new_mesh_dir / "model.ply"
    mesh = trimesh.load(
        str(mesh_dir / "model.obj"), skip_materials=True, process=False, maintain_order=True
    )
    mesh = as_mesh(mesh)
    mesh.apply_scale(scale)
    mesh.apply_scale(1000)
    Path(new_mesh_path).parent.mkdir(exist_ok=True, parents=True)
    mesh.export(str(new_mesh_path), encoding="ascii")


def make_obj_pc(obj_id):
    n_points = 2000
    mesh_dir = Path(GSO_NORMALIZED_DIR) / obj_id / "meshes"
    new_mesh_dir = Path(GSO_POINTCLOUD_DIR) / obj_id / "meshes"
    new_mesh_path = new_mesh_dir / "model.obj"
    mesh = trimesh.load(
        str(mesh_dir / "model.obj"), skip_materials=True, process=False, maintain_order=True
    )
    mesh = as_mesh(mesh)
    points = trimesh.sample.sample_surface(mesh, n_points)[0]
    mesh_pc = trimesh.PointCloud(points)
    Path(new_mesh_path).parent.mkdir(exist_ok=True, parents=True)
    mesh_pc.export(str(new_mesh_path))


if __name__ == "__main__":
    trimesh.util.log.setLevel("ERROR")
    obj_dataset = make_object_dataset("gso.orig")
    for n, obj in tqdm(enumerate(obj_dataset.objects)):
        obj_id = obj["label"].split("gso_")[1]
        make_obj_normalized(obj_id)
        make_ply_scaled(obj_id)
        make_obj_pc(obj_id)
