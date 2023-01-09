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
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Process, Queue
from pathlib import Path

# Third Party
import trimesh
from tqdm import tqdm

# MegaPose
from megapose.config import SHAPENET_DIR
from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.lib3d.rigid_mesh_database import as_mesh

SPLIT_NAME = "models_pointcloud"
TARGETS_MODEL_DIR = Path(str(SHAPENET_DIR)) / SPLIT_NAME


def make_obj_pc(mesh_path):
    n_points = 2000
    n, mesh_path = mesh_path
    print(n, mesh_path)
    mesh_path = Path(mesh_path)
    new_mesh_path = str(mesh_path.with_suffix("")) + "_pointcloud.obj"
    new_mesh_path = new_mesh_path.replace("models_orig", SPLIT_NAME)

    mesh = trimesh.load(
        mesh_path, group_material=False, process=False, skip_materials=True, maintain_order=True
    )
    mesh = as_mesh(mesh)
    points = trimesh.sample.sample_surface(mesh, n_points)[0]
    mesh_pc = trimesh.PointCloud(points)
    Path(new_mesh_path).parent.mkdir(exist_ok=True, parents=True)
    mesh_pc.export(new_mesh_path)


def make_obj_pc_(mesh_path):
    p = Process(target=make_obj_pc, args=(mesh_path,))
    p.start()
    p.join()


if __name__ == "__main__":
    trimesh.util.log.setLevel("ERROR")
    obj_dataset = make_object_dataset("shapenet.orig")
    print(TARGETS_MODEL_DIR)
    if TARGETS_MODEL_DIR.exists():
        shutil.rmtree(TARGETS_MODEL_DIR)
    TARGETS_MODEL_DIR.mkdir()
    shutil.copy(
        (SHAPENET_DIR / "models_orig" / "taxonomy.json"), TARGETS_MODEL_DIR / "taxonomy.json"
    )
    n_procs = 20
    mesh_paths = []
    for n, obj in tqdm(enumerate(obj_dataset.objects)):
        mesh_path = Path(obj["mesh_path"])
        mesh_paths.append([n, mesh_path])

    for mesh_path in tqdm(mesh_paths):
        make_obj_pc_(mesh_path)

    time.sleep(60)
