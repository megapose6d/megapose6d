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

SCALE = 0.1
MODELS_DIR_TEMPLATE = "models_bop-renderer_scale={scale}"


def make_ply_scaled(mesh_path, scale=SCALE):
    n, mesh_path = mesh_path
    mesh_path = Path(mesh_path)
    new_mesh_path = str(mesh_path.with_suffix("")) + "_scaled.ply"
    new_mesh_path = new_mesh_path.replace("models_orig", MODELS_DIR_TEMPLATE.format(scale=scale))
    mesh = trimesh.load(str(mesh_path), skip_materials=True, process=False, maintain_order=True)
    mesh = as_mesh(mesh)
    mesh.apply_scale(scale)
    mesh.apply_scale(1000)
    Path(new_mesh_path).parent.mkdir(exist_ok=True, parents=True)
    mesh.export(str(new_mesh_path), encoding="ascii")
    print(n, new_mesh_path)


def make_ply_scaled_(mesh_path):
    p = Process(target=make_ply_scaled, args=(mesh_path,))
    p.start()
    p.join()


if __name__ == "__main__":
    trimesh.util.log.setLevel("ERROR")
    obj_dataset = make_object_dataset("shapenet.orig")
    target_models_dir = SHAPENET_DIR / MODELS_DIR_TEMPLATE.format(scale=SCALE)
    if target_models_dir.exists():
        shutil.rmtree(target_models_dir)
    target_models_dir.mkdir()
    shutil.copy((SHAPENET_DIR / "models_orig" / "taxonomy.json"), target_models_dir)
    n_procs = 20
    mesh_paths = []
    for n, obj in tqdm(enumerate(obj_dataset.objects)):
        mesh_path = Path(obj["mesh_path"])
        mesh_paths.append([n, mesh_path])

    for mesh_path in tqdm(mesh_paths):
        make_ply_scaled_(mesh_path)


    time.sleep(60)
