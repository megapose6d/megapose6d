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
import io
import json
from concurrent.futures import ProcessPoolExecutor as Pool
from contextlib import redirect_stdout
from multiprocessing import Process, Queue
from pathlib import Path

# Third Party
import numpy as np
from tqdm import tqdm

# MegaPose
from megapose.config import SHAPENET_DIR
from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.panda3d_renderer.panda3d_scene_renderer import App


def measure_memory(gltf_path):
    app = App()
    obj_node = app.loader.load_model(gltf_path, noCache=True)
    f = io.StringIO()
    with redirect_stdout(f):
        obj_node.analyze()
    s = f.getvalue()
    s = s.splitlines()
    mems = []
    for l in s:
        if "GeomVertexData arrays occupy" in l:
            print(l)
            l_ = l.split(" ")
            idx = [n for n, w in enumerate(l_) if w == "occupy"][0]
            mems.append(float(l_[idx + 1]))
        elif "GeomPrimitive arrays occupy" in l:
            print(l)
            l_ = l.split(" ")
            idx = [n for n, w in enumerate(l_) if w == "occupy"][0]
            mems.append(float(l_[idx + 1]))
        elif "texture memory required" in l:
            print(l)
            l_ = l.split(" ")
            idx = [n for n, w in enumerate(l_) if w == "minimum"][0]
            mems.append(float(l_[idx + 1]))
    tot_mem_kb = sum(mems)
    stats = dict(
        tot_mem_kb=tot_mem_kb,
    )
    (gltf_path.parent / "stats.json").write_text(json.dumps(stats))
    return


def measure_memory_(gltf_path):
    p = Process(target=measure_memory, args=(gltf_path,))
    p.start()
    p.join()


if __name__ == "__main__":
    panda3d_obj_dataset = make_object_dataset("shapenet.panda3d_bam")
    panda3d_map = {obj["label"]: obj for obj in panda3d_obj_dataset.objects}
    panda3d_objects = set(list(panda3d_map.keys()))
    pc_obj_dataset = make_object_dataset("shapenet.pointcloud")
    pc_map = {obj["label"]: obj for obj in pc_obj_dataset.objects}
    pc_objects = set(list(pc_map.keys()))
    vanilla_obj_dataset = make_object_dataset("shapenet.orig")
    vanilla_objects = set([obj["label"] for obj in vanilla_obj_dataset.objects])
    stats = []
    for n, obj in enumerate(tqdm(vanilla_obj_dataset.objects)):
        stats_ = dict()
        label = obj["label"]
        stats_["label"] = label
        stats_["has_pointcloud"] = label in pc_objects
        stats_["has_panda3d"] = label in panda3d_objects
        if stats_["has_panda3d"] and stats_["has_pointcloud"]:
            panda3d_obj_dir = Path(panda3d_map[label]["mesh_path"]).parent
            tot_mem_kb = sum(
                [f.stat().st_size / 1024 for f in panda3d_obj_dir.iterdir() if f.is_file()]
            )
        else:
            tot_mem_kb = np.nan
        stats_["tot_mem_kb"] = tot_mem_kb
        stats.append(stats_)

    stats_dir = SHAPENET_DIR / "stats"
    stats_dir.mkdir(exist_ok=True)
    (stats_dir / "stats_all_objects.json").write_text(json.dumps(stats))
