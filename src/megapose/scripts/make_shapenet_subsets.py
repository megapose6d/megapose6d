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
import pandas as pd
from tqdm import tqdm

# MegaPose
from megapose.config import SHAPENET_DIR
from megapose.datasets.datasets_cfg import make_object_dataset


def read_all_stats(obj_dataset):
    gltf_paths = []
    statistics = []
    for n, obj in tqdm(enumerate(obj_dataset.objects)):
        mesh_path = Path(obj["mesh_path"])
        infos = dict()
        stats_path = mesh_path.parent / "stats.json"
        pc_path = mesh_path.parent / "model_normalized_pointcloud.obj"
        infos["label"] = obj["label"]
        if stats_path.exists() and pc_path.exists():
            infos["is_gltf_correct"] = True
            infos.update(json.loads(stats_path.read_text()))
        else:
            infos["is_gltf_correct"] = False
            infos["tot_mem_kb"] = np.nan

        statistics.append(infos)
    statistics = pd.DataFrame(statistics)
    return statistics


def get_labels_split(statistics, max_model_mem_kb, max_tot_mem_kb):
    statistics = statistics.copy()
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    statistics = statistics.iloc[np.where(np.isfinite(statistics["tot_mem_kb"]))[0]]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    statistics = statistics.iloc[np.where(statistics["tot_mem_kb"] <= max_model_mem_kb)[0]]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)

    np_random = np.random.RandomState(0)
    statistics = statistics.iloc[np_random.permutation(np.arange(len(statistics)))]
    max_id = np.where(np.cumsum(statistics["tot_mem_kb"]) <= max_tot_mem_kb)[0][-1]
    statistics = statistics.iloc[:max_id]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    return statistics["label"].tolist()


def get_labels_split_max_objects(statistics, max_model_mem_kb, max_num_objects):
    statistics = statistics.copy()
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    statistics = statistics.iloc[np.where(np.isfinite(statistics["tot_mem_kb"]))[0]]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    statistics = statistics.iloc[np.where(statistics["tot_mem_kb"] <= max_model_mem_kb)[0]]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)

    np_random = np.random.RandomState(0)
    statistics = statistics.iloc[np_random.permutation(np.arange(len(statistics)))]
    max_id = max_num_objects
    statistics = statistics.iloc[:max_id]
    print(len(statistics), np.nansum(statistics["tot_mem_kb"]) / 1e6)
    return statistics["label"].tolist()


if __name__ == "__main__":
    ds_dir = SHAPENET_DIR
    ds_stats_path = ds_dir / "stats/stats_all_objects.json"

    statistics = pd.read_json(ds_stats_path)

    splits = [
        dict(
            name="shapenet_10mb_100",
            max_model_mem_kb=10e3,
            max_num_objects=100,
        ),
        dict(
            name="shapenet_10mb_1k",
            max_model_mem_kb=10e3,
            max_num_objects=1000,
        ),
        dict(
            name="shapenet_10mb_2k",
            max_model_mem_kb=10e3,
            max_num_objects=2500,
        ),
        dict(
            name="shapenet_10mb_5k",
            max_model_mem_kb=10e3,
            max_num_objects=5000,
        ),
        dict(
            name="shapenet_10mb_10k",
            max_model_mem_kb=10e3,
            max_num_objects=10000,
        ),
        dict(
            name="shapenet_10mb_15k",
            max_model_mem_kb=10e3,
            max_num_objects=15000,
        ),
        dict(
            name="shapenet_10mb_20k",
            max_model_mem_kb=10e3,
            max_num_objects=20000,
        ),
        dict(
            name="shapenet_100mb_200gb",
            max_model_mem_kb=100e3,
            max_tot_mem_kb=200e6,
        ),
        dict(
            name="shapenet_10mb_200gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=200e6,
        ),
        dict(
            name="shapenet_10mb_50gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=50e6,
        ),
        dict(
            name="shapenet_20mb_50gb",
            max_model_mem_kb=20e3,
            max_tot_mem_kb=50e6,
        ),
        dict(
            name="shapenet_10mb_100gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=100e6,
        ),
        dict(
            name="shapenet_10mb_32gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=32e6,
        ),
        dict(
            name="shapenet_2mb_32gb",
            max_model_mem_kb=2e3,
            max_tot_mem_kb=32e6,
        ),
        dict(
            name="shapenet_10mb_8gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=8e6,
        ),
        dict(
            name="shapenet_10mb_1gb",
            max_model_mem_kb=10e3,
            max_tot_mem_kb=1e6,
        ),
        dict(
            name="shapenet_2mb_1gb",
            max_model_mem_kb=2e3,
            max_tot_mem_kb=1e6,
        ),
    ]

    for split in splits:
        if "max_num_objects" in split:
            labels = get_labels_split_max_objects(
                statistics, split["max_model_mem_kb"], split["max_num_objects"]
            )
        else:
            labels = get_labels_split(
                statistics, split["max_model_mem_kb"], split["max_tot_mem_kb"]
            )
        split_path = (ds_dir / "stats" / split["name"]).with_suffix(".json")
        split_path.write_text(json.dumps(labels))
        print("wrote", split_path, len(labels))
