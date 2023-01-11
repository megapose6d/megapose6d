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
import os
from pathlib import Path

# Third Party
import pandas as pd
from joblib import Memory

# MegaPose
import megapose

PROJECT_ROOT = Path(megapose.__file__).parent.parent.parent
PROJECT_DIR = PROJECT_ROOT
LOCAL_DATA_DIR = Path(os.environ.get("MEGAPOSE_DATA_DIR", Path(PROJECT_DIR) / "local_data"))
BOP_DS_DIR = LOCAL_DATA_DIR / "bop_datasets"
NB_DATA_DIR = LOCAL_DATA_DIR / "notebook_data"
SHAPENET_DIR = LOCAL_DATA_DIR / "shapenetcorev2"
WDS_DS_DIR = LOCAL_DATA_DIR / "webdatasets"

BOP_TOOLKIT_DIR = PROJECT_ROOT / "deps" / "bop_toolkit_challenge"
BLENDER_PBR_DS_DIR = LOCAL_DATA_DIR / "blender_pbr_datasets"
CC_TEXTURE_FOLDER = str(LOCAL_DATA_DIR / "cctextures")
BLENDER_PROC_DIR = PROJECT_DIR.parent / "blenderproc"
BLENDER_VERSION = "blender-2.93.8-linux-x64"
BLENDER_INSTALL_DIR = LOCAL_DATA_DIR / BLENDER_VERSION
if not BLENDER_INSTALL_DIR.exists():
    BLENDER_INSTALL_DIR = Path(os.environ["HOME"]) / BLENDER_VERSION
PYTHON_BIN_PATH = Path(os.environ["CONDA_PREFIX"]) / "bin/python"

BOP_PANDA3D_DS_DIR = LOCAL_DATA_DIR / "bop_models_panda3d"

GSO_DIR = LOCAL_DATA_DIR / "google_scanned_objects"
GSO_ORIG_DIR = GSO_DIR / "models_orig"
GSO_NORMALIZED_DIR = GSO_DIR / "models_normalized"
GSO_SCALED_DIR = GSO_DIR / "models_bop-renderer_scale=0.1"
GSO_POINTCLOUD_DIR = GSO_DIR / "models_pointcloud"
GSO_SCALE = 0.1

EXP_DIR = LOCAL_DATA_DIR / "experiments"
RESULTS_DIR = LOCAL_DATA_DIR / "results"
DEBUG_RESULTS_DIR = LOCAL_DATA_DIR / "debug/results"
DEBUG_DATA_DIR = LOCAL_DATA_DIR / "debug_data"
CACHE_DIR = LOCAL_DATA_DIR / "joblib_cache"

assert LOCAL_DATA_DIR.exists()
EXP_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DEBUG_DATA_DIR.mkdir(exist_ok=True)
NB_DATA_DIR.mkdir(exist_ok=True)
BLENDER_PBR_DS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

MEMORY = Memory(CACHE_DIR, verbose=2)


BOP_DS_NAMES_ROOT = [
    "lm",
    "lmo",
    "tless",
    "tudl",
    "icbin",
    "itodd",
    "hb",
    "ycbv",
    "hope",
]

MODELNET_TEST_CATEGORIES = [
    "bathtub",
    "bookshelf",
    "guitar",
    "range_hood",
    "sofa",
    "wardrobe",
    "tv_stand",
]

SHAPENET_MODELNET_CATEGORIES = set(
    [
        "guitar",
        "bathtub,bathing tub,bath,tub",
        "bookshelf",
        "sofa,couch,lounge",
    ]
)

YCBV_OBJECT_NAMES = [
    ["obj_000001", "01_master_chef_can"],
    ["obj_000002", "02_cracker_box"],
    ["obj_000003", "03_sugar_box"],
    ["obj_000004", "04_tomatoe_soup_can"],
    ["obj_000005", "05_mustard_bottle"],
    ["obj_000006", "06_tuna_fish_can"],
    ["obj_000007", "07_pudding_box"],
    ["obj_000008", "08_gelatin_box"],
    ["obj_000009", "09_potted_meat_can"],
    ["obj_000010", "10_banana"],
    ["obj_000011", "11_pitcher_base"],
    ["obj_000012", "12_bleach_cleanser"],
    ["obj_000013", "13_bowl"],
    ["obj_000014", "14_mug"],
    ["obj_000015", "15_power_drill"],
    ["obj_000016", "16_wood_block"],
    ["obj_000017", "17_scissors"],
    ["obj_000018", "18_large_marker"],
    ["obj_000019", "19_large_clamp"],
    ["obj_000020", "20_extra_large_clamp"],
    ["obj_000021", "21_foam_brick"],
]


YCBV_SYMMETRIC_OBJECTS = ["obj_000013"]

YCBV_OBJECT_NAMES = pd.DataFrame(YCBV_OBJECT_NAMES, columns=["label", "description"])


LMO_OBJECT_NAMES = [
    ["obj_000001", "monkey"],
    ["obj_000005", "watering can"],
    ["obj_000006", "kitty"],
    ["obj_000008", "drill"],
    ["obj_000009", "duck"],
    ["obj_000010", "egg carton"],
    ["obj_000011", "bottle"],
    ["obj_000012", "whole punch"],
]
