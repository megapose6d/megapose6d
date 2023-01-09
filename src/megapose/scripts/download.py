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
import argparse
import logging
import subprocess
import zipfile
from pathlib import Path

# Third Party
import wget

# MegaPose
from megapose.bop_config import (
    PBR_COARSE,
    PBR_DETECTORS,
    PBR_REFINER,
    SYNT_REAL_COARSE,
    SYNT_REAL_DETECTORS,
    SYNT_REAL_REFINER,
)
from megapose.config import BOP_DS_DIR, LOCAL_DATA_DIR, PROJECT_DIR
from megapose.utils.logging import get_logger

logger = get_logger(__name__)

RCLONE_CFG_PATH = PROJECT_DIR / "megapose" / "rclone.conf"
RCLONE_ROOT = "megapose:"
DOWNLOAD_DIR = LOCAL_DATA_DIR / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

BOP_SRC = "https://bop.felk.cvut.cz/media/data/bop_datasets/"
BOP_DATASETS = {
    "hope": {"test_splits": ["test_all", "val"]},
    "ycbv": {
        "test_splits": ["test_all"],
        "train_splits": ["train_real", "train_synt"],
    },
    "tless": {
        "test_splits": [
            "test_primesense_all",
        ],
        "train_splits": [
            "train_primesense",
        ],
    },
    "hb": {
        "test_splits": ["test_primesense_all", "val_primesense"],
    },
    "icbin": {
        "test_splits": ["test_all"],
    },
    "itodd": {
        "test_splits": ["val", "test_all"],
    },
    "lm": {
        "test_splits": ["test_all"],
    },
    "lmo": {
        "test_splits": ["test_all"],
        "has_pbr": False,
    },
    "tudl": {
        "test_splits": [
            "test_all",
        ],
        "train_splits": [
            "train_real",
        ],
    },
    "ruapc": {
        "test_splits": [
            "test_all",
        ]
    },
    "tyol": {
        "test_splits": [
            "test_all",
        ]
    },
}

BOP_DS_NAMES = list(BOP_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser("Megapose download utility")
    parser.add_argument("--bop_dataset", default="", type=str, choices=BOP_DS_NAMES)
    parser.add_argument("--bop_src", default="bop", type=str, choices=["bop", "gdrive"])
    parser.add_argument("--bop_extra_files", default="", type=str, choices=["ycbv", "tless"])
    parser.add_argument("--model", default="", type=str)
    parser.add_argument("--urdf_models", default="", type=str)
    parser.add_argument("--ycbv_compat_models", action="store_true")
    parser.add_argument("--texture_dataset", action="store_true")
    parser.add_argument("--result_id", default="", type=str)
    parser.add_argument("--bop_result_id", default="", type=str)
    parser.add_argument("--synt_dataset", default="", type=str)
    parser.add_argument("--detections", default="", type=str)
    parser.add_argument("--pbr_training_images", action="store_true")
    parser.add_argument("--train_splits", action="store_true")
    parser.add_argument("--all_bop20_results", action="store_true")
    parser.add_argument("--all_bop20_models", action="store_true")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.bop_dataset:
        if args.bop_src == "bop":
            download_bop_original(
                args.bop_dataset,
                args.pbr_training_images and BOP_DATASETS[args.bop_dataset].get("has_pbr", True),
                args.train_splits,
            )
        elif args.bop_src == "gdrive":
            download_bop_gdrive(args.bop_dataset)

    if args.bop_extra_files:
        if args.bop_extra_files == "tless":
            # https://github.com/kirumang/Pix2Pose#download-pre-trained-weights
            gdrive_download(f"bop_datasets/tless/all_target_tless.json", BOP_DS_DIR / "tless")
        elif args.bop_extra_files == "ycbv":
            # Friendly names used with YCB-Video
            gdrive_download(f"bop_datasets/ycbv/ycbv_friendly_names.txt", BOP_DS_DIR / "ycbv")
            # Offsets between YCB-Video and BOP (extracted from BOP readme)
            gdrive_download(f"bop_datasets/ycbv/offsets.txt", BOP_DS_DIR / "ycbv")
            # Evaluation models for YCB-Video (used by other works)
            gdrive_download(f"bop_datasets/ycbv/models_original", BOP_DS_DIR / "ycbv")
            # Keyframe definition
            gdrive_download(f"bop_datasets/ycbv/keyframe.txt", BOP_DS_DIR / "ycbv")

    if args.urdf_models:
        gdrive_download(f"urdfs/{args.urdf_models}", LOCAL_DATA_DIR / "urdfs")

    if args.ycbv_compat_models:
        gdrive_download(f"bop_datasets/ycbv/models_bop-compat", BOP_DS_DIR / "ycbv")
        gdrive_download(f"bop_datasets/ycbv/models_bop-compat_eval", BOP_DS_DIR / "ycbv")

    if args.model:
        gdrive_download(f"experiments/{args.model}", LOCAL_DATA_DIR / "experiments")

    if args.detections:
        gdrive_download(
            f"saved_detections/{args.detections}.pkl", LOCAL_DATA_DIR / "saved_detections"
        )

    if args.result_id:
        gdrive_download(f"results/{args.result_id}", LOCAL_DATA_DIR / "results")

    if args.bop_result_id:
        csv_name = args.bop_result_id + ".csv"
        gdrive_download(f"bop_predictions/{csv_name}", LOCAL_DATA_DIR / "bop_predictions")
        gdrive_download(
            f"bop_eval_outputs/{args.bop_result_id}", LOCAL_DATA_DIR / "bop_predictions"
        )

    if args.texture_dataset:
        gdrive_download("zip_files/textures.zip", DOWNLOAD_DIR)
        logger.info("Extracting textures ...")
        zipfile.ZipFile(DOWNLOAD_DIR / "textures.zip").extractall(
            LOCAL_DATA_DIR / "texture_datasets"
        )

    if args.synt_dataset:
        zip_name = f"{args.synt_dataset}.zip"
        gdrive_download(f"zip_files/{zip_name}", DOWNLOAD_DIR)
        logger.info("Extracting textures ...")
        zipfile.ZipFile(DOWNLOAD_DIR / zip_name).extractall(LOCAL_DATA_DIR / "synt_datasets")

    if args.all_bop20_models:
        for model_dict in (
            PBR_DETECTORS,
            PBR_COARSE,
            PBR_REFINER,
            SYNT_REAL_DETECTORS,
            SYNT_REAL_COARSE,
            SYNT_REAL_REFINER,
        ):
            for model in model_dict.values():
                gdrive_download(f"experiments/{model}", LOCAL_DATA_DIR / "experiments")

    if args.all_bop20_results:
        # SRL
        # Third Party
        from nerfpose.bop_config import (
            PBR_INFERENCE_ID,
            SYNT_REAL_4VIEWS_INFERENCE_ID,
            SYNT_REAL_8VIEWS_INFERENCE_ID,
            SYNT_REAL_ICP_INFERENCE_ID,
            SYNT_REAL_INFERENCE_ID,
        )

        for result_id in (
            PBR_INFERENCE_ID,
            SYNT_REAL_INFERENCE_ID,
            SYNT_REAL_ICP_INFERENCE_ID,
            SYNT_REAL_4VIEWS_INFERENCE_ID,
            SYNT_REAL_8VIEWS_INFERENCE_ID,
        ):
            gdrive_download(f"results/{result_id}", LOCAL_DATA_DIR / "results")


def run_rclone(cmd, args, flags):
    rclone_cmd = ["rclone", cmd] + args + flags + ["--config", str(RCLONE_CFG_PATH)]
    logger.debug(" ".join(rclone_cmd))
    subprocess.run(rclone_cmd)


def gdrive_download(gdrive_path, local_path):
    gdrive_path = Path(gdrive_path)
    if gdrive_path.name != local_path.name:
        local_path = local_path / gdrive_path.name
    rclone_path = RCLONE_ROOT + str(gdrive_path)
    local_path = str(local_path)
    logger.info(f"Copying {rclone_path} to {local_path}")
    run_rclone("copyto", [rclone_path, local_path], flags=["-P"])


def download_bop_original(ds_name, download_pbr, download_train):
    filename = f"{ds_name}_base.zip"
    wget_download_and_extract(BOP_SRC + filename, BOP_DS_DIR)

    suffixes = ["models"] + BOP_DATASETS[ds_name]["test_splits"]
    if download_pbr:
        suffixes += ["train_pbr"]
    if download_train:
        suffixes += BOP_DATASETS[ds_name].get("train_splits", [])
    for suffix in suffixes:
        wget_download_and_extract(BOP_SRC + f"{ds_name}_{suffix}.zip", BOP_DS_DIR / ds_name)


def download_bop_gdrive(ds_name):
    gdrive_download(f"bop_datasets/{ds_name}", BOP_DS_DIR / ds_name)


def wget_download_and_extract(url, out):
    tmp_path = DOWNLOAD_DIR / url.split("/")[-1]
    if tmp_path.exists():
        logger.info(f"{url} already downloaded: {tmp_path}...")
    else:
        logger.info(f"Download {url} at {tmp_path}...")
        wget.download(url, out=tmp_path.as_posix())
    logger.info(f"Extracting {tmp_path} at {out}.")
    zipfile.ZipFile(tmp_path).extractall(out)


if __name__ == "__main__":
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
