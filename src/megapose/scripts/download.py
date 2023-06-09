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
from pathlib import Path

# MegaPose
from megapose.config import LOCAL_DATA_DIR, PROJECT_DIR
from megapose.utils.logging import get_logger

logger = get_logger(__name__)

RCLONE_CFG_PATH = PROJECT_DIR / "rclone.conf"
RCLONE_ROOT = "inria_data:"


def main():
    parser = argparse.ArgumentParser("Megapose download utility")
    parser.add_argument("--megapose_models", action="store_true")
    parser.add_argument("--example_data", action="store_true")
    parser.add_argument("--data_subset", type=str, default=None)
    parser.add_argument("--data_object_models", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.megapose_models:
        # rclone copyto inria_data:megapose-models/ megapose-models/
        #     --exclude="**epoch**" --config $MEGAPOSE_DIR/rclone.conf -P
        download(
            f"megapose-models/",
            LOCAL_DATA_DIR / "megapose-models/",
            flags=["--exclude", "*epoch*"],
        )

    if args.example_data:
        # rclone copyto inria_data:examples/ examples/
        #     --config $MEGAPOSE_DIR/rclone.conf -P
        download(f"examples/", LOCAL_DATA_DIR / "examples")

    if args.data_subset:
        # rclone copyto inria_data:webdatasets/ webdatasets/
        #     --include "0000000*.tar" --include "*.json" --include "*.feather"
        #     --config $MEGAPOSE_DIR/rclone.conf -P
        download(
            f"webdatasets/",
            LOCAL_DATA_DIR / "webdatasets/",
            flags=[
                "--include",
                args.data_subset,
                "--include",
                "*.json",
                "--include",
                "*.feather",
            ],
        )

    if args.data_object_models:
        # rclone copyto inria_data:tars/ tars/
        #     --include "shapenetcorev2.zip" --include "google_scanned_objects.zip"
        #     --config $MEGAPOSE_DIR/rclone.conf -P
        # unzip tars/shapenetcorev2.zip
        # unzip tars/google_scanned_objects.zip
        download(
            f"tars/",
            LOCAL_DATA_DIR / "tars/",
            flags=[
                "--include",
                "shapenetcorev2.zip",
                "--include",
                "google_scanned_objects.zip",
            ],
        )
        subprocess.run(["unzip", LOCAL_DATA_DIR / "tars/shapenetcorev2.zip", LOCAL_DATA_DIR])
        subprocess.run(
            ["unzip", LOCAL_DATA_DIR / "tars/google_scanned_objects.zip", LOCAL_DATA_DIR]
        )


def run_rclone(cmd, args, flags):
    rclone_cmd = ["rclone", cmd] + args + flags + ["--config", str(RCLONE_CFG_PATH)]
    logger.debug(" ".join(rclone_cmd))
    subprocess.run(rclone_cmd)


def download(download_path, local_path, flags=[]):
    download_path = Path(download_path)
    if download_path.name != local_path.name:
        local_path = local_path / download_path.name
    rclone_path = RCLONE_ROOT + str(download_path) + "/"
    local_path = str(local_path)
    logger.info(f"Copying {rclone_path} to {local_path}")
    run_rclone("copyto", [rclone_path, local_path], flags=flags + ["-P"])


if __name__ == "__main__":
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    main()
