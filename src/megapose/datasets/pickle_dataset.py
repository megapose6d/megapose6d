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
import glob
import pickle
from pathlib import Path

# MegaPose
from megapose.datasets.utils import SceneData


class PickleDataset:
    def __init__(self, ds_dir) -> None:
        self.ds_dir = Path(ds_dir)
        assert self.ds_dir.exists(), f"The directory {ds_dir} doesn't exist"

        glob_str = f"{self.ds_dir}/*_data.pkl"
        files = list(glob.glob(glob_str))

        self._length = len(files)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """Return SceneData object for that index."""
        fname = self.ds_dir / f"{idx:06}_data.pkl"
        data = pickle.load(open(fname, "rb"))
        data["TWC"] = data["world_t_camera"]
        data["scene_id"] = 0
        data["im_idx"] = idx
        mask = None
        infos = dict()
        infos["camera"] = {"TWC": data["world_t_camera"], "K": data["intrinsics"]}
        infos["frame_info"] = {"scene_id": 0, "view_id": idx, "cam_name": "cam", "cam_id": "cam"}
        scene_data = SceneData(data["rgb"], data["depth"], mask, infos)
        return scene_data
