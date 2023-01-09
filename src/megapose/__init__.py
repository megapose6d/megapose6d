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
import cv2


def assign_gpu() -> None:
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
        device_ids = device_ids.split(",")
    else:
        device_ids = range(int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert local_rank < len(device_ids)
    cuda_id = int(device_ids[local_rank])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    if "SLURM_JOB_NODELIST" in os.environ:
        os.environ["EGL_VISIBLE_DEVICES"] = str(cuda_id)


assign_gpu()
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

if "EGL_VISIBLE_DEVICES" not in os.environ:
    os.environ['EGL_VISIBLE_DEVICES'] = '0'

for k in (
    "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES", "EGL_VISIBLE_DEVICES"):
    if k in os.environ:
        print(f"{k}: {os.environ[k]}")
