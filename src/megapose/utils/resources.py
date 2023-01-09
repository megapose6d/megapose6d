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
import subprocess
import xml.etree.ElementTree as ET
from shutil import which

# Third Party
import psutil
import torch


def is_egl_available():
    return is_gpu_available and "EGL_VISIBLE_DEVICES" in os.environ


def is_gpu_available():
    return which("nvidia-smi") is not None


def is_slurm_available():
    return which("sinfo") is not None


def get_total_memory():
    current_process = psutil.Process(os.getpid())
    mem = current_process.memory_full_info().pss
    for child in current_process.children():
        mem += child.memory_full_info().pss
    return mem / (1024**3)


def get_cuda_memory():
    return torch.cuda.max_memory_allocated() / (1024.0**3)


def get_gpu_memory():

    devices = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
    ).split(",")
    assert len(devices) == 1
    out = subprocess.check_output(["nvidia-smi", "--id=" + str(devices[0]), "-q", "--xml-format"])
    tree = ET.fromstring(out)
    gpu = tree.findall("gpu")[0]
    memory = float(gpu.find("fb_memory_usage").find("used").text.split(" ")[0]) / 1024
    return memory


def assign_gpu():
    device_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    device_ids = device_ids.split(",")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    assert local_rank < len(device_ids)
    cuda_id = int(device_ids[local_rank])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    return
