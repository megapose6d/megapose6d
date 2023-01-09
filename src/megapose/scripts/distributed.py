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
import datetime
import os
import sys
from pathlib import Path

# Third Party
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# MegaPose
import megapose.utils.hostlist as hostlist
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def get_tmp_dir():
    if "JOB_DIR" in os.environ:
        tmp_dir = Path(os.environ["JOB_DIR"]) / "tmp"
    else:
        tmp_dir = Path("/tmp/cosypose_job")
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir


def sync_config(cfg, local_fields=[]):
    cfg_path = get_tmp_dir() / "config.yaml"
    if get_rank() == 0:
        OmegaConf.save(cfg, cfg_path)
    dist.barrier()
    my_cfg = cfg
    cfg = OmegaConf.load(cfg_path)
    for local_field in local_fields:
        if local_field in my_cfg:
            cfg[local_field] = my_cfg[local_field]
    dist.barrier()
    return cfg


def sync_model(model):
    sync_dir = get_tmp_dir() / "models"
    sync_dir.mkdir(exist_ok=True)
    sync_ckpt = sync_dir / "sync.checkpoint"
    if get_rank() == 0 and get_world_size() > 1:
        logger.info("save")
        torch.save(model.state_dict(), sync_ckpt)
    dist.barrier()
    if get_rank() > 0:
        logger.info("load")
        model.load_state_dict(torch.load(sync_ckpt))
    dist.barrier()
    return model


def redirect_output():
    if "JOB_DIR" in os.environ:
        rank = get_rank()
        output_file = Path(os.environ["JOB_DIR"]) / f"stdout{rank}.out"
        sys.stdout = open(output_file, "w")
        sys.stderr = open(output_file, "w")
    return


def get_rank():
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    return rank


def get_world_size():
    if not torch.distributed.is_initialized():
        world_size = 1
    else:
        world_size = torch.distributed.get_world_size()
    return world_size


def init_distributed_mode():
    assert torch.cuda.device_count() == 1
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    if "SLURM_STEPS_GPUS" in os.environ:
        gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
        os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))
    else:
        os.environ["MASTER_PORT"] = str(12345)

    if "SLURM_JOB_NODELIST" in os.environ:
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    os.environ["NCCL_BLOCKING_WAIT"] = 1

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=1800 * 4),
    )
    torch.distributed.barrier()


def reduce_dict(input_dict, average=True):
    """
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        reduced_dict = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.tensor(values).float().cuda()
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict
