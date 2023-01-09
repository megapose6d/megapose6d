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

# Third Party
import torch
from torch.distributed.elastic.multiprocessing.errors import record

# MegaPose
from megapose.utils.distributed import (
    get_rank,
    get_tmp_dir,
    get_world_size,
    init_distributed_mode,
)
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


@record
def main():
    init_distributed_mode()
    proc_id = get_rank()
    n_tasks = get_world_size()
    n_cpus = os.environ.get("N_CPUS", "not specified")
    logger.info(f"Number of processes (=num GPUs): {n_tasks}")
    logger.info(f"Process ID: {proc_id}")
    logger.info(f"TMP Directory for this job: {get_tmp_dir()}")
    logger.info(f"GPU CUDA ID: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"Num GPUS available: {torch.cuda.device_count()}")
    logger.info(f"GPU Properties: {torch.cuda.get_device_properties(0)}")
    logger.info(f"Max number of CPUs for this process: {n_cpus}")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
