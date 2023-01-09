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
import contextlib
import os
import time
import random
import webdataset as wds

# Third Party
import torch
import numpy as np
import pinocchio as pin


def make_seed(*args):
    """Copied from webdataset"""
    seed = 0
    for arg in args:
        seed = (seed * 31 + hash(arg)) & 0x7FFFFFFF
    return seed


@contextlib.contextmanager
def temp_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_unique_seed() -> int:
    worker_seed = wds.utils.pytorch_worker_seed()
    seed = make_seed(
        worker_seed,
        os.getpid(),
        time.time_ns(),
        os.urandom(4),
    )
    return seed


def set_seed(seed: int) -> None:
    pin.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
