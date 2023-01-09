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
from collections import OrderedDict

# Third Party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# MegaPose
import megapose.utils.tensor_collection as tc
from megapose.datasets.samplers import DistributedSceneSampler
from megapose.datasets.scene_dataset import SceneDataset, SceneObservation
from megapose.evaluation.data_utils import parse_obs_data
from megapose.utils.distributed import get_rank, get_tmp_dir, get_world_size


class EvaluationRunner:
    def __init__(self, scene_ds, meters, batch_size=64, cache_data=True, n_workers=4, sampler=None):

        self.rank = get_rank()
        self.world_size = get_world_size()
        self.tmp_dir = get_tmp_dir()

        self.scene_ds = scene_ds
        if sampler is None:
            sampler = DistributedSceneSampler(
                scene_ds, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )
        dataloader = DataLoader(
            scene_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            sampler=sampler,
            collate_fn=SceneObservation.collate_fn,
        )

        if cache_data:
            self.dataloader = list(tqdm(dataloader))
        else:
            self.dataloader = dataloader

        self.meters = meters
        self.meters = OrderedDict(
            {k: v for k, v in sorted(self.meters.items(), key=lambda item: item[0])}
        )

    @staticmethod
    def make_empty_predictions():
        infos = dict(
            view_id=np.empty(0, dtype=np.int),
            scene_id=np.empty(0, dtype=np.int),
            label=np.empty(0, dtype=np.object),
            score=np.empty(0, dtype=np.float),
        )
        poses = torch.empty(0, 4, 4, dtype=torch.float)
        return tc.PandasTensorCollection(infos=pd.DataFrame(infos), poses=poses)

    def evaluate(self, obj_predictions, device="cuda"):
        for meter in self.meters.values():
            meter.reset()
        obj_predictions = obj_predictions.to(device)
        for data in tqdm(self.dataloader):
            for k, meter in self.meters.items():
                meter.add(obj_predictions, data["gt_data"].to(device))
        return self.summary()

    def summary(self):
        summary, dfs = dict(), dict()
        for meter_k, meter in self.meters.items():
            if len(meter.datas) > 0:
                meter.gather_distributed(tmp_dir=self.tmp_dir)
                summary_, df_ = meter.summary()
                dfs[meter_k] = df_
                for k, v in summary_.items():
                    summary[meter_k + "/" + k] = v
        return summary, dfs
