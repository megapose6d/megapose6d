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
import functools
import os
import time
from collections import defaultdict
from typing import Any, Dict, List

# Third Party
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

# MegaPose
from megapose.config import EXP_DIR
from megapose.datasets.datasets_cfg import make_object_dataset, make_scene_dataset
from megapose.datasets.object_dataset import RigidObjectDataset, concat_object_datasets
from megapose.datasets.pose_dataset import PoseDataset
from megapose.datasets.scene_dataset import (
    IterableMultiSceneDataset,
    IterableSceneDataset,
    RandomIterableSceneDataset,
    SceneDataset,
)
from megapose.datasets.web_scene_dataset import IterableWebSceneDataset, WebSceneDataset
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.panda3d_renderer.panda3d_batch_renderer import Panda3dBatchRenderer
from megapose.training.megapose_forward_loss import megapose_forward_loss
from megapose.training.pose_models_cfg import check_update_config, create_model_pose
from megapose.training.training_config import DatasetConfig, TrainingConfig
from megapose.training.utils import (
    CudaTimer,
    make_lr_ratio_function,
    make_optimizer,
    write_logs,
)
from megapose.utils.distributed import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    reduce_dict,
    sync_config,
    sync_model,
)
from megapose.utils.logging import get_logger
from megapose.utils.random import get_unique_seed, set_seed, temp_numpy_seed
from megapose.utils.resources import get_cuda_memory, get_gpu_memory, get_total_memory


def worker_init_fn(worker_id: int) -> None:
    set_seed(get_unique_seed())


def train_megapose(cfg: TrainingConfig) -> None:
    logger = get_logger("main")
    cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_num_threads(1)

    cfg = check_update_config(cfg)
    logger.info(f"Training with cfg: \n {OmegaConf.to_yaml(cfg)}")

    init_distributed_mode()
    cfg = sync_config(cfg)

    set_seed(get_rank())
    world_size = get_world_size()

    logger.info(f"Connection established with {world_size} gpus.")
    cfg.global_batch_size = world_size * cfg.batch_size
    assert cfg.hardware.n_gpus == world_size

    def split_objects_across_gpus(obj_dataset: RigidObjectDataset) -> RigidObjectDataset:
        rank, world_size = get_rank(), get_world_size()
        if cfg.split_objects_across_gpus:
            with temp_numpy_seed(0):
                this_rank_labels = set(
                    np.array_split(
                        np.random.permutation(np.array([obj.label for obj in obj_dataset.objects])),
                        world_size,
                    )[rank].tolist()
                )
        else:
            this_rank_labels = set([obj.label for obj in renderer_obj_dataset.objects])
        if cfg.n_max_objects is not None:
            this_rank_labels = set(list(this_rank_labels)[: cfg.n_max_objects])

        obj_dataset = RigidObjectDataset(
            [obj for obj in obj_dataset.objects if obj.label in this_rank_labels]
        )
        return obj_dataset

    # Object datasets
    renderer_obj_dataset = concat_object_datasets(
        [
            split_objects_across_gpus(make_object_dataset(ds_cfg.renderer_obj_ds_name))
            for ds_cfg in cfg.train_datasets + cfg.val_datasets
        ]
    )
    mesh_obj_dataset = concat_object_datasets(
        [
            split_objects_across_gpus(make_object_dataset(ds_cfg.mesh_obj_ds_name))
            for ds_cfg in cfg.train_datasets + cfg.val_datasets
        ]
    )
    this_rank_labels = set([obj.label for obj in renderer_obj_dataset.objects])
    assert len(renderer_obj_dataset) == len(mesh_obj_dataset)
    logger.info(f"Number of objects to train on (this rank):  {len(mesh_obj_dataset)})")

    # Scene dataset
    def make_iterable_scene_dataset(
        dataset_configs: List[DatasetConfig],
        deterministic: bool = False,
    ) -> IterableMultiSceneDataset:
        scene_dataset_iterators = []
        for this_dataset_config in dataset_configs:
            ds = make_scene_dataset(
                this_dataset_config.ds_name,
                load_depth=cfg.input_depth,
            )
            if isinstance(ds, WebSceneDataset):
                assert not deterministic
                iterator: IterableSceneDataset = IterableWebSceneDataset(
                    ds, buffer_size=cfg.sample_buffer_size
                )
            else:
                assert isinstance(ds, SceneDataset)
                iterator = RandomIterableSceneDataset(ds, deterministic=deterministic)
            logger.info(f"Loaded dataset {this_dataset_config.ds_name}")

            for _ in range(this_dataset_config.n_repeats):
                scene_dataset_iterators.append(iterator)
        return IterableMultiSceneDataset(scene_dataset_iterators)

    scene_ds_train = make_iterable_scene_dataset(cfg.train_datasets)

    # Datasets
    ds_train = PoseDataset(
        scene_ds_train,
        resize=cfg.input_resize,
        apply_rgb_augmentation=cfg.rgb_augmentation,
        apply_background_augmentation=cfg.background_augmentation,
        apply_depth_augmentation=cfg.depth_augmentation,
        min_area=cfg.min_area,
        depth_augmentation_level=cfg.depth_augmentation_level,
        keep_labels_set=this_rank_labels,
    )

    ds_iter_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        num_workers=cfg.n_dataloader_workers,
        collate_fn=ds_train.collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        pin_memory=True,
    )
    iter_train = iter(ds_iter_train)

    ds_iter_val = None
    if len(cfg.val_datasets) > 0:
        scene_ds_val = make_iterable_scene_dataset(cfg.val_datasets, deterministic=True)
        ds_val = PoseDataset(
            scene_ds_val,
            resize=cfg.input_resize,
            apply_rgb_augmentation=False,
            apply_depth_augmentation=False,
            apply_background_augmentation=False,
            min_area=cfg.min_area,
            keep_labels_set=this_rank_labels,
        )
        ds_iter_val = DataLoader(
            ds_val,
            batch_size=cfg.batch_size,
            num_workers=cfg.n_dataloader_workers,
            worker_init_fn=worker_init_fn,
            collate_fn=ds_train.collate_fn,
            persistent_workers=True,
            pin_memory=True,
        )

    renderer = Panda3dBatchRenderer(
        object_dataset=renderer_obj_dataset,
        n_workers=cfg.n_rendering_workers,
        preload_cache=False,
        split_objects=True,
    )

    mesh_db = (
        MeshDataBase.from_object_ds(mesh_obj_dataset)
        .batched(n_sym=cfg.n_symmetries_batch, resample_n_points=cfg.resample_n_points)
        .cuda()
        .float()
    )

    model = create_model_pose(cfg=cfg, renderer=renderer, mesh_db=mesh_db).cuda()

    if cfg.run_id_pretrain is not None:
        pretrain_path = EXP_DIR / cfg.run_id_pretrain / "checkpoint.pth.tar"
        pretrain_ckpt = torch.load(pretrain_path)["state_dict"]
        model.load_state_dict(pretrain_ckpt)
        logger.info(f"Using pretrained model from {pretrain_path}.")

    if cfg.resume_run_id:
        resume_run_dir = EXP_DIR / cfg.resume_run_id
        ckpt = None
        try:
            ckpt_path = resume_run_dir / "checkpoint.pth.tar"
            ckpt = torch.load(ckpt_path)
        except EOFError:
            print(
                "Unable to load checkpoint.pth.tar. Falling back to checkpoint_epoch=last.pth.tar"
            )
            ckpt_path = resume_run_dir / "checkpoint_epoch=last.pth.tar"
            ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Resuming: Loaded checkpoint from {ckpt_path}")
        start_epoch = ckpt["epoch"] + 1
    else:
        start_epoch = 1

    if cfg.sync_batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = sync_model(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device()
    )

    optimizer = make_optimizer(model.parameters(), cfg)

    this_rank_epoch_size = cfg.epoch_size // get_world_size()
    this_rank_n_batch_per_epoch = this_rank_epoch_size // cfg.batch_size
    # NOTE: LR schedulers "epoch" actually correspond to "batch"
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_ratio_function(cfg))
    lr_scheduler.last_epoch = (  # type: ignore
        start_epoch * this_rank_epoch_size // cfg.batch_size - 1
    )

    lr_scheduler.step()
    # Just remove the annoying warnings
    optimizer._step_count = 1  # type: ignore
    lr_scheduler.step()
    optimizer._step_count = 0  # type: ignore

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, cfg.n_epochs + 1):
        meters_train: Dict[str, AverageValueMeter] = defaultdict(lambda: AverageValueMeter())
        meters_val: Dict[str, AverageValueMeter] = defaultdict(lambda: AverageValueMeter())

        if cfg.add_iteration_epoch_interval is None:
            n_iterations = cfg.n_iterations
        else:
            n_iterations = min(epoch // cfg.add_iteration_epoch_interval + 1, cfg.n_iterations)

        forward_loss_fn = functools.partial(
            megapose_forward_loss, model=model, cfg=cfg, n_iterations=n_iterations, mesh_db=mesh_db
        )

        def train() -> None:
            meters = meters_train
            set_seed(epoch * get_rank() + get_rank())
            model.train()
            pbar = tqdm(
                range(this_rank_n_batch_per_epoch), ncols=120, disable=cfg.logging_style != "tqdm"
            )
            for n in pbar:
                start_iter = time.time()
                t = time.time()
                data = next(iter_train)
                time_data = time.time() - t

                optimizer.zero_grad()

                debug_dict: Dict[str, Any] = dict()
                timer_forward = CudaTimer(enabled=cfg.cuda_timing)
                timer_forward.start()
                with torch.cuda.amp.autocast():
                    loss = forward_loss_fn(
                        data=data,
                        meters=meters,
                        train=True,
                        debug_dict=debug_dict,
                    )

                time_render = debug_dict["time_render"]
                meters["loss_total"].add(loss.item())
                timer_forward.end()

                timer_backward = CudaTimer(enabled=cfg.cuda_timing)
                timer_backward.start()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.clip_grad_norm, norm_type=2
                )
                meters["grad_norm"].add(torch.as_tensor(total_grad_norm).item())

                scaler.step(optimizer)
                scaler.update()
                timer_backward.end()

                lr_scheduler.step()

                time_iter = time.time() - start_iter
                if n > 0:
                    meters["time_iter"].add(time_iter)

                infos = dict(
                    loss=f"{loss.item():.2e}",
                    tf=f"{timer_forward.elapsed():.3f}",
                    tb=f"{timer_backward.elapsed():.3f}",
                    tr=f"{time_render:.3f}",
                    td=f"{time_data:.3f}",
                    tt=f"{time_iter:.3f}",
                )
                infos["it/s"] = f"{1. / time_iter:.2f}"
                if not pbar.disable:
                    pbar.set_postfix(**infos)
                else:
                    log_str = f"Epochs [{epoch}/{cfg.n_epochs}]"
                    log_str += " " + f"Iter [{n+1}/{this_rank_n_batch_per_epoch}]"
                    log_str += " " + " ".join([f"{k}={v}" for k, v in infos.items()])

                    logger.info(log_str)

                # Only add timing info after the first 10 iters times
                if epoch > 1 or n > 10:
                    meters["time_backward"].add(timer_backward.elapsed())
                    meters["time_forward"].add(timer_forward.elapsed())
                    meters["time_render"].add(time_render)
                    meters["time_iter"].add(time_iter)
                    meters["time_data"].add(time_data)

        @torch.no_grad()
        def validation() -> None:
            assert ds_iter_val is not None
            model.eval()
            iter_val = iter(ds_iter_val)
            n_batch = (cfg.val_size // get_world_size()) // cfg.batch_size
            pbar = tqdm(range(n_batch), ncols=120)
            for n in pbar:
                data = next(iter_val)
                loss = forward_loss_fn(
                    data=data,
                    meters=meters_val,
                    train=False,
                )
                meters_val["loss_total"].add(loss.item())

        train()

        do_eval = epoch % cfg.val_epoch_interval == 0 or epoch == 1
        if do_eval and ds_iter_val is not None:
            validation()

        log_dict = dict()
        log_dict.update(
            {
                "grad_norm": meters_train["grad_norm"].mean,
                "grad_norm_std": meters_train["grad_norm"].std,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "time_forward": meters_train["time_forward"].mean,
                "time_backward": meters_train["time_backward"].mean,
                "time_data": meters_train["time_data"].mean,
                "cuda_memory": get_cuda_memory(),
                "gpu_memory": get_gpu_memory(),
                "cpu_memory": get_total_memory(),
                "time": time.time(),
                "n_iterations": epoch * cfg.epoch_size // cfg.batch_size,
                "n_datas": epoch * this_rank_n_batch_per_epoch * cfg.batch_size,
            }
        )

        for string, meters in zip(("train", "val"), (meters_train, meters_val)):
            for k in dict(meters).keys():
                log_dict[f"{string}_{k}"] = meters[k].mean

        log_dict = reduce_dict(log_dict)

        if get_rank() == 0:
            logger.info(cfg.run_id)
            write_logs(
                cfg,
                model,
                epoch,
                log_dict=log_dict,
            )

        dist.barrier()
    os._exit(0)
