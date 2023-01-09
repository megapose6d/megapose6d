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
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

# Third Party
import simplejson as json
import torch
from bokeh import document
from bokeh.io import export_png, save
from bokeh.io.export import get_screenshot_as_png
from omegaconf import OmegaConf

# MegaPose
from megapose.training.training_config import TrainingConfig
from megapose.utils.distributed import get_rank, get_world_size
from megapose.utils.logging import get_logger

logger = get_logger(__name__)

RGB_DIMS = [0, 1, 2]
DEPTH_DIMS = [3]


def cast(obj: torch.Tensor) -> torch.Tensor:
    return obj.cuda(non_blocking=True)


def cast_to_numpy(obj, dtype=None):
    if isinstance(obj, torch.Tensor):
        obj = obj.cpu().numpy()

    if dtype is not None:
        obj = obj.astype(dtype)
    return obj


def cast_images(rgb: torch.Tensor, depth: Optional[torch.Tensor]) -> torch.Tensor:
    """Convert rgb and depth to a single to cuda FloatTensor.

    Arguments:
        rgb: (bsz, 3, h, w) uint8 tensor, with values in [0, 1]
        depth: (bsz, h, w) float tensor, or None

    Returns:
        images: (bsz, 3, h, w) RGB or (bsz, 4, h, w) RGB-D images.
    """
    rgb_tensor = cast(rgb).float() / 255
    if depth is None:
        return rgb_tensor
    else:
        depth_tensor = cast(depth).unsqueeze(1)
        return torch.cat((rgb_tensor, depth_tensor), dim=1)


def cast_tensor_image_to_numpy(images):
    """Convert images to

    Args:
        images: [B,C,H,W]
    """
    images = (images[:, :3] * 255).to(torch.uint8)
    images = images.permute([0, 2, 3, 1])
    images = images.cpu().numpy()
    return images


def cast_raw_numpy_images_to_tensor(images):
    """
    Casts numpy images to tensor.

    Args:
        images: [B,H,W,C] numpy array, RGB values in [0,255], depth in meters

    """
    B, H, W, C = images.shape
    assert C in [
        3,
        4,
    ], f"images must have shape [B,H,W,C] with C=3 (rgb) or C=4 (rgbd), encountered C={C}"
    images = torch.as_tensor(images)

    max_rgb = torch.max(images[:, RGB_DIMS])
    if max_rgb < 1.5:
        raise Warning("You are about to divide by 255 but the max rgb pixel value is less than 1.5")

    # [B,C,H,W]
    images = images.permute(0, 3, 1, 2).cuda().float()
    images[:, RGB_DIMS] /= 255  # normalize RGB channels to be in [0,1]
    return images


def make_optimizer(
    parameters: Iterator[torch.nn.Parameter],
    cfg: TrainingConfig
) -> torch.optim.Optimizer:

    optimizer: Optional[torch.optim.Optimizer] = None
    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            parameters, lr=cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(cfg.optimizer)
    return optimizer


def make_lr_ratio_function(cfg: TrainingConfig) -> Callable:

    def lr_ratio(batch: int) -> float:
        this_rank_epoch_size = cfg.epoch_size // get_world_size()
        n_batch_per_epoch = this_rank_epoch_size // cfg.batch_size
        epoch_id = batch // n_batch_per_epoch

        if cfg.n_epochs_warmup == 0:
            lr_ratio = 1.0
        else:
            n_batches_warmup = cfg.n_epochs_warmup * n_batch_per_epoch
            lr_ratio = min(max(batch, 1) / n_batches_warmup, 1.0)

        lr_ratio /= 10 ** (epoch_id // cfg.lr_epoch_decay)
        return lr_ratio

    return lr_ratio


def write_logs(cfg, model, epoch, log_dict=None, test_dict=None, bokeh_docs=None):
    assert get_rank() == 0

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(exist_ok=True)

    if log_dict is not None:
        log_dict.update(epoch=epoch)

    if not (save_dir / "config.yaml").exists():
        OmegaConf.save(cfg, save_dir / "config.yaml")

    def save_checkpoint(model, postfix=None):
        ckpt_name = "checkpoint"
        if postfix is not None:
            ckpt_name += postfix
        ckpt_name += ".pth.tar"
        path = save_dir / ckpt_name
        torch.save({"state_dict": model.module.state_dict(), "epoch": epoch}, path)
        logger.debug(f"Wrote checkpoint: {path.parent / path.name}")

    save_checkpoint(model)
    save_checkpoint(model, postfix="_epoch=last")

    if cfg.save_epoch_interval is not None and epoch % cfg.save_epoch_interval == 0:
        save_checkpoint(model, postfix=f"_epoch={epoch}")

    bokeh_doc_path = None
    if bokeh_docs is not None and len(bokeh_docs) > 0:
        bokeh_doc_dir = Path(cfg.save_dir) / "visualization"
        bokeh_doc_dir.mkdir(exist_ok=True)
        for bokeh_doc_postfix, bokeh_doc_json in bokeh_docs.items():
            if cfg.vis_save_only_last:
                bokeh_doc_path = bokeh_doc_dir / f"epoch=last_{bokeh_doc_postfix}.html"
            else:
                bokeh_doc_path = bokeh_doc_dir / f"epoch={epoch}_{bokeh_doc_postfix}.html"
            if bokeh_doc_path.exists():
                bokeh_doc_path.unlink()
            bokeh_doc = document.Document.from_json(bokeh_doc_json)
            # Lightest option
            save(bokeh_doc, bokeh_doc_path)
            bokeh_doc.clear()
        logger.info(f"Wrote visualization: {bokeh_doc_path}")

    if log_dict is not None:
        log_dict.update(has_visualization=bokeh_doc_path is not None)
        with open(save_dir / "log.txt", "a") as f:
            f.write(json.dumps(log_dict, ignore_nan=True) + "\n")
        logger.info(log_dict)

    if test_dict is not None:
        for ds_name, ds_errors in test_dict.items():
            ds_errors["epoch"] = epoch
            with open(save_dir / f"errors_{ds_name}.txt", "a") as f:
                f.write(json.dumps(test_dict[ds_name], ignore_nan=True) + "\n")
        logger.info(test_dict)

    return


class SimpleTimer:
    def __init__(self) -> None:
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def end(self):
        self.stop()

    def elapsed(self):
        return self.end_time - self.start_time


class CudaTimer:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_sec = None

        self.start_called = False
        self.end_called = False

    def start(self) -> None:
        if not self.enabled:
            return

        self.start_called = True
        self.start_event.record()

    def end(self) -> None:
        if not self.enabled:
            return

        self.end_called = True
        self.end_event.record()
        torch.cuda.synchronize()
        self.elapsed_sec = self.start_event.elapsed_time(self.end_event) / 1000.0

    def stop(self) -> None:
        self.end()

    def elapsed(self) -> float:
        """Return the elapsed time (in seconds)."""
        if not self.enabled:
            return 0.0

        if not self.start_called:
            raise ValueError("You must call CudaTimer.start() before querying the elapsed time")

        if not self.end_called:
            raise ValueError("You must call CudaTimer.end() before querying the elapsed time")

        return self.elapsed_sec
