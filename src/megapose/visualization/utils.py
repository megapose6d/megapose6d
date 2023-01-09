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
from typing import Any, Dict, Optional, Tuple, Union

# Third Party
import cv2
import numpy as np
import torch
from PIL import ImageEnhance


def image_to_np_uint8(im: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Returns a np.uint8 image"""
    if isinstance(im, torch.Tensor):
        im_np = im.detach().cpu().numpy()
    else:
        im_np = im

    if im_np.shape[0] in {
        3,
    }:
        im_np = im_np.transpose((1, 2, 0))

    if im_np.max() <= 1.0:
        im_np = (im_np * 255).astype(np.uint8)
    else:
        assert im_np.dtype == np.uint8
    return im_np


def get_mask_from_rgb(img: np.ndarray) -> np.ndarray:
    img_t = torch.as_tensor(img)
    mask = torch.zeros_like(img_t)
    mask[img_t > 0] = 255
    mask = torch.max(mask, dim=-1)[0]
    mask_np = mask.numpy().astype(np.bool_)
    return mask_np


def make_contour_overlay(
    img: np.ndarray,
    render: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
    dilate_iterations: int = 1,
) -> Dict[str, Any]:

    if color is None:
        color = (0, 255, 0)

    mask_bool = get_mask_from_rgb(render)
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
    mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)

    # maybe dilate this a bit to make edges thicker
    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    # dilate
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    img_contour = np.copy(img)
    img_contour[canny > 0] = color

    return {
        "img": img_contour,
        "mask": mask_bool,
        "canny": canny,
    }


def adjust_brightness(img, factor=1.5):
    """Higher factor makes it brighter."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def tensor_image_to_uint8(x):
    # x [3, 240, 320]
    return (x.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()


def get_ds_info(ds_name):
    if ds_name == "ycbv":
        ds_name = "ycbv"
        urdf_ds_name = "ycbv"
        obj_ds_name = "ycbv.panda3d"
    elif ds_name in ["lm", "lmo"]:
        urdf_ds_name = "lm"
        obj_ds_name = "lm.panda3d"
    elif ds_name == "tless":
        urdf_ds_name = "tless.cad"  # is this correct?
        obj_ds_name = "tless.panda3d"
    elif ds_name == "hope":
        urdf_ds_name = "hope.cad"  # not sure if this exists
        obj_ds_name = "hope.panda3d"
    elif ds_name == "hb":
        urdf_ds_name = "hb.cad"  # not sure if this exists
        obj_ds_name = "hb.panda3d"
    elif ds_name == "tudl":
        urdf_ds_name = "tudl.cad"  # not sure if this exists
        obj_ds_name = "tudl.panda3d"
    elif ds_name == "custom":
        urdf_ds_name = None  # not sure if this exists
        obj_ds_name = "custom.panda3d"
    else:
        raise ValueError("Unknown dataset")

    return urdf_ds_name, obj_ds_name


# draw a single bounding box onto a numpy array image
def draw_bounding_box(
    img,
    bbox: np.ndarray,
    color=None,
):
    """Draw a bounding box onto a numpy array image.

    Args:
        bbox: [xmin, ymin, xmax, ymax]
    """

    if color is None:
        color = [255, 0, 0]

    x_min, y_min, x_max, y_max = bbox.astype(np.int64)
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    return img
