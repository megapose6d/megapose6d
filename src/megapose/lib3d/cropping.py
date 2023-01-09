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



# Third Party
import torch
import torchvision

# MegaPose
from megapose.datasets.pose_dataset import PoseDataset

# Local Folder
from .camera_geometry import boxes_from_uv, project_points, project_points_robust


def deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, lamb=1.4, im_size=(240, 320), clamp=False):
    """
    gt_boxes: N x 4
    crop_boxes: N x 4
    """
    lobs, robs, uobs, dobs = obs_boxes[:, [0, 2, 1, 3]].t()
    lrend, rrend, urend, drend = rend_boxes[:, [0, 2, 1, 3]].t()
    xc = rend_center_uv[..., 0, 0]
    yc = rend_center_uv[..., 0, 1]
    lobs, robs = lobs.unsqueeze(-1), robs.unsqueeze(-1)
    uobs, dobs = uobs.unsqueeze(-1), dobs.unsqueeze(-1)
    lrend, rrend = lrend.unsqueeze(-1), rrend.unsqueeze(-1)
    urend, drend = urend.unsqueeze(-1), drend.unsqueeze(-1)

    xc, yc = xc.unsqueeze(-1), yc.unsqueeze(-1)
    w = max(im_size)
    h = min(im_size)
    r = w / h

    xdists = torch.cat(
        ((lobs - xc).abs(), (lrend - xc).abs(), (robs - xc).abs(), (rrend - xc).abs()), dim=1
    )
    ydists = torch.cat(
        ((uobs - yc).abs(), (urend - yc).abs(), (dobs - yc).abs(), (drend - yc).abs()), dim=1
    )
    xdist = xdists.max(dim=1)[0]
    ydist = ydists.max(dim=1)[0]
    width = torch.max(xdist, ydist * r) * 2 * lamb
    height = torch.max(xdist / r, ydist) * 2 * lamb

    xc, yc = xc.squeeze(-1), yc.squeeze(-1)
    x1, y1, x2, y2 = xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2
    boxes = torch.cat((x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1)), dim=1)
    assert not clamp
    if clamp:
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, h - 1)
    return boxes


def deepim_crops(images, obs_boxes, K, TCO_pred, O_vertices, output_size=None, lamb=1.4):
    batch_size, _, h, w = images.shape
    device = images.device
    if output_size is None:
        output_size = (h, w)
    uv = project_points(O_vertices, K, TCO_pred)
    rend_boxes = boxes_from_uv(uv)
    rend_center_uv = project_points(torch.zeros(batch_size, 1, 3).to(device), K, TCO_pred)
    boxes = deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, im_size=(h, w), lamb=lamb)
    boxes = torch.cat((torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1)
    crops = crop_images(images, boxes, output_size=output_size, sampling_ratio=4)
    return boxes[:, 1:], crops


def deepim_crops_robust(
    images,
    obs_boxes,
    K,
    TCO_pred,
    tCR_in,
    O_vertices,
    output_size=None,
    lamb=1.4,
    return_crops=True,
):
    batch_size, _, h, w = images.shape
    batch_size = TCO_pred.shape[0]
    device = images.device
    if output_size is None:
        output_size = (h, w)
    uv = project_points_robust(O_vertices, K, TCO_pred)
    rend_boxes = boxes_from_uv(uv)
    TCR = TCO_pred.clone()
    TCR[:, :3, -1] = tCR_in
    rend_center_uv = project_points_robust(torch.zeros(batch_size, 1, 3).to(device), K, TCR)
    boxes = deepim_boxes(rend_center_uv, obs_boxes, rend_boxes, im_size=(h, w), lamb=lamb)
    boxes = torch.cat((torch.arange(batch_size).unsqueeze(1).to(device).float(), boxes), dim=1)
    crops = None
    if return_crops:
        crops = crop_images(images, boxes, output_size=output_size, sampling_ratio=4)
    return boxes[:, 1:], crops


def crop_images(images, boxes, output_size, sampling_ratio):
    """Crop RGB/RGBD images.

    Properly handles using roi_align with a depth image (which contains invalid pixels)
    """
    batch_size, nchannels, h, w = images.shape
    assert nchannels in [3, 4]  # doesn't handle grayscale currently
    has_depth = nchannels == 4

    DEPTH_DIMS = PoseDataset.DEPTH_DIMS

    if not has_depth:
        crops = torchvision.ops.roi_align(
            images, boxes, output_size=output_size, sampling_ratio=sampling_ratio
        )
    else:
        crops = torchvision.ops.roi_align(images, boxes, output_size=output_size, sampling_ratio=4)

        # roi_align can result in invalid depth measurements
        # since it does interpolation. Simply set those to zero
        # [B,1,H,W]
        depth = images[:, DEPTH_DIMS]
        depth_valid = torch.zeros_like(images[:, DEPTH_DIMS])
        depth_valid[depth > 0] = 1
        depth_valid_crops = torchvision.ops.roi_align(
            depth_valid, boxes, output_size=output_size, sampling_ratio=4
        )
        depth_mask = torch.ones_like(depth_valid_crops)
        depth_mask[depth_valid_crops < 0.99] = 0
        crops[:, DEPTH_DIMS] *= depth_mask  # set invalid depth to zero

    return crops
