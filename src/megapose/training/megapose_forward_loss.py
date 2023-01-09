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
from typing import Any, Dict

# Third Party
import numpy as np
import torch
import torchnet
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from torch import nn

# MegaPose
from megapose.datasets.pose_dataset import BatchPoseData
from megapose.lib3d.camera_geometry import (
    project_points_robust as project_points_robust,
)
from megapose.lib3d.cosypose_ops import (
    TCO_init_from_boxes_zup_autodepth,
    loss_refiner_CO_disentangled_reference_point,
)
from megapose.lib3d.multiview import make_TCO_multiview
from megapose.lib3d.rigid_mesh_database import BatchedMeshes
from megapose.lib3d.transform_ops import add_noise, invert_transform_matrices
from megapose.models.pose_rigid import PosePredictor
from megapose.training.training_config import TrainingConfig
from megapose.training.utils import cast, cast_images
from megapose.visualization.bokeh_plotter import BokehPlotter


def megapose_forward_loss(
    model: PosePredictor,
    cfg: TrainingConfig,
    data: BatchPoseData,
    meters: Dict[str, torchnet.meter.AverageValueMeter],
    mesh_db: BatchedMeshes,
    n_iterations: int,
    debug_dict: Dict[str, Any],
    make_visualization: bool = False,
    train: bool = True,
    is_notebook: bool = False,
) -> torch.Tensor:

    # Normalize RGB dims to be in [0,1] from [0,255]
    # Don't tamper with depth
    images = cast_images(rgb=data.rgbs, depth=data.depths)
    K = cast(data.K).float()
    TCO_gt = cast(data.TCO).float()
    labels_gt = np.array([obj.label for obj in data.object_datas])
    bboxes_gt = cast(data.bboxes).float()

    batch_size, nchannels, h, w = images.shape
    device, dtype = TCO_gt.device, TCO_gt.dtype

    n_hypotheses = cfg.n_hypotheses
    bce_loss = nn.BCEWithLogitsLoss(reduction="none").cuda()

    hypotheses_image_ids = (
        torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, cfg.n_hypotheses)
    )
    hypotheses_labels = np.repeat(
        np.expand_dims(np.array(labels_gt, dtype=object), axis=1), cfg.n_hypotheses, axis=1
    ).copy()

    if cfg.hypotheses_init_method == "coarse_z_up+auto-depth":
        assert cfg.n_hypotheses == 1
        points_3d = mesh_db.select(np.ravel(hypotheses_labels).tolist()).sample_points(200)
        TCO_init_zup = TCO_init_from_boxes_zup_autodepth(bboxes_gt, points_3d, K)
        TCO_init_zup = add_noise(
            TCO_init_zup, euler_deg_std=[0, 0, 0], trans_std=[0.01, 0.01, 0.05]
        )
        hypotheses_TCO_init = TCO_init_zup.unsqueeze(1)
        is_hypothesis_positive = None

    elif cfg.hypotheses_init_method == "refiner_gt+noise":
        TCO_init_random = add_noise(
            TCO_gt.unsqueeze(1).repeat(1, n_hypotheses, 1, 1).flatten(0, 1),
            euler_deg_std=cfg.init_euler_deg_std,
            trans_std=cfg.init_trans_std,
        ).view(batch_size, cfg.n_hypotheses, 4, 4)
        hypotheses_TCO_init = TCO_init_random
        is_hypothesis_positive = None

    elif cfg.hypotheses_init_method == "coarse_classif_multiview_paper":
        assert cfg.predict_rendered_views_logits

        TCO_gt_noise = add_noise(
            TCO_gt,
            euler_deg_std=cfg.init_euler_deg_std,
            trans_std=cfg.init_trans_std,
        )
        tOR = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        tCR = TCO_gt_noise[..., :3, [-1]] + TCO_gt_noise[..., :3, :3] @ tOR.unsqueeze(-1)
        tCR = tCR.squeeze(-1)
        TCV_O = make_TCO_multiview(
            TCO_gt_noise,
            tCR,
            multiview_type="sphere_26views",
            remove_TCO_rendering=True,
            views_inplane_rotations=True,
        )
        n_candidate_views = TCV_O.shape[1]

        is_hypothesis_positive = np.zeros((batch_size, n_hypotheses), dtype=int)
        views_permutation = np.empty((2, batch_size, n_hypotheses), dtype=int)
        for b in range(batch_size):
            views_permutation[0, b, :] = b
            views_permutation[1, b, :] = np.random.permutation(n_candidate_views)[:n_hypotheses]
            positive_idx = np.where(views_permutation[1, b] == 0)[0]
            is_hypothesis_positive[b, positive_idx] = 1
            if len(positive_idx) == 0:
                # 30% of the time set positive view to be in the batch
                if np.random.rand() > 0.3:
                    positive_idx = np.random.randint(cfg.n_rendered_views)
                    views_permutation[1, b, positive_idx] = 0
                    is_hypothesis_positive[b, positive_idx] = 1
        hypotheses_TCO_init = TCV_O[views_permutation]

    elif cfg.hypotheses_init_method == "coarse_classif_multiview_SO3_grid":
        # TODO
        raise NotImplementedError

    else:
        raise ValueError(cfg.hypotheses_init_method)

    hypotheses_image_ids_flat = hypotheses_image_ids.flatten(0, 1)
    outputs = model(
        images=images[hypotheses_image_ids_flat],
        K=K[hypotheses_image_ids_flat],
        TCO=hypotheses_TCO_init.flatten(0, 1),
        labels=np.ravel(hypotheses_labels),
        n_iterations=n_iterations,
        random_ambient_light=cfg.random_ambient_light,
    )

    meshes = mesh_db.select(labels_gt)
    points = meshes.sample_points(cfg.n_points_loss)
    TCO_possible_gt = TCO_gt.unsqueeze(1) @ meshes.symmetries
    TCO_possible_gt = TCO_possible_gt.unsqueeze(1).repeat(1, n_hypotheses, 1, 1, 1).flatten(0, 1)
    points = points.unsqueeze(1).repeat(1, n_hypotheses, 1, 1).flatten(0, 1)

    list_losses_pose = []
    list_rendering_logits = []

    time_render = 0

    for n in range(n_iterations):
        iter_outputs = outputs[f"iteration={n+1}"]

        loss_TCO_iter, loss_TCO_iter_data = None, None
        if cfg.predict_pose_update:
            loss_TCO_iter, loss_TCO_iter_data = loss_refiner_CO_disentangled_reference_point(
                TCO_possible_gt=TCO_possible_gt,
                points=points,
                TCO_input=iter_outputs.TCO_input,
                refiner_outputs=iter_outputs.network_outputs["pose"],
                K_crop=iter_outputs.K_crop,
                tCR=iter_outputs.tCR,
            )
            loss_TCO_iter = loss_TCO_iter.view(batch_size, n_hypotheses)
            meters[f"loss_TCO-iter={n+1}"].add(loss_TCO_iter.mean().item())
            list_losses_pose.append(loss_TCO_iter)

        if cfg.predict_rendered_views_logits:
            list_rendering_logits.append(
                iter_outputs.renderings_logits.view(batch_size, n_hypotheses, -1)
            )

        time_render += iter_outputs.timing_dict["render"]
        # Add all the losses from loss-data
        if loss_TCO_iter_data is not None:
            for key, val in loss_TCO_iter_data.items():
                if key == "loss":
                    continue
                # reshape and remove invalid indices
                # [B, n_hypotheses]
                x = val.view(batch_size, n_hypotheses)
                save_key = f"loss_TCO-iter={n+1}-{key}"
                meters[save_key].add(x.mean().item())

    meters["time_render"].add(time_render)

    # Batch size x N hypotheses x N iterations
    loss_hypotheses = torch.zeros(
        (batch_size, n_hypotheses, n_iterations), device=device, dtype=dtype
    )
    if cfg.predict_pose_update:
        losses_pose = torch.stack(list_losses_pose).permute(1, 2, 0)
        loss_hypotheses += cfg.loss_alpha_pose * losses_pose
        loss_pose = losses_pose.mean(dim=(-1, -2))
        meters["loss_TCO"].add(loss_pose.mean().item())

    if cfg.predict_rendered_views_logits:
        # Batch size x N hypotheses x N iterations x N views
        assert cfg.n_rendered_views == 1
        assert n_iterations == 1
        rendering_logits = torch.stack(list_rendering_logits).permute(1, 2, 0, 3)
        rendering_logits /= cfg.renderings_logits_temperature
        loss_renderings_confidence = bce_loss(
            rendering_logits.flatten(1, 3),
            torch.tensor(is_hypothesis_positive, dtype=torch.float, device=device),
        ).unsqueeze(-1)
        meters["loss_renderings_confidence"].add(loss_renderings_confidence.mean().item())
        loss_hypotheses += cfg.loss_alpha_renderings_confidence * loss_renderings_confidence

    loss = loss_hypotheses.mean()

    meters["loss_total"].add(loss.item())

    if make_visualization:

        def add_mask_to_image(
            image: torch.Tensor, mask: torch.Tensor, color: str = "red"
        ) -> torch.Tensor:
            t_color = torch.zeros_like(image)
            idx = dict(red=0, green=1, blue=2)[color]
            t_color[idx, mask > 0] = 1.0
            output = image * 0.8 + t_color * 0.2
            return output

        plotter = BokehPlotter(is_notebook=is_notebook)
        grid = []

        n_views = cfg.n_rendered_views
        last_iter_outputs = outputs[f"iteration={n_iterations}"]
        images_crop = last_iter_outputs.images_crop
        images_crop = images_crop.view(batch_size, n_hypotheses, *images_crop.shape[-3:])
        renders = last_iter_outputs.renders
        renders = renders.view(
            batch_size, n_hypotheses, n_views, renders.shape[1] // n_views, *renders.shape[-2:]
        )

        KV_crop = last_iter_outputs.KV_crop
        KV_crop = KV_crop.view(batch_size, n_hypotheses, *KV_crop.shape[1:])

        K_crop = last_iter_outputs.K_crop
        K_crop = K_crop.view(batch_size, n_hypotheses, *K_crop.shape[1:])

        tCR = last_iter_outputs.tCR
        tCR = tCR.view(batch_size, n_hypotheses, *tCR.shape[1:])

        TCV_O = last_iter_outputs.TCV_O_input
        TCV_O = TCV_O.view(batch_size, n_hypotheses, n_views, *TCV_O.shape[2:])

        TCV0_O = TCV_O[..., 0, :4, :4]
        TCV0_R = TCV_O[..., 0, :4, :4].clone()
        TCV0_R[..., :3, [-1]] = tCR.unsqueeze(-1)
        TO_R = invert_transform_matrices(TCV0_O) @ TCV0_R

        TCV_R = TCV_O.clone()
        TCV_R = TCV_R @ TO_R.unsqueeze(2)

        points_orig = torch.zeros(1, 1, 3).to(K_crop.device)

        for batch_idx in range(min(cfg.vis_batch_size, batch_size)):
            for view_idx in range(n_views):
                row = []
                for init_idx in range(n_hypotheses):
                    image_crop_ = images_crop[batch_idx, init_idx]
                    render_ = renders[batch_idx, init_idx, view_idx]
                    KV_crop_ = KV_crop[[batch_idx], init_idx, view_idx]
                    TCO_ = TCV_O[[batch_idx], init_idx, view_idx]
                    TCR_ = TCV_R[[batch_idx], init_idx, view_idx]


                    image_crop_ = add_mask_to_image(image_crop_[:3], image_crop_[-1])
                    image_crop_ = add_mask_to_image(image_crop_[:3], render_[-1], "green")
                    f = plotter.plot_image(image_crop_)
                    f.title.text = f"init of iteration {n_iterations}"
                    row.append(f)

                    n_channels = render_.shape[0]
                    ref_point_uv = project_points_robust(points_orig, KV_crop_, TCR_).flatten()
                    origin_uv = project_points_robust(points_orig, KV_crop_, TCO_).flatten()
                    f = plotter.plot_image(render_[:3])
                    f.circle(
                        [int(ref_point_uv[0])],
                        [int(render_.shape[1] - ref_point_uv[1])],
                        color="red",
                    )
                    f.circle(
                        [int(origin_uv[0])], [int(render_.shape[1] - origin_uv[1])], color="green"
                    )
                    f.title.text = f"idx={batch_idx},view={view_idx},init={init_idx}"
                    if cfg.predict_rendered_views_logits:
                        assert is_hypothesis_positive is not None
                        is_positive = is_hypothesis_positive[batch_idx, init_idx]
                        f.title.text = (
                            f"idx={batch_idx},view={view_idx},init={init_idx},target={is_positive}"
                        )
                    row.append(f)

                    if n_channels == 6:
                        f = plotter.plot_image(render_[3:])
                        row.append(f)

                grid.append(row)

        doc = curdoc()
        sizing_mode = "scale_width" if is_notebook else None
        plot = gridplot(grid, toolbar_location=None, sizing_mode=sizing_mode)
        if is_notebook:
            meters["bokeh_doc_hypotheses"] = plot
            meters["bokeh_grid"] = grid
        else:
            doc.add_root(plot)
            meters["bokeh_doc_hypotheses"] = doc.to_json()
            doc.clear()

    if debug_dict is not None:
        debug_dict["outputs"] = outputs
        debug_dict["time_render"] = time_render

    return loss
