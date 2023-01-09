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


from __future__ import annotations

# Standard Library
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Third Party
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# MegaPose
import megapose
import megapose.inference.utils
import megapose.utils.tensor_collection as tc
from megapose.inference.depth_refiner import DepthRefiner
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.lib3d.cosypose_ops import TCO_init_from_boxes_autodepth_with_R
from megapose.training.utils import CudaTimer, SimpleTimer
from megapose.utils import transform_utils
from megapose.utils.logging import get_logger
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.utils.timer import Timer

logger = get_logger(__name__)


class PoseEstimator(torch.nn.Module):
    """Performs inference for pose estimation."""

    def __init__(
        self,
        refiner_model: Optional[torch.nn.Module] = None,
        coarse_model: Optional[torch.nn.Module] = None,
        detector_model: Optional[torch.nn.Module] = None,
        depth_refiner: Optional[DepthRefiner] = None,
        bsz_objects: int = 8,
        bsz_images: int = 256,
        SO3_grid_size: int = 576,
    ) -> None:

        super().__init__()
        self.coarse_model = coarse_model
        self.refiner_model = refiner_model
        self.detector_model = detector_model
        self.depth_refiner = depth_refiner
        self.bsz_objects = bsz_objects
        self.bsz_images = bsz_images

        # Load the SO3 grid if was passed in
        if SO3_grid_size is not None:
            self.load_SO3_grid(SO3_grid_size)

        # load cfg and mesh_db from refiner model
        if self.refiner_model is not None:
            self.cfg = self.refiner_model.cfg
            self.mesh_db = self.refiner_model.mesh_db
        elif self.coarse_model is not None:
            self.cfg = self.coarse_model.cfg
            self.mesh_db = self.coarse_model.mesh_db
        else:
            raise ValueError("At least one of refiner_model or " " coarse_model must be specified.")

        self.eval()

        self.keep_all_outputs = False
        self.keep_all_coarse_outputs = False
        self.refiner_outputs = None
        self.coarse_outputs = None
        self.debug_dict: dict = dict()

    def load_SO3_grid(self, grid_size: int) -> None:
        """Loads the SO(3) grid."""
        self._SO3_grid = transform_utils.load_SO3_grid(grid_size)
        self._SO3_grid = self._SO3_grid.cuda()

    @torch.no_grad()
    def forward_refiner(
        self,
        observation: ObservationTensor,
        data_TCO_input: PoseEstimatesType,
        n_iterations: int = 5,
        keep_all_outputs: bool = False,
        cuda_timer: bool = False,
        **refiner_kwargs,
    ) -> Tuple[dict, dict]:
        """Runs the refiner model for the specified number of iterations.


        Will actually use the batched_model_predictions to stay within
        batch size limit.

        Returns:
            (preds, extra_data)

            preds:
                A dict with keys 'refiner/iteration={n}' for
                n=1,...,n_iterations

                Each value is a data_TCO_type.

            extra_data:
                A dict containing additional information such as timing

        """

        timer = Timer()
        timer.start()

        start_time = time.time()

        assert self.refiner_model is not None

        B = data_TCO_input.poses.shape[0]
        ids = torch.arange(B)
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=self.bsz_objects)
        device = observation.images.device

        preds = defaultdict(list)
        all_outputs = []

        model_time = 0.0

        for (batch_idx, (batch_ids,)) in enumerate(dl):
            data_TCO_input_ = data_TCO_input[batch_ids]
            df_ = data_TCO_input_.infos
            TCO_input_ = data_TCO_input_.poses

            # Add some additional fields to df_
            df_["refiner_batch_idx"] = batch_idx
            df_["refiner_instance_idx"] = np.arange(len(df_))

            labels_ = df_["label"].tolist()
            batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            images_ = observation.images[batch_im_ids_]
            K_ = observation.K[batch_im_ids_]

            timer_ = CudaTimer(enabled=cuda_timer)
            timer_.start()
            outputs_ = self.refiner_model(
                images=images_,
                K=K_,
                TCO=TCO_input_,
                n_iterations=n_iterations,
                labels=labels_,
                **refiner_kwargs,
            )
            timer_.stop()
            model_time += timer_.elapsed()

            if keep_all_outputs:
                all_outputs.append(outputs_)

            # Collect the data
            for n in range(1, n_iterations + 1):
                iter_outputs = outputs_[f"iteration={n}"]

                infos = df_
                batch_preds = PandasTensorCollection(
                    infos,
                    poses=iter_outputs.TCO_output,
                    poses_input=iter_outputs.TCO_input,
                    K_crop=iter_outputs.K_crop,
                    K=iter_outputs.K,
                    boxes_rend=iter_outputs.boxes_rend,
                    boxes_crop=iter_outputs.boxes_crop,
                )

                preds[f"iteration={n}"].append(batch_preds)

        for k, v in preds.items():
            preds[k] = tc.concatenate(v)

        timer.stop()

        elapsed = time.time() - start_time

        extra_data = {
            "n_iterations": n_iterations,
            "outputs": all_outputs,
            "model_time": model_time,
            "time": elapsed,
        }

        logger.debug(
            f"Pose prediction on {B} poses (n_iterations={n_iterations}):" f" {timer.stop()}"
        )

        return preds, extra_data

    @torch.no_grad()
    def forward_scoring_model(
        self,
        observation: ObservationTensor,
        data_TCO: PoseEstimatesType,
        cuda_timer: bool = False,
        return_debug_data: bool = False,
    ) -> Tuple[PoseEstimatesType, dict]:

        """Score the estimates using the coarse model.


        Adds the 'pose_score' field to data_TCO.infos

        Modifies PandasTensorCollection in-place.
        """

        start_time = time.time()

        assert self.coarse_model is not None
        bsz_images = self.bsz_images

        # Add M rows for each row in detections
        df = data_TCO.infos

        ids = torch.arange(len(df))
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=bsz_images)
        device = observation.images.device

        images_crop_list = []
        renders_list = []
        logits_list = []
        scores_list = []

        render_time = 0
        model_time = 0

        for (batch_ids,) in dl:
            data_TCO_ = data_TCO[batch_ids]
            df_ = data_TCO_.infos
            TCO_ = data_TCO_.poses
            labels_ = df_["label"].tolist()
            batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            images_ = observation.images[batch_im_ids_]
            K_ = observation.K[batch_im_ids_]

            out_ = self.coarse_model.forward_coarse(
                images=images_,
                K=K_,
                labels=labels_,
                TCO_input=TCO_,
                cuda_timer=cuda_timer,
                return_debug_data=return_debug_data,
            )

            render_time += out_["render_time"]
            model_time += out_["model_time"]

            logits_list.append(out_["logits"])
            scores_list.append(out_["scores"])

            if return_debug_data:
                images_crop_list.append(out_["images_crop"])
                renders_list.append(out_["renders"])

        debug_data = dict()

        # Combine together the data from the different batches
        logits = torch.cat(logits_list)
        scores = torch.cat(scores_list)
        if return_debug_data:
            images_crop: torch.tensor = torch.cat(images_crop_list)
            renders: torch.tensor = torch.cat(renders_list)

            H = images_crop.shape[2]
            W = images_crop.shape[3]

            debug_data = {
                "images_crop": images_crop,
                "renders": renders,
            }

        df["pose_logit"] = logits.cpu().numpy()
        df["pose_score"] = scores.cpu().numpy()

        elapsed = time.time() - start_time

        timing_str = (
            f"time: {elapsed:.2f}, model_time: {model_time:.2f}, render_time: {render_time:.2f}"
        )

        extra_data = {
            "render_time": render_time,
            "model_time": model_time,
            "time": elapsed,
            "logits": logits,  # [B,]
            "scores": scores,  # [B,]
            "debug": debug_data,
            "n_batches": len(dl),
            "timing_str": timing_str,
        }

        data_TCO.infos = df
        return data_TCO, extra_data

    @torch.no_grad()
    def forward_coarse_model(
        self,
        observation: ObservationTensor,
        detections: DetectionsType,
        cuda_timer: bool = False,
        return_debug_data: bool = False,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Generates pose hypotheses and scores them with the coarse model.

        - Generates coarse hypotheses using the SO(3) grid.
        - Scores them using the coarse model.
        """

        start_time = time.time()

        megapose.inference.types.assert_detections_valid(detections)

        bsz_images = self.bsz_images
        coarse_model = self.coarse_model
        SO3_grid = self._SO3_grid
        B = len(detections)
        M = self._SO3_grid.shape[0]

        # Add M rows for each row in detections
        df = detections.infos
        df_concat = []
        for tc_idx, row in df.iterrows():
            df_tmp = pd.DataFrame([row] * M)
            df_tmp["hypothesis_id"] = list(range(M))
            df_tmp["bbox_id"] = tc_idx
            df_concat.append(df_tmp)

        # Each row in detections is now repeated M times
        # and has the hypothesis_id field.
        df_hypotheses = pd.concat(df_concat)
        df_hypotheses.reset_index()

        ids = torch.arange(len(df_hypotheses))
        ds = TensorDataset(ids)
        dl = DataLoader(ds, batch_size=bsz_images)
        device = observation.images.device

        images_crop_list = []
        renders_list = []
        logits_list = []
        scores_list = []
        bboxes_list = []

        render_time = 0
        model_time = 0
        TCO_init = []

        for (batch_ids,) in dl:

            # b = bsz_images
            df_ = df_hypotheses.iloc[batch_ids.cpu().numpy()]

            batch_im_ids_ = torch.as_tensor(df_["batch_im_id"].values, device=device)

            m_idx = torch.as_tensor(df_["hypothesis_id"].values, device=device)

            labels_ = df_["label"].tolist()
            bbox_ids_ = torch.as_tensor(df_["bbox_id"].values, device=device)

            images_ = observation.images[batch_im_ids_]
            K_ = observation.K[batch_im_ids_]

            # We are indexing into the original detections TensorCollection.
            bboxes_ = detections.bboxes[bbox_ids_]
            meshes_ = coarse_model.mesh_db.select(labels_)

            # [b,N,3]
            points_ = meshes_.points

            # [b,3,3]
            SO3_grid_ = SO3_grid[m_idx]

            # Compute the initial poses
            # [b,4,4]
            TCO_init_ = TCO_init_from_boxes_autodepth_with_R(
                bboxes_,
                points_,
                K_,
                SO3_grid_,
            )

            del points_

            out_ = coarse_model.forward_coarse(
                images=images_,
                K=K_,
                labels=labels_,
                TCO_input=TCO_init_,
                cuda_timer=cuda_timer,
                return_debug_data=return_debug_data,
            )

            render_time += out_["render_time"]
            model_time += out_["model_time"]

            logits_list.append(out_["logits"])
            scores_list.append(out_["scores"])
            bboxes_list.append(bboxes_)
            TCO_init.append(TCO_init_)

            if return_debug_data:
                images_crop_list.append(out_["images_crop"])
                renders_list.append(out_["renders"])

        # Combine all the information into data_TCO_type
        logits = torch.cat(logits_list)
        logits = logits.reshape([B, M])

        scores = torch.cat(scores_list)
        scores = scores.reshape([B, M])

        bboxes = torch.cat(bboxes_list, dim=0)

        # [B*M, 4, 4]
        TCO = torch.cat(TCO_init)
        TCO_reshape = TCO.reshape([B, M, 4, 4])

        debug_data = dict()

        if return_debug_data:
            images_crop = torch.cat(images_crop_list)
            renders = torch.cat(renders_list)

            H = images_crop.shape[2]
            W = images_crop.shape[3]

            debug_data = {
                "images_crop": images_crop.reshape([B, M, -1, H, W]),
                "renders": renders.reshape([B, M, -1, H, W]),
            }

        df_hypotheses["coarse_logit"] = logits.flatten().cpu().numpy()
        df_hypotheses["coarse_score"] = scores.flatten().cpu().numpy()

        elapsed = time.time() - start_time

        timing_str = (
            f"time: {elapsed:.2f}, model_time: {model_time:.2f}, render_time: {render_time:.2f}"
        )

        extra_data = {
            "render_time": render_time,
            "model_time": model_time,
            "time": elapsed,
            "logits": logits,  # [B,]
            "scores": scores,  # [B,]
            "TCO": TCO_reshape,  # [B,M,4,4]
            "debug": debug_data,
            "n_batches": len(dl),
            "timing_str": timing_str,
        }

        data_TCO = PandasTensorCollection(df_hypotheses, poses=TCO, bboxes=bboxes)
        return data_TCO, extra_data

    @torch.no_grad()
    def forward_detection_model(
        self,
        observation: ObservationTensor,
        *args: Any,
        **kwargs: Any,
    ) -> DetectionsType:
        """Runs the detector."""

        return self.detector_model.get_detections(observation, *args, **kwargs)

    def run_depth_refiner(
        self,
        observation: ObservationTensor,
        predictions: PoseEstimatesType,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs the depth refiner."""
        assert self.depth_refiner is not None, "You must specify a depth refiner"
        depth = observation.depth
        K = observation.K

        refined_preds, extra_data = self.depth_refiner.refine_poses(predictions, depth=depth, K=K)

        return refined_preds, extra_data

    @torch.no_grad()
    def run_inference_pipeline(
        self,
        observation: ObservationTensor,
        detections: Optional[DetectionsType] = None,
        run_detector: Optional[bool] = None,
        n_refiner_iterations: int = 5,
        n_pose_hypotheses: int = 1,
        keep_all_refiner_outputs: bool = False,
        detection_filter_kwargs: Optional[dict] = None,
        run_depth_refiner: bool = False,
        bsz_images: Optional[int] = None,
        bsz_objects: Optional[int] = None,
        cuda_timer: bool = False,
        coarse_estimates: Optional[PoseEstimatesType] = None,
    ) -> Tuple[PoseEstimatesType, dict]:
        """Runs the entire pose estimation pipeline.

        Performs the following steps

        1. Run detector (or use detections that were passed in)
        2. Run coarse model
        3. Extract n_pose_hypotheses from coarse model
        4. Run refiner for n_refiner_iterations
        5. Score refined hypotheses
        6. Select highest scoring refined hypotheses.

        Returns:
            data_TCO_final: final predictions
            data: Dict containing additional data about the different
                steps in the pipeline.

        """

        timing_str = ""
        timer = SimpleTimer()
        timer.start()

        if bsz_images is not None:
            self.bsz_images = bsz_images

        if bsz_objects is not None:
            self.bsz_objects = bsz_objects

        if coarse_estimates is None:
            assert detections is not None or run_detector, (
                "You must " "either pass in `detections` or set run_detector=True"
            )
            if detections is None and run_detector:
                start_time = time.time()
                detections = self.forward_detection_model(observation)
                detections = detections.cuda()
                elapsed = time.time() - start_time
                timing_str += f"detection={elapsed:.2f}, "

            # Ensure that detections has the instance_id column
            assert detections is not None
            detections = megapose.inference.utils.add_instance_id(detections)

            # Filter detections
            if detection_filter_kwargs is not None:
                detections = megapose.inference.utils.filter_detections(
                    detections, **detection_filter_kwargs
                )

            # Run the coarse estimator using gt_detections
            data_TCO_coarse, coarse_extra_data = self.forward_coarse_model(
                observation=observation,
                detections=detections,
                cuda_timer=cuda_timer,
            )
            timing_str += f"coarse={coarse_extra_data['time']:.2f}, "

            # Extract top-K coarse hypotheses
            data_TCO_filtered = self.filter_pose_estimates(
                data_TCO_coarse, top_K=n_pose_hypotheses, filter_field="coarse_logit"
            )
        else:
            data_TCO_coarse = coarse_estimates
            coarse_extra_data = None
            data_TCO_filtered = coarse_estimates

        preds, refiner_extra_data = self.forward_refiner(
            observation,
            data_TCO_filtered,
            n_iterations=n_refiner_iterations,
            keep_all_outputs=keep_all_refiner_outputs,
            cuda_timer=cuda_timer,
        )
        data_TCO_refined = preds[f"iteration={n_refiner_iterations}"]
        timing_str += f"refiner={refiner_extra_data['time']:.2f}, "

        # Score the refined poses using the coarse model.
        data_TCO_scored, scoring_extra_data = self.forward_scoring_model(
            observation,
            data_TCO_refined,
            cuda_timer=cuda_timer,
        )
        timing_str += f"scoring={scoring_extra_data['time']:.2f}, "

        # Extract the highest scoring pose estimate for each instance_id
        data_TCO_final_scored = self.filter_pose_estimates(
            data_TCO_scored, top_K=1, filter_field="pose_logit"
        )

        # Optionally run ICP or TEASER++
        if run_depth_refiner:
            depth_refiner_start = time.time()
            data_TCO_depth_refiner, _ = self.run_depth_refiner(observation, data_TCO_final_scored)
            data_TCO_final = data_TCO_depth_refiner
            depth_refiner_time = time.time() - depth_refiner_start
            timing_str += f"depth refiner={depth_refiner_time:.2f}"
        else:
            data_TCO_depth_refiner = None
            data_TCO_final = data_TCO_final_scored

        timer.stop()
        timing_str = f"total={timer.elapsed():.2f}, {timing_str}"

        extra_data: dict = dict()
        extra_data["coarse"] = {"preds": data_TCO_coarse, "data": coarse_extra_data}
        extra_data["coarse_filter"] = {"preds": data_TCO_filtered}
        extra_data["refiner_all_hypotheses"] = {"preds": preds, "data": refiner_extra_data}
        extra_data["scoring"] = {"preds": data_TCO_scored, "data": scoring_extra_data}
        extra_data["refiner"] = {"preds": data_TCO_final_scored, "data": refiner_extra_data}
        extra_data["timing_str"] = timing_str
        extra_data["time"] = timer.elapsed()

        if run_depth_refiner:
            extra_data["depth_refiner"] = {"preds": data_TCO_depth_refiner}

        return data_TCO_final, extra_data

    def filter_pose_estimates(
        self,
        data_TCO: PoseEstimatesType,
        top_K: int,
        filter_field: str,
        ascending: bool = False,
    ) -> PoseEstimatesType:
        """Filter the pose estimates by retaining only the top-K coarse model scores.

        Retain only the top_K estimates corresponding to each hypothesis_id

        Args:
            top_K: how many estimates to retain
            filter_field: The field to filter estimates by
        """

        df = data_TCO.infos

        group_cols = ["batch_im_id", "label", "instance_id"]
        # Logic from https://stackoverflow.com/a/40629420
        df = df.sort_values(filter_field, ascending=ascending).groupby(group_cols).head(top_K)

        data_TCO_filtered = data_TCO[df.index.tolist()]

        return data_TCO_filtered
