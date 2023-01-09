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
from pathlib import Path
from typing import Any, Dict, Optional

# Third Party
import torch
from omegaconf import OmegaConf

# MegaPose
import megapose
import megapose.datasets.datasets_cfg
import megapose.evaluation.eval_runner
import megapose.inference.utils
from megapose.datasets.datasets_cfg import make_object_dataset
from megapose.evaluation.eval_config import EvalConfig
from megapose.evaluation.evaluation_runner import EvaluationRunner
from megapose.evaluation.meters.modelnet_meters import ModelNetErrorMeter
from megapose.evaluation.prediction_runner import PredictionRunner
from megapose.evaluation.runner_utils import format_results
from megapose.inference.depth_refiner import DepthRefiner
from megapose.inference.icp_refiner import ICPRefiner
from megapose.inference.pose_estimator import PoseEstimator
from megapose.inference.teaserpp_refiner import TeaserppRefiner
from megapose.lib3d.rigid_mesh_database import MeshDataBase
from megapose.utils.distributed import get_rank, get_tmp_dir
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def generate_save_key(detection_type: str, coarse_estimation_type: str) -> str:
    return f"{detection_type}+{coarse_estimation_type}"


def get_save_dir(cfg: EvalConfig) -> Path:
    """Returns a save dir.

    Example

    .../ycbv.bop19/gt+SO3_grid

    You must remove the '.bop19' from the name in order for the
    bop_toolkit_lib to process it correctly.

    """
    save_key = generate_save_key(cfg.inference.detection_type, cfg.inference.coarse_estimation_type)

    assert cfg.save_dir is not None
    assert cfg.ds_name is not None
    save_dir = Path(cfg.save_dir) / cfg.ds_name / save_key
    return save_dir


def run_eval(
    cfg: EvalConfig,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run eval for a single setting on a single dataset.

    A single setting is a (detection_type, coarse_estimation_type) such
    as ('maskrcnn', 'SO3_grid').

    Saves the results to the directory below (if one is not passed in).

    cfg.save_dir / ds_name / eval_key / results.pth.tar

    Returns:
        dict: If you are rank_0 process, otherwise returns None

    """

    save_key = generate_save_key(cfg.inference.detection_type, cfg.inference.coarse_estimation_type)
    if save_dir is None:
        save_dir = get_save_dir(cfg)

    cfg.save_dir = str(save_dir)

    logger.info(f"Running eval on ds_name={cfg.ds_name} with setting={save_key}")

    # Load the dataset
    ds_kwargs = dict(load_depth=True)
    scene_ds = megapose.datasets.datasets_cfg.make_scene_dataset(cfg.ds_name, **ds_kwargs)
    urdf_ds_name, obj_ds_name = megapose.datasets.datasets_cfg.get_obj_ds_info(cfg.ds_name)

    # drop frames if this was specified
    if cfg.n_frames is not None:
        scene_ds.frame_index = scene_ds.frame_index[: cfg.n_frames].reset_index(drop=True)

    # Load detector model
    if cfg.inference.detection_type == "detector":
        assert cfg.detector_run_id is not None
        detector_model = megapose.inference.utils.load_detector(cfg.detector_run_id)
    elif cfg.inference.detection_type == "gt":
        detector_model = None
    else:
        raise ValueError(f"Unknown detection_type={cfg.inference.detection_type}")

    # Load the coarse and mrefiner models
    # Needed to deal with the fact that str and Optional[str] are incompatible types.
    # See https://stackoverflow.com/a/53287330
    assert cfg.coarse_run_id is not None
    assert cfg.refiner_run_id is not None
    coarse_model, refiner_model, mesh_db = megapose.inference.utils.load_pose_models(
        coarse_run_id=cfg.coarse_run_id,
        refiner_run_id=cfg.refiner_run_id,
        n_workers=cfg.n_rendering_workers,
        obj_ds_name=obj_ds_name,
        urdf_ds_name=urdf_ds_name,
        force_panda3d_renderer=True,
    )

    renderer = refiner_model.renderer

    if cfg.inference.run_depth_refiner:
        if cfg.inference.depth_refiner == "icp":
            depth_refiner: Optional[DepthRefiner] = ICPRefiner(mesh_db, renderer)
        elif cfg.inference.depth_refiner == "teaserpp":
            depth_refiner = TeaserppRefiner(mesh_db, renderer)
        else:
            depth_refiner = None
    else:
        depth_refiner = None

    pose_estimator = PoseEstimator(
        refiner_model=refiner_model,
        coarse_model=coarse_model,
        detector_model=detector_model,
        depth_refiner=depth_refiner,
        SO3_grid_size=cfg.inference.SO3_grid_size,
    )

    # Create the prediction runner and run inference
    assert cfg.batch_size == 1
    pred_runner = PredictionRunner(
        scene_ds=scene_ds,
        inference_cfg=cfg.inference,
        batch_size=cfg.batch_size,
        n_workers=cfg.n_dataloader_workers,
    )

    # Run inference
    with torch.no_grad():
        all_preds = pred_runner.get_predictions(pose_estimator)

    logger.info(f"Done with inference on ds={cfg.ds_name}")
    logger.info(f"Predictions: {all_preds.keys()}")

    # Keep it simple for now. Only eval the final prediction
    eval_keys = set()
    eval_keys.add("refiner/final")
    eval_keys.add("depth_refiner")

    # Compute eval metrics
    # TODO (lmanuelli): Fix this up.
    # TODO (ylabbe): Clean this.
    eval_metrics, eval_dfs = dict(), dict()
    if not cfg.skip_evaluation:
        assert "modelnet" in cfg.ds_name
        object_ds = make_object_dataset(obj_ds_name)
        mesh_db = MeshDataBase.from_object_ds(object_ds)
        meters = {
            "modelnet": ModelNetErrorMeter(mesh_db, sample_n_points=None),
        }
        eval_runner = EvaluationRunner(
            scene_ds,
            meters,
            n_workers=cfg.n_dataloader_workers,
            cache_data=False,
            batch_size=1,
            sampler=pred_runner.sampler,
        )
        for preds_k, preds in all_preds.items():
            do_eval = preds_k in set(eval_keys)
            if do_eval:
                logger.info(f"Evaluation of predictions: {preds_k} (n={len(preds)})")
                eval_metrics[preds_k], eval_dfs[preds_k] = eval_runner.evaluate(preds)
            else:
                logger.info(f"Skipped: {preds_k} (n={len(all_preds)})")

    # Gather predictions from different processes
    logger.info("Waiting on barrier.")
    torch.distributed.barrier()
    logger.info("Gathering predictions from all processes.")
    for k, v in all_preds.items():
        all_preds[k] = v.gather_distributed(tmp_dir=get_tmp_dir()).cpu()

    torch.distributed.barrier()
    logger.info("Finished gathering predictions from all processes.")

    # Save results to disk
    if get_rank() == 0:
        results_path = save_dir / "results.pth.tar"
        assert cfg.save_dir is not None
        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Finished evaluation on {cfg.ds_name}, setting={save_key}")
        results = format_results(all_preds, eval_metrics, eval_dfs)
        torch.save(results, results_path)
        torch.save(results.get("summary"), save_dir / "summary.pth.tar")
        torch.save(results.get("predictions"), save_dir / "predictions.pth.tar")
        torch.save(results.get("dfs"), save_dir / "error_dfs.pth.tar")
        torch.save(results.get("metrics"), save_dir / "metrics.pth.tar")
        (save_dir / "summary.txt").write_text(results.get("summary_txt", ""))
        (save_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        logger.info(f"Saved predictions+metrics in {save_dir}")

        return {
            "results": results,
            "pred_keys": list(all_preds.keys()),
            "save_dir": save_dir,
            "results_path": results_path,
        }
    else:
        return None
