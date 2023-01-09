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
import copy
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third Party
from omegaconf import OmegaConf

# MegaPose
import megapose.evaluation.bop
import megapose.evaluation.evaluation
from megapose.bop_config import (
    PBR_COARSE,
    PBR_DETECTORS,
    PBR_REFINER,
    SYNT_REAL_COARSE,
    SYNT_REAL_DETECTORS,
    SYNT_REAL_REFINER,
)
from megapose.config import (
    DEBUG_RESULTS_DIR,
    EXP_DIR,
    MODELNET_TEST_CATEGORIES,
    RESULTS_DIR,
)
from megapose.evaluation.eval_config import (
    BOPEvalConfig,
    EvalConfig,
    FullEvalConfig,
    HardwareConfig,
)
from megapose.evaluation.evaluation import generate_save_key, run_eval
from megapose.utils.distributed import get_rank, get_world_size, init_distributed_mode
from megapose.utils.logging import get_logger, set_logging_level

logger = get_logger(__name__)

BOP_DATASET_NAMES = [
    "lm",
    "lmo",
    "tless",
    "tudl",
    "icbin",
    "itodd",
    "hb",
    "ycbv",
    # 'hope',
]

BOP_TEST_DATASETS = [
    "lmo.bop19",
    "tless.bop19",
    "tudl.bop19",
    "icbin.bop19",
    "itodd.bop19",
    "hb.bop19",
    "ycbv.bop19",
]


MODELNET_TEST_DATASETS = [f"modelnet.{category}.test" for category in MODELNET_TEST_CATEGORIES]


def create_eval_cfg(
    cfg: EvalConfig,
    detection_type: str,
    coarse_estimation_type: str,
    ds_name: str,
) -> Tuple[str, EvalConfig]:

    cfg = copy.deepcopy(cfg)

    cfg.inference.detection_type = detection_type
    cfg.inference.coarse_estimation_type = coarse_estimation_type
    cfg.ds_name = ds_name

    if detection_type == "detector":
        assert cfg.detector_run_id is not None

        ds_name_root = cfg.ds_name.split(".")[0]
        if cfg.detector_run_id == "bop_pbr":
            cfg.detector_run_id = PBR_DETECTORS[ds_name_root]
    elif detection_type == "gt":
        pass
    else:
        raise ValueError(f"Unknown detector type {cfg.detector_type}")

    name = generate_save_key(detection_type, coarse_estimation_type)

    return name, cfg


def run_full_eval(cfg: FullEvalConfig) -> None:

    bop_eval_cfgs = []

    init_distributed_mode()
    print("World size", get_world_size())

    assert (
        cfg.detection_coarse_types is not None and len(cfg.detection_coarse_types) > 0
    ), "You must specify some detector_coarse_types to evaluate."

    assert cfg.ds_names is not None

    # Iterate over each dataset
    for ds_name in cfg.ds_names:

        # create the EvalConfig objects that we will call `run_eval` on
        eval_configs: Dict[str, EvalConfig] = dict()
        for (detection_type, coarse_estimation_type) in cfg.detection_coarse_types:
            name, cfg_ = create_eval_cfg(cfg, detection_type, coarse_estimation_type, ds_name)
            eval_configs[name] = cfg_

        # For each eval_cfg run the evaluation.
        # Note that the results get saved to disk
        for save_key, eval_cfg in eval_configs.items():

            # Run the inference
            if not cfg.skip_inference:
                eval_out = run_eval(eval_cfg)

            # If we are skpping the inference mimic the output that run_eval
            # would have produced so that we can run the bop_eval
            else:  # Otherwise hack the output so we can run the BOP eval
                if get_rank() == 0:
                    results_dir = megapose.evaluation.evaluation.get_save_dir(eval_cfg)
                    pred_keys = ["refiner/final"]
                    if eval_cfg.inference.run_depth_refiner:
                        pred_keys.append("depth_refiner")
                    eval_out = {
                        "results_path": results_dir / "results.pth.tar",
                        "pred_keys": pred_keys,
                        "save_dir": results_dir,
                    }

                    assert Path(
                        eval_out["results_path"]
                    ).is_file(), f"The file {eval_out['results_path']} doesn't exist"

            # Run the bop eval for each type of prediction
            if cfg.run_bop_eval and get_rank() == 0:

                bop_eval_keys = set(("refiner/final", "depth_refiner"))
                bop_eval_keys = bop_eval_keys.intersection(set(eval_out["pred_keys"]))

                for method in bop_eval_keys:
                    if not "bop19" in ds_name:
                        continue

                    bop_eval_cfg = BOPEvalConfig(
                        results_path=eval_out["results_path"],
                        dataset=ds_name,
                        split="test",
                        eval_dir=eval_out["save_dir"] / "bop_evaluation",
                        method=method,
                        convert_only=False,
                    )
                    bop_eval_cfgs.append(bop_eval_cfg)

    # Run the bop eval for each config
    # TODO (lmanuelli): Parallelize this using subprocess
    # if desired.
    if get_rank() == 0:
        if cfg.run_bop_eval:
            for bop_eval_cfg in bop_eval_cfgs:
                megapose.evaluation.bop.run_evaluation(bop_eval_cfg)

    logger.info(f"Process {get_rank()} reached end of script")


def update_cfg_debug(cfg: EvalConfig) -> FullEvalConfig:
    cfg.batch_size = 1
    cfg.n_frames = 2 * cfg.batch_size * cfg.hardware.n_gpus

    assert cfg.result_id is not None
    cfg.save_dir = str(DEBUG_RESULTS_DIR / cfg.result_id)
    return cfg


if __name__ == "__main__":
    print("Running eval")
    set_logging_level("debug")

    cli_cfg = OmegaConf.from_cli()
    logger.info(f"CLI config: \n {OmegaConf.to_yaml(cli_cfg)}")

    cfg: FullEvalConfig = OmegaConf.structured(FullEvalConfig)
    cfg.hardware = HardwareConfig(
        n_cpus=int(os.environ.get("N_CPUS", 10)),
        n_gpus=int(os.environ.get("WORLD_SIZE", 1)),
    )

    cfg = OmegaConf.merge(cfg, cli_cfg)

    assert cfg.coarse_run_id is not None
    assert cfg.refiner_run_id is not None
    assert cfg.result_id is not None

    if cfg.ds_names == "BOP_TEST_DATASETS":
        cfg.ds_names = BOP_TEST_DATASETS

    cfg.save_dir = RESULTS_DIR / cfg.result_id

    if cfg.debug:
        cfg = update_cfg_debug(cfg)

    run_full_eval(cfg)
