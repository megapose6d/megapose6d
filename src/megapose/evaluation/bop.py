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
import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Third Party
import numpy as np
import torch
from tqdm import tqdm

# MegaPose
from megapose.config import BOP_TOOLKIT_DIR, LOCAL_DATA_DIR, PROJECT_DIR
from megapose.evaluation.eval_config import BOPEvalConfig

# Note we are actually using the bop_toolkit_lib that is directly conda installed
# inside the docker image. This is just to access the scripts.
POSE_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19.py"
DETECTION_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop_coco.py"
DUMMY_EVAL_SCRIPT_PATH = BOP_TOOLKIT_DIR / "scripts/eval_bop19_dummy.py"


# Third Party
import bop_toolkit_lib
from bop_toolkit_lib import inout  # noqa


def main():
    parser = argparse.ArgumentParser("Bop evaluation")
    parser.add_argument("--results_path", default="", type=str)
    parser.add_argument("--eval_dir", default="", type=str)
    parser.add_argument("--dataset", default="", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--method", default="", type=str)
    parser.add_argument("--detection-method", default="", type=str)
    parser.add_argument("--csv_path", default="", type=str)
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    run_evaluation(args)


def convert_results_to_coco(results_path, out_json_path, detection_method):
    sys.path = [p for p in sys.path if "bop_toolkit" not in str(p)]
    TOOLKIT_MASTER_DIR = Path(PROJECT_DIR).parent / "bop_toolkit_master"
    sys.path.append(TOOLKIT_MASTER_DIR.as_posix())
    importlib.reload(sys.modules["bop_toolkit_lib"])
    # Third Party
    from bop_toolkit_lib.pycoco_utils import binary_mask_to_polygon

    results = torch.load(results_path)
    predictions = results["predictions"][detection_method]
    print("Detections from:", results_path)
    print("Detection method:", detection_method)
    print("Number of detections: ", len(predictions))

    infos = []
    for n in tqdm(range(len(predictions))):
        row = predictions.infos.iloc[n]
        x1, y1, x2, y2 = predictions.bboxes[n].tolist()
        x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
        score = row.score
        category_id = int(row.label.split("_")[-1])
        mask = predictions.masks[n].numpy().astype(np.uint8)
        rle = binary_mask_to_polygon(mask)
        info = dict(
            scene_id=int(row.scene_id),
            view_id=int(row.view_id),
            category_id=category_id,
            bbox=[x, y, w, h],
            score=score,
            segmentation=rle,
        )
        infos.append(info)
    Path(out_json_path).write_text(json.dumps(infos))
    return


def convert_results_to_bop(
    results_path: Path, out_csv_path: Path, method: str,
    use_pose_score: bool = True
):
    predictions = torch.load(results_path)["predictions"]
    predictions = predictions[method]
    print("Predictions from:", results_path)
    print("Method:", method)
    print("Number of predictions: ", len(predictions))

    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split("_")[-1])
        if use_pose_score:
            score = row.pose_score
        else:
            score = row.score
        if "time" in row:
            time = row.time
        else:
            time = -1
        pred = dict(
            scene_id=row.scene_id,
            im_id=row.view_id,
            obj_id=obj_id,
            score=score,
            t=t,
            R=R,
            time=time,
        )
        preds.append(pred)
    print("Wrote:", out_csv_path)
    Path(out_csv_path).parent.mkdir(exist_ok=True)
    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path

def _run_bop_evaluation(filename, eval_dir, eval_detection=False, dummy=False):
    myenv = os.environ.copy()
    myenv["PYTHONPATH"] = BOP_TOOLKIT_DIR.as_posix()
    ld_library_path = os.environ['LD_LIBRARY_PATH']
    conda_prefix = os.environ['CONDA_PREFIX']
    myenv["LD_LIBRARY_PATH"] = f'{conda_prefix}/lib:{ld_library_path}'
    myenv["BOP_DATASETS_PATH"] = str(LOCAL_DATA_DIR / "bop_datasets")
    myenv["BOP_RESULTS_PATH"] = str(eval_dir)
    myenv["BOP_EVAL_PATH"] = str(eval_dir)
    if dummy:
        cmd = [
            "python",
            str(DUMMY_EVAL_SCRIPT_PATH),
            "--renderer_type",
            "cpp",
            "--result_filenames",
            filename,
        ]
    else:
        if eval_detection:
            cmd = [
                "python",
                str(DETECTION_EVAL_SCRIPT_PATH),
                "--result_filenames",
                filename,
            ]
        else:
            cmd = [
                "python",
                str(POSE_EVAL_SCRIPT_PATH),
                "--result_filenames",
                filename,
                "--renderer_type",
                "cpp",
            ]

    subprocess.call(cmd, env=myenv, cwd=BOP_TOOLKIT_DIR.as_posix())


def run_evaluation(cfg: BOPEvalConfig) -> None:
    """Runs the bop evaluation for the given setting."""
    print(cfg)
    results_path = Path(cfg.results_path)
    eval_dir = Path(cfg.eval_dir)

    if cfg.dataset == "hb" and cfg.split == "test":
        cfg.convert_only = True
    if cfg.dataset == "itodd" and cfg.split == "test":
        cfg.convert_only = True

    scores_pose_path = None
    if cfg.method is not None:
        method = cfg.method.replace("/", "-")
        method = method.replace("_", "-")

        # The csv file needs naming like <anything>_ycbv-test.csv since
        # this is what is expected by bop_toolkit_lib
        csv_path = eval_dir / f"{method}_{cfg.dataset.split('.')[0]}-{cfg.split}.csv"

        convert_results_to_bop(results_path, csv_path, cfg.method)

        if not cfg.convert_only:
            _run_bop_evaluation(csv_path, cfg.eval_dir, eval_detection=False)
        scores_pose_path = eval_dir / csv_path.with_suffix("").name / "scores_bop19.json"

    scores_detection_path = None
    if cfg.detection_method is not None:
        raise NotImplementedError
        method = cfg.detection_method.replace("/", "-")
        method = method.replace("_", "-")
        json_path = eval_dir / f"{method}_{cfg.dataset}-{cfg.split}.json"
        convert_results_to_coco(results_path, json_path, cfg.detection_method)
        if not cfg.convert_only:
            _run_bop_evaluation(json_path, cfg.eval_dir, eval_detection=True)
        scores_detection_path = (
            eval_dir / csv_path.with_suffix("").name / "scores_bop22_coco_bbox.json"
        )

    return scores_pose_path, scores_detection_path


if __name__ == "__main__":
    main()
