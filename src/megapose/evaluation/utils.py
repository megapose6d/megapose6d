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
import re

# Third Party
import numpy as np
import pandas as pd

# thirdparty
import roma
import scipy
import torch
import torchgeometry as tgm
from bop_toolkit_lib.misc import get_symmetry_transformations

# MegaPose
# megapose
from megapose.config import LOCAL_DATA_DIR
from megapose.utils.tensor_collection import PandasTensorCollection


def get_symmetry_transformations_torch(trans_list):
    S = len(trans_list)
    syms = torch.zeros([S, 4, 4])
    syms[:, 3, 3] = 1

    for idx, data in enumerate(trans_list):
        syms[idx, :3, :3] = torch.tensor(data["R"])
        syms[idx, :3, 3] = torch.tensor(data["t"]).squeeze()

    return syms


def compute_pose_error(T1, T2):
    """
    Args:
    Two sets of poses in world frame
        T1: [B,4,4]
        T2: [B,4,4]
    """

    trans_err = torch.linalg.norm(T1[..., :3, 3] - T2[..., :3, 3], dim=-1)
    R1 = T1[..., :3, :3]
    R2 = T2[..., :3, :3]
    dR = torch.matmul(R2, torch.linalg.inv(R1))
    rotvec = roma.rotmat_to_rotvec(dR)  # [B, 3]
    roterr = torch.linalg.norm(rotvec, dim=-1)
    roterr_deg = torch.rad2deg(roterr)

    return {"trans_err": trans_err, "roterr_deg": roterr_deg}


def compute_errors(preds, method, obj_dataset, max_sym_rot_step_deg=1):
    """
    Compute the errors between gt_pose and predicted pose.

    Args:

        preds: This is results['predictions'] where results is from results.pth.tar
        method: The type of method we should use for evaluation

        methods: str, e.g. 'gt_detections+coarse_init'

    """

    preds_gt = preds[f"{method}/ground_truth"]
    TCO_gt = preds_gt.poses.cuda()  # [B,4,4]
    device = TCO_gt.device
    TOC_gt = torch.linalg.inv(TCO_gt)

    for key, p in preds.items():
        if not key.startswith(method):
            continue
        if re.search("refiner/iteration=\d*$", key) or re.search("refiner/init$", key):
            pass
        else:
            continue

        print(f"Processing key {key}")

        object_labels = p.infos.label.unique()
        object_labels.sort()

        obj_info_dict = dict()
        for val in obj_dataset.objects:
            obj_info_dict[val["label"]] = val

        # Need to go by each object
        for obj_label in object_labels:
            obj_info = obj_info_dict[obj_label]
            assert obj_info["label"] == obj_label
            if obj_info["is_symmetric"]:
                bop_info = obj_info["bop_info"]
                max_sym_rot_step = np.deg2rad(max_sym_rot_step_deg)
                trans_list = get_symmetry_transformations(
                    bop_info, max_sym_disc_step=max_sym_rot_step
                )
                syms = get_symmetry_transformations_torch(trans_list)
            else:
                syms = torch.eye(4, device=device).unsqueeze(0)

            syms = syms.to(device)

            df = p.infos
            df = df[df.label == obj_label]
            idx_list = df.index.tolist()
            TCO_pred_obj = p.poses[idx_list].cuda()
            TCO_gt_obj = TCO_gt[idx_list]


            # Assumes symmetries don't have any offsets
            pts = create_default_object_pts().to(device)
            mssd_out = mssd_torch(TCO_pred_obj, TCO_gt_obj, pts, syms)
            TCO_gt_sym_obj = mssd_out["T_gt_sym"]

            err_out = compute_pose_error(TCO_gt_sym_obj, TCO_pred_obj)
            trans_err = err_out["trans_err"].cpu()
            roterr_deg = err_out["roterr_deg"].cpu()

            # Set the value on the original pandas.DataFrame
            p.infos.loc[idx_list, "trans_err"] = trans_err.tolist()
            p.infos.loc[idx_list, "rot_err_deg"] = roterr_deg.tolist()



    p_init = preds[f"{method}/refiner/init"]
    for key, p in preds.items():
        if not key.startswith(method):
            continue
        if re.search("refiner/iteration=\d*$", key):
            pass
        else:
            continue

        p.infos["trans_err_init"] = p_init.infos["trans_err"]
        p.infos["rot_err_deg_init"] = p_init.infos["rot_err_deg"]

    return preds


def create_plots(result_name):
    """Make the png figures from the"""
    pass


def create_default_object_pts():
    pts = []
    pts.append([0, 0, 0])
    pts.append([1, 0, 0])
    pts.append([-1, 0, 0])
    pts.append([0, 1, 0])
    pts.append([0, -1, 0])
    pts.append([0, 0, 1])
    pts.append([0, 0, -1])

    return torch.tensor(pts, dtype=torch.float)


def mssd_torch(T_est, T_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    Based on https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/pose_error.py#L96

    Args:
        T_est: [B,4,4] tensor, estimated pose
        T_gt: [B,4,4] tensor, ground-truth pose
        pts: [N,3] tensor, 3D model points
        syms: [S,4,4] tensor, set of symmetry transformations


    Returns:
        err: [B,] mssd
        T_gt_sym: [B,4,4] the closest symmetry aware transform
        sym: [B,4,4] symmetry transform that led to T_gt_sym
        errs: [B,S] full list of errors

    """
    B = T_est.shape[0]
    S = syms.shape[0]
    N = pts.shape[0]

    # M = B*S
    # [B,S,4,4]
    T_gt_syms = torch.matmul(T_gt.unsqueeze(1), syms)

    # [B*S,4,4]
    T_gt_syms_flat = T_gt_syms.flatten(0, 1)

    # [B,1,3]
    pts_expand = pts.unsqueeze(0).expand([B, S, N, 3])

    # [M,N,3]
    pts_expand = pts_expand.reshape([B * S, N, 3])

    # [B,S,N,3] = [B*S,4,4] * [B*S,N,3]
    pts_gt_sym = tgm.transform_points(T_gt_syms_flat, pts_expand).reshape([B, S, N, 3])

    # [B,N,3] = [B,4,4] * [B,N,3]
    pts_est = tgm.transform_points(T_est, pts.unsqueeze(0).expand([B, N, 3]))

    # [B,S,N,3]
    pts_delta = pts_gt_sym - pts_est.unsqueeze(1)

    # [B,S]
    errs = torch.linalg.norm(pts_delta, dim=-1).mean(dim=-1)

    # ([B,], [B,])
    min_err, min_idx = torch.min(errs, dim=-1)

    syms_exp = syms.unsqueeze(0).expand([B, S, 4, 4])

    # [B,4,4]
    T_sym = syms_exp[torch.arange(B), min_idx]
    T_gt_sym = T_gt_syms[torch.arange(B), min_idx]

    return {
        "errs": errs,
        "err": min_err,
        "sym": T_sym,
        "T_gt_sym": T_gt_sym,
        "idx": min_idx,
    }


def load_zephyr_hypotheses(ds_name, device="cuda", debug=False, hypotheses_type="all"):
    """Load Zephyr ppf hypotheses (and SIFT)

    Args:
        ds_name: str ['ycbv.bop19', 'lmo.bop19']
        hypotheses_type: ['all', 'ppf', 'sift']

    Returns:
        PandasTensorCollection:
            poses: [N,4,4]
            infos: has columns ['pose_hypothesis_id']

    """

    assert hypotheses_type in ["ppf", "sift", "all"]
    zephyr_dir = LOCAL_DATA_DIR / "external_detections/zephyr"
    if ds_name == "ycbv.bop19":
        fname = zephyr_dir / f"ycbv_test_pose_hypotheses_{hypotheses_type}.pth"
    elif ds_name == "lmo.bop19":
        fname = zephyr_dir / f"lmo_test_pose_hypotheses_{hypotheses_type}.pth"
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    p = torch.load(fname)
    p.infos = p.infos.rename(columns={"object_label": "label"})
    p = p.cuda()
    return p


def load_ppf_hypotheses(ds_name, device="cuda", debug=False):
    """Load Zephyr ppf hypotheses

    The columns of the dataframe are

    ['ObjectId', 'SceneId', 'ImageId', 'Score', 'XTrans', 'YTrans', 'ZTrans',
       'XRot', 'YRot', 'ZRot', 'Type', 'Time']
    """
    zephyr_dir = LOCAL_DATA_DIR / "external_detections/zephyr/ppf_hypos"
    "external_detections/zephyr/ppf_hypos/ycbv_list_bop_test.txt"
    if ds_name == "ycbv.bop19":
        fname = zephyr_dir / "ycbv_list_bop_test.txt"
    elif ds_name == "lmo.bop19":
        fname = zephyr_dir / "lmo_list_bop_test_v1.txt"
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    df = pd.read_csv(fname, delim_whitespace=True)

    if debug:
        df = df[:10]

    df["scene_id"] = df.SceneId
    df["object_id"] = df.ObjectId
    df["view_id"] = df.ImageId
    df["label"] = ""  # empty

    for object_id in df["ObjectId"].unique():
        df.loc[df.ObjectId == object_id, "label"] = f"obj_{object_id:06}"

    # Make a PandasTensorCollection with poses
    translation_scale = 1000.0  # Things are in cm?
    N = len(df)

    poses = torch.zeros([N, 4, 4], device=device)
    poses[:, 0, 3] = torch.tensor(df["XTrans"], device=device) / translation_scale
    poses[:, 1, 3] = torch.tensor(df["YTrans"], device=device) / translation_scale
    poses[:, 2, 3] = torch.tensor(df["ZTrans"], device=device) / translation_scale

    xrot = torch.deg2rad(torch.tensor(df["XRot"], device=device))
    yrot = torch.deg2rad(torch.tensor(df["YRot"], device=device))
    zrot = torch.deg2rad(torch.tensor(df["ZRot"], device=device))
    euler = torch.stack((xrot, yrot, zrot), dim=-1)

    r = scipy.spatial.transform.Rotation.from_euler("XYZ", euler.cpu().numpy())
    poses[:, :3, :3] = torch.from_numpy(r.as_matrix()).to(device)

    columns_to_remove = [
        "SceneId",
        "ObjectId",
        "ImageId",
        "XTrans",
        "YTrans",
        "ZTrans",
        "XRot",
        "YRot",
        "ZRot",
        "Type",
        "Time",
    ]
    for column_name in columns_to_remove:
        df.drop(column_name, axis=1, inplace=True)

    p = PandasTensorCollection(df, poses=poses)
    return p


def load_dtoid_detections(ds_name):
    dtoid_dir = LOCAL_DATA_DIR / "external_detections/dtoid"
    if ds_name == "lmo.bop19":
        fname = dtoid_dir / "lmo_preds.csv"
    elif ds_name == "lm.bop19":
        fname = dtoid_dir / "lm_preds.csv"
    else:
        raise ValueError(f"Unknown dataset {ds_name}")
    df = pd.read_csv(fname)

    def parse_image_fn(image_fn):
        ds, split, scene_id, modality, ext = image_fn.split("/")
        scene_id = int(scene_id)
        view_id = int(ext.split(".")[0])
        return dict(scene_id=scene_id, view_id=view_id)

    x1 = df.loc[:, "x"].values
    y1 = df.loc[:, "y"].values
    x2 = x1 + df.loc[:, "w"].values
    y2 = y1 + df.loc[:, "h"].values

    infos = pd.DataFrame([parse_image_fn(image_fn) for image_fn in df["image_fn"]])
    infos.loc[:, "label"] = [f"obj_{object_id:06d}" for object_id in df["object_id"]]
    infos.loc[:, "score"] = -1
    bboxes = np.concatenate([x1[:, None], y1[:, None], x2[:, None], y2[:, None]], axis=1)
    bboxes = torch.tensor(bboxes).float()
    ids_valids = (bboxes >= 0).all(dim=1).nonzero().flatten().tolist()
    bboxes = bboxes[ids_valids]
    infos = infos.iloc[ids_valids].reset_index(drop=True)

    detections = PandasTensorCollection(infos, bboxes=bboxes)
    return detections


def compute_errors_single_object(TCO_gt, TCO_pred, obj_label, obj_dataset, max_sym_rot_step_deg=1):
    """
    Compute the errors between gt_pose and predicted pose.

    Args:

        TCO_gt: [4,4] The pose you want to compute error relative to
        poses: [B,4,4]
        obj_dataset:

    """

    device = TCO_pred.device
    B = TCO_pred.shape[0]
    obj_info_dict = dict()
    for val in obj_dataset.objects:
        obj_info_dict[val["label"]] = val

    obj_info = obj_info_dict[obj_label]
    assert obj_info["label"] == obj_label
    if obj_info["is_symmetric"]:
        bop_info = obj_info["bop_info"]
        max_sym_rot_step = np.deg2rad(max_sym_rot_step_deg)
        trans_list = get_symmetry_transformations(bop_info, max_sym_disc_step=max_sym_rot_step)
        syms = get_symmetry_transformations_torch(trans_list)
    else:
        syms = torch.eye(4, device=device).unsqueeze(0)

    syms = syms.to(device)

    pts = create_default_object_pts().to(device)

    TCO_gt_expand = TCO_gt.unsqueeze(0).expand([B, -1, -1])
    mssd_out = mssd_torch(TCO_pred, TCO_gt_expand, pts, syms)
    TCO_gt_sym_obj = mssd_out["T_gt_sym"]

    err_out = compute_pose_error(TCO_gt_sym_obj, TCO_pred)
    trans_err = err_out["trans_err"].cpu()
    roterr_deg = err_out["roterr_deg"].cpu()

    return {
        "trans_err": trans_err,
        "roterr_deg": roterr_deg,
    }
