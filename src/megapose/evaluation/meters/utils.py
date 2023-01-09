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
from collections import OrderedDict

# Third Party
import numpy as np
import pandas as pd


def one_to_one_matching(
    pred_infos, gt_infos, keys=("scene_id", "view_id"), allow_pred_missing=False
):
    keys = list(keys)
    pred_infos["pred_id"] = np.arange(len(pred_infos))
    gt_infos["gt_id"] = np.arange(len(gt_infos))
    matches = pred_infos.merge(gt_infos, on=keys)

    matches_gb = matches.groupby(keys).groups
    assert all([len(v) == 1 for v in matches_gb.values()])
    if not allow_pred_missing:
        assert len(matches) == len(gt_infos)
    return matches


def add_inst_num(infos, group_keys=["scene_id", "view_id", "label"], key="pred_inst_num"):

    inst_num = np.empty(len(infos), dtype=int)
    for group_name, group_ids in infos.groupby(group_keys).groups.items():
        inst_num[group_ids.values] = np.arange(len(group_ids))
    infos[key] = inst_num
    return infos


def get_top_n_ids(
    infos, group_keys=("scene_id", "view_id", "label"), top_key="score", n_top=-1, targets=None
):

    infos["id_before_top_n"] = np.arange(len(infos))
    group_keys = list(group_keys)

    if targets is not None:
        targets_inst_count = dict()
        for k, ids in targets.groupby(group_keys).groups.items():
            targets_inst_count[k] = targets.loc[ids[0], "inst_count"]

    def get_top_n(group_k):
        if n_top > 0:
            return n_top
        elif targets is not None:
            return targets_inst_count.get(group_k, 0)
        else:
            return None

    keep_ids = []
    for k, ids in infos.groupby(group_keys).groups.items():
        group = infos.iloc[ids].sort_values(top_key, ascending=False)
        top_n = get_top_n(k)
        if top_n is None:
            top_n = len(group)
        keep_ids.append(group["id_before_top_n"].values[:top_n])
    if len(keep_ids) > 0:
        keep_ids = np.concatenate(keep_ids)
    else:
        keep_ids = []
    del infos["id_before_top_n"]
    return keep_ids


def add_valid_gt(
    gt_infos, group_keys=("scene_id", "view_id", "label"), visib_gt_min=-1, targets=None
):

    if visib_gt_min > 0:
        gt_infos["valid"] = gt_infos["visib_fract"] >= visib_gt_min
        if targets is not None:
            gt_infos["valid"] = np.logical_and(
                gt_infos["valid"], np.isin(gt_infos["label"], targets["label"])
            )
    elif targets is not None:
        valid_ids = get_top_n_ids(
            gt_infos, group_keys=group_keys, top_key="visib_fract", targets=targets
        )
        gt_infos["valid"] = False
        gt_infos.loc[valid_ids, "valid"] = True
    else:
        gt_infos["valid"] = True
    return gt_infos


def get_candidate_matches(
    pred_infos, gt_infos, group_keys=["scene_id", "view_id", "label"], only_valids=True
):
    pred_infos["pred_id"] = np.arange(len(pred_infos))
    gt_infos["gt_id"] = np.arange(len(gt_infos))
    group_keys = list(group_keys)
    cand_infos = pred_infos.merge(gt_infos, on=group_keys)
    if only_valids:
        cand_infos = cand_infos[cand_infos["valid"]].reset_index(drop=True)
    cand_infos["cand_id"] = np.arange(len(cand_infos))
    return cand_infos


def match_poses(cand_infos, group_keys=["scene_id", "view_id", "label"]):
    assert "error" in cand_infos

    matches = []

    def match_label_preds(group):
        gt_ids_matched = set()
        group = group.reset_index(drop=True)
        gb_pred = group.groupby("pred_id", sort=False)
        ids_sorted = gb_pred.first().sort_values("score", ascending=False)
        gb_pred_groups = gb_pred.groups
        for idx, _ in ids_sorted.iterrows():
            pred_group = group.iloc[gb_pred_groups[idx]]
            best_error = np.inf
            best_match = None
            for _, tentative_match in pred_group.iterrows():
                if (
                    tentative_match["error"] < best_error
                    and tentative_match["gt_id"] not in gt_ids_matched
                ):
                    best_match = tentative_match
                    best_error = tentative_match["error"]

            if best_match is not None:
                gt_ids_matched.add(best_match["gt_id"])
                matches.append(best_match)

    if len(cand_infos) > 0:
        cand_infos.groupby(group_keys).apply(match_label_preds)
        matches = pd.DataFrame(matches).reset_index(drop=True)
    else:
        matches = cand_infos
    return matches


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids - 1]) * mpre[ids]).sum() * 10
    return ap
