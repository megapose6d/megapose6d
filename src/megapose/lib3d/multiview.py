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
import numpy as np
import torch
import transforms3d
from panda3d.core import NodePath, RenderState, TransformState

# MegaPose
from megapose.lib3d.transform import Transform
from megapose.lib3d.transform_ops import invert_transform_matrices
from megapose.panda3d_renderer.utils import compute_view_mat


def _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0):
    TCO = TCO.copy()
    tCR = tCR.copy()
    root = NodePath("root")

    obj = NodePath("object")
    obj.reparentTo(root)
    obj.setPos(0, 0, 0)

    TCCGL = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)

    tCR = np.array(tCR.tolist())
    TOC = Transform(np.array(TCO.tolist())).inverse().toHomogeneousMatrix()
    if not np.isfinite(TOC).all():
        TOC = np.eye(4)
        tCR = np.zeros(3)

    view_mat_C0 = compute_view_mat(TOC)
    cam = NodePath("camera")
    cam.reparentTo(root)
    cam.setMat(view_mat_C0)

    ref = NodePath("reference_point")
    ref.reparentTo(root)
    tWR = TOC[:3, :3] @ tCR.reshape((3, 1)) + TOC[:3, [-1]]
    ref.setPos(*tWR[:3])

    radius = np.linalg.norm(np.array(tCR)[:3])
    cam_positions_wrt_cam0 = cam_positions_wrt_cam0 * radius

    ref_pos = ref.getPos()
    ref_pos = (ref_pos[0], ref_pos[1], ref_pos[2])

    up = np.array(cam.getMat()).transpose()[:3, 2]
    up_vector = (up[0], up[1], up[2])

    TC0_CV = []
    cam_pointing_to_ref = NodePath("camera-pointing-to-ref")
    cam_pointing_to_ref.reparentTo(root)
    cam_pointing_to_ref.setPos(cam, (0, 0, 0))
    cam_pointing_to_ref.lookAt(ref_pos, up_vector)

    for n, cam_pos in enumerate(cam_positions_wrt_cam0):
        node = NodePath(f"camera-pointing-to-ref-{n}")
        node.reparentTo(root)
        node.setPos(cam_pointing_to_ref, *cam_pos)
        node.lookAt(ref_pos, up_vector)
        view_mat = np.array(node.getMat(cam)).transpose()
        view_mat = TCCGL @ view_mat @ np.linalg.inv(TCCGL)
        TC0_CV.append(view_mat)

    for p in root.getChildren():
        p.clear_texture()
        p.clear_light()
        p.remove_node()
    root.clear_texture()
    root.clear_light()
    root.remove_node()
    for _ in range(3):
        RenderState.garbageCollect()
        TransformState.garbageCollect()
    return TC0_CV


def get_1_view_TCO_pos_front(TCO, tCR):
    cam_positions_wrt_cam0 = np.array(
        [
            [0, 0, 0],
        ]
    )
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)


def get_3_views_TCO_pos_front(TCO, tCR):
    cam_positions_wrt_cam0 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ]
    )
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)


def get_5_views_TCO_pos_front(TCO, tCR):
    cam_positions_wrt_cam0 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)


def get_3_views_TCO_pos_sphere(TCO, tCR):
    cam_positions_wrt_cam0 = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [-1, 0, 0],
        ]
    )
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)


def get_6_views_TCO_pos_sphere(TCO, tCR):
    cam_positions_wrt_cam0 = np.array(
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 2, 0],
            [0, 1, 1],
            [-1, 1, 0],
            [0, 1, -1],
        ]
    )
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)


def get_26_views_TCO_pos_sphere(TCO, tCR):
    cam_positions_wrt_cam0 = []
    for y in [0, 1, 2]:
        for x in [0, -1, 1]:
            for z in [0, 1, -1]:
                if x == 0 and y == 1 and z == 0:
                    pass
                else:
                    cam_positions_wrt_cam0.append([x, y, z])
    cam_positions_wrt_cam0 = np.array(cam_positions_wrt_cam0, dtype=float)
    return _get_views_TCO_pos_sphere(TCO, tCR, cam_positions_wrt_cam0)

def make_TCO_multiview(
    TCO: torch.Tensor,
    tCR: torch.Tensor,
    multiview_type: str = "front_3views",
    n_views: int = 4,
    remove_TCO_rendering: bool = False,
    views_inplane_rotations: bool = False,
):
    """_summary_

    Args:
        TCO (torch.Tensor): (bsz, 4, 4)
        tCR (torch.Tensor): (bsz, 3)


    Returns:
        _type_: _description_
    """
    bsz = TCO.shape[0]
    device, dtype = TCO.device, TCO.dtype

    TCO_np = TCO.cpu().numpy()
    tCR_np = tCR.cpu().numpy()

    if n_views == 1:
        TC0_CV = []
        for b in range(bsz):
            TC0_CV_ = [np.eye(4)]
            TC0_CV.append(TC0_CV_)
        TC0_CV = torch.as_tensor(np.stack(TC0_CV), device=device, dtype=dtype)
        TCV_O = invert_transform_matrices(TC0_CV) @ TCO.unsqueeze(1)

    elif multiview_type == "TCO+front_1view":
        TC0_CV = []
        for b in range(bsz):
            if remove_TCO_rendering:
                TC0_CV_ = []
            else:
                TC0_CV_ = [np.eye(4)]
            TC0_CV_ += get_1_view_TCO_pos_front(TCO_np[b], tCR_np[b])
            TC0_CV.append(TC0_CV_)
        TC0_CV = torch.as_tensor(np.stack(TC0_CV), device=device, dtype=dtype)
        TCV_O = invert_transform_matrices(TC0_CV) @ TCO.unsqueeze(1)

    elif multiview_type == "TCO+front_3views":
        TC0_CV = []
        for b in range(bsz):
            if remove_TCO_rendering:
                TC0_CV_ = []
            else:
                TC0_CV_ = [np.eye(4)]
            TC0_CV_ += get_3_views_TCO_pos_front(TCO_np[b], tCR_np[b])
            TC0_CV.append(TC0_CV_)
        TC0_CV = torch.as_tensor(np.stack(TC0_CV), device=device, dtype=dtype)
        TCV_O = invert_transform_matrices(TC0_CV) @ TCO.unsqueeze(1)

    elif multiview_type == "sphere_26views":
        TC0_CV = []
        for b in range(bsz):
            if remove_TCO_rendering:
                TC0_CV_ = []
            else:
                TC0_CV_ = [np.eye(4)]
            TC0_CV_ += get_26_views_TCO_pos_sphere(TCO_np[b], tCR_np[b])
            TC0_CV.append(TC0_CV_)
        TC0_CV = torch.as_tensor(np.stack(TC0_CV), device=device, dtype=dtype)
        TCV_O = invert_transform_matrices(TC0_CV) @ TCO.unsqueeze(1)

    else:
        raise ValueError(multiview_type)

    if views_inplane_rotations:
        assert remove_TCO_rendering
        TCV_O = TCV_O.unsqueeze(2).repeat(1, 1, 4, 1, 1)
        for idx, angle in enumerate([np.pi / 2, np.pi, 3 * np.pi / 2]):
            idx = idx + 1
            dR = torch.as_tensor(
                transforms3d.euler.euler2mat(0, 0, angle), device=device, dtype=dtype
            )
            TCV_O[:, :, idx, :3, :3] = dR @ TCV_O[:, :, idx, :3, :3]
        TCV_O = TCV_O.flatten(1, 2)
    return TCV_O
