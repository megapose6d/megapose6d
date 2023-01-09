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
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import numpy as np
import numpy.typing as npt

# Local Folder
from .rotations import euler2quat
from .transform import Transform


@dataclass
class ContinuousSymmetry:
    """A representation of a continuous object symmetry.

    See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
    """

    offset: npt.NDArray[np.float_]
    axis: npt.NDArray[np.float_]


@dataclass
class DiscreteSymmetry:
    """
    pose: (4, 4) homogeneous matrix
    """

    pose: npt.NDArray[np.float_]


def make_symmetries_poses(
    symmetries_discrete: List[DiscreteSymmetry] = [],
    symmetries_continuous: List[ContinuousSymmetry] = [],
    n_symmetries_continuous: int = 8,
    units: str = "mm",
    scale: Optional[float] = None,
) -> np.ndarray:
    """Generates the set of object symmetries.

    Returns:
        (num_symmetries, 4, 4) array
    """
    # Note: See https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/misc.py
    if scale is None:
        scale = {"m": 1, "mm": 0.001}[units]
    all_M_discrete = [Transform((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0))]
    all_M_continuous = []
    all_M = []
    for sym_d_n in symmetries_discrete:
        M = sym_d_n.pose
        M[:3, -1] *= scale
        all_M_discrete.append(Transform(M))
    for sym_c_n in symmetries_continuous:
        assert np.allclose(sym_c_n.offset, 0)
        axis = np.array(sym_c_n.axis)  # convert to np.array from list
        assert axis.sum() == 1
        for n in range(n_symmetries_continuous):
            euler = axis * 2 * np.pi * n / n_symmetries_continuous
            q = euler2quat(euler)
            all_M_continuous.append(Transform(q, (0.0, 0.0, 0.0)))
    for sym_d in all_M_discrete:
        if len(all_M_continuous) > 0:
            for sym_c in all_M_continuous:
                all_M.append((sym_c * sym_d).toHomogeneousMatrix())
        else:
            all_M.append(sym_d.toHomogeneousMatrix())
    return np.array(all_M)
