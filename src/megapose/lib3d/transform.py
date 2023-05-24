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
from typing import Tuple, Union

# Third Party
import numpy as np
import pinocchio as pin
import torch


class Transform:
    """A representation of a SE(3) object based on pinocchio's pin.SE3."""

    def __init__(
        self,
        *args: Union[
            Union[pin.SE3, np.ndarray, torch.Tensor],  # T
            Union[
                pin.Quaternion, np.ndarray, torch.Tensor, Tuple[float, float, float, float]
            ],  # rotation
            Union[np.ndarray, torch.Tensor, Tuple[float, float, float]],  # translation
        ]
    ):
        """
        - Transform(T): SE3 or (4, 4) array
        - Transform(quaternion, translation), where
            quaternion: pin.Quaternion, 4-array representing a xyzw quaternion,
                or a 3x3 rotation matrix
            translation: 3-array
        """
        if len(args) == 1:
            arg_T = args[0]
            if isinstance(arg_T, pin.SE3):
                self._T = arg_T
            elif isinstance(arg_T, np.ndarray):
                assert arg_T.shape == (4, 4)
                R = arg_T[:3, :3].copy()
                t = arg_T[:3, -1].copy()
                self._T = pin.SE3(R, t.reshape(3, 1))
            elif isinstance(arg_T, torch.Tensor):
                T = arg_T.detach().cpu().numpy().copy()
                R = T[:3, :3]
                t = T[:3, -1]
                self._T = pin.SE3(R, t.reshape(3, 1))
            else:
                raise ValueError

        elif len(args) == 2:
            rotation, translation = args
            if isinstance(rotation, pin.Quaternion):
                R = rotation.matrix()
            elif isinstance(rotation, tuple):
                rotation_np = np.array(rotation)
            elif isinstance(rotation, (np.ndarray, torch.Tensor)):
                if isinstance(rotation, torch.Tensor):
                    rotation_np = rotation.detach().cpu().numpy().copy()
                else:
                    rotation_np = rotation
            else:
                raise ValueError

            if rotation_np.size == 4:
                quaternion_xyzw = rotation_np.flatten().tolist()
                quaternion_wxyz = [quaternion_xyzw[-1], *quaternion_xyzw[:-1]]
                q = pin.Quaternion(*quaternion_wxyz)
                q.normalize()
                R = q.matrix()
            elif rotation_np.size == 9:
                assert rotation_np.shape == (3, 3)
                R = rotation_np
            else:
                raise ValueError
            t = np.asarray(translation)
            self._T = pin.SE3(R, t.reshape(3, 1))

        else:
            raise ValueError

    def __mul__(self, other: "Transform") -> "Transform":
        T = self._T * other._T
        return Transform(T)

    def inverse(self) -> "Transform":
        return Transform(self._T.inverse())

    def __str__(self) -> str:
        return str(self._T)

    def toHomogeneousMatrix(self) -> np.ndarray:
        return self._T.homogeneous

    @property
    def translation(self) -> np.ndarray:
        return self._T.translation.reshape(3)

    @property
    def quaternion(self) -> pin.Quaternion:
        return pin.Quaternion(self._T.rotation)

    @property
    def matrix(self) -> np.ndarray:
        """Returns 4x4 homogeneous matrix representations"""
        return self._T.homogeneous
