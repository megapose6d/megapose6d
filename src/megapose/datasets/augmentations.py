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
import dataclasses
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Union

# Third Party
import cv2
import numpy as np
import PIL
import torch
from PIL import ImageEnhance, ImageFilter
from torchvision.datasets import ImageFolder

# MegaPose
from megapose.datasets.scene_dataset import Resolution, SceneObservation
from megapose.datasets.utils import make_detections_from_segmentation
from megapose.lib3d.camera_geometry import get_K_crop_resize


class SceneObservationTransform:
    def __call__(self, obs: SceneObservation) -> SceneObservation:
        raise NotImplementedError


class SceneObservationAugmentation(SceneObservationTransform):
    def __init__(
        self,
        transform: Union[SceneObservationTransform, List["SceneObservationAugmentation"]],
        p: float = 1.0,
    ):
        self.p = p
        self.transform = transform

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.rgb is not None
        if random.random() <= self.p:
            if isinstance(self.transform, list):
                for transform_ in self.transform:
                    obs = transform_(obs)
            else:
                obs = self.transform(obs)
        return obs


class PillowRGBTransform(SceneObservationTransform):
    def __init__(self, pillow_fn: PIL.ImageEnhance._Enhance, factor_interval: Tuple[float, float]):
        self.pillow_fn = pillow_fn
        self.factor_interval = factor_interval

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        rgb_pil = PIL.Image.fromarray(obs.rgb)
        rgb_pil = self.pillow_fn(rgb_pil).enhance(factor=random.uniform(*self.factor_interval))
        obs = dataclasses.replace(obs, rgb=np.array(rgb_pil))
        return obs


class PillowSharpness(PillowRGBTransform):
    def __init__(self, factor_interval: Tuple[float, float] = (0.0, 50.0)):
        super().__init__(pillow_fn=ImageEnhance.Sharpness, factor_interval=factor_interval)


class PillowContrast(PillowRGBTransform):
    def __init__(self, factor_interval: Tuple[float, float] = (0.2, 50.0)):
        super().__init__(pillow_fn=ImageEnhance.Contrast, factor_interval=factor_interval)


class PillowBrightness(PillowRGBTransform):
    def __init__(self, factor_interval: Tuple[float, float] = (0.1, 6.0)):
        super().__init__(pillow_fn=ImageEnhance.Brightness, factor_interval=factor_interval)


class PillowColor(PillowRGBTransform):
    def __init__(self, factor_interval: Tuple[float, float] = (0, 20.0)):
        super().__init__(pillow_fn=ImageEnhance.Color, factor_interval=factor_interval)


class PillowBlur(SceneObservationTransform):
    def __init__(self, factor_interval: Tuple[int, int] = (1, 3)):
        self.factor_interval = factor_interval

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.rgb is not None
        rgb_pil = PIL.Image.fromarray(obs.rgb)
        k = random.randint(*self.factor_interval)
        rgb_pil = rgb_pil.filter(ImageFilter.GaussianBlur(k))
        obs = dataclasses.replace(obs, rgb=np.array(rgb_pil))
        return obs


class DepthTransform(SceneObservationTransform):
    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.depth is not None
        depth = self._transform_depth(obs.depth)
        return dataclasses.replace(obs, depth=depth)


class DepthGaussianNoiseTransform(DepthTransform):
    """Adds random Gaussian noise to the depth image."""

    def __init__(self, std_dev: float = 0.02):
        self.std_dev = std_dev

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        noise = np.random.normal(scale=self.std_dev, size=depth.shape)
        depth[depth > 0] += noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        return depth


class DepthCorrelatedGaussianNoiseTransform(DepthTransform):
    """Adds random Gaussian noise to the depth image."""

    def __init__(
        self,
        std_dev: float = 0.01,
        gp_rescale_factor_min: float = 15.0,
        gp_rescale_factor_max: float = 40.0,
    ):
        self.std_dev = std_dev
        self.gp_rescale_factor_min = gp_rescale_factor_min
        self.gp_rescale_factor_max = gp_rescale_factor_max
        self.gp_rescale_factor_bounds = [gp_rescale_factor_min, gp_rescale_factor_max]

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        depth = np.copy(depth)
        rescale_factor = np.random.uniform(
            low=self.gp_rescale_factor_min,
            high=self.gp_rescale_factor_max,
        )

        small_H, small_W = (np.array([H, W]) / rescale_factor).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=self.std_dev, size=(small_H, small_W))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        depth[depth > 0] += additive_noise[depth > 0]
        depth = np.clip(depth, 0, np.finfo(np.float32).max)
        return depth


class DepthMissingTransform(DepthTransform):
    """Randomly drop-out parts of the depth image."""

    def __init__(self, max_missing_fraction: float = 0.2, debug: bool = False):
        self.max_missing_fraction = max_missing_fraction
        self.debug = debug

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        v_idx, u_idx = np.where(depth > 0)
        if not self.debug:
            missing_fraction = np.random.uniform(0, self.max_missing_fraction)
        else:
            missing_fraction = self.max_missing_fraction
        dropout_ids = np.random.choice(
            np.arange(len(u_idx)), int(missing_fraction * len(u_idx)), replace=False
        )
        depth[v_idx[dropout_ids], u_idx[dropout_ids]] = 0
        return depth


class DepthDropoutTransform(DepthTransform):
    """Set the entire depth image to zero."""

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.zeros_like(depth)
        return depth


class DepthEllipseDropoutTransform(DepthTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
    ) -> None:
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    @staticmethod
    def generate_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(noise_params["ellipse_dropout_mean"])

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T  # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(
            nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout
        )
        # Shape: [num_ellipses_to_dropout x 2]
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        y_radii = np.random.gamma(
            noise_params["ellipse_gamma_shape"],
            noise_params["ellipse_gamma_scale"],
            size=num_ellipses_to_dropout,
        )
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        return x_radii, y_radii, angles, dropout_centers

    @staticmethod
    def dropout_random_ellipses(
        depth_img: np.ndarray, noise_params: Dict[str, float]
    ) -> np.ndarray:
        """Randomly drop a few ellipses in the image for robustness.

        Adapted from:
        https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        This is adapted from the DexNet 2.0 code:
        https://github.com/chrisdxie/uois/blob/master/src/data_augmentation.py#L53


        @param depth_img: a [H x W] set of depth z values
        """

        depth_img = depth_img.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img, noise_params=noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            depth_img = cv2.ellipse(
                depth_img,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=0,
                thickness=-1,
            )

        return depth_img

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = self.dropout_random_ellipses(depth, self._noise_params)
        return depth


class DepthEllipseNoiseTransform(DepthTransform):
    def __init__(
        self,
        ellipse_dropout_mean: float = 10.0,
        ellipse_gamma_shape: float = 5.0,
        ellipse_gamma_scale: float = 1.0,
        std_dev: float = 0.01,
    ) -> None:
        self.std_dev = std_dev
        self._noise_params = {
            "ellipse_dropout_mean": ellipse_dropout_mean,
            "ellipse_gamma_scale": ellipse_gamma_scale,
            "ellipse_gamma_shape": ellipse_gamma_shape,
        }

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth_img = depth
        depth_aug = depth_img.copy()

        (
            x_radii,
            y_radii,
            angles,
            dropout_centers,
        ) = DepthEllipseDropoutTransform.generate_random_ellipses(
            depth_img, noise_params=self._noise_params
        )

        num_ellipses_to_dropout = x_radii.shape[0]

        additive_noise = np.random.normal(loc=0.0, scale=self.std_dev, size=x_radii.shape)

        # Dropout ellipses
        noise = np.zeros_like(depth)
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            noise = cv2.ellipse(
                noise,
                tuple(center[::-1]),
                (x_radius, y_radius),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=additive_noise[i],
                thickness=-1,
            )

        depth_aug[depth > 0] += noise[depth > 0]
        depth = depth_aug

        return depth


class DepthBlurTransform(DepthTransform):
    def __init__(self, factor_interval: Tuple[int, int] = (3, 7)):
        self.factor_interval = factor_interval

    def _transform_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.copy(depth)
        k = random.randint(*self.factor_interval)
        depth = cv2.blur(depth, (k, k))
        return depth


class DepthBackgroundDropoutTransform(SceneObservationTransform):
    """Set all background depth values to zero."""

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.depth is not None
        assert obs.segmentation is not None

        # Set background to zero
        depth = np.copy(obs.depth)
        depth[obs.segmentation == 0] = 0
        return dataclasses.replace(obs, depth=depth)


class BackgroundImageDataset:
    def __getitem__(self, idx: int) -> np.ndarray:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ReplaceBackgroundTransform(SceneObservationTransform):
    def __init__(
        self,
        image_dataset: BackgroundImageDataset,
    ):
        self.image_dataset = image_dataset

    def get_bg_image(self, idx: int) -> PIL.Image:
        return self.image_dataset[idx]

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.rgb is not None
        assert obs.segmentation is not None
        rgb = obs.rgb.copy()
        h, w, c = rgb.shape
        rgb_bg_pil = self.get_bg_image(random.randint(0, len(self.image_dataset) - 1))
        rgb_bg = np.asarray(rgb_bg_pil.resize((w, h)))
        mask_bg = obs.segmentation == 0
        rgb[mask_bg] = rgb_bg[mask_bg]
        return dataclasses.replace(obs, rgb=rgb)


class VOCBackgroundAugmentation(ReplaceBackgroundTransform):
    def __init__(self, voc_root: Path):
        image_dataset = ImageFolder(voc_root)
        super().__init__(image_dataset)

    def get_bg_image(self, idx: int) -> np.ndarray:
        return self.image_dataset[idx][0]


class CropResizeToAspectTransform(SceneObservationTransform):
    def __init__(self, resize: Resolution = (480, 640)):
        assert resize[1] >= resize[0]
        self.resize = resize
        self.aspect = max(resize) / min(resize)

    def __call__(self, obs: SceneObservation) -> SceneObservation:
        assert obs.rgb is not None
        assert obs.segmentation is not None
        assert obs.binary_masks is None
        assert obs.camera_data is not None
        assert obs.object_datas is not None
        assert obs.segmentation.dtype == np.uint32

        rgb_pil = PIL.Image.fromarray(obs.rgb)
        w, h = rgb_pil.size

        if (h, w) == self.resize:
            return obs

        segmentation_pil = PIL.Image.fromarray(obs.segmentation)
        assert segmentation_pil.mode == "I"
        depth_pil = None
        if obs.depth is not None:
            assert obs.depth.dtype == np.float32
            depth_pil = PIL.Image.fromarray(obs.depth)
            assert depth_pil.mode == "F"

        # Match the width on input image with an image of target aspect ratio.
        if not np.isclose(w / h, self.aspect):
            r = self.aspect
            crop_h = w * 1 / r
            x0, y0 = w / 2, h / 2
            crop_box_size = (crop_h, w)
            crop_h, crop_w = min(crop_box_size), max(crop_box_size)
            x1, y1, x2, y2 = x0 - crop_w / 2, y0 - crop_h / 2, x0 + crop_w / 2, y0 + crop_h / 2
            box = (x1, y1, x2, y2)
            rgb_pil = rgb_pil.crop(box)
            segmentation_pil = segmentation_pil.crop(box)
            if depth_pil is not None:
                depth_pil = depth_pil.crop(box)
            new_K = get_K_crop_resize(
                torch.tensor(obs.camera_data.K).unsqueeze(0),
                torch.tensor(box).unsqueeze(0),
                orig_size=(h, w),
                crop_resize=(crop_h, crop_w),
            )[0].numpy()
        else:
            new_K = obs.camera_data.K

        # Resize to target size
        w, h = rgb_pil.size
        w_resize, h_resize = max(self.resize), min(self.resize)
        rgb_pil = rgb_pil.resize((w_resize, h_resize), resample=PIL.Image.BILINEAR)
        segmentation_pil = segmentation_pil.resize((w_resize, h_resize), resample=PIL.Image.NEAREST)
        if depth_pil is not None:
            depth_pil = depth_pil.resize((w_resize, h_resize), resample=PIL.Image.NEAREST)
        box = (0, 0, w, h)
        new_K = get_K_crop_resize(
            torch.tensor(new_K).unsqueeze(0),
            torch.tensor(box).unsqueeze(0),
            orig_size=(h, w),
            crop_resize=(h_resize, w_resize),
        )[0].numpy()

        new_obs = deepcopy(obs)
        new_obs.camera_data.K = new_K
        new_obs.camera_data.resolution = (h_resize, w_resize)
        new_obs.rgb = np.array(rgb_pil, dtype=np.uint8)
        new_obs.segmentation = np.array(segmentation_pil, dtype=np.int32)
        if depth_pil is not None:
            new_obs.depth = np.array(depth_pil, dtype=np.float_)

        # Update modal object bounding boxes
        dets_gt = make_detections_from_segmentation(new_obs.segmentation[None])[0]
        new_object_datas = []
        for obj in obs.object_datas:
            if obj.unique_id in dets_gt:
                new_obj = dataclasses.replace(
                    obj, bbox_modal=dets_gt[obj.unique_id], bbox_amodal=None, visib_fract=None
                )
                new_object_datas.append(new_obj)
        new_obs.object_datas = new_object_datas
        return new_obs
