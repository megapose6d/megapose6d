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
import builtins
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Set
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Set

# Third Party
import numpy as np
import panda3d as p3d
from direct.showbase.ShowBase import ShowBase
from tqdm import tqdm

# MegaPose
from megapose.datasets.object_dataset import RigidObjectDataset

# Local Folder
from .types import (
    CameraRenderingData,
    Panda3dCamera,
    Panda3dCameraData,
    Panda3dLightData,
    Panda3dObjectData,
    Resolution,
    RgbaColor,
)
from .utils import make_rgb_texture_normal_map, np_to_lmatrix4


@dataclass
class Panda3dDebugData:
    timings: Dict[str, float]


class App(ShowBase):
    """Panda3d App."""

    def __init__(self) -> None:
        p3d.core.load_prc_file_data(
            __file__,
            "load-display pandagl\n"
            "notify-level-assimp fatal\n"
            "notify-level-egldisplay fatal\n"
            "notify-level-glgsg fatal\n"
            "notify-level-glxdisplay fatal\n"
            "notify-level-x11display fatal\n"
            "notify-level-device fatal\n"
            "texture-minfilter mipmap\n"
            "texture-anisotropic-degree 16\n"
            "framebuffer-multisample 1\n"
            "multisamples 4\n"
            "background-color 0.0 0.0 0.0 0.0\n"
            "load-file-type p3assimp\n"
            "track-memory-usage 1\n"
            "transform-cache 0\n"
            "state-cache 0\n"
            "audio-library-name null\n"
            "model-cache-dir\n",
        )
        assert "CUDA_VISIBLE_DEVICES" in os.environ
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        assert len(devices) == 1
        if "EGL_VISIBLE_DEVICES" not in os.environ:
            out = subprocess.check_output(
                ["nvidia-smi", "--id=" + str(devices[0]), "-q", "--xml-format"]
            )
            tree = ET.fromstring(out)
            gpu = tree.findall("gpu")[0]
            assert gpu is not None
            minor_number_el = gpu.find("minor_number")
            assert minor_number_el is not None
            dev_id = minor_number_el.text
            os.environ["EGL_VISIBLE_DEVICES"] = str(dev_id)

        super().__init__(windowType="offscreen")
        self.render.set_shader_auto()
        self.render.set_antialias(p3d.core.AntialiasAttrib.MAuto)
        self.render.set_two_sided(True)


def make_scene_lights(
    ambient_light_color: RgbaColor = (0.1, 0.1, 0.1, 1.0),
    point_lights_color: RgbaColor = (0.4, 0.4, 0.4, 1.0),
) -> List[Panda3dLightData]:
    """Creates 1 ambient light + 6 point lights to illuminate a panda3d scene."""
    pos = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ]
    )

    def pos_fn(
        root_node: p3d.core.NodePath, light_node: p3d.core.NodePath, pos: np.ndarray
    ) -> None:
        radius = root_node.getBounds().radius
        xyz_ = pos * radius * 10
        light_node.setPos(tuple(xyz_.tolist()))
        return

    light_datas = [Panda3dLightData(light_type="ambient", color=ambient_light_color)]
    for pos_n in pos:
        pos_fn_ = partial(pos_fn, pos=pos_n)
        light_datas.append(
            Panda3dLightData(
                light_type="point", color=point_lights_color, positioning_function=pos_fn_
            )
        )
    return light_datas


class Panda3dSceneRenderer:
    """A class used to render scenes including objects, cameras, lights.

    Rendering is done using panda3d.
    """

    def __init__(
        self,
        asset_dataset: RigidObjectDataset,
        preload_labels: Set[str] = set(),
        debug: bool = False,
        verbose: bool = False,
    ):

        self._asset_dataset = asset_dataset
        self._label_to_node: Dict[str, p3d.core.NodePath] = dict()
        self.verbose = verbose
        self.debug = debug
        self.debug_data = Panda3dDebugData(timings=dict())

        self._cameras_pool: Dict[Resolution, List[Panda3dCamera]] = defaultdict(list)
        if hasattr(builtins, "base"):
            self._app = builtins.base  # type: ignore
        else:
            self._app = App()
        self._app.cam.node().setActive(0)
        self._app.render.clear_light()
        self._rgb_texture = make_rgb_texture_normal_map(size=32)

        assert isinstance(preload_labels, set)
        for label in tqdm(preload_labels, disable=not verbose):
            self.get_object_node(label)

    def create_new_camera(self, resolution: Resolution) -> Panda3dCamera:
        idx = sum([len(x) for x in self._cameras_pool.values()])
        cam = Panda3dCamera.create(f"camera={idx}", resolution=resolution, app=self._app)
        self._cameras_pool[resolution].append(cam)
        return cam

    def get_cameras(self, data_cameras: List[Panda3dCameraData]) -> List[Panda3dCamera]:
        resolution_to_data_cameras: Dict[Resolution, List[Panda3dCameraData]] = defaultdict(list)
        for data_camera in data_cameras:
            resolution_to_data_cameras[data_camera.resolution].append(data_camera)

        for resolution_, data_cameras_ in resolution_to_data_cameras.items():
            for idx in range(len(data_cameras_)):
                if idx >= len(self._cameras_pool[resolution_]):
                    self.create_new_camera(resolution_)

        cameras = []
        available_cameras = {k: v.copy() for k, v in self._cameras_pool.items()}
        for data_camera in data_cameras:
            camera = available_cameras[data_camera.resolution].pop()
            cameras.append(camera)
        return cameras

    def get_object_node(self, label: str) -> p3d.core.NodePath:
        if label in self._label_to_node:
            return self._label_to_node[label]

        asset = self._asset_dataset.get_object_by_label(label)
        scale = asset.scaling_factor_mesh_units_to_meters * asset.scaling_factor
        y, p, r = asset.ypr_offset_deg

        node = self._app.loader.load_model(str(asset.mesh_path), noCache=True)
        node.setScale(scale)
        node.setPos(0, 0, 0)
        node.setHpr(y, p, r)
        self._label_to_node[label] = node
        return node

    def use_normals_texture(self, obj_node: p3d.core.NodePath) -> p3d.core.NodePath:
        obj_node.setMaterialOff(1)
        obj_node.set_color(p3d.core.Vec4((1.0, 1.0, 1.0, 1.0)))
        obj_node.setTextureOff(1)
        obj_node.setTexGen(p3d.core.TextureStage.getDefault(), p3d.core.TexGenAttrib.MEyeNormal)
        obj_node.setTexture(self._rgb_texture)
        return obj_node

    def setup_scene(
        self, root_node: p3d.core.NodePath, data_objects: List[Panda3dObjectData]
    ) -> List[p3d.core.NodePath]:
        obj_nodes = []
        for n, data_obj in enumerate(data_objects):
            label = data_obj.label
            obj_node = root_node.attach_new_node(f"label={label}-object={n}")
            self.get_object_node(label).instanceTo(obj_node)
            if data_obj.remove_mesh_material:
                obj_node.setMaterialOff(1)
            TWO = np_to_lmatrix4(data_obj.TWO.toHomogeneousMatrix())
            obj_node.setMat(TWO)
            if data_obj.positioning_function is not None:
                data_obj.positioning_function(root_node, obj_node)
            obj_node.setScale(data_obj.scale)
            if data_obj.color is not None:
                data_obj.set_node_material_and_transparency(obj_node)
            obj_nodes.append(obj_node)
        return obj_nodes

    def setup_cameras(
        self, root_node: p3d.core.NodePath, data_cameras: List[Panda3dCameraData]
    ) -> List[Panda3dCamera]:
        cameras = self.get_cameras(data_cameras)

        for data_camera, camera in zip(data_cameras, cameras):
            camera_node_path = camera.node_path
            camera_node_path.node().setActive(1)
            camera_node_path.reparentTo(root_node)

            data_camera.set_lens_parameters(camera_node_path.node().getLens())
            view_mat = data_camera.compute_view_mat()
            camera_node_path.setMat(view_mat)
            if data_camera.positioning_function is not None:
                data_camera.positioning_function(root_node, camera_node_path)
        return cameras

    def render_images(
        self, cameras: List[Panda3dCamera], copy_arrays: bool = True, render_depth: bool = False
    ) -> List[CameraRenderingData]:

        self._app.graphicsEngine.renderFrame()
        self._app.graphicsEngine.syncFrame()

        renderings = []
        for camera in cameras:
            rgb = camera.get_rgb_image()
            if copy_arrays:
                rgb = rgb.copy()
            rendering = CameraRenderingData(rgb)

            if render_depth:
                rendering.depth = camera.get_depth_image()
            renderings.append(rendering)
        return renderings

    def setup_lights(
        self, root_node: p3d.core, light_datas: List[Panda3dLightData]
    ) -> List[p3d.core.NodePath]:
        light_node_paths = []
        for n, light_data in enumerate(light_datas):
            if light_data.light_type == "point":
                light_node = p3d.core.PointLight(f"{n}_point")
                assert light_data.positioning_function is not None
            elif light_data.light_type == "ambient":
                light_node = p3d.core.AmbientLight(f"{n}_ambient")
            elif light_data.light_type == "directional":
                light_node = p3d.core.DirectionalLight("{n}_directional")
                assert light_data.positioning_function is not None
            else:
                raise NotImplementedError(light_data.light_type)

            light_node.set_color(light_data.color)
            light_node_path = root_node.attach_new_node(light_node)
            root_node.set_light(light_node_path)
            if light_data.positioning_function is not None:
                light_data.positioning_function(root_node, light_node_path)
            light_node_paths.append(light_node_path)
        return light_node_paths

    def render_scene(
        self,
        object_datas: List[Panda3dObjectData],
        camera_datas: List[Panda3dCameraData],
        light_datas: List[Panda3dLightData],
        render_depth: bool = False,
        copy_arrays: bool = True,
        render_binary_mask: bool = False,
        render_normals: bool = False,
        clear: bool = True,
    ) -> List[CameraRenderingData]:

        start = time.time()
        root_node = self._app.render.attachNewNode("world")
        object_nodes = self.setup_scene(root_node, object_datas)
        cameras = self.setup_cameras(root_node, camera_datas)
        light_nodes = self.setup_lights(root_node, light_datas)
        setup_time = time.time() - start

        start = time.time()
        renderings = self.render_images(cameras, copy_arrays=copy_arrays, render_depth=render_depth)
        if render_normals:
            for object_node in object_nodes:
                self.use_normals_texture(object_node)
                root_node.clear_light()
                light_data = Panda3dLightData(light_type="ambient", color=(1.0, 1.0, 1.0, 1.0))
                light_nodes += self.setup_lights(root_node, [light_data])
            normals_renderings = self.render_images(cameras, copy_arrays=copy_arrays)
            for n, rendering in enumerate(renderings):
                rendering.normals = normals_renderings[n].rgb

        if render_binary_mask:
            for rendering_n in renderings:
                assert rendering_n.depth is not None
                h, w = rendering_n.depth.shape[:2]
                binary_mask = np.zeros((h, w), dtype=np.bool_)
                binary_mask[rendering_n.depth[..., 0] > 0] = 1
                rendering.binary_mask = binary_mask

        render_time = time.time() - start

        if clear:
            for camera in cameras:
                camera.node_path.node().setActive(0)
            for object_node in object_nodes:
                object_node.clear_texture()  # TODO: Is this necessary ?
                object_node.clear_light()  # TODO: Is this necessary ?
                object_node.detach_node()
            for light_node in light_nodes:
                light_node.detach_node()
            root_node.clear_light()
            root_node.detach_node()

            for _ in range(3):
                # TODO: Is this necessary ?
                p3d.core.RenderState.garbageCollect()
                p3d.core.TransformState.garbageCollect()

        self.debug_data.timings["setup_time"] = setup_time
        self.debug_data.timings["render_time"] = render_time
        return renderings
