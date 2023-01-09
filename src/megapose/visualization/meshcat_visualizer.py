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
import io
from pathlib import Path

# Third Party
import meshcat
import numpy as np
import trimesh
from meshcat.geometry import (
    ImageTexture,
    MeshLambertMaterial,
    ObjMeshGeometry,
    PngImage,
)

# MegaPose
from megapose.datasets.datasets_cfg import make_object_dataset

# Local Folder
from .meshcat_utils import create_visualizer, trimesh_to_meshcat_geometry


class MeshcatSceneViewer:
    def __init__(self, obj_ds_name, use_textures=True, zmq_url="tcp://127.0.0.1:6000", clear=True):

        self.obj_ds = make_object_dataset(obj_ds_name)
        self.label_to_object = {}
        self.visualizer = create_visualizer(zmq_url=zmq_url, clear=clear)
        self.use_textures = use_textures
        self.clear = clear

    def get_meshcat_object(self, label):
        if label in self.label_to_object:
            pass
        else:
            objects = [obj for obj in self.obj_ds.objects if obj.label == label]
            obj = objects[0]
            mesh_path = Path(obj.mesh_path)
            mesh = trimesh.load(mesh_path)
            scale = obj.scale
            mesh.apply_scale(scale)

            obj_path = "/dev/shm/mesh.obj"
            trimesh.exchange.export.export_mesh(mesh, obj_path)
            geometry = ObjMeshGeometry.from_file(obj_path)

            material = None

            # Needed to deal with the fact that some objects might
            # be saved as trimesh.Scene instead of trimesh.Trimesh
            if hasattr(mesh, "visual"):
                if isinstance(mesh.visual, trimesh.visual.TextureVisuals) and self.use_textures:
                    texture_path = f"/dev/shm/{label}_texture.png"
                    mesh.visual.material.image.save(texture_path)
                    material = MeshLambertMaterial(
                        map=ImageTexture(image=PngImage.from_file(texture_path))
                    )
            self.label_to_object[label] = (geometry, material)
        return self.label_to_object[label]

    def visualize_scene(self, obj_infos):
        if self.clear:
            self.visualizer.delete()
        self.visualizer.open()

        for n, obj_info in enumerate(obj_infos):
            label = str(obj_info["name"])
            geometry, material = self.get_meshcat_object(label)
            node_name = obj_info.get("node_name", f"label={label}-object={n}")

            TWO = np.array(obj_info["TWO"].tolist())
            self.visualizer[node_name].set_object(geometry, material)
            self.visualizer[node_name].set_transform(TWO)
        return
