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
from typing import List

# Third Party
import numpy as np
import numpy.typing as npt
import panda3d as p3d

# MegaPose
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer.geometry import make_axes, make_box, make_sphere

def compute_view_mat(TWC):
    TCCGL = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)
    TCCGL = Transform(TCCGL)
    TWC = Transform(TWC)
    TWCGL = TWC * TCCGL
    view_mat = TWCGL.toHomogeneousMatrix()
    view_mat = p3d.core.LMatrix4f(*view_mat.transpose().flatten().tolist())
    return view_mat

def np_to_lmatrix4(np_array: npt.NDArray) -> p3d.core.LMatrix4f:
    return p3d.core.LMatrix4f(*np_array.transpose().flatten().tolist())


def depth_image_from_depth_buffer(
    depth_buffer: npt.NDArray[np.float32], z_near: float, z_far: float, eps: float = 0.001
) -> npt.NDArray[np.float32]:
    """Convert depth image to depth buffer.

    See https://developer.nvidia.com/content/depth-precision-visualized#:~:text=GPU%20hardware%20depth%20buffers%20don,reciprocal%20of%20world%2Dspace%20depth.
    """
    a = 1.0 / (1 / z_far - 1 / z_near)
    b = -a / z_near
    depth_image = a / (depth_buffer - b)
    depth_image[depth_buffer > (1 - eps)] = 0  # missing/infinite depth
    return depth_image


def make_rgb_texture_normal_map(size: int = 32) -> p3d.core.Texture:
    tex = p3d.core.Texture()
    tex.setup3dTexture(size, size, size, p3d.core.Texture.T_unsigned_byte, p3d.core.Texture.F_rgb8)
    im = np.ones((size, size, size, 3), dtype=np.uint8) * 255
    for x in range(size):
        for y in range(size):
            for z in range(size):
                color = (np.array([x, y, z]) * 255 / size).astype(np.uint8)
                im[x, y, z] = color.astype(np.uint8)
    tex.setRamImage(im.tostring())
    return tex


def make_cube_node(scale, color=(1, 0, 0, 1)):
    cube = GeomNode("cube")
    cube.add_geom(make_box())
    node = NodePath(cube)
    node.setScale(*scale)
    node.setPos(0, 0, 0)
    node.set_light_off()
    node.set_render_mode_wireframe()
    node.set_color(color)
    node.set_render_mode_thickness(4)
    node.set_antialias(AntialiasAttrib.MLine)
    node.set_material(Material(), 1)
    return node


def show_node_axes(node, radius=None):
    n_rand = f"{np.random.randint(1000)}"
    axes = GeomNode(f"axes-{n_rand}")
    axes.add_geom(make_axes())

    axes_node = NodePath(axes)
    bounds = node.getBounds()
    if bounds.is_empty():
        if radius is None:
            radius = 1
    else:
        radius = bounds.get_radius()
    axes_node.set_scale(radius * 0.1)
    axes_node.reparentTo(node)
    axes_node.setPos(0, 0, 0)
    axes_node.set_light_off()
    axes_node.set_render_mode_wireframe()
    axes_node.set_render_mode_thickness(4)
    axes_node.set_antialias(AntialiasAttrib.MLine)
    axes_node.set_material(Material(), 1)
    return axes_node


def show_node_center(node, radius=None):
    sphere = GeomNode("sphere")
    sphere.add_geom(make_sphere())

    sphere_node = NodePath(sphere)
    bounds = node.getBounds()
    if bounds.is_empty():
        if radius is None:
            radius = 1
    else:
        radius = bounds.get_radius()
    sphere_node.set_scale(radius * 0.1)
    set_material(sphere_node, (1, 1, 1, 1))

    sphere_node.reparentTo(node)
    sphere_node.setPos(0, 0, 0)
    sphere_node.set_material(Material(), 1)
    return sphere_node
