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


"""
https://github.com/ikalevatykh/panda3d_viewer/blob/master/panda3d_viewer/geometry.py
"""

# pylint: disable=invalid-name, too-many-locals

# Standard Library
import itertools

# Third Party
import numpy as np
from panda3d.core import (
    Geom,
    GeomLines,
    GeomPoints,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
)

__all__ = (
    "make_axes",
    "make_grid",
    "make_cylinder",
    "make_box",
    "make_plane",
    "make_sphere",
    "make_points",
)
__all__ = ("ViewerError", "ViewerClosedError")


class ViewerError(Exception):
    """Base class for all viewer exceptions."""


class ViewerClosedError(ViewerError):
    """Raised when a method is called in a closed viewer."""


def make_axes():
    """Make an axes geometry.

    Returns:
        Geom -- p3d geometry
    """
    vformat = GeomVertexFormat.get_v3c4()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(6)

    vertex = GeomVertexWriter(vdata, "vertex")
    color = GeomVertexWriter(vdata, "color")

    for x, y, z in np.eye(3):
        vertex.addData3(0, 0, 0)
        color.addData4(x, y, z, 1)
        vertex.addData3(x, y, z)
        color.addData4(x, y, z, 1)

    prim = GeomLines(Geom.UHStatic)
    prim.addNextVertices(6)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_grid(num_ticks=10, step=1.0):
    """Make a grid geometry.

    Keyword Arguments:
        step {float} -- step in meters (default: {1.0})
        num_ticks {int} -- ticks number per axis (default: {5})

    Returns:
        Geom -- p3d geometry
    """
    ticks = np.arange(-num_ticks // 2, num_ticks // 2 + 1) * step

    vformat = GeomVertexFormat.get_v3()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(len(ticks) * 4)

    vertex = GeomVertexWriter(vdata, "vertex")

    for t in ticks:
        vertex.addData3(t, ticks[0], 0)
        vertex.addData3(t, ticks[-1], 0)
        vertex.addData3(ticks[0], t, 0)
        vertex.addData3(ticks[-1], t, 0)

    prim = GeomLines(Geom.UHStatic)
    prim.addNextVertices(len(ticks) * 4)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_capsule(radius, length, num_segments=16, num_rings=16):
    """Make capsule geometry.

    Arguments:
        radius {float} -- capsule radius
        length {float} -- capsule length

    Keyword Arguments:
        num_segments {int} -- segments number (default: {16})
        num_rings {int} -- rings number (default: {16})

    Returns:
        Geom -- p3d geometry
    """
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(num_segments * num_rings)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tcoord = GeomVertexWriter(vdata, "texcoord")

    for u in np.linspace(0, np.pi, num_rings):
        for v in np.linspace(0, 2 * np.pi, num_segments):
            x, y, z = np.cos(v) * np.sin(u), np.sin(v) * np.sin(u), np.cos(u)
            offset = np.sign(z) * 0.5 * length
            vertex.addData3(x * radius, y * radius, z * radius + offset)
            normal.addData3(x, y, z)
            tcoord.addData2(u / np.pi, v / (2 * np.pi))

    prim = GeomTriangles(Geom.UHStatic)
    for i in range(num_rings - 1):
        for j in range(num_segments - 1):
            r0 = i * num_segments + j
            r1 = r0 + num_segments
            if i < num_rings - 2:
                prim.addVertices(r0, r1, r1 + 1)
            if i > 0:
                prim.addVertices(r0, r1 + 1, r0 + 1)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_cylinder(num_segments=16, closed=True):
    """Make a uniform cylinder geometry.

    Keyword Arguments:
        num_segments {int} -- segments number (default: {16})
        closed {bool} -- add caps (default: {True})

    Returns:
        Geom -- p3d geometry
    """
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tcoord = GeomVertexWriter(vdata, "texcoord")

    cyl_rows = num_segments * 2
    cap_rows = num_segments + 1
    if closed:
        vdata.uncleanSetNumRows(cyl_rows + 2 * cap_rows)
    else:
        vdata.uncleanSetNumRows(cyl_rows)

    for phi in np.linspace(0, 2 * np.pi, num_segments):
        x, y = np.cos(phi), np.sin(phi)
        for z in (-1, 1):
            vertex.addData3(x, y, z * 0.5)
            normal.addData3(x, y, 0)
            tcoord.addData2(phi / (2 * np.pi), (z + 1) / 2)

    prim = GeomTriangles(Geom.UHStatic)
    for i in range(num_segments - 1):
        prim.addVertices(i * 2, i * 2 + 3, i * 2 + 1)
        prim.addVertices(i * 2, i * 2 + 2, i * 2 + 3)

    if closed:
        for z in (-1, 1):
            vertex.addData3(0, 0, z * 0.5)
            normal.addData3(0, 0, z)
            tcoord.addData2(0, 0)

            for phi in np.linspace(0, 2 * np.pi, num_segments):
                x, y = np.cos(phi), np.sin(phi)
                vertex.addData3(x, y, z * 0.5)
                normal.addData3(0, 0, z)
                tcoord.addData2(x, y)

        for i in range(num_segments):
            r0 = cyl_rows
            r1 = r0 + cap_rows
            prim.addVertices(r0, r0 + i + 1, r0 + i)
            prim.addVertices(r1, r1 + i, r1 + i + 1)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_box():
    """Make a uniform box geometry.

    Returns:
        Geom -- p3d geometry
    """
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(24)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tcoord = GeomVertexWriter(vdata, "texcoord")

    axes = itertools.permutations(np.eye(3), r=2)
    quad = ((0, 0), (1, 0), (0, 1), (1, 1))

    for x, y in axes:
        z = np.cross(x, y)
        for u, v in quad:
            vertex.addData3(*(x * (u - 0.5) + y * (v - 0.5) + z * 0.5))
            normal.addData3(*z)
            tcoord.addData2(u, v)

    prim = GeomTriangles(Geom.UHStatic)
    for i in range(0, 24, 4):
        prim.addVertices(i + 0, i + 1, i + 2)
        prim.addVertices(i + 2, i + 1, i + 3)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_plane(size=(1.0, 1.0)):
    """Make a plane geometry.

    Arguments:
        size {tuple} -- plane size x,y

    Returns:
        Geom -- p3d geometry
    """
    vformat = GeomVertexFormat.get_v3n3t2()
    vdata = GeomVertexData("vdata", vformat, Geom.UHStatic)
    vdata.uncleanSetNumRows(4)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    tcoord = GeomVertexWriter(vdata, "texcoord")

    quad = ((0, 0), (1, 0), (0, 1), (1, 1))

    for u, v in quad:
        vertex.addData3((u - 0.5) * size[0], (v - 0.5) * size[1], 0)
        normal.addData3(0, 0, 1)
        tcoord.addData2(u, v)

    prim = GeomTriangles(Geom.UHStatic)
    prim.addVertices(0, 1, 2)
    prim.addVertices(2, 1, 3)

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    return geom


def make_sphere(num_segments=16, num_rings=16):
    """Make a uniform UV sphere geometry.

    Keyword Arguments:
        num_segments {int} -- segments number (default: {16})
        num_rings {int} -- rings number (default: {16})

    Returns:
        Geom -- p3d geometry
    """
    return make_capsule(1.0, 0.0, num_segments, num_rings)


def make_points(vertices, colors=None, texture_coords=None, geom=None):
    """Make or update existing points set geometry.

    Arguments:
        root_path {str} -- path to the group's root node
        name {str} -- node name within a group
        vertices {list} -- point coordinates (and other data in a point cloud format)

    Keyword Arguments:
        colors {list} -- colors (default: {None})
        texture_coords {list} -- texture coordinates (default: {None})
        geom {Geom} -- geometry to update (default: {None})

    Returns:
        Geom -- p3d geometry
    """
    if not isinstance(vertices, np.ndarray):
        vertices = np.asarray(vertices, dtype=np.float32)

    if colors is not None:
        if not isinstance(colors, np.ndarray):
            colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = np.uint8(colors * 255)
        vertices = np.column_stack(
            (vertices.view(dtype=np.uint32).reshape(-1, 3), colors.view(dtype=np.uint32))
        )

    if texture_coords is not None:
        if not isinstance(texture_coords, np.ndarray):
            texture_coords = np.asarray(texture_coords)
        vertices = np.column_stack(
            (
                vertices.view(dtype=np.uint32).reshape(-1, 3),
                texture_coords.view(dtype=np.uint32).reshape(-1, 2),
            )
        )

    data = vertices.tostring()

    if geom is None:
        if vertices.strides[0] == 12:
            vformat = GeomVertexFormat.get_v3()
        elif vertices.strides[0] == 16:
            vformat = GeomVertexFormat.get_v3c4()
        elif vertices.strides[0] == 20:
            vformat = GeomVertexFormat.get_v3t2()
        else:
            raise ViewerError(
                "Incompatible point clout format: {},{}".format(vertices.dtype, vertices.shape)
            )

        vdata = GeomVertexData("vdata", vformat, Geom.UHDynamic)
        vdata.unclean_set_num_rows(len(vertices))
        vdata.modify_array_handle(0).set_subdata(0, len(data), data)

        prim = GeomPoints(Geom.UHDynamic)
        prim.clear_vertices()
        prim.add_consecutive_vertices(0, len(vertices))
        prim.close_primitive()

        geom = Geom(vdata)
        geom.add_primitive(prim)
    else:
        vdata = geom.modify_vertex_data()
        vdata.unclean_set_num_rows(len(vertices))
        vdata.modify_array_handle(0).set_subdata(0, len(data), data)

        prim = geom.modify_primitive(0)
        prim.clear_vertices()
        prim.add_consecutive_vertices(0, len(vertices))
        prim.close_primitive()

    return geom
