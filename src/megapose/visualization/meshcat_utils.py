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
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import numpy as np
import trimesh
import trimesh.transformations as tra

"""
Some code borrowed from https://github.com/google-research/ravens
under Apache license
"""


def isRotationMatrix(M, tol=1e-4):
    """Checks if something is a valid rotation matrix."""
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (np.abs(np.linalg.det(M) - 1) < tol):
        tag = True

    if tag is False:
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag


def trimesh_to_meshcat_geometry(mesh, use_vertex_colors=False):
    """Converts a trimesh geometry to a meshcat geometry.

    Only deals with vertices/faces, no textures
    Args:
        mesh: trimesh.TriMesh object
    """

    if use_vertex_colors:
        visual = mesh.visual
        if isinstance(visual, trimesh.visual.TextureVisuals):
            visual = visual.to_color()

        vertex_colors = visual.vertex_colors[:, :3]
        vertex_colors = vertex_colors / 255.0
    else:
        vertex_colors = None
    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces, vertex_colors)


def rgb2hex(rgb):
    """
    Converts rgb color to hex

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def visualize_mesh(vis, mesh, transform=None, color=None, texture_png=None):
    mesh_vis = trimesh_to_meshcat_geometry(mesh)
    material = None

    if color is not None:
        color = np.array(color)
        if isinstance(color, str) and color == "random":
            color = np.random.randint(low=0, high=256, size=3)
            color_hex = rgb2hex(tuple(color))
            material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
        else:  # color is np.ndarray, e.g. [1,0,0]
            if not np.issubdtype(color.dtype, np.int):
                color = (color * 255).astype(np.int32)
            color_hex = rgb2hex(tuple(color))
            material = meshcat.geometry.MeshPhongMaterial(color=color_hex)

    if texture_png is not None:
        material = g.MeshLambertMaterial(
            map=g.ImageTexture(image=g.PngImage.from_file(texture_png))
        )
        print("material")

    vis.set_object(mesh_vis, material)
    if transform is not None:
        vis.set_transform(transform)


def visualize_scene(vis, object_dict, randomize_color=True):

    for name, data in object_dict.items():

        # try assigning a random color
        if randomize_color:
            if "color" in data:
                color = data["color"]

                # if it's not an integer, convert it to [0,255]
                if not np.issubdtype(color.dtype, np.int):
                    color = (color * 255).astype(np.int32)
            else:
                color = np.random.randint(low=0, high=256, size=3)
                data["color"] = color
        else:
            color = [0, 255, 0]

        mesh_vis = trimesh_to_meshcat_geometry(data["mesh"])
        color_hex = rgb2hex(tuple(color))
        material = meshcat.geometry.MeshPhongMaterial(color=color_hex)

        mesh_name = f"{name}/mesh"
        vis[mesh_name].set_object(mesh_vis, material)
        vis[mesh_name].set_transform(data["T_world_object"])

        frame_name = f"{name}/transform"
        make_frame(vis, frame_name, T=data["T_world_object"])


def create_visualizer(clear=True, zmq_url="tcp://127.0.0.1:6000"):
    print(
        "Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a"
        f" server. Communicating on zmq_url={zmq_url}"
    )
    vis = meshcat.Visualizer(zmq_url=zmq_url)
    if clear:
        vis.delete()

    print("Created meschat visualizer!")
    return vis


def make_frame(
    vis, name, h=0.15, radius=0.001, o=1.0, T=None, transform=None, ignore_invalid_transform=False
):
    """Add a red-green-blue triad to the Meschat visualizer.

    # Update to use `meshcat.geometry.triad`
    Args:
      vis (MeshCat Visualizer): the visualizer
      name (string): name for this frame (should be unique)
      h (float): height of frame visualization
      radius (float): radius of frame visualization
      o (float): opacity
    """
    vis[name]["x"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o),
    )
    rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
    rotate_x[0, 3] = h / 2
    vis[name]["x"].set_transform(rotate_x)

    vis[name]["y"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o),
    )
    rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
    rotate_y[1, 3] = h / 2
    vis[name]["y"].set_transform(rotate_y)

    vis[name]["z"].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[name]["z"].set_transform(rotate_z)

    if T is not None:
        transform = T

    if transform is not None:

        if not ignore_invalid_transform:
            is_valid = isRotationMatrix(transform[:3, :3])
            if not is_valid:
                raise ValueError("meshcat_utils:attempted to visualize invalid transform T")

        vis[name].set_transform(transform)


def draw_grasp(vis, line_name, transform, h=0.15, radius=0.001, o=1.0, color=[255, 0, 0]):
    """Draws line to the Meshcat visualizer.
    Args:
      vis (Meshcat Visualizer): the visualizer
      line_name (string): name for the line associated with the grasp.
      transform (numpy array): 4x4 specifying transformation of grasps.
      radius (float): radius of frame visualization
      o (float): opacity
      color (list): color of the line.
    """
    vis[line_name].set_object(
        g.Cylinder(height=h, radius=radius),
        g.MeshLambertMaterial(color=rgb2hex(tuple(color)), reflectivity=0.8, opacity=o),
    )
    rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
    rotate_z[2, 3] = h / 2
    vis[line_name].set_transform(transform @ rotate_z)


def visualize_pointcloud(vis, name, pc, color=None, transform=None, **kwargs):
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        # Resize the color np array if needed.
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)

        # Divide it by 255 to make sure the range is between 0 and 1,
        color = color.astype(np.float32) / 255
    else:
        color = np.ones_like(pc)

    vis[name].set_object(meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs))

    if transform is not None:
        vis[name].set_transform(transform)


def visualize_bbox(vis, name, dims, transform=None, T=None):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (MeshCat Visualizer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry

    """
    material = meshcat.geometry.MeshBasicMaterial(wireframe=True)
    bbox = meshcat.geometry.Box(dims)
    vis[name].set_object(bbox, material)

    if T is not None:
        transform = T

    if transform is not None:
        vis[name].set_transform(transform)


def visualize_transform_manager(vis, tm, frame, **kwargs):
    """Visualizes pytransform3d TransformManager."""
    for node in tm.nodes:
        t = tm.get_transform(node, frame)
        make_frame(vis, node, transform=t, **kwargs)


def get_pointcloud(depth, intrinsics, flatten=False, remove_zero_depth_points=True):
    """Projects depth image to pointcloud

    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
        flatten: whether to flatten pointcloud

    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)

    if flatten:
        points = np.reshape(points, [height * width, 3])
    return points
