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
import json
import os
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from re import I

# Third Party
import numpy as np
import torch.distributed as dist
import yaml
from colorama import Fore, Style
from omegaconf import OmegaConf
from tqdm import tqdm

# MegaPose
from megapose.config import (
    BLENDER_INSTALL_DIR,
    BLENDER_PROC_DIR,
    BOP_TOOLKIT_DIR,
    GSO_DIR,
    GSO_NORMALIZED_DIR,
    GSO_ORIG_DIR,
    LOCAL_DATA_DIR,
    MEMORY,
    PROJECT_DIR,
    PYTHON_BIN_PATH,
    SHAPENET_DIR,
)
from megapose.datasets.bop import BOPDataset
from megapose.datasets.gso_dataset import GoogleScannedObjectDataset, make_gso_infos
from megapose.datasets.hdf5_scene_dataset import write_scene_ds_as_hdf5
from megapose.datasets.shapenet_object_dataset import (
    ShapeNetObjectDataset,
    make_shapenet_infos,
)
from megapose.datasets.web_scene_dataset import write_scene_ds_as_wds
from megapose.utils.distributed import get_rank, get_tmp_dir, init_distributed_mode
from megapose.utils.logging import get_logger

logger = get_logger(__name__)

CC_TEXTURE_FOLDER = str(LOCAL_DATA_DIR / "cctextures")
VERBOSE_KWARGS = {
    True: dict(stdout=None, stderr=None),
    False: dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
}
SHAPENET_ORIG_DIR = SHAPENET_DIR / "models_orig"
SHAPENET_SCALED_DIR = SHAPENET_DIR / "models_bop-renderer_scale=0.1"
GSO_ORIG_DIR = GSO_DIR / "models_orig"
GSO_SCALED_DIR = GSO_DIR / "models_bop-renderer_scale=0.1"


def make_setup_dict():
    blender_install_dir = BLENDER_INSTALL_DIR
    setup_dict = {
        "custom_blender_path": str(blender_install_dir),
        "blender_install_path": str(blender_install_dir),
        "pip": [
            "h5py",
            "scikit-image",
            "pypng==0.0.20",
            "scipy",
            "matplotlib",
            "pytz",
            "numpy==1.20",
            "imageio==2.9.0",
        ],
    }
    return setup_dict


def make_initializer(output_dir):
    return {
        "module": "main.Initializer",
        "config": {
            "global": {
                "output_dir": str(output_dir),
            }
        },
    }


def make_box_scene(used_assets=[]):
    cube_dicts = [
        {
            "module": "constructor.BasicMeshInitializer",
            "config": {
                "meshes_to_add": [
                    {"type": "plane", "name": "ground_plane0", "scale": [2, 2, 1]},
                    {
                        "type": "plane",
                        "name": "ground_plane1",
                        "scale": [2, 2, 1],
                        "location": [0, -2, 2],
                        "rotation": [
                            -1.570796,
                            0,
                            0,
                        ],  # switch the sign to turn the normals to the outside
                    },
                    {
                        "type": "plane",
                        "name": "ground_plane2",
                        "scale": [2, 2, 1],
                        "location": [0, 2, 2],
                        "rotation": [1.570796, 0, 0],
                    },
                    {
                        "type": "plane",
                        "name": "ground_plane4",
                        "scale": [2, 2, 1],
                        "location": [2, 0, 2],
                        "rotation": [0, -1.570796, 0],
                    },
                    {
                        "type": "plane",
                        "name": "ground_plane5",
                        "scale": [2, 2, 1],
                        "location": [-2, 0, 2],
                        "rotation": [0, 1.570796, 0],
                    },
                    {
                        "type": "plane",
                        "name": "light_plane",
                        "location": [0, 0, 10],
                        "scale": [3, 3, 1],
                    },
                ]
            },
        },
        {
            "module": "manipulators.MaterialManipulator",
            "config": {
                "selector": {
                    "provider": "getter.Material",
                    "conditions": {"name": "light_plane_material"},
                },
                "cf_switch_to_emission_shader": {
                    "color": {
                        "provider": "sampler.Color",
                        "min": [0.5, 0.5, 0.5, 1.0],
                        "max": [1.0, 1.0, 1.0, 1.0],
                    },
                    "strength": {"provider": "sampler.Value", "type": "float", "min": 3, "max": 6},
                },
            },
        },
        {
            "module": "loader.CCMaterialLoader",
            "config": {
                "folder_path": str(CC_TEXTURE_FOLDER),
                "used_assets": used_assets,
            },
        },
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {"provider": "getter.Entity", "conditions": {"name": "ground_plane.*"}},
                "mode": "once_for_all",
                "cf_randomize_materials": {
                    "randomization_level": 1,
                    "materials_to_replace_with": {
                        "provider": "getter.Material",
                        "random_samples": 1,
                        "conditions": {"cp_is_cc_texture": True},
                    },
                },
            },
        },
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                "selector": {"provider": "getter.Entity", "conditions": {"name": ".*plane.*"}},
                "cp_physics": False,
                "cp_physics_collision_shape": "BOX",
                "cp_category_id": 333,
                "cf_add_modifier": {"name": "Solidify", "thickness": 0.0025},
            },
        },
    ]
    return cube_dicts


def make_shapenet_loader(synset_id, category_id, source_id=None, scale=None):
    # NOTE: No random scale.
    loader_dict = [
        {
            "module": "loader.ShapeNetLoader",
            "config": {
                "data_path": str(SHAPENET_ORIG_DIR),
                "used_synset_id": synset_id,
                "move_object_origin": False,
                "add_properties": {
                    "cp_physics": True,
                    "cp_shapenet_object": True,
                    "cp_category_id": str(category_id),
                },
            },
        },
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                # get all shape net objects, as we have only loaded one, this returns only one entity
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_shapenet_object": True,
                        "cp_category_id": str(category_id),
                        "type": "MESH",
                    },
                },
                "cf_add_modifier": {"name": "Solidify", "thickness": 0.0025},
            },
        },
    ]
    if source_id is not None:
        loader_dict[0]["config"]["used_source_id"] = source_id
    if scale is not None:
        loader_dict[1]["config"]["scale"] = scale
    return loader_dict


def make_gso_loader(obj_id, category_id, scale=None):
    # NOTE: No random scale.
    loader_dict = [
        {
            "module": "loader.GoogleScannedObjectsLoader",
            "config": {
                "dataset_path": str(GSO_NORMALIZED_DIR),
                "obj_id": obj_id,
                "move_object_origin": False,
                "add_properties": {
                    "cp_physics": True,
                    "cp_shapenet_object": True,
                    "cp_category_id": str(category_id),
                },
            },
        },
        {
            "module": "manipulators.EntityManipulator",
            "config": {
                # get all shape net objects, as we have only loaded one, this returns only one entity
                "selector": {
                    "provider": "getter.Entity",
                    "conditions": {
                        "cp_shapenet_object": True,
                        "cp_category_id": str(category_id),
                        "type": "MESH",
                    },
                },
                "cf_add_modifier": {"name": "Solidify", "thickness": 0.0025},
            },
        },
    ]
    if scale is not None:
        loader_dict[1]["config"]["scale"] = scale
    return loader_dict


def make_physics_positioning():
    physics_positioning = {
        "module": "object.PhysicsPositioning",
        "config": {
            "min_simulation_time": 3,
            "max_simulation_time": 10,
            "check_object_interval": 1,
            "solver_iters": 25,
            "substeps_per_frame": 50,
            "collision_margin": 0.0005,
            "friction": 100.0,
            "linear_damping": 0.99,
            "angular_damping": 0.99,
        },
    }
    return physics_positioning


def make_object_pose_sampler():
    object_pose_sampler = {
        "module": "object.ObjectPoseSampler",
        "config": {
            "objects_to_sample": {"provider": "getter.Entity", "conditions": {"cp_physics": True}},
            "pos_sampler": {
                "provider": "sampler.Uniform3d",
                "min": {
                    "provider": "sampler.Uniform3d",
                    "min": [-0.2, -0.2, 0.0],
                    "max": [-0.1, -0.1, 0.0],
                },
                "max": {
                    "provider": "sampler.Uniform3d",
                    "min": [0.1, 0.1, 0.4],
                    "max": [0.2, 0.2, 0.6],
                },
            },
            "rot_sampler": {"provider": "sampler.UniformSO3"},
        },
    }
    return object_pose_sampler


def make_light_sampler(radius_min=1, radius_max=1.5, energy=100):
    light_sampler = {
        "module": "lighting.LightSampler",
        "config": {
            "lights": [
                {
                    "location": {
                        "provider": "sampler.Shell",
                        "center": [0, 0, 0],
                        "radius_min": radius_min,  # now depends on the bottom area of the box
                        "radius_max": radius_max,  # this one too
                        "elevation_min": 5,
                        "elevation_max": 89,
                        "uniform_elevation": True,
                    },
                    "color": {
                        "provider": "sampler.Color",
                        "min": [0.5, 0.5, 0.5, 1.0],
                        "max": [1.0, 1.0, 1.0, 1.0],
                    },
                    "type": "POINT",
                    "energy": 100,
                }
            ]
        },
    }
    return light_sampler


def make_material_randomization():
    return {
        "module": "manipulators.MaterialManipulator",
        "config": {
            "selector": {
                "provider": "getter.Material",
                "conditions": {"name": ".*", "use_nodes": True},
            },
            "cf_set_specular": {
                "provider": "sampler.Value",
                "type": "float",
                "min": 0.3,
                "max": 1.0,
            },
            "cf_set_roughness": {
                "provider": "sampler.Value",
                "type": "float",
                "min": 0.0,
                "max": 0.5,
            },
            "cf_set_metallic": {
                "provider": "sampler.Value",
                "type": "float",
                "min": 0.0,
                "max": 0.5,
            },
        },
    }


def make_camera_sampler(cam_intrinsics, num_samples=25, radius_min=0.4, radius_max=1.5):
    camera_sampler = {
        "module": "camera.CameraSampler",
        "config": {
            "intrinsics": cam_intrinsics,
            "cam_poses": [
                {
                    "proximity_checks": {"min": 0.3},
                    "excluded_objs_in_proximity_check": {
                        "provider": "getter.Entity",
                        "conditions": {"name": "ground_plane.*", "type": "MESH"},
                    },
                    "number_of_samples": num_samples,
                    "location": {
                        "provider": "sampler.Shell",
                        "center": [0, 0, 0],
                        "radius_min": radius_min,
                        "radius_max": radius_max,
                        "elevation_min": 5,
                        "elevation_max": 89,
                        "uniform_elevation": True,
                    },
                    "rotation": {
                        "format": "look_at",
                        "value": {
                            "provider": "getter.POI",
                            "selector": {
                                "provider": "getter.Entity",
                                "conditions": {
                                    "cp_shapenet_object": True,
                                    "type": "MESH",
                                },
                                "random_samples": 15,
                            },
                        },
                        "inplane_rot": {
                            "provider": "sampler.Value",
                            "type": "float",
                            "min": -3.14159,
                            "max": 3.14159,
                        },
                    },
                }
            ],
        },
    }
    return camera_sampler


def make_renderer():
    renderer = {
        "module": "renderer.RgbRenderer",
        "config": {"samples": 50, "render_distance": True, "image_type": "JPEG"},
    }
    return renderer


def make_writer(depth_scale=0.1, ignore_dist_thresh=5.0):
    return {
        "module": "writer.BopWriter",
        "config": {
            "append_to_existing_output": False,
            "depth_scale": depth_scale,
            "ignore_dist_thres": ignore_dist_thresh,
            "postprocessing_modules": {"distance": [{"module": "postprocessing.Dist2Depth"}]},
        },
    }


def make_script(output_dir, objects, textures, cfg, seed):
    np_random = np.random.RandomState(seed)
    output_dir = Path(output_dir)

    fx = np_random.uniform(*cfg.focal_interval)
    fy = fx + np_random.uniform(*cfg.diff_focal_interval)
    h, w = min(cfg.render_size), max(cfg.render_size)
    cx, cy = w / 2, h / 2
    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ]
    ).tolist()
    intrinsics = dict(cam_K=K, resolution_x=w, resolution_y=h)

    modules = [
        make_initializer(output_dir),
        *make_box_scene(used_assets=textures),
    ]

    for obj in objects:
        if obj["category_id"].startswith("shapenet"):
            modules += make_shapenet_loader(
                synset_id=obj["synset_id"],
                scale=obj["scale"],
                category_id=obj["category_id"],
                source_id=obj["source_id"],
            )
        elif obj["category_id"].startswith("gso"):
            modules += make_gso_loader(
                obj_id=obj["obj_id"], scale=obj["scale"], category_id=obj["category_id"]
            )
        else:
            raise ValueError(obj)

    modules += [
        make_material_randomization(),
        make_object_pose_sampler(),
        make_physics_positioning(),
        make_light_sampler(radius_min=cfg.light_radius_min, radius_max=cfg.light_radius_max),
        make_camera_sampler(
            cam_intrinsics=intrinsics,
            num_samples=cfg.camera_num_samples_per_chunk,
            radius_min=cfg.camera_radius_min,
            radius_max=cfg.camera_radius_max,
        ),
        make_renderer(),
        make_writer(),
    ]
    script = {
        "version": 3,
        "setup": make_setup_dict(),
        "modules": modules,
    }
    return script


def run_script(script, script_path, verbose=True):
    Path(script_path).write_text(json.dumps(script))
    seed = script["seed"]
    env = os.environ.copy()
    env["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    run_path = BLENDER_PROC_DIR / "run.py"
    subprocess.run(
        [str(PYTHON_BIN_PATH), str(run_path), str(script_path)], env=env, **VERBOSE_KWARGS[verbose]
    )
    return


@MEMORY.cache
def load_textures_names():
    texture_names = [
        p.name for p in Path(CC_TEXTURE_FOLDER).iterdir() if len(list(p.glob("*2K_Color.jpg"))) > 0
    ]
    return texture_names


def make_one_scene_script(cfg, output_dir, seed):
    np_random = np.random.RandomState(seed)
    if cfg.model_type == "shapenet":
        synsets = make_shapenet_infos(SHAPENET_ORIG_DIR, "model_normalized.obj")
        main_synsets = [
            synset
            for synset in synsets
            if len(synset.parents) == 0 and len(synset.models_descendants) > 0
        ]
        objects = []
        for n in range(cfg.n_objects):
            synset = np_random.choice(main_synsets)
            source_id = np_random.choice(synset.models_descendants)
            obj = dict(
                synset_id=synset.synset_id,
                source_id=source_id,
                category_id=f"shapenet_{synset.synset_id}_{source_id}",
                scale=[cfg.scale, cfg.scale, cfg.scale],
            )
            objects.append(obj)
    elif cfg.model_type == "gso":
        object_ids = make_gso_infos(GSO_NORMALIZED_DIR)
        objects = []
        for n in range(cfg.n_objects):
            obj_id = np_random.choice(object_ids)
            obj = dict(
                obj_id=obj_id,
                category_id=f"gso_{obj_id}",
                scale=[cfg.scale, cfg.scale, cfg.scale],
            )
            objects.append(obj)
    else:
        raise ValueError(cfg.model_type)

    textures = load_textures_names()
    this_scene_floor_textures = [np_random.choice(textures)]
    script = make_script(output_dir, objects, this_scene_floor_textures, cfg, seed)
    script["seed"] = seed
    scene_infos = dict(objects=objects, floor_textures=this_scene_floor_textures, seed=seed)
    return scene_infos, script


def make_masks_and_gt_infos(chunk_dir, is_shapenet=True, verbose=True):
    bop_toolkit_dir = BOP_TOOLKIT_DIR
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + str(bop_toolkit_dir)
    env["COSYPOSE_DIR"] = str(PROJECT_DIR)
    script_path = bop_toolkit_dir / "scripts/calc_masks_custom.py"
    success = True
    cmd = ["python", str(script_path), "--chunk-dir", chunk_dir, "--overwrite-models"]
    if is_shapenet:
        cmd += ["--shapenet-dir", str(SHAPENET_SCALED_DIR)]
    else:
        cmd += ["--gso-dir", str(GSO_SCALED_DIR)]
    p = subprocess.run(cmd, env=env, **VERBOSE_KWARGS[verbose])
    success = success and p.returncode == 0

    script_path = bop_toolkit_dir / "scripts/calc_gt_info_custom.py"
    cmd = ["python", str(script_path), "--chunk-dir", chunk_dir, "--overwrite-models"]
    if is_shapenet:
        cmd += ["--shapenet-dir", str(SHAPENET_SCALED_DIR)]
    else:
        cmd += ["--gso-dir", str(GSO_SCALED_DIR)]
    p = subprocess.run(cmd, env=env, **VERBOSE_KWARGS[verbose])
    success = success and p.returncode == 0
    return success


def make_shapenet_model_infos():
    dataset = ShapeNetObjectDataset(SHAPENET_DIR, split="orig")
    all_labels = [obj["label"] for obj in dataset.objects]
    return all_labels


def make_dataset_cfg(cfg):
    cfg.light_radius_min = 1.0
    cfg.light_radius_max = 1.5

    cfg.camera_radius_min = 0.4
    cfg.camera_radius_max = 1.5
    cfg.camera_num_samples_per_chunk = 25
    cfg.render_size = (720, 540)
    cfg.focal_interval = (500, 3000)
    cfg.diff_focal_interval = (1, 50)

    cfg.save_hdf5 = False
    cfg.save_webdataset = True
    cfg.save_files = False

    cfg.n_objects = 3
    cfg.scale = 0.1

    cfg.n_scenes = 2

    cfg.hardware = dict()
    cfg.hardware.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.hardware.rank = int(os.environ.get("RANK", 0))
    cfg.hardware.n_proc_per_gpu = 3
    cfg.model_type = "shapenet"

    if cfg.dataset_id == "shapenet_50k":
        cfg.camera_num_samples_per_chunk = 25
        cfg.n_objects = 30
        cfg.n_scenes = int(50e3 / cfg.camera_num_samples_per_chunk)
        cfg.ds_name = "shapenet_50k"

    elif cfg.dataset_id == "shapenet_1M":
        cfg.camera_num_samples_per_chunk = 20
        cfg.n_objects = 40
        cfg.n_scenes = int(1e6 / cfg.camera_num_samples_per_chunk)
        cfg.ds_name = "shapenet_1M"

    elif cfg.dataset_id == "gso_1M":
        cfg.camera_radius_min = 0.8
        cfg.camera_radius_max = 1.8
        cfg.camera_num_samples_per_chunk = 40
        cfg.n_objects = 20
        cfg.scale = 0.1
        cfg.model_type = "gso"
        cfg.n_scenes = int(1e6 / cfg.camera_num_samples_per_chunk)
        cfg.ds_name = "gso_1M"

    elif cfg.resume_dataset is not None:
        pass

    else:
        raise ValueError(cfg.config_id)

    if cfg.resume_dataset is not None:
        logger.info(f"{Fore.RED}Resuming {cfg.resume_dataset} {Style.RESET_ALL}")
        resume_cfg = OmegaConf.load(
            LOCAL_DATA_DIR / "blender_pbr_datasets" / cfg.resume_dataset / "config.yaml"
        )
        resume_cfg = OmegaConf.merge(
            resume_cfg, OmegaConf.masked_copy(cfg, ["resume_dataset", "hardware", "verbose"])
        )
        cfg = resume_cfg
    else:
        logger.info(f"{Fore.GREEN}Recording dataset: {cfg.dataset_id} {Style.RESET_ALL}")

    if cfg.debug:
        cfg.camera_num_samples_per_chunk = 5
        cfg.n_scenes = 2
        cfg.ds_name += "_debug"
        cfg.n_objects = 2

    if cfg.few:
        cfg.ds_name += "_debug_few"
        cfg.n_scenes = 5

    cfg.ds_dir = str(LOCAL_DATA_DIR / "blender_pbr_datasets" / cfg.ds_name)
    return cfg


def record_chunk(cfg, ds_dir, chunk_id):
    script_path = ds_dir / "configs" / f"chunk={chunk_id}.yaml"
    script_path.parent.mkdir(exist_ok=True)
    output_dir = ds_dir / f"chunks/chunk={chunk_id}"
    output_dir.parent.mkdir(exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Generate script
    scene_infos, script = make_one_scene_script(cfg, output_dir, seed=chunk_id)

    # Generate data with Blender
    run_script(script, script_path, verbose=cfg.verbose)
    chunk_info = dict(
        chunk_id=chunk_id,
        script_path=str(script_path),
        output_dir=str(output_dir),
        scene_infos=scene_infos,
        scale=cfg["scale"],
    )
    gt_path = output_dir / f"bop_data/train_pbr/{0:06d}/scene_gt.json"
    gt = json.loads(gt_path.read_text())
    for im_id, im_gt in gt.items():
        # Remove non shapenet objects (e.g. walls)
        im_gt = [d for d in im_gt if isinstance(d["obj_id"], str)]
        gt[im_id] = im_gt
    gt_path.write_text(json.dumps(gt))
    (output_dir / "chunk_infos.json").write_text(json.dumps(chunk_info))

    # Generate masks and gt infos
    success = make_masks_and_gt_infos(
        output_dir, verbose=cfg.verbose, is_shapenet=cfg.model_type == "shapenet"
    )

    # HDF5 dataset generation
    if cfg.save_hdf5:
        shutil.copy(
            ds_dir / "shapenet_labels.json", output_dir / "bop_data" / "shapenet_labels.json"
        )
        scene_ds = BOPDataset(
            output_dir / "bop_data",
            split="train_pbr",
            load_depth=True,
            allow_cache=False,
            per_view_annotations=False,
        )
        write_scene_ds_as_hdf5(
            scene_ds, output_dir / f"bop_data/train_pbr/{0:06d}/data.hdf5", n_reading_workers=4
        )

    if cfg.save_webdataset:
        shutil.copy(
            ds_dir / "shapenet_labels.json", output_dir / "bop_data" / "shapenet_labels.json"
        )
        scene_ds = BOPDataset(
            output_dir / "bop_data",
            split="train_pbr",
            load_depth=True,
            allow_cache=False,
            per_view_annotations=False,
        )
        write_scene_ds_as_wds(
            scene_ds, output_dir / f"bop_data/train_pbr/{0:06d}/", n_reading_workers=4
        )

    # Move everything to base directory
    chunk_scene_dir = output_dir / f"bop_data/train_pbr/{0:06d}"
    train_pbr_dir = ds_dir / "train_pbr"
    target_dir = train_pbr_dir / f"{chunk_id:06d}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    if cfg.save_files and success:
        shutil.copytree(chunk_scene_dir, target_dir)
    if cfg.save_hdf5 and success:
        target_dir.mkdir(exist_ok=True)
        shutil.copy(chunk_scene_dir / "data.hdf5", target_dir / "data.hdf5")
    if cfg.save_webdataset and success:
        target_dir.mkdir(exist_ok=True)
        shutil.copy(chunk_scene_dir / "shard-00000000.tar", target_dir / "shard-00000000.tar")
        shutil.copy(chunk_scene_dir / "ds_infos.json", target_dir / "ds_infos.json")
    shutil.rmtree(output_dir)
    return


def find_chunks_to_record(cfg, chunk_ids):
    this_chunk_ids = np.array_split(chunk_ids, cfg.hardware.world_size)[cfg.hardware.rank].tolist()
    chunk_ids = []
    for chunk_id in this_chunk_ids:
        if not (ds_dir / f"train_pbr/{chunk_id:06d}").exists():
            chunk_ids.append(chunk_id)
    return chunk_ids


if __name__ == "__main__":
    cfg = OmegaConf.create(
        dict(
            dataset_id="test",
            resume_dataset=None,
            debug=False,
            verbose=False,
            run_comment="",
            overwrite=False,
            few=False,
            chunk_ids=None,
        )
    )
    cli_cfg = OmegaConf.from_cli()
    if cli_cfg is not None:
        cfg = OmegaConf.merge(
            cfg,
            cli_cfg,
        )
    cfg = make_dataset_cfg(cfg)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    logger.info(f"Recording dataset cfg: \n {OmegaConf.to_yaml(cfg)}")

    init_distributed_mode()

    ds_dir = Path(cfg.ds_dir)
    if get_rank() == 0:
        if ds_dir.exists():
            if cfg.resume_dataset is not None:
                pass
            elif cfg.chunk_ids is not None:
                pass
            elif cfg.overwrite:
                shutil.rmtree(cfg.ds_dir)
            else:
                raise ValueError("There is already a dataset with this name")

        if cfg.resume_dataset is None:
            ds_dir.mkdir(exist_ok=cfg.chunk_ids is not None)
            OmegaConf.save(cfg, ds_dir / "config.yaml")
            if cfg.model_type == "gso":
                object_labels = make_gso_infos(GSO_NORMALIZED_DIR)
            else:
                object_labels = make_shapenet_model_infos()
            shapenet_labels_path = ds_dir / "shapenet_labels.json"
            if not shapenet_labels_path.exists():
                (ds_dir / "shapenet_labels.json").write_text(json.dumps(object_labels))

        train_pbr_dir = ds_dir / "train_pbr"
        train_pbr_dir.mkdir(exist_ok=True)

    dist.barrier()

    if cfg.chunk_ids is None:
        chunk_ids = np.arange(cfg.n_scenes)
    else:
        chunk_ids = cfg.chunk_ids
    chunk_ids = find_chunks_to_record(cfg, chunk_ids)
    for chunk_id in tqdm(chunk_ids, ncols=80):
        chunk_id = int(chunk_id)
        record_chunk(cfg, ds_dir, chunk_id)
    dist.barrier()

# TODO: Add multi process, multi-gpu, arguments for ti.
# TODO: Generate some images for the meeting.
# TODO: Add multi-GPU recording.
