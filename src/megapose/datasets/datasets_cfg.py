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
from typing import List, Optional, Tuple

# Third Party
import numpy as np
import pandas as pd

# MegaPose
from megapose.config import (
    BOP_DS_DIR,
    BOP_PANDA3D_DS_DIR,
    GSO_DIR,
    LOCAL_DATA_DIR,
    SHAPENET_DIR,
    SHAPENET_MODELNET_CATEGORIES,
    WDS_DS_DIR,
    YCBV_OBJECT_NAMES,
)
from megapose.datasets.bop_object_datasets import BOPObjectDataset
from megapose.datasets.bop_scene_dataset import BOPDataset, remap_bop_targets
from megapose.datasets.deepim_modelnet import DeepImModelNetDataset
from megapose.datasets.gso_dataset import GoogleScannedObjectDataset
from megapose.datasets.modelnet_object_dataset import ModelNetObjectDataset
from megapose.datasets.object_dataset import RigidObjectDataset
from megapose.datasets.scene_dataset import SceneDataset
from megapose.datasets.shapenet_object_dataset import ShapeNetObjectDataset
from megapose.datasets.urdf_dataset import UrdfDataset
from megapose.datasets.web_scene_dataset import WebSceneDataset
from megapose.utils.logging import get_logger

logger = get_logger(__name__)


def keep_bop19(ds: SceneDataset) -> SceneDataset:
    assert ds.frame_index is not None
    assert isinstance(ds, BOPDataset)
    targets = pd.read_json(ds.ds_dir / "test_targets_bop19.json")
    targets = remap_bop_targets(targets)
    targets = targets.loc[:, ["scene_id", "view_id"]].drop_duplicates()
    index = ds.frame_index.merge(targets, on=["scene_id", "view_id"]).reset_index(drop=True)
    assert len(index) == len(targets)
    ds.frame_index = index
    return ds


def make_scene_dataset(
    ds_name: str,
    load_depth: bool = False,
    n_frames: Optional[int] = None,
) -> SceneDataset:

    # BOP challenge splits
    if ds_name == "hb.bop19":
        ds_dir = BOP_DS_DIR / "hb"
        ds: SceneDataset = BOPDataset(ds_dir, split="test_primesense", label_format="hb-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "icbin.bop19":
        ds_dir = BOP_DS_DIR / "icbin"
        ds = BOPDataset(ds_dir, split="test", label_format="icbin-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "itodd.bop19":
        ds_dir = BOP_DS_DIR / "itodd"
        ds = BOPDataset(ds_dir, split="test", label_format="itodd-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "lmo.bop19":
        ds_dir = BOP_DS_DIR / "lmo"
        ds = BOPDataset(ds_dir, split="test", label_format="lm-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "tless.bop19":
        ds_dir = BOP_DS_DIR / "tless"
        ds = BOPDataset(ds_dir, split="test_primesense", label_format="tless-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "tudl.bop19":
        ds_dir = BOP_DS_DIR / "tudl"
        ds = BOPDataset(ds_dir, split="test", label_format="tudl-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "ycbv.bop19":
        ds_dir = BOP_DS_DIR / "ycbv"
        ds = BOPDataset(ds_dir, split="test", label_format="ycbv-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "ruapc.bop19":
        ds_dir = BOP_DS_DIR / "ruapc"
        ds = BOPDataset(ds_dir, split="test", label_format="ruapc-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "hope.bop19":
        ds_dir = BOP_DS_DIR / "hope"
        ds = BOPDataset(ds_dir, split="test", label_format="hope-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "tyol.bop19":
        ds_dir = BOP_DS_DIR / "tyol"
        ds = BOPDataset(ds_dir, split="test", label_format="tyol-{label}")
        ds = keep_bop19(ds)
    elif ds_name == "lm.bop19":
        ds_dir = BOP_DS_DIR / "lm"
        ds = BOPDataset(ds_dir, split="test", label_format="lm-{label}")
        ds = keep_bop19(ds)

    # Non-BOP challenge BOP dataset splits
    elif ds_name == "tless.primesense.train":
        ds_dir = BOP_DS_DIR / "tless"
        ds = BOPDataset(ds_dir, split="train_primesense", label_format="tless-{label}")
    elif ds_name == "tless.primesense.test":
        ds_dir = BOP_DS_DIR / "tless"
        ds = BOPDataset(ds_dir, split="test_primesense", label_format="tless-{label}")
    elif ds_name == "ycbv.train.real":
        ds_dir = BOP_DS_DIR / "ycbv"
        ds = BOPDataset(ds_dir, split="train_real", label_format="ycbv-{label}")
    elif ds_name == "ycbv.test":
        ds_dir = BOP_DS_DIR / "ycbv"
        ds = BOPDataset(ds_dir, split="test", label_format="ycbv-{label}")
    elif ds_name == "lmo.test":
        ds_dir = BOP_DS_DIR / "lmo"
        ds = BOPDataset(ds_dir, split="test", label_format="lm-{label}")
    elif ds_name == "hb.val":
        ds_dir = BOP_DS_DIR / "hb"
        ds = BOPDataset(ds_dir, split="val_primesense", label_format="hb-{label}")
    elif ds_name == "itodd.val":
        ds_dir = BOP_DS_DIR / "itodd"
        ds = BOPDataset(ds_dir, split="val", label_format="itodd-{label}")
    elif ds_name == "tudl.train.real":
        ds_dir = BOP_DS_DIR / "tudl"
        ds = BOPDataset(ds_dir, split="train_real", label_format="tudl-{label}")

    # PBR training sets
    elif ds_name == "hb.pbr":
        ds_dir = BOP_DS_DIR / "hb"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="hb-{label}")
    elif ds_name == "icbin.pbr":
        ds_dir = BOP_DS_DIR / "icbin"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="icbin-{label}")
    elif ds_name == "itodd.pbr":
        ds_dir = BOP_DS_DIR / "itodd"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="itodd-{label}")
    elif ds_name == "lm.pbr":
        ds_dir = BOP_DS_DIR / "lm"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="lm-{label}")
    elif ds_name == "tless.pbr":
        ds_dir = BOP_DS_DIR / "tless"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="tless-{label}")
    elif ds_name == "tudl.pbr":
        ds_dir = BOP_DS_DIR / "tudl"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="tudl-{label}")
    elif ds_name == "ycbv.pbr":
        ds_dir = BOP_DS_DIR / "ycbv"
        ds = BOPDataset(ds_dir, split="train_pbr", label_format="ycbv-{label}")

    # ModelNet40
    elif ds_name.startswith("modelnet."):
        _, category, split = ds_name.split(".")
        n_objects = (
            30
            if category
            in {"bathtub", "bookshelf", "guitar", "range_hood", "sofa", "wardrobe", "tv_stand"}
            else 50
        )
        ds = DeepImModelNetDataset(
            LOCAL_DATA_DIR / "modelnet40",
            category=category,
            split=split,
            n_objects=n_objects,
            n_images_per_object=50,
        )

    # Datasets in webdataset format
    elif ds_name.startswith("webdataset."):
        ds_name = ds_name[len("webdataset.") :]
        ds = WebSceneDataset(WDS_DS_DIR / ds_name)

    else:
        raise ValueError(ds_name)

    ds.load_depth = load_depth
    if n_frames is not None:
        assert ds.frame_index is not None
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    return ds


def make_object_dataset(ds_name: str) -> RigidObjectDataset:
    # BOP original models
    if ds_name == "tless.cad":
        ds: RigidObjectDataset = BOPObjectDataset(
            BOP_DS_DIR / "tless/models_cad", label_format="tless-{label}"
        )
    elif ds_name == "tless.eval":
        ds = BOPObjectDataset(BOP_DS_DIR / "tless/models_eval", label_format="tless-{label}")
    elif ds_name == "tless.reconst":
        ds = BOPObjectDataset(BOP_DS_DIR / "tless/models_reconst", label_format="tless-{label}")
    elif ds_name == "ycbv":
        ds = BOPObjectDataset(BOP_DS_DIR / "ycbv/models", label_format="ycbv-{label}")
    elif ds_name == "hb":
        ds = BOPObjectDataset(BOP_DS_DIR / "hb/models", label_format="hb-{label}")
    elif ds_name == "icbin":
        ds = BOPObjectDataset(BOP_DS_DIR / "icbin/models", label_format="icbin-{label}")
    elif ds_name == "itodd":
        ds = BOPObjectDataset(BOP_DS_DIR / "itodd/models", label_format="itodd-{label}")
    elif ds_name in {"lm", "lmo"}:
        ds = BOPObjectDataset(BOP_DS_DIR / "lm/models", label_format="lm-{label}")
    elif ds_name == "tudl":
        ds = BOPObjectDataset(BOP_DS_DIR / "tudl/models", label_format="tudl-{label}")
    elif ds_name == "tyol":
        ds = BOPObjectDataset(BOP_DS_DIR / "tyol/models", label_format="tyol-{label}")
    elif ds_name == "ruapc":
        ds = BOPObjectDataset(BOP_DS_DIR / "ruapc/models", label_format="ruapc-{label}")
    elif ds_name == "hope":
        ds = BOPObjectDataset(BOP_DS_DIR / "hope/models", label_format="hope-{label}")

    # BOP models converted for Panda3D
    # TODO: Is this necessary ?
    elif ds_name == "hb.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "hb/models", label_format="hb-{label}")
    elif ds_name == "icbin.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "icbin/models", label_format="icbin-{label}")
    elif ds_name == "itodd.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "itodd/models", label_format="itodd-{label}")
    elif ds_name == "lm.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "lm/models", label_format="lm-{label}")
    elif ds_name == "tless.cad.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "tless/models_cad", label_format="tless-{label}")
    elif ds_name == "ycbv.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "ycbv/models", label_format="ycbv-{label}")
    elif ds_name == "tudl.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "tudl/models", label_format="tudl-{label}")
    elif ds_name == "tyol.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "tyol/models", label_format="tyol-{label}")
    elif ds_name == "ruapc.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "ruapc/models", label_format="ruapc-{label}")
    elif ds_name == "hope.panda3d":
        ds = BOPObjectDataset(BOP_PANDA3D_DS_DIR / "hope/models", label_format="hope-{label}")

    # GSO
    elif ds_name == "gso.orig":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="orig")
    elif ds_name == "gso.normalized":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="normalized")
    elif ds_name == "gso.panda3d":
        ds = GoogleScannedObjectDataset(GSO_DIR, split="panda3d")

    # ModelNet
    elif ds_name.startswith("modelnet."):
        _, category, split, orig_str = ds_name.split(".")
        rescaled = "rescaled" in orig_str
        n_objects = (
            30
            if category
            in {"bathtub", "bookshelf", "guitar", "range_hood", "sofa", "wardrobe", "tv_stand"}
            else 50
        )
        ds = ModelNetObjectDataset(
            LOCAL_DATA_DIR / "modelnet40",
            category=category,
            split=split,
            rescaled=rescaled,
            n_objects=n_objects,
        )

    # ShapeNet
    # shapenet.{filters=20mb_50k,remove_modelnet,...}.split
    elif ds_name.startswith("shapenet."):
        ds_name = ds_name[len("shapenet.") :]

        filters_list: List[str] = []
        if ds_name.startswith("filters="):
            filter_str = ds_name.split(".")[0]
            filters_list = filter_str.split("filters=")[1].split(",")
            ds_name = ds_name[len(filter_str) + 1 :]

        model_split = ds_name
        ds = ShapeNetObjectDataset(SHAPENET_DIR, split=model_split)

        for filter_str in filters_list:
            if filter_str == "remove_modelnet":
                keep_labels = set(
                    [
                        obj.label
                        for obj in ds.objects
                        if obj.category not in SHAPENET_MODELNET_CATEGORIES
                    ]
                )
            else:
                keep_labels = set(
                    json.loads(
                        (SHAPENET_DIR / "stats" / ("shapenet_" + filter_str))
                        .with_suffix(".json")
                        .read_text()
                    )
                )
            ds = ds.filter_objects(keep_labels)

    # GSO
    # gso.{nobjects=500,...}.split
    elif ds_name.startswith("gso."):
        ds_name = ds_name[len("gso.") :]

        n_objects_: Optional[int] = None
        if ds_name.startswith("nobjects="):
            nobjects_str = ds_name.split(".")[0]
            n_objects_ = int(nobjects_str.split("=")[1])
            ds_name = ds_name[len(nobjects_str) + 1 :]

        model_split = ds_name
        ds = GoogleScannedObjectDataset(GSO_DIR, split=model_split)
        if n_objects_ is not None:
            np_random = np.random.RandomState(0)
            keep_labels = set(
                np_random.choice(
                    [obj.label for obj in ds.objects], n_objects_, replace=False
                ).tolist()
            )
            ds = ds.filter_objects(keep_labels)

    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name: str) -> RigidObjectDataset:
    # BOP
    if ds_name == "tless.cad":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "tless.cad", mesh_units="mm", label_format="tless-{label}"
        )
    elif ds_name == "tless.reconst":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "tless.reconst",
            mesh_units="mm",
            label_format="tless-{label}",
        )

    elif ds_name == "tless":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "tless.cad", mesh_units="mm", label_format="tless-{label}"
        )
    elif ds_name == "ycbv":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "ycbv", mesh_units="mm", label_format="ycbv-{label}"
        )
    elif ds_name == "hb":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "hb", mesh_units="mm", label_format="hb-{label}"
        )
    elif ds_name == "icbin":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "icbin", mesh_units="mm", label_format="icbin-{label}"
        )
    elif ds_name == "itodd":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "itodd", mesh_units="mm", label_format="itodd-{label}"
        )
    elif ds_name == "lm":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "lm", mesh_units="mm", label_format="lm-{label}"
        )
    elif ds_name == "tudl":
        ds = UrdfDataset(
            LOCAL_DATA_DIR / "urdfs" / "tudl", mesh_units="mm", label_format="tudl-{label}"
        )

    else:
        raise ValueError(ds_name)
    return ds


def get_obj_ds_info(ds_name: str) -> Tuple[Optional[str], str]:
    urdf_ds_name = None # Only used for bullet compatibility
    if ds_name == "ycbv.bop19":
        ds_name = "ycbv"
        urdf_ds_name = "ycbv"
        obj_ds_name = "ycbv.panda3d"
    elif ds_name in ["lm.bop19", "lmo.bop19"]:
        urdf_ds_name = "lm"
        obj_ds_name = "lm.panda3d"
    elif ds_name == "tless.bop19":
        obj_ds_name = "tless.panda3d"
    elif ds_name == "hope.bop19":
        obj_ds_name = "hope.panda3d"
    elif ds_name == "hb.bop19":
        obj_ds_name = "hb.panda3d"
    elif ds_name == "tudl.bop19":
        obj_ds_name = "tudl.panda3d"
    elif ds_name == "custom":
        obj_ds_name = "custom.panda3d"
    elif "modelnet." in ds_name:
        category = ds_name.split(".")[1]
        obj_ds_name = f"modelnet.{category}.test.rescaled"
    else:
        raise ValueError("Unknown dataset")

    return urdf_ds_name, obj_ds_name


def get_object_label(ds_name, description):
    df = None
    if ds_name == "ycbv":
        df = YCBV_OBJECT_NAMES
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    x = df[df.description == description]
    if len(x) == 0:
        raise ValueError(f"Couldn't find object '{description}' in ds {ds_name}")

    return x.iloc[0].label
