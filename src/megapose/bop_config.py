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

# Cosypose models


BOP_CONFIG = dict()
BOP_CONFIG["hb"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="hb",
    obj_ds_name="hb",
    train_pbr_ds_name=["hb.pbr"],
    inference_ds_name=["hb.bop19"],
    test_ds_name=[],
)

BOP_CONFIG["icbin"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="icbin",
    obj_ds_name="icbin",
    train_pbr_ds_name=["icbin.pbr"],
    inference_ds_name=["icbin.bop19"],
    test_ds_name=["icbin.bop19"],
)


BOP_CONFIG["itodd"] = dict(
    input_resize=(1280, 960),
    urdf_ds_name="itodd",
    obj_ds_name="itodd",
    train_pbr_ds_name=["itodd.pbr"],
    inference_ds_name=["itodd.bop19"],
    test_ds_name=[],
    val_ds_name=["itodd.val"],
)


BOP_CONFIG["lmo"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="lm",
    obj_ds_name="lm",
    train_pbr_ds_name=["lm.pbr"],
    train_synt_real_ds_names=[
        ("lm.pbr", 1),
    ],
    inference_ds_name=["lmo.bop19"],
    test_ds_name=["lmo.bop19"],
)

BOP_CONFIG["lm"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="lm",
    obj_ds_name="lm",
    train_pbr_ds_name=["lm.pbr"],
    train_synt_real_ds_names=[
        ("lm.pbr", 1),
    ],
)


BOP_CONFIG["tless"] = dict(
    input_resize=(720, 540),
    urdf_ds_name="tless.cad",
    obj_ds_name="tless.cad",
    train_pbr_ds_name=["tless.pbr"],
    inference_ds_name=["tless.bop19"],
    test_ds_name=["tless.bop19"],
    train_synt_real_ds_names=[("tless.pbr", 4), ("tless.primesense.train", 1)],
    train_opengl_ds_names=[("tless.opengl", 1)],
    train_mysynt_ds_names=[("synthetic.tless-1M.train", 1)],
)

BOP_CONFIG["tudl"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="tudl",
    obj_ds_name="tudl",
    train_pbr_ds_name=["tudl.pbr"],
    inference_ds_name=["tudl.bop19"],
    test_ds_name=["tudl.bop19"],
    train_synt_real_ds_names=[("tudl.pbr", 10), ("tudl.train.real", 1)],
    train_opengl_ds_names=[("tudl.opengl", 1)],
    train_mysynt_ds_names=[("synthetic.tudl-1M.train", 1)],
)


BOP_CONFIG["ycbv"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="ycbv",
    obj_ds_name="ycbv",
    train_pbr_ds_name=["ycbv.pbr"],
    train_pbr_real_ds_names=[("ycbv.pbr", 1), ()],
    inference_ds_name=["ycbv.bop19"],
    test_ds_name=["ycbv.bop19"],
    train_synt_real_ds_names=[("ycbv.pbr", 20), ("ycbv.train.synt", 1), ("ycbv.train.real", 3)],
    train_opengl_ds_names=[("ycbv.opengl", 1)],
    train_mysynt_ds_names=[("synthetic.ycbv-1M.train", 1)],
)

BOP_CONFIG["ruapc"] = dict(
    # TODO: input resize
    input_resize=(640, 480),
    urdf_ds_name="ruapc",
    obj_ds_name="ruapc",
    train_pbr_ds_name=[],
    train_pbr_real_ds_names=[],
    inference_ds_name=["ruapc.bop19"],
    test_ds_name=["ruapc.bop19"],
)

BOP_CONFIG["tyol"] = dict(
    # TODO: input resize
    input_resize=(640, 480),
    urdf_ds_name="tyol",
    obj_ds_name="tyol",
    train_pbr_ds_name=[],
    train_pbr_real_ds_names=[],
    inference_ds_name=["tyol.bop19"],
    test_ds_name=["tyol.bop19"],
)

BOP_CONFIG["moped"] = dict(
    input_resize=(640, 480),
    urdf_ds_name="moped",
    obj_ds_name="moped",
    train_pbr_ds_name=[],
    train_pbr_real_ds_names=[],
    inference_ds_name=["moped"],
    test_ds_name=["moped"],
)

for k, v in BOP_CONFIG.items():
    v["panda3d_obj_ds_name"] = v["obj_ds_name"] + ".panda3d"

PBR_DETECTORS = dict(
    hb="detector-bop-hb-pbr--497808",
    icbin="detector-bop-icbin-pbr--947409",
    itodd="detector-bop-itodd-pbr--509908",
    lmo="detector-bop-lmo-pbr--517542",
    tless="detector-bop-tless-pbr--873074",
    tudl="detector-bop-tudl-pbr--728047",
    ycbv="detector-bop-ycbv-pbr--970850",
    hope="detector-bop-hope-pbr--15246",
)

PBR_COARSE = dict(
    hb="coarse-bop-hb-pbr--70752",
    icbin="coarse-bop-icbin-pbr--915044",
    itodd="coarse-bop-itodd-pbr--681884",
    lmo="coarse-bop-lmo-pbr--707448",
    tless="coarse-bop-tless-pbr--506801",
    tudl="coarse-bop-tudl-pbr--373484",
    ycbv="coarse-bop-ycbv-pbr--724183",
    hope="bop-hope-pbr-coarse-transnoise-zxyavg-225203",
)

PBR_REFINER = dict(
    hb="refiner-bop-hb-pbr--247731",
    icbin="refiner-bop-icbin-pbr--841882",
    itodd="refiner-bop-itodd-pbr--834427",
    lmo="refiner-bop-lmo-pbr--325214",
    tless="refiner-bop-tless-pbr--233420",
    tudl="refiner-bop-tudl-pbr--487212",
    ycbv="refiner-bop-ycbv-pbr--604090",
    hope="bop-hope-pbr-refiner--955392",
)

SYNT_REAL_DETECTORS = dict(
    tudl="detector-bop-tudl-synt+real--298779",
    tless="detector-bop-tless-synt+real--452847",
    ycbv="detector-bop-ycbv-synt+real--292971",
)

SYNT_REAL_COARSE = dict(
    tudl="coarse-bop-tudl-synt+real--610074",
    tless="coarse-bop-tless-synt+real--160982",
    ycbv="coarse-bop-ycbv-synt+real--822463",
)

SYNT_REAL_REFINER = dict(
    tudl="refiner-bop-tudl-synt+real--423239",
    tless="refiner-bop-tless-synt+real--881314",
    ycbv="refiner-bop-ycbv-synt+real--631598",
)

for k, v in PBR_COARSE.items():
    if k not in SYNT_REAL_COARSE:
        SYNT_REAL_COARSE[k] = v

for k, v in PBR_REFINER.items():
    if k not in SYNT_REAL_REFINER:
        SYNT_REAL_REFINER[k] = v

for k, v in PBR_DETECTORS.items():
    if k not in SYNT_REAL_DETECTORS:
        SYNT_REAL_DETECTORS[k] = v


PBR_INFERENCE_ID = "bop-pbr--223026"
SYNT_REAL_INFERENCE_ID = "bop-synt+real--815712"
SYNT_REAL_ICP_INFERENCE_ID = "bop-synt+real-icp--121351"
