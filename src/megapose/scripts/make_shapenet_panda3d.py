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
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Process
from pathlib import Path

# Third Party
from tqdm import tqdm

# MegaPose
from megapose.config import SHAPENET_DIR
from megapose.datasets.datasets_cfg import make_object_dataset


def fix_normals(obj_path):
    obj = obj_path.read_text()
    lines = obj.splitlines()
    elements = {
        "vt": [],
        "v": [],
        "vn": [],
        "blocks": [],
        "mtllib": "",
    }

    is_block = False

    def make_new_block():
        return dict(
            g="",
            usemtl="",
            f=[],
            l=[],
        )

    for line in lines:
        if line.startswith("mtllib"):
            assert is_block is False
            elements["mtllib"] = line

        elif line.startswith("v "):
            assert is_block is False
            elements["v"].append(line)

        elif line.startswith("vt "):
            assert is_block is False
            elements["vt"].append(line)

        elif line.startswith("vn "):
            assert is_block is False
            elements["vn"].append(line)

        else:
            if not is_block:
                if line.startswith("g"):
                    block = make_new_block()
                    block["g"] = line
                    is_block = True

            if is_block and len(line) <= 1 and len(elements["vn"]) > 0:
                is_block = False
                elements["blocks"].append(block)

            if is_block:
                if line.startswith("usemtl"):
                    block["usemtl"] = line
                elif line.startswith("f"):
                    block["f"].append(line)
                elif line.startswith("l"):
                    block["l"].append(line)
                else:
                    pass

    n_vn_orig = len(elements["vn"])

    vn_flipped = []
    for line_vn in elements["vn"]:
        normal = map(float, line_vn[3:].split(" "))
        normal_flipped = " ".join([f"{-x:.7f}" for x in normal])
        vn_flipped.append(f"vn {normal_flipped}")
    elements["vn"] += vn_flipped

    for block in elements["blocks"]:
        f_flipped = []
        for line_f in block["f"]:
            face = line_f[3:].split(" ")
            face = [f.split("/") for f in face]
            face_flipped = " ".join([f"{x[0]}/{x[1]}/{int(x[2])+n_vn_orig}" for x in face])
            f_flipped.append(f"f  {face_flipped}")
        block["f"] += f_flipped

    new_obj = "\n"
    new_obj += elements["mtllib"]
    new_obj += "\n"
    new_obj += "\n"
    new_obj += "\n".join(elements["v"])
    new_obj += "\n"
    new_obj += "\n"
    new_obj += "\n".join(elements["vt"])
    new_obj += "\n"
    new_obj += "\n"
    new_obj += "\n".join(elements["vn"])
    new_obj += "\n"

    for block in elements["blocks"]:
        text_block = "\n"
        assert len(block["g"]) > 0
        text_block += block["g"]
        text_block += "\n"
        text_block += block["usemtl"]
        text_block += "\n"
        if len(block["f"]) > 0:
            text_block += "\n".join(block["f"])
        if len(block["l"]) > 0:
            text_block += "\n".join(block["l"])
        text_block += "\n"
        new_obj += text_block
    return new_obj


def convert_obj_to_gltf(obj_path):
    n, obj_path = obj_path
    print(n, obj_path)
    obj_path = Path(obj_path)
    new_obj = fix_normals(obj_path)
    binormals_obj_path = Path((str(obj_path.with_suffix("")) + "_binormals.obj"))
    binormals_obj_path.write_text(new_obj)
    proc = subprocess.run(["obj2gltf", "-i", str(binormals_obj_path)])
    gltf_path = binormals_obj_path.with_suffix(".gltf")
    p = Process(target=convert_gltf, args=(gltf_path,))
    p.start()
    p.join()
    bam_path = gltf_path.with_suffix(".bam")
    bam_exists = bam_path.exists()
    if bam_exists:
        new_models_dir = Path(str(obj_path.parent).replace("models_orig", "models_panda3d_bam"))
        Path(new_models_dir).mkdir(exist_ok=True, parents=True)
        img_dir = obj_path.parent.parent / "images"
        new_img_dir = new_models_dir
        if img_dir.exists():
            for p in img_dir.iterdir():
                shutil.copy(p, new_img_dir / p.name)
        shutil.move(bam_path, new_models_dir / bam_path.name)
    binormals_obj_path.unlink()
    if gltf_path.exists():
        gltf_path.unlink()
    return


def convert_obj_to_gltf_(obj_path):
    p = Process(target=convert_obj_to_gltf, args=(obj_path,))
    p.start()
    p.join()
    return


def convert_gltf(gltf_path):
    # Third Party
    from gltf.converter import convert

    gltf_path = Path(gltf_path)
    convert(str(gltf_path), str(gltf_path.with_suffix(".bam")), None)
    return True


def convert_batch_obj_to_gltf(obj_paths):
    for obj_path in obj_paths:
        convert_obj_to_gltf(obj_path)


if __name__ == "__main__":
    obj_dataset = make_object_dataset("shapenet.orig")
    new_dir = SHAPENET_DIR / "models_panda3d_bam"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir()
    shutil.copy(SHAPENET_DIR / "models_orig/taxonomy.json", new_dir / "taxonomy.json")
    n_procs = 20
    mesh_paths = []
    for n, obj in tqdm(enumerate(obj_dataset.objects)):
        mesh_path = Path(obj["mesh_path"])
        mesh_paths.append([n, mesh_path])

    for mesh_path in tqdm(mesh_paths):
        convert_obj_to_gltf(mesh_path)

    time.sleep(60)
