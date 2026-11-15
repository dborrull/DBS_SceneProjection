# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Generated meshes for labels that ask for them ('sofa:object:mesh'), no Nuke dependency.

Each SAM3 instance of such a label is cropped on the ComfyUI server (ObjectMeshCrop) and turned
into a mesh by Pixal3D (ComfyUI's own nodes, geometry only: the projection is the texture).
The GLBs go to <run>/meshes/inst_<instance>.glb; Blender places them (blender/meshfit.py).
A mesh is reused from an earlier version of the same node when the plate and the instance's
mask are the same (<run>/meshes/inst_<i>.key)."""

import ast
import glob
import hashlib
import os
import shutil
import struct
import zipfile

from .comfy_client import ComfyError, wait_for

MESH = "mesh"
MIN_SCORE = 0.40          # as geofit.MIN_OBJ_SCORE: weaker masks get no object at all
MAX_MESHES = 8            # about a minute each on an 8 GB card
MODEL = "pixal3d_int8_convrot.safetensors"
FILES = {"diffusion_models": MODEL, "clip_vision": "dino_v3_L_naf_fp32.safetensors",
         "vae": "trellis_2_shape_vae_bf16.safetensors"}
NODES = ("ObjectMeshCrop", "Pixal3DConditioning", "EmptyTrellis2LatentStructure", "VaeDecodeStructureTrellis2",
         "Trellis2ShapeStage", "Trellis2UpsampleStage", "VaeDecodeShapeTrellis", "CFGOverride", "SaveGLB")
UPSAMPLE = 1024           # shape voxels; the template's 1536 is more detail than a projection needs
KEY_VERSION = "pixal3d-1"


def mesh_labels(labels):
    return [k for k, v in (labels or {}).items() if v == "object:" + MESH]


# ---- seg.npz without numpy (Nuke has none) ------------------------------------------------

def _npy_header(f):
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ComfyError("seg.npz: not a numpy file")
    major = f.read(2)[0]
    n = struct.unpack("<H" if major == 1 else "<I", f.read(2 if major == 1 else 4))[0]
    return ast.literal_eval(f.read(n).decode("latin1"))


def _read_small(z, name):
    """A 1-D array of an .npz as a list (ints, floats or str)."""
    with z.open(name + ".npy") as f:
        h = _npy_header(f)
        data = f.read()
    dt, n = h["descr"], h["shape"][0] if h["shape"] else 1
    if dt[1] == "U":                                   # fixed-width UTF-32 strings
        w = int(dt[2:]) * 4
        return [data[i * w:(i + 1) * w].decode("utf-32-le" if dt[0] in "<|" else "utf-32-be").rstrip("\x00")
                for i in range(n)]
    code = {"i4": "i", "i8": "q", "f4": "f", "f8": "d", "b1": "?"}[dt[1:]]
    return list(struct.unpack(("<" if dt[0] in "<|" else ">") + code * n, data[:n * struct.calcsize(code)]))


def mesh_instances(seg_path, labels):
    """Instances of SegmentLabels' file to generate: labels whose shape is 'mesh', confident
    enough, best first, at most MAX_MESHES. -> [(instance, label)]."""
    wanted = set(mesh_labels(labels))
    if not wanted:
        return []
    with zipfile.ZipFile(seg_path) as z:
        names = _read_small(z, "labels")
        ml = _read_small(z, "mask_label")
        sc = _read_small(z, "scores")
    cands = [(sc[i], i, names[ml[i]]) for i in range(len(ml)) if names[ml[i]] in wanted and sc[i] >= MIN_SCORE]
    cands.sort(reverse=True)
    return [(i, lab) for _, i, lab in cands[:MAX_MESHES]]


def mask_hashes(seg_path, instances):
    """sha1 of each instance's mask bytes (the zip's timestamps change between runs, the
    masks do not)."""
    out = {}
    want = set(instances)
    with zipfile.ZipFile(seg_path) as z, z.open("masks.npy") as f:
        h = _npy_header(f)
        if h.get("fortran_order"):
            raise ComfyError("seg.npz: unexpected mask layout")
        n, hh, ww = h["shape"]
        size = hh * ww
        for i in range(n):
            buf = f.read(size)
            if i in want:
                out[i] = hashlib.sha1(buf).hexdigest()
            if len(out) == len(want):
                break
    return out


def _file_sha1(path):
    s = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            s.update(chunk)
    return s.hexdigest()


# ---- ComfyUI ------------------------------------------------------------------------------

def check(client):
    """Raise with what is missing on the server for Pixal3D meshes."""
    missing = client.missing_nodes(NODES)
    if missing:
        raise ComfyError("%s is missing %s for 'mesh' labels: update ComfyUI (Pixal3D / TRELLIS.2 nodes are "
                         "built in from late 2026) and comfy_nodes/nuke_geo_projection, then restart it."
                         % (client.label, ", ".join(missing)))
    for folder, name in FILES.items():
        have = client.list_models(folder)
        if have is not None and name not in have:
            raise ComfyError("%s has no %s in models/%s, needed for 'mesh' labels (Comfy-Org/Pixal3D on "
                             "Hugging Face, about 7.9 GB for the three files)." % (client.label, name, folder))


def add_branch(wf, n, image_id, seg_file, instance, fov_x, prefix):
    """Nodes for one instance: crop -> Pixal3D structure -> shape -> upsample -> GLB.
    Loaders are shared ("m1" model, "m2" DINO, "m3" VAE); sampler settings as ComfyUI's
    Pixal3D template."""
    k = "mesh%d_" % n
    wf[k + "crop"] = {"class_type": "ObjectMeshCrop", "inputs": {
        "image": [image_id, 0], "seg_file": seg_file, "instance": int(instance), "fov_x": float(fov_x),
        "pad": 1.1, "size": 1024}}
    wf[k + "cond"] = {"class_type": "Pixal3DConditioning", "inputs": {
        "clip_vision_model": ["m2", 0], "image": [k + "crop", 0], "camera_angle_x": [k + "crop", 1]}}
    wf[k + "lat"] = {"class_type": "EmptyTrellis2LatentStructure", "inputs": {"batch_size": 1}}
    wf[k + "s1"] = {"class_type": "KSampler", "inputs": {
        "model": ["mA", 0], "seed": 56, "steps": 12, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal",
        "positive": [k + "cond", 0], "negative": [k + "cond", 1], "latent_image": [k + "lat", 0], "denoise": 1}}
    wf[k + "vox"] = {"class_type": "VaeDecodeStructureTrellis2", "inputs": {
        "samples": [k + "s1", 0], "vae": ["m3", 0], "resolution": "32"}}
    wf[k + "shape"] = {"class_type": "Trellis2ShapeStage", "inputs": {
        "positive": [k + "cond", 0], "negative": [k + "cond", 1], "voxel": [k + "vox", 0]}}
    wf[k + "s2"] = {"class_type": "KSampler", "inputs": {
        "model": ["mB", 0], "seed": 42, "steps": 20, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal",
        "positive": [k + "shape", 0], "negative": [k + "shape", 1], "latent_image": [k + "shape", 2], "denoise": 1}}
    wf[k + "up"] = {"class_type": "Trellis2UpsampleStage", "inputs": {
        "positive": [k + "cond", 0], "negative": [k + "cond", 1], "shape_latent": [k + "s2", 0],
        "vae": ["m3", 0], "target_resolution": UPSAMPLE}}
    wf[k + "s3"] = {"class_type": "KSampler", "inputs": {
        "model": ["mB", 0], "seed": 42, "steps": 12, "cfg": 7.5, "sampler_name": "euler", "scheduler": "simple",
        "positive": [k + "up", 0], "negative": [k + "up", 1], "latent_image": [k + "up", 2], "denoise": 1}}
    wf[k + "dec"] = {"class_type": "VaeDecodeShapeTrellis", "inputs": {"samples": [k + "s3", 0], "vae": ["m3", 0]}}
    # Saved as decoded (1-6 M triangles): ComfyUI's DecimateMesh takes ~8 min per mesh on these;
    # Blender thins them in seconds (build_scene.load_generated).
    wf[k + "save"] = {"class_type": "SaveGLB", "inputs": {"mesh": [k + "dec", 0], "filename_prefix": prefix}}
    return k + "save"


def loaders(image_name):
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "m1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "m2": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": FILES["clip_vision"]}},
        "m3": {"class_type": "VAELoader", "inputs": {"vae_name": FILES["vae"]}},
        # structure stage / shape stages: guidance as the template (CFG override + rescale)
        "mA0": {"class_type": "CFGOverride", "inputs": {"model": ["m1", 0], "cfg": 1, "start_percent": 0.667,
                                                        "end_percent": 1}},
        "mA1": {"class_type": "RescaleCFG", "inputs": {"model": ["mA0", 0], "multiplier": 0.7}},
        "mA": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["mA1", 0], "shift": 5}},
        "mB0": {"class_type": "CFGOverride", "inputs": {"model": ["m1", 0], "cfg": 1, "start_percent": 0.769,
                                                        "end_percent": 1}},
        "mB": {"class_type": "RescaleCFG", "inputs": {"model": ["mB0", 0], "multiplier": 0.5}},
    }


def generate(client, geo, labels, out_dir, is_cancelled=lambda: False, log=print):
    """Meshes for the 'mesh' labels of an estimate_geometry() result (needs its seg, the
    uploaded plate name and the seg file's server item). -> meshes dir, or None (no such labels
    or instances)."""
    insts = mesh_instances(geo["seg"], labels)
    if not insts:
        if mesh_labels(labels):
            log("mesh labels %s: no instance confident enough" % ", ".join(mesh_labels(labels)))
        return None
    mdir = os.path.join(out_dir, "meshes")
    os.makedirs(mdir, exist_ok=True)
    plate = _file_sha1(geo["plate"])
    hashes = mask_hashes(geo["seg"], [i for i, _ in insts])
    keys = {i: hashlib.sha1(("%s|%s|%s|%s|%d" % (KEY_VERSION, MODEL, plate, hashes[i], UPSAMPLE)).encode()).hexdigest()
            for i, _ in insts}
    # Reuse: same key in an earlier version of this node (sibling vNNN folders).
    todo = []
    for i, lab in insts:
        dest = os.path.join(mdir, "inst_%d.glb" % i)
        hit = None
        for kf in glob.glob(os.path.join(os.path.dirname(os.path.abspath(out_dir)), "v*", "meshes", "inst_*.key")):
            try:
                with open(kf) as f:
                    if f.read().strip() == keys[i] and os.path.isfile(kf[:-4] + ".glb"):
                        hit = kf[:-4] + ".glb"
                        break
            except OSError:
                continue
        if hit:
            shutil.copyfile(hit, dest)
            log("mesh %s (instance %d): reused %s" % (lab, i, hit))
        else:
            todo.append((i, lab))
        with open(dest[:-4] + ".key", "w") as f:
            f.write(keys[i])
    if todo:
        check(client)
        item = geo["seg_item"]
        seg_file = (item.get("subfolder") + "/" if item.get("subfolder") else "") + item["filename"]
        wf = loaders(geo["comfy_image"])
        saves = {}
        for n, (i, lab) in enumerate(todo):
            saves[add_branch(wf, n, "img", seg_file, i, geo["info"]["fov_x_deg"], "geo_projection/mesh")] = (i, lab)
        log("Pixal3D: %d mesh(es) for %s (about a minute each)" % (len(todo), ", ".join(l for _, l in todo)))
        hist = wait_for(client, client.queue(wf), is_cancelled=is_cancelled)
        for sid, (i, lab) in saves.items():
            files = hist.get("outputs", {}).get(sid, {})
            item = next((x for v in files.values() if isinstance(v, list) for x in v
                         if isinstance(x, dict) and str(x.get("filename", "")).lower().endswith(".glb")), None)
            dest = os.path.join(mdir, "inst_%d.glb" % i)
            if item is None:
                os.remove(dest[:-4] + ".key")
                log("mesh %s (instance %d): Pixal3D returned no GLB" % (lab, i))
                continue
            client.download(item, dest)
    return mdir.replace("\\", "/")
