# GeoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Compare MoGe 2 with Depth Anything 3 (mono large, metric large, base) on one plate.

Blender's Python (numpy + OpenImageIO):
    python tests/compare_depth_models.py <plate.png> <out_dir> [--gt <synthetic_dir>] [--res 1008]

Runs all models in one ComfyUI job, downloads their float outputs and reports focal, absolute
scale, shape (abs-rel after a median scale) and edges (abs-rel in a band around depth jumps).
With --gt (a make_synthetic_plate.py folder) errors are against ground truth; without it the
DA3 models are compared to MoGe. Also writes normals_<model>.png (normals from each depth with
one shared focal) to look at edges.
"""

import argparse
import json
import os
import sys

import numpy as np
import OpenImageIO as oiio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nuke"))
from geoproj.comfy_client import ComfyClient, wait_for  # noqa: E402

DA3 = {"da3_mono": "depth_anything_3_mono_large.safetensors",
       "da3_metric": "depth_anything_3_metric_large.safetensors",
       "da3_base": "depth_anything_3_base.safetensors"}
EDGE_JUMP = 0.05       # relative depth step that counts as an edge
EDGE_BAND = 2          # samples each side of an edge


def workflow(image_name, res):
    wf = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_2_vitl_normal_fp16.safetensors"}},
        "3": {"class_type": "MoGeInference", "inputs": {
            "moge_model": ["2", 0], "image": ["1", 0], "resolution_level": 9, "fov_x_degrees": 0.0,
            "batch_size": 1, "force_projection": True, "apply_mask": True, "refine_steps": 3}},
        "4": {"class_type": "SaveMoGeGeometry", "inputs": {"moge_geometry": ["3", 0],
                                                            "filename_prefix": "geo_projection/cmp_moge"}},
    }
    saves = {"4": "moge"}
    for k, (key, model) in enumerate(DA3.items()):
        l, i, s = str(10 + 3 * k), str(11 + 3 * k), str(12 + 3 * k)
        wf[l] = {"class_type": "LoadDA3Model", "inputs": {"model_name": model, "weight_dtype": "default"}}
        inputs = {"da3_model": [l, 0], "image": ["1", 0], "resolution": res,
                  "resize_method": "upper_bound_resize"}
        if key == "da3_base":   # one view: only ray_pose gives intrinsics (cam_dec needs 2+ views)
            inputs.update({"mode": "multiview", "mode.ref_view_strategy": "first", "mode.pose_method": "ray_pose"})
        else:
            inputs["mode"] = "mono"
        wf[i] = {"class_type": "DA3Inference", "inputs": inputs}
        wf[s] = {"class_type": "SaveDA3Geometry", "inputs": {"da3_geometry": [i, 0],
                                                             "filename_prefix": "geo_projection/cmp_" + key}}
        saves[s] = key
    return wf, saves


def run(plate, out_dir, res, server="127.0.0.1:8188"):
    c = ComfyClient(server)
    wf, saves = workflow(c.upload_image(plate), res)
    hist = wait_for(c, c.queue(wf))
    files = {}
    for nid, key in saves.items():
        for item in hist["outputs"][nid]["geo"]:
            ext = item["filename"].rsplit(".", 1)[-1]
            files[(key, ext)] = c.download(item, os.path.join(out_dir, "%s.%s" % (key, ext)))
    return files


def normals(depth, fx_norm, valid):
    """Camera-space normals from a depth map and a normalized focal (fx / width)."""
    H, W = depth.shape
    f = fx_norm * W
    u, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    p = np.stack([u * depth / f, v * depth / f, depth], -1)
    dx, dy = np.gradient(p, axis=1), np.gradient(p, axis=0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    img = (0.5 - 0.5 * n) * valid[..., None]
    return img.astype(np.float32)


def save_png(path, img):
    o = oiio.ImageOutput.create(path)
    o.open(path, oiio.ImageSpec(img.shape[1], img.shape[0], 3, oiio.UINT8))
    o.write_image(np.clip(img, 0, 1).astype(np.float32))
    o.close()


def edge_band(ref, valid, step):
    """Samples within EDGE_BAND of a relative depth jump > EDGE_JUMP in the reference."""
    j = np.zeros_like(valid)
    jh = np.abs(np.diff(ref, axis=1)) > EDGE_JUMP * np.minimum(ref[:, 1:], ref[:, :-1])
    jv = np.abs(np.diff(ref, axis=0)) > EDGE_JUMP * np.minimum(ref[1:], ref[:-1])
    j[:, 1:] |= jh
    j[:, :-1] |= jh
    j[1:] |= jv
    j[:-1] |= jv
    band = j.copy()
    for _ in range(EDGE_BAND):
        b = band.copy()
        b[1:] |= band[:-1]
        b[:-1] |= band[1:]
        b[:, 1:] |= band[:, :-1]
        b[:, :-1] |= band[:, 1:]
        band = b
    return band & valid


def affine_to(pred, target, m):
    """Least-squares a * pred + b ~= target over m (DA3 mono / metric output carries an unknown
    scale *and* shift)."""
    A = np.stack([pred[m], np.ones(m.sum())], 1)
    a, b = np.linalg.lstsq(A, target[m], rcond=None)[0]
    return a * pred + b


def stats(pred, ref, valid, band, moge=None):
    """Errors against ref: scale-only alignment, affine alignment (both need ref, so only
    possible with ground truth) and, given moge, 'fused' = affine-aligned to MoGe instead -
    what a production run can actually do."""
    m = valid & (pred > 0) & np.isfinite(pred)
    out = {}
    variants = {"scale": pred * np.median(ref[m] / pred[m]), "affine": affine_to(pred, ref, m)}
    if moge is not None:
        # Shape of the fused depth: MoGe's absolute scale is a separate (known bad) question,
        # so this still gets a median scale to ref like the "scale" column.
        f = affine_to(pred, moge, m)
        variants["fused_to_moge"] = f * np.median(ref[m] / f[m])
    for name, p in variants.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(p - ref) / ref
        out[name] = {"abs_rel_median": round(float(np.median(rel[m & ~band])), 5),
                     "edge_abs_rel_median": round(float(np.median(rel[m & band])), 5) if (m & band).any() else None,
                     "edge_abs_rel_p90": round(float(np.percentile(rel[m & band], 90)), 5) if (m & band).any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plate")
    ap.add_argument("out_dir")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--res", type=int, default=1008)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    run(a.plate, a.out_dir, a.res)

    moge = np.load(os.path.join(a.out_dir, "moge.npz"))
    minfo = json.load(open(os.path.join(a.out_dir, "moge.json")))
    H, W = moge["depth"].shape
    depth = {"moge": moge["depth"].astype(np.float64)}
    focal = {"moge": minfo["intrinsics_normalized"]["fx"]}
    sky = {}
    for key in DA3:
        d = np.load(os.path.join(a.out_dir, key + ".npz"))
        depth[key] = d["depth"][0].astype(np.float64)
        if "sky" in d:
            sky[key] = d["sky"][0]
        if "intrinsics" in d and d["intrinsics"].size:
            K = d["intrinsics"][0, 0]
            if abs(K[0, 0] - 1) > 1e-6:          # identity = no estimate
                focal[key] = float(K[0, 0] / (2 * K[0, 2]))   # fx / processed width (cx ~ W/2)

    gt = None
    if a.gt:
        g = json.load(open(os.path.join(a.gt, "ground_truth.json")))
        step_file = next(f for f in os.listdir(a.gt) if f.startswith("gt_depth_step"))
        step = int(step_file[len("gt_depth_step"):-4])
        gt = np.load(os.path.join(a.gt, step_file)).astype(np.float64)
        focal["ground truth"] = g["focal_pixels"] / g["width"]
    else:
        step = 2
    s = slice(step // 2, None, step)
    sub = {k: v[s, s][:(gt.shape[0] if gt is not None else None), :(gt.shape[1] if gt is not None else None)]
           for k, v in depth.items()}
    valid = moge["mask"][s, s][:sub["moge"].shape[0], :sub["moge"].shape[1]].astype(bool)
    if gt is not None:
        valid &= gt > 0
    for key, sk in sky.items():
        valid &= sk[s, s][:valid.shape[0], :valid.shape[1]] < 0.5

    # DA3 metric: canonical output -> metres with the focal in model-input pixels (upstream:
    # metric = focal * output / 300). The model ran with the long side at --res.
    proc_w = a.res if W >= H else a.res * W / H
    f_for_metric = focal.get("ground truth", focal["moge"])
    sub["da3_metric_m"] = sub["da3_metric"] * f_for_metric * proc_w / 300.0

    ref = gt if gt is not None else sub["moge"]
    band = edge_band(ref, valid, step)
    report = {"focal_fx_over_width": focal, "reference": "ground truth" if gt is not None else "moge",
              "edge_band_fraction": float(band.mean()), "models": {}}
    for key in ("moge", "da3_mono", "da3_metric", "da3_base"):
        if gt is None and key == "moge":
            continue
        report["models"][key] = stats(sub[key], ref, valid, band,
                                      sub["moge"] if (key != "moge" and gt is not None) else None)
    m = valid & (sub["da3_metric_m"] > 0)
    report["da3_metric_absolute_scale_ref_over_pred"] = float(np.median(ref[m] / sub["da3_metric_m"][m]))
    m = valid & (sub["moge"] > 0)
    report["moge_absolute_scale_ref_over_pred"] = float(np.median(ref[m] / sub["moge"][m]))
    json.dump(report, open(os.path.join(a.out_dir, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))

    fx = focal.get("ground truth", focal["moge"])
    full_valid = moge["mask"].astype(bool)
    fm = full_valid & (depth["moge"] > 0)
    for key in ("moge", "da3_mono", "da3_metric", "da3_base"):
        # DA3 depth aligned to MoGe first: normals of an unaligned affine depth are meaningless.
        d = depth[key] if key == "moge" else affine_to(depth[key], depth["moge"], fm)
        save_png(os.path.join(a.out_dir, "normals_%s.png" % key), normals(d, fx, full_valid))


if __name__ == "__main__":
    main()
