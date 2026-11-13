# DBS_ComfyProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Step 2 of the pipeline: plate image -> MoGe geometry (float EXR + JSON + NPZ). No Nuke dependency.

python -m geoproj.estimate <plate.png> <out_base> [--server 127.0.0.1:8188] [--fov-x 0]
Each run goes to a new <out_base>/vNNN folder.
"""

import argparse
import json
import os

from .comfy_client import ComfyClient, ComfyError, next_version_dir, wait_for
from .labels import DEFAULT_PRESET, load_preset, parse_labels  # noqa: F401 (re-exported)

WORKFLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows", "moge_geometry_api.json")
LOAD_ID, MODEL_ID, INFER_ID, SAVE_ID = "1", "2", "3", "4"

# Text prompt -> role in the fit: see labels.py and label_presets/.
DEFAULT_LABELS = load_preset(DEFAULT_PRESET)


SEG_ID = "20"


def add_segmentation(wf, labels, threshold=0.35):
    """Append SegmentLabels (SAM3, one model load + one image encode for all labels)."""
    wf[SEG_ID] = {"class_type": "SegmentLabels", "inputs": {
        "image": [LOAD_ID, 0], "labels": ", ".join(labels), "confidence_threshold": threshold,
        "filename_prefix": "geo_projection/seg"}}


SCALE_ID = "30"
SHARP_LOAD_ID, SPLAT_ID = "40", "41"


def add_splat(wf):
    """Append SHARP (comfyui-sharp's model loader + our SharpSplat at MoGe's focal)."""
    wf[SHARP_LOAD_ID] = {"class_type": "LoadSharpModel", "inputs": {"precision": "auto", "checkpoint_path": ""}}
    wf[SPLAT_ID] = {"class_type": "SharpSplat", "inputs": {
        "sharp_model": [SHARP_LOAD_ID, 0], "image": [LOAD_ID, 0], "moge_geometry": [INFER_ID, 0],
        "filename_prefix": "geo_projection/splat"}}


def estimate_geometry(image_path, out_dir, server="127.0.0.1:8188", fov_x_degrees=0.0,
                      resolution_level=9, model_name=None, labels=None, is_cancelled=lambda: False,
                      log=print, scale_frames=None, cameras=None, splat=False):
    """Run MoGe (and SAM3 per label, if labels) on one image. Files go to out_dir:
    geo.{exr,json,npz}, seg_<label>.npz and labels.json ({label: role}).
    Returns {"dir", "exr", "json", "npz", "info", "labels"} (+ "scale", "splat").

    fov_x_degrees > 0 fixes the horizontal FOV (known lens); 0 lets MoGe estimate it.
    scale_frames [(frame, image path)] + cameras (SolveTrackScale's JSON, as a dict): also solve
    the scale between MoGe and the tracked camera -> scale.json, result["scale"].
    splat: also a SHARP Gaussian splat (camera space, MoGe's focal) -> sharp.ply, result["splat"].
    """
    client = ComfyClient(server)
    client.check()
    with open(WORKFLOW) as f:
        wf = json.load(f)
    if labels:
        add_segmentation(wf, labels)
    missing = client.missing_nodes(n["class_type"] for n in wf.values())
    if missing:
        raise ComfyError("%s is missing nodes: %s\nInstall comfy_nodes/nuke_geo_projection and update ComfyUI."
                         % (client.label, ", ".join(missing)))
    if labels and client.missing_nodes(["SAM3Segment"]):
        raise ComfyError("%s has no SAM3Segment: install comfyui-rmbg (it runs SAM3 for the object "
                         "masks) and restart ComfyUI. Or turn 'semantic segmentation' off." % client.label)
    if splat:
        add_splat(wf)
        missing = client.missing_nodes(["LoadSharpModel", "SharpPredict", "SharpSplat"])
        if missing:
            raise ComfyError("%s is missing %s for the Gaussian splat: install comfyui-sharp (SHARP) and "
                             "update comfy_nodes/nuke_geo_projection, then restart ComfyUI. Or turn "
                             "'Gaussian splat' off." % (client.label, ", ".join(missing)))

    if model_name:
        wf[MODEL_ID]["inputs"]["model_name"] = model_name
    available = client.list_models("geometry_estimation")
    if available is not None and wf[MODEL_ID]["inputs"]["model_name"] not in available:
        raise ComfyError("%s has no %s in models/geometry_estimation (found: %s)."
                         % (client.label, wf[MODEL_ID]["inputs"]["model_name"], ", ".join(available) or "none"))

    wf[LOAD_ID]["inputs"]["image"] = client.upload_image(image_path)
    if scale_frames:
        wf[SCALE_ID] = {"class_type": "SolveTrackScale", "inputs": {
            "moge_geometry": [INFER_ID, 0], "image": [LOAD_ID, 0],
            "frames": ",".join("%d:%s" % (f, client.upload_image(p)) for f, p in scale_frames),
            "cameras": json.dumps(cameras), "filename_prefix": "geo_projection/scale"}}
        missing = client.missing_nodes(["SolveTrackScale"])
        if missing:
            raise ComfyError("%s is missing SolveTrackScale: update comfy_nodes/nuke_geo_projection and "
                             "restart ComfyUI." % client.label)
    wf[INFER_ID]["inputs"]["fov_x_degrees"] = float(fov_x_degrees)
    wf[INFER_ID]["inputs"]["resolution_level"] = int(resolution_level)

    log("queue on %s" % client.label)
    prompt_id = client.queue(wf)
    hist = wait_for(client, prompt_id, is_cancelled=is_cancelled)
    files = hist.get("outputs", {}).get(SAVE_ID, {}).get("geo") or []
    if not files:
        raise ComfyError("The workflow finished but SaveMoGeGeometry returned no files.")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    result = {"dir": out_dir}
    for item in files:
        ext = item["filename"].rsplit(".", 1)[-1]
        dest = os.path.join(out_dir, "geo." + ext).replace("\\", "/")
        client.download(item, dest)
        result[ext] = dest
    with open(result["json"]) as f:
        result["info"] = json.load(f)

    result["labels"] = {}
    if labels:
        seg = hist.get("outputs", {}).get(SEG_ID, {}).get("geo") or []
        if not seg:
            raise ComfyError("The workflow finished but SegmentLabels returned no file.")
        result["seg"] = client.download(seg[0], os.path.join(out_dir, "seg.npz").replace("\\", "/"))
        with open(os.path.join(out_dir, "labels.json"), "w") as f:
            json.dump(labels, f, indent=2)
        result["labels"] = dict(labels)
    if scale_frames:
        sc = hist.get("outputs", {}).get(SCALE_ID, {}).get("geo") or []
        if sc:
            with open(client.download(sc[0], os.path.join(out_dir, "scale.json").replace("\\", "/"))) as f:
                result["scale"] = json.load(f)
    if splat:
        sp = hist.get("outputs", {}).get(SPLAT_ID, {}).get("geo") or []
        if not sp:
            raise ComfyError("The workflow finished but SharpSplat returned no file.")
        result["splat"] = client.download(sp[0], os.path.join(out_dir, "sharp.ply").replace("\\", "/"))
    log("geometry in %s" % out_dir)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("out_base")
    ap.add_argument("--server", default=os.environ.get("GEO_PROJECTION_SERVER", "127.0.0.1:8188"))
    ap.add_argument("--fov-x", type=float, default=0.0)
    ap.add_argument("--resolution-level", type=int, default=9)
    ap.add_argument("--labels", default=None,
                    help="'floor:floor, table:object, ...'; default: DEFAULT_LABELS; 'none' = no SAM3")
    args = ap.parse_args(argv)
    labels = DEFAULT_LABELS if args.labels is None else ({} if args.labels == "none" else parse_labels(args.labels))
    r = estimate_geometry(args.image, next_version_dir(args.out_base), args.server, args.fov_x,
                          args.resolution_level, labels=labels)
    print(json.dumps({k: v for k, v in r.items() if k != "info"}, indent=2))
    print(json.dumps(r["info"], indent=2))


if __name__ == "__main__":
    main()
