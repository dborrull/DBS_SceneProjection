"""Offline check of the SolveTrackScale ComfyUI node on a synthetic sequence (ComfyUI's Python):

    C:/ComfyUI/python_embeded/python.exe tests/solve_scale_offline.py <run dir> <seq dir> <frame> [...]

<run dir>: a DBS_ComfyProjection run on frame 1 of the sequence (geo.npz, plate.0001.png, fit.json);
<seq dir>: make_synthetic_plate.py SEQ_FRAMES output (plate.%04d.png, ground_truth.json with
camera_frames). Prints the solved scale, its error and the true one (MoGe camera height vs
the ground truth's), so the scale solve can be checked without Nuke.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, r"C:\ComfyUI\ComfyUI")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comfy_nodes"))
from nuke_geo_projection import SolveTrackScale  # noqa: E402
from PIL import Image  # noqa: E402

run, seq, frames = sys.argv[1], sys.argv[2], [int(f) for f in sys.argv[3:]]
gt = json.load(open(os.path.join(seq, "ground_truth.json")))
C = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)          # Blender Z up -> Nuke Y up


def nuke_matrix(mb):
    m = np.eye(4)
    mb = np.array(mb, float)
    m[:3, :3] = C @ mb[:3, :3]
    m[:3, 3] = C @ mb[:3, 3]
    return m.ravel().tolist()


cams = {"ref": 1, "frames": {str(f): {"matrix": nuke_matrix(gt["camera_frames"][str(f)]),
                                      "focal": gt["focal_mm"], "haperture": gt["sensor_width_mm"]}
                             for f in [1] + frames}}
geo = np.load(os.path.join(run, "geo.npz"))
ref = np.array(Image.open(os.path.join(run, "plate.0001.png")).convert("RGB"), np.float32) / 255
import shutil  # noqa: E402
import folder_paths  # noqa: E402
names = {}
for f in frames:                                   # the node reads frames from ComfyUI's input folder
    os.makedirs(os.path.join(folder_paths.get_input_directory(), "geo_projection_test"), exist_ok=True)
    names[f] = "geo_projection_test/plate.%04d.png" % f
    shutil.copy(os.path.join(seq, "plate.%04d.png" % f), os.path.join(folder_paths.get_input_directory(), names[f]))
out = SolveTrackScale().solve({"points": torch.from_numpy(geo["points"][None])}, torch.from_numpy(ref[None]),
                              ",".join("%d:%s" % (f, names[f]) for f in frames),
                              json.dumps(cams), "geo_projection/test_scale")
d = out["ui"]["geo"][0]
res = json.load(open(os.path.join(folder_paths.get_output_directory(), d["subfolder"], d["filename"])))
fit = json.load(open(os.path.join(run, "fit.json")))
moge_h = fit["camera"]["location"][2] / (fit.get("units_scale") or 1.0)
print(json.dumps(res, indent=1))
print("true scale (GT camera height / MoGe's): %.4f" % (gt["camera_location"][2] / moge_h))
