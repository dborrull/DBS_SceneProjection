"""Compare a geofit result (fit.json + fit_proxy_depth.npy) with the synthetic plate's ground truth.

python tests/eval_fit.py <fit.json> <synthetic_dir>
Run geofit with --camera-height set to the ground-truth height so units are metres.
"""

import json
import math
import os
import sys

import numpy as np

fit_path, gt_dir = sys.argv[1:3]
fit = json.load(open(fit_path))
gt = json.load(open(os.path.join(gt_dir, "ground_truth.json")))
proxy = np.load(fit_path.rsplit(".", 1)[0] + "_proxy_depth.npy") * fit["units_scale"]
step = fit["verification"]["step"]
gt_z = np.load(os.path.join(gt_dir, "gt_depth_step%d.npy" % step))

ok = (gt_z > 0) & (proxy > 0)
rel = np.abs(proxy[ok] - gt_z[ok]) / gt_z[ok]
print("camera height %.3f (gt %.3f)   lens %.2f mm (gt %.2f)" % (
    fit["camera_height"], gt["camera_location"][2], fit["camera"]["lens_mm"], gt["focal_mm"]))
print("proxy depth vs gt: coverage %.1f%%  abs-rel median %.2f%%  mean %.2f%%  p90 %.2f%%" % (
    100 * ok.mean(), 100 * np.median(rel), 100 * rel.mean(), 100 * np.percentile(rel, 90)))
s = np.median(gt_z[ok] / proxy[ok])
rel_s = np.abs(proxy[ok] * s - gt_z[ok]) / gt_z[ok]
print("  shape only (x%.3f): abs-rel median %.2f%%  p90 %.2f%%" % (s, 100 * np.median(rel_s), 100 * np.percentile(rel_s, 90)))

# Match each ground-truth box to the fitted box with the nearest centre (world XY, camera foot = origin).
cam = np.array(gt["camera_location"][:2])
fwd = np.array([math.cos(gt["camera_rotation_euler_xyz_rad"][2] + math.pi / 2),
                math.sin(gt["camera_rotation_euler_xyz_rad"][2] + math.pi / 2)])
right = np.array([fwd[1], -fwd[0]])
boxes = [p for p in fit["primitives"] if p["type"] == "box"]
for g in (p for p in gt["primitives"] if p["type"] == "box"):
    rel_xy = np.array(g["location"][:2]) - cam
    g_xy = np.array([rel_xy @ right, rel_xy @ fwd])   # into the fit's frame (Y = camera forward)
    b = min(boxes, key=lambda b: np.linalg.norm(np.array(b["center"][:2]) - g_xy))
    gs, bs = sorted(g["size"][:2]), sorted(b["size"][:2])
    print("%-8s -> %-6s centre err %.2f m | footprint %.2fx%.2f (gt %.2fx%.2f) | height %.2f (gt %.2f)%s" % (
        g["name"], b["name"], np.linalg.norm(np.array(b["center"][:2]) - g_xy),
        bs[0], bs[1], gs[0], gs[1], b["size"][2], g["size"][2],
        "  [depth guessed]" if b.get("depth_guessed") else ""))
