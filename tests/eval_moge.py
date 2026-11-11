"""Compare a SaveMoGeGeometry result against the synthetic plate's ground truth.

python tests/eval_moge.py <moge_geo.exr> <moge_geo.json> <synthetic_dir>
Needs numpy + OpenImageIO (Blender's bundled Python has both).
"""

import json
import os
import sys

import numpy as np
import OpenImageIO as oiio

exr_path, json_path, gt_dir = sys.argv[1:4]

inp = oiio.ImageInput.open(exr_path)
names = list(inp.spec().channelnames)
pix = inp.read_image("float")
inp.close()
z = pix[..., names.index("Z")]
mask = pix[..., names.index("mask")] > 0.5

info = json.load(open(json_path))
gt = json.load(open(os.path.join(gt_dir, "ground_truth.json")))
step_file = next(f for f in os.listdir(gt_dir) if f.startswith("gt_depth_step"))
step = int(step_file[len("gt_depth_step"):-4])
gt_z = np.load(os.path.join(gt_dir, step_file))

# Sample the prediction at the same pixel centres the GT rays went through.
zs = z[step // 2::step, step // 2::step][:gt_z.shape[0], :gt_z.shape[1]]
ms = mask[step // 2::step, step // 2::step][:gt_z.shape[0], :gt_z.shape[1]]
valid = ms & (gt_z > 0) & (zs > 0)

scale = np.median(gt_z[valid] / zs[valid])
rel_raw = np.abs(zs[valid] - gt_z[valid]) / gt_z[valid]
rel_aligned = np.abs(zs[valid] * scale - gt_z[valid]) / gt_z[valid]

print("focal px   : moge %.1f  gt %.1f  (err %+.1f%%)" % (
    info["focal_pixels"], gt["focal_pixels"], 100 * (info["focal_pixels"] / gt["focal_pixels"] - 1)))
print("fov x deg  : moge %.2f  gt %.2f" % (info["fov_x_deg"], gt["fov_x_deg"]))
print("focal mm   : moge %.2f  gt %.2f  (36 mm back)" % (info["focal_mm_for_haperture_36"], gt["focal_mm"]))
print("principal  : cx %.4f cy %.4f (0.5 = centred)" % (
    info["intrinsics_normalized"]["cx"], info["intrinsics_normalized"]["cy"]))
print("valid px   : %.1f%% of compared samples" % (100 * valid.mean()))
print("scale gt/moge (median): %.4f" % scale)
print("depth abs-rel, raw metric : mean %.2f%%  median %.2f%%" % (100 * rel_raw.mean(), 100 * np.median(rel_raw)))
print("depth abs-rel, scale-fixed: mean %.2f%%  median %.2f%%  p90 %.2f%%" % (
    100 * rel_aligned.mean(), 100 * np.median(rel_aligned), 100 * np.percentile(rel_aligned, 90)))
