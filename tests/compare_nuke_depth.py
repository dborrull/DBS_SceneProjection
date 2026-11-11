"""Compare Nuke's ScanlineRender depth (tests/nuke_e2e.py) with the fit's proxy depth.

python tests/compare_nuke_depth.py <run_dir>     (Blender's Python: numpy + OpenImageIO)

Both come from the same fitted camera + primitives, one in Nuke (Camera3 knobs + Alembic),
one ray traced in geofit, so they only agree if the hand-off to Nuke is right.
"""

import json
import os
import sys

import numpy as np
import OpenImageIO as oiio

run = sys.argv[1]
fit = json.load(open(os.path.join(run, "fit.json")))
step = fit["verification"]["step"]
proxy = np.load(os.path.join(run, "fit_proxy_depth.npy")) * fit["units_scale"]

inp = oiio.ImageInput.open(os.path.join(run, "..", "nuke_depth.exr"))
names = list(inp.spec().channelnames)
img = inp.read_image("float")
inp.close()
ch = next(i for i, n in enumerate(names) if n.lower().endswith("z"))
inv_z = img[..., ch]                 # classic ScanlineRender writes 1/z (0 = nothing)
nuke_z = np.where(inv_z > 0, 1.0 / np.maximum(inv_z, 1e-12), 0.0)
nuke_s = nuke_z[step // 2::step, step // 2::step][:proxy.shape[0], :proxy.shape[1]]

ok = (proxy > 0) & (nuke_s > 0)
rel = np.abs(nuke_s[ok] - proxy[ok]) / proxy[ok]
print("channels %s" % names)
print("nuke covers %.1f%% of fitted pixels" % (100 * ok.sum() / max((proxy > 0).sum(), 1)))
print("nuke vs fit depth: abs-rel median %.3f%%  p99 %.3f%%  max %.2f%%" % (
    100 * np.median(rel), 100 * np.percentile(rel, 99), 100 * rel.max()))
