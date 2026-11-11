"""Check that the projection setup's render reproduces the plate (tests/nuke_e2e.py output).

python tests/compare_nuke_rgb.py <out_dir>      (Blender's Python: numpy + OpenImageIO)

Rendered through the projection camera itself, projected geometry must give the plate back
wherever there is geometry; the Read turns the sRGB plate into linear, so compare in linear.
"""

import os
import sys

import numpy as np
import OpenImageIO as oiio

out = sys.argv[1]
run = os.path.join(out, "v001")
plate = next(os.path.join(run, f) for f in os.listdir(run) if f.startswith("plate."))
inp = oiio.ImageInput.open(os.path.join(out, "nuke_depth.exr"))
sp = inp.spec()
names = list(sp.channelnames)
px = inp.read_image("float")
inp.close()
full = np.zeros((sp.full_height, sp.full_width, len(names)), np.float32)
full[sp.y:sp.y + sp.height, sp.x:sp.x + sp.width] = px
rgb = full[..., [names.index(c) for c in ("R", "G", "B")]]
alpha = full[..., names.index("A")] > 0.5
p = oiio.ImageBuf(plate).get_pixels(oiio.FLOAT)[..., :3]
lin = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
d = np.abs(rgb - lin).max(-1)
print("rgb: alpha %.1f%% | |render - plate| mean %.4f | pixels within 0.02: %.1f%%" % (
    100 * alpha.mean(), d[alpha].mean(), 100 * (d[alpha] < 0.02).mean()))
