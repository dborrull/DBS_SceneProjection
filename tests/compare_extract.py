"""Compare nuke_extract.py's renders: the extract covers only its object (a small part of the
full render) and, where both render, gives the same pixels unless something nearer hides it in
the full one. Writes preview.jpg (full | extract).

    python tests/compare_extract.py <out dir>      (OPENCV_IO_ENABLE_OPENEXR=1)
"""
import os
import sys

import cv2
import numpy as np

d = sys.argv[1]
f = cv2.imread(os.path.join(d, "full.exr"), cv2.IMREAD_UNCHANGED)
e = cv2.imread(os.path.join(d, "extract.exr"), cv2.IMREAD_UNCHANGED)
cf, ce = f[..., :3].max(-1) > 1e-4, e[..., :3].max(-1) > 1e-4
m = cf & ce
diff = np.abs(f[..., :3] - e[..., :3]).max(-1)
print("cover full %.3f | extract %.4f | where both: median diff %.5f, p99 %.4f"
      % (cf.mean(), ce.mean(), np.median(diff[m]) if m.any() else -1, np.percentile(diff[m], 99) if m.any() else -1))
prev = np.clip(np.concatenate([f[..., :3], e[..., :3]], 1) ** (1 / 2.2) * 255, 0, 255).astype("uint8")
cv2.imwrite(os.path.join(d, "preview.jpg"), cv2.resize(prev, None, fx=0.35, fy=0.35))
