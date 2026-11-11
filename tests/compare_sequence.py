"""Sequence mode check: the setup's render (tests/nuke_e2e.py --render-frame F) against the real
plate at frame F. Away from the reference frame, a single projection leaves stretched or
wrong texture where the camera now sees what the reference frame did not; extra frames
should fill those areas with the right pixels.

    python tests/compare_sequence.py <render.exr> <plate.%04d.png> <frame>   (OPENCV_IO_ENABLE_OPENEXR=1)

Prints coverage (rendered pixels), median error and the share of rendered pixels that are
clearly wrong (> 0.05 linear), and writes <render>_vs_plate.jpg (render | plate | error x4).
"""
import sys

import cv2
import numpy as np

render, plate, frame = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = cv2.imread(render, cv2.IMREAD_UNCHANGED)[..., :3]
p = cv2.imread(plate % frame, cv2.IMREAD_UNCHANGED)[..., :3] / 255.0
p = np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)      # sRGB PNG -> linear
cover = r.max(-1) > 1e-4
err = np.abs(r - p).max(-1)
print("frame %d: coverage %.3f | median error %.4f | wrong (> 0.05) %.2f%% of rendered"
      % (frame, cover.mean(), np.median(err[cover]), 100 * (err[cover] > 0.05).mean()))
to8 = lambda x: np.clip(x ** (1 / 2.2) * 255, 0, 255).astype("uint8")
e = np.repeat(np.clip(err * 4, 0, 1)[..., None], 3, -1)
cv2.imwrite(render[:-4] + "_vs_plate.jpg", cv2.resize(np.concatenate([to8(r), to8(p), (e * 255).astype("uint8")], 1),
                                                      None, fx=0.3, fy=0.3))
