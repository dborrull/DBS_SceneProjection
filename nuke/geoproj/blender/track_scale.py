# DBS_ComfyProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Scale between MoGe and a camera track from the tracker's own 3D points (numpy only).

The points come from Nuke already projected through the shot camera at the reference frame:
[(u, v, depth)], u, v normalized from the top-left, depth in track units along the view axis.
Each one is compared with MoGe's depth at that pixel: scale = track units per MoGe unit.
Points hidden behind something nearer (MoGe sees the occluder) or off a thin edge disagree,
so the ratios are trimmed around their median before the final median.
"""

import numpy as np

TRIM = 0.25          # keep ratios within this relative distance of the first median


def solve(points, depth, valid):
    """-> {"scale", "used", "seen", "spread"} or {} when too few points land on valid pixels.
    spread: median relative deviation of the kept ratios (how well MoGe agrees with the track)."""
    H, W = valid.shape
    a = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(a):
        return {}
    x = np.clip((a[:, 0] * W).astype(np.int64), 0, W - 1)
    y = np.clip((a[:, 1] * H).astype(np.int64), 0, H - 1)
    z = depth[y, x]
    ok = valid[y, x] & (z > 0) & (a[:, 2] > 0)
    r = a[ok, 2] / z[ok]
    if len(r) < 3:
        return {}
    med = np.median(r)
    keep = np.abs(r / med - 1.0) <= TRIM
    if keep.sum() >= 3:
        r = r[keep]
    s = float(np.median(r))
    return {"scale": s, "used": int(len(r)), "seen": int(len(a)),
            "spread": float(np.median(np.abs(r / s - 1.0)))}
