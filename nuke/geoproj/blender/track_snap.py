# AutoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Snap MoGe's depth to the camera track's own 3D points, region by region (numpy only).

MoGe guesses depth from one frame; a few percent off is enough for projected paint to slide
once the camera moves. The tracker's points are triangulated from that motion, so they are
right by construction. For every labelled region (one object, the floor, a wall, a backdrop
card) with enough tracker points on it, MoGe's inverse depth gets a correction a + b*u + c*v
fitted to those points:

- depth only changes along the camera rays, so the reference frame still lines up;
- adding a linear function to inverse depth is a projective map of space that keeps the
  camera centre: planes stay planes, and a plane MoGe got tilted or too near becomes exactly
  the plane through the tracker points.

Regions with too few points keep MoGe's shape with the global scale (track_scale.solve).
The corrected point map is written as a new geo.npz, so every later step (fit, splat) uses it.
"""

import numpy as np

import track_scale

MIN_POINTS = 4        # tracker points on a region to correct its depth (constant term)
MIN_POINTS_TILT = 10  # ... and to fit the tilt terms too
MIN_UV_SPREAD = 0.02  # spread of those points across the image (normalized) for the tilt terms
EDGE_PX = 4           # points this close to another region are skipped (mask edges, occlusion)
TRIM = 0.25           # first pass: drop points whose depth ratio is this far off the region's
MAX_RATIO = 1.6       # a correction changing any depth by more than this factor is not trusted
REFIT_ROUNDS = 3


def regions_of(seg_path, roles, shape, load_roles):
    """Region id per pixel, as the fit sees them: object / backdrop instances (mask index) and
    one region per floor / wall / ceiling role. Ignored and unlabelled pixels: -1.
    -> (region grid, {region: name})."""
    role, inst, inst_label, _ = load_roles(seg_path, roles, shape, 1)
    reg = np.where(inst >= 0, inst, -1)
    names = {i: "%s_%d" % (lb, i) for i, lb in inst_label.items()}
    for name, code in (("floor", 4), ("wall", 2), ("ceiling", 3)):
        m = (role == code) & (inst < 0)
        if m.any():
            rid = 10000 + code
            reg[m] = rid
            names[rid] = name
    return reg, names


def _sample(points, W, H):
    a = np.asarray(points, np.float64).reshape(-1, 3)
    x = np.clip((a[:, 0] * W).astype(np.int64), 0, W - 1)
    y = np.clip((a[:, 1] * H).astype(np.int64), 0, H - 1)
    return a, x, y


def _away_from_edges(reg, x, y, r):
    H, W = reg.shape
    own = reg[y, x]
    ok = np.ones(len(x), bool)
    for dx, dy in ((-r, -r), (r, -r), (-r, r), (r, r), (0, -r), (0, r), (-r, 0), (r, 0)):
        ok &= reg[np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)] == own
    return ok


def _fit(u, v, r, tilt):
    """Robust least squares r ~ a + b u + c v (or r ~ a). -> (coef[3], inlier mask)."""
    A = np.stack([np.ones_like(u), u, v], 1) if tilt else np.ones((len(u), 1))
    keep = np.ones(len(u), bool)
    for _ in range(REFIT_ROUNDS):
        c, *_ = np.linalg.lstsq(A[keep], r[keep], rcond=None)
        res = np.abs(A @ c - r)
        mad = np.median(res[keep]) + 1e-12
        new = res <= max(3.0 * 1.4826 * mad, 1e-9)
        if new.sum() < A.shape[1] + 1 or (new == keep).all():
            break
        keep = new
    coef = np.zeros(3)
    coef[:A.shape[1]] = c
    return coef, keep


def correct(npz_path, seg_path, roles, points, out_path, load_roles):
    """Writes the corrected point map to out_path (same keys as geo.npz) -> report dict:
    {"scale", "regions": [{name, points, used, tilt, before, after}], "corrected_share"}.
    before / after: median relative depth error against the tracker points on that region."""
    g = np.load(npz_path)
    data = {k: g[k] for k in g.files}
    pts = data["points"].astype(np.float64)
    z = pts[..., 2]
    valid = data["mask"].astype(bool) & np.isfinite(z) & (z > 0)
    H, W = z.shape
    glob = track_scale.solve(points, z, valid)
    if not glob.get("scale"):
        return {}
    s = glob["scale"]
    reg, names = regions_of(seg_path, roles, (H, W), load_roles) if seg_path else (np.full((H, W), -1), {})

    a, x, y = _sample(points, W, H)
    ok = valid[y, x] & (a[:, 2] > 0) & (reg[y, x] >= 0) & _away_from_edges(reg, x, y, EDGE_PX)
    u_img = (np.arange(W) + 0.5) / W
    v_img = (np.arange(H) + 0.5) / H
    report = []
    new_z = z.copy()
    changed = np.zeros((H, W), bool)
    for rid in np.unique(reg[y[ok], x[ok]]):
        sel = ok & (reg[y, x] == rid)
        zt = a[sel, 2] / s                      # tracker depth in MoGe units
        zm = z[y[sel], x[sel]]
        ratio = zt / zm
        near = np.abs(ratio / np.median(ratio) - 1.0) <= TRIM
        entry = {"name": names.get(int(rid), str(rid)), "points": int(sel.sum()), "used": 0, "tilt": False,
                 "before": float(np.median(np.abs(ratio - 1.0)))}
        report.append(entry)
        if near.sum() < MIN_POINTS:
            continue
        u, v = a[sel, 0][near], a[sel, 1][near]
        r = 1.0 / zt[near] - 1.0 / zm[near]
        tilt = (near.sum() >= MIN_POINTS_TILT and u.std() > MIN_UV_SPREAD and v.std() > MIN_UV_SPREAD)
        coef, keep = _fit(u, v, r, tilt)
        m = reg == rid
        rows, cols = np.nonzero(m)
        w_new = 1.0 / z[rows, cols] + coef[0] + coef[1] * u_img[cols] + coef[2] * v_img[rows]
        q = (1.0 / np.maximum(w_new, 1e-12)) / z[rows, cols]
        if tilt and ((w_new <= 0).any() or np.percentile(q, 99.5) > MAX_RATIO
                     or np.percentile(q, 0.5) < 1.0 / MAX_RATIO):
            # The tilt extrapolates badly across the region: offset only.
            coef, keep = _fit(u, v, r, False)
            tilt = False
            w_new = 1.0 / z[rows, cols] + coef[0]
            q = (1.0 / np.maximum(w_new, 1e-12)) / z[rows, cols]
        if (w_new <= 0).any() or q.max() > MAX_RATIO or q.min() < 1.0 / MAX_RATIO:
            entry["skipped"] = "correction too large"
            continue
        new_z[rows, cols] = z[rows, cols] * q
        changed[rows, cols] = True
        zc = 1.0 / (1.0 / zm + coef[0] + coef[1] * a[sel, 0] + coef[2] * a[sel, 1])
        entry.update(used=int(keep.sum()), tilt=bool(tilt),
                     after=float(np.median(np.abs(zt[near][keep] / zc[near][keep] - 1.0))))

    q = np.where(valid, new_z / np.where(valid, z, 1.0), 1.0)
    data["points"] = (pts * q[..., None]).astype(np.float32)
    if "depth" in data:
        data["depth"] = (data["depth"] * q).astype(np.float32)
    if changed.any():
        data.pop("normal", None)        # tilted regions: the fit recomputes normals from the points
    np.savez(out_path, **data)
    report.sort(key=lambda e: -e["points"])
    rp = reg[y, x]
    return {"scale": s, "regions": report,
            "point_region": [names.get(int(v)) if v >= 0 else None for v in rp],
            "corrected_share": float((changed & valid).sum() / max(valid.sum(), 1))}
