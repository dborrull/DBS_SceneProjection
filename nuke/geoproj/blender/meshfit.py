# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Place a generated mesh (Pixal3D, meshgen.py) on its SAM3 instance. numpy only.

Pixal3D's mesh is pixel aligned with the crop it was made from: seen from the front (Blender
axes after the glTF import: x right, z up, looking along +y) its outline is the crop's. So:
  1. turn it to the camera ray through the instance's mask (two candidates: as is and mirrored),
  2. refine scale and image offset on the silhouette (the mask with its holes filled: holes are
     things in front of the object, and the mesh rightly goes on behind them),
  3. scale it about the camera so its visible surface sits on MoGe's depth (median ratio).
The silhouette fixes the outline; MoGe fixes the distance. Depth along the ray is the model's
own guess: nothing in one frame checks it.

All in MoGe's OpenCV camera space (x right, y down, z forward) on geofit's sample grid."""

import numpy as np

MIN_IOU = 0.6             # below this the mesh is not used (the label falls back to a fitted shape)
PAD = 1.1                 # as ObjectMeshCrop: crop side = longest bbox side x PAD


def dilate(m, n=1):
    out = m.copy()
    for _ in range(n):
        g = out.copy()
        g[1:] |= out[:-1]; g[:-1] |= out[1:]; g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        out = g
    return out


def erode(m, n=1):
    return ~dilate(~m, n)


def fill_holes(m):
    """m with every region not connected to the border set (numpy flood fill)."""
    outside = np.zeros_like(m)
    outside[0, :], outside[-1, :], outside[:, 0], outside[:, -1] = ~m[0, :], ~m[-1, :], ~m[:, 0], ~m[:, -1]
    while True:
        grown = dilate(outside) & ~m
        if (grown == outside).all():
            return ~outside
        outside = grown


def _window(m, margin):
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    H, W = m.shape
    return (max(rows[0] - margin, 0), min(rows[-1] + margin + 1, H),
            max(cols[0] - margin, 0), min(cols[-1] + margin + 1, W))


class Grid(object):
    """Projection of camera-space points onto the sample grid (rows, cols as geofit samples them:
    sample k sits at pixel k*step + step//2 + 0.5)."""

    def __init__(self, info, shape, step):
        k = info["intrinsics_normalized"]
        self.fx, self.fy = k["fx"] * info["width"], k["fy"] * info["height"]
        self.cx, self.cy = k["cx"] * info["width"], k["cy"] * info["height"]
        self.shape, self.step = shape, step

    def pixel_of(self, rc):
        return rc * self.step + self.step // 2 + 0.5

    def project(self, X):
        z = np.maximum(X[:, 2], 1e-6)
        return self.fx * X[:, 0] / z + self.cx, self.fy * X[:, 1] / z + self.cy

    def splat(self, X, win):
        """(silhouette, nearest depth) of the points X inside the window (r0, r1, c0, c1)."""
        r0, r1, c0, c1 = win
        u, v = self.project(X)
        c = np.round((u - self.step // 2 - 0.5) / self.step).astype(int) - c0
        r = np.round((v - self.step // 2 - 0.5) / self.step).astype(int) - r0
        ok = (X[:, 2] > 0) & (r >= 0) & (r < r1 - r0) & (c >= 0) & (c < c1 - c0)
        zb = np.full((r1 - r0, c1 - c0), np.inf)
        np.minimum.at(zb, (r[ok], c[ok]), X[ok, 2])
        sil = fill_holes(erode(dilate(np.isfinite(zb), 1), 1))
        return sil, zb


SAMPLES_PER_CELL = 8      # surface samples per sample-grid cell of the object's window
MIN_SAMPLES, MAX_SAMPLES = 50000, 600000


def _dense(V, F, n, seed=0):
    """n points spread evenly over the surface (by triangle area), plus the vertices: a remeshed
    and decimated mesh has large triangles on flat sides, and its vertices alone leave gaps in
    the splatted silhouette."""
    if not len(F):
        return V
    T = V[F]
    area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    if area.sum() <= 0:
        return V
    rng = np.random.default_rng(seed)
    tri = rng.choice(len(F), size=n, p=area / area.sum())
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    S = T[tri, 0] + a[:, None] * (T[tri, 1] - T[tri, 0]) + b[:, None] * (T[tri, 2] - T[tri, 0])
    return np.concatenate([V, S])


def place(V, F, mask, pts, valid, info, step):
    """V, F: the generated mesh (Blender axes, any scale). mask: the instance on the sample grid.
    -> (camera-space vertices, report) or (None, report)."""
    grid = Grid(info, valid.shape, step)
    mfull = fill_holes(mask)
    win = _window(mfull, 6)
    r0, r1, c0, c1 = win
    mw = mfull[r0:r1, c0:c1]
    rows, cols = np.nonzero(mask)
    pu0, pu1 = grid.pixel_of(cols.min()) - step / 2, grid.pixel_of(cols.max()) + step / 2
    pv0, pv1 = grid.pixel_of(rows.min()) - step / 2, grid.pixel_of(rows.max()) + step / 2
    mu, mv, mh = (pu0 + pu1) / 2, (pv0 + pv1) / 2, pv1 - pv0
    ray = np.array([(mu - grid.cx) / grid.fx, (mv - grid.cy) / grid.fy, 1.0])
    ray /= np.linalg.norm(ray)
    xax = np.cross([0.0, 1.0, 0.0], ray)
    xax /= np.linalg.norm(xax)
    Rray = np.stack([xax, np.cross(ray, xax), ray], 1)   # camera z -> the ray, image up kept up
    P = _dense(V, F, int(np.clip(SAMPLES_PER_CELL * mw.size, MIN_SAMPLES, MAX_SAMPLES)))
    report = {}
    best = None
    for mirrored in (False, True):
        s = -1.0 if mirrored else 1.0
        def turn(Q):
            return np.stack([Q[:, 0] * s, -Q[:, 2], Q[:, 1] * s], 1) @ Rray.T
        C = turn(P)
        cm = C.mean(0)
        C = C - cm
        k = (mh / grid.fy) / max(np.ptp(C[:, 1]), 1e-9)   # height spans the mask at distance 1
        X0 = C * k + ray

        def put(sc, du, dv):
            X = (X0 - ray) * sc + ray
            X[:, 0] += du / grid.fx
            X[:, 1] += dv / grid.fy
            return X

        def iou(X):
            sil, _ = grid.splat(X, win)
            return (sil & mw).sum() / max((sil | mw).sum(), 1)

        sc, du, dv = 1.0, 0.0, 0.0
        for px, ds in ((4 * step, 0.08), (2 * step, 0.04), (step, 0.02), (step / 2.0, 0.01)):
            cands = [(sc + a * ds, du + b * px, dv + c * px) for a in (-1, 0, 1) for b in (-1, 0, 1)
                     for c in (-1, 0, 1)]
            sc, du, dv = max(cands, key=lambda t: iou(put(*t)))
        X = put(sc, du, dv)
        sil, zb = grid.splat(X, win)
        z = pts[r0:r1, c0:c1, 2]
        both = sil & mask[r0:r1, c0:c1] & valid[r0:r1, c0:c1] & np.isfinite(zb)
        if both.sum() < 5:
            continue
        lam = float(np.median(z[both] / zb[both]))
        Vc = (turn(V) - cm) * k * sc + ray
        Vc[:, 0] += du / grid.fx
        Vc[:, 1] += dv / grid.fy
        Vc *= lam
        sil, zb = grid.splat(X * lam, win)
        both = sil & mask[r0:r1, c0:c1] & valid[r0:r1, c0:c1] & np.isfinite(zb)
        rel = np.abs(zb[both] - z[both]) / z[both]
        r = {"iou": float((sil & mw).sum() / max((sil | mw).sum(), 1)),
             "miss": float((mw & ~sil).sum() / max(mw.sum(), 1)),
             "spill": float((sil & ~mw).sum() / max(sil.sum(), 1)),
             "depth_rel_median": float(np.median(rel)) if rel.size else None,
             "depth_rel_p90": float(np.percentile(rel, 90)) if rel.size else None,
             "mirrored": mirrored, "scale_refine": round(sc, 3)}
        if best is None or r["iou"] > best[1]["iou"]:
            best = (Vc, r)
    if best is None:
        return None, {"reason": "mesh does not cover the mask"}
    report = best[1]
    if report["iou"] < MIN_IOU:
        report["reason"] = "silhouette IoU %.2f < %.2f" % (report["iou"], MIN_IOU)
        return None, report
    return best[0], report

