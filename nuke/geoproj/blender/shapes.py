# AutoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Object primitives beyond boxes, fitted to an object's MoGe points in the fit's world
(Z up, floor at Z = 0). numpy only.

Round shapes share one vertical axis (gravity is known from the floor), so half an object
seen from one side is enough: cylinder, cone, sphere and revolve (a lathe profile, for vases
and lamps) are all lathe meshes. Each fitter returns a primitive dict with world-space
"vertices" / "faces" and a "residual": the median distance from the points to the surface,
used to pick the best shape."""

import math

import numpy as np

SHAPES = ("auto", "box", "card", "cylinder", "cone", "sphere", "revolve")
# Extra cost of each shape when picking automatically: a more flexible shape must fit
# clearly better to be chosen (and a card only wins on really flat objects).
COMPLEXITY = {"box": 1.0, "cylinder": 1.1, "cone": 1.2, "sphere": 1.2, "revolve": 1.35, "card": 1.6}
SEGMENTS = 32            # lathe segments around the axis
MIN_ROUND_SAMPLES = 40   # fewer points: boxes only
PROFILE_BINS = 16


# ---- lathe mesh -----------------------------------------------------------------------------

def lathe(center_xy, profile, segments=SEGMENTS, cap=True):
    """profile: [(z, r), ...] bottom to top. Returns (vertices (N, 3), faces (M, 3)), faces
    wound outwards."""
    prof = [(float(z), max(float(r), 1e-4)) for z, r in profile]
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], 1)
    verts = []
    for z, r in prof:
        verts.append(np.column_stack([center_xy[0] + r * ring[:, 0], center_xy[1] + r * ring[:, 1], np.full(segments, z)]))
    v = np.concatenate(verts)
    faces = []
    for i in range(len(prof) - 1):
        for j in range(segments):
            a0, a1 = i * segments + j, i * segments + (j + 1) % segments
            b0, b1 = a0 + segments, a1 + segments
            faces += [[a0, a1, b1], [a0, b1, b0]]
    if cap:
        # The caps get their own rim vertices: shared with the side, Nuke smooths the normals
        # across the rim (and 'project on: front' misjudges the faces there).
        nb = len(v)
        top0 = (len(prof) - 1) * segments
        v = np.vstack([v, [center_xy[0], center_xy[1], prof[0][0]], [center_xy[0], center_xy[1], prof[-1][0]],
                       v[:segments], v[top0:top0 + segments]])
        b0, t0 = nb + 2, nb + 2 + segments
        for j in range(segments):
            faces.append([nb, b0 + (j + 1) % segments, b0 + j])               # bottom, facing down
            faces.append([nb + 1, t0 + j, t0 + (j + 1) % segments])           # top, facing up
    return v, np.array(faces, np.int64)


# ---- fitters --------------------------------------------------------------------------------

def _circle(xy, iters=2):
    """Algebraic (Kasa) circle fit, refitted on the best 80%. -> (center, radius)."""
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        p = xy[keep]
        A = np.column_stack([2 * p, np.ones(len(p))])
        sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
        c = sol[:2]
        r = math.sqrt(max(sol[2] + c @ c, 1e-12))
        res = np.abs(np.linalg.norm(xy - c, axis=1) - r)
        keep = res <= np.percentile(res, 80)
    return c, r


def _extent_xy(p):
    lo, hi = np.percentile(p[:, :2], 2, 0), np.percentile(p[:, :2], 98, 0)
    return float(np.linalg.norm(hi - lo))


def _span_z(p, bottom):
    top = float(np.percentile(p[:, 2], 99))
    return bottom, max(top, bottom + 1e-3)


ROUND_MAX_R = 0.75       # radius at most this fraction of the points' footprint: half a round
                         # object spans ~2 radii; a flat face gives a huge radius


def _round_ok(c, r, p):
    """Reject circle fits that are really a flat face (a huge radius from points on a line)
    or that sit far from the points."""
    ext = _extent_xy(p)
    return 0 < r <= ROUND_MAX_R * max(ext, 1e-3) and np.linalg.norm(np.median(p[:, :2], 0) - c) <= 1.5 * r


def fit_cylinder(p, bottom):
    c, r = _circle(p[:, :2])
    if not _round_ok(c, r, p):
        return None
    z0, z1 = _span_z(p, bottom)
    v, f = lathe(c, [(z0, r), (z1, r)])
    res = np.median(np.abs(np.linalg.norm(p[:, :2] - c, axis=1) - r))
    return {"type": "cylinder", "center": [float(c[0]), float(c[1]), (z0 + z1) / 2], "radius": r,
            "size": [2 * r, 2 * r, z1 - z0], "vertices": v, "faces": f, "residual": float(res)}


def fit_cone(p, bottom):
    c, r = _circle(p[:, :2])
    if not _round_ok(c, r, p):
        return None
    z0, z1 = _span_z(p, bottom)
    rho = np.linalg.norm(p[:, :2] - c, axis=1)
    A = np.column_stack([np.ones(len(p)), p[:, 2]])
    a, b = np.linalg.lstsq(A, rho, rcond=None)[0]
    r0, r1 = max(a + b * z0, 1e-3), max(a + b * z1, 1e-3)
    v, f = lathe(c, [(z0, r0), (z1, r1)])
    res = np.median(np.abs(rho - (a + b * p[:, 2])))
    return {"type": "cone", "center": [float(c[0]), float(c[1]), (z0 + z1) / 2], "radius": max(r0, r1),
            "size": [2 * max(r0, r1)] * 2 + [z1 - z0], "radius_bottom": r0, "radius_top": r1,
            "vertices": v, "faces": f, "residual": float(res)}


def fit_revolve(p, bottom):
    """Lathe profile: the median radius at each height around one vertical axis."""
    c, r = _circle(p[:, :2])
    if not _round_ok(c, r, p):
        return None
    z0, z1 = _span_z(p, bottom)
    rho = np.linalg.norm(p[:, :2] - c, axis=1)
    edges = np.linspace(z0, z1, PROFILE_BINS + 1)
    zs, rs = [], []
    for i in range(PROFILE_BINS):
        m = (p[:, 2] >= edges[i]) & (p[:, 2] <= edges[i + 1])
        if m.sum() >= 4:
            zs.append((edges[i] + edges[i + 1]) / 2)
            rs.append(np.median(rho[m]))
    if len(zs) < 3:
        return None
    rs = np.array(rs)
    sm = rs.copy()                       # 3-tap median: one noisy bin does not make a ridge
    for i in range(1, len(rs) - 1):
        sm[i] = np.median(rs[i - 1:i + 2])
    prof = [(z0, sm[0])] + list(zip(zs, sm)) + [(z1, sm[-1])]
    v, f = lathe(c, prof)
    res = np.median(np.abs(rho - np.interp(p[:, 2], [q[0] for q in prof], [q[1] for q in prof])))
    return {"type": "revolve", "center": [float(c[0]), float(c[1]), (z0 + z1) / 2], "radius": float(sm.max()),
            "size": [2 * float(sm.max())] * 2 + [z1 - z0], "profile": [[float(z), float(r)] for z, r in prof],
            "vertices": v, "faces": f, "residual": float(res)}


def fit_sphere(p, bottom=None):
    A = np.column_stack([2 * p, np.ones(len(p))])
    sol = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    r = math.sqrt(max(sol[3] + c @ c, 1e-12))
    if not (0 < r <= ROUND_MAX_R * max(_extent_xy(p), 1e-3)):
        return None
    n = 12                                # lathe of a half circle
    prof = [(c[2] - r * math.cos(math.pi * i / n), r * math.sin(math.pi * i / n)) for i in range(n + 1)]
    v, f = lathe(c[:2], prof, cap=False)
    res = np.median(np.abs(np.linalg.norm(p - c, axis=1) - r))
    return {"type": "sphere", "center": c.tolist(), "radius": r, "size": [2 * r] * 3,
            "vertices": v, "faces": f, "residual": float(res)}


def box_residual(box, p):
    """Median distance of the points to a box's surface (box dict from geofit.box_from_points)."""
    a = math.radians(box["rotation_z_deg"])
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    q = np.abs((p - np.array(box["center"])) @ R) - np.array(box["size"]) / 2
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(q.max(1), 0)
    return float(np.median(np.abs(outside + inside)))


def card_residual(card, p):
    return float(np.median(np.abs((p - np.array(card["center"])) @ np.array(card["normal"]))))


# ---- verification: depth of triangle meshes on the sample grid --------------------------------

def rasterize_depth(prims, R, cam_w, scale, info, shape, step):
    """Camera depth (MoGe units) of the nearest triangle per sample, for primitives carrying
    "vertices" / "faces" (round shapes). Z-buffer over the sample grid, perspective-correct.
    -> (depth grid with inf where empty, primitive index grid)."""
    Hs, Ws = shape
    k = info["intrinsics_normalized"]
    W, H = info["width"], info["height"]
    depth = np.full(shape, np.inf)
    owner = np.full(shape, -1)
    gx = (np.arange(Ws) * step + step // 2 + 0.5)
    gy = (np.arange(Hs) * step + step // 2 + 0.5)
    for idx, p in prims:
        # world -> OpenCV camera: world = scale * (cam_cv @ R.T)  =>  cam_cv = (world - cam) @ R / scale
        cv = (p["vertices"] - cam_w) @ R / scale
        z = cv[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (k["cx"] + k["fx"] * cv[:, 0] / z) * W
            v = (k["cy"] + k["fy"] * cv[:, 1] / z) * H
        for tri in p["faces"]:
            if np.any(z[tri] <= 1e-6):
                continue
            tu, tv, tz = u[tri], v[tri], z[tri]
            c0 = max(int(np.searchsorted(gx, tu.min())), 0)
            c1 = min(int(np.searchsorted(gx, tu.max())), Ws)
            r0 = max(int(np.searchsorted(gy, tv.min())), 0)
            r1 = min(int(np.searchsorted(gy, tv.max())), Hs)
            if c0 >= c1 or r0 >= r1:
                continue
            X, Y = np.meshgrid(gx[c0:c1], gy[r0:r1])
            d = (tv[1] - tv[2]) * (tu[0] - tu[2]) + (tu[2] - tu[1]) * (tv[0] - tv[2])
            if abs(d) < 1e-12:
                continue
            w0 = ((tv[1] - tv[2]) * (X - tu[2]) + (tu[2] - tu[1]) * (Y - tv[2])) / d
            w1 = ((tv[2] - tv[0]) * (X - tu[2]) + (tu[0] - tu[2]) * (Y - tv[2])) / d
            w2 = 1 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            zz = 1.0 / (w0 / tz[0] + w1 / tz[1] + w2 / tz[2])
            blk = depth[r0:r1, c0:c1]
            better = inside & (zz < blk)
            blk[better] = zz[better]
            owner[r0:r1, c0:c1][better] = idx
    return depth, owner
