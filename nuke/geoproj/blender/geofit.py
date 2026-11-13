# DBS_ComfyProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Fit projection primitives (floor, walls, boxes) + camera to a MoGe geometry dump. numpy only.

Input is what SaveMoGeGeometry writes: geo.npz (points / depth / mask / normal, OpenCV camera
space: x right, y down, z forward) and geo.json (normalized intrinsics).

Output (fit.json) is in a Blender-style world: Z up, floor at Z = 0, camera above the origin
looking down +Y. Units are MoGe's unless camera_height rescales them.

python geofit.py geo.npz geo.json fit.json [--camera-height 1.6] [--step 4]
"""

import argparse
import json
import math

import numpy as np

import shapes

# Tunables (fractions of the median scene depth / of the valid samples).
DIST_TOL = 0.012        # plane inlier distance
ANGLE_TOL_DEG = 20.0    # plane inlier normal agreement
EDGE_TOL = 0.03         # depth jump that breaks connectivity between neighbour samples
MIN_PLANE = 0.01        # smallest plane segment kept
MIN_WALL = 0.025        # smallest vertical segment that can be a wall
PLANE_MARGIN = 0.1      # floor / walls extend this fraction past their points (hidden in a room)
NEAR_PLANE_TOL = 3.0    # samples this many DIST_TOL from a floor / wall are not objects
MIN_OBJECT = 0.003      # smallest object cluster turned into a box
HORIZONTAL_DEG = 25.0   # |normal . up| for floor / tops
VERTICAL_DEG = 20.0     # walls: normal this close to horizontal
RANSAC_ITERS = 400
MAX_PLANES = 16
MIN_PLANE_SAMPLES = 12  # absolute floor of MIN_PLANE: a few labelled pixels are not a plane


def load(npz_path, json_path, step, principal=None):
    """principal: (cx, cy) normalized principal point of the real camera (a shot camera with a
    window offset). MoGe assumes a centred one, so the points are rebuilt from its depth along
    the real camera's rays; the normals are kept."""
    data = np.load(npz_path)
    info = json.load(open(json_path))
    s = slice(step // 2, None, step)
    pts = data["points"][s, s].astype(np.float64)
    if principal is not None:
        info["intrinsics_normalized"]["cx"], info["intrinsics_normalized"]["cy"] = map(float, principal)
        pts = sample_rays(info, pts.shape[:2], step) * pts[..., 2:3]
    valid = data["mask"][s, s].astype(bool) & np.isfinite(pts).all(-1) & (pts[..., 2] > 0)
    if "normal" in data:
        nrm = data["normal"][s, s].astype(np.float64)
    else:
        nrm = _normals_from_grid(pts)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-12
    # Normals face the camera (the point map's view direction is +z).
    flip = (nrm * pts).sum(-1) > 0
    nrm[flip] *= -1
    return pts, nrm, valid, info


def _normals_from_grid(pts):
    dx = np.zeros_like(pts)
    dy = np.zeros_like(pts)
    dx[:, 1:-1] = pts[:, 2:] - pts[:, :-2]
    dy[1:-1] = pts[2:] - pts[:-2]
    return np.cross(dx, dy)


# ---- connected components on the sample grid (numpy label propagation) ----------------

def grid_edges(pts, active):
    z = pts[..., 2]
    eh = active[:, :-1] & active[:, 1:] & (np.abs(z[:, 1:] - z[:, :-1]) < EDGE_TOL * np.minimum(z[:, 1:], z[:, :-1]))
    ev = active[:-1] & active[1:] & (np.abs(z[1:] - z[:-1]) < EDGE_TOL * np.minimum(z[1:], z[:-1]))
    return eh, ev


def components(active, eh, ev):
    """Label 4-connected components of `active` linked by edges eh / ev. -1 = inactive."""
    H, W = active.shape
    big = H * W
    lab = np.where(active, np.arange(big).reshape(H, W), big)
    while True:
        prev = lab
        lab = lab.copy()
        m = np.minimum(prev[:, :-1], prev[:, 1:])
        lab[:, :-1] = np.where(eh, np.minimum(lab[:, :-1], m), lab[:, :-1])
        lab[:, 1:] = np.where(eh, np.minimum(lab[:, 1:], m), lab[:, 1:])
        m = np.minimum(prev[:-1], prev[1:])
        lab[:-1] = np.where(ev, np.minimum(lab[:-1], m), lab[:-1])
        lab[1:] = np.where(ev, np.minimum(lab[1:], m), lab[1:])
        flat = lab.ravel()
        on = flat < big
        flat[on] = flat[flat[on]]  # pointer jumping: follow the label's own label
        lab = flat.reshape(H, W)
        if np.array_equal(lab, prev):
            break
    out = np.full((H, W), -1)
    ids = np.unique(lab[active])
    remap = {v: i for i, v in enumerate(ids)}
    out[active] = [remap[v] for v in lab[active]]
    return out


# ---- planes ---------------------------------------------------------------------------

def fit_plane(p):
    c = p.mean(0)
    # Normal = least-variance direction of the 3x3 covariance: any number of points, and no
    # N x N matrix (a full SVD of the N x 3 points builds one: minutes on a big plane).
    d = p - c
    _, vecs = np.linalg.eigh(d.T @ d)
    return vecs[:, 0], c


def extract_planes(pts, nrm, valid, rng):
    """Sequential RANSAC (point + its normal as hypothesis), keeping the largest connected
    piece of each plane. Returns a label grid (-1 = no plane) and [(normal, point)]."""
    tol = DIST_TOL * np.median(pts[valid][:, 2])
    cos_tol = math.cos(math.radians(ANGLE_TOL_DEG))
    free = valid.copy()
    labels = np.full(valid.shape, -1)
    planes = []
    n_valid = valid.sum()
    min_plane = max(MIN_PLANE * n_valid, MIN_PLANE_SAMPLES)
    eh_all, ev_all = grid_edges(pts, valid)
    for _ in range(MAX_PLANES):
        idx = np.flatnonzero(free)
        if idx.size < min_plane:
            break
        P, N = pts.reshape(-1, 3)[idx], nrm.reshape(-1, 3)[idx]
        best, best_n = None, 0
        for s in rng.choice(idx.size, size=min(RANSAC_ITERS, idx.size), replace=False):
            n = N[s]
            inl = (np.abs((P - P[s]) @ n) < tol) & (N @ n > cos_tol)
            k = inl.sum()
            if k > best_n:
                best, best_n = inl, k
        if best_n < min_plane:
            break
        n, c = fit_plane(P[best])
        if n @ N[best].mean(0) < 0:
            n = -n
        inl = (np.abs((P - c) @ n) < tol) & (N @ n > cos_tol)
        grid = np.zeros(valid.shape, bool)
        grid.ravel()[idx[inl]] = True
        if not grid.any():
            # The refit plane lost its inliers (curved / noisy region): drop the hypothesis's.
            free.ravel()[idx[best]] = False
            continue
        eh, ev = eh_all & grid[:, :-1] & grid[:, 1:], ev_all & grid[:-1] & grid[1:]
        comp = components(grid, eh, ev)
        counts = np.bincount(comp[grid])
        keep = comp == counts.argmax()
        if keep.sum() < min_plane:
            # Largest piece too small: take these samples out of the pool so RANSAC moves on.
            free &= ~grid
            continue
        n, c = fit_plane(pts[keep])
        if n @ nrm[keep].mean(0) < 0:
            n = -n
        labels[keep] = len(planes)
        planes.append((n, c))
        free &= ~keep
    return labels, planes


# ---- world frame ----------------------------------------------------------------------

FLOOR_PITCH_DEG = 70.0   # second pass: a camera looking down this much (a desk shot from above)
PARALLEL_DEG = 10.0      # pieces of one support surface: normals this close
FLOOR_MIN_PIECES = 0.03  # ...together at least this share of the samples


def choose_floor(labels, planes, pts, valid):
    """Floor = the large near-horizontal plane farthest below the camera (OpenCV up is -y).
    First for a roughly level camera (normal within HORIZONTAL_DEG of image up); if none, for
    one pitched down up to FLOOR_PITCH_DEG (the normal leans towards the camera, the roll stays
    small). Not in one pass: a camera pitched up would then take a wall for the floor."""
    try:
        return _choose_floor(labels, planes, valid, _level)
    except RuntimeError:
        return _choose_floor(labels, planes, valid, _pitched)


def _level(n):
    return -n[1] >= math.cos(math.radians(HORIZONTAL_DEG))


def _pitched(n):
    return -n[1] >= math.cos(math.radians(FLOOR_PITCH_DEG)) and n[2] < 0 and roll_of(n) <= HORIZONTAL_DEG


def _choose_floor(labels, planes, valid, upright):
    n_valid = valid.sum()
    best, best_score = None, -np.inf
    for i, (n, c) in enumerate(planes):
        if not upright(n):           # normal (towards camera) must point roughly up
            continue
        size = (labels == i).sum() / n_valid
        height = n @ (np.zeros(3) - c)      # camera height above the plane
        if height <= 0 or size < 0.03:
            continue
        score = height + 0.5 * size * height
        if score > best_score:
            best, best_score = i, score
    if best is None:
        raise RuntimeError("No floor found: no large near-horizontal plane below the camera.")
    return best


def world_frame(floor_n, floor_c):
    """Rotation R (rows = world X, Y, Z in camera coords) and origin (camera foot on floor)."""
    up = floor_n / np.linalg.norm(floor_n)
    h = up @ (np.zeros(3) - floor_c)
    origin = -h * up
    fwd = np.array([0.0, 0.0, 1.0]) - up[2] * up
    if np.linalg.norm(fwd) < 1e-6:
        raise RuntimeError("Camera looks straight down: forward direction is undefined.")
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    return np.stack([x, fwd, up]), origin, h


# ---- primitives -----------------------------------------------------------------------

def min_area_rect(xy):
    """Minimum-area rectangle of 2D points (1-degree search). -> centre, (w, d), angle."""
    best = None
    for a in np.radians(np.arange(0, 90, 1.0)):
        c, s = math.cos(a), math.sin(a)
        r = xy @ np.array([[c, -s], [s, c]])
        lo, hi = np.percentile(r, 2, 0), np.percentile(r, 98, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            centre = ((lo + hi) / 2) @ np.array([[c, s], [-s, c]])  # back from the rotated frame
            best = (area, centre, hi - lo, a)
    return best[1], best[2], best[3]


def plane_quad(name, kind, n_world, pw, margin=PLANE_MARGIN):
    """Quad covering the points pw (world) of a plane with normal n_world, grown by `margin`
    of its size on every side: in a room the extra floor / wall ends up behind other walls."""
    if kind in ("floor", "ceiling"):
        u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    else:
        v = np.array([0, 0, 1.0])
        u = np.cross(v, n_world)
        u /= np.linalg.norm(u)
    a, b = pw @ u, pw @ v
    lo_a, hi_a = a.min(), a.max()
    lo_b, hi_b = b.min(), b.max()
    ga, gb = margin * (hi_a - lo_a), margin * (hi_b - lo_b)
    lo_a, hi_a, lo_b, hi_b = lo_a - ga, hi_a + ga, lo_b - gb, hi_b + gb
    if kind in ("wall", "card"):
        lo_b = 0.0                       # walls and cards stand on the floor
    off = np.median(pw @ n_world)
    centre = n_world * off + u * (lo_a + hi_a) / 2 + v * (lo_b + hi_b) / 2
    return {"name": name, "type": "plane", "kind": kind, "center": centre.tolist(),
            "normal": n_world.tolist(), "u_axis": u.tolist(), "v_axis": v.tolist(),
            "size_u": float(hi_a - lo_a), "size_v": float(hi_b - lo_b), "n_samples": int(len(pw))}


def box_from_points(name, pw, cam_xy, bottom=0.0):
    """Box around the points from `bottom` (0 = standing on the floor) to their top. When only
    one side face is visible the depth is unknown: it is guessed as min(width, height) and
    grown away from the camera."""
    centre_xy, size, a = min_area_rect(pw[:, :2])
    top = np.percentile(pw[:, 2], 99) - bottom
    axes = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])
    thin = int(np.argmin(size))
    guessed = size[thin] < 0.25 * size[1 - thin]
    if guessed:
        new = min(size[1 - thin], top)
        away = np.sign((centre_xy - cam_xy) @ axes[thin]) or 1.0
        centre_xy = centre_xy + axes[thin] * away * (new - size[thin]) / 2
        size = size.copy()
        size[thin] = new
    return {"name": name, "type": "box", "center": [float(centre_xy[0]), float(centre_xy[1]), float(bottom + top / 2)],
            "size": [float(size[0]), float(size[1]), float(top)], "rotation_z_deg": float(math.degrees(a)),
            "depth_guessed": bool(guessed), "n_samples": int(len(pw))}


FACE_TOL = 0.012          # box face inlier distance, x the object's median distance to the camera
FACE_MIN_SHARE = 0.15     # a face needs this share of the object's points
FACE_PERP_DEG = 12.0      # the second face: perpendicular to the first within this
FACE_ITERS = 300


def _line_inliers(xy, d, o, tol):
    """Points of xy (N, 2) within tol of the line {x: x . n = o}, n the normal of direction d."""
    n = np.array([-d[1], d[0]])
    return np.abs(xy @ n - o) <= tol


def box_from_planes(name, pw, cam_xy, bottom=0.0, rng=None):
    """Upright box whose visible side faces sit on the object's own vertical planes (RANSAC
    lines in the footprint): the faces a camera sees land where the points are, not at the
    outer extent of the points, which stray samples (edges, background caught by the mask)
    push out. The far sides, unseen, come from the extent of the points. None when no
    vertical face holds enough of the points."""
    rng = rng or np.random.default_rng(0)
    xy = pw[:, :2]
    if len(xy) > 6000:
        xy = xy[rng.choice(len(xy), 6000, replace=False)]
    tol = FACE_TOL * float(np.median(np.linalg.norm(xy - cam_xy, axis=1)))
    best = None
    for _ in range(FACE_ITERS):
        i, j = rng.choice(len(xy), 2, replace=False)
        d = xy[j] - xy[i]
        L = np.linalg.norm(d)
        if L < 4 * tol:
            continue
        d = d / L
        n = np.array([-d[1], d[0]])
        k = int(_line_inliers(xy, d, xy[i] @ n, tol).sum())
        if best is None or k > best[0]:
            best = (k, d)
    if best is None or best[0] < FACE_MIN_SHARE * len(xy):
        return None
    d1 = best[1]
    n1 = np.array([-d1[1], d1[0]])
    # Refine face 1: its offset = the densest band of the points along its normal, then the
    # direction from the band's principal axis and the offset from its median.
    proj = np.sort(xy @ n1)
    win = np.searchsorted(proj, proj + 2 * tol) - np.arange(len(proj))
    s = int(np.argmax(win))
    in1 = np.abs(xy @ n1 - np.median(proj[s:s + int(win[s])])) <= tol
    c = xy[in1] - xy[in1].mean(0)
    _, vecs = np.linalg.eigh(c.T @ c)
    d1 = vecs[:, 1] / np.linalg.norm(vecs[:, 1])
    n1 = np.array([-d1[1], d1[0]])
    o1 = float(np.median(xy[in1] @ n1))
    in1 = np.abs(xy @ n1 - o1) <= tol
    # Box axes: x along face 1, y its normal pointing away from the camera.
    if (cam_xy @ n1 - o1) > 0:
        n1, o1 = -n1, -o1
    # Face 2: perpendicular to face 1 = a peak of the points along d1 (among the others).
    rest = ~in1
    face2 = None
    if rest.sum() >= FACE_MIN_SHARE * len(xy):
        a = np.sort(xy[rest] @ d1)
        win = np.searchsorted(a, a + 2 * tol) - np.arange(len(a))
        k = int(win.max())
        if k >= FACE_MIN_SHARE * len(xy):
            s = int(np.argmax(win))
            o2 = float(np.median(a[s:s + k]))
            pts2 = xy[rest][np.abs(xy[rest] @ d1 - o2) <= tol]
            # Perpendicular check: face 2's own direction must be close to n1.
            c2 = pts2 - pts2.mean(0)
            _, v2 = np.linalg.eigh(c2.T @ c2)
            if abs(v2[:, 1] @ n1) >= math.cos(math.radians(FACE_PERP_DEG)):
                face2 = (o2, pts2)
    u_all, w_all = xy @ d1, xy @ n1
    w_lo = o1                                            # the seen face
    if face2 is not None:
        o2, pts2 = face2
        w_hi = float(np.percentile(pts2 @ n1, 98))       # face 2 spans the depth
        # Face 2 is at one end along d1: the end nearer the camera's side of the points.
        other = float(np.percentile(u_all, 2 if o2 > np.median(u_all) else 98))
        u_lo, u_hi = sorted((o2, other))
    else:
        u_lo, u_hi = (float(v) for v in np.percentile(xy[in1] @ d1, [2, 98]))
        w_hi = float(np.percentile(w_all, 98))
    guessed = False
    depth = w_hi - w_lo
    top = np.percentile(pw[:, 2], 99) - bottom
    if face2 is None and depth < 0.25 * (u_hi - u_lo):
        depth = min(u_hi - u_lo, top)                    # one face only: depth unknown
        guessed = True
    depth = max(depth, 2 * tol)
    centre_xy = d1 * (u_lo + u_hi) / 2 + n1 * (w_lo + depth / 2)
    a = math.atan2(d1[1], d1[0])
    # Box local axes (box_residual): x = (cos a, sin a) = d1, y = (-sin a, cos a) = +-n1.
    return {"name": name, "type": "box", "center": [float(centre_xy[0]), float(centre_xy[1]), float(bottom + top / 2)],
            "size": [float(u_hi - u_lo), float(depth), float(top)], "rotation_z_deg": float(math.degrees(a)),
            "depth_guessed": bool(guessed), "n_samples": int(len(pw)), "fit": "planes",
            "planes_seen": 1 + (face2 is not None)}


# ---- verification ---------------------------------------------------------------------

def ray_depths(prims, rays_w, origin_w):
    """Camera-space depth (MoGe z) where each ray first hits a primitive. rays_w: (N, 3) world
    directions with camera z = 1, so the ray parameter t is the depth. -> (t, primitive id)."""
    t_best = np.full(len(rays_w), np.inf)
    id_best = np.full(len(rays_w), -1)
    for k, p in enumerate(prims):
        if "vertices" in p and p["type"] != "plane":
            continue                        # relief meshes and round shapes: handled in _finish
        if p["type"] == "plane":
            n, c = np.array(p["normal"]), np.array(p["center"])
            u, v = np.array(p["u_axis"]), np.array(p["v_axis"])
            den = rays_w @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((c - origin_w) @ n) / den
            hit = origin_w + t[:, None] * rays_w - c
            ok = (t > 0) & (np.abs(hit @ u) <= p["size_u"] / 2) & (np.abs(hit @ v) <= p["size_v"] / 2)
            if "cut" in p and len(p["cut"]) == len(rays_w):
                ok &= p["cut"]                  # a cutout card: only where its mesh is
        else:
            a = math.radians(p["rotation_z_deg"])
            R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            o = (origin_w - np.array(p["center"])) @ R
            d = rays_w @ R
            half = np.array(p["size"]) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (-half - o) / d, (half - o) / d
            tn = np.nanmax(np.minimum(t1, t2), 1)
            tf = np.nanmin(np.maximum(t1, t2), 1)
            ok = (tf >= tn) & (tn > 0)
            t = tn
        better = ok & (t < t_best)
        t_best[better] = t[better]
        id_best[better] = k
    return t_best, id_best


def sample_rays(info, shape, step):
    k = info["intrinsics_normalized"]
    Hs, Ws = shape
    cols = (np.arange(Ws) * step + step // 2 + 0.5) / info["width"]
    rows = (np.arange(Hs) * step + step // 2 + 0.5) / info["height"]
    x = (cols[None, :] - k["cx"]) / k["fx"]
    y = (rows[:, None] - k["cy"]) / k["fy"]
    return np.stack(np.broadcast_arrays(x, y, np.ones((Hs, Ws))), -1)


# ---- semantic path (SAM3 labels) --------------------------------------------------------

# Geometry types: what each object becomes ("card", "box", or "auto" = the best of
# shapes.SHAPES per object, overridable per label), whether each backdrop instance (a tree, a
# hill) gets its own card at its own distance, and whether the visible surfaces become relief
# meshes (MoGe's point map, split by label) with the floor / wall planes pushed behind as fill.
# Sample spacing (px) is set separately ("precision").
GEOMETRY = {
    "cards": {"objects": "card", "cards": True, "mesh": False},
    "boxes": {"objects": "box", "cards": True, "mesh": False},
    "shapes": {"objects": "auto", "cards": True, "mesh": False},
    "relief": {"objects": None, "cards": True, "mesh": True},     # cards: fill behind each tree's relief
}
DEFAULT_STEP = 4
# Pre-0.5 detail levels -> (geometry, step), for nodes made before the geometry menu.
LEGACY_DETAIL = {"low": ("boxes", 6), "medium": ("boxes", 4), "high": ("relief", 3)}
# Hierarchy group (top-level Alembic / USD prim) of each kind of primitive.
GROUP_OF = {"floor": "structure", "wall": "structure", "ceiling": "structure", "backdrop": "backdrop",
            "card": "backdrop"}
MIN_MESH_FACES = 20      # smaller relief pieces are dropped
FILL_GAP = 0.01          # fill planes sit this fraction of the median depth behind the relief

ROLE_CODES = {"backdrop": 1, "wall": 2, "ceiling": 3, "floor": 4, "object": 5, "ignore": 6}
MIN_SEG_SAMPLES = 30     # fewest samples of a floor / ceiling / backdrop to fit it
MIN_OBJ_SAMPLES = 15     # fewest samples of an object instance to make a box
MIN_OBJ_SCORE = 0.40     # less confident object masks are left unlabelled (stray detections)
IN_BACKDROP = 0.5        # object masks mostly inside a backdrop mask (a "house" in the trees) too
FLOAT_GAP = 0.15         # an object whose bottom is above this fraction of its top floats
GROUND_TOL = 3.0         # ground inlier distance in DIST_TOL: grass and terrain are bumpy
GROUND_ANGLE_DEG = 30.0
GROUND_ROLL_DEG = 20.0   # a ground plane that would roll the camera more than this is a slope
                         # (an embankment), not the floor the shot stands on
FLOOR_MIN_SHARE = 0.05   # labelled ground under this share of the samples: also try the
                         # largest unlabelled floor plane (a road nobody labelled)


def _role_of(roles, name):
    """roles value is 'role' or 'role:shape' (labels.json keys may use _ for spaces)."""
    r = roles.get(name.replace(" ", "_"), roles.get(name, ""))
    base, _, shape = r.partition(":")
    return base, (shape or None)


def load_roles(seg_path, roles, shape, step):
    """Per-sample role code and object instance. Later rules win where masks overlap:
    backdrop < wall < ceiling < floor < object (smaller instance wins) < ignore.
    Returns (role grid, instance grid, {instance: label}, {instance: forced shape or None})."""
    d = np.load(seg_path)
    names, masks, ml = list(d["labels"]), d["masks"], d["mask_label"]
    scores = d["scores"] if "scores" in d else np.ones(len(masks))
    s = slice(step // 2, None, step)
    masks = masks[:, s, s][:, :shape[0], :shape[1]]
    code_of = [ROLE_CODES.get(_role_of(roles, names[ml[i]])[0], 0) for i in range(len(masks))]
    bd_union = np.zeros(shape, bool)
    for i, c in enumerate(code_of):
        if c == ROLE_CODES["backdrop"]:
            bd_union |= masks[i]
    code_of = [0 if c == ROLE_CODES["object"] and (
        scores[i] < MIN_OBJ_SCORE or (masks[i] & bd_union).sum() > IN_BACKDROP * max(masks[i].sum(), 1))
        else c for i, c in enumerate(code_of)]
    # One object often answers several prompts ("box" and "column"): keep the most confident
    # mask of overlapping object masks, else the object is split into two partial boxes.
    drop = set()
    objs = sorted((i for i in range(len(masks)) if code_of[i] == ROLE_CODES["object"]), key=lambda i: -scores[i])
    for a_i, a in enumerate(objs):
        if a in drop:
            continue
        for b in objs[a_i + 1:]:
            if b in drop:
                continue
            inter = np.logical_and(masks[a], masks[b]).sum()
            small = min(masks[a].sum(), masks[b].sum())
            if inter > 0.5 * np.logical_or(masks[a], masks[b]).sum() or inter > 0.8 * small:
                drop.add(b)
    role = np.zeros(shape, np.int8)
    inst = np.full(shape, -1)        # object and backdrop instances (mask index)
    inst_label = {}
    entries = [(code_of[i], -int(masks[i].sum()), i) for i in range(len(masks)) if i not in drop]
    for code, _, i in sorted(entries):
        if code == 0:
            continue
        role[masks[i]] = code
        if code in (ROLE_CODES["object"], ROLE_CODES["backdrop"]):
            inst[masks[i]] = i
            inst_label[i] = names[ml[i]]
    inst[~np.isin(role, (ROLE_CODES["object"], ROLE_CODES["backdrop"]))] = -1
    inst_shape = {i: _role_of(roles, lb)[1] for i, lb in inst_label.items()}
    return role, inst, inst_label, inst_shape


def robust_inliers(z, gap=EDGE_TOL):
    """Samples of one instance in the unbroken run of depths around its median: background
    (or foreground) that bleeds in at the mask edges sits behind a depth jump. A MAD cut would
    also clip objects that are long in depth (a table seen end-on)."""
    zs = np.sort(z)
    med = np.median(z)
    jumps = np.flatnonzero(np.diff(zs) > gap * med)
    k = np.searchsorted(zs, med)
    lo = zs[jumps[jumps < k][-1] + 1] if np.any(jumps < k) else zs[0]
    hi = zs[jumps[jumps >= k][0]] if np.any(jumps >= k) else zs[-1]
    return (z >= lo) & (z <= hi)


def roll_of(n):
    """Camera roll (degrees, unsigned) a floor with normal n (camera coords) implies; over 90
    when n points down the image."""
    return math.degrees(math.atan2(abs(n[0]), -n[1]))


def geometric_floor(pts, nrm, mask, rng, pitched=True):
    """The floor guessed from geometry alone (choose_floor) among the masked samples.
    pitched: if no level floor, also try a camera looking down (only when nothing is labelled
    ground: next to some labelled ground, the level floor or nothing).
    -> (normal, point, inlier grid) or None."""
    lab, planes = extract_planes(pts, nrm, mask, rng)
    try:
        fi = _choose_floor(lab, planes, mask, _level)
        n, c = planes[fi]
        return n, c, lab == fi
    except RuntimeError:
        if not pitched:
            return None
    # Camera looking down (a desk from above): the support surface tends to come out as several
    # parallel pieces, none big enough alone. Gather the pieces parallel to each candidate,
    # keep the biggest group and fit one loose plane through it.
    cands = [i for i, (n, c) in enumerate(planes) if _pitched(n) and n @ (np.zeros(3) - c) > 0]
    if not cands:
        return None
    size = {i: int((lab == i).sum()) for i in cands}
    cos_par = math.cos(math.radians(PARALLEL_DEG))
    groups = [[j for j in cands if planes[j][0] @ planes[i][0] > cos_par] for i in cands]
    group = max(groups, key=lambda g: sum(size[j] for j in g))
    if sum(size[j] for j in group) < FLOOR_MIN_PIECES * mask.sum():
        return None
    return fit_ground(pts, nrm, np.isin(lab, group), rng)


def fit_ground(pts, nrm, mask, rng):
    """One plane through everything labelled floor / ground, with a loose tolerance and no
    connectivity: a lawn, a path and a patch behind a bench are the same ground even when
    bumpy or cut apart. -> (normal towards the camera, point, inlier grid) or None."""
    idx = np.flatnonzero(mask)
    P, N = pts.reshape(-1, 3)[idx], nrm.reshape(-1, 3)[idx]
    tol = GROUND_TOL * DIST_TOL * np.median(P[:, 2])
    cos_tol = math.cos(math.radians(GROUND_ANGLE_DEG))
    best, best_n = None, 0
    for s in rng.choice(len(P), size=min(RANSAC_ITERS, len(P)), replace=False):
        if roll_of(N[s]) > GROUND_ROLL_DEG:
            continue
        inl = (np.abs((P - P[s]) @ N[s]) < tol) & (N @ N[s] > cos_tol)
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    if best is None or best_n < MIN_SEG_SAMPLES:
        return None
    n, c = fit_plane(P[best])
    if n @ (np.zeros(3) - c) < 0:           # face the camera (it is above the ground)
        n = -n
    if roll_of(n) > GROUND_ROLL_DEG:
        return None
    inl = np.abs((P - c) @ n) < tol
    grid = np.zeros(mask.shape, bool)
    grid.ravel()[idx[inl]] = True
    return n, c, grid


def backdrop_quad(pw_bd, cam, R, info, min_dist=0.0, pct=50):
    """Vertical plane facing the camera at the pct-th percentile distance of the backdrop
    samples (but no nearer than min_dist, so it never cuts through labelled objects), big
    enough to fill the frame there and down to the ground: what projection lands on behind
    everything else."""
    dist = max(np.percentile(pw_bd[:, 1] - cam[1], pct), min_dist)   # world Y = camera forward
    if dist <= 0:
        return None
    k = info["intrinsics_normalized"]
    corners = []
    for u, v in ((0, 0), (1, 0), (0, 1), (1, 1)):
        d = np.array([(u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0]) @ R.T
        if d[1] <= 1e-6:
            continue
        corners.append(cam + d * dist / d[1])
    if len(corners) < 2:
        return None
    pts = np.array(corners + [[cam[0], cam[1] + dist, 0.0]])
    return plane_quad("backdrop", "backdrop", np.array([0, -1.0, 0]), pts)


def card_quad(name, p, cam_xy):
    """Vertical card for one backdrop instance (a tree, a hill): faces the camera from the
    instance's horizontal direction, at its own depth, standing on the ground."""
    d = np.median(p[:, :2], 0) - cam_xy
    if np.linalg.norm(d) < 1e-6:
        return None
    n = np.array([-d[0], -d[1], 0.0]) / np.linalg.norm(d)
    q = plane_quad(name, "card", n, p, margin=0.05)
    return q


CUT_DILATE = 2           # backdrop card cutouts: the instance mask grown by this many samples


def _dilate(m, n):
    out = m.copy()
    for _ in range(n):
        g = out.copy()
        g[1:] |= out[:-1]; g[:-1] |= out[1:]; g[:, 1:] |= out[:, :-1]; g[:, :-1] |= out[:, 1:]
        out = g
    return out


def card_cutout(card, mask, rays_w, cam_w, z=None, valid=None):
    """A backdrop card cut to its instance's shape: the card's plane, but only where the
    (slightly grown) mask is, plus straight down to the floor under it (what nearer things hide
    of the trunk). A rectangle would stand in front of everything farther inside it: the trees
    behind a 'forest' card. rays_w: (H, W, 3) world rays of the sample grid. Adds "vertices",
    "faces" (a grid mesh on the plane, facing the camera) and "cut" (the samples it covers, for
    the verification); the card keeps its plane (center, normal, size) for the rest.
    z / valid (MoGe depth of the grid): the growth and the fill never cover samples farther than
    the card (another tree behind it)."""
    n, c = np.array(card["normal"]), np.array(card["center"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((c - cam_w) @ n) / (rays_w @ n)
    P = cam_w + t[..., None] * rays_w
    region = _dilate(mask, CUT_DILATE)
    region = np.maximum.accumulate(region, axis=0)       # and everything below it in the image
    ok = region & np.isfinite(t) & (t > 0) & (P[..., 2] >= 0)
    if z is not None:
        ok &= mask | ~(valid & (z > t * (1 + SIL_DEPTH_TOL)))
    cell = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    i, j = np.nonzero(cell)
    if len(i) * 2 < MIN_MESH_FACES:
        return card                                       # too small to cut: keep the rectangle
    W = mask.shape[1]
    a, b, cc, d = i * W + j, (i + 1) * W + j, i * W + j + 1, (i + 1) * W + j + 1
    tris = np.concatenate([np.stack([a, b, cc], 1), np.stack([cc, b, d], 1)])
    used, inv = np.unique(tris, return_inverse=True)
    cut = np.zeros(mask.size, bool)
    cut[used] = True
    card = dict(card)
    card.update(vertices=P.reshape(-1, 3)[used], faces=inv.reshape(-1, 3), cut=cut)
    return card


STACK_GAP = 0.35         # a floating object rests on a labelled object below if the gap is
                         # under this fraction of its own height (else it hangs: wall cabinets)
SUPPORT_COVER = 0.6      # ...or reaches the floor when unlabelled points fill this much of the
                         # column under it (a lamp's pedestal, a pot's stand)
SUPPORT_MIN = 10         # fewest such points


def _footprint(p):
    lo, hi = np.percentile(p[:, :2], 5, 0), np.percentile(p[:, :2], 95, 0)
    return (lo + hi) / 2, max(float(np.max(hi - lo)) / 2, 1e-3)


def ground_objects(items, pw_loose):
    """items: [{"p": points, ...}]. Sets each item's "bottom": 0 for objects standing on the
    floor; the top of a labelled object right below (stacking); 0 plus the unlabelled points of
    the column under it when they fill it (that support becomes part of the object); else its
    own lowest point (it hangs). pw_loose: unlabelled samples that may hold an object up (not
    those lying on a wall: the wall under a niche holds nothing)."""
    for it in items:
        p = it["p"]
        it["top"] = float(np.percentile(p[:, 2], 99))
        it["low"] = float(np.percentile(p[:, 2], 2))
        it["c"], it["reach"] = _footprint(p)
    for it in items:
        p = it["p"]
        if it["low"] <= FLOAT_GAP * it["top"]:
            it["bottom"] = 0.0
            continue
        height = it["top"] - it["low"]
        below = [o for o in items if o is not it and o["top"] <= it["low"] + 0.1 * height
                 and np.linalg.norm(it["c"] - o["c"]) <= o["reach"] + 0.5 * it["reach"]
                 and it["low"] - o["top"] <= STACK_GAP * height]
        if below:
            it["bottom"] = max(o["top"] for o in below)
            it["support"] = "stacked"
            continue
        col = pw_loose[(np.linalg.norm(pw_loose[:, :2] - it["c"], axis=1) <= it["reach"])
                       & (pw_loose[:, 2] > 0.02 * it["low"]) & (pw_loose[:, 2] < it["low"])]
        bins = np.linspace(0, it["low"], 9)
        cover = np.mean([((col[:, 2] >= bins[k]) & (col[:, 2] < bins[k + 1])).any() for k in range(8)]) if len(col) else 0
        if len(col) >= SUPPORT_MIN and cover >= SUPPORT_COVER:
            it["p"] = np.vstack([p, col])
            it["bottom"] = 0.0
            it["support"] = "column"
        elif it.get("occluded_below"):
            it["bottom"] = 0.0               # its foot is hidden behind something nearer
            it["support"] = "occluded"
        else:
            it["bottom"] = it["low"]
            it["support"] = "hanging"


SIL_DEPTH_TOL = 0.05     # silhouette check: a sample this much deeper than the object's proxy
                         # is background the proxy would wrongly take the projection of
SIL_WARN = 0.25          # an object whose proxy misses or spills over this share is flagged
OCCLUDER_NEARER = 0.92   # a sample below an object at < this fraction of its depth hides its foot
OCCLUDED_COLS = 0.5      # fraction of the object's image columns that must be hidden that way


OCCLUDED_BAND = 0.15     # only the object's lowest part counts (a lamp shade is wider than its foot)


def occluded_below(mask, z, valid, look=3):
    """True when, under the object's lowest visible samples, the next samples down belong to
    something nearer the camera in most image columns (a sofa arm in front of a lamp
    pedestal): its real bottom is hidden, not missing."""
    cols = np.flatnonzero(mask.any(0))
    if not len(cols):
        return False
    rows = np.flatnonzero(mask.any(1))
    lowest = np.array([np.flatnonzero(mask[:, c]).max() for c in cols])
    cols = cols[lowest >= rows.max() - OCCLUDED_BAND * (rows.max() - rows.min())]
    hidden = 0
    for c in cols:
        r = np.flatnonzero(mask[:, c]).max()
        below = slice(r + 1, min(r + 1 + look, mask.shape[0]))
        zb, vb = z[below, c], valid[below, c] & ~mask[below, c]
        if vb.any() and np.median(zb[vb]) < OCCLUDER_NEARER * z[r, c]:
            hidden += 1
    return hidden >= OCCLUDED_COLS * len(cols)


def make_object(name, label, p, bottom, cam_xy, mode):
    """One object as the requested shape ('card', 'box', a round shape) or, with 'auto', the
    shape whose surface stays closest to the points (weighted by shapes.COMPLEXITY)."""
    card = card_quad(name, p, cam_xy)
    box = box_from_points(name, p, cam_xy, bottom)
    cands = {"box": (box, shapes.box_residual(box, p))}
    boxp = box_from_planes(name, p, cam_xy, bottom)
    if boxp is not None:
        rp = shapes.box_residual(boxp, p)
        if rp <= cands["box"][1]:
            cands["box"] = (boxp, rp)
    if card is not None:
        cands["card"] = (card, shapes.card_residual(card, p))
    fitters = {"cylinder": shapes.fit_cylinder, "cone": shapes.fit_cone, "sphere": shapes.fit_sphere,
               "revolve": shapes.fit_revolve}
    want = list(fitters) if mode == "auto" else ([mode] if mode in fitters else [])
    if len(p) >= shapes.MIN_ROUND_SAMPLES:
        for k in want:
            prim = fitters[k](p, bottom)
            if prim is not None:
                cands[k] = (prim, prim["residual"])
    if mode in cands:
        kind = mode
    elif mode == "auto":
        # A card hugs the one face a camera sees best, but it is no volume: only on request.
        kind = min((k for k in cands if k != "card"), key=lambda k: cands[k][1] * shapes.COMPLEXITY[k])
    else:
        kind = "box"                     # the requested round shape did not fit: fall back
    prim = dict(cands[kind][0])
    prim.update(name=name, label=label, shape=kind, group="objects",
                residuals={k: round(v[1], 5) for k, v in cands.items()})
    return prim


def fit_semantic(pts, nrm, valid, info, seg_path, roles, step, camera_height, rng, geom):
    role, inst, inst_label, inst_shape = load_roles(seg_path, roles, valid.shape, step)
    floor = valid & (role == ROLE_CODES["floor"])
    ground = fit_ground(pts, nrm, floor, rng) if floor.sum() >= MIN_SEG_SAMPLES else None
    floor_source = "labels"
    if ground is None or ground[2].sum() < FLOOR_MIN_SHARE * valid.sum():
        # Little or no level labelled ground (only the verges of an unlabelled road): try the
        # largest floor-like plane among the ground and the unlabelled samples, keep the bigger.
        geo = geometric_floor(pts, nrm, valid & ((role == 0) | (role == ROLE_CODES["floor"])), rng,
                              pitched=ground is None)
        if geo is not None and (ground is None or geo[2].sum() > ground[2].sum()):
            ground, floor_source = geo, "geometry"
    if ground is None:
        # Still nothing: the geometric choice on everything not ignored.
        ground = geometric_floor(pts, nrm, valid & (role != ROLE_CODES["ignore"]), rng)
        floor_source = "geometry"
    if ground is None:
        raise RuntimeError("No floor found: no large near-horizontal plane below the camera.")
    floor_n, floor_c, floor_mask = ground
    R, origin, cam_h = world_frame(floor_n, floor_c)
    scale = camera_height / cam_h if camera_height else 1.0

    def to_world(p):
        return scale * (p - origin) @ R.T

    pw = to_world(pts)
    prims = [plane_quad("floor", "floor", np.array([0, 0, 1.0]), pw[floor_mask])]
    prims[0]["source"] = floor_source

    ceiling = valid & (role == ROLE_CODES["ceiling"])
    if ceiling.sum() >= MIN_SEG_SAMPLES:
        zc = pw[ceiling][:, 2]
        prims.append(plane_quad("ceiling", "ceiling", np.array([0, 0, -1.0]), pw[ceiling][robust_inliers(zc)]))

    walls = valid & (role == ROLE_CODES["wall"])
    if walls.sum() >= MIN_SEG_SAMPLES:
        wl, wplanes = extract_planes(pts, nrm, walls, rng)
        sin_v = math.sin(math.radians(VERTICAL_DEG))
        for i, (n, _) in enumerate(wplanes):
            nw = R @ n
            if abs(nw[2]) >= sin_v:
                continue                  # labelled wall but not vertical (a sloped bit, noise)
            nw = np.array([nw[0], nw[1], 0.0]) / math.hypot(nw[0], nw[1])
            prims.append(plane_quad("wall_%d" % (len(prims)), "wall", nw, pw[wl == i]))

    relief_inst = []                      # (instance, group) made relief by their label
    backdrop = valid & (role == ROLE_CODES["backdrop"])
    if backdrop.sum() >= MIN_SEG_SAMPLES:
        cam_w = to_world(np.zeros(3))
        solid = valid & np.isin(role, (ROLE_CODES["object"], ROLE_CODES["wall"]))
        behind = 1.05 * np.percentile(pw[solid][:, 1] - cam_w[1], 95) if solid.any() else 0.0
        if geom["cards"]:
            # A card per backdrop instance, cut to the instance's shape; the full backdrop moves
            # to the far end and only fills what the cards miss (sky edges, gaps between trees).
            rays_g = scale * sample_rays(info, valid.shape, step).reshape(-1, 3) @ R.T
            rays_g = rays_g.reshape(valid.shape + (3,))
            n_cards = 0
            for i in sorted(set(inst[backdrop].tolist())):
                m = backdrop & (inst == i)
                if m.sum() < MIN_SEG_SAMPLES:
                    continue
                if inst_shape.get(i) == "relief" and not geom["mesh"]:
                    relief_inst.append((i, "backdrop"))    # 'tree:backdrop:relief': a mesh, no card
                    continue
                p = pw[m][robust_inliers(pts[m][:, 2])]
                n_cards += 1
                card = card_quad("card_%d" % n_cards, p, cam_w[:2])
                if card is not None:
                    card["label"] = inst_label[i]
                    card["bd_inst"] = int(i)        # relief: pushed behind this instance's mesh
                    prims.append(card_cutout(card, m, rays_g, cam_w, pts[..., 2], valid))
            bd = backdrop_quad(pw[backdrop], cam_w, R, info, behind, pct=95)
            rel = np.isin(inst, [i for i, g in relief_inst if g == "backdrop"]) & backdrop
            if bd is not None and rel.any():
                # Trees made relief by their label can stand farther than the 95th percentile:
                # the far backdrop goes behind them, as it does in relief mode.
                bd = push_back(bd, pw, rel, FILL_GAP * np.median(pts[valid][:, 2]) * scale)
        else:
            bd = backdrop_quad(pw[backdrop], cam_w, R, info, behind)
        if bd is not None:
            prims.append(bd)

    cam_xy = to_world(np.zeros(3))[:2]
    # Plane inlier distance: a shadow or a picture sits on the wall, a wall cabinet's front
    # stands well off it.
    on_wall_tol = DIST_TOL * np.median(pts[valid][:, 2]) * scale
    wall_quads = [p for p in prims if p["kind"] == "wall"]

    def on_a_wall(p):
        """Most points lie on a fitted wall quad (inside its extent, not just its plane):
        that wall already covers them."""
        for q in wall_quads:
            d = p - np.array(q["center"])
            inside = (np.abs(d @ np.array(q["normal"])) < on_wall_tol) & \
                     (np.abs(d @ np.array(q["u_axis"])) <= q["size_u"] / 2) & \
                     (np.abs(d @ np.array(q["v_axis"])) <= q["size_v"] / 2)
            if inside.mean() > 0.5:
                return True
        return False
    counts = {}
    min_obj = MIN_OBJ_SAMPLES
    objects = valid & (role == ROLE_CODES["object"])
    items = []
    for i in sorted(set(inst[objects].tolist())):
        m = objects & (inst == i)
        if m.sum() < min_obj:
            continue
        p = pw[m][robust_inliers(pts[m][:, 2])]
        if len(p) < MIN_OBJ_SAMPLES:
            continue
        if on_a_wall(p):
            continue                      # flat on a wall (a picture, a shadow): no object
        items.append({"i": i, "p": p, "occluded_below": occluded_below(m, pts[..., 2], valid)})
    if geom["objects"] is not None:
        loose = valid & ((role == 0) | (role == ROLE_CODES["ignore"]))
        for q in wall_quads:                 # samples on a wall are no support
            d = pw - np.array(q["center"])
            loose &= ~((np.abs(d @ np.array(q["normal"])) < 2 * on_wall_tol)
                       & (np.abs(d @ np.array(q["u_axis"])) <= q["size_u"] / 2))
        ground_objects(items, pw[loose])
        for it in items:
            label = inst_label[it["i"]]
            name = label.replace(" ", "_")
            counts[name] = counts.get(name, 0) + 1
            mode = inst_shape.get(it["i"]) or geom["objects"]
            if mode == "relief":
                relief_inst.append((it["i"], "objects"))   # 'rock:object:relief': its surface as a mesh
                continue
            if geom["objects"] in ("card", "box"):
                mode = geom["objects"]    # the menu's single shape wins over per-label shapes
            prim = make_object("%s_%d" % (name, counts[name]), label, it["p"], it["bottom"], cam_xy, mode)
            prim["support"] = it.get("support", "floor")
            prim["inst"] = int(it["i"])       # its SAM3 instance: the silhouette check
            prims.append(prim)
    labelled = valid & (role > 0) & (role != ROLE_CODES["ignore"])
    ctx = {"pw": pw, "role": role, "inst": inst, "inst_label": inst_label, "min_obj": min_obj,
           "relief_inst": relief_inst}
    return prims, R, origin, cam_h, scale, labelled, ctx


# ---- relief meshes (detail high) ---------------------------------------------------------

def relief_meshes(pts, pw, valid, groups, names):
    """Triangulate the sample grid: a quad of four valid, depth-continuous samples gives two
    triangles and goes to the group of its top-left sample. groups: int grid (-1 = none);
    names: {group: (name, label)}. Returns (mesh primitives, per-sample primitive index grid
    relative to the returned list). Winding faces the camera (image down x right in OpenCV
    camera space is -z)."""
    H, W = valid.shape
    eh, ev = grid_edges(pts, valid)
    ok = eh[:-1, :] & eh[1:, :] & ev[:, :-1] & ev[:, 1:]
    gq = groups[:-1, :-1]
    meshes, owner = [], np.full(valid.shape, -1)
    for g, (name, label) in names.items():
        i, j = np.nonzero(ok & (gq == g))
        if len(i) * 2 < MIN_MESH_FACES:
            continue
        a, b, c, d = i * W + j, (i + 1) * W + j, i * W + j + 1, (i + 1) * W + j + 1
        tris = np.concatenate([np.stack([a, b, c], 1), np.stack([c, b, d], 1)])
        used, inv = np.unique(tris, return_inverse=True)
        k = len(meshes)
        owner.ravel()[used] = np.where(owner.ravel()[used] < 0, k, owner.ravel()[used])
        meshes.append({"name": name, "type": "mesh", "kind": "mesh", "label": label,
                       "vertices": pw.reshape(-1, 3)[used], "faces": inv.reshape(-1, 3)})
    return meshes, owner


def label_relief(pts, valid, ctx):
    """Relief meshes for the instances whose label asks for relief ('rock:object:relief',
    'tree:backdrop:relief'), the rest of the fit staying primitives."""
    groups = np.full(valid.shape, -1)
    names, group_of, counts = {}, {}, {}
    for i, grp in ctx["relief_inst"]:
        m = valid & (ctx["inst"] == i)
        label = ctx["inst_label"][i]
        nm = label.replace(" ", "_")
        counts[nm] = counts.get(nm, 0) + 1
        groups[m] = 100 + i
        names[100 + i] = ("%s_%d" % (nm, counts[nm]), label)
        group_of[names[100 + i][0]] = grp
    meshes, owner = relief_meshes(pts, ctx["pw"], valid, groups, names)
    for me in meshes:
        me["group"] = group_of[me["name"]]
    return meshes, owner


def push_back(plane, pw, mask, gap):
    """Move a fill plane behind the relief samples it stands for (mask), so it never hides
    them (niches, recesses) and only shows where the relief has holes."""
    c, n = np.array(plane["center"]), np.array(plane["normal"])
    u, v = np.array(plane["u_axis"]), np.array(plane["v_axis"])
    d = pw[mask] - c
    inside = (np.abs(d @ u) <= plane["size_u"] / 2) & (np.abs(d @ v) <= plane["size_v"] / 2)
    off = d[inside] @ n
    behind = max(0.0, -np.percentile(off, 0.5)) if off.size else 0.0
    plane = dict(plane)
    plane["center"] = (c - n * (behind + gap)).tolist()
    if "vertices" in plane:              # a cutout card: its mesh moves with it
        plane["vertices"] = plane["vertices"] - n * (behind + gap)
    return plane


def mesh_mode(prims, pts, valid, scale, ctx=None, pw=None):
    """Detail high: boxes give way to relief meshes of every visible surface, split by label;
    floor / wall / ceiling / backdrop planes stay as fill, pushed behind their relief."""
    gap = FILL_GAP * np.median(pts[valid][:, 2]) * scale
    groups = np.full(valid.shape, -1)
    names = {}
    if ctx is None:
        groups[valid] = 0
        names[0] = ("relief", "relief")
        fills = [push_back(p, pw, valid, gap) for p in prims if p["type"] == "plane"]
    else:
        pw, role, inst = ctx["pw"], ctx["role"], ctx["inst"]
        other = valid & ((role == 0) | (role == ROLE_CODES["ignore"]))
        groups[other] = 0
        names[0] = ("other_mesh", "other")
        for k, (r, code) in enumerate(ROLE_CODES.items()):
            if r in ("object", "ignore"):
                continue
            g = 1 + k
            groups[valid & (role == code)] = g
            names[g] = ("%s_mesh" % r, r)
        counts = {}
        objects = valid & (role == ROLE_CODES["object"])
        for i in sorted(set(inst[objects].tolist())):
            m = objects & (inst == i)
            if m.sum() < ctx["min_obj"]:
                groups[m] = 0
                continue
            label = ctx["inst_label"][i]
            nm = label.replace(" ", "_")
            counts[nm] = counts.get(nm, 0) + 1
            g = 100 + i
            groups[m] = g
            names[g] = ("%s_%d" % (nm, counts[nm]), label)
        kind_role = {"floor": "floor", "wall": "wall", "ceiling": "ceiling", "backdrop": "backdrop"}
        fills = [push_back(p, pw, valid & (role == ROLE_CODES[kind_role[p["kind"]]]), gap)
                 for p in prims if p["type"] == "plane" and p.get("kind") in kind_role]
        # Backdrop cards: each behind its own instance's relief, filling its holes at the
        # tree's own distance (the far backdrop plane alone slides there).
        bd = valid & (role == ROLE_CODES["backdrop"])
        fills += [push_back(p, pw, bd & (inst == p["bd_inst"]), gap)
                  for p in prims if p.get("kind") == "card" and "bd_inst" in p]
    meshes, owner = relief_meshes(pts, pw, valid, groups, names)
    return fills, meshes, owner


# ---- main -----------------------------------------------------------------------------

def fit(npz_path, json_path, camera_height=None, step=None, seed=0, seg_path=None, roles=None,
        geometry="shapes", principal=None, detail=None):
    """seg_path + roles ({label: role or role:shape}): planes and objects follow the labels.
    Without them, floor / walls / objects are guessed from geometry alone.
    geometry: cards | boxes | shapes | relief (see GEOMETRY); step: sample spacing in px.
    detail (low | medium | high): pre-0.5 levels, mapped to a geometry + step."""
    if detail:
        geometry, legacy_step = LEGACY_DETAIL[detail]
        step = step or legacy_step
    det = GEOMETRY[geometry]
    step = step or DEFAULT_STEP
    pts, nrm, valid, info = load(npz_path, json_path, step, principal)
    rng = np.random.default_rng(seed)
    n_valid = valid.sum()
    if seg_path:
        prims, R, origin, cam_h, scale, labelled, ctx = fit_semantic(
            pts, nrm, valid, info, seg_path, roles or {}, step, camera_height, rng, det)
        meshes = owner = None
        if det["mesh"]:
            prims, meshes, owner = mesh_mode(prims, pts, valid, scale, ctx)
        elif ctx["relief_inst"]:
            meshes, owner = label_relief(pts, valid, ctx)
        return _finish(prims, R, origin, cam_h, scale, pts, valid, info, step, labelled, meshes, owner,
                       objects=valid & (ctx["role"] == ROLE_CODES["object"]), inst=ctx["inst"])

    labels, planes = extract_planes(pts, nrm, valid, rng)
    fi = choose_floor(labels, planes, pts, valid)
    R, origin, cam_h = world_frame(*planes[fi])
    scale = camera_height / cam_h if camera_height else 1.0

    def to_world(p):
        return scale * (p - origin) @ R.T

    cos_v = math.sin(math.radians(VERTICAL_DEG))
    H, W = valid.shape
    prims, used = [], np.zeros_like(valid)
    walls = 0
    for i, (n, c) in enumerate(planes):
        m = labels == i
        nw = R @ n
        if i == fi:
            kind = "floor"
            nw = np.array([0, 0, 1.0])
        elif abs(nw[2]) < cos_v and m.sum() >= MIN_WALL * n_valid:
            rows, cols = np.nonzero(m)
            if not (rows.min() == 0 or cols.min() == 0 or cols.max() == W - 1):
                continue                    # vertical but bounded in frame: an object face
            kind = "wall"
            nw = np.array([nw[0], nw[1], 0.0])
            nw /= np.linalg.norm(nw)
            walls += 1
        else:
            continue
        prims.append(plane_quad(kind if kind == "floor" else "wall_%d" % walls, kind, nw, to_world(pts[m])))
        used |= m

    near_tol = NEAR_PLANE_TOL * DIST_TOL * np.median(pts[valid][:, 2]) * scale
    pw_all = to_world(pts)
    rest = valid & ~used & (pw_all[..., 2] > near_tol)
    for p in prims:                      # leftovers hugging a floor / wall are not objects
        rest &= np.abs((pw_all - np.array(p["center"])) @ np.array(p["normal"])) > near_tol
    eh, ev = grid_edges(pts, rest)
    comp = components(rest, eh, ev)
    nb = 0
    if comp.max() >= 0 and det["objects"] is not None:
        for ci in np.argsort(-np.bincount(comp[rest])):
            m = comp == ci
            if m.sum() < MIN_OBJECT * n_valid:
                break
            nb += 1
            prims.append(make_object("box_%d" % nb, "object", pw_all[m], 0.0, to_world(np.zeros(3))[:2],
                                     det["objects"]))
    meshes = owner = None
    if det["mesh"]:
        prims, meshes, owner = mesh_mode(prims, pts, valid, scale, pw=pw_all)
    return _finish(prims, R, origin, cam_h, scale, pts, valid, info, step, None, meshes, owner)


def assign_groups(prims):
    """Top-level group of each primitive in the Alembic / USD hierarchy: structure (floor,
    walls, ceiling), objects, backdrop (backdrop plane and cards), other (unlabelled relief)."""
    mesh_group = {"floor_mesh": "structure", "wall_mesh": "structure", "ceiling_mesh": "structure",
                  "backdrop_mesh": "backdrop", "other_mesh": "other", "relief": "other"}
    for p in prims:
        if "group" in p:
            continue
        if p["type"] == "plane":
            p["group"] = GROUP_OF.get(p.get("kind"), "structure")
        elif p["type"] == "mesh":
            p["group"] = mesh_group.get(p["name"], "objects")
        else:
            p["group"] = "objects"


def silhouettes(prims, objects, inst, z, valid, rays_w, R, cam_w, scale, info, step):
    """Each object's proxy against its SAM3 mask, on the sample grid: {name: {miss, spill, samples}}.
    miss: share of the mask the proxy leaves out (projected onto whatever is behind instead).
    spill: share of the proxy's own footprint over background (deeper than the proxy, or no
    depth: sky): the projection would paint that background onto the object. Samples where
    something nearer hides the proxy are fine and not counted."""
    out = {}
    for p in prims:
        if "inst" not in p:
            continue
        m = objects & (inst == p["inst"])
        if m.sum() < MIN_OBJ_SAMPLES:
            continue
        if "vertices" in p:
            t, _ = shapes.rasterize_depth([(0, p)], R, cam_w, scale, info, valid.shape, step)
        else:
            t = ray_depths([p], rays_w, cam_w)[0].reshape(valid.shape)
        foot = np.isfinite(t)
        behind = foot & ~m & (~valid | (z > t * (1 + SIL_DEPTH_TOL)))
        out[p["name"]] = {"miss": float((m & ~foot).sum() / m.sum()),
                          "spill": float(behind.sum() / max(foot.sum(), 1)), "samples": int(m.sum())}
    return out


def _finish(prims, R, origin, cam_h, scale, pts, valid, info, step, labelled, meshes=None, owner=None,
            objects=None, inst=None):
    """Camera block + verification (proxy depth vs MoGe depth) shared by both paths. Relief
    meshes are made of the MoGe samples themselves: where they have a vertex, that is the
    nearest surface unless a primitive is in front."""
    def to_world(p):
        return scale * (p - origin) @ R.T

    n_valid = valid.sum()
    assign_groups(prims + (meshes or []))
    # Camera: Blender camera axes in world = R @ diag(1, -1, -1) (OpenCV -> Blender camera).
    k = info["intrinsics_normalized"]
    cam = {
        "location": to_world(np.zeros(3)).tolist(),
        "rotation_matrix": (R @ np.diag([1.0, -1.0, -1.0])).tolist(),
        "sensor_width_mm": 36.0,
        "lens_mm": k["fx"] * 36.0,
        "shift_x": 0.5 - k["cx"],
        "shift_y": (k["cy"] - 0.5) * info["height"] / info["width"],
        "width": info["width"], "height": info["height"],
    }

    # Verification: proxy depth vs MoGe depth on the sample grid.
    rays = sample_rays(info, valid.shape, step)
    rays_w = scale * rays.reshape(-1, 3) @ R.T
    t, pid = ray_depths(prims, rays_w, np.array(cam["location"]))
    t = t.reshape(valid.shape)               # ray param with camera z = 1: MoGe depth units
    pid = pid.reshape(valid.shape)
    shaped = [(k, p) for k, p in enumerate(prims) if p["type"] not in ("plane", "box", "mesh")]
    if shaped:
        tz, towner = shapes.rasterize_depth(shaped, R, np.array(cam["location"]), scale, info, valid.shape, step)
        closer = tz < t
        t = np.where(closer, tz, t)
        pid = np.where(closer, towner, pid)
    z = pts[..., 2]
    if meshes:
        on_mesh = (owner >= 0) & valid & ((z < t) | ~np.isfinite(t))
        t = np.where(on_mesh, z, t)
        pid = np.where(on_mesh, len(prims) + owner, pid)
        prims = prims + meshes
    hit = valid & np.isfinite(t) & (z > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(t - z) / z
    per = {}
    for k_, p in enumerate(prims):
        m = hit & (pid == k_)
        if m.any():
            per[p["name"]] = {"coverage": float(m.sum() / n_valid), "abs_rel_median": float(np.median(rel[m])),
                              "abs_rel_p90": float(np.percentile(rel[m], 90))}
    report = {
        "valid_samples": int(n_valid), "step": step,
        "coverage": float(hit.sum() / n_valid),
        "abs_rel_median": float(np.median(rel[hit])) if hit.any() else None,
        "abs_rel_p90": float(np.percentile(rel[hit], 90)) if hit.any() else None,
        "per_primitive": per,
    }
    if labelled is not None:
        # Labelled, non-ignored samples are the ones that are meant to get geometry.
        lh = hit & labelled
        report["labelled_fraction"] = float(labelled.sum() / n_valid)
        report["labelled_coverage"] = float(lh.sum() / max(labelled.sum(), 1))
        report["labelled_abs_rel_median"] = float(np.median(rel[lh])) if lh.any() else None
    if inst is not None:
        report["silhouette"] = silhouettes(prims, objects, inst, z, valid, rays_w, R,
                                           np.array(cam["location"]), scale, info, step)
    proxy_depth = np.where(np.isfinite(t), t, 0).astype(np.float32)
    return {"units_scale": scale, "camera_height": float(cam_h * scale), "camera": cam,
            "primitives": prims, "verification": report}, proxy_depth


def json_safe(result):
    """fit() result without the relief meshes' arrays (vertex / face counts instead)."""
    out = dict(result)
    out["primitives"] = [
        dict({k: v for k, v in p.items() if k not in ("vertices", "faces", "cut")},
             **({"n_vertices": int(len(p["vertices"])), "n_faces": int(len(p["faces"]))} if "vertices" in p else {}))
        for p in result["primitives"]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("json")
    ap.add_argument("out")
    ap.add_argument("--camera-height", type=float, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--geometry", default="shapes", choices=sorted(GEOMETRY))
    ap.add_argument("--detail", default=None, choices=sorted(LEGACY_DETAIL), help="pre-0.5 levels")
    ap.add_argument("--seg", default=None, help="seg.npz from SegmentLabels")
    ap.add_argument("--roles", default=None, help="labels.json ({label: role})")
    ap.add_argument("--principal", type=float, nargs=2, default=None, help="normalized cx cy")
    a = ap.parse_args(argv)
    roles = json.load(open(a.roles)) if a.roles else None
    result, proxy = fit(a.npz, a.json, a.camera_height, a.step, seg_path=a.seg, roles=roles, geometry=a.geometry,
                        principal=a.principal, detail=a.detail)
    with open(a.out, "w") as f:
        json.dump(json_safe(result), f, indent=2)
    np.save(a.out.rsplit(".", 1)[0] + "_proxy_depth.npy", proxy)
    v = result["verification"]
    print("camera height %.3f  lens %.1f mm" % (result["camera_height"], result["camera"]["lens_mm"]))
    for p in result["primitives"]:
        print("  %-8s %-6s %s" % (p["name"], p["type"], v["per_primitive"].get(p["name"])))
    print("coverage %.1f%%  abs-rel median %.2f%%  p90 %.2f%%" % (
        100 * v["coverage"], 100 * (v["abs_rel_median"] or 0), 100 * (v["abs_rel_p90"] or 0)))


if __name__ == "__main__":
    main()
