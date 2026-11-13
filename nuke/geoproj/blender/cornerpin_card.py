# DBS_ComfyProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""A card on the four corners of a CornerPin, in the world of a setup's Alembic.

blender -b --factory-startup --python cornerpin_card.py -- <run dir> <corners json> <out.abc> <name>

<run dir>: a build's vNNN folder (geo.npz or geo_snapped.npz, geo.json, fit.json).
<corners json>: [[u, v], x4] the CornerPin's 'to' corners at the reference frame, normalized
image coordinates from the top left (to1 .. to4: bottom left, bottom right, top right, top left).

The surface under the CornerPin is MoGe's (snapped to the track when the build had tracker
points): a plane is fitted to its points inside the quad, and each corner's camera ray meets
that plane. So the card lies on the real surface whatever the setup's geometry there is, and
its corners land exactly on the CornerPin's at the reference frame. The card has UVs 0-1 from
to1 to to3 (the pinned image's own corners). Written as <name> under a 'cornerpin' group, like
the setup's objects: /cornerpin/<name>/<name>. Prints CARD {json}: corners, plane residual, points.

The maths (card_corners) is plain numpy, for tests without Blender."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geofit  # noqa: E402

STEP = 2              # MoGe samples used (every STEP px)
SHRINK = 0.85         # the quad shrunk towards its centre for the plane fit: its edges bleed
MIN_POINTS = 30


class CardError(RuntimeError):
    pass


def _inside(u, v, quad):
    """Points (u, v) inside the convex or concave polygon quad [(u, v) x n] (even-odd rule)."""
    inside = np.zeros(u.shape, bool)
    n = len(quad)
    for k in range(n):
        (u0, v0), (u1, v1) = quad[k], quad[(k + 1) % n]
        cross = (v0 > v) != (v1 > v)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = u0 + (v - v0) * (u1 - u0) / (v1 - v0)
        inside ^= cross & (u < x)
    return inside


def _plane(p, iters=4):
    """Plane through points p (N, 3), refitted on the inliers: -> (normal, offset, residual)."""
    keep = np.ones(len(p), bool)
    for _ in range(iters):
        c = p[keep].mean(0)
        n = np.linalg.svd(p[keep] - c)[2][-1]
        d = np.abs((p - c) @ n)
        mad = np.median(d[keep]) + 1e-12
        keep = d <= 3.0 * 1.4826 * mad
    c = p[keep].mean(0)
    n = np.linalg.svd(p[keep] - c)[2][-1]
    return n, float(n @ c), float(np.median(np.abs((p[keep] - c) @ n))), int(keep.sum())


def principal_of(fit):
    """The principal point the fit used (a shot camera's window offset), from its camera shift."""
    cam = fit["camera"]
    cx = 0.5 - cam["shift_x"]
    cy = 0.5 + cam["shift_y"] * cam["width"] / float(cam["height"])
    return None if abs(cx - 0.5) < 1e-9 and abs(cy - 0.5) < 1e-9 else (cx, cy)


def card_corners(pts, valid, info, step, corners, fit):
    """-> (4 corners in the fit's world (Z up), info dict). pts / valid: geofit.load at `step`."""
    k = info["intrinsics_normalized"]
    Hs, Ws = valid.shape
    gu = (np.arange(Ws) * step + step // 2 + 0.5) / info["width"]
    gv = (np.arange(Hs) * step + step // 2 + 0.5) / info["height"]
    U, V = np.meshgrid(gu, gv)
    q = np.array(corners, float)
    centre = q.mean(0)
    inside = _inside(U, V, centre + SHRINK * (q - centre)) & valid
    if inside.sum() < MIN_POINTS:
        raise CardError("Only %d depth samples inside the CornerPin (need %d): is it on the plate, "
                        "and at the reference frame?" % (inside.sum(), MIN_POINTS))
    n, off, res, used = _plane(pts[inside])
    rays = np.column_stack([(q[:, 0] - k["cx"]) / k["fx"], (q[:, 1] - k["cy"]) / k["fy"], np.ones(4)])
    den = rays @ n
    if np.any(np.abs(den) < 1e-9):
        raise CardError("The surface under the CornerPin is seen edge-on: no card.")
    t = off / den
    if np.any(t <= 0):
        raise CardError("A CornerPin corner is beyond the surface's horizon: keep the corners on it.")
    P = rays * t[:, None]                                   # OpenCV camera space, MoGe units
    cam = fit["camera"]
    R = np.array(cam["rotation_matrix"]) @ np.diag([1.0, -1.0, -1.0])
    W = fit.get("units_scale", 1.0) * P @ R.T + np.array(cam["location"])
    depth = float(np.median(pts[inside][:, 2]))
    return W, {"residual_rel": res / depth, "points": used, "depth": depth}


def load_run(run):
    npz = os.path.join(run, "geo_snapped.npz")
    if not os.path.isfile(npz):
        npz = os.path.join(run, "geo.npz")
    fit = json.load(open(os.path.join(run, "fit.json")))
    pts, _, valid, info = geofit.load(npz, os.path.join(run, "geo.json"), STEP, principal_of(fit))
    return pts, valid, info, fit


def main(argv):
    import bpy
    run, corners_json, out, name = argv[:4]
    pts, valid, info, fit = load_run(run)
    try:
        W, rep = card_corners(pts, valid, info, STEP, json.load(open(corners_json)), fit)
    except CardError as e:
        print("CARD ERROR " + str(e))
        sys.exit(2)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    verts = [tuple(v) for v in W]
    face = [0, 1, 2, 3]
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    # Facing the camera ('project on: front'): flip the winding if the normal looks away.
    cam = np.array(fit["camera"]["location"])
    nrm = np.cross(W[1] - W[0], W[3] - W[0])
    if nrm @ (cam - W.mean(0)) < 0:
        face, uvs = [0, 3, 2, 1], [uvs[0], uvs[3], uvs[2], uvs[1]]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], [face])
    uv = mesh.uv_layers.new(name="uv")
    for li, loop in enumerate(mesh.polygons[0].loop_indices):
        uv.data[loop].uv = uvs[li]
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    grp = bpy.data.objects.new("cornerpin", None)
    for o in (grp, obj):
        bpy.context.scene.collection.objects.link(o)
    obj.parent = grp
    bpy.context.scene.frame_start = bpy.context.scene.frame_end = 1
    bpy.ops.wm.alembic_export(filepath=os.path.abspath(out), start=1, end=1, selected=False)
    rep["corners"] = W.tolist()
    print("CARD " + json.dumps(rep))


if __name__ == "__main__" and "--" in sys.argv:
    main(sys.argv[sys.argv.index("--") + 1:])
