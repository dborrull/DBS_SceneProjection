# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""SHARP Gaussian splat -> the fit's world, Y up (the Alembic's space), as a standard 3DGS PLY.

numpy only: runs inside Blender after the fit, or standalone:
python splat.py <sharp.ply> <geo.npz> <geo.json> <fit.json> <out.ply>

SHARP gives Gaussians in its camera space (OpenCV: x right, y down, z forward) at MoGe's focal
(SharpSplat node), so each one sits on a MoGe ray but at SHARP's own depth scale. That scale is
matched to MoGe on the front-most Gaussian of each pixel (median ratio), then the fit's camera
-> world transform is applied (world = units_scale * R p + camera location, Z up) and the world
turned Y up for Nuke. Positions, rotations and scales of the Gaussians are transformed; colours
are taken from SHARP's sRGB export back to linear (Nuke renders in linear).
"""

import json
import sys

import numpy as np

SH_C0 = 0.28209479177387814
# Blender world (Z up) -> Nuke / Alembic world (Y up): (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1",
          "int": "<i4", "int32": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def read_ply(path):
    """(structured vertex array, header lines). Binary little-endian PLY, vertex element first."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("%s: no end_header" % path)
            line = line.decode("latin1").strip()
            header.append(line)
            if line == "end_header":
                break
        data = f.read()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("%s: only binary little-endian PLY is supported" % path)
    props, n, element = [], 0, None
    for line in header:
        t = line.split()
        if t[0] == "element":
            element = t[1]
            if element == "vertex":
                n = int(t[2])
        elif t[0] == "property" and element == "vertex":
            if t[1] == "list":
                raise ValueError("%s: list properties in the vertex element" % path)
            props.append((t[2], _TYPES[t[1]]))
    return np.frombuffer(data, np.dtype(props), count=n).copy(), header


def write_ply(path, v):
    names = v.dtype.names
    rev = {"<f4": "float", "<f8": "double", "|u1": "uchar", "<i4": "int", "<u4": "uint",
           "<i2": "short", "<u2": "ushort"}
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(v)
    head += "".join("property %s %s\n" % (rev[v.dtype[k].str], k) for k in names)
    head += "end_header\n"
    with open(path, "wb") as f:
        f.write(head.encode("ascii"))
        f.write(np.ascontiguousarray(v).tobytes())


def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def quat_from_matrix(m):
    """(w, x, y, z) of a proper rotation matrix."""
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0] * 4
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """a * b, a one quaternion (4,), b many (N, 4); all (w, x, y, z)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def depth_scale(xyz, points, valid, fx_px, cx_px, cy_px):
    """MoGe depth / SHARP depth on the front-most Gaussian of each pixel (median), and the
    median relative difference left after that scale (how well SHARP's shape agrees)."""
    H, W = valid.shape
    z = xyz[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(xyz[:, 0] / z * fx_px + cx_px)
        v = np.floor(xyz[:, 1] / z * fx_px + cy_px)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok].astype(np.int64), v[ok].astype(np.int64), z[ok]
    keep = valid[v, u]
    u, v, z = u[keep], v[keep], z[keep]
    if len(z) < 100:
        raise ValueError("SHARP splat does not overlap MoGe's valid pixels")
    key = v * W + u
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    first = np.r_[True, key[1:] != key[:-1]]
    key, z = key[first], z[first]
    mz = points.reshape(-1, 3)[key, 2]
    ratio = mz / z
    k = float(np.median(ratio))
    return k, float(np.median(np.abs(ratio / k - 1.0))), int(len(z))


def to_world(sharp_ply, npz, info, fit, out_ply):
    """Write the splat in the fit's Y-up world. Returns a report for fit.json."""
    v, _ = read_ply(sharp_ply)
    for k in ("x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3", "scale_0", "scale_1", "scale_2"):
        if k not in v.dtype.names:
            raise ValueError("%s: not a 3DGS PLY (no %s)" % (sharp_ply, k))
    g = np.load(npz)
    points, valid = g["points"], g["mask"].astype(bool)
    H, W = valid.shape
    kn = info["intrinsics_normalized"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    # SharpSplat gives SHARP MoGe's focal; SHARP's principal point is the image centre.
    k, rel, n_px = depth_scale(xyz, points, valid, kn["fx"] * W, W / 2.0, H / 2.0)

    cam = fit["camera"]
    C = np.array(cam["rotation_matrix"], np.float64)  # Blender camera axes in world
    R = C @ np.diag([1.0, -1.0, -1.0])                 # OpenCV camera -> world (Z up)
    Q = Z_UP_TO_Y_UP @ R
    s = float(fit.get("units_scale") or 1.0) * k
    t = Z_UP_TO_Y_UP @ np.array(cam["location"], np.float64)
    p = s * xyz @ Q.T + t
    v["x"], v["y"], v["z"] = p[:, 0], p[:, 1], p[:, 2]
    q = np.stack([v["rot_%d" % i] for i in range(4)], 1).astype(np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    q = quat_mul(quat_from_matrix(Q), q)
    for i in range(4):
        v["rot_%d" % i] = q[:, i]
    for i in range(3):
        v["scale_%d" % i] = v["scale_%d" % i] + np.log(s)   # log scales
    if all("f_dc_%d" % i in v.dtype.names for i in range(3)):
        for i in range(3):
            c = srgb_to_linear(0.5 + SH_C0 * v["f_dc_%d" % i].astype(np.float64))
            v["f_dc_%d" % i] = (c - 0.5) / SH_C0
    write_ply(out_ply, v)
    return {"file": out_ply.replace("\\", "/"), "gaussians": int(len(v)), "sharp_to_moge": k,
            "shape_rel_median": rel, "pixels_matched": n_px}


def main(argv=None):
    a = (argv or sys.argv[1:])
    sharp, npz, info_json, fit_json, out = a[:5]
    rep = to_world(sharp, npz, json.load(open(info_json)), json.load(open(fit_json)), out)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
