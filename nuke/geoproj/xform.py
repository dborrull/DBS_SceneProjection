# AutoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Small 3x3 / rigid-transform helpers in plain Python (Nuke's Python may lack numpy).

Euler angles are Nuke's (and Blender's) rot_order XYZ in degrees: R = Rz * Ry * Rx."""

import math


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]


def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]


def euler_xyz(deg):
    x, y, z = (math.radians(d) for d in deg)
    rx = [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]
    ry = [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]]
    rz = [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]
    return matmul(rz, matmul(ry, rx))


def to_euler_xyz(r):
    """Inverse of euler_xyz (degrees)."""
    sy = max(-1.0, min(1.0, -r[2][0]))
    y = math.asin(sy)
    if abs(sy) < 0.999999:
        x = math.atan2(r[2][1], r[2][2])
        z = math.atan2(r[1][0], r[0][0])
    else:                                   # gimbal lock: put it all in x
        x = math.atan2(-r[1][2], r[1][1])
        z = 0.0
    return [math.degrees(a) for a in (x, y, z)]


def rigid_from_world_matrix(wm):
    """Nuke world_matrix (16 values, row-major, translation in the last column) ->
    (rotation 3x3 without scale, translation)."""
    r = [[wm[4 * i + j] for j in range(3)] for i in range(3)]
    for j in range(3):                      # remove per-axis scale from the columns
        n = math.sqrt(sum(r[i][j] ** 2 for i in range(3))) or 1.0
        for i in range(3):
            r[i][j] /= n
    return r, [wm[3], wm[7], wm[11]]


def project_points(points, r, t, focal, haperture, win, aspect):
    """World points (track space) seen by a Nuke camera (rotation r, its columns the camera axes
    in world, position t) -> [(u, v, depth)] for the points in front of it and inside the frame:
    u, v normalized image coordinates from the top-left (OpenCV), depth along the view axis.
    win: win_translate; aspect: image width / height."""
    rt = transpose(r)
    f = focal / haperture                   # focal in image widths
    cx, cy = 0.5 - win[0] / 2.0, 0.5 + win[1] / 2.0 * aspect
    out = []
    for p in points:
        x, y, z = matvec(rt, [p[0] - t[0], p[1] - t[1], p[2] - t[2]])
        d = -z                               # Nuke cameras look down -Z, Y up
        if d <= 1e-9:
            continue
        u = x / d * f + cx
        v = -y / d * f * aspect + cy
        if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
            out.append((u, v, d))
    return out


def alignment(fit_t, fit_rot_deg, shot_r, shot_t):
    """Rigid transform taking the fitted camera onto the shot camera: x' = Ra x + ta.
    Returns (Ra, Ra @ fit_t, shot_t): with a uniform scale s about the camera,
    x' = s Ra x + (shot_t - s Ra fit_t)."""
    r_fit = euler_xyz(fit_rot_deg)
    ra = matmul(shot_r, transpose(r_fit))
    return ra, matvec(ra, fit_t), list(shot_t)
