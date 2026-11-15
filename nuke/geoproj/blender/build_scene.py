# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Step 4-5 of the pipeline: MoGe geometry -> fitted camera + primitives -> .blend + Alembic.

blender -b --factory-startup --python nuke/geoproj/blender/build_scene.py -- <geo.npz> <geo.json> <out_dir>
        [--camera-height 1.6] [--geometry shapes|boxes|cards|relief] [--step 4]
        [--seg seg.npz --roles labels.json] [--splat sharp.ply] [--meshes <run>/meshes]

Writes <out_dir>/fit.json (camera, primitives, verification, nuke_camera knob values), fit_proxy_depth.npy,
scene.blend and scene.abc (camera + one mesh per primitive; Alembic is Y-up for Nuke).
"""

import argparse
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geofit  # noqa: E402
import splat  # noqa: E402
import track_scale  # noqa: E402
import track_snap  # noqa: E402


def clear_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)


def add_mesh(name, verts, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def add_plane(p):
    c, u, v = (np.array(p[k]) for k in ("center", "u_axis", "v_axis"))
    hu, hv = u * p["size_u"] / 2, v * p["size_v"] / 2
    verts = [c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv]
    # Winding so the face normal matches the plane normal (towards the camera).
    face = [0, 1, 2, 3] if np.cross(u, v) @ np.array(p["normal"]) > 0 else [0, 3, 2, 1]
    obj = add_mesh(p["name"], verts, [face])
    obj["geo_kind"] = p["kind"]
    return obj


def add_box(p):
    sx, sy, sz = (s / 2 for s in p["size"])
    corners = [(x, y, z) for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]
    quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]
    # Each face gets its own 4 vertices: with the 8 corners shared, Nuke smooths the normals
    # across the edges, and 'project on: front' then drops part of a face seen edge-on.
    verts = [corners[i] for q in quads for i in q]
    faces = [[4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3] for k in range(6)]
    obj = add_mesh(p["name"], verts, faces)
    obj.rotation_euler = (0, 0, np.radians(p["rotation_z_deg"]))
    obj.location = p["center"]
    obj["geo_kind"] = "box"
    obj["depth_guessed"] = p.get("depth_guessed", False)
    return obj


def add_relief(p):
    """A triangle mesh already in world space: a relief mesh (MoGe's own surface for one label)
    or a round shape (cylinder, cone, sphere, revolve)."""
    mesh = bpy.data.meshes.new(p["name"])
    mesh.from_pydata(p["vertices"].tolist(), [], p["faces"].tolist())
    mesh.update()
    obj = bpy.data.objects.new(p["name"], mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj["geo_kind"] = "mesh"
    obj["label"] = p.get("label", "")
    return obj


GEN_VOXELS = 64           # remesh cells along the longest side: Pixal3D meshes are not quite closed,
                          # and finer voxels drop parts of them (truck: 98% of the outline kept at 64,
                          # 68% at 128)
GEN_FACES = 20000         # triangles kept after the remesh


def load_generated(folder):
    """{SAM3 instance: (vertices, triangles)} from <folder>/inst_<i>.glb (meshgen.py), in
    Blender's axes after the glTF import (x right, z up, the crop seen along +y).
    Pixal3D gives 1-6 M triangles. A voxel remesh makes them one closed surface (no holes, all
    normals outward, which 'project on: front' needs), then a decimate takes it to GEN_FACES.
    Vertex clustering left holes and inward faces: black patches in Nuke."""
    import bmesh
    import glob
    import re
    import time
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "inst_*.glb"))):
        i = int(re.search(r"inst_(\d+)\.glb$", path).group(1))
        t0 = time.time()
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=path)
        objs = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        V, F, base, n_in = [], [], 0, 0
        for o in objs:
            n_in += len(o.data.polygons)
            rem = o.modifiers.new("remesh", "REMESH")
            rem.mode = "VOXEL"
            rem.voxel_size = max(max(o.dimensions), 1e-6) / GEN_VOXELS
            dg = bpy.context.evaluated_depsgraph_get()
            n = len(o.evaluated_get(dg).data.polygons)
            if n > GEN_FACES:
                o.modifiers.new("decimate", "DECIMATE").ratio = GEN_FACES / float(n)
            dg = bpy.context.evaluated_depsgraph_get()
            bm = bmesh.new()
            bm.from_mesh(o.evaluated_get(dg).to_mesh())
            bmesh.ops.triangulate(bm, faces=bm.faces[:])
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.verts.index_update()
            m = np.array(o.matrix_world)
            v = np.array([p.co[:] for p in bm.verts], float) @ m[:3, :3].T + m[:3, 3]
            f = np.array([[p.index for p in t.verts] for t in bm.faces], np.int64)
            bm.free()
            V.append(v)
            F.append(f + base)
            base += len(v)
        for o in [o for o in bpy.data.objects if o not in before]:
            bpy.data.objects.remove(o)
        if V:
            out[i] = (np.concatenate(V), np.concatenate(F))
            print("GEOFIT generated mesh instance %d: %d -> %d faces, %.1fs" % (
                i, n_in, len(out[i][1]), time.time() - t0))
    return out


def add_camera(cam):
    data = bpy.data.cameras.new("plate_cam")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = cam["sensor_width_mm"]
    data.sensor_height = cam["sensor_width_mm"] * cam["height"] / cam["width"]
    data.lens = cam["lens_mm"]
    data.shift_x, data.shift_y = cam["shift_x"], cam["shift_y"]
    data.clip_start, data.clip_end = 0.01, 1000.0
    obj = bpy.data.objects.new("plate_cam", data)
    bpy.context.scene.collection.objects.link(obj)
    rot = Matrix([list(r) for r in cam["rotation_matrix"]])
    obj.matrix_world = Matrix.Translation(Vector(cam["location"])) @ rot.to_4x4()
    scene = bpy.context.scene
    scene.camera = obj
    scene.render.resolution_x, scene.render.resolution_y = cam["width"], cam["height"]
    scene.render.resolution_percentage = 100
    return obj


# Blender world (Z up) -> Nuke / Alembic world (Y up): (x, y, z) -> (x, z, -y).
Z_UP_TO_Y_UP = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))


def nuke_camera(cam_obj, cam):
    """Camera3 knob values. Nuke and Blender cameras share local axes (look down -Z, Y up) and
    Nuke's rot_order XYZ is Rz*Ry*Rx like Blender's XYZ Euler, so only the world changes."""
    m = Z_UP_TO_Y_UP @ cam_obj.matrix_world
    return {
        "translate": list(m.to_translation()),
        "rotate": [np.degrees(a) for a in m.to_euler("XYZ")],
        "rot_order": "XYZ",
        "focal": cam["lens_mm"],
        "haperture": cam["sensor_width_mm"],
        "vaperture": cam["sensor_width_mm"] * cam["height"] / cam["width"],
        # Nuke's screen window spans 2 units across the width; Blender shift is in sensor widths.
        "win_translate": [2.0 * cam["shift_x"], 2.0 * cam["shift_y"]],
    }


def geo_ratios(points, proxy, step, scale):
    """Per tracker point: depth of the fitted geometry along its reference-frame ray / the point's
    own depth (1 = on the geometry), or None where the ray misses. Nuke turns it into slide in px."""
    Hs, Ws = proxy.shape
    out = []
    for u, v, d in points:
        # proxy samples sit at step//2 + k*step on the full-resolution grid
        c = int(round((u * Ws * step - step // 2) / step))
        r = int(round((v * Hs * step - step // 2) / step))
        t = proxy[min(max(r, 0), Hs - 1), min(max(c, 0), Ws - 1)] if d > 0 else 0
        out.append(float(t * scale / d) if t > 0 else None)
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("npz")
    ap.add_argument("json")
    ap.add_argument("out_dir")
    ap.add_argument("--camera-height", type=float, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--geometry", default="shapes", choices=sorted(geofit.GEOMETRY))
    ap.add_argument("--detail", default=None, choices=sorted(geofit.LEGACY_DETAIL), help="pre-0.5 levels")
    ap.add_argument("--seg", default=None, help="seg.npz from SegmentLabels")
    ap.add_argument("--roles", default=None, help="labels.json ({label: role})")
    ap.add_argument("--principal", type=float, nargs=2, default=None,
                    help="normalized cx cy of the real camera (shot camera window offset)")
    ap.add_argument("--splat", default=None, help="SHARP splat PLY -> <out_dir>/scene_splat.ply")
    ap.add_argument("--track-points", default=None,
                    help="JSON [(u, v, depth)] of tracker points at the reference frame -> track scale")
    ap.add_argument("--meshes", default=None, help="folder of inst_<i>.glb generated for 'mesh' labels")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    roles = json.load(open(a.roles)) if a.roles else None
    snap = None
    if a.track_points:
        # The tracker's points are the truth: MoGe's depth is snapped to them region by region
        # before anything is fitted, so every primitive (and the splat) sits on the track.
        snapped = os.path.join(a.out_dir, "geo_snapped.npz")
        snap = track_snap.correct(a.npz, a.seg, roles or {}, json.load(open(a.track_points)), snapped,
                                  geofit.load_roles)
        if snap:
            a.npz = snapped
            for e in snap["regions"][:12]:
                print("GEOFIT snap %-22s %4d pts used %4d tilt %-5s error %.2f%% -> %s" % (
                    e["name"], e["points"], e["used"], e["tilt"], 100 * e["before"],
                    "%.2f%%" % (100 * e["after"]) if "after" in e else e.get("skipped", "kept")))
    gen = load_generated(a.meshes) if a.meshes else None
    result, proxy = geofit.fit(a.npz, a.json, a.camera_height, a.step, seg_path=a.seg, roles=roles,
                               geometry=a.geometry, principal=a.principal, detail=a.detail, gen_meshes=gen)
    for p in result["primitives"]:
        if "mesh_fit" in p or "mesh_fallback" in p:
            print("GEOFIT mesh %s: %s %s" % (p["name"], p.get("mesh_fallback") and "fallback to %s (%s)"
                                             % (p.get("shape"), p["mesh_fallback"]) or "placed", p.get("mesh_fit")))
    np.save(os.path.join(a.out_dir, "fit_proxy_depth.npy"), proxy)

    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = scene.frame_end = 1
    cam_obj = add_camera(result["camera"])
    # One empty per group (structure, objects, backdrop, other): the files get /objects/lamp_1/
    # lamp_1 etc., so Nuke can mask a whole group or one object by path.
    groups = {}
    for p in result["primitives"]:
        if "vertices" in p:
            obj = add_relief(p)              # relief, round shapes, cutout cards
            obj["geo_kind"] = p.get("kind", "mesh")
        else:
            obj = (add_plane if p["type"] == "plane" else add_box)(p)
        g = p.get("group", "objects")
        if g not in groups:
            groups[g] = bpy.data.objects.new(g, None)
            bpy.context.scene.collection.objects.link(groups[g])
        obj.parent = groups[g]
        obj["shape"] = p.get("shape", p["type"])
    result["nuke_camera"] = nuke_camera(cam_obj, result["camera"])
    if a.track_points:
        g = np.load(a.npz)
        result["track_points"] = track_scale.solve(json.load(open(a.track_points)), g["points"][..., 2],
                                                   g["mask"].astype(bool))
        tp = result["track_points"]
        if tp and snap:
            tp["snap"] = snap
        if tp:
            tp["geo_ratio"] = geo_ratios(json.load(open(a.track_points)), proxy, a.step or geofit.DEFAULT_STEP,
                                         tp["scale"])
        print("GEOFIT track points: scale %s used %s spread %s" % (tp.get("scale"), tp.get("used"), tp.get("spread")))
    if a.splat:
        # Same world as the Alembic (Y up), so the Group moves both with one transform.
        result["splat"] = splat.to_world(a.splat, a.npz, json.load(open(a.json)), result,
                                         os.path.abspath(os.path.join(a.out_dir, "scene_splat.ply")))
        print("GEOFIT splat: %(gaussians)d gaussians, SHARP -> MoGe x%(sharp_to_moge).4f, shape "
              "difference %(shape_rel_median).4f" % result["splat"])
    with open(os.path.join(a.out_dir, "fit.json"), "w") as f:
        json.dump(geofit.json_safe(result), f, indent=2, default=float)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(os.path.join(a.out_dir, "scene.blend")))
    abc = os.path.abspath(os.path.join(a.out_dir, "scene.abc"))
    bpy.ops.wm.alembic_export(filepath=abc, start=1, end=1, selected=False)
    # Same scene as USD, for other DCCs (Blender's own exporter; Nuke reads the Alembic).
    try:
        bpy.ops.wm.usd_export(filepath=os.path.abspath(os.path.join(a.out_dir, "scene.usd")),
                              selected_objects_only=False, export_animation=False)
    except Exception as e:                   # an older / different USD exporter: not fatal
        print("GEOFIT usd export skipped:", e)
    v = result["verification"]
    print("GEOFIT camera height %.3f lens %.2f mm | %d primitives | coverage %.1f%% abs-rel median %.2f%%"
          % (result["camera_height"], result["camera"]["lens_mm"], len(result["primitives"]),
             100 * v["coverage"], 100 * (v["abs_rel_median"] or 0)))
    print("GEOFIT wrote", abc)


main()
