"""Render a synthetic test plate with known camera + geometry (ground truth for the PoC).

blender -b --factory-startup --python tests/make_synthetic_plate.py -- <out_dir>

Writes <out_dir>/plate.png (display-referred beauty), gt_depth_step4.npy (planar
depth sampled every 4 px) and ground_truth.json (camera + primitives). With SEQ_FRAMES=n
also a moving-camera sequence plate.0001..n.png (see below).
"""

import json
import math
import os
import sys

import bpy
from mathutils import Vector

out_dir = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
os.makedirs(out_dir, exist_ok=True)

W, H = 1280, 720
FOCAL_MM, SENSOR_MM = 28.0, 36.0

scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)


def material(name, color, scale):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = [c * 0.55 for c in color] + [1.0]
    ramp.color_ramp.elements[1].color = list(color) + [1.0]
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8
    return m


primitives = []


def box(name, size, location, rot_z_deg, mat):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=(0, 0, math.radians(rot_z_deg)))
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    o.data.materials.append(mat)
    primitives.append({"name": name, "type": "box", "size": list(size), "location": list(location),
                       "rotation_z_deg": rot_z_deg})
    return o


def plane(name, size, location, rotation_deg, mat):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location,
                                     rotation=[math.radians(a) for a in rotation_deg])
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], 1.0)
    o.data.materials.append(mat)
    primitives.append({"name": name, "type": "plane", "size": list(size), "location": list(location),
                       "rotation_deg": list(rotation_deg)})
    return o


# Room: floor z=0, back wall y=6, left wall x=-4 (Blender Z-up, metres).
plane("floor", (10, 14), (0, 1, 0), (0, 0, 0), material("floor", (0.55, 0.45, 0.35), 12))
plane("back_wall", (10, 4), (0, 6, 2), (90, 0, 0), material("back", (0.75, 0.75, 0.7), 5))
plane("left_wall", (14, 4), (-4, 1, 2), (90, 0, 90), material("left", (0.5, 0.6, 0.7), 5))
box("crate_a", (1.2, 1.2, 1.2), (-1.6, 3.2, 0.6), 20, material("crate_a", (0.7, 0.4, 0.2), 20))
box("crate_b", (0.8, 2.0, 0.6), (1.4, 2.0, 0.3), -15, material("crate_b", (0.3, 0.5, 0.3), 25))
box("pillar", (0.5, 0.5, 3.0), (2.6, 4.5, 1.5), 0, material("pillar", (0.8, 0.8, 0.8), 10))

bpy.ops.object.camera_add(location=(0.4, -4.0, 1.6))
cam = bpy.context.active_object
cam.name = "plate_cam"
direction = Vector((-0.2, 3.0, 0.7)) - cam.location
cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
cam.data.lens = FOCAL_MM
cam.data.sensor_fit = "HORIZONTAL"
cam.data.sensor_width = SENSOR_MM
scene.camera = cam

bpy.ops.object.light_add(type="SUN", rotation=(math.radians(50), 0, math.radians(30)))
bpy.context.active_object.data.energy = 3.0
bpy.ops.object.light_add(type="AREA", location=(0, 1, 3.8))
bpy.context.active_object.data.energy = 400.0
bpy.context.active_object.data.size = 4.0

world = scene.world or bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
bg.inputs["Strength"].default_value = 0.3

try:
    scene.render.engine = "CYCLES"
except TypeError as e:
    print("engine:", e)
scene.cycles.samples = 64
scene.cycles.use_denoising = True
try:
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.device = "GPU"
except Exception as e:
    print("GPU setup failed, using CPU:", e)

scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.resolution_percentage = 100
vl = scene.view_layers[0]
vl.use_pass_z = True
vl.use_pass_position = True

scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join(out_dir, "plate.png")
bpy.ops.render.render(write_still=True)

# Sequence (SEQ_FRAMES=n): frames 1..n as plate.%04d.png, the camera trucking SEQ_TRUCK metres
# to the right (and a little forward) while aiming at the same point; frame 1 = the still above.
# Ground truth gets each frame's camera matrix ("camera_frames").
SEQ_FRAMES = int(os.environ.get("SEQ_FRAMES", 0))
camera_frames = {}
if SEQ_FRAMES:
    truck = float(os.environ.get("SEQ_TRUCK", 2.0))
    start = cam.location.copy()
    scene.cycles.samples = int(os.environ.get("SEQ_SAMPLES", 32))
    for f in range(1, SEQ_FRAMES + 1):
        a = (f - 1) / max(SEQ_FRAMES - 1, 1)
        cam.location = start + Vector((truck * a, 0.6 * a, 0.0))
        cam.rotation_euler = (Vector((-0.2, 3.0, 0.7)) - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        camera_frames[f] = [list(r) for r in cam.matrix_world]
        scene.render.filepath = os.path.join(out_dir, "plate.%04d.png" % f)
        bpy.ops.render.render(write_still=True)
    cam.location = start
    cam.rotation_euler = (Vector((-0.2, 3.0, 0.7)) - start).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()

# Ground-truth planar depth (distance along the camera axis) by ray casting at GT_STEP
# pixel spacing, sampled at pixel centres of the full-res plate. 0 = no hit.
GT_STEP = int(os.environ.get("GT_STEP", 4))
import numpy as np

depsgraph = bpy.context.evaluated_depsgraph_get()
mw = cam.matrix_world
origin = mw.translation.copy()
rot = mw.to_3x3()
forward = (rot @ Vector((0, 0, -1))).normalized()
f_px = FOCAL_MM / SENSOR_MM * W
gt_depth = np.zeros((H // GT_STEP, W // GT_STEP), np.float32)
for j in range(gt_depth.shape[0]):
    v = j * GT_STEP + 0.5 * GT_STEP
    for i in range(gt_depth.shape[1]):
        u = i * GT_STEP + 0.5 * GT_STEP
        d = rot @ Vector(((u - W / 2) / f_px, -(v - H / 2) / f_px, -1.0))
        hit, loc, *_ = scene.ray_cast(depsgraph, origin, d.normalized())
        if hit:
            gt_depth[j, i] = (loc - origin).dot(forward)
np.save(os.path.join(out_dir, "gt_depth_step%d.npy" % GT_STEP), gt_depth)

gt = {
    "width": W, "height": H,
    "focal_mm": FOCAL_MM, "sensor_width_mm": SENSOR_MM,
    "focal_pixels": FOCAL_MM / SENSOR_MM * W,
    "fov_x_deg": math.degrees(2 * math.atan(SENSOR_MM / 2 / FOCAL_MM)),
    "camera_location": list(cam.location),
    "camera_rotation_euler_xyz_rad": list(cam.rotation_euler),
    "camera_matrix_world": [list(r) for r in mw],
    "blender_convention": "Z up, camera looks down -Z, +Y up in camera space",
    "primitives": primitives,
}
if camera_frames:
    gt["camera_frames"] = camera_frames
with open(os.path.join(out_dir, "ground_truth.json"), "w") as f:
    json.dump(gt, f, indent=2)
print("wrote", out_dir)
