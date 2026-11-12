"""End-to-end check of the AutoProjection gizmo in Nuke terminal mode (works in Non-commercial).

Nuke17.1.exe --nc -t tests/nuke_e2e.py <plate> <out_dir> [--height 1.6] [--geometry shapes] [--step 4]
        [--preset all] [--shot ground_truth.json [--win u v]] [--system USD|classic]

Creates the gizmo on the plate, presses Build (ComfyUI must be running), checks the pasted
setup and renders rgba + depth.Z through the setup's render camera to <out_dir>/nuke_depth.exr, so
it can be compared with the fit's own proxy depth (tests/compare_nuke_depth.py).
--shot: build a shot Camera3 from a synthetic plate's ground truth (Blender Z-up -> Nuke Y-up)
and connect it to the 'cam' input (keyed per frame when the ground truth has camera_frames).
Sequence: <plate> = .../plate.%04d.png --last N --frames "13 25" --render-frame 25 (then
tests/compare_sequence.py compares the render with the real plate at that frame).
"""

import argparse
import json
import math
import os
import sys

import nuke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nuke"))
import geo_projection  # noqa: E402
from geoproj import xform  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("plate")
ap.add_argument("out_dir")
ap.add_argument("--height", type=float, default=0.0)
ap.add_argument("--geometry", default="shapes")
ap.add_argument("--step", type=int, default=4)
ap.add_argument("--preset", default="all")
ap.add_argument("--shot", default=None)
ap.add_argument("--system", default="USD")
ap.add_argument("--last", type=int, default=1, help="sequence: plate.%%04d.png frames 1..last")
ap.add_argument("--frames", default="", help="sequence: the gizmo's 'extra frames'")
ap.add_argument("--render-frame", type=int, default=1, help="frame the rgba / depth render is written at")
ap.add_argument("--dot", action="store_true", help="connect the shot camera through a Dot")
ap.add_argument("--runs", type=int, default=1, help="2: run twice, check the first Group stays connected")
ap.add_argument("--win", type=float, nargs=2, default=None, help="shot camera win_translate u v")
ap.add_argument("--splat", action="store_true", help="also the SHARP Gaussian splat (needs comfyui-sharp)")
ap.add_argument("--output", default="", help="splat setups: the Group's output (projection, splat, ...)")
a = ap.parse_args(sys.argv[1:])

read = nuke.nodes.Read(file=a.plate.replace("\\", "/"), first=1, last=a.last)
nuke.root()["first_frame"].setValue(1)
nuke.root()["last_frame"].setValue(a.last)
# ScanlineRender without a bg renders at the project format: make it the plate's
# (the default one is over Non-commercial's 1920x1080 limit).
nuke.addFormat("%d %d 1 e2e_plate" % (read.width(), read.height()))
nuke.root()["format"].setValue("e2e_plate")
geo_projection.create_node()
gp = nuke.selectedNode()
gp.setInput(0, read)
print("E2E knobs:", [k for k in ("label_preset", "extra_labels", "show_labels", "geometry", "precision", "system")
                     if k in gp.knobs()], "| presets:", gp["label_preset"].values(), "|",
      gp["ver"].label())
gp["ref_frame"].setValue(1)
gp["in_color"].setValue("sRGB")    # no Viewer in terminal mode
gp["work_size"].setValue(0)
gp["camera_height"].setValue(a.height)
gp["geometry"].setValue(a.geometry)
gp["precision"].setValue(a.step)
gp["label_preset"].setValue(a.preset)
gp["system"].setValue(a.system)
gp["out_dir"].setValue(a.out_dir.replace("\\", "/"))
gp["proj_frames"].setValue(a.frames)
gp["splat"].setValue(a.splat)

if a.shot:
    gt = json.load(open(a.shot))
    mb = gt["camera_matrix_world"]                       # Blender, Z up
    c = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]               # Z up -> Y up
    r = xform.matmul(c, [row[:3] for row in mb[:3]])
    t = xform.matvec(c, [mb[0][3], mb[1][3], mb[2][3]])
    shot = nuke.nodes.Camera3(name="ShotCam")
    shot["translate"].setValue(t)
    shot["rotate"].setValue(xform.to_euler_xyz(r))
    for f, mf in sorted(gt.get("camera_frames", {}).items(), key=lambda kv: int(kv[0])):
        # Sequence: the tracked camera, keyed on every frame.
        rf = xform.matmul(c, [row[:3] for row in mf[:3]])
        tf = xform.matvec(c, [mf[0][3], mf[1][3], mf[2][3]])
        shot["translate"].setAnimated()
        shot["rotate"].setAnimated()
        for i in range(3):
            shot["translate"].setValueAt(tf[i], int(f), i)
            shot["rotate"].setValueAt(xform.to_euler_xyz(rf)[i], int(f), i)
    shot["rot_order"].setValue("XYZ")
    shot["focal"].setValue(gt["focal_mm"])
    shot["haperture"].setValue(gt["sensor_width_mm"])
    shot["vaperture"].setValue(gt["sensor_width_mm"] * gt["height"] / gt["width"])
    if a.win:
        shot["win_translate"].setValue(a.win)
    if a.dot:
        dot = nuke.nodes.Dot(name="ShotDot")         # wired through a Dot, as in a real script
        dot.setInput(0, shot)
        gp.setInput(1, dot)
    else:
        gp.setInput(1, shot)
    print("E2E shot camera:", [round(v, 3) for v in t], [round(v, 2) for v in xform.to_euler_xyz(r)])

geo_projection.run(gp)
print("E2E status:", gp["gp_status"].value(), "|", gp["res_info"].value())

group = "GeoProj_" + gp["gp_status"].value()
if a.runs > 1:
    # A second run must leave the first Group connected as it was (pasting onto the selected
    # gizmo used to rewire it below the new Group).
    geo_projection.run(gp)
    second = "GeoProj_" + gp["gp_status"].value()
    for g in (group, second):
        print("E2E run inputs: %s <- %s%s" % (g, nuke.tcl("full_name [input root.%s 0]" % g),
              " | cam <- " + nuke.tcl("full_name [input root.%s 1]" % g) if a.shot else ""))
# The setup is one Group; read its insides by path (Node objects are scarce in Non-commercial).
print("E2E group:", nuke.exists(group), "| scene_scale", nuke.knob(group + ".scene_scale"),
      "| inside:", [k for k in ("plate", "cam", "Hold", "Camera", "Project", "Geo", "Bind", "Xform", "Scene", "Render")
                    if nuke.exists("%s.%s" % (group, k))])
print("E2E projection camera:", nuke.knob(group + ".Camera.translate"), nuke.knob(group + ".Camera.rotate"),
      nuke.knob(group + ".Camera.focal"))
print("E2E xform:", nuke.knob(group + ".Xform.translate"), nuke.knob(group + ".Xform.rotate"),
      nuke.knob(group + ".Xform.uniform_scale"))

if a.splat:
    print("E2E splat:", nuke.knob(group + ".Splat.file"), "| colorspace", nuke.knob(group + ".SplatRender.output_colorspace"),
          "| output", nuke.knob(group + ".output_mode"))
    if a.output:
        nuke.knob(group + ".output_mode", a.output)
grp = nuke.toNode(group)
if a.shot:
    print("E2E group cam input:", grp.input(1).name() if grp.input(1) else None)
w = nuke.nodes.Write(file=os.path.join(a.out_dir, "nuke_depth.exr").replace("\\", "/"),
                     channels="all", file_type="exr", datatype="32 bit float")
w.setInput(0, grp)
nuke.execute(w, a.render_frame, a.render_frame)
print("E2E rendered", w["file"].value())
if nuke.exists(group + ".Bind"):
    bind = nuke.toNode(group + ".Bind")
    from pxr import UsdShade
    bound = [UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0] for p in bind.getStage().Traverse()
             if p.GetTypeName() == "Mesh"]
    print("E2E bind: error %s | material %s | meshes bound %d/%d" % (
        bind.hasError(), bind["material_path"].value(), sum(1 for b in bound if b), len(bound)))
