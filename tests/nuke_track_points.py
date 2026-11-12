"""Scale from tracker points on a real tracked shot (Nuke terminal, Non-commercial is fine).

Nuke17.1.exe --nc -t tests/nuke_track_points.py <script.nk> <read> <camera> <point cloud> <out_dir>
        [--ref 0] [--preset all]

Opens the script, makes a new AutoProjection on <read> with <camera> on 'cam' and <point cloud>
(e.g. CameraTrackerPointCloud1) on 'points', builds it and prints the scale it got. Then
tests/check_track_points.py compares the fitted geometry's depth with the tracker points.
"""

import argparse
import os
import sys

import nuke

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "nuke"))
import geo_projection  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("script", "read", "camera", "points", "out_dir"):
    ap.add_argument(k)
ap.add_argument("--ref", type=int, default=0)
ap.add_argument("--preset", default="all")
ap.add_argument("--geometry", default=None)
a = ap.parse_args(sys.argv[1:])

nuke.scriptOpen(a.script)
geo_projection.create_node()
gp = nuke.selectedNode()
gp.setInput(0, nuke.toNode(a.read))
gp.setInput(1, nuke.toNode(a.camera))
gp.setInput(2, nuke.toNode(a.points))
gp["ref_frame"].setValue(a.ref)
gp["in_color"].setValue("sRGB")    # no Viewer in terminal mode
gp["label_preset"].setValue(a.preset)
if a.geometry:
    gp["geometry"].setValue(a.geometry)
gp["out_dir"].setValue(a.out_dir.replace("\\", "/"))
geo_projection.run(gp)
print("TP status:", gp["gp_status"].value())
print("TP info:", gp["res_info"].value())
print("TP scene_scale:", nuke.knob("GeoProj_%s.scene_scale" % gp["gp_status"].value()))
