"""Nuke (--nc -t): build a setup Group from an existing run, 'extract for paint' one object, and
render both: the extract must show only that object, and match the full render there.

    Nuke17.1.exe --nc -t tests/nuke_extract.py <run dir> <show only> [USD|classic] [out dir] [shot]

With 'shot', a shot camera (a copy of the projection camera) feeds both Groups' 'cam' input.
"""
import json
import os
import sys

import nuke

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nuke"))
from geoproj import nuke_ui  # noqa: E402
from geoproj.templates import CLASSES, setup_template  # noqa: E402

run, show = sys.argv[1], sys.argv[2]
system = sys.argv[3] if len(sys.argv) > 3 else "USD"
out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(run, "extract_test")
shot = len(sys.argv) > 5 and sys.argv[5] == "shot"
os.makedirs(out, exist_ok=True)
fit = json.load(open(os.path.join(run, "fit.json")))
cam = fit["nuke_camera"]
plate = [f for f in os.listdir(run) if f.startswith("plate.")][0]
frame = int(plate.split(".")[1])
nuke.root()["first_frame"].setValue(frame)
nuke.root()["last_frame"].setValue(frame)
nuke.addFormat("%d %d 1 gp_plate" % (fit["camera"]["width"], fit["camera"]["height"]))
nuke.knob("root.format", "gp_plate")           # the render's size (NC: at most 1920 x 1080)


def paste(text):
    path = os.path.join(out, "paste.nk")
    open(path, "w").write(text)
    nuke.nodePaste(path)


def source(node, i):
    return nuke.tcl("full_name [input root.%s %d]" % (node, i))


nuke.selectAll()
nuke.invertSelection()
vec = lambda v: "{%s}" % " ".join("%.6g" % x for x in v)
paste('Read {\n inputs 0\n file "%s"\n first %d\n last %d\n name Plate\n}\n'
      % (os.path.join(run, "plate.%04d.png").replace("\\", "/"), frame, frame)
      + ('%s {\n inputs 0\n translate %s\n rotate %s\n focal %.6g\n haperture %.6g\n vaperture %.6g\n'
         ' name ShotCam\n}\n' % (CLASSES[system]["camera"], vec(cam["translate"]), vec(cam["rotate"]),
                                 cam["focal"], cam["haperture"], cam["vaperture"]) if shot else ""))
align = {"rotate": [0.0, 0.0, 0.0], "ra_ft": cam["translate"], "shot_t": cam["translate"]}
paste(setup_template("GeoProj_t", frame, os.path.join(run, "scene.abc").replace("\\", "/"), cam,
                     [(p.get("group", "objects"), p["name"]) for p in fit["primitives"]], align,
                     system=system, use_shot_cam=shot, plate="root.Plate", shot_cam=shot and "root.ShotCam"))
grp = nuke.toNode("GeoProj_t")                               # Node object 1
nuke.knob("GeoProj_t.show_only", show)
nuke_ui.extract(grp)                                         # + the root graph
p = show.strip("/*").replace("/", "_") if "*" in show else show.rsplit("/", 1)[-1]
for n, i in (("GeoProj_t", 0), ("_Hold", 0), ("_Paint", 0), ("_Project", 0), ("_Project", 1), ("_CamHold", 0),
             ("_Cam", 0), ("_Geo", 0), ("_Xform", 0), ("_Render", 1), ("_Render", 2)):
    n = n if n.startswith("GeoProj") else p + n
    if nuke.exists(n):
        print("CONNECT %s[%d] <- %s" % (n, i, source(n, i)))
print("BACKDROP", nuke.knob(p + "_Backdrop.label"), "| at", nuke.knob(p + "_Backdrop.xpos"),
      nuke.knob(p + "_Backdrop.ypos"), "| Render at", nuke.knob(p + "_Render.xpos"), nuke.knob(p + "_Render.ypos"))

for src, tag in (("GeoProj_t", "full"), (p + "_Render", "extract")):
    # Crop to the format: the EXR's data window is the bbox, which readers may crop to.
    paste('push [node root.%s]\nCrop {\n box {0 0 %d %d}\n crop false\n}\nWrite {\n file "%s"\n file_type exr\n'
          ' channels rgba\n name W_%s\n}\n' % (src, fit["camera"]["width"], fit["camera"]["height"],
                                               os.path.join(out, tag + ".exr").replace("\\", "/"), tag))
    nuke.execute("W_" + tag, frame, frame, 1)
if system == "classic":
    sv = nuke.knob(p + "_Geo.scene_view").split()
    items = sv[sv.index("items:") + 1:]
    sel = [items[int(i)] for i in sv[sv.index("selected:") + 1:sv.index("items:")]]
    print("CLASSIC selected after render:", sel[:5], "of", len(items))
print("DONE", out)