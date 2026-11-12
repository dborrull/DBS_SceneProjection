# AutoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Steps 2-5 without Nuke: plate PNG -> MoGe (ComfyUI) -> Blender fit -> fit.json + scene.abc.

python -m geoproj.pipeline <plate.png> <out_base> [--camera-height 1.6] [--focal 28 --haperture 36]
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import time

from .comfy_client import ComfyError, next_version_dir
from .estimate import estimate_geometry
from .labels import DEFAULT_PRESET, effective_labels, list_presets

BUILD_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender", "build_scene.py")


def find_blender(explicit=""):
    """Blender executable: explicit path, $GEO_PROJECTION_BLENDER, else the newest
    'Blender Foundation/Blender X.Y' install that has a blender.exe."""
    for p in (explicit, os.environ.get("GEO_PROJECTION_BLENDER", "")):
        if p and os.path.isfile(p):
            return p
    found = []
    for root in (os.environ.get("ProgramFiles", r"C:\Program Files"), "/Applications", "/opt", "/usr/local"):
        for exe in glob.glob(os.path.join(root, "Blender Foundation", "Blender *", "blender.exe")) + \
                glob.glob(os.path.join(root, "blender*", "blender")):
            m = re.search(r"(\d+)\.(\d+)", exe)
            found.append(((int(m.group(1)), int(m.group(2))) if m else (0, 0), exe))
    if not found:
        raise ComfyError("Blender not found. Set the 'blender' knob or $GEO_PROJECTION_BLENDER.")
    return max(found)[1]


def fov_from_focal(focal_mm, haperture_mm):
    return math.degrees(2 * math.atan(haperture_mm / (2.0 * focal_mm))) if focal_mm > 0 else 0.0


# Variables Blender may inherit. Inside Nuke the environment carries Nuke's own USD / Python /
# Qt setup that Blender has no use for.
_ENV_KEEP = {"SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "USERPROFILE", "USERNAME",
             "USERDOMAIN", "HOMEDRIVE", "HOMEPATH", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
             "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "COMMONPROGRAMFILES",
             "COMMONPROGRAMFILES(X86)", "COMMONPROGRAMW6432", "COMPUTERNAME", "NUMBER_OF_PROCESSORS",
             "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "PROCESSOR_LEVEL", "PROCESSOR_REVISION",
             "OS", "PATHEXT", "COMSPEC", "LANG", "LC_ALL", "DISPLAY", "XDG_RUNTIME_DIR",
             "CUDA_VISIBLE_DEVICES", "GEO_PROJECTION_BLENDER"}


def clean_env(environ=None):
    """(env for Blender, names dropped). Allow-list, plus PATH without Nuke's folders."""
    environ = dict(os.environ if environ is None else environ)
    env = {k: v for k, v in environ.items() if k.upper() in _ENV_KEEP}
    path = [p for p in environ.get("PATH", "").split(os.pathsep) if p and "nuke" not in p.lower()]
    env["PATH"] = os.pathsep.join(path)
    dropped = sorted(k for k in environ if k.upper() not in _ENV_KEEP and k.upper() != "PATH")
    return env, dropped


def run_blender(blender, npz, json_path, out_dir, camera_height=0.0, seg=None, roles_json=None,
                geometry=("shapes", 4), principal=None, is_cancelled=lambda: False, log=print, timeout=600,
                splat=None, track_points=None):
    cmd = [blender, "-b", "--factory-startup", "--python", BUILD_SCENE, "--", npz, json_path, out_dir,
           "--geometry", geometry[0]] + (["--step", str(geometry[1])] if geometry[1] else [])
    if splat:
        cmd += ["--splat", splat]
    if track_points:
        tp = os.path.join(out_dir, "track_points.json")
        with open(tp, "w") as f:
            json.dump([list(p) for p in track_points], f)
        cmd += ["--track-points", tp]
    if camera_height > 0:
        cmd += ["--camera-height", str(camera_height)]
    if seg and roles_json:
        cmd += ["--seg", seg, "--roles", roles_json]
    if principal is not None:
        cmd += ["--principal", "%.8g" % principal[0], "%.8g" % principal[1]]
    log("blender: " + " ".join('"%s"' % c if " " in c else c for c in cmd))
    log_path = os.path.join(out_dir, "blender.log")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env, dropped = clean_env()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("[AutoProjection] env vars not passed to Blender: %s\n\n" % ", ".join(dropped))
        lf.flush()
        # cwd: Nuke's GUI runs in its install folder and Windows looks for DLLs in the current
        # directory, so Blender would pick up Foundry's USD libraries there and exit.
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags, env=env,
                                cwd=out_dir)
        t0 = time.time()
        while proc.poll() is None:
            if is_cancelled():
                proc.kill()
                raise ComfyError("Cancelled by the user.")
            if time.time() - t0 > timeout:
                proc.kill()
                raise ComfyError("Blender timed out (see %s)." % log_path)
            time.sleep(0.2)
    fit_path = os.path.join(out_dir, "fit.json")
    abc = os.path.join(out_dir, "scene.abc")
    if proc.returncode != 0 or not os.path.isfile(abc):
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = "".join(f.readlines()[-15:])
        what = ("the fit script failed (Blender exits 0 on a Python error)" if proc.returncode == 0
                else "Blender failed (exit %s)" % proc.returncode)
        raise ComfyError("%s. End of %s:\n%s" % (what[0].upper() + what[1:], log_path, tail))
    with open(fit_path, encoding="utf-8") as f:
        fit = json.load(f)
    res = {"fit": fit, "fit_json": fit_path.replace("\\", "/"), "abc": abc.replace("\\", "/")}
    if splat:
        if not fit.get("splat") or not os.path.isfile(fit["splat"]["file"]):
            raise ComfyError("The splat was not converted (see %s)." % log_path)
        res["splat"] = fit["splat"]["file"]
    return res


def build(image_path, out_dir, server="127.0.0.1:8188", focal_mm=0.0, haperture_mm=36.0,
          camera_height=0.0, blender="", labels=None, geometry=("shapes", 4), principal=None,
          is_cancelled=lambda: False, log=print, on_step=lambda s: None, scale_frames=None, cameras=None,
          splat=False, track_points=None):
    """The whole chain for one plate image. labels: {text prompt: role} for SAM3 (None / {} =
    geometry-only fit); geometry: (cards | boxes | shapes | relief, sample step in px).
    scale_frames / cameras: solve the tracked camera's scale too (estimate_geometry).
    splat: also a SHARP Gaussian splat, in the Alembic's world (res["splat"]: scene_splat.ply).
    track_points: [(u, v, depth)] tracker points at the reference frame -> fit["track_points"]
    (scale between MoGe and the track).
    Returns run_blender's dict + "geo"."""
    blender_exe = find_blender(blender)        # before ComfyUI: no Blender -> say so at once
    on_step(("MoGe + SAM3" if labels else "MoGe") + (" + SHARP" if splat else "") + " (ComfyUI)")
    geo = estimate_geometry(image_path, out_dir, server, fov_from_focal(focal_mm, haperture_mm),
                            labels=labels, is_cancelled=is_cancelled, log=log,
                            scale_frames=scale_frames, cameras=cameras, splat=splat)
    on_step("Blender fit")
    roles_json = os.path.join(out_dir, "labels.json") if geo.get("seg") else None
    res = run_blender(blender_exe, geo["npz"], geo["json"], out_dir, camera_height,
                      geo.get("seg"), roles_json, geometry, principal, is_cancelled=is_cancelled, log=log,
                      splat=geo.get("splat"), track_points=track_points)
    res["geo"] = geo
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image")
    ap.add_argument("out_base")
    ap.add_argument("--server", default=os.environ.get("GEO_PROJECTION_SERVER", "127.0.0.1:8188"))
    ap.add_argument("--focal", type=float, default=0.0, help="known focal in mm (0 = estimate)")
    ap.add_argument("--haperture", type=float, default=36.0, help="film back width in mm")
    ap.add_argument("--camera-height", type=float, default=0.0)
    ap.add_argument("--blender", default="")
    ap.add_argument("--preset", default=DEFAULT_PRESET, help="label preset (%s)" % ", ".join(list_presets()))
    ap.add_argument("--labels", default="", help="extra 'text:role' labels on top of the preset")
    ap.add_argument("--no-seg", action="store_true", help="geometry-only fit, no SAM3")
    ap.add_argument("--geometry", default="shapes", choices=("shapes", "boxes", "cards", "relief"))
    ap.add_argument("--step", type=int, default=4, help="precision: sample spacing in px")
    ap.add_argument("--splat", action="store_true", help="also a SHARP Gaussian splat (comfyui-sharp)")
    a = ap.parse_args(argv)
    labels = {} if a.no_seg else effective_labels(a.preset, a.labels)
    r = build(a.image, next_version_dir(a.out_base), a.server, a.focal, a.haperture, a.camera_height,
              a.blender, labels, (a.geometry, a.step), splat=a.splat)
    if r.get("splat"):
        print("splat: %s" % r["splat"])
    v = r["fit"]["verification"]
    print("abc: %s\ncoverage %.1f%%  abs-rel median %.2f%%" % (r["abc"], 100 * v["coverage"], 100 * v["abs_rel_median"]))


if __name__ == "__main__":
    main()
