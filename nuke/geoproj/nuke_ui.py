# GeoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Nuke side of GeoProjection. Every entry point gets the GeoProjection node from its button
(nuke.thisNode()) and reaches internal nodes by path, so a run costs at most four Node
objects (this node, the shot camera, the active Viewer, the graph to paste into); pasted
setups connect themselves by name - Nuke Non-commercial allows 10 per session. Helpers marked "ComfyBridge" are
copied from it."""

import json
import math
import os
import re
import tempfile

import nuke

from .colour import IN_VIEWER, LUT, _CS_SRGB, ocio_names, parse_viewer_process, pick_colorspace
from .comfy_client import ComfyClient, ComfyError, next_version_dir
from .labels import DEFAULT_PRESET, ROLES, LabelError, effective_labels, list_presets
from .pipeline import build, find_blender
from .templates import extract_template, node_template, select_prims, setup_template
from .xform import alignment, project_points, rigid_from_world_matrix, to_euler_xyz

SETUP_DX, SETUP_DY = 150, 60


def _log(msg):
    print("[GeoProjection] %s" % msg)


# ---- ComfyBridge helpers ------------------------------------------------------------------

def _paste(text, beside=None, positions=None, select_beside=False, clear_selection=False):
    """Paste .nk text into the graph that contains `beside` (button callbacks of a Group run
    inside it). Costs one Node object when `beside` is inside another group.
    positions: {node name: (x, y)} applied by name (no Node objects).
    select_beside: make `beside` the only selected node so the paste connects to it.
    clear_selection: paste with nothing selected (the text connects itself)."""
    fd, path = tempfile.mkstemp(suffix=".nk", prefix="geoprojection_")
    os.close(fd)

    def _do():
        with open(path, "w") as f:
            f.write(text)
        if select_beside or clear_selection:
            nuke.selectAll()
            nuke.invertSelection()
        if select_beside:
            beside.setSelected(True)
        nuke.nodePaste(path)
        for name, (x, y) in (positions or {}).items():
            try:
                nuke.knob(name + ".xpos", str(int(x)))
                nuke.knob(name + ".ypos", str(int(y)))
            except Exception as e:
                _log("could not place %s: %s" % (name, e))

    try:
        if beside is not None:
            full = beside.fullName()
            ctx = nuke.toNode(full.rsplit(".", 1)[0]) if "." in full else nuke.root()
            with ctx:
                _do()
        else:
            _do()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _k(node, internal, knob, value=None):
    path = "%s.%s.%s" % (node.fullName(), internal, knob)
    return nuke.knob(path) if value is None else nuke.knob(path, str(value))


def _working_space():
    for k in ("workingSpaceLUT", "colorManagement_workingSpace"):
        try:
            v = nuke.knob("root." + k)
            if v:
                return v
        except Exception:
            pass  # expected: not every Nuke setup has it
    return "scene_linear"


def _splat_colorspace():
    """SplatRender output_colorspace for the script's working space. The splat's colours are
    linear Rec.709 (SHARP's linear output); SplatRender converts them to this."""
    ws = str(_working_space()).lower()
    for keys, cs in ((("acescg", "ap1"), "lin_ap1_scene"), (("aces2065", "ap0"), "lin_ap0_scene"),
                     (("rec2020",), "lin_rec2020_scene"), (("p3",), "lin_p3d65_scene")):
        if any(k in ws for k in keys):
            return cs
    return "lin_rec709_scene"


def _viewer_process():
    """Viewer transform of the active Viewer (costs one Node object in NC)."""
    name = None
    try:
        av = nuke.activeViewer()
        if av is not None:
            name = av.node().name()
    except Exception:
        name = None
    for n in ([name] if name else []) + ["Viewer1"]:
        try:
            return parse_viewer_process(nuke.knob(n + ".viewerProcess"))
        except Exception:
            continue
    return None


def _send_colour(node, dv, work_cs, available):
    """Plate colour for the PNG: viewer transform (OCIO display/view or Nuke LUT) or sRGB."""
    ocio_mode = dv is not None and dv[0] != LUT
    node["gp_in_ocio"].setValue(ocio_mode)
    node["gp_in_raw"].setValue(ocio_mode)
    if ocio_mode:
        _k(node, "GP_InDisplay", "colorspace", work_cs)
        _k(node, "GP_InDisplay", "display", dv[0])
        _k(node, "GP_InDisplay", "view", dv[1])
    elif dv is not None:
        _k(node, "GP_Write", "colorspace", dv[1])
    else:
        _k(node, "GP_Write", "colorspace", pick_colorspace(_CS_SRGB, available))


def _graph_prefix(beside):
    """'' for the root graph, 'Group1.' for nodes inside Group1."""
    full = beside.fullName()
    return full.rsplit(".", 1)[0] + "." if "." in full else ""


def _unique_name(base, beside):
    """base, or base_2, base_3... free in the graph that contains `beside`."""
    prefix = _graph_prefix(beside)
    if not nuke.exists(prefix + base):
        return base
    i = 2
    while nuke.exists("%s%s_%d" % (prefix, base, i)):
        i += 1
    return "%s_%d" % (base, i)


# ---- entry points -------------------------------------------------------------------------

def create_node():
    try:
        blender = find_blender()
    except ComfyError:
        blender = ""
    presets = list(list_presets())
    if DEFAULT_PRESET in presets:              # default first: an Enumeration starts there
        presets.insert(0, presets.pop(presets.index(DEFAULT_PRESET)))
    _paste(node_template(os.environ.get("GEO_PROJECTION_SERVER", "127.0.0.1:8188"),
                         nuke.frame(), presets, blender.replace("\\", "/")))


def _labels(node):
    if not node["segment"].value():
        return {}
    try:
        return effective_labels(node["label_preset"].value(), node["extra_labels"].value())
    except LabelError as e:
        raise ComfyError(str(e))


def show_labels(node):
    try:
        labels = _labels(node)
    except ComfyError as e:
        nuke.message(str(e))
        return
    if not labels:
        nuke.message("Semantic segmentation is off: the fit guesses floor / walls / boxes from geometry.")
        return
    by_role = {}
    for k, r in labels.items():
        base, _, shape = r.partition(":")
        by_role.setdefault(base, []).append("%s (%s)" % (k, shape) if shape else k)
    nuke.message("%d labels (%s + extra):\n\n%s" % (len(labels), node["label_preset"].value(), "\n".join(
        "%s: %s" % (r, ", ".join(by_role[r])) for r in ROLES if r in by_role)))


def test_connection(node):
    try:
        s = ComfyClient(node["server"].value()).system_stats()
        dev = s.get("devices", [{}])[0]
        nuke.message("Connected to ComfyUI %s\n%s\nVRAM: %.1f GB" % (
            s.get("system", {}).get("comfyui_version", ""), dev.get("name", "?"),
            dev.get("vram_total", 0) / (1024.0 ** 3)))
    except ComfyError as e:
        nuke.message(str(e))


def _default_out_dir(node):
    try:
        script = nuke.scriptName()  # raises on an unsaved script
    except Exception:
        script = ""
    base = os.path.dirname(script) if script else tempfile.gettempdir()
    return os.path.join(base, "geo_projection", node.name()).replace("\\", "/")


def run(node):
    try:
        _run(node)
    except ComfyError as e:
        node["gp_status"].setValue("error")
        nuke.message("GeoProjection:\n" + str(e))


def _camera_at(cam, frame):
    k = lambda name, i=None: cam[name].valueAt(frame) if i is None else cam[name].valueAt(frame, i)
    wm = [cam["world_matrix"].getValueAt(frame, i) for i in range(16)]
    r, t = rigid_from_world_matrix(wm)
    return {"node": cam, "name": cam.fullName(), "r": r, "t": t, "matrix": wm, "focal": k("focal"),
            "haperture": k("haperture"), "vaperture": k("vaperture"),
            "win_translate": [k("win_translate", 0), k("win_translate", 1)]}


PASSTHROUGH = ("Dot", "NoOp", "PostageStamp")
PASSTHROUGH_MAX = 20     # Node objects are scarce in Non-commercial: a long chain of Dots costs one each


def _shot_camera(node, frame, extra=()):
    """The camera connected to the 'cam' input, read at the reference frame (one Node object),
    or None. With extra frames, its knob values at each of them too ("extra": [(frame, values)])."""
    cam = node.input(1)
    if cam is None:
        return None
    link = cam.fullName()            # what the user wired in (maybe a Dot): the Groups connect to it
    for _ in range(PASSTHROUGH_MAX):
        # Through Dots and other pass-through nodes to the camera.
        if cam is None or cam.Class() not in PASSTHROUGH:
            break
        cam = cam.input(0)
    if cam is None or "world_matrix" not in cam.knobs() or "focal" not in cam.knobs():
        raise ComfyError("The 'cam' input must be a camera (got %s)." % (cam.Class() if cam else "nothing"))
    shot = _camera_at(cam, frame)
    shot["link"] = link
    shot["extra"] = []
    for f in extra:
        c = _camera_at(cam, f)
        shot["extra"].append((f, {"translate": c["t"], "rotate": to_euler_xyz(c["r"]), "rot_order": "XYZ",
                                  "focal": c["focal"], "haperture": c["haperture"],
                                  "vaperture": c["vaperture"], "win_translate": c["win_translate"]}))
    return shot


TRACK_POINTS_MIN = 8      # tracker points seen at the reference frame needed to fix the scale
TRACK_POINTS_BAD_SPREAD = 0.25   # flagged when the points' depth ratios spread more than this


def _points_of(src):
    """World positions of the points of the node on the 'points' input (one Node object):
    classic geometry (CameraTrackerPointCloud, ReadGeo2 of an exported cloud or locators, any
    node with a geometry selection) or a USD stage (GeoImport ...)."""
    for _ in range(PASSTHROUGH_MAX):
        if src is None or src.Class() not in PASSTHROUGH:
            break
        src = src.input(0)
    if src is None:
        return []
    if src.Class() == "CameraTracker":
        raise ComfyError("Connect the CameraTracker's point cloud to 'points', not the CameraTracker "
                         "itself: in the CameraTracker, Export: Point Cloud, then Create.")
    pts = []
    if "geo_select" in src.knobs():
        for obj in src["geo_select"].getGeometry():
            m = obj.transform()
            for p in obj.points():
                w = m.transform(p)
                pts.append((w.x, w.y, w.z))
        return pts
    stage = src.getStage() if hasattr(src, "getStage") else None
    if stage is not None:
        from pxr import Usd, UsdGeom
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointBased) or prim.GetTypeName() == "ParticleField3DGaussianSplat":
                a = prim.GetAttribute("points") or prim.GetAttribute("positions")
                vals = a.Get() if a else None
                if not vals:
                    continue
                m = cache.GetLocalToWorldTransform(prim)
                pts += [tuple(m.Transform(v)) for v in vals]
            elif prim.IsA(UsdGeom.Xformable) and not prim.GetChildren():
                # A locator (empty Xform): its position.
                pts.append(tuple(cache.GetLocalToWorldTransform(prim).ExtractTranslation()))
        return pts
    raise ComfyError("The 'points' input must be 3D points: a CameraTrackerPointCloud, a ReadGeo / "
                     "GeoImport of a point cloud, ... (got %s)." % src.Class())


SLIDE_FRAMES = 9         # frames across the plate where the slide is measured
SLIDE_BAD_PX = 2.0       # an object whose median slide is above this many px is flagged


def _slide_report(node, shot, tp, aspect):
    """How far the fitted geometry is from the tracker points once the camera moves: each point
    is put on the geometry along its reference-frame ray (fit's geo_ratio) and both are
    projected through the shot camera over the plate's range. -> {region: (points, median px,
    p90 px)} with the worst frame per point, and the frames used."""
    world, ratios = shot.get("track_world") or [], tp.get("geo_ratio") or []
    regions = (tp.get("snap") or {}).get("point_region") or [None] * len(ratios)
    if not world or len(ratios) != len(world):
        return {}, []
    first, last = int(node.firstFrame()), int(node.lastFrame())
    n = min(SLIDE_FRAMES, last - first + 1)
    frames = sorted({int(round(first + i * (last - first) / max(n - 1, 1))) for i in range(n)})
    cams = [_camera_at(shot["node"], f) for f in frames]
    W, H = float(node.width()), float(node.height())
    c0 = shot["t"]
    per = {}
    for p, q, reg in zip(world, ratios, regions):
        if not q:
            continue
        g = [c0[i] + (p[i] - c0[i]) * q for i in range(3)]
        worst = 0.0
        for c in cams:
            a = project_points([p, g], c["r"], c["t"], c["focal"], c["haperture"], c["win_translate"], aspect)
            if len(a) == 2:
                worst = max(worst, math.hypot((a[0][0] - a[1][0]) * W, (a[0][1] - a[1][1]) * H))
        per.setdefault(reg or "unlabelled", []).append(worst)
    out = {}
    for reg, v in per.items():
        v.sort()
        out[reg] = (len(v), v[len(v) // 2], v[min(len(v) - 1, int(0.9 * len(v)))])
    return out, frames


def _track_points(node, shot, aspect):
    """Tracker points on the 'points' input seen by the shot camera at the reference frame ->
    [(u, v, depth)], or None when nothing is connected."""
    try:
        src = node.input(2)
    except Exception:
        src = None                           # nodes made before the 'points' input
    if src is None:
        return None
    if not shot:
        raise ComfyError("'points' needs the tracked camera on the 'cam' input too.")
    pts = _points_of(src)
    seen, world = [], []
    for p in pts:
        uvd = project_points([p], shot["r"], shot["t"], shot["focal"], shot["haperture"],
                             shot["win_translate"], aspect)
        if uvd:
            seen.append(uvd[0])
            world.append(p)
    shot["track_world"] = world      # same order as the returned points: for the slide check
    _log("tracker points: %d, %d seen at the reference frame" % (len(pts), len(seen)))
    if len(seen) < TRACK_POINTS_MIN:
        raise ComfyError("Only %d of the %d tracker points on 'points' are in view at the reference "
                         "frame (need %d). Check the point cloud belongs to this camera, or "
                         "disconnect 'points'." % (len(seen), len(pts), TRACK_POINTS_MIN))
    return seen


SCALE_BAD_PX = 0.005    # track scale flagged when its median reprojection error is over this x width
SCALE_BAD_SPREAD = 1.1   # ...or when the frames disagree by more than this ratio
SCALE_SAMPLES = 50      # frames of the plate looked at to pick the one for the scale solve


def _scale_frames(node, shot, frame, extra):
    """Frames whose plate + tracked camera fix the scale between MoGe and the track: the extra
    frames, else the frame of the plate where the camera is farthest from the reference one.
    [] when the camera does not move (a nodal pan: the scale changes nothing then).
    -> ([frames], SolveTrackScale's cameras dict)."""
    frames = list(extra)
    if not frames:
        first, last = int(node.firstFrame()), int(node.lastFrame())
        step = max(1, (last - first) // SCALE_SAMPLES)
        dist = lambda f: sum((a - b) ** 2 for a, b in zip(_camera_at(shot["node"], f)["t"], shot["t"])) ** 0.5
        far = max(range(first, last + 1, step), key=dist, default=frame)
        if far != frame and dist(far) > 1e-6:
            frames = [far]
    cams = {"ref": frame, "frames": {}}
    for f in [frame] + frames:
        c = _camera_at(shot["node"], f)
        cams["frames"][str(f)] = {"matrix": c["matrix"], "focal": c["focal"], "haperture": c["haperture"]}
    return frames, cams


def _extra_frames(node):
    """The 'extra frames' knob as ints in the order typed, without the reference frame or repeats."""
    if "proj_frames" not in node.knobs():
        return []
    ref, out = int(node["ref_frame"].value()), []
    for tok in node["proj_frames"].value().replace(",", " ").split():
        try:
            f = int(float(tok))
        except ValueError:
            raise ComfyError("'extra frames' takes frame numbers separated by spaces (got '%s')." % tok)
        if f != ref and f not in out:
            out.append(f)
    return out


def add_frame(node):
    """'add current frame': append the Viewer's frame to 'extra frames'."""
    frames = [str(f) for f in _extra_frames(node)]
    f = str(int(nuke.frame()))
    if f == str(int(node["ref_frame"].value())):
        nuke.message("GeoProjection: frame %s is the reference frame (always projected, on top)." % f)
        return
    if f not in frames:
        frames.append(f)
    node["proj_frames"].setValue(" ".join(frames))


def _geometry(node):
    """(geometry, step) from the node; nodes made before 0.5 have a detail level instead."""
    if "geometry" in node.knobs():
        return node["geometry"].value(), int(node["precision"].value()) or None
    return {"low": ("boxes", 6), "medium": ("boxes", 4), "high": ("relief", 3)}[node["detail"].value()]


def _run(node):
    frame = int(node["ref_frame"].value())
    extra = _extra_frames(node)
    shot = _shot_camera(node, frame, extra)
    if extra and not shot:
        raise ComfyError("'extra frames' needs the tracked shot camera on the 'cam' input: without "
                         "it the camera at those frames is unknown.\nConnect it, or clear 'extra frames'.")
    # Nodes made before the 3D system menu existed keep the classic setup.
    system = node["system"].value() if "system" in node.knobs() else "classic"
    want_splat = "splat" in node.knobs() and bool(node["splat"].value())
    if want_splat and system != "USD":
        raise ComfyError("'Gaussian splat' needs the USD 3D system (the classic 3D system has no "
                         "splats).\nSet '3D system' to USD, or turn the splat off.")
    focal, haperture = float(node["focal"].value()), float(node["haperture"].value())
    if shot:
        focal, haperture = shot["focal"], shot["haperture"]
        _log("lens from %s at frame %d: %.2f mm / %.2f mm" % (shot["name"], frame, focal, haperture))
    labels = _labels(node)
    dv = None
    if node["in_color"].value() == IN_VIEWER:
        dv = _viewer_process()
        if dv is None:
            raise ComfyError("Cannot read the Viewer transform.\nOpen a Viewer or set 'input color' to sRGB.")
    _send_colour(node, dv, _working_space(), ocio_names())

    out_dir = next_version_dir(node["out_dir"].evaluate() or _default_out_dir(node))
    os.makedirs(out_dir)
    plate_png = os.path.join(out_dir, "plate.%04d.png" % frame).replace("\\", "/")
    node["gp_in_file"].setValue(plate_png.replace("%04d" % frame, "%04d"))
    par = float(node.pixelAspect() or 1.0)
    if "gp_par" in node.knobs():
        # Non-square pixels (DVCPRO HD, anamorphic): the frames are written stretched to square
        # pixels, as MoGe / SHARP expect; the camera maths below use the display aspect.
        node["gp_par"].setValue(par)
    elif abs(par - 1.0) > 1e-6:
        raise ComfyError("The plate has non-square pixels (pixel aspect %.3g) and this GeoProjection "
                         "node is older than v0.8: create a new one." % par)
    aspect = node.width() * par / float(node.height())
    if abs(par - 1.0) > 1e-6:
        _log("pixel aspect %.3g: frames sent with square pixels (%dx%d display)"
             % (par, round(node.width() * par), node.height()))

    task = nuke.ProgressTask("GeoProjection")
    try:
        task.setMessage("Writing frame %d" % frame)
        try:
            nuke.execute("%s.GP_Write" % node.fullName(), frame, frame, 1)
            why = ""
        except RuntimeError as e:
            why = str(e)
        if not os.path.isfile(plate_png):
            src = node.input(0)
            chans = set(c.split(".")[-1] for c in (src.channels() if src else []))
            hint = ""
            if src is None:
                hint = "Nothing is connected to the plate input."
            elif not {"red", "green", "blue"} <= chans:
                hint = ("The plate input (%s, a %s) has no rgb at frame %d. Connect the plate itself "
                        "(the Read, or the plate after an undistort), not a render of the geometry."
                        % (src.fullName(), src.Class(), frame))
            raise ComfyError("The plate frame %d was not written:\n%s\n%s%s" % (
                frame, plate_png, hint, ("\n" + why) if why else ""))
        track_points = _track_points(node, shot, aspect)
        # The tracker's own 3D points fix the scale directly; without them it is solved from
        # matches between frames (SolveTrackScale).
        scale_frames, cams = (_scale_frames(node, shot, frame, extra) if shot and not track_points
                              else ([], None))
        for f in scale_frames:
            # The tracked camera's scale is solved from these frames (and, in sequence mode,
            # they are the ones projected): written like the reference frame.
            task.setMessage("Writing frame %d" % f)
            nuke.execute("%s.GP_Write" % node.fullName(), f, f, 1)
        scale_frames = [(f, plate_png.replace("%04d" % frame, "%04d" % f)) for f in scale_frames]
        steps = iter((20, 70))

        def on_step(s):
            task.setMessage(s)
            task.setProgress(next(steps, 90))

        principal = None
        if shot and any(abs(w) > 1e-9 for w in shot["win_translate"]):
            # Nuke's screen window spans 2 units across the width (and width/height up and
            # down): a window offset moves the principal point off the image centre.
            u, v = shot["win_translate"]
            principal = (0.5 - u / 2.0, 0.5 + v / 2.0 * aspect)
            _log("principal point from %s window offset: %.4f %.4f" % (shot["name"], principal[0], principal[1]))
        res = build(plate_png, out_dir, node["server"].value(), focal, haperture,
                    float(node["camera_height"].value()), node["blender"].evaluate(), labels,
                    _geometry(node), principal, is_cancelled=task.isCancelled, log=_log,
                    on_step=on_step, scale_frames=scale_frames, cameras=cams, splat=want_splat,
                    track_points=track_points)
    finally:
        del task

    fit = res["fit"]
    fit_cam = fit["nuke_camera"]
    cam = dict(fit_cam)
    if shot:
        # Project from the shot camera as it is at the reference frame; the geometry is moved
        # into its space by the TransformGeo.
        cam.update(translate=shot["t"], rotate=to_euler_xyz(shot["r"]), rot_order="XYZ",
                   focal=shot["focal"], haperture=shot["haperture"], vaperture=shot["vaperture"],
                   win_translate=shot["win_translate"])
        ra, ra_ft, shot_t = alignment(fit_cam["translate"], fit_cam["rotate"], shot["r"], shot["t"])
        align = {"rotate": to_euler_xyz(ra), "ra_ft": ra_ft, "shot_t": shot_t}
    else:
        if focal > 0:
            # MoGe was given this lens: keep the user's film back so the knobs read as entered.
            cam["focal"], cam["haperture"] = focal, haperture
            cam["vaperture"] = haperture * fit["camera"]["height"] / fit["camera"]["width"]
        align = {"rotate": [0.0, 0.0, 0.0], "ra_ft": fit_cam["translate"], "shot_t": fit_cam["translate"]}
    v = fit["verification"]
    version = os.path.basename(out_dir)
    by_shape = {}
    for p in fit["primitives"]:
        k = p.get("shape") or {"mesh": "relief mesh"}.get(p["type"], p["type"])
        by_shape[k] = by_shape.get(k, 0) + 1
    shapes = ", ".join("%d %s" % (n, k) for k, n in sorted(by_shape.items(), key=lambda kv: -kv[1]))
    info = "%s: %s | lens %.1f mm | error %.2f%% | coverage %.0f%%" % (
        version, shapes, cam["focal"], 100 * (v["abs_rel_median"] or 0), 100 * v["coverage"])
    if "labelled_coverage" in v:
        info += " (labelled areas %.0f%%)" % (100 * v["labelled_coverage"])
    if extra:
        info += " | frames %s" % " > ".join(str(f) for f in [frame] + extra)
    scene_scale = 1.0
    solved = res["geo"].get("scale") or {}
    tp = fit.get("track_points") or {}
    if tp.get("scale"):
        # Track units per MoGe unit from the tracker's points; the fit may have rescaled MoGe.
        scene_scale = tp["scale"] / (fit.get("units_scale") or 1.0)
        info += " | track scale %.3g from %d tracker points (spread %.0f%%)" % (
            scene_scale, tp["used"], 100 * tp["spread"])
        if tp["spread"] > TRACK_POINTS_BAD_SPREAD:
            info += " UNRELIABLE: the points disagree with MoGe's depth - check 'scene scale'"
        snap = tp.get("snap") or {}
        if snap:
            fixed = [e for e in snap["regions"] if "after" in e]
            info += " | snapped to tracker: %d regions (%.0f%% of the frame)" % (
                len(fixed), 100 * snap["corrected_share"])
        slide, frames = _slide_report(node, shot, tp, aspect)
        if slide:
            rows = sorted(slide.items(), key=lambda kv: -kv[1][0])
            _log("slide vs tracker points over frames %s (median / p90 px, worst frame per point):" % frames)
            for reg, (k, med, p90) in rows:
                _log("  %-24s %4d pts  %6.2f / %6.2f px%s" % (reg, k, med, p90,
                                                             "  SLIDES" if med > SLIDE_BAD_PX else ""))
            main = [(reg, v) for reg, v in rows if reg != "unlabelled"][:3]
            info += " | slide " + ", ".join("%s %.1f px" % (reg, v[1]) for reg, v in main)
            bad = [reg for reg, v in rows if reg != "unlabelled" and v[1] > SLIDE_BAD_PX and v[0] >= 5]
            if bad:
                info += " (sliding: %s)" % ", ".join(bad[:4])
    elif solved.get("scale"):
        # Track units per fit unit: the solve is per MoGe unit, the fit may have rescaled MoGe.
        scene_scale = solved["scale"] / (fit.get("units_scale") or 1.0)
        info += " | track scale %.3g (%d pts, %.1f px)" % (scene_scale, solved["matches"], solved["median_px"])
        per = solved.get("per_frame") or {}
        spread = [p["scale"] for p in per.values()]
        if (solved["median_px"] > SCALE_BAD_PX * fit["camera"]["width"]
                or (len(spread) > 1 and max(spread) > SCALE_BAD_SPREAD * min(spread))):
            info += " UNRELIABLE: per frame %s - check 'scene scale', or use extra frames nearer the reference" % (
                ", ".join("%s: %.3g" % (f, p["scale"] / (fit.get("units_scale") or 1.0)) for f, p in sorted(per.items())))
    elif shot and scale_frames:
        info += " | track scale not solved (too few matches): set 'scene scale' by hand"
    if res.get("splat"):
        sp = fit.get("splat") or {}
        info += " | splat %.1fM gaussians" % (sp.get("gaussians", 0) / 1e6)
    _log(info)

    group = _unique_name("GeoProj_" + version, node)
    text = setup_template(group, frame, res["abc"], cam,
                          [(p.get("group", "objects"), p["name"]) for p in fit["primitives"]], align,
                          use_shot_cam=bool(shot), label="%s %s %s%s%s" % (
                              version, _geometry(node)[0], system, " +%d fr" % len(extra) if extra else "",
                              " +splat" if res.get("splat") else ""),
                          info="%s | %s" % (info, res["abc"]), system=system, geometry=_geometry(node)[0],
                          plate="root." + node.fullName(), shot_cam=shot and "root." + shot["link"],
                          extra=shot["extra"] if shot else (), scale=scene_scale,
                          splat=res.get("splat"), splat_colorspace=_splat_colorspace())
    x, y = int(node["xpos"].value()), int(node["ypos"].value())
    # Connected by name in the text, pasted with nothing selected: pasting onto the selected
    # gizmo would rewire the Groups of earlier runs below it to this one.
    _paste(text, beside=node, positions={group: (x, y + SETUP_DY)}, clear_selection=True)
    node["gp_status"].setValue(version)
    node["res_info"].setValue(info)


EXTRACT_DX, EXTRACT_STEP = 200, 640    # first backdrop right of the Group; each next one further


def _abc_order(abc, prims):
    """prims in the Alembic's own order. A classic ReadGeo2 re-lists its items in that order
    when it validates and keeps the selected indices, so a partial selection written in any
    other order picks the wrong objects. Read with Nuke's USD (pxr opens .abc)."""
    try:
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.Open(abc)
        order = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    except Exception as e:
        _log("could not read the Alembic's order (%s): the classic extract may pick other objects" % e)
        return prims
    pos = {path: i for i, path in enumerate(order)}
    return sorted(prims, key=lambda gn: pos.get("/%s/%s/%s" % (gn[0], gn[1], gn[1]), len(pos)))


def isolate_classic(group):
    """Classic setup Group: 'show only' / 'pattern' -> the ReadGeo2 'Geo' imports everything but
    only outputs the picked objects (its selected items), so projection and render show only
    them. Knob paths only, no Node objects (Non-commercial). The items are read back from the
    knob: ReadGeo2 lists them in the Alembic's order, which the indices must follow."""
    path = group + ".Geo.scene_view"
    sv = nuke.knob(path)
    if not sv or "items:" not in sv or "selected:" not in sv:
        return
    head, items = sv.split("items:", 1)
    head = head.split("selected:", 1)[0]
    items = items.split()
    prims = []
    for it in items:                         # /root/<group>/<name>/<name>
        parts = it.strip("/").split("/")
        prims.append((parts[1], parts[2]) if len(parts) >= 3 else (parts[-1], parts[-1]))
    picked = select_prims(prims, nuke.knob(group + ".show_only"), nuke.knob(group + ".show_pattern"))
    keep = [i for i, p in enumerate(prims) if not picked or p in picked]
    nuke.knob(path, "%sselected: %s items: %s" % (head, " ".join(str(i) for i in keep), " ".join(items)))


def extract(group):
    """'extract for paint' on a setup Group: paste, in a backdrop to the right of the Group,
    loose nodes projecting onto what 'show only' / 'pattern' picks (templates.extract_template),
    fed by the same plate and camera (connected by name in the pasted text). Each extract
    goes one backdrop further right. Costs two Node objects (the Group, the graph it is in)."""
    if "gp_setup" not in group.knobs():
        nuke.message("GeoProjection: this setup was built before v0.6.\nBuild it again to extract objects.")
        return
    setup = json.loads(group["gp_setup"].getText())      # value() would evaluate the [ ] as TCL
    prims = [tuple(p) for p in setup["prims"]]
    if setup["system"] == "classic":
        prims = setup["prims"] = _abc_order(setup["abc"], prims)
    pattern = group["show_pattern"].value().strip()
    chosen = select_prims(prims, group["show_only"].value(), pattern)
    if not chosen:
        nuke.message("GeoProjection: pick an object or group in 'show only' (or type a 'pattern') first.")
        return
    if not pattern:
        pattern = "/" + group["show_only"].value()
    tag = chosen[0][1] if len(chosen) == 1 else (pattern.split()[0].strip("/*").replace("/", "_") or "sel")
    tag = re.sub(r"\W+", "_", tag)
    prefix, p, i = _graph_prefix(group), tag, 1
    while nuke.exists("%s%s_Render" % (prefix, p)):
        i += 1
        p = "%s_%d" % (tag, i)
    full = "root." + group.fullName()

    def source(i):
        # The node on input i, by name through TCL: no Node object.
        n = nuke.tcl("full_name [input %s %d]" % (full, i))
        return None if n in ("", "0") else "root." + n

    title = "geoproject %s frame %03d" % (tag, setup["ref"])
    text, offsets = extract_template(p, setup, chosen, title, scale=group["scene_scale"].value(),
                                     plate=source(0), cam=source(1) if setup["use_shot_cam"] else None)
    n = int(group["gp_extracts"].value()) if "gp_extracts" in group.knobs() else 0
    x, y = int(group["xpos"].value()) + EXTRACT_DX + n * EXTRACT_STEP, int(group["ypos"].value()) - 100
    _paste(text, beside=group, positions={k: (x + dx, y + dy) for k, (dx, dy) in offsets.items()},
           clear_selection=True)
    if "gp_extracts" in group.knobs():
        group["gp_extracts"].setValue(n + 1)
    _log("extracted %s: %s" % (p, " ".join(n for _, n in chosen)))


ICON = "geo_projection.png"   # next to geo_projection.py (in the plugin path), + @2x


def install_menu():
    """DBS_GeoProjection under the DBS button in the Nodes toolbar (David Borrull Silva's tools),
    and in Nodes > 3D."""
    cmd = "import geo_projection; geo_projection.create_node()"
    tb = nuke.menu("Nodes").addMenu("DBS", icon=ICON)
    tb.addCommand("DBS_GeoProjection", cmd, icon=ICON)
    nuke.menu("Nodes").addMenu("3D").addCommand("DBS_GeoProjection", cmd, icon=ICON)
