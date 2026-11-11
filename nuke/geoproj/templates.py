# GeoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
""".nk templates: pasting them creates nodes without Python Node objects
(Nuke Non-commercial lets Python touch only 10 per session). Pattern from ComfyBridge."""

import fnmatch
import json

from . import __version__
from .colour import IN_SRGB, IN_VIEWER

GEOMETRY_TYPES = ("shapes", "boxes", "cards", "relief")
# Node colour per geometry type (gizmo, live; setup Groups, as built): teal, orange, blue, purple.
GEOMETRY_COLORS = {"shapes": 0x3f7f6fff, "boxes": 0xb0702aff, "cards": 0x3a6ab0ff, "relief": 0x7a4a9aff}
# Gizmo knobChanged: recolour the node when its geometry menu changes (path only, no Node object).
TINT_CALLBACK = (
    "def _geoproj_tint(k):\n"
    "    if k.name() != 'geometry':\n"
    "        return\n"
    "    colors = (%s)\n"
    "    types = (%s)\n"
    "    if k.value() in types:\n"
    "        g = k.fullyQualifiedName().rsplit('.', 1).pop(0)\n"
    "        nuke.knob(g + '.tile_color', str(colors[types.index(k.value())]))\n"
    "_geoproj_tint(nuke.thisKnob())\n"
    ) % (", ".join(str(GEOMETRY_COLORS[t]) for t in GEOMETRY_TYPES),
         ", ".join("'%s'" % t for t in GEOMETRY_TYPES))


def nk_quote(s):
    # Copied from ComfyBridge (MIT, David Borrull).
    return '"%s"' % str(s).replace("\\", "/").replace('"', "").replace("[", "\\[").replace("]", "\\]")


def _py(cmd):
    return nk_quote("import geo_projection; geo_projection.%s(nuke.thisNode())" % cmd)


TIPS = {
    "server": "ComfyUI address (host:port) with MoGe and the nuke_geo_projection nodes installed.",
    "ref_frame": "Frame of the plate that is analysed and projected (a FrameHold keeps it).",
    "in_color": "viewer transform: send what you see in the Viewer (MoGe expects a display image). "
                "sRGB: always plain sRGB.",
    "work_size": "Long edge in px sent to ComfyUI (0 = original). The camera is unaffected.",
    "cam_info": "Connect your shot camera to the 'cam' input: its lens at the reference frame is "
                "used, the geometry is placed in its space and the render follows it. Connect the "
                "tracker's 3D points to 'points' (CameraTrackerPointCloud, or a ReadGeo / GeoImport "
                "of the point cloud or locators from 3DEqualizer, SynthEyes ...): they fix the scale "
                "of the geometry exactly. Without them the scale is solved from other frames of the "
                "plate.",
    "proj_frames": "Sequence mode (needs the shot camera on the 'cam' input): more frames of the "
                   "plate projected onto the same geometry, e.g. '25 50'. The reference frame stays "
                   "on top; each extra frame, in this order, fills what the frames before it did "
                   "not see (each projection is occluded by the geometry). The geometry still "
                   "comes from the reference frame only.",
    "focal": "Known focal length in mm (without a 'cam' input). 0 = let MoGe estimate it "
             "(expect ~10-20% error).",
    "haperture": "Film back width in mm that goes with the focal.",
    "camera_height": "Camera height above the floor in scene units (e.g. 1.6 m). 0 = MoGe's own units.",
    "segment": "Label the image with SAM3 first, so floor, walls and objects come from the labels "
               "instead of being guessed from geometry. Needs comfyui-rmbg (SAM3) in ComfyUI.",
    "label_preset": "Label library: text files in geoproj/label_presets and "
                    "~/.nuke/geo_projection_labels (one text:role per line).",
    "extra_labels": "Your own labels on top of the preset, one per line, text:role. Roles: floor, "
                    "wall, ceiling, object (a box per instance), backdrop (far cards / plane: trees, "
                    "hills), ignore (no geometry). A label already in the preset takes this role.",
    "show_labels": "Show the labels that will be sent (preset + extra).",
    "geometry": "What each object becomes. shapes: the best fitting primitive per object (box, "
                "cylinder, cone, sphere or revolve for vases and lamps; force one per label with "
                "text:object:shape). boxes: a box each. cards: a camera-facing card each. relief: "
                "MoGe's own surface as meshes (heavy). Floor / walls / ceiling / backdrop are "
                "planes in all of them.",
    "precision": "Sample spacing in pixels used to fit (and the relief mesh's vertex spacing). "
                 "Smaller = more detail and heavier; 4 is a good default, 2-3 for relief.",
    "system": "Which Nuke 3D system the projection setup is built with. USD: GeoImport, "
              "Project3DShader + GeoBindMaterial, GeoTransform, GeoScene, Camera4, ScanlineRender2. "
              "classic: ReadGeo2, Project3D2, TransformGeo, Scene, Camera3, ScanlineRender.",
    "splat": "Also make a Gaussian splat of the plate with SHARP (comfyui-sharp in ComfyUI), in the "
             "same space as the geometry. The setup Group then renders it with SplatRender and its "
             "'output' menu picks projection, splat, or projection over splat (the splat fills "
             "what the geometry misses). USD 3D system only.",
    "blender": "blender executable. Empty = newest Blender found in Program Files.",
    "out_dir": "Where each run is written (vNNN subfolders). Empty = next to the script.",
    "run": "Write the frame, estimate geometry in ComfyUI, fit planes / boxes in Blender and "
           "paste the projection setup next to this node as one Group (GeoProj_vNNN) with a live "
           "'scene scale' and a 'cam' input for the shot camera.",
}


def _tip(name):
    return (" t " + nk_quote(TIPS[name])) if name in TIPS else ""


def node_template(server, ref_frame, presets, blender=""):
    # The frame sent to ComfyUI has square pixels (MoGe / SHARP assume them): the display width
    # is width * pixel aspect (gp_par, set by run() from the plate), then the long edge is
    # brought down to 'working size'.
    wd = "Input1.width*parent.gp_par"
    long_edge = "max(%s,Input1.height)" % wd
    s = "(parent.work_size>0 && %s>parent.work_size ? parent.work_size/%s : 1)" % (long_edge, long_edge)
    W = "floor(%s*%s/2+0.5)*2" % (wd, s)
    H = "floor(Input1.height*%s/2+0.5)*2" % s
    resize_off = "parent.gp_par==1 && (parent.work_size<=0 || %s<=parent.work_size)" % long_edge
    t = {k: _tip(k) for k in TIPS}
    return """set cut_paste_input [stack 0]
push $cut_paste_input
Group {
 name GeoProjection1
 tile_color %(tile)d
 knobChanged %(tint)s
 label "\\[value gp_status]\\n\\[value geometry] \\[value system]"
 addUserKnob {20 geoproj l GeoProjection}
 addUserKnob {1 server l server%(server_t)s}
 server %(server)s
 addUserKnob {22 test l "test connection" -STARTLINE T %(py_test)s}
 addUserKnob {26 div_plate l plate}
 addUserKnob {3 ref_frame l "reference frame"%(ref_frame_t)s}
 ref_frame %(ref)d
 addUserKnob {22 set_ref l "set to current frame" -STARTLINE T %(py_set_ref)s}
 addUserKnob {4 in_color l "input color" M {%(in_viewer)s %(in_srgb)s}%(in_color_t)s}
 addUserKnob {3 work_size l "working size"%(work_size_t)s}
 work_size 1920
 addUserKnob {26 div_cam l camera}
 addUserKnob {26 cam_info l "" +STARTLINE T "<i>shot camera to 'cam', its tracker points to 'points'</i>"%(cam_info_t)s}
 addUserKnob {1 proj_frames l "extra frames"%(proj_frames_t)s}
 addUserKnob {22 add_frame l "add current frame" -STARTLINE T %(py_add)s}
 addUserKnob {7 focal l "focal (mm, 0 = estimate)" R 0 200%(focal_t)s}
 addUserKnob {7 haperture l "film back width (mm)" R 1 60%(haperture_t)s}
 haperture 36
 addUserKnob {7 camera_height l "camera height (0 = auto)" R 0 10%(camera_height_t)s}
 addUserKnob {26 div_seg l labels}
 addUserKnob {6 segment l "semantic segmentation (SAM3)" +STARTLINE%(segment_t)s}
 segment true
 addUserKnob {4 label_preset l "label preset" M {%(presets)s}%(label_preset_t)s}
 addUserKnob {22 show_labels l "show labels" -STARTLINE T %(py_show)s%(show_labels_t)s}
 addUserKnob {52 labels_ui l "extra labels" +STARTLINE T "__import__('geoproj.widgets', fromlist=\\['LabelsEditor'\\]).LabelsEditor()"%(extra_labels_t)s}
 addUserKnob {1 extra_labels +INVISIBLE}
 addUserKnob {26 div_build l geometry}
 addUserKnob {4 geometry l geometry M {%(geometries)s}%(geometry_t)s}
 addUserKnob {3 precision l "precision (px)" -STARTLINE%(precision_t)s}
 precision 4
 addUserKnob {4 system l "3D system" M {%(systems)s}%(system_t)s}
 addUserKnob {6 splat l "Gaussian splat (SHARP)" +STARTLINE%(splat_t)s}
 addUserKnob {26 div_out l output}
 addUserKnob {2 out_dir l "output folder"%(out_dir_t)s}
 addUserKnob {2 blender l blender%(blender_t)s}
 blender %(blender)s
 addUserKnob {26 div_run l ""}
 addUserKnob {22 run l "<b>  Build camera + geometry  </b>" +STARTLINE T %(py_run)s%(run_t)s}
 addUserKnob {26 res_info l "" +STARTLINE T " "}
 addUserKnob {26 ver l "" +STARTLINE T "<small>GeoProjection v%(version)s  -  by David Borrull</small>"}
 addUserKnob {1 gp_in_file +INVISIBLE}
 addUserKnob {6 gp_in_ocio +INVISIBLE}
 addUserKnob {6 gp_in_raw +INVISIBLE}
 addUserKnob {1 gp_status +INVISIBLE}
 addUserKnob {7 gp_par +INVISIBLE}
 gp_par 1
}
 Input {
  inputs 0
  name Input1
  xpos 0
  ypos -300
 }
set N_in [stack 0]
 OCIODisplay {
  name GP_InDisplay
  disable {{!parent.gp_in_ocio}}
  xpos 150
  ypos -260
 }
 Reformat {
  type "to box"
  box_width {{"%(W)s"}}
  box_height {{"%(H)s"}}
  box_fixed true
  box_pixel_aspect 1
  resize distort
  filter Cubic
  name GP_InResize
  disable {{"%(resize_off)s"}}
  xpos 150
  ypos -220
 }
 Write {
  file "\\[value parent.gp_in_file]"
  raw {{parent.gp_in_raw}}
  file_type png
  datatype "8 bit"
  create_directories true
  name GP_Write
  xpos 150
  ypos -170
 }
 Input {
  inputs 0
  name cam
  number 1
  xpos 300
  ypos -300
 }
 Input {
  inputs 0
  name points
  number 2
  xpos 450
  ypos -300
 }
push $N_in
 Output {
  name Output1
  xpos 0
  ypos -100
 }
end_group
""" % dict({k + "_t": v for k, v in t.items()}, **{
        "server": nk_quote(server), "ref": int(ref_frame), "blender": nk_quote(blender), "W": W, "H": H,
        "resize_off": resize_off,
        "in_viewer": nk_quote(IN_VIEWER), "in_srgb": nk_quote(IN_SRGB),
        "presets": " ".join(nk_quote(p) for p in presets), "geometries": " ".join(GEOMETRY_TYPES),
        "systems": " ".join(SYSTEMS),
        "py_test": _py("test_connection"), "py_run": _py("run"), "py_show": _py("show_labels"),
        "py_add": _py("add_frame"),
        "py_set_ref": nk_quote("nuke.thisNode()['ref_frame'].setValue(int(nuke.frame()))"),
        "version": __version__,
        "tile": GEOMETRY_COLORS["shapes"], "tint": _nk_string(TINT_CALLBACK),
    })


def _scene_view(items, keep=None):
    """ReadGeo2 scene_view listing every Alembic item, with those in `keep` (all by default)
    imported and selected. The indices refer to this full list. Left to itself a pasted
    ReadGeo2 sets this during validate and may keep only one object."""
    every = "".join("\n  %d" % i for i in range(len(items)))
    idx = "".join("\n  %d" % i for i, it in enumerate(items) if keep is None or it in keep)
    return " scene_view {\n  {0}\n  imported:%s\n  selected:%s\n  items:%s\n }" % (
        every, idx, "".join("\n  %s" % it for it in items))


# Node classes per 3D system. Classic = Nuke's original 3D; USD = the USD-based 3D system
# (Nuke 15+), where projection is a Project3DShader material bound with GeoBindMaterial.
CLASSES = {
    "classic": {"camera": "Camera3", "project": "Project3D2", "geo": "ReadGeo2", "xform": "TransformGeo",
                "scene": "Scene", "render": "ScanlineRender"},
    "USD": {"camera": "Camera4", "project": "Project3DShader", "geo": "GeoImport", "xform": "GeoTransform",
            "scene": "GeoScene", "render": "ScanlineRender2"},
}
SYSTEMS = ("USD", "classic")


# Material path Project3DShader gives its material in the USD stage (it follows the node's own
# name; inside the setup Group that name is fixed, so copying / renaming the Group is safe).
MATERIAL_PATH = "/materials/NukeMaterialOps/%s_NdkSurfaceShader"

# knobChanged of the setup Group (USD): 'show only' / 'pattern' rewrite the Isolate node's mask.
# Nuke re-cooks a geo node when its knob text changes, so an expression in the mask would not
# update live. Paths only (nuke.knob), no Node objects: Non-commercial's 10 stay free.
# Wrapped in a function: knobChanged code runs in a namespace shared with other scripts.
ISOLATE_CALLBACK = (
    "def _geoproj_isolate(k):\n"
    "    if k.name() not in ('show_only', 'show_pattern'):\n"
    "        return\n"
    "    g = k.fullyQualifiedName().rsplit('.', 1).pop(0)\n"
    "    pat = ' '.join(t if t.startswith('/') else '//' + t for t in nuke.knob(g + '.show_pattern').split())\n"
    "    sel = nuke.knob(g + '.show_only')\n"
    "    mask = pat or ('' if sel == 'all' else '/' + sel)\n"
    "    nuke.knob(g + '.Isolate.mask_patterns', mask or '/*')\n"
    "    nuke.knob(g + '.Isolate.disable', 'false' if mask else 'true')\n"
    "_geoproj_isolate(nuke.thisKnob())\n")
# Classic: the same knobs rewrite the ReadGeo2's selected items (nuke_ui.isolate_classic).
ISOLATE_CALLBACK_CLASSIC = (
    "def _geoproj_isolate_classic(k):\n"
    "    if k.name() not in ('show_only', 'show_pattern'):\n"
    "        return\n"
    "    import geo_projection\n"
    "    geo_projection.isolate_classic(k.fullyQualifiedName().rsplit('.', 1).pop(0))\n"
    "_geoproj_isolate_classic(nuke.thisKnob())\n")


def _nk_string(s):
    """A Python snippet as a .nk quoted string: newlines as \\n, quotes and brackets escaped."""
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("[", "\\[").replace("]", "\\]")
    return '"%s"' % s.replace("\n", "\\n")


def select_prims(prims, show_only, pattern):
    """The [(group, name)] that 'show only' / 'pattern' pick, as the Isolate node would show
    them (a path picks its whole subtree). [] when they pick everything ('all', no pattern)."""
    tokens = pattern.split() or ([] if show_only in ("", "all") else ["/" + show_only])
    out = []
    for g, n in prims:
        paths = ("/%s" % g, "/%s/%s" % (g, n), "/%s/%s/%s" % (g, n, n))
        for t in tokens:
            pat = t if t.startswith("/") else "*/" + t
            if any(fnmatch.fnmatchcase(p, pat) for p in paths):
                out.append((g, n))
                break
    return out


def _lighter(rgba, k=0.45):
    """A tile colour mixed k of the way to white (backdrops read better light)."""
    c = [(rgba >> s) & 0xff for s in (24, 16, 8)]
    r, g, b = (int(v + (255 - v) * k) for v in c)
    return (r << 24) | (g << 16) | (b << 8) | 0xff


def extract_template(p, setup, prims, title, scale=1.0, plate=None, cam=None):
    """Loose nodes in a backdrop to paint on `prims` (David's layout, as he builds it by hand):

        plate -> FrameHold -> RotoPaint (paint here) -> Project <- FrameHold <- camera
        -> geometry (only `prims`) -> transform -> ScanlineRender, cam: the camera

    The camera is the shot camera when the setup used one (a Dot linked to it: the
    projection holds it at the reference frame, the render follows it), else the baked
    projection camera. Classic: ReadGeo2 (its 'img' input takes the projection) ->
    TransformGeo -> ScanlineRender. USD: GeoImport -> GeoBindMaterial (Project3DShader) ->
    GeoTransform -> GeoVisibility (only `prims`) -> GeoScene -> ScanlineRender2; the material
    path follows the Project node's name, so renaming that node breaks the bind.
    p: node name prefix ('lamp_1' -> lamp_1_Hold, lamp_1_Paint...). setup: the source Group's
    gp_setup. scale: its scene scale, baked into the transform. plate / cam: full names to
    connect to by 'push [node ...]'; paste with nothing selected.
    -> (.nk text, {node name: (dx, dy)} offsets from the backdrop's top left corner)."""
    s = setup
    usd = s["system"] == "USD"
    c = CLASSES[s["system"]]
    shot = s["use_shot_cam"] and cam
    vec = lambda v: "{%s}" % " ".join("%.6g" % x for x in v)
    al, k = s["align"], s["cam"]
    xlate = [st - scale * rf for st, rf in zip(al["shot_t"], al["ra_ft"])]
    paths = " ".join("/%s/%s" % gn for gn in prims)
    v = dict(p=p, ref=int(s["ref"]), abc=nk_quote(s["abc"]), cam_cls=c["camera"], proj_cls=c["project"],
             geo_cls=c["geo"], xform_cls=c["xform"], render_cls=c["render"], t=vec(k["translate"]),
             r=vec(k["rotate"]), order=k["rot_order"], focal=k["focal"], hap=k["haperture"],
             vap=k["vaperture"], win=vec(k["win_translate"]), xlate=vec(xlate), ar=vec(al["rotate"]),
             scale=scale, paths=nk_quote(paths), material=MATERIAL_PATH % (p + "_Project"),
             plate="push [node %s]\n" % plate if plate else "push 0\n",
             title=nk_quote(title), tile=_lighter(GEOMETRY_COLORS.get(s["geometry"], GEOMETRY_COLORS["shapes"])))
    # Layout: one column for the image -> render chain, the camera to the right, its FrameHold
    # between it and the Project; USD puts the GeoImport to the left of its bind.
    x0 = 170 if usd else 40
    rows = (["Hold", "Paint", "Project", "Bind", "Xform", "Isolate", "Scene", "Render"] if usd
            else ["Hold", "Paint", "Project", "Geo", "Xform", "Render"])
    L = {n: (x0, 100 + 70 * i) for i, n in enumerate(rows)}
    yp = L["Project"][1]
    L.update(CamHold=(x0 + 140, yp), Camera=(x0 + 290, yp - 30), Cam=(x0 + 324, yp + 4),
             RenderDot=(x0 + 324, L["Render"][1] + 4))
    if usd:
        L.update(Geo=(x0 - 150, L["Bind"][1]), SceneDot=(x0 + 324, L["Scene"][1] + 4))
    width, height = x0 + 420, L["Render"][1] + 80
    text = """ BackdropNode {
  inputs 0
  name %(p)s_Backdrop
  label %(title)s
  note_font_size 42
  tile_color %(tile)d
  bdwidth %(w)d
  bdheight %(h)d
 }
%(plate)s FrameHold {
  first_frame %(ref)d
  name %(p)s_Hold
 }
 RotoPaint {
  output rgb
  label "paint here"
  name %(p)s_Paint
 }
set N_paint [stack 0]
""" % dict(v, w=width, h=height)
    if shot:
        text += "push [node %s]\n Dot {\n  label camera\n  name %s_Cam\n }\n" % (cam, p)
    else:
        text += """ %(cam_cls)s {
  inputs 0
  translate %(t)s
  rotate %(r)s
  rot_order %(order)s
  focal %(focal).6g
  haperture %(hap).6g
  vaperture %(vap).6g
  win_translate %(win)s
  label "projection camera"
  name %(p)s_Camera
 }
""" % v
    text += """set N_cam [stack 0]
 FrameHold {
  first_frame %(ref)d
  name %(p)s_CamHold
 }
push $N_paint
 %(proj_cls)s {
  inputs 2
  name %(p)s_Project
 }
""" % v
    xform = """ %(xform_cls)s {
  inputs 1
  translate %(xlate)s
  rotate %(ar)s
  rot_order XYZ
  uniform_scale %(scale).6g
  label "scene scale + camera space"
  name %(p)s_Xform
 }
""" % v
    if usd:
        text += """set N_proj [stack 0]
 %(geo_cls)s {
  inputs 0
  file %(abc)s
  name %(p)s_Geo
 }
set N_geo [stack 0]
push $N_proj
push $N_geo
 GeoBindMaterial {
  inputs 2
  mask_patterns %(paths)s
  material_path %(material)s
  name %(p)s_Bind
 }
""" % v + xform + """ GeoVisibility {
  mask_patterns %(paths)s
  invert_mask true
  name %(p)s_Isolate
 }
set N_geo_out [stack 0]
push $N_cam
 Dot {
  name %(p)s_SceneDot
 }
set N_sdot [stack 0]
 Dot {
  name %(p)s_RenderDot
 }
set N_rdot [stack 0]
push $N_sdot
push $N_geo_out
 GeoScene {
  inputs 2
  name %(p)s_Scene
 }
set N_scene [stack 0]
""" % v
    else:
        text += """ %(geo_cls)s {
  inputs 1
  file %(abc)s
%(scene_view)s
  name %(p)s_Geo
 }
""" % dict(v, scene_view=_scene_view(["/root/%s/%s/%s" % (g, n, n) for g, n in s["prims"]],
                                     ["/root/%s/%s/%s" % (g, n, n) for g, n in prims])) + xform + """set N_scene [stack 0]
push $N_cam
 Dot {
  name %(p)s_RenderDot
 }
set N_rdot [stack 0]
""" % v
    text += """push $N_rdot
push $N_scene
push 0
 %(render_cls)s {
  inputs 3
  name %(p)s_Render
 }
""" % v
    if not shot:
        L.pop("Cam")
    else:
        L.pop("Camera")
    return text, {"%s_%s" % (p, n): xy for n, xy in dict(L, Backdrop=(0, 0)).items()}


EXTRA_DX = 200    # setup Group: each extra frame's projection column, to the left of the main one


def _extra_projections(extra, c, usd, vec):
    """Sequence mode, inside the setup Group: one projection per extra frame of the plate,
    each from the shot camera baked at that frame, layered under the main one:

        Project (reference frame) over Project_<f1> over Project_<f2> ...   -> Blend

    Every projection has occlusion 'world' (it only lands on surfaces its camera saw: what the
    main frame did not see is filled by the next frame that did) and an alpha of 1 inside its
    frame (Alpha_<f>), so 'over' falls through only outside it or where it is occluded.
    extra: [(frame, camera knob values)] in priority order. Expects N_plate and N_proj (the
    main projection) set; leaves the blend on the stack as N_proj. -> (text, layout)."""
    merge = "MergeLayerShader" if usd else "MergeMat"
    text, L = "", {}
    for k, (f, cam) in enumerate(extra):
        x = -EXTRA_DX * (k + 1)
        L.update({"Hold_%d" % f: (x, -740), "Alpha_%d" % f: (x, -700), "Camera_%d" % f: (x - 110, -690),
                  "Project_%d" % f: (x, -640)})
        text += """push $N_plate
 FrameHold {
  first_frame %(f)d
  name Hold_%(f)d
%(pos_hold)s }
 Expression {
  expr3 1
  label "alpha 1"
  name Alpha_%(f)d
%(pos_alpha)s }
set N_img_%(f)d [stack 0]
 %(cam_cls)s {
  inputs 0
  translate %(t)s
  rotate %(r)s
  rot_order %(order)s
  focal %(focal).6g
  haperture %(hap).6g
  vaperture %(vap).6g
  win_translate %(win)s
  label "shot camera at %(f)d"
  name Camera_%(f)d
%(pos_cam)s }
push $N_img_%(f)d
 %(proj_cls)s {
  inputs 2
  project_on front
  occlusion_mode world
  name Project_%(f)d
%(pos_proj)s }
set N_proj_%(f)d [stack 0]
""" % dict(f=f, cam_cls=c["camera"], proj_cls=c["project"], t=vec(cam["translate"]),
           r=vec(cam["rotate"]), order=cam["rot_order"], focal=cam["focal"], hap=cam["haperture"],
           vap=cam["vaperture"], win=vec(cam["win_translate"]),
           pos_hold="  xpos %d\n  ypos %d\n" % L["Hold_%d" % f],
           pos_alpha="  xpos %d\n  ypos %d\n" % L["Alpha_%d" % f],
           pos_cam="  xpos %d\n  ypos %d\n" % L["Camera_%d" % f],
           pos_proj="  xpos %d\n  ypos %d\n" % L["Project_%d" % f])
    # Merge from the lowest priority up: input 1 (A) goes over input 0 (B).
    acc = "$N_proj_%d" % extra[-1][0]
    for k in range(len(extra) - 2, -1, -1):
        f = extra[k][0]
        name = "Blend_%d" % f
        L[name] = (-EXTRA_DX * (k + 1), -605)
        text += "push $N_proj_%d\npush %s\n %s {\n  inputs 2\n  name %s\n  xpos %d\n  ypos %d\n }\nset N_%s [stack 0]\n" % (
            f, acc, merge, name, L[name][0], L[name][1], name)
        acc = "$N_%s" % name
    L["Blend"] = (0, -605)
    text += "push $N_proj\npush %s\n %s {\n  inputs 2\n  label \"reference frame over the others\"\n  name Blend\n  xpos 0\n  ypos -605\n }\nset N_proj [stack 0]\n" % (acc, merge)
    return text, L


def setup_template(group, ref_frame, abc, cam, prims, align, use_shot_cam=False, label="",
                   info="", system="USD", geometry="shapes", plate=None, shot_cam=None, extra=(),
                   scale=1.0, splat=None, splat_colorspace="lin_rec709_scene"):
    """Projection setup as one Group pasted next to the GeoProjection node (which must be
    selected so the Group's 'plate' input connects to it), for the classic or USD 3D system.

    Inside the Group the node names are fixed and every reference is parent.<knob> or an input,
    so copying, pasting or renaming the Group (or the gizmo, or the shot camera) breaks nothing.
    Group knob scene_scale: live scale of the geometry about the projection camera.
    Group input 'cam' (only with use_shot_cam): the shot camera, connected after the paste; the
    render goes through it, the projection uses the baked camera at the reference frame.

    cam: projection camera knob values (translate, rotate, rot_order, focal, haperture,
    vaperture, win_translate). prims: [(group, name)] of the Alembic objects; Blender puts each
    under its group's empty: /root/<group>/<name>/<name> in ReadGeo2, /<group>/<name>/<name> in USD.
    USD setups get an Isolate node (GeoVisibility) driven by the Group's 'show only' / 'pattern'.
    align: {"rotate": xyz deg, "ra_ft": Ra @ fit camera position, "shot_t": camera position} for
    the transform x' = s Ra x + (shot_t - s Ra ft).
    plate / shot_cam: full names ('root.GeoProjection1') the Group connects to by
    'push [node ...]'; paste with nothing selected. The arguments are kept on the Group
    (gp_setup) for 'extract for paint'.
    extra: sequence mode, [(frame, shot camera knob values at that frame)] projected under the
    reference frame (_extra_projections); the main projection then gets occlusion 'world' and
    an alpha of 1 too.
    splat (USD only): the Gaussian splat PLY in the Alembic's world. GeoImport Splat ->
    SplatXform (follows Xform) -> SplatRender through the render camera; the Group's 'output'
    menu switches projection / splat / projection over splat (the plate gets an alpha of 1 so
    the projection covers the splat where there is geometry). Returns the .nk text."""
    if splat and system != "USD":
        raise ValueError("The Gaussian splat needs the USD 3D system.")
    setup = dict(ref=int(ref_frame), abc=abc, cam=cam, prims=[list(p) for p in prims], align=align,
                 use_shot_cam=bool(use_shot_cam), system=system, geometry=geometry, info=info)
    c = CLASSES[system]
    usd = system == "USD"
    vec = lambda v: "{%s}" % " ".join("%.6g" % x for x in v)
    xlate = " ".join('{"%.6g - %.6g*parent.scene_scale"}' % (st, rf)
                     for st, rf in zip(align["shot_t"], align["ra_ft"]))
    # USD prims are selected by path pattern: the geometry groups. Not /*: the stage's roots
    # also hold /materials (where Project3DShader puts its material) and the Alembic camera,
    # which must not get the projection bound to them.
    order = ["structure", "objects", "backdrop", "other"]
    groups = sorted(set(g for g, _ in prims), key=lambda g: order.index(g) if g in order else 9)
    mask = (' mask_patterns "%s"\n' % " ".join("/" + g for g in groups)) if usd else ""
    show = ["all"] + ["%s/*" % g for g in groups] + ["%s/%s" % gn for gn in prims]
    what = "Show only this object or group (projection and render), and what 'extract for paint' takes. all = everything."
    isolate_knobs = """ addUserKnob {4 show_only l "show only" t %s M {%s}}
 addUserKnob {1 show_pattern l "pattern" t %s}
 addUserKnob {22 extract l "extract for paint" t %s T %s}
""" % (nk_quote(what), " ".join(nk_quote(x) for x in show),
       nk_quote("Free mask instead of the menu (when filled, it wins): names or paths, space "
                "separated, * and ? wildcards; a name without a leading / is searched at any "
                "depth. Examples: lamp_1   *chair*   vase* sofa_1   /objects/*   /structure/floor"),
       nk_quote("Paste, in a backdrop beside this Group, loose nodes to paint on what 'show only' / "
                "'pattern' picks: FrameHold -> RotoPaint (paint here) -> Project <- FrameHold of the "
                "camera; that geometry only -> ScanlineRender through the camera."),
       _py("extract"))
    v = dict(group=group, ref=int(ref_frame), t=vec(cam["translate"]), r=vec(cam["rotate"]),
             order=cam["rot_order"], focal=cam["focal"], hap=cam["haperture"], vap=cam["vaperture"],
             win=vec(cam["win_translate"]), abc=nk_quote(abc), label=nk_quote(label), info=nk_quote(info),
             xlate=xlate, ar=vec(align["rotate"]), mask=mask,
             material=MATERIAL_PATH % ("Blend" if extra else "Project"),
             occlusion="  occlusion_mode world\n" if extra else "",
             version=__version__, n_in=2 if use_shot_cam else 1,
             tile=GEOMETRY_COLORS.get(geometry, GEOMETRY_COLORS["shapes"]),
             isolate_knobs=isolate_knobs, setup=_nk_string(json.dumps(setup)), scale=float(scale),
             knob_changed=" knobChanged %s\n" % _nk_string(ISOLATE_CALLBACK if usd else ISOLATE_CALLBACK_CLASSIC),
             scale_t=nk_quote("Live scale of the geometry about the projection camera: the projection "
                              "does not change. Use it to match the scale of a tracked scene."),
             **{k + "_cls": cls for k, cls in c.items()})
    # Layout (David's): one column plate -> Hold -> Project -> Bind -> Xform -> Isolate -> Scene ->
    # Render; Geo to the left of Bind; the camera to the right, its line dropping through two Dots
    # to Scene and Render.
    L = {"plate": (0, -760), "Hold": (0, -700), "Project": (0, -640), "Bind": (0, -570),
         "Xform": (0, -490), "Isolate": (0, -410), "Scene": (0, -320), "Render": (0, -230),
         "Output1": (0, -170), "Camera": (200, -680), "Geo": (-190, -575), "cam": (360, -680)}
    if not usd:
        L["Geo"] = L["Bind"]
    extra_text = ""
    if extra or splat:
        L.update(plate=(0, -790), Hold=(0, -740), Alpha=(0, -700))
    if splat:
        L.update(Splat=(-380, -575), SplatXform=(-380, -490), SplatRender=(-380, -230),
                 SplatOver=(0, -170), Out=(0, -120), Output1=(0, -60))
    if extra:
        extra_text, extra_L = _extra_projections(list(extra), c, usd, vec)
        L.update(extra_L)
    rx = L["cam"][0] if use_shot_cam else L["Camera"][0]
    L["DotScene"], L["DotRender"] = (rx + 34, L["Scene"][1] + 30), (rx + 34, L["Render"][1] + 4)
    pos = lambda n: "  xpos %d\n  ypos %d\n" % L[n]
    v.update({"pos_" + k: pos(k) for k in L})
    v["alpha"] = (' Expression {\n  expr3 1\n  label "alpha 1"\n  name Alpha\n%s }\n' % pos("Alpha")) if extra or splat else ""
    v["output_knob"] = (' addUserKnob {4 output_mode l output t %s M {projection splat "projection over splat"}}\n'
                        ' output_mode "projection over splat"\n' % nk_quote(
                            "projection: the plate projected on the geometry. splat: the Gaussian splat "
                            "(SplatRender). projection over splat: the projection where there is "
                            "geometry, the splat everywhere else.")) if splat else ""
    # plate / shot_cam: full names to connect to by 'push [node ...]' (paste with nothing
    # selected; pasting onto a selected node would rewire everything below it). Without
    # them, the Group connects to the selected node.
    ref = lambda n: "push [node %s]\n" % n
    head = "set cut_paste_input [stack 0]\n%s%s" % (
        (ref(shot_cam) if shot_cam else "push 0\n") if use_shot_cam else "",
        ref(plate) if plate else "push $cut_paste_input\n")
    text = (head + """Group {
 inputs %(n_in)d
 name %(group)s
 tile_color %(tile)d
 label %(label)s
%(knob_changed)s addUserKnob {20 geoproj l GeoProjection}
 addUserKnob {7 scene_scale l "scene scale" t %(scale_t)s R 0.1 10}
 scene_scale %(scale).6g
%(output_knob)s%(isolate_knobs)s addUserKnob {26 info l "" +STARTLINE T %(info)s}
 addUserKnob {26 ver l "" +STARTLINE T "<small>GeoProjection v%(version)s  -  by David Borrull</small>"}
 addUserKnob {1 gp_setup +INVISIBLE}
 gp_setup %(setup)s
 addUserKnob {3 gp_extracts +INVISIBLE}
}
 Input {
  inputs 0
  name plate
%(pos_plate)s }
set N_plate [stack 0]
 FrameHold {
  first_frame %(ref)d
  name Hold
%(pos_Hold)s }
%(alpha)sset N_hold [stack 0]
 %(camera_cls)s {
  inputs 0
  translate %(t)s
  rotate %(r)s
  rot_order %(order)s
  focal %(focal).6g
  haperture %(hap).6g
  vaperture %(vap).6g
  win_translate %(win)s
  label "projection camera"
  name Camera
%(pos_Camera)s }
set N_cam [stack 0]
push $N_cam
push $N_hold
 %(project_cls)s {
  inputs 2
  project_on front
%(occlusion)s  name Project
%(pos_Project)s }
set N_proj [stack 0]
""") % v + extra_text
    if usd:
        text += """ %(geo_cls)s {
  inputs 0
  file %(abc)s
  name Geo
%(pos_Geo)s }
set N_geo [stack 0]
push $N_proj
push $N_geo
 GeoBindMaterial {
  inputs 2
 %(mask)s  material_path %(material)s
  name Bind
%(pos_Bind)s }
""" % v
    else:
        text += """ %(geo_cls)s {
  file %(abc)s
%(scene_view)s
  name Geo
%(pos_Geo)s }
""" % dict(v, scene_view=_scene_view(["/root/%s/%s/%s" % (g, n, n) for g, n in prims]))
    text += """ %(xform_cls)s {
  inputs 1
 %(mask)s  translate {%(xlate)s}
  rotate %(ar)s
  rot_order XYZ
  uniform_scale {{parent.scene_scale}}
  label "scene scale + shot camera space"
  name Xform
%(pos_Xform)s }
""" % v
    if usd:
        text += """ GeoVisibility {
  mask_patterns /*
  invert_mask true
  disable true
  label "show only"
  name Isolate
%(pos_Isolate)s }
""" % v
    text += "set N_geo_out [stack 0]\n"
    if use_shot_cam:
        text += """ Input {
  inputs 0
  name cam
  number 1
%(pos_cam)s }
set N_rcam [stack 0]
""" % v
        render_cam = "$N_rcam"
    else:
        render_cam = "$N_cam"
    # The render camera drops through two Dots, to the Scene (USD: the camera goes into the
    # scene, as in Foundry's own setups) and to the Render's cam input.
    text += """push %s
 Dot {
  name DotScene
%s }
set N_dot_scene [stack 0]
 Dot {
  name DotRender
%s }
set N_dot_render [stack 0]
""" % (render_cam, v["pos_DotScene"], v["pos_DotRender"])
    if usd:
        text += "push $N_dot_scene\npush $N_geo_out\n %s {\n  inputs 2\n  name Scene\n%s }\n" % (c["scene"], v["pos_Scene"])
    else:
        text += "push $N_geo_out\n %s {\n  inputs 1\n  name Scene\n%s }\n" % (c["scene"], v["pos_Scene"])
    text += """set N_scene [stack 0]
push $N_dot_render
push $N_scene
push 0
 %s {
  inputs 3
  name Render
%s }
""" % (c["render"], v["pos_Render"])
    if splat:
        # SplatRender inputs: 0 bg, 1 cam, 2 scene. The splat is in the Alembic's world, so its
        # transform follows Xform (scene scale + shot camera space).
        text += """set N_render [stack 0]
 GeoImport {
  inputs 0
  file %(file)s
  name Splat
%(pos_Splat)s }
 GeoTransform {
  inputs 1
  translate {{Xform.translate.x} {Xform.translate.y} {Xform.translate.z}}
  rotate {{Xform.rotate.x} {Xform.rotate.y} {Xform.rotate.z}}
  rot_order XYZ
  uniform_scale {{Xform.uniform_scale}}
  label "follows Xform"
  name SplatXform
%(pos_SplatXform)s }
set N_splat_geo [stack 0]
push $N_splat_geo
push $N_dot_render
push 0
 SplatRender {
  inputs 3
  output_colorspace %(cs)s
  name SplatRender
%(pos_SplatRender)s }
set N_splat [stack 0]
push $N_render
push $N_splat
 Merge2 {
  inputs 2
  label "projection over splat"
  name SplatOver
%(pos_SplatOver)s }
set N_over [stack 0]
push $N_over
push $N_splat
push $N_render
 Switch {
  inputs 3
  which {{parent.output_mode}}
  label "output"
  name Out
%(pos_Out)s }
""" % dict(v, file=nk_quote(splat), cs=splat_colorspace)
    text += """ Output {
  name Output1
%s }
end_group
""" % v["pos_Output1"]
    return text
