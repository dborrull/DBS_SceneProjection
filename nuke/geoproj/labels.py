# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Label library: presets are text files, one `text:role` per line (# for comments).

Searched in the package's label_presets folder, then in ~/.nuke/geo_projection_labels and
$GEO_PROJECTION_LABELS (a user preset with the same name replaces the built-in one).
Roles: floor, wall, ceiling, object (one box per instance), backdrop (far planes behind
everything) and ignore (no geometry)."""

import collections
import os

ROLES = ("floor", "wall", "ceiling", "object", "backdrop", "ignore")
SHAPES = ("auto", "box", "card", "cylinder", "cone", "sphere", "revolve")   # blender/shapes.py
RELIEF = "relief"     # per label: that label's surface as a relief mesh (objects and backdrop)
DEFAULT_PRESET = "all"
BUILTIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_presets")


class LabelError(ValueError):
    pass


def preset_dirs():
    dirs = [BUILTIN_DIR, os.path.join(os.path.expanduser("~"), ".nuke", "geo_projection_labels")]
    if os.environ.get("GEO_PROJECTION_LABELS"):
        dirs.append(os.environ["GEO_PROJECTION_LABELS"])
    return dirs


def list_presets():
    """{name: path}, built-ins first; later folders override by name."""
    found = collections.OrderedDict()
    for d in preset_dirs():
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".txt"):
                    found[f[:-4]] = os.path.join(d, f)
    return found


def parse_labels(text):
    """'floor:floor, table:object, lamp:object:revolve' or one per line ('#' comments; a bare
    'table' = object) -> ordered {label: role or role:shape}. The shape forces what the fit
    makes of that label (objects: a shape or relief; backdrop: relief); without it the fit picks. A later line for the same
    label wins."""
    out = collections.OrderedDict()
    items = [i for line in (text or "").splitlines() for i in line.split("#", 1)[0].split(",")]
    for item in items:
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        name = parts[0]
        role = (parts[1].lower() if len(parts) > 1 else "") or "object"
        shape = parts[2].lower() if len(parts) > 2 else ""
        if role not in ROLES:
            raise LabelError("Unknown role '%s' for '%s' (%s)." % (role, name, ", ".join(ROLES)))
        if shape and not ((role == "object" and shape in SHAPES + (RELIEF,))
                          or (role == "backdrop" and shape == RELIEF)):
            raise LabelError("Unknown shape '%s' for '%s' (objects: %s; backdrop: relief)."
                             % (shape, name, ", ".join(SHAPES + (RELIEF,))))
        out[name] = role + (":" + shape if shape and shape != "auto" else "")
    return out


def load_preset(name):
    presets = list_presets()
    if name not in presets:
        raise LabelError("No label preset '%s' (have: %s)." % (name, ", ".join(presets)))
    with open(presets[name], encoding="utf-8") as f:
        return parse_labels(f.read())


def effective_labels(preset, extra=""):
    """Preset + extra lines; an extra line for a label already in the preset changes its role,
    `label:ignore` included."""
    labels = load_preset(preset) if preset else collections.OrderedDict()
    labels.update(parse_labels(extra))
    return labels
