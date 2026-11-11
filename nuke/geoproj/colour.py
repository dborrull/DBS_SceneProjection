# Copied from ComfyBridge (nuke-comfy-pipe) v0.8.2 - by David Borrull
# https://github.com/dborrull/nuke-comfy-pipe
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Viewer transform parsing and colour space names (no Nuke dependency).

MoGe, like the diffusion models ComfyBridge feeds, is trained on display-referred
images, so the plate is sent through the active Viewer transform."""

import re

IN_VIEWER = "viewer transform"
IN_SRGB = "sRGB"
LUT = "__lut__"

_CS_SRGB = ["sRGB - Texture", "srgb_tx", "sRGB Encoded Rec.709 (sRGB)", "Output - sRGB", "sRGB"]


def parse_viewer_process(text):
    """'ACES 1.0 - SDR Video (sRGB - Display)' -> ('sRGB - Display', 'ACES 1.0 - SDR Video')
    'sRGB' (Nuke colour management)        -> (LUT, 'sRGB')
    '' / 'None'                              -> None"""
    text = (text or "").strip()
    m = re.match(r"^(.+?)\s*\((.+)\)$", text)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    if text and text.lower() not in ("none", "raw"):
        return LUT, text
    return None


def ocio_names():
    """Colour space names of the active OCIO config, or None if unavailable."""
    try:
        import PyOpenColorIO as OCIO
        cfg = OCIO.GetCurrentConfig()
        return [cs.getName() for cs in cfg.getColorSpaces()]
    except Exception:
        return None


def pick_colorspace(candidates, available=None):
    if available:
        low = {a.lower(): a for a in available}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
    return candidates[-1] if candidates else None
