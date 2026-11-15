# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Minimal OpenEXR writer: scanline, uncompressed, FLOAT channels. numpy only.

Used so depth stays float32 end to end without depending on OpenEXR/Imath or
OpenCV's OPENCV_IO_ENABLE_OPENEXR flag.
"""

import struct

import numpy as np

_FLOAT = 2


def _attr(name, type_name, payload):
    return name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(payload)) + payload


def write_exr(path, channels):
    """Write {channel_name: HxW float array} to an uncompressed float32 EXR."""
    names = sorted(channels)
    planes = [np.ascontiguousarray(channels[n], dtype="<f4") for n in names]
    h, w = planes[0].shape
    for n, p in zip(names, planes):
        if p.shape != (h, w):
            raise ValueError("channel %s has shape %s, expected %s" % (n, p.shape, (h, w)))

    chlist = b"".join(n.encode() + b"\0" + struct.pack("<iB3xii", _FLOAT, 0, 1, 1) for n in names) + b"\0"
    box = struct.pack("<iiii", 0, 0, w - 1, h - 1)
    header = (
        struct.pack("<ii", 20000630, 2)
        + _attr("channels", "chlist", chlist)
        + _attr("compression", "compression", b"\0")
        + _attr("dataWindow", "box2i", box)
        + _attr("displayWindow", "box2i", box)
        + _attr("lineOrder", "lineOrder", b"\0")
        + _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
        + _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
        + _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
        + b"\0"
    )

    # Each scanline: y, byte count, then every channel's row in channel-name order.
    line_bytes = 4 * w * len(names)
    first = len(header) + 8 * h
    offsets = np.arange(h, dtype="<u8") * (8 + line_bytes) + first
    rows = np.stack(planes, axis=1)  # (h, nch, w)

    with open(path, "wb") as f:
        f.write(header)
        f.write(offsets.tobytes())
        prefix = struct.pack("<i", line_bytes)
        for y in range(h):
            f.write(struct.pack("<i", y) + prefix)
            f.write(rows[y].tobytes())
