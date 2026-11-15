# DBS_SceneProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Draws the toolbar icon: nuke/geo_projection.png (24 px) and nuke/geo_projection@2x.png (48 px).

python tools/make_icon.py        (needs Pillow)

A rounded teal square (the gizmo's 'shapes' colour) with a white projector throwing its
frustum onto a box, drawn at 8x and scaled down.
"""

import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEAL = (63, 127, 111, 255)
WHITE = (255, 255, 255, 255)
SOFT = (255, 255, 255, 90)
S = 8                      # supersampling


def draw(size):
    n = size * S
    u = n / 48.0                                   # design units: a 48 x 48 grid
    im = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([1 * u, 1 * u, 47 * u, 47 * u], radius=9 * u, fill=TEAL)

    def P(x, y):
        return (x * u, y * u)

    lw = max(int(round(2.6 * u)), 1)
    # Box on the right, in a simple oblique view: front face, top, side.
    fx0, fy0, fx1, fy1 = 25, 20, 38, 36            # front face
    dx, dy = 5, -5                                 # depth offset
    front = [P(fx0, fy0), P(fx1, fy0), P(fx1, fy1), P(fx0, fy1)]
    top = [P(fx0, fy0), P(fx0 + dx, fy0 + dy), P(fx1 + dx, fy0 + dy), P(fx1, fy0)]
    side = [P(fx1, fy0), P(fx1 + dx, fy0 + dy), P(fx1 + dx, fy1 + dy), P(fx1, fy1)]
    # The projected image lands on the front face: a light fill there.
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(over).polygon(front, fill=SOFT)
    im.alpha_composite(over)
    d = ImageDraw.Draw(im)
    # Frustum from the projector's lens to the front face's corners.
    lens = (13.5, 28)
    for c in ((fx0, fy0), (fx0, fy1)):
        d.line([P(*lens), P(*c)], fill=WHITE, width=max(int(round(1.6 * u)), 1))
    for poly in (front, top, side):
        d.line(poly + [poly[0]], fill=WHITE, width=lw, joint="curve")
    # Projector body: a rounded block with a lens.
    d.rounded_rectangle([5 * u, 23 * u, 14 * u, 33 * u], radius=2 * u, fill=WHITE)
    d.ellipse([11.5 * u, 25.5 * u, 16.5 * u, 30.5 * u], fill=WHITE, outline=TEAL, width=max(int(round(1.2 * u)), 1))
    return im.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    for size, name in ((24, "geo_projection.png"), (48, "geo_projection@2x.png")):
        path = os.path.join(ROOT, "nuke", name)
        draw(size).save(path)
        print("wrote", path)
