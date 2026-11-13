# DBS_ComfyProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""Draws the README banner: docs/images/banner.jpg (1600 x 440), in the style of ComfyBridge's.

python tools/make_banner.py [--image picture.jpg]      (needs Pillow; fonts: Segoe UI, Windows)

Left: the toolbar icon, large, on a teal gradient, with the name and two lines of text.
Right: --image (fitted to the height, faded in from the left), else a drawn illustration of
the idea: a projector throwing the plate onto boxes standing on a floor grid.
"""

import argparse
import os
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_icon  # noqa: E402

W, H = 1600, 440
S = 2                                       # supersampling of the drawn illustration
BG = (22, 24, 26)
TEAL_GLOW = (47, 104, 90)
FONTS = "C:/Windows/Fonts"
TITLE = "DBS_ComfyProjection"
SUB = "One frame -> projection setup in Nuke"
FEATURES = "MoGe 2  \u00b7  SAM 3  \u00b7  Blender  \u00b7  snapped to your camera track"
TITLE_MAX_W = 700                           # px: the title ends before the picture
TITLE_PREFIX = (120, 205, 180)              # the DBS_ prefix, in the icon's teal
FADE_X0, FADE_X1 = 1090, 1200               # the right picture fades in over these columns


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTS, name), size)


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def background():
    im = Image.new("RGB", (W, H), BG)
    px = im.load()
    for x in range(W):
        # Brightest around the icon, falling off to the dark ground by mid width.
        t = min(max((x - 60) / 820.0, 0.0), 1.0)
        t = t * t * (3 - 2 * t)
        c = lerp(TEAL_GLOW, BG, t)
        for y in range(H):
            v = 1.0 - 0.18 * abs(y - H / 2) / (H / 2)            # soft vertical vignette
            px[x, y] = tuple(int(ch * v) for ch in c)
    return im


def illustration():
    """Projector -> frustum -> boxes on a floor grid, drawn at S x and scaled down (RGBA)."""
    w, h = W * S, H * S
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def P(x, y):
        return (x * S, y * S)

    vp = (1400, 150)                                       # vanishing point on the horizon
    grid = (255, 255, 255, 34)
    for x in range(800, 2200, 90):                         # lines to the vanishing point
        d.line([P(*vp), P(x, H)], fill=grid, width=S)
    y, k = 165, 4.0
    while y < H:
        d.line([P(1000, y), P(W, y)], fill=grid, width=S)
        y += k
        k *= 1.45

    def face(pts, top, bottom):
        """A face filled with a vertical gradient: the projected picture."""
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).polygon([P(*p) for p in pts], fill=255)
        ys = [p[1] for p in pts]
        grad = Image.new("RGBA", (w, h))
        gd = ImageDraw.Draw(grad)
        y0, y1 = int(min(ys) * S), int(max(ys) * S)
        for yy in range(y0, y1 + 1):
            gd.line([(0, yy), (w, yy)], fill=lerp(top, bottom, (yy - y0) / max(y1 - y0, 1)) + (255,))
        im.paste(grad, (0, 0), mask)

    edge = (150, 230, 205, 255)

    def box(x0, y0, x1, y1, dx, dy, top, bottom):
        front = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        side = [(x1, y0), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy), (x1, y1)]
        lid = [(x0, y0), (x0 + dx, y0 + dy), (x1 + dx, y0 + dy), (x1, y0)]
        face(front, top, bottom)
        face(side, lerp(top, (0, 0, 0), 0.45), lerp(bottom, (0, 0, 0), 0.45))
        face(lid, lerp(top, (255, 255, 255), 0.15), lerp(top, (0, 0, 0), 0.1))
        for poly in (front, side, lid):
            d.line([P(*q) for q in poly + [poly[0]]], fill=edge, width=2 * S, joint="curve")
        return front

    # Projected picture: sky-warm at the top, green-brown at the bottom (a plate in miniature).
    small = box(1500, 232, 1555, 298, 22, -14, (214, 170, 120), (88, 104, 64))
    front = box(1290, 196, 1440, 330, 50, -32, (232, 196, 140), (70, 98, 62))
    # Frustum from the projector's lens to the big box's front face.
    lens = (1205, 140)
    ray = (255, 255, 255, 120)
    for c in (front[0], front[3]):
        d.line([P(*lens), P(*c)], fill=ray, width=S)
    # Projector body (as in the icon).
    d.rounded_rectangle([P(1157, 120), P(1201, 160)], radius=6 * S, fill=(255, 255, 255, 255))
    d.ellipse([P(1193, 129), P(1217, 153)], fill=(255, 255, 255, 255), outline=(63, 127, 111, 255),
              width=3 * S)
    return im.resize((W, H), Image.LANCZOS)


def picture(path):
    """--image fitted to the banner height, right-aligned (RGBA)."""
    src = Image.open(path).convert("RGB")
    s = H / float(src.height)
    src = src.resize((int(src.width * s), H), Image.LANCZOS)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    out.paste(src, (W - src.width, 0))
    return out


def fade(layer):
    """Fade the right picture in from FADE_X0 to FADE_X1."""
    a = layer.getchannel("A")
    ramp = Image.new("L", (W, H), 0)
    rp = ramp.load()
    for x in range(W):
        t = min(max((x - FADE_X0) / float(FADE_X1 - FADE_X0), 0.0), 1.0)
        for y in range(H):
            rp[x, y] = int(255 * t * t * (3 - 2 * t))
    layer.putalpha(ImageChops.multiply(a, ramp))
    return layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="picture for the right side (default: drawn illustration)")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "images", "banner.jpg"))
    a = ap.parse_args()

    im = background().convert("RGBA")
    right = picture(a.image) if a.image else illustration()
    im.alpha_composite(fade(right))

    icon = make_icon.draw(250)
    shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), (100, 104, 350, 354), icon.getchannel("A"))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(14)))
    im.alpha_composite(icon, (95, 95))

    d = ImageDraw.Draw(im)
    # Title: the DBS_ prefix in the icon's teal, the name in white, sized to end before the
    # picture on the right (TITLE_MAX_W), on the same baseline as before.
    prefix, name = TITLE.split("_", 1) if "_" in TITLE else ("", TITLE)
    prefix += "_" if prefix else ""
    size = 104
    while size > 40 and font("segoeui.ttf", size).getlength(prefix + name) > TITLE_MAX_W:
        size -= 2
    f = font("segoeui.ttf", size)
    d.text((402, 248), prefix, font=f, fill=TITLE_PREFIX, anchor="ls")
    d.text((402 + f.getlength(prefix), 248), name, font=f, fill=(255, 255, 255), anchor="ls")
    d.text((406, 272), SUB.replace("->", "\u2192"), font=font("segoeuil.ttf", 37), fill=(214, 226, 222))
    d.text((406, 326), FEATURES, font=font("segoeuil.ttf", 26), fill=(150, 168, 162))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    im.convert("RGB").save(a.out, quality=92, optimize=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
