# GeoProjection (nuke_geo_projection) - by David Borrull
# MIT License - Copyright (c) 2026 David Borrull. See LICENSE.
"""ComfyUI nodes for nuke_geo_projection: dump MoGe geometry to disk in float.

The core MoGeRender node only gives normalized disparity; this keeps metric depth,
the point map and the intrinsics so Nuke/Blender can rebuild camera + geometry.

Conventions (MoGe / OpenCV camera space): X right, Y down, Z forward, units = MoGe
metric (roughly metres, scale not guaranteed). Invalid pixels (sky, masked) are
written as 0 with mask = 0.
"""

import json
import math
import os

import numpy as np

import folder_paths

from .exr_io import write_exr


class SaveMoGeGeometry:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_geometry": ("MOGE_GEOMETRY",),
                "filename_prefix": ("STRING", {"default": "geo_projection/moge"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "nuke_geo_projection"

    def save(self, moge_geometry, filename_prefix):
        points = moge_geometry["points"].float().cpu().numpy()
        B, H, W, _ = points.shape
        depth = _get(moge_geometry, "depth", B)
        mask = _get(moge_geometry, "mask", B)
        normal = _get(moge_geometry, "normal", B)
        K = _get(moge_geometry, "intrinsics", B)

        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), W, H)

        json_paths = []
        files = []  # history entries, so a client can fetch them with /view
        for b in range(B):
            valid = np.isfinite(points[b]).all(-1)
            if mask is not None:
                valid &= mask[b].astype(bool)
            z = depth[b] if depth is not None else points[b, ..., 2]

            def clean(a):
                return np.where(valid[..., None] if a.ndim == 3 else valid, np.nan_to_num(a, posinf=0, neginf=0), 0).astype(np.float32)

            base = "%s_%05d_" % (name, counter + b)
            channels = {"Z": clean(z), "mask": valid.astype(np.float32),
                        "P.x": clean(points[b, ..., 0]), "P.y": clean(points[b, ..., 1]), "P.z": clean(points[b, ..., 2])}
            if normal is not None:
                n = clean(normal[b])
                channels.update({"N.x": n[..., 0], "N.y": n[..., 1], "N.z": n[..., 2]})
            write_exr(os.path.join(out_dir, base + "geo.exr"), channels)

            arrays = {"points": clean(points[b]), "depth": clean(z), "mask": valid}
            if normal is not None:
                arrays["normal"] = clean(normal[b])
            np.savez_compressed(os.path.join(out_dir, base + "geo.npz"), **arrays)

            zv = z[valid]
            info = {
                "width": W,
                "height": H,
                "convention": "opencv (x right, y down, z forward)",
                "valid_ratio": float(valid.mean()),
                "depth_min": float(zv.min()) if zv.size else None,
                "depth_median": float(np.median(zv)) if zv.size else None,
                "depth_max": float(zv.max()) if zv.size else None,
            }
            if K is not None:
                # MoGe intrinsics are normalized: fx by width, fy by height.
                k = K[b]
                fx, fy, cx, cy = (float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]))
                info["intrinsics_normalized"] = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
                info["focal_pixels"] = fx * W
                info["fov_x_deg"] = math.degrees(2 * math.atan(0.5 / fx))
                info["fov_y_deg"] = math.degrees(2 * math.atan(0.5 / fy))
                # focal (mm) = focal_normalized_by_width * horizontal aperture
                info["focal_mm_for_haperture_36"] = fx * 36.0
            json_path = os.path.join(out_dir, base + "geo.json")
            with open(json_path, "w") as f:
                json.dump(info, f, indent=2)
            json_paths.append(json_path)
            files += [{"filename": base + ext, "subfolder": subfolder, "type": "output"}
                      for ext in ("geo.exr", "geo.json", "geo.npz")]

        # custom "geo" key: the web UI does not try to preview it
        return {"ui": {"geo": files}, "result": (json_paths[0],)}


class SaveDA3Geometry:
    """Depth Anything 3 output (DA3Inference) -> <prefix>_da3.npz: depth (B, H, W) float32 raw
    model depth, plus sky / confidence and, in multiview mode, pixel intrinsics (1, S, 3, 3)
    and world-to-camera extrinsics (1, S, 3, 4) when the model gave them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"da3_geometry": ("DA3_GEOMETRY",),
                             "filename_prefix": ("STRING", {"default": "geo_projection/da3"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "nuke_geo_projection"

    def save(self, da3_geometry, filename_prefix):
        arrays = {}
        for k in ("depth", "sky", "confidence", "intrinsics", "extrinsics"):
            v = da3_geometry.get(k)
            if v is not None:
                arrays[k] = v.float().cpu().numpy()
        H, W = arrays["depth"].shape[-2:]
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), W, H)
        fname = "%s_%05d_.npz" % (name, counter)
        np.savez_compressed(os.path.join(out_dir, fname), mode=str(da3_geometry.get("mode", "")), **arrays)
        return {"ui": {"geo": [{"filename": fname, "subfolder": subfolder, "type": "output"}]}}


_SAM3_NODE = None  # one comfyui-rmbg SAM3Segment instance: its model cache lives on it


class SegmentLabels:
    """SAM3 text grounding for many labels on one image: the model loads once and the image is
    encoded once; each label only runs the text + grounding heads. Writes <prefix>_seg.npz with
    labels (L,), masks (N, H, W) bool, mask_label (N,) index into labels, scores (N,).

    Uses comfyui-rmbg's SAM3 (models/sam3/sam3.pt). Wiring one SAM3Segment node per label
    instead builds a model copy per node (minutes, then out of VRAM on 8 GB)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "labels": ("STRING", {"default": "floor, wall, table, chair", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.01}),
                "filename_prefix": ("STRING", {"default": "geo_projection/seg"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "segment"
    OUTPUT_NODE = True
    CATEGORY = "nuke_geo_projection"

    def segment(self, image, labels, confidence_threshold, filename_prefix):
        global _SAM3_NODE
        import nodes
        import torch
        from PIL import Image

        if _SAM3_NODE is None:
            cls = nodes.NODE_CLASS_MAPPINGS.get("SAM3Segment")
            if cls is None:
                raise RuntimeError("SegmentLabels needs comfyui-rmbg (SAM3Segment) installed.")
            _SAM3_NODE = cls()
        processor, device = _SAM3_NODE._load_processor("Auto")
        # SAM3 (~3.4 GB) is outside ComfyUI's model management: make room for it, and park it on
        # the CPU when done (below) so MoGe / SHARP after it get the VRAM back on an 8 GB card.
        import comfy.model_management as mm
        mm.unload_all_models()
        processor.model.to(device)
        names =[s.strip() for s in labels.replace("\n", ",").split(",") if s.strip()]
        img = Image.fromarray((image[0, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        H, W = img.height, img.width

        masks, mask_label, scores = [], [], []
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.no_grad()
        with torch.inference_mode(), autocast:
            state = processor.set_image(img)
            for li, name in enumerate(names):
                processor.reset_all_prompts(state)
                processor.set_confidence_threshold(confidence_threshold, state)
                state = processor.set_text_prompt(name, state)
                m = state.get("masks")
                if m is None or m.numel() == 0:
                    continue
                m = (m.squeeze(1) if m.ndim == 4 else m).bool().cpu().numpy()
                sc = state.get("scores")
                sc = sc.float().cpu().numpy().reshape(-1) if sc is not None else np.ones(len(m))
                for k in range(len(m)):
                    if m[k].any():
                        masks.append(m[k])
                        mask_label.append(li)
                        scores.append(float(sc[k]) if k < len(sc) else 1.0)
        del state
        processor.model.to("cpu")
        mm.soft_empty_cache()

        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), W, H)
        fname = "%s_%05d_.npz" % (name, counter)
        np.savez_compressed(os.path.join(out_dir, fname), labels=np.array(names),
                            masks=np.array(masks, bool).reshape(-1, H, W),
                            mask_label=np.array(mask_label, np.int32), scores=np.array(scores, np.float32))
        return {"ui": {"geo": [{"filename": fname, "subfolder": subfolder, "type": "output"}]}}


class SolveTrackScale:
    """Scale between MoGe's units and a tracked camera's: SIFT matches between the reference
    image and other frames of the plate, each match lifted to 3D with MoGe's point at the
    reference pixel, placed in the track's world by the reference camera and a scale s, and
    reprojected through the camera at the other frame. s = the one with the smallest median
    reprojection error (searched on a log grid, then refined).

    cameras (JSON): {"ref": frame, "frames": {frame: {"matrix": 16 floats, row-major world
    matrix, translation in the last column (Nuke: camera looks down -Z, +Y up), "focal",
    "haperture"}}}. frames: uploaded image names, "frame:name" comma separated, the same size
    as the reference image. Writes <prefix>_scale.json: scale, median_px, matches, per_frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "moge_geometry": ("MOGE_GEOMETRY",),
                "image": ("IMAGE",),
                "frames": ("STRING", {"default": ""}),
                "cameras": ("STRING", {"default": "{}", "multiline": True}),
                "filename_prefix": ("STRING", {"default": "geo_projection/scale"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "solve"
    OUTPUT_NODE = True
    CATEGORY = "nuke_geo_projection"

    def solve(self, moge_geometry, image, frames, cameras, filename_prefix):
        import cv2
        from PIL import Image

        cams = json.loads(cameras)
        pts = moge_geometry["points"].float().cpu().numpy()[0]
        ref = (image[0, ..., :3].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Hi, Wi = ref.shape[:2]
        H, W = pts.shape[:2]
        # Low contrast threshold + local equalisation: plates often have soft, low-contrast texture.
        sift = cv2.SIFT_create(4000, contrastThreshold=0.01)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        gray = lambda rgb: clahe.apply(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        kr, dr = sift.detectAndCompute(gray(ref), None)
        mref = np.array(cams["frames"][str(cams["ref"])]["matrix"], float).reshape(4, 4)
        lifted, seen, per_frame = [], [], {}
        for item in [s for s in frames.split(",") if s.strip()]:
            f, name = item.strip().split(":", 1)
            img = np.array(Image.open(folder_paths.get_annotated_filepath(name)).convert("RGB").resize((Wi, Hi)))
            kf, df = sift.detectAndCompute(gray(img), None)
            if dr is None or df is None:
                continue
            good = [m for m, n in cv2.BFMatcher().knnMatch(dr, df, k=2) if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                continue
            a = np.float32([kr[m.queryIdx].pt for m in good])
            b = np.float32([kf[m.trainIdx].pt for m in good])
            _, inl = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, 1.5, 0.999)
            if inl is not None:
                a, b = a[inl.ravel() > 0], b[inl.ravel() > 0]
            # MoGe point at each reference pixel (OpenCV axes) -> Nuke camera axes.
            ix = np.clip((a[:, 0] * W / Wi).astype(int), 0, W - 1)
            iy = np.clip((a[:, 1] * H / Hi).astype(int), 0, H - 1)
            p = pts[iy, ix]
            ok = np.isfinite(p).all(1) & (p[:, 2] > 0)
            cam = cams["frames"][f]
            mf = np.array(cam["matrix"], float).reshape(4, 4)
            fpx = cam["focal"] / cam["haperture"] * Wi
            lifted.append((p[ok] * [1, -1, -1], b[ok], mf, fpx))
            seen.append(f)

        def errors(s, items):
            out = []
            for xc, uv, mf, fpx in items:
                w = (mref[:3, :3] @ (s * xc).T).T + mref[:3, 3]
                c = (mf[:3, :3].T @ (w - mf[:3, 3]).T).T
                front = c[:, 2] < -1e-6
                u = Wi / 2 + fpx * c[:, 0] / -c[:, 2]
                v = Hi / 2 - fpx * c[:, 1] / -c[:, 2]
                e = np.hypot(u - uv[:, 0], v - uv[:, 1])
                out.append(np.where(front, e, 1e6))
            return np.concatenate(out) if out else np.zeros(0)

        def best(items):
            grid = np.logspace(-3, 3, 601)
            med = [np.median(errors(s, items)) for s in grid]
            k = int(np.argmin(med))
            fine = np.logspace(np.log10(grid[max(k - 1, 0)]), np.log10(grid[min(k + 1, len(grid) - 1)]), 201)
            med = [np.median(errors(s, items)) for s in fine]
            k = int(np.argmin(med))
            return float(fine[k]), float(med[k])

        result = {"scale": None, "median_px": None, "matches": int(sum(len(x[0]) for x in lifted)),
                  "per_frame": {}}
        if result["matches"] >= 8:
            result["scale"], result["median_px"] = best(lifted)
            for f, item in zip(seen, lifted):
                if len(item[0]) >= 8:
                    s, e = best([item])
                    result["per_frame"][f] = {"scale": s, "median_px": e, "matches": len(item[0])}
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), Wi, Hi)
        fname = "%s_%05d_.json" % (name, counter)
        with open(os.path.join(out_dir, fname), "w") as fh:
            json.dump(result, fh, indent=2)
        return {"ui": {"geo": [{"filename": fname, "subfolder": subfolder, "type": "output"}]}}


class SharpSplat:
    """Gaussian splat of the plate with Apple's SHARP (comfyui-sharp), at MoGe's focal so the
    splat lines up with MoGe's point map (same rays; the depth scale differs and is matched later,
    in the fit). SHARP's own focal input is a 35 mm equivalent on the diagonal, so MoGe's pixel
    focal is converted. Writes <prefix>_NNNNN_.ply (standard 3DGS PLY, SHARP's camera space:
    OpenCV axes, x right, y down, z forward; colours sRGB as SHARP exports them)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sharp_model": ("SHARP_MODEL",),
                "image": ("IMAGE",),
                "moge_geometry": ("MOGE_GEOMETRY",),
                "filename_prefix": ("STRING", {"default": "geo_projection/splat"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "predict"
    OUTPUT_NODE = True
    CATEGORY = "nuke_geo_projection"

    def predict(self, sharp_model, image, moge_geometry, filename_prefix):
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get("SharpPredict")
        if cls is None:
            raise RuntimeError("SharpSplat needs comfyui-sharp (SharpPredict) installed.")
        H, W = int(image.shape[1]), int(image.shape[2])
        K = _get(moge_geometry, "intrinsics", 1)
        if K is None:
            raise RuntimeError("SharpSplat: the MoGe geometry has no intrinsics.")
        f_px = float(K[0][0, 0]) * W                 # MoGe: fx normalized by the width
        f_mm = f_px * math.hypot(36.0, 24.0) / math.hypot(W, H)
        out_root = folder_paths.get_output_directory()
        out_dir, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, out_root, W, H)
        os.makedirs(out_dir, exist_ok=True)          # SharpPredict does not create subfolders
        rel = os.path.relpath(os.path.join(out_dir, "%s_%05d" % (name, counter)), out_root)
        # ComfyUI's partial loading moves SHARP's modules but not its bare parameters (cls tokens,
        # layer scales): when VRAM is short its forward fails on mixed cuda / cpu tensors. Free
        # the other models (SegmentLabels parks SAM3 on the CPU) and load SHARP (2.7 GB) whole.
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        mm.load_models_gpu([sharp_model], force_full_load=True)
        res = cls().predict(model=sharp_model, image=image[:1], focal_length_mm=f_mm, output_prefix=rel)
        path = res[0] if isinstance(res, tuple) else res["result"][0]
        return {"ui": {"geo": [{"filename": os.path.basename(path), "subfolder": subfolder, "type": "output"}]}}


def _get(geo, key, B):
    v = geo.get(key)
    if v is None:
        return None
    a = v.float().cpu().numpy() if hasattr(v, "float") else np.asarray(v)
    if a.shape[0] != B:
        a = np.broadcast_to(a, (B,) + a.shape[1:])
    return a


NODE_CLASS_MAPPINGS = {"SaveMoGeGeometry": SaveMoGeGeometry, "SegmentLabels": SegmentLabels,
                       "SaveDA3Geometry": SaveDA3Geometry, "SolveTrackScale": SolveTrackScale,
                       "SharpSplat": SharpSplat}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveMoGeGeometry": "Save MoGe Geometry (float EXR/NPZ)",
                              "SegmentLabels": "Segment Labels (SAM3, one model)",
                              "SaveDA3Geometry": "Save DA3 Geometry (NPZ)",
                              "SolveTrackScale": "Solve Track Scale (MoGe vs tracked camera)",
                              "SharpSplat": "SHARP Splat at MoGe focal (PLY)"}
