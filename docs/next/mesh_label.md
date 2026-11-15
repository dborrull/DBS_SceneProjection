# Next version: generated mesh per label (`car:object:mesh`)

Status: planned, not built. Discussed 2026-11-15.

## Why

Boxes, cylinders and relief come from MoGe's points of one frame. Small or round objects at a
depth edge (the truck's wheels) get points smeared along the view ray, and no fitted primitive
is right. An image-to-3D model gives a closed, plausible shape (round wheels, a car's volume,
back faces when the camera turns). It does not know where the object is or its exact size: that
still comes from MoGe, the track and the SAM3 mask.

## What the user does

- Extra label with the `mesh` shape: `car:object:mesh`, `statue:object:mesh`.
- Only the labels that ask for it are generated. The rest of the scene builds as today.
- Like `box` and `relief`, a label's shape wins over the geometry menu in every mode.
- Not a geometry menu mode: generating every object is minutes and VRAM per object, and floor,
  walls and backdrop cannot be generated anyway.

## Pipeline

1. **ComfyUI.** For each `mesh` instance, crop the plate with its SAM3 mask (padding about 10%,
   square, about 512 px) and run an image-to-3D node. Geometry only: the projection gives the
   texture, so no texture or paint stage (that is what does not fit in 8 GB).
2. **Blender, `blender/meshfit.py` (new).** Place the mesh in the fit world:
   - start pose: the model's own pose if it gives one (SAM 3D Objects), otherwise the box fitted
     to the instance (`make_object` box) for center, size and heading, with the mesh's up axis
     to Z;
   - refine scale, yaw and translation (similarity, roll and pitch locked to the floor unless
     the object floats) against two terms: MoGe points (trimmed nearest-point distance; ignore
     the worst 20%, they are the smeared edge points) and the mask silhouette from the
     reference camera (miss and spill as in the silhouette check);
   - rest it on what is under it, as other objects.
3. **Checks.** Silhouette check as today. If the mesh misses over 25% of its mask, or fits the
   points worse than 3x the box residual, use the box and say so in the info line
   (`car_1 mesh -> box (silhouette 38%)`).
4. **Output.** Decimate to about 20k faces, own vertices per hard edge so `project on: front`
   behaves (the smoothed-normal problem fixed in 0.8.1), into the Alembic / USD like any object
   under `/objects`.

## Model choice (to test)

| Model | ComfyUI | Notes |
|---|---|---|
| SAM 3D Objects (Meta) | ComfyUI-SAM3DObjects (PozzettiAndrea) | Takes image + mask, gives mesh **and pose**; has pose optimisation (ICP / render) nodes. First choice: we already have the SAM3 masks. |
| TRELLIS 2 | native / community nodes | 8 GB with low VRAM mode at 512, geometry only. |
| Hunyuan3D 2.1 | ComfyUI-Hunyuan3d-2-1 | Shape only reported at about 6-10 GB; texturing needs 20 GB+ (not needed). |

Installed already: ComfyUI-GeometryPack (mesh utilities, same author as SAM3DObjects). Not
installed: any of the three generators. Installing one needs David's OK (download size, custom
node code).

## Depends on

- **Reuse of ComfyUI results** (also on the next-version list): cache generated meshes in the
  run, keyed by plate frame, mask and model, so switching geometry mode or precision does not
  regenerate. Without it, every rebuild costs minutes per mesh.
- SAM3 and the generator must not sit in VRAM together (unload SAM3 first, as done for SHARP).

## Test plan

1. One object on a plate that is not the truck (stock footage, not for public images): a car or
   a statue, label `xxx:object:mesh`.
2. Measure: silhouette miss / spill, slide in px @HD against the box for the same object over
   the shot, generation time and peak VRAM on the 8 GB card.
3. Keep it only if the slide is not worse than the box. A plausible shape with wrong proportions
   slides more than a box fitted to the points.

## First test (2026-11-15)

Pixal3D (`pixal3d_int8_convrot`, native ComfyUI nodes, geometry only: structure -> shape ->
upsample 1024 -> `VaeDecodeShapeTrellis` -> `SaveGLB`; no texture stage) on the interior plate
(GeoProjection2/v001), two objects where primitives fail. RTX 4060 Ti 8 GB: about 1 minute
per object, no out-of-memory. SAM 3D Objects was skipped: gated weights on Hugging Face.

Placement (numpy prototype, no Blender needed): the mesh is pixel aligned. Its glTF front view
(Blender x right, z up, looking along y) matches the crop. Map it to the camera, turn it to the
ray through the crop centre, refine scale and image offset on the silhouette (mask holes filled:
they are occluders), then scale about the camera so its front surface sits on MoGe's median
depth. The crop's FOV is passed to `Pixal3DConditioning`. Left/right sign: pick the better
silhouette.

| Object | Old primitive | Generated mesh | Silhouette IoU | Depth vs MoGe (median) | Bottom |
|---|---|---|---|---|---|
| Tall sculpture | revolve, 1.82 m wide | 0.53 wide, 1.70 high (MoGe 1.70) | 0.93 | 0.6% | z 0.04 |
| Sofa | cylinder, 1.42 m | 1.00 x 1.09 x 0.65 | 0.92 | 1.1% | z 0.00 |

- Both rest on the floor without being told to: an independent check on scale and depth.
- The sofa mesh is complete behind the side table that occludes it.
- **Weak point: depth along the view ray is invented.** The sculpture is 1.46 m deep (side
  "wings"); the silhouette and MoGe's front surface do not constrain it. Needs a rule, e.g.
  limit depth to a multiple of width, or flatten the back to MoGe's points.
- Depth limit test (same day): three seeds give the sculpture the same depth, 3.4-4x its
  width, so it is not model noise: Pixal3D is sure of it, and one still cannot say who is
  right. Squashing depth along the ray to at most 1x width **before** placing (then refine the
  silhouette and depth as usual) keeps silhouette IoU 0.93 and improves depth vs MoGe (p90
  7.4% -> 2.6%): sculpture 0.40 x 0.44 x 1.68. Squashing after placing broke the fit (IoU
  0.85, 5% depth error). Sofa barely changes (depth 0.99 -> 0.92 of width). A fixed ratio is
  wrong for long objects seen end-on (a car from the front), so it cannot be the default.
  Proposal: place both candidates (as generated, squashed) and pick the one whose silhouette
  matches the SAM3 mask on a second frame of the tracked shot; on a still, keep the model's
  depth unless the label asks for the limit.
- Left/right sign is chosen by silhouette IoU, not by depth error (depth picked the mirrored
  mesh once).
- Meshes are 1.2-3 M vertices (42 MB GLB): decimate before Alembic.
- Not measured: slide over a moving shot (still image). Next test needs a tracked plate.

Prototype scripts were in the session scratchpad (`prep.py`, `make_wf.py`, `meshfit_proto.py`).
Models downloaded to C:\ComfyUI: `diffusion_models/pixal3d_int8_convrot.safetensors`,
`clip_vision/dino_v3_L_naf_fp32.safetensors`, `vae/trellis_2_shape_vae_bf16.safetensors`.

## Open questions

- Fit a generated mesh by points alone when the model gives no pose: symmetric objects (wheels,
  vases) leave yaw free; take yaw from the box.
- Thin parts (mirrors, antennas) that MoGe cannot see: keep them or cut them by the mask?
- Wheels specifically may be cheaper as cylinders anchored to the vehicle box than as
  generated meshes.

## Wider test (closed-remesh version, interior shot)

Six Pixal3D meshes on the interior plate, Blender step 28 s for all six.

| object | IoU | depth vs MoGe | result |
|---|---|---|---|
| chair, whole (inst 30, thin legs) | 0.91 | 1.8% | placed |
| chair cut by the left edge (inst 35) | 0.65 | 2.4% | placed, 28% spill |
| sculpture / vase (inst 60) | 0.89 | 0.5% | placed |
| sofa (inst 46) | 0.76 | 2.9% | placed (0.87 before the remesh) |
| hanging lamp (inst 57) | 0.10 | - | fallback to box |
| plant (inst 63) | - | - | not reached: not kept as an object |

With the truck shot: 5 of 7 placed, 2 fallbacks (cut-off truck, lamp).
Whole, well-seen objects work; cut-off and hanging/thin ones do not.
The 64-voxel remesh costs the wide sofa 11 IoU points.
