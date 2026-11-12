# Changelog

## 0.8.0

- **Tracker points input** (`points`): the tracker's point cloud fixes the track scale directly
  and MoGe's depth is snapped to it region by region before fitting (`blender/track_snap.py`).
  Truck shot, 188 frames: projection slide 14 px -> 1.2 px median on the truck.
- Slide report: per-object slide in px over the plate's range in the info line and Script Editor.
- **Gaussian splat** (optional, USD): Apple SHARP via ComfyUI-Sharp (`SharpSplat` node), placed in
  the geometry's world; setup Group output menu projection / splat / projection over splat.
- Boxes put their visible faces on the object's own planes (not the extent of the points).
- Non-square pixel plates (DVCPRO HD, anamorphic) sent to ComfyUI with square pixels.
- Projections land on front faces only (`project_on front`).
- Classic 3D system: live **show only** / **pattern** isolate.
- Renamed **AutoProjection** (was GeoProjection). The node is `DBS_AutoProjection` (prefix of
  David Borrull Silva's tools), under a **DBS** toolbar button with its own icon. Python modules
  (`geo_projection`, `geoproj`) keep their names, so existing installs and scripts still work.
- **set to current frame** button next to the reference frame.
- SAM3 is moved off the GPU after use so SHARP loads fully on 8 GB cards.
- Clearer error when the plate input has no rgb (e.g. a render connected instead of the plate).

## 0.7.0

- Sequence mode: extra frames projected onto the same geometry, occluded, under the reference frame.
- Automatic track scale (`SolveTrackScale`): SIFT matches between frames lifted with MoGe.

## 0.6.0

- **extract for paint**: loose FrameHold -> RotoPaint -> Project nodes for one object.

## 0.5.x

- Geometry menu shapes / boxes / cards / relief; best-fit primitive per object; precision knob.
- Group hierarchy /structure /objects /backdrop /other, scene.usd; USD show only / pattern.
- Floor rejects planes that would roll the camera > 20°.

## 0.4.0

- The setup is one self-contained Group (copy / paste / rename safe).

## 0.3.0

- Relief meshes; USD / classic 3D system menu; shot camera window offset.

## 0.2.0

- Label presets, shot camera input, live scene scale.
