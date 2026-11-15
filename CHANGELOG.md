# Changelog

## Unreleased

- Renamed AutoProjection -> DBS_SceneProjection (node, menu, README, banner). Python modules,
  the `geoproj` tab and the `GeoProj_v###` Groups keep their names: old scripts still work.
- Info line says `track check n/a` for cameras not from a Nuke CameraTracker (FBX / Alembic
  from another tracker), instead of saying nothing.
- README: cameras tracked elsewhere (Alembic preferred, undistorted plate, FBX caveats), and
  what happens without a point cloud (PointCloudGenerator in NukeX, or the tracking team's
  locators). Checked: a camera and point cloud written to Alembic by Blender read back exactly.
- Silhouette check: each object's proxy against its SAM3 mask (`fit.json` verification
  `silhouette`: miss / spill per object). The info line flags proxies leaving out over 25% of
  their mask, and counts loose fits (over 50% of the footprint on background: open shapes).
  On the truck shot all objects pass; in the interior test, box tables and vases miss 45-50%
  (10% as shapes) and chairs are loose.
- The panel's info line is split into lines (it was cut off).
- No point cloud: SolveTrackScale also triangulates its matches with the tracked camera at the
  two frames (rays over 1° apart, under 2 px, within 1.5x of MoGe's depth) and those points
  stand in for the tracker's: snap and slide check without NukeX. Checked numerically (0.6%
  depth error at 1 px noise) and through the node on a synthetic wall (scale 2.0008 for 2.0).
  ComfyUI needs a restart to load it.
- Relief keeps a card per backdrop instance, pushed just behind that instance's mesh, as fill
  for its holes (was: only the far backdrop plane).
- Relief per label: `rock:object:relief`, `tree:backdrop:relief` make that label a relief mesh
  in any geometry mode. The far backdrop plane is pushed behind those meshes.
- Backdrop cards are cut to their instance's shape (mask grown 2 samples, plus down to the
  floor, never over anything farther). Rectangles stood in front of farther things: on the
  truck shot a `forest` card at depth 42 hid trees at 61-66, which slid 19-26 px. Tree tracker
  points on the geometry: median ratio 0.62-0.68 -> 0.997.
- **card from CornerPin** (button on the setup Group): a card on a selected CornerPin2D's four
  corners at the reference frame, with the pinned image projected onto it (backdrop beside
  the Group). A menu beside it: **standing, facing camera** (default: upright, turned to the
  camera, bottom edge where MoGe sees it) or **on the surface** (on a plane fitted to MoGe's
  points inside the corners). Corners reproject to 0 px; on the truck shot a checker pinned
  to the road stays on it through the shot.
- The extra-labels hint lists `relief`.

## 0.8.1

- Boxes give each face its own vertices, and round shapes their caps: with shared corners Nuke
  smoothed the normals across the edges, and `project on: front` left part of a face seen
  edge-on unprojected (e.g. the back of a truck).
- "extract for paint" Project nodes also use `project on: front` (they were left on `both`).
- Build checks for Blender and for comfyui-rmbg (SAM3) before sending anything to ComfyUI, so a
  missing one is reported at once instead of after MoGe has run.
- README: the first build with segmentation downloads SAM3 (about 3.4 GB).
- Track check: the CameraTracker's solve error in the info line (found through the camera's
  expressions or the point cloud), with a warning over 1 px `@HD`; for other trackers, a warning
  when over 30% of the tracker points disagree with the rest.
- Slide and track errors are judged in px of a 1920-wide frame (`@HD`), so the thresholds hold at
  any resolution; plates that are not 1920 wide show both values.

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
