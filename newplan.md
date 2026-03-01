# TRIFFID UGV Plan (ROS2 Humble + Rosbag)

Goal: Produce **3D detections** (class, confidence, persistent ID, XYZ) from the available rosbags **without map/GPS**, publishing in `b2/base_link`.

This plan works with:
- USB RGB camera (`/camera_front/raw_image`, frame `f_oc_link`)
- RealSense depth (`/camera_front/realsense_front/depth/image_rect_raw`, frame `f_depth_optical_frame`, `16UC1` mm)
- TF static transforms (`/tf_static`) linking `b2/base_link` ↔ camera frames

---


general depth pipeline


Detect on /camera_front/raw_image (RGB)
For each box:
sample small set of depth pixels in depth image (grid)
transform those points to f_oc_link (tf frame attached to the front USB RGB camera) 
using CameraInfo intrinsics f_depth_optical_frame (fx_d, fy_d, cx_d, cy_d) -> 3D point in Depth camera frame
Xd = (ud - cx_d) * Z / fx_d
Yd = (vd - cy_d) * Z / fy_d
Zd = Z
then transform 3D point into the RGB camera frame (f_oc_link) using TF2 -> 3D point expressed in the RGB camera coordinate system
project to RGB, using intrinsics from /camera_front/camera_info -> pixel location in the RGB image where that depth point would appear
u = fx_rgb * (X / Z) + cx_rgb
u = fx_rgb * (X / Z) + cx_rgb
keep the ones that project inside the detected box
median point -> object position
transform final object point to b2/base_link using TF2
f_oc_link -> b2/base_link
Publish in b2/base_link frame and /ugv/perception/detections_3d topic



## 0) Inputs / Outputs (Frozen Interface)

### Inputs (subscribe)
- `/camera_front/raw_image` (`sensor_msgs/Image`, `bgr8`)  
- `/camera_front/camera_info` (`sensor_msgs/CameraInfo`)  
- `/camera_front/realsense_front/depth/image_rect_raw` (`sensor_msgs/Image`, `16UC1`)  
- `/camera_front/realsense_front/depth/camera_info` (`sensor_msgs/CameraInfo`)  
- `/tf`, `/tf_static` (TF2 transforms)

### Output (publish)
- `/ugv/perception/detections_3d` (`vision_msgs/Detection3DArray`)
  - `header.frame_id = "b2/base_link"`
  - Units: **meters**
  - Timestamp: **copied from RGB header** (documented choice)

### Additional for mapping partner
- `/dog_odom` (`nav_msgs/Odometry`) to build a semantic map in `odom` (since there is no `map` frame in bag)

---

## 1) Confirmed Facts from Bag

- Depth frame: `f_depth_optical_frame`
- Depth encoding: `16UC1` (millimetres) → convert `Z_m = Z_mm / 1000.0`
- RGB frame: `f_oc_link`
- TF static contains:
  - `b2/base_link -> f_oc_link`
  - `b2/base_link -> f_depth_optical_frame`

No `map` frame and no GPS `/fix` data in current rosbags.

---

## 2) High-Level Pipeline

### Why we cannot sample depth at RGB pixel
RGB and depth come from different sensors/resolutions, so `(u,v)` in RGB does not correspond to `(u,v)` in depth.

### Robust association strategy (fast demo version)
For each RGB detection bbox:
1. Sample a **small set of depth pixels** (grid) from depth image
2. Convert sampled depth pixels → 3D points in `f_depth_optical_frame` (pinhole + depth intrinsics)
3. Transform those 3D points → `f_oc_link` using TF2
4. Project 3D points → RGB pixels using RGB intrinsics
5. Keep only points whose projected pixel lies inside the bbox
6. Median of remaining 3D points = object position (in `f_oc_link`)
7. Transform object position → `b2/base_link`
8. Assign persistent ID (IoU tracking on RGB bboxes)
9. Publish `Detection3DArray`

---

## 3) Math Details

### 3.1 Depth pixel → 3D in depth camera frame
Given depth pixel `(ud, vd)` and depth `Z` (meters), with depth intrinsics `(fx_d, fy_d, cx_d, cy_d)` from `/camera_front/realsense_front/depth/camera_info`:

Xd = (ud - cx_d) * Z / fx_d
Yd = (vd - cy_d) * Z / fy_d
Zd = Z


### 3.2 Transform depth-3D → RGB camera frame
Use TF2 to transform a `geometry_msgs/PointStamped` from `f_depth_optical_frame` to `f_oc_link`.

### 3.3 Project 3D in RGB frame → RGB pixel
Using RGB intrinsics from `/camera_front/camera_info`:
- fx = 500
- fy = 500
- cx = 640
- cy = 360

For point `(Xr, Yr, Zr)` in `f_oc_link`:


u = fx * (Xr / Zr) + cx
v = fy * (Yr / Zr) + cy



Only keep if:
- `Zr > 0`
- `0 <= u < 1280`, `0 <= v < 720`
- bbox.xmin <= u <= bbox.xmax AND bbox.ymin <= v <= bbox.ymax

### 3.4 Transform object 3D → base_link
Transform the final median point from `f_oc_link` to `b2/base_link` with TF2.

---

## 4) Depth Sampling Strategy (to keep it fast)

Instead of using all 640×480 depth pixels:

For each bbox:
- Compute bbox center `(uc, vc)` in RGB
- Sample a fixed number of depth pixels in a **coarse grid** (e.g., 12×12 = 144 points) over the entire depth image OR over a selected ROI.

Recommended fast method:

1) Use TF to know rough relative camera placement (already present).
2) Sample depth pixels on a grid:
   - `ud ∈ {0, 64, 128, ..., 576}`
   - `vd ∈ {0, 48, 96, ..., 432}`

This yields ~100–200 candidate 3D points per frame, enough for a proof-of-function.

Optional improvement:
- If performance allows, sample a denser grid (e.g., 20×15 = 300 points).

---

## 5) Tracking / Persistent IDs

Tracking runs in RGB image space:
- Use IoU matching between current frame bboxes and previous frame bboxes
- Assign persistent integer ID per track
- Never reuse IDs
- Tracks can expire after N frames not seen (retain ID history if required)

Output includes track ID in:
- `Detection3D.results[0].hypothesis.class_id` (class)
- store ID in `Detection3D.id` if available, or use `results[0].hypothesis.score` + custom field if you use a custom message

(If standard message cannot carry ID cleanly, define a custom message type for production; for demo, keep ID in `Detection3D.id` if supported.)

---

## 6) ROS2 Node Execution (with rosbag)

### Play bag

ros2 bag play <bag_folder> --loop --clock


### Run perception node

ros2 run triffid_ugv_perception ugv_node


### Check output

ros2 topic echo /ugv/perception/detections_3d


---

## 7) Validation Checklist (for integration milestone)

- [ ] Node runs fully from rosbag (no hardware)
- [ ] Depth converted from mm to meters (`16UC1`)
- [ ] TF2 transforms succeed:
  - `f_depth_optical_frame -> f_oc_link`
  - `f_oc_link -> b2/base_link`
- [ ] Output `header.frame_id = b2/base_link`
- [ ] Output timestamp copied from RGB header stamp
- [ ] IDs persist across frames

---

## 8) Known Constraints (accepted for “works today”)

- No `map` frame: partner uses `/dog_odom` + your `b2/base_link` detections to build a semantic map in `odom`.
- No GPS: no GeoJSON absolute coordinates can be produced from these rosbags.
- Depth/RGB association is approximate due to sparse sampling; will be improved later by denser sampling or full projection method.

---

## 9) Next Improvements (after demo works)

- Increase depth sampling density or use full depth projection (optimized).
- Add ROI-based sampling to reduce compute.
- Upgrade tracker to ByteTrack.
- Add support for output in `odom` as optional secondary topic (derived from `/dog_odom` or TF if available).
