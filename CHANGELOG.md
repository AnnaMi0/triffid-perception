# Changelog

All notable changes to the TRIFFID Perception pipeline are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.2.0] — 2026-04-22

### Added — RealSense D435i Live Camera Support

- **YUYV colour encoding** (`ugv_node.py`): explicit `yuv422`/`yuv422_yuy2`/`YUV422` check in `rgb_callback`. YUYV frames are converted via `cv2.COLOR_YUV2BGR_YUY2` directly from the raw buffer (without `cv_bridge`, which may lack this conversion in older driver builds). All other encodings (including `bgr8` from rosbag) continue to go through `cv_bridge` — no regression on the existing test flow.

- **Camera topic overrides** (`run.sh`): three optional env vars pass through to the `ugv_node` ROS parameters:
  - `RGB_TOPIC`, `DEPTH_TOPIC`, `CAMERA_INFO_TOPIC` — leave empty for rosbag mode (preserves default `/b2/camera/` namespace); set to the 435i topics for live operation:
    ```
    RGB_TOPIC=/camera_front_435i/realsense_front_435i/color/image_raw
    DEPTH_TOPIC=/camera_front_435i/realsense_front_435i/depth/image_rect_raw
    CAMERA_INFO_TOPIC=/camera_front_435i/realsense_front_435i/color/camera_info
    ```
  - Depth alignment intentionally disabled for PoC (`align_depth.enable: False` on Jetson); raw `/depth/image_rect_raw` is used.

- **`./run.sh camtest`**: one-shot frame-grab command that:
  1. Subscribes to the live colour and depth topics (defaults to 435i topics; overrideable via `CAMTEST_RGB_TOPIC` / `CAMTEST_DEPTH_TOPIC` — independent from `RGB_TOPIC`/`DEPTH_TOPIC` to avoid picking up rosbag replay overrides from the shell environment)
  2. Applies YUYV→BGR conversion (same logic as `ugv_node`)
  3. Saves `samples/camtest_color.png` and a false-colour `samples/camtest_depth.png`
  4. Exits with code 0 on success, 1 on timeout (10 s default)
  - New file: `src/triffid_ugv_perception/scripts/camtest.py`

### Added — MQTT Port Configurability

- **`MQTT_PORT` env var** (`run.sh`, `docker-compose.yml`): Mosquitto broker port defaults to `1883` but is configurable when another user on the shared host occupies that port:
  ```bash
  MQTT_PORT=1884 ./run.sh start
  ```
  - `run.sh`: reads `MQTT_PORT` and passes it to `mosquitto -p $MQTT_PORT` and to `geojson_bridge --ros-args -p mqtt_port:=$MQTT_PORT`
  - `docker-compose.yml`: bare `- MQTT_PORT` passthrough in `environment:` block
  - `run_uav.sh` already handled this correctly; only the UGV side needed updating

### Added — GeoJSON Pipeline: Class Name Normalisation + Spatial Deduplication

- **Lowercase class names**: all 63 `TARGET_CLASSES` values lowercased at source in both `ugv_node.py` and `uav_node.py`. All lookup dict keys (`_POINT_CLASSES`, `_LINE_CLASSES`, `_CLASS_COLORS`, `_CLASS_CATEGORIES`, `_CLASS_SYMBOLS`) updated to match in both packages.

- **Spatial deduplication in `geojson_bridge`** (`geojson_bridge.py`): before publishing each UGV FeatureCollection, overlapping features of the same class are merged (highest confidence kept):
  - Algorithm: greedy, sort by confidence descending, suppress any feature within `dedup_radius_m` metres of an already-kept feature of the same class
  - Haversine distance used for GPS-projected features; `local_frame: true` features always pass through unchanged
  - New ROS parameter: `dedup_radius_m` (default `3.0 m`)
  - New module-level helpers: `_haversine_m()`, `_feature_centroid()`, `_deduplicate_features()`

- **Cross-platform deduplication in bridge** (`bridge.py`): `_merge()` now deduplicates after combining UGV + UAV features using the same algorithm with a larger 10 m radius (`_CROSS_DEDUP_RADIUS_M`) to account for UAV geo-projection uncertainty. `local_frame: true` UGV features always pass through.

- **Class name normalisation in bridge** (`bridge.py`): `Bridge._normalize()` lowercases the `class` property of every incoming feature in `_on_message()`, making the bridge the single point of truth for class name casing regardless of source.

### Added — TELESTO Bridge: Auto-Start + Smoke Test

- **Bridge wired into `run.sh`**: `cmd_start()` now starts `bridge.py` as a daemon after mosquitto, `cmd_stop()` kills it. Log at `/tmp/bridge.log`, shown in `./run.sh logs`. New env vars `TELESTO_BASE_URL` and `TELESTO_OBSERVER_URL` passed through `docker-compose.yml` to override the built-in endpoint defaults.

- **`./run.sh bridge-test`**: publishes mock UGV + UAV GeoJSON to the local MQTT broker, starts a short `--dry-run` bridge session, and prints the merged deduplicated output — no Telesto upload needed.

- **`smoke_test.py`** (`src/triffid_telesto/smoke_test.py`): standalone smoke test that publishes a 4-feature scenario (UGV fire + UAV fire overlap, local_frame debris, unique vehicle) and asserts the bridge produces exactly 3 features with correct source and confidence.

### Fixed

- **Integration test crash on `./run.sh test`** (`integration_test.py`): `shutil.rmtree('/ws/output_rosbag')` failed with `OSError: [Errno 16] Device or resource busy` because `/ws/output_rosbag` is a Docker volume mount point. Fixed to clear the directory contents without deleting the mount point itself.

- **No detections in rosbag mode** (`geojson_bridge.py`): GPS validity gate (`if not self.gps_valid: return`) completely blocked GeoJSON publishing when no `/fix` topic was present (rosbag datasets typically lack GPS). Hard block removed; node now logs a throttled warning and publishes with `"local_frame": true` — the property was already set correctly, features were just never reaching the publisher.

### Changed

- **Test suite** — 252 total (was 214 across UGV + UAV, plus 30 Telesto):
  - UGV `test_unit.py`: all class name literals updated to lowercase — 164 tests, count unchanged
  - UAV `test_unit.py`: class name literals updated to lowercase — 50 tests, count unchanged
  - Telesto `test_telesto.py`: 30 → 38 tests; added `TestBridgeCrossPlatformDedup` (6 tests covering same-class overlap, different-class non-overlap, distance threshold, `local_frame` passthrough, source ordering) and two normalization tests in `TestBridgeOnMessage`; fixed `test_merge_both_sources` to use distinct classes/locations; updated `_point_feature` fixture to accept `confidence` and `local_frame` parameters and use lowercase class defaults

---

## [2.1.0] — 2026-04-13

### GeoJSON Field Alignment

- **Renamed** `ellipsoidal_alt_m` (UGV) and `gnss_altitude_m` (UAV) to unified `altitude_m` across both modules
- Both UGV and UAV now produce identical GeoJSON property schemas

### TELESTO Integration

- Added `triffid_telesto` package (`src/triffid_telesto/`)
- `telesto_client.py`: `TelestoClient` with GET/PUT/PATCH/DELETE for Map Manager API, `sync_collection()` for smart sync, `clear_source()`, observer notifications
- `bridge.py`: MQTT→TELESTO bridge — subscribes to UGV + UAV GeoJSON topics, merges features, syncs to TELESTO backend with observer notification
- 30 unit tests (client CRUD, sync logic, bridge merge, field alignment)

---

## [2.0.0] — 2026-04-13

### Summary

**BREAKING**: Replaced the cross-camera depth–RGB fusion pipeline with a pixel-aligned RGB-D pipeline. The UGV node now assumes the depth image is aligned to the colour image (e.g. Intel RealSense `aligned_depth_to_color`), sharing the same pixel grid, resolution, and intrinsics. This eliminates the need for separate depth/RGB camera frames, grid sampling, cross-camera TF transforms, and body↔optical coordinate conversion within the node.

---

### Changed — `ugv_node.py` (main perception node)

#### Module-level constants

| Before | After |
|--------|-------|
| `DEPTH_FRAME = 'f_depth_optical_frame'` | **Removed** |
| `RGB_FRAME = 'f_oc_link'` | **Removed** |
| `DEFAULT_GRID_STEP_U = 64` | **Removed** |
| `DEFAULT_GRID_STEP_V = 48` | **Removed** |
| — | `DEFAULT_RGB_TOPIC = '/b2/camera/color/image_raw'` |
| — | `DEFAULT_DEPTH_TOPIC = '/b2/camera/aligned_depth_to_color/image_raw'` |
| — | `DEFAULT_CAMERA_INFO_TOPIC = '/b2/camera/color/camera_info'` |
| — | `_MAX_DEPTH_SAMPLES = 500` |

**Practical effect**: The node no longer hard-codes camera frame IDs. Topic names are configurable via ROS parameters. The maximum depth sample count caps computation for large segmentation masks.

#### Class docstring

Changed from "Cross-camera depth–RGB fusion pipeline" to "Pixel-aligned RGB-D perception pipeline" with updated step descriptions reflecting the simplified flow.

#### ROS parameters

| Before | After |
|--------|-------|
| `depth_grid_step_u` (int, default 64) | **Removed** |
| `depth_grid_step_v` (int, default 48) | **Removed** |
| — | `rgb_image_topic` (string, default `/b2/camera/color/image_raw`) |
| — | `depth_image_topic` (string, default `/b2/camera/aligned_depth_to_color/image_raw`) |
| — | `camera_info_topic` (string, default `/b2/camera/color/camera_info`) |

**Practical effect**: Users can now configure which camera topics the node subscribes to via ROS parameters, rather than changing source code. Grid step tuning is no longer needed since depth is sampled per-detection rather than globally.

#### Node state

| Before | After |
|--------|-------|
| `self.rgb_camera_info` | **Removed** |
| `self.depth_camera_info` | **Removed** |
| — | `self.camera_info` (single shared CameraInfo) |
| — | `self.camera_frame` (read from CameraInfo header.frame_id) |

**Practical effect**: One CameraInfo instead of two. Camera frame ID is discovered at runtime from the CameraInfo message header, not hard-coded.

#### Subscribers

| Before | After |
|--------|-------|
| 2 CameraInfo subscribers (RGB + depth) | 1 CameraInfo subscriber (shared) |
| Hard-coded topic names | Topic names from ROS parameters |
| `rgb_info_callback` + `depth_info_callback` | Single `camera_info_callback` |

**Practical effect**: Subscribes to 3 topics (RGB, depth, CameraInfo) instead of 4 (RGB, depth, RGB CameraInfo, depth CameraInfo). Callback also extracts the camera frame ID from the header.

#### `rgb_callback` (main processing pipeline)

**Before** (8-step cross-camera pipeline):
1. YOLO detection on RGB
2. Grid-sample the depth image (coarse grid, ~100 points) → back-project to `f_depth_optical_frame`
3. TF batch transform `f_depth_optical_frame → f_oc_link` (matrix multiply)
4. Pinhole project all depth points to RGB pixel coordinates (body→optical conversion + `CameraInfo.K`)
5. For each detection, filter depth points whose projected pixel falls inside the mask/bbox
6. Median 3D position in `f_oc_link`, TF to `b2/base_link`
7. Extent estimation from point cloud spread or bbox back-projection
8. ByteTrack → publish

**After** (7-step pixel-aligned pipeline):
1. YOLO detection on RGB
2. Resolution check: verify depth and RGB share the same resolution
3. For each detection, `_sample_depth_for_detection()` reads depth at mask/bbox pixels and back-projects directly to camera_optical_frame using shared intrinsics
4. Median 3D position in camera_optical_frame, TF to `b2/base_link`
5. Extent estimation from point cloud spread or bbox back-projection (optical convention)
6. ByteTrack → publish

**Practical effect**: No global depth grid, no cross-camera TF lookup, no body→optical conversion, no separate projection step. Depth is sampled only where detections exist (more efficient, more accurate — no background depth contamination from grid points landing between objects).

**New guard**: If depth and RGB resolutions differ, the frame is skipped with a warning. This catches misconfigured `aligned_depth_to_color` streams.

#### Removed methods

| Method | Purpose | Why removed |
|--------|---------|-------------|
| `_depth_grid_to_rgb_frame()` | Grid-sample depth → back-project → TF to RGB frame | No grid sampling needed; depth samples per-detection instead |
| `_project_to_rgb()` | Project 3D points to RGB pixel plane with body→optical conversion | No projection needed; depth pixels already correspond to RGB pixels |

#### Added methods

| Method | Purpose |
|--------|---------|
| `_sample_depth_for_detection(det, depth_img, fx, fy, cx, cy)` | Sample depth at detection mask/bbox pixels, back-project to camera_optical_frame. Handles mask vs bbox fallback, sub-sampling for large masks (capped at `_MAX_DEPTH_SAMPLES`), zero-depth rejection. Returns (M, 3) array in optical frame or None. |

#### Modified methods

| Method | Change |
|--------|--------|
| `_bbox_to_3d_corners()` | **Signature changed**: now takes explicit `fx, fy, cx, cy` parameters instead of reading from `self.rgb_camera_info`. **Convention changed**: outputs in camera_optical_frame (X=right, Y=down, **Z=forward** = depth) instead of body frame (X=forward, Y=left, Z=up). The `depth` parameter is now the Z component (forward distance in optical frame), not the X component (forward distance in body frame). |

---

### Changed — `ugv_perception.launch.py`

#### Removed: 4 static TF publisher nodes

| Node | Frames | Why removed |
|------|--------|-------------|
| `tf_base_to_rgb` | `b2/base_link → f_oc_link` | No separate RGB camera body frame |
| `tf_base_to_dc` | `b2/base_link → f_dc_link` | No separate depth camera mount frame |
| `tf_dc_to_depth` | `f_dc_link → f_depth_frame` | No depth frame chain |
| `tf_depth_to_optical` | `f_depth_frame → f_depth_optical_frame` | No depth optical frame |

**Practical effect**: The launch file now starts only `ugv_node` and `geojson_bridge`. The camera→base_link transform must be provided externally by the robot's localization stack (Angel's TF tree: `map → b2/map → b2/odom → b2/base_link → b2/camera_optical_frame`).

#### Removed: launch arguments

- `depth_grid_step_u`
- `depth_grid_step_v`

#### Added: launch arguments

- `rgb_image_topic` (default: `/b2/camera/color/image_raw`)
- `depth_image_topic` (default: `/b2/camera/aligned_depth_to_color/image_raw`)
- `camera_info_topic` (default: `/b2/camera/color/camera_info`)

#### Updated: node parameter mapping

The `ugv_perception_node` parameter dictionary now passes `rgb_image_topic`, `depth_image_topic`, `camera_info_topic` instead of `depth_grid_step_u/v`.

---

### Changed — `test_unit.py`

#### Removed test classes (18 tests removed)

| Class | Tests | Why removed |
|-------|-------|-------------|
| `TestProjectToRGB` | 6 | Tested `_project_to_rgb()` which no longer exists |
| `TestBackProjection` | 5 | Tested back-projection for the old separate depth camera |
| `TestDepthGridSampling` | 5 | Tested depth grid creation logic which no longer exists |
| `TestCrossCameraPipeline` | 6 | Tested end-to-end cross-camera geometry which no longer exists |
| `TestMaskBasedDepthMatching` | 4 | Tested mask filtering of projected grid points (replaced by per-detection sampling) |
| `TestBodyToOpticalConversion` | 6 | Tested body→optical axis conversion which is no longer done in the node |

Note: Some of these classes (TestBackProjection, TestDepthGridSampling, TestCrossCameraPipeline, TestMaskBasedDepthMatching) test pure math that is still valid — they were removed because they test patterns no longer in the codebase. The underlying math (pinhole model, grid construction, mask filtering) is well-established.

#### Added test classes (15 tests added)

| Class | Tests | What it tests |
|-------|-------|---------------|
| `TestBackProjectionOptical` | 7 | Pinhole back-projection in camera_optical_frame (X=right, Y=down, Z=forward) |
| `TestSampleDepthForDetection` | 6 | `_sample_depth_for_detection()` — mask sampling, bbox fallback, zero depth, empty mask, principal point, sub-sampling cap |
| `TestPixelAlignedPipeline` | 3 | End-to-end: pixel-aligned depth → back-project → median position |

#### Reworked test classes (existing classes updated)

| Class | Changes |
|-------|---------|
| `TestFrameConstants` | Removed `test_depth_frame_id` (DEPTH_FRAME), `test_rgb_frame_id` (RGB_FRAME), `test_grid_step_defaults`. Added `test_default_rgb_topic`, `test_default_depth_topic`, `test_default_camera_info_topic`, `test_max_depth_samples`. |
| `TestIntrinsicsConsistency` | Replaced dual-camera (RGB + depth) intrinsic checks with single shared-camera checks. Removed `test_depth_*`, `test_rgb_*` individual tests. Added `test_principal_point_inside_image`, `test_focal_lengths_positive`, `test_k_matrix_last_row`, `test_near_square_pixel`, `test_principal_point_near_centre`. |
| `TestBboxTo3DCorners` | Updated for new method signature (explicit `fx, fy, cx, cy` params). Changed assertions from body convention (X=depth, Y/Z extent) to optical convention (Z=depth, X/Y extent). |
| `TestPipelineMathValidation` | Removed `test_depth_grid_count_matches_expectation` (grid no longer exists). |

#### Test count: 182 → 164

| Category | Before | After | Delta |
|----------|--------|-------|-------|
| Removed (obsolete cross-camera) | 32 | 0 | −32 |
| Added (pixel-aligned) | 0 | 15 | +15 |
| Reworked (constants/intrinsics/bbox) | — | — | −1 (removed grid count test) |
| Unchanged | 150 | 149 | −1 |
| **Total** | **182** | **164** | **−18** |

---

### Changed — Documentation

#### `README.md`

- **Description**: "Fuses RGB and depth from separate cameras via cross-camera projection" → "Uses a pixel-aligned RGB-D camera (Intel RealSense with depth aligned to colour)"
- **Architecture diagram**: Replaced dual-camera ASCII art (separate RGB USB + depth RealSense) with single RGB-D camera diagram showing pixel-aligned streams and configurable topic names
- **Hardware Assumptions**: Replaced two-column table (Front RGB vs Front Depth) with single-camera table; noted pixel-alignment and shared intrinsics
- **Subscribed topics**: Updated from `/camera_front/*` to `/b2/camera/*` with note about configurability
- **TF Frame Tree**: Replaced 4-level static chain with single dynamic frame from CameraInfo; removed body↔optical conversion documentation
- **Pipeline Steps**: Replaced 8 steps (grid sample → TF batch transform → pinhole project → mask filter → median) with 7 steps (per-detection depth sample → back-project → median); added resolution check step
- **Node description**: Updated to reflect pixel-aligned camera, configurable topics, dynamic camera frame
- **Parameters table**: Replaced `depth_grid_step_u/v` with `rgb_image_topic`, `depth_image_topic`, `camera_info_topic`
- **Detection3DArray**: Removed "(v1.6)" tracker fix annotation from `bbox.size` description
- **Debug image**: "depth grid points" → "depth pixels"
- **Launch instructions**: Removed "ugv_node + static TF publishers" reference; added note about external TF requirement
- **Rosbag required topics**: Updated to new topic names; added legacy bag compatibility note
- **Project structure**: Updated test count (182 → 164); launch file comment ("Launch all nodes + static TFs" → "Launch nodes (no static TFs)"); ugv_node.py comment ("Main perception node" → "Main perception node (pixel-aligned RGB-D)")

#### `INTERFACE.md`

- **Version**: 1.9 → 2.0
- **Date**: 2026-03-24 → 2026-04-13
- **ROS 2 Distribution**: Humble → Jazzy
- **Subscribed Topics**: Updated from 4 camera topics to 3 (shared CameraInfo); new `/b2/camera/*` names; added configurability note
- **TF Frames**: Replaced 4-level static chain with single dynamic frame; documented Angel's TF tree; removed body↔optical conversion section
- **Segmentation header**: `f_oc_link` → "f_oc_link or camera frame from CameraInfo"
- **Debug image**: Updated depth point description
- **Detection3DArray notes**: Added note about camera_optical_frame convention
- **Changelog**: Added v2.0 entry with full breaking change description
- **Limitations**: Added #7 (pixel-aligned depth required) and #8 (external TF required)

---

### Migration Guide

For teams upgrading from v1.x (cross-camera) to v2.0 (pixel-aligned):

1. **Camera setup**: Configure the RealSense to publish `aligned_depth_to_color` — depth must share the same resolution and pixel grid as the colour stream.

2. **TF tree**: Provide a `camera_optical_frame → b2/base_link` transform externally (via robot driver, URDF, or static publisher). The 4 old static transforms are no longer published by the launch file.

3. **Topic names**: Update any rosbag playback or topic remapping to use the new defaults:
   - RGB: `/b2/camera/color/image_raw`
   - Depth: `/b2/camera/aligned_depth_to_color/image_raw`
   - CameraInfo: `/b2/camera/color/camera_info`
   
   Or override via launch arguments:
   ```bash
   ros2 launch triffid_ugv_perception ugv_perception.launch.py \
     rgb_image_topic:=/my/rgb \
     depth_image_topic:=/my/depth \
     camera_info_topic:=/my/camera_info
   ```

4. **Parameters**: Remove any `depth_grid_step_u/v` overrides from your launch configs (they will be ignored / cause "unknown parameter" warnings).

5. **Integration tests**: The integration test will need updated topic names and TF checks. The old frame IDs (`f_oc_link`, `f_depth_optical_frame`, etc.) no longer appear in the TF tree.

---

## [1.9.0] — 2026-03-24

- Renamed `gnss_altitude_m` GeoJSON property to `ellipsoidal_alt_m` to clarify it is WGS-84 ellipsoidal height (not geoid/MSL)
- `run.sh record` now records subscribed input topics alongside output topics

## [1.8.0] — 2026-03-24

- **Class gate**: Cross-class tracker matches now forbidden (cost +1e5), preventing ID hijacking when bboxes overlap across classes
- **Majority-vote class label**: Track class determined by most-frequent observation (not overwritten each frame)
- `geojson_raw.json` added to sample output (all confirmed tracks, no spatial dedup)

## [1.7.0] — 2026-03-23

- Added `/ugv/detections/front/debug_image` topic to interface spec (lazy-published `bgr8` debug overlay with bbox, class, track ID, and depth-point count `d=N`)
- Documented `samples/` output artefacts

## [1.6.0] — 2026-03-23

- Class-dependent geometry types (Point/LineString/Polygon) per 63-class ontology
- Wall added to LineString classes
- Polygon no longer falls back to Point when extent is zero (uses 0.3 m minimum)
- Tracker now propagates `extent`, `n_depth_pts`, `class_id` to published detections

## [1.5.0] — 2026-03-23

- Topic rename: `/ugv/perception/front/*` → `/ugv/detections/front/*`
- GeoJSON ROS topic `/triffid/front/geojson` → `/ugv/detections/front/geojson`
- MQTT topic default `triffid/front/geojson` → `ugv/detections/front/geojson`

## [1.4.0] — 2026-03-23

- ByteTrack tracker (Kalman + Hungarian + 3D gate) replaces greedy IoU tracker
- `max_age` default 10→30
- New params: `tracker_iou_threshold_low`, `tracker_conf_high`, `tracker_n_init`, `tracker_pos_gate`
- `scipy` dependency added
- Spatial deduplication in merged GeoJSON (same-class within 1 m → keep highest confidence)

## [1.3.0] — 2026-03-04

- MQTT output: GeoJSON published to local Mosquitto broker (`triffid/front/geojson`, QoS 0)
- MQTT trace capture in sample collector

## [1.2.0] — 2026-03-04

- GPS gating (no publish until fix received)
- Polygon emitted when *either* bbox dimension > 0 (was: both)
- 0.3 m minimum extent for zero-dimension polygons
- Merged GeoJSON sample output

## [1.1.0] — 2026-03-02

- 3D coordinates, heading rotation, GPS filtering, `/dog_odom` subscription

## [1.0.0] — 2026-03-01

- Initial frozen interface
