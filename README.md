# TRIFFID Perception

Perception pipelines for the TRIFFID project — covering both **UGV** (ground robot) and **UAV** (DJI M30T drone) platforms.

| Platform | Package | Runtime | Output |
|----------|---------|---------|--------|
| UGV | `triffid_ugv_perception` | ROS 2 Humble + Docker (GPU) | ROS 2 topics + MQTT + API |
| UAV | `triffid_uav_perception` | Pure Python + Docker (CPU) | MQTT + GeoJSON files |

For UAV-specific documentation see [src/triffid_uav_perception/README.md](src/triffid_uav_perception/README.md).

---

## UGV Perception

Real-time 3D object detection pipeline for the TRIFFID UGV platform.  
Uses a pixel-aligned RGB-D camera (Intel RealSense with depth aligned to colour), runs a fine-tuned YOLOv11l-seg model for 2D detection, samples depth directly at detection pixels and back-projects to 3D, tracks objects across frames, and publishes results as ROS 2 messages and GeoJSON.

---

## Table of Contents

1. [UGV Perception](#ugv-perception)
2. [Architecture Overview](#architecture-overview)
3. [Hardware Assumptions](#hardware-assumptions)
4. [ROS 2 Topics](#ros-2-topics)
5. [TF Frame Tree](#tf-frame-tree)
6. [Pipeline Steps](#pipeline-steps)
7. [Node Descriptions](#node-descriptions)
8. [Parameters](#parameters)
9. [Docker Setup](#docker-setup)
10. [Quick Start (`run.sh`)](#quick-start-runsh)
11. [Running the Integration Test](#running-the-integration-test)
12. [Rosbag Datasets](#rosbag-datasets)
13. [Project Structure](#project-structure)
14. [Interface Specification](#interface-specification)

---

## Architecture Overview

```
 ┌──────────────────────────────┐
 │  Intel RealSense (RGB-D)     │
 │  Depth aligned to colour     │
 │  Shared intrinsics & grid    │
 │                              │
 │  RGB: /b2/camera/color/      │
 │       image_raw (bgr8)       │
 │  Depth: /b2/camera/          │
 │    aligned_depth_to_color/   │
 │    image_raw (16UC1, mm)     │
 │  Info: /b2/camera/color/     │
 │        camera_info            │
 └──────────────┬───────────────┘
                │
                ▼
 ┌─────────────────────────────────┐
 │         ugv_node                │
 │  1. YOLO on RGB → 2D bboxes    │
 │  2. Sample depth at det pixels  │
 │  3. Back-project → 3D optical   │
 │  4. TF → b2/base_link           │
 │  5. Median 3D position          │
 │  6. ByteTrack tracker → ID      │
 │  7. Publish Detection3DArray    │
 └──────────┬──────────────────────┘
            │
            ▼
 ┌──────────────────────────────────┐
 │ geojson_bridge                   │
 │ Detection3D → GeoJSON            │
 │ GPS + heading rotation           │
 │ Optional API PUT                 │
 │                                  │
 │ /ugv/detections/front/geojson  │
 │ (ROS2 + MQTT)                  │
 └──────────┬───────────────────────┘
            │
            ▼
 ┌──────────────────────┐
 │ Mosquitto broker     │
 │ localhost:1883       │
 └──────────────────────┘
```

---

## Hardware Assumptions

The UGV platform uses an Intel RealSense camera with **depth aligned to colour** (`aligned_depth_to_color`). Both streams share the same pixel grid, resolution, and intrinsics — no cross-camera projection is needed.

| Property | Value |
|---|---|
| **Camera** | Intel RealSense (RGB-D, depth aligned to colour) |
| **RGB topic** | `/b2/camera/color/image_raw` (configurable) |
| **Depth topic** | `/b2/camera/aligned_depth_to_color/image_raw` (configurable) |
| **CameraInfo topic** | `/b2/camera/color/camera_info` (configurable) |
| **Resolution** | Shared (e.g. 1280 × 720) |
| **RGB encoding** | `bgr8` |
| **Depth encoding** | `16UC1` (millimetres) |
| **TF frame** | Read dynamically from `CameraInfo.header.frame_id` (e.g. `b2/camera_optical_frame`) |
| **Intrinsics** | Shared K matrix from the single `CameraInfo` topic |

Because depth is pixel-aligned to the colour image, the pipeline samples depth directly at detection pixels — there is no separate depth camera frame, no grid sampling, and no cross-camera TF transform.

The robot also publishes `/tf_static` with the transform chain (including `camera_optical_frame → b2/base_link`), `/dog_odom`, `/dog_imu_raw`, and various Unitree topics. Only the camera, CameraInfo, and TF topics are consumed by this package.

---

## ROS 2 Topics

### Subscribed (Input — from rosbag or live robot)

| Topic | Type | Rate | Description |
|---|---|---|---|
| `/b2/camera/color/image_raw` | `sensor_msgs/Image` | ~15 Hz | RGB image (bgr8) |
| `/b2/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | ~15 Hz | Pixel-aligned depth image (16UC1, mm) |
| `/b2/camera/color/camera_info` | `sensor_msgs/CameraInfo` | ~15 Hz | Shared intrinsics matrix K |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | — | Transform tree (see below) |
| `/fix` | `sensor_msgs/NavSatFix` | ~0.4 Hz | GPS fix (optional, for GeoJSON GPS coords) |
| `/dog_odom` | `nav_msgs/Odometry` | ~500 Hz | Odometry with magnetometer-fused heading (optional, for GeoJSON heading rotation) |

> **Note**: Topic names are configurable via ROS parameters (`rgb_image_topic`, `depth_image_topic`, `camera_info_topic`). The defaults above use the `/b2/camera/` namespace.

### Published (Output)

| Topic | Type | Node | Description |
|---|---|---|---|
| `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` | `ugv_node` | 3D detections in `b2/base_link` frame |
| `/ugv/detections/front/segmentation` | `sensor_msgs/Image` | `ugv_node` | Semantic label map (`mono8`, pixel = class ID) |
| `/ugv/detections/front/debug_image` | `sensor_msgs/Image` | `ugv_node` | Debug overlay: RGB + bboxes, class, track ID, depth-point count (only published when subscribed) |
| `/ugv/detections/front/geojson` | `std_msgs/String` | `geojson_bridge` | GeoJSON FeatureCollection (RFC 7946) |

> The same GeoJSON payload is also published as MQTT to `localhost:1883` on topic `ugv/detections/front/geojson` (Mosquitto, running inside the container). See [GeoJSON Bridge Parameters](#geojson-bridge-parameters).

**MQTT output**: `geojson_bridge` also publishes identical GeoJSON payloads to the local Mosquitto broker on topic `ugv/detections/front/geojson` (port 1883). Subscribe from any host with:
```bash
mosquitto_sub -h localhost -t 'ugv/detections/front/geojson'
```

### Detection3DArray Message Structure

Each `Detection3D` in the array contains:

- **`header.frame_id`**: `b2/base_link`
- **`header.stamp`**: Copied from the triggering RGB frame (rosbag time, not wall clock)
- **`id`**: Persistent tracking ID (positive integer, never reused)
- **`bbox.center.position`**: Median 3D position (x, y, z) in metres relative to `b2/base_link`
- **`bbox.size`**: 3D bounding box extent (x, y, z) in metres in `b2/base_link`. Derived from the spread of matched depth points (or back-projected 2D bbox corners when depth points cluster too tightly). May be `(0, 0, 0)` when the TF transform for extent corners fails.
- **`results[0].hypothesis.class_id`**: Class name string (e.g. `First responder`, `Civilian vehicle`, `Flame`, `Debris`)
- **`results[0].hypothesis.score`**: YOLO confidence (0–1)

### Segmentation Topic (`mono8`)

The segmentation topic publishes a `sensor_msgs/Image` with encoding `mono8` (1280×720).
Each pixel value is a 1-based class ID from the YOLO model:

- `0` = background (no detection)
- `1`–`63` = `TARGET_CLASSES[pixel_value - 1]` (e.g. pixel 15 → class ID 14 → `Building`)

When masks overlap, the highest-confidence detection’s class ID wins.

---

## TF Frame Tree

The pipeline requires a single transform from the camera's optical frame to `b2/base_link`. The camera frame ID is **read dynamically** from the `CameraInfo.header.frame_id` field (typically `b2/camera_optical_frame`).

```
b2/base_link
└── b2/camera_optical_frame     (from CameraInfo header)
      Convention: ROS optical (X=right, Y=down, Z=forward)
      Published by robot's driver / localization stack
```

The static transforms for the old cross-camera setup (`f_oc_link`, `f_dc_link`, `f_depth_frame`, `f_depth_optical_frame`) have been removed from the launch file. The transform chain is now managed by Angel's localization stack: `map → b2/map → b2/odom → b2/base_link → b2/camera_optical_frame`.

The node uses **camera_optical_frame convention** (X=right, Y=down, Z=forward) directly for all back-projection. There is no body↓optical conversion inside the node.

---

## Pipeline Steps

The core processing runs on every RGB frame in `rgb_callback`:

1. **YOLO Detection** — Run the fine-tuned YOLOv11l-seg model on the RGB image. The model detects 63 disaster-response classes (see `classes.txt`): people (citizens, first responders, military personnel), vehicles (civilian, police, army, fire truck, ambulance, excavator), hazards (flame, smoke, debris, destroyed buildings), terrain (roads, grass, mud), equipment (helmets, SCBA, fire hose, extinguisher), and more. Or use a full-image dummy bbox for testing without YOLO.

2. **Pixel-aligned Depth Sampling** — For each detection, sample depth at the instance mask pixels (or bbox pixels if no mask). Because the depth image is aligned to the colour image (same resolution, same pixel grid, shared intrinsics), depth is read directly at the detection's pixel coordinates — no grid sampling, no cross-camera TF transform, no pinhole projection. Large masks are sub-sampled to a maximum of 500 depth samples for efficiency. Zero-depth pixels (no reading) are discarded.

3. **Back-project to 3D** — Using the shared intrinsics from the single `CameraInfo` topic, back-project the valid depth samples to 3D points in **camera_optical_frame** (X=right, Y=down, Z=forward).

4. **Median 3D Position** — Take the median of the back-projected 3D points as the detection's position in camera_optical_frame. Transform this point to `b2/base_link` via TF.

5. **3D Extent Estimation** — Compute the 3D bounding box extent from the spread of matched points. When depth points cluster too tightly, fall back to back-projecting the 2D bbox corners at the median depth.

6. **ByteTrack Tracking** — A ByteTrack-style tracker with Kalman-filter motion prediction, Hungarian (optimal) assignment, and two-pass association (high-confidence detections first, then low-confidence). Tracks go through TENTATIVE → CONFIRMED → LOST states; only confirmed tracks (seen for `n_init` consecutive frames) are published. A 3D position gate allows re-identification when the 2D IoU is low but the 3D distance is close (useful for small objects like poles). Tracks are retired after `max_age=30` frames without a match. A **class gate** prevents cross-class matches (e.g. a Fence track cannot be reassigned to a Civilian vehicle detection). Class labels use **majority voting** — each observation votes for a class, and the track takes the class with the most votes, making labels stable even if one frame disagrees. Requires `scipy` for optimal assignment (falls back to greedy if unavailable).

7. **Publish** — Emit a `Detection3DArray` with all tracked detections and a per-pixel semantic label map on `/ugv/detections/front/segmentation` (`mono8`, pixel = 1-based class ID, 0 = background).

---

## Node Descriptions

### `ugv_node` (main perception)

The core pipeline node. Subscribes to RGB, depth, CameraInfo, and TF. Publishes `Detection3DArray` and segmentation overlay.

- Uses fine-tuned `yolo11l-seg` (ultralytics) for 2D detection + instance segmentation (63 classes)
- Segmentation masks used for pixel-precise depth sampling (depth read at mask pixels, not via grid)
- Publishes semantic segmentation label map on `/ugv/detections/front/segmentation` (mono8, only when subscribed)
- 3D NMS deduplication: overlapping detections at the same 3D position are merged (highest confidence kept)
- Pixel-aligned RGB-D: single camera with shared intrinsics, no cross-camera TF required
- Camera frame read dynamically from `CameraInfo.header.frame_id`
- Configurable topic names via ROS parameters (default `/b2/camera/` namespace)
- ByteTrack-style tracker (Kalman + Hungarian + class gate + majority-vote class + 3D gate) for persistent object IDs
- Frame synchronisation: uses latest available depth image when an RGB frame arrives (not strict time-sync)

### `geojson_bridge`

Converts `Detection3DArray` messages to GeoJSON (RFC 7946) for the TRIFFID mapping API.

- **GPS gating**: does not publish until at least one valid `/fix` message is received, preventing body-frame metre coordinates from appearing as lon/lat
- If GPS origin is available (from `/fix` topic or parameters), converts local (x, y) to WGS-84 (lon, lat) using equirectangular approximation; GPS positions are median-filtered (window = 7)
- If heading is available (from `/dog_odom`), rotates body-frame offsets by yaw (ENU) before GPS projection
- Geometry type is **class-dependent**: Point (31 small/mobile classes), LineString (Fence, Wall), Polygon (30 area classes). Polygon dimensions are clamped to a minimum extent of 0.3 m even when depth evidence is sparse.
- **MQTT**: publishes compact JSON to a local Mosquitto broker (default `localhost:1883`, topic `ugv/detections/front/geojson`). Enabled by default; disable with `mqtt_enabled:=false`.
- Optionally PUTs to the TRIFFID API at `https://crispres.com/wp-json/map-manager/v1/features` (disabled by default)
- Each detection becomes a GeoJSON Feature with SimpleStyle properties (`marker-color`, `marker-symbol`, etc.)

---

## Parameters

All parameters are declared on `ugv_node` and configurable via the launch file:

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `/ws/best.pt` | YOLO model weights file (mounted from host) |
| `confidence_threshold` | `0.35` | Minimum YOLO confidence to accept a detection |
| `target_frame` | `b2/base_link` | Output frame for 3D positions |
| `rgb_image_topic` | `/b2/camera/color/image_raw` | RGB image topic name |
| `depth_image_topic` | `/b2/camera/aligned_depth_to_color/image_raw` | Depth image topic name |
| `camera_info_topic` | `/b2/camera/color/camera_info` | CameraInfo topic name (shared intrinsics) |
| `use_dummy_detections` | `false` | Bypass YOLO with a full-image dummy bbox (for testing) |
| `yolo_imgsz` | `1280` | YOLO input resolution (pixels) |
| `tracker_iou_threshold` | `0.30` | High-confidence IoU threshold for first-pass matching |
| `tracker_iou_threshold_low` | `0.15` | Low-confidence IoU threshold for second-pass matching |
| `tracker_conf_high` | `0.40` | Confidence split: detections above this go to first pass |
| `tracker_max_age` | `30` | Frames before a lost track is retired |
| `tracker_n_init` | `3` | Consecutive hits before a track is confirmed (published) |
| `tracker_pos_gate` | `2.0` | 3D distance gate (m) — allows matching when IoU is low |

### GeoJSON Bridge Parameters

| Parameter | Default | Description |
|---|---|---|
| `api_url` | `https://crispres.com/...` | TRIFFID mapping API endpoint |
| `publish_to_api` | `false` | Enable HTTP PUT to the API |
| `gps_origin_lat` | `0.0` | GPS origin latitude (or auto-set from `/fix`) |
| `gps_origin_lon` | `0.0` | GPS origin longitude |
| `mqtt_enabled` | `true` | Enable MQTT publishing to local broker |
| `mqtt_host` | `localhost` | MQTT broker hostname |
| `mqtt_port` | `1883` | MQTT broker port |
| `mqtt_topic` | `ugv/detections/front/geojson` | MQTT topic for GeoJSON output |

---

## Docker Setup

The pipeline runs inside a Docker container based on `ros:humble-perception-jammy` with NVIDIA GPU support.

### Container Details

- **Image**: `triffid_perception:latest`
- **Base**: ROS 2 Humble on Ubuntu 22.04
- **GPU**: NVIDIA (via `nvidia-container-toolkit`)
- **Network**: `host` mode (shares host DDS network)
- **IPC**: `host` (shared memory for fast DDS transport)
- **ROS_DOMAIN_ID**: `42` (isolated from host's default domain 0)
- **DDS**: CycloneDDS with increased fragment buffers for large images
- **MQTT**: Mosquitto broker (`mosquitto` + `mosquitto-clients`) installed in image, started automatically by `run.sh start`

### Volumes

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src/` | `/ws/src/` | Source code (editable from host) |
| `./new_rosbag/rosbag2_active_*/` | `/ws/rosbag/` | Rosbag dataset (read-only) |
| `./install/` | `/ws/install/` | Build artifacts (persisted) |
| `./build/` | `/ws/build/` | Build artifacts (persisted) |
| `./log/` | `/ws/log/` | Build logs |
| `./cyclonedds.xml` | `/ws/cyclonedds.xml` | DDS configuration |
| `./best.pt` | `/ws/best.pt` | YOLO model weights (read-only) |

---

## Quick Start (`run.sh`)

The all-in-one `run.sh` script wraps Docker operations:

```bash
cd ~/hua_ws
./run.sh build          # Build the workspace inside Docker
./run.sh start          # Launch pipeline (ugv_node + geojson_bridge + rosbag)
./run.sh stop           # Stop everything
./run.sh restart        # Rebuild + restart
./run.sh logs           # Tail node logs
./run.sh sample [SEC]   # Collect output samples + merged GeoJSON (default: 15s)
./run.sh test           # Run integration test
./run.sh unit           # Run unit tests
./run.sh shell          # Open a bash shell in the container
./run.sh status         # Check topic rates
```

Environment variables: `BAG_RATE` (default 1.0), `BAG_START` (offset sec), `YOLO_IMGSZ` (default 1280), `TIMEOUT` (test timeout).

**Notes:**
- `build` and `start` perform a clean build (`rm -rf build/* install/* log/*`) before `colcon build` to avoid stale artefact conflicts with bind-mounted directories.
- `sample` saves the following files to `./samples/`:
  - `rgb_frame_0.jpg` … `rgb_frame_4.jpg` — 5 raw RGB frames
  - `detections_3d.yaml` — first non-empty `Detection3DArray` message
  - `segmentation.png` — first semantic label map (`mono8`)
  - `geojson.json` — first non-empty GeoJSON `FeatureCollection`
  - `geojson_raw.json` — all confirmed tracks (one feature per track ID, highest-confidence snapshot), no spatial deduplication — every ID the tracker confirmed is present
  - `geojson_merged.json` — same as raw but spatially deduplicated per class (same-class features within 1 m merged, highest confidence kept)
  - `mqtt_trace.jsonl` — every MQTT GeoJSON message received during the window, one compact JSON object per line
  - `tracking_debug.mp4` — H.264 video of the debug overlay (bboxes, class names, track IDs, depth-point count `d=N`)
  - `track_lifecycle.csv` — per-frame track presence/absence table for all track IDs
  - `possible_id_switches.csv` — detected potential tracker ID switches

### Manual Quick Start

### 1. Build the container

```bash
cd ~/hua_ws
sudo docker compose build
```

### 2. Build the ROS 2 package

```bash
sudo docker compose exec perception bash -c "
  cd /ws && colcon build --packages-select triffid_ugv_perception
"
```

### 3. Run the pipeline

Inside the container, launch both nodes:

```bash
sudo docker compose exec perception bash -c "
  source /ws/install/setup.bash &&
  ros2 launch triffid_ugv_perception ugv_perception.launch.py
"
```

> **Note**: The launch file no longer publishes static TF transforms. The camera→base_link transform chain must be provided by the robot's driver or localization stack.

### 4. Play the rosbag (in a separate terminal)

The rosbag must be played on the **host** with matching `ROS_DOMAIN_ID`, or inside the container:

```bash
# Inside the container:
sudo docker compose exec perception bash -c "
  ros2 bag play /ws/rosbag --rate 0.5 \
    --qos-profile-overrides-path /ws/src/triffid_ugv_perception/config/bag_qos_overrides.yaml
"
```

### 5. Verify output

```bash
# Echo 3D detections:
sudo docker compose exec perception bash -c "
  ROS_DOMAIN_ID=42 ros2 topic echo /ugv/detections/front/detections_3d --once"

# Echo GeoJSON (ROS 2 topic):
sudo docker compose exec perception bash -c "
  ROS_DOMAIN_ID=42 ros2 topic echo /ugv/detections/front/geojson --once"

# Subscribe to GeoJSON via MQTT (from inside container):
sudo docker compose exec perception bash -c "
  mosquitto_sub -h localhost -t 'ugv/detections/front/geojson'"
```

### Testing with Dummy Detections (no YOLO)

To test the depth, TF, and tracking pipeline without YOLO:

```bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py use_dummy_detections:=true
```

This generates a full-image bounding box on every frame, allowing you to verify that depth points project correctly and 3D positions are computed.

---

## Running the Integration Test

The integration test verifies 9 checks across all pipeline requirements:

```bash
sudo docker compose exec perception bash -c "
  cd /ws && source install/setup.bash &&
  python3 /ws/src/triffid_ugv_perception/test/integration_test.py --timeout 45
"
```

### What it checks

| # | Check | Requirement |
|---|---|---|
| 1 | Output rosbag recorded | Replayable dataset |
| 2 | All expected topics receive messages | Topic liveness |
| 3 | Message types match spec | Interface definitions |
| 4 | CameraInfo valid (frame, resolution, intrinsics) | Sensor validation |
| 5 | Required TF transforms available | Coordinate frames |
| 6 | Detection timestamps from rosbag, not wall clock; depth–RGB sync <500ms | Timestamp consistency |
| 7 | Detection fields populated (id, class, score, frame_id) | Interface definitions |
| 8 | 3D positions finite, non-zero, within 30m range | Depth pipeline sanity |
| 9 | Persistent tracking IDs, no duplicates per frame | Tracking correctness |
| 10 | GeoJSON RFC-7946 valid, required properties present | GeoJSON schema |
| 11 | Segmentation label map (mono8) published | Segmentation output |

### Options

```bash
python3 integration_test.py --check timestamps   # run a single check
python3 integration_test.py --no-launch           # skip launching nodes (if already running)
python3 integration_test.py --no-bag              # skip automatic rosbag playback
python3 integration_test.py --timeout 60          # increase timeout
```

---

## Rosbag Datasets

The rosbag is not included in the repository (17+ GB binary). Mount it via `docker-compose.yml`.

### Required topics in the bag

- `/b2/camera/color/image_raw` — RGB frames
- `/b2/camera/color/camera_info` — Camera intrinsics (shared)
- `/b2/camera/aligned_depth_to_color/image_raw` — Pixel-aligned depth frames
- `/tf_static` — Static transforms (camera_optical_frame → base_link)

> **Legacy bags**: Older rosbags with separate RGB/depth camera topics (`/camera_front/raw_image`, `/camera_front/realsense_front/depth/...`) are no longer compatible. Use topic remapping or re-record with the new sensor configuration.

### Note on YOLO detection

The pipeline detects 63 disaster-response classes using a fine-tuned YOLOv11l-seg model (see `classes.txt` for the full list). If the recorded scene contains none of the trained classes, the pipeline will publish empty `Detection3DArray` messages — this is expected behaviour, not a bug. Use `use_dummy_detections:=true` to test the depth pipeline independently.

---

## Project Structure

```
hua_ws/
├── docker-compose.yml          # UGV container config (GPU, volumes, network)
├── docker-compose.uav.yml      # UAV container config (lightweight, no ROS2)
├── Dockerfile                  # UGV image: ROS2 Humble + CUDA + ultralytics
├── Dockerfile.uav              # UAV image: Python 3.10 + ultralytics
├── run.sh                      # UGV all-in-one runner
├── run_uav.sh                  # UAV all-in-one runner
├── docker_entrypoint.sh        # Source ROS, build if needed
├── cyclonedds.xml              # DDS tuning for large image fragments
├── .gitignore
│
└── src/
    ├── triffid_ugv_perception/     # UGV perception (ROS2 + MQTT)
    │   ├── package.xml
    │   ├── setup.py
    │   ├── setup.cfg
    │   ├── config/
    │   │   └── bag_qos_overrides.yaml    # QoS overrides for ros2 bag play
    │   ├── launch/
    │   │   └── ugv_perception.launch.py  # Launch nodes (no static TFs)
    │   ├── triffid_ugv_perception/
    │   │   ├── __init__.py
    │   │   ├── ugv_node.py               # Main perception node (pixel-aligned RGB-D)
    │   │   ├── tracker.py                # ByteTrack-style multi-object tracker
    │   │   └── geojson_bridge.py         # Detection3D → GeoJSON + API
    │   ├── scripts/
    │   │   └── collect_samples.py        # Output sample collector
    │   └── test/
    │       ├── integration_test.py       # End-to-end integration test
    │       └── test_unit.py              # 164 unit tests
    │
    └── triffid_uav_perception/     # UAV perception (standalone, MQTT only)
        ├── requirements.txt
        ├── README.md
        ├── triffid_uav_perception/
        │   ├── __init__.py
        │   ├── metadata.py              # DJI XMP metadata extraction
        │   ├── geo.py                   # Geo-projection (pixel → GPS)
        │   └── uav_node.py             # Main pipeline + MQTT
        └── test/
            └── test_unit.py             # 34 unit tests
```

---

## Interface Specification

For integration partners: the full frozen interface document is in **[INTERFACE.md](INTERFACE.md)**.
It specifies every topic, message field, frame convention, GeoJSON schema, class list, and encoding in a single reference.
