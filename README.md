# TRIFFID Perception Pipeline

**Robotic vision for object detection and segmentation on UGV and UAV platforms.**

- **Platform:** ROS2 (Humble inside Docker, Jazzy on host)
- **Hardware:** Unitree Go robot dog (UGV), UAV (pending)
- **GPU:** 2x NVIDIA GeForce RTX 5090
- **Workspace:** `/home/triffid/hua_ws`

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Repository Structure](#2-repository-structure)
3. [Docker Setup](#3-docker-setup)
4. [UGV Perception Node](#4-ugv-perception-node)
5. [UAV Perception Node](#5-uav-perception-node)
6. [GeoJSON Bridge](#6-geojson-bridge)
7. [IoU Tracker](#7-iou-tracker)
8. [ROS2 Topics & Messages](#8-ros2-topics--messages)
9. [Rosbag Dataset](#9-rosbag-dataset)
10. [Quick Start](#10-quick-start)
11. [Configuration & Parameters](#11-configuration--parameters)
12. [Swapping in Your Own Model](#12-swapping-in-your-own-model)
13. [Known Limitations & TODOs](#13-known-limitations--todos)

---

## 1. Architecture Overview

The pipeline consists of three ROS2 nodes running inside a Docker container:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Docker Container                                 │
│                   (ROS2 Humble + CUDA + Ultralytics)                    │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────────┐  │
│  │  UGV Node    │───▶│  GeoJSON Bridge  │───▶│  TRIFFID Mapping API  │  │
│  │  (ugv_node)  │    │  (geojson_bridge)│    │  (crispres.com)       │  │
│  └──────────────┘    └──────────────────┘    └───────────────────────┘  │
│         │                    ▲                                          │
│         │   ┌──────────────┐ │                                          │
│         │   │  UAV Node    │─┘                                          │
│         │   │  (uav_node)  │  (skeleton, awaiting rosbag)               │
│         │   └──────────────┘                                            │
│         ▼                                                               │
│  ┌──────────────────┐                                                   │
│  │  Diagnostics     │──▶ /diagnostics + /triffid/heartbeat              │
│  │  (health monitor)│                                                   │
│  └──────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
         ▲
         │ ROS2 DDS (host network)
         │
┌────────┴────────┐
│  Rosbag Replay  │
│  (or live robot) │
└─────────────────┘
```

### Data Flow (UGV Pipeline)

```
RGB Image ──────────────────┐
                            ▼
                    ┌───────────────┐
                    │  YOLO Model   │  Detect objects in each frame
                    │  (detection)  │  → bounding boxes + class + confidence
                    └───────┬───────┘
                            │
Depth Image ────────────────▼
                    ┌───────────────┐
                    │ Depth Sample  │  Sample depth at bbox centroid
                    │ (5×5 median)  │  → get Z in metres
                    └───────┬───────┘
                            │
CameraInfo ─────────────────▼
                    ┌───────────────┐
                    │ Back-Project  │  Pixel (u,v) + depth Z → 3D point
                    │   to 3D      │  X = (u - cx) * Z / fx
                    │              │  Y = (v - cy) * Z / fy
                    └───────┬───────┘
                            │
TF Transforms ─────────────▼
                    ┌───────────────┐
                    │  TF2 Lookup   │  camera_frame → map frame
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  IoU Tracker  │  Assign persistent ID to each object
                    │  (greedy)     │  IDs are NEVER reused
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Publish      │  vision_msgs/Detection3DArray
                    │  ROS2 topic   │  on /ugv/perception/detections_3d
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ GeoJSON Bridge│  Local XY → lat/lon → RFC-7946 GeoJSON
                    │              │  → /triffid/geojson (ROS2 topic)
                    │              │  → HTTP PUT to TRIFFID API (optional)
                    └───────────────┘
```

### Mermaid Diagram

See the diagram [here](https://mermaid.live/edit#pako:eNqNlNty2jAQhl9Fo6t2SsKZABed8QEICWCPUTLT1h1G2LJxAhIjmyZpJrd9gD5in6QrGQgmh9YXHnv17Vr7-1894kCEDHdxtBR3wYLKDBHT5wiudDOPJV0vkOdMTWPwzceeSOc0RmU0Sn4w5Im5yHz8PafV5Q1MoMoBXTFJZ5EUPCtLejdLVjRmBdLuueT8JcvoMmU8Zdv3kK2zRVlnzyQLshkUK5SxjPFw0nf-s9AWSXgkClVIXxXIIvQJwX2WZjRLggIxcKcKiZL7fZjx0OdHQl0NrgGDO3KZDOCbieBoAvoe9U56FgHwizNykM0y6AxAYNDJyWc0NcbuqAfLtto0mtLVepnwuFAiZzRuGtal6zkXkFC3kUmD2xNXipt9zeekHanTiGdMpn3HG0Me6dcQkZSnkZCrojI7apdjXQI_FFeKD26ZPKatS026V6ZuYNta3TakpA_vamd6Q3ug2h4wcTF1JsiUSXjkGsuZXPc8Jd1IBHSJ_vz6rX7NTjm1UYMclOhDP7Ro0ZzRONiaOO7Q0j9fJlGUhOWYiZv0SLeDFMMdAn1OiAstEvX6bk_2UI-NndCYixRMlb50w9iZDInjqd8g1kmgR4uzNPV97tGMpSU0feBBCUy6a1NVzQUuh8-VXyuq8fOe4RGzp4XZ97lgMOlzdiDOq342tJ-Na71t9CG9ZUuWCf6x8DFYn4Gn3zS0Wv-XdfbMPuPIQrW3LQSnTq6Lnqs8ps-Xg3HKo9vjojA3-QrpF6cij4K3dHjruzwIG3sZ3O74wIZ5HJdwDD7G3UxuWAnD-bOi6hU_qnUfZwu2Aj904TGk8tbHPn-CnDXlX4VY7dKk2MQL3I3UoVbCm3UIxlCmknS1j0pQhElLbHiGu61OSxfB3Ud8j7u1auW006m3G5Vmpd5oNZr1En6AcPvstNas16vtZqtZq7Q6jacS_qm_Wzk9q9Sa1Uq70662IKnWevoLuXzSKA):


---

## 2. Repository Structure

```
hua_ws/
├── Dockerfile                          # Docker image definition
├── docker-compose.yml                  # Docker Compose config (GPU, volumes)
├── docker_entrypoint.sh                # Container entrypoint (sources ROS2)
├── plan.md                             # Original integration plan
├── email.txt                           # GeoJSON API spec from partners
├── README.md                           # ← You are here
│
├── rosbag2_active_20260220_164455/     # UGV rosbag dataset (~65 seconds)
│   ├── metadata.yaml
│   └── rosbag2_active_20260220_164455_0.db3
│
└── src/
    ├── triffid_ugv_perception/         # UGV perception package
    │   ├── package.xml                 # ROS2 package manifest
    │   ├── setup.py                    # Python package config
    │   ├── setup.cfg
    │   ├── resource/
    │   ├── launch/
    │   │   └── ugv_perception.launch.py
    │   ├── test/
    │   │   └── integration_test.py     # Automated end-to-end integration test
    │   └── triffid_ugv_perception/
    │       ├── __init__.py
    │       ├── ugv_node.py             # Main UGV perception node
    │       ├── tracker.py              # IoU-based multi-object tracker
    │       ├── geojson_bridge.py       # Detection → GeoJSON converter
    │       └── diagnostics.py          # Health monitoring + heartbeat
    │
    └── triffid_uav_perception/         # UAV perception package
        ├── package.xml
        ├── setup.py
        ├── setup.cfg
        ├── resource/
        ├── launch/
        │   └── uav_perception.launch.py
        └── triffid_uav_perception/
            ├── __init__.py
            └── uav_node.py             # UAV perception node (placeholder)
```

---

## 3. Docker Setup

### Image Details

| Layer | Contents |
|---|---|
| **Base** | `ros:humble-perception-jammy` (Ubuntu 22.04 + ROS2 Humble) |
| **ROS2 packages** | `cv_bridge`, `vision_msgs`, `tf2_ros`, `tf2_geometry_msgs`, `image_transport` |
| **Python packages** | `ultralytics` (YOLO), `opencv-python-headless`, `numpy<2` |
| **GPU** | NVIDIA runtime (RTX 5090 passthrough via `nvidia-container-toolkit`) |

### Build & Run

```bash
# Build the Docker image (first time or after Dockerfile changes)
sudo docker compose build

# Enter an interactive shell inside the container
sudo docker compose run --rm perception bash

# Inside the container — build the ROS2 workspace
cd /ws && colcon build --symlink-install
source install/setup.bash
```

### Volume Mounts

The `docker-compose.yml` bind-mounts these directories so you can edit code on the host and it's immediately reflected inside the container:

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src/` | `/ws/src/` | Source code (editable from host) |
| `./rosbag2_active_*/` | `/ws/rosbag/` | Rosbag dataset (read-only) |
| `./build/` | `/ws/build/` | Build artifacts (persisted) |
| `./install/` | `/ws/install/` | Install artifacts (persisted) |
| `./log/` | `/ws/log/` | Build logs (persisted) |

### Networking

- `network_mode: host` — the container shares the host's network stack, so ROS2 DDS discovery works transparently between the container and any other ROS2 nodes on the same machine.
- `ipc: host` — enables shared-memory transport for faster DDS communication.
- **Cross-machine communication** uses a **Zenoh bridge** (not raw DDS multicast). The Zenoh bridge handles topic forwarding between the UGV, UAV, and base station machines, so no special DDS XML config is needed.

---

## 4. UGV Perception Node

**File:** `src/triffid_ugv_perception/triffid_ugv_perception/ugv_node.py`
**Executable:** `ugv_node`
**ROS2 Node Name:** `ugv_perception_node`

### What It Does

Processes RGB + Depth images from the Unitree Go robot dog's front camera to detect, localise in 3D, and track objects of interest.

### Subscriptions

| Topic | Type | QoS | Purpose |
|---|---|---|---|
| `/camera_front/raw_image` | `sensor_msgs/Image` | BEST_EFFORT | RGB image from USB camera |
| `/camera_front/realsense_front/depth/image_rect_raw` | `sensor_msgs/Image` | BEST_EFFORT | 16-bit depth image (mm) from RealSense D430i |
| `/camera_front/realsense_front/depth/camera_info` | `sensor_msgs/CameraInfo` | RELIABLE | Camera intrinsics (fx, fy, cx, cy) |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | (via TF2 listener) | Coordinate transforms |

### Publications

| Topic | Type | Purpose |
|---|---|---|
| `/ugv/perception/detections_3d` | `vision_msgs/Detection3DArray` | 3D detections with tracking IDs |

### Processing Pipeline (per RGB frame)

1. **Wait for prerequisites** — CameraInfo and at least one depth frame must be received before processing starts.

2. **Convert ROS Image → OpenCV** — Uses `cv_bridge` to convert the ROS `sensor_msgs/Image` message to a BGR OpenCV `numpy` array.

3. **Run detection model** — Passes the BGR image to the YOLO model. Returns a list of bounding boxes with class IDs and confidence scores. Currently filters for COCO classes: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`. If ultralytics is not installed, the node logs an error and produces no detections.

4. **Sample depth at detection centroid** — For each bounding box, computes the centroid pixel `(cu, cv)` and samples a 5x5 patch from the depth image around that pixel. Takes the **median of valid (non-zero) pixels** for robustness against noise. If the depth value is > 100, it's assumed to be in millimetres and converted to metres.

5. **3D back-projection** — Converts the 2D pixel + depth into a 3D point in the camera's coordinate frame using the pinhole camera model:
   ```
   X = (u - cx) * Z / fx
   Y = (v - cy) * Z / fy
   Z = Z (depth in metres)
   ```
   Where `fx`, `fy`, `cx`, `cy` come from the `CameraInfo` `K` matrix.

6. **TF2 transform** — Transforms the 3D point from the camera frame to the `target_frame` (default: `map`) using the ROS2 TF2 tree. If the transform fails (e.g., frame not in TF tree), falls back to camera-frame coordinates with a warning.

7. **IoU tracking** — Passes all detections to the `IoUTracker` which assigns persistent IDs. See [Section 7](#7-iou-tracker) for details.

8. **Publish Detection3DArray** — Each tracked detection becomes a `Detection3D` message with:
   - `bbox.center.position` — the 3D coordinates
   - `results[0].hypothesis.class_id` — the class name (string)
   - `results[0].hypothesis.score` — the confidence
   - `id` — the persistent tracking ID (string)
   - `header.stamp` — **copied from the input RGB image** (preserves timing)
   - `header.frame_id` — set to the `target_frame` parameter

---

## 5. UAV Perception Node

**File:** `src/triffid_uav_perception/triffid_uav_perception/uav_node.py`
**Executable:** `uav_node`
**ROS2 Node Name:** `uav_perception_node`
**Status:** SKELETON — awaiting UAV rosbag

### What It Will Do

Process RGB images from the UAV camera to detect and track objects, publishing 2D detections.

### Expected Subscriptions (placeholder topics, to be updated)

| Topic | Type | Purpose |
|---|---|---|
| `/uav/camera/image_raw` | `sensor_msgs/Image` | RGB image from UAV camera |
| `/uav/gps/fix` | `sensor_msgs/NavSatFix` | UAV GPS position |

### Publications

| Topic | Type | Purpose |
|---|---|---|
| `/uav/perception/detections_2d` | `vision_msgs/Detection2DArray` | 2D detections with tracking IDs |

### Pipeline

```
RGB Image → YOLO Detection → IoU Tracker → Detection2DArray
```

### What Needs to Be Done

1. Get a UAV rosbag and identify the actual topic names
2. Update the `rgb_topic` and `gps_topic` parameters
3. Implement geo-projection if TRIFFID requires lat/lon output from the UAV (see plan.md Step 5)

---

## 6. GeoJSON Bridge

**File:** `src/triffid_ugv_perception/triffid_ugv_perception/geojson_bridge.py`
**Executable:** `geojson_bridge`
**ROS2 Node Name:** `geojson_bridge`

### What It Does

Converts ROS2 detection messages into RFC-7946 compliant GeoJSON and optionally pushes them to the TRIFFID mapping REST API.

### Subscriptions

| Topic | Type | Purpose |
|---|---|---|
| `/ugv/perception/detections_3d` | `vision_msgs/Detection3DArray` | UGV 3D detections |
| `/uav/perception/detections_2d` | `vision_msgs/Detection2DArray` | UAV 2D detections (when available) |
| `/fix` | `sensor_msgs/NavSatFix` | GPS fix to set coordinate origin |

### Publications

| Topic | Type | Purpose |
|---|---|---|
| `/triffid/geojson` | `std_msgs/String` | GeoJSON as a JSON string (for debugging) |

### Coordinate Conversion

Detections from the UGV node are in **local map frame** (metres relative to the robot's starting position). To produce GPS coordinates for GeoJSON, the bridge uses **equirectangular approximation**:

```
d_lat = y_metres / R * (180 / π)
d_lon = x_metres / (R * cos(lat_origin)) * (180 / π)

latitude  = origin_lat + d_lat
longitude = origin_lon + d_lon
```

Where `R = 6,378,137 m` (Earth radius). This is accurate to ~1 metre for displacements under ~1 km from the origin.

The GPS origin is set by:
1. The `gps_origin_lat` / `gps_origin_lon` parameters (highest priority), OR
2. The first valid message on the `/fix` topic (auto-detected)

### GeoJSON Output Format

Each detection becomes a GeoJSON Point Feature. Example output:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [23.734806, 37.975502]
      },
      "properties": {
        "name": "person",
        "category": "detection",
        "source": "ugv",
        "track_id": "1",
        "confidence": 0.99,
        "marker-color": "#ff0000",
        "marker-size": "medium",
        "marker-symbol": "pitch"
      }
    }
  ]
}
```

This is compatible with the TRIFFID mapping API and can be validated at [geojson.io](https://geojson.io).

### SimpleStyle Markers (Mapbox v1.1.0)

Each class gets a colour and icon for map visualisation:

| Class | marker-color | marker-symbol |
|---|---|---|
| person | `#ff0000` (red) | pitch |
| car | `#0000ff` (blue) | car |
| truck | `#00008b` (dark blue) | truck |
| bus | `#000080` (navy) | bus |
| bicycle | `#00ff00` (green) | bicycle |
| motorcycle | `#008000` (dark green) | marker |
| debris | `#ff8c00` (orange) | marker |

### API Integration

When `publish_to_api=true`, the bridge sends the GeoJSON to the TRIFFID API via HTTP PUT in a background thread (non-blocking). The API endpoint is configurable via the `api_url` parameter.

**Default:** `https://crispres.com/wp-json/map-manager/v1/features`

---

## 7. IoU Tracker

**File:** `src/triffid_ugv_perception/triffid_ugv_perception/tracker.py`
**Class:** `IoUTracker`

### Purpose

Maintains persistent object identities across video frames. Required by the TRIFFID spec.

### Algorithm

1. **IoU matrix** — Compute Intersection-over-Union between every existing track's bounding box and every new detection's bounding box.

2. **Greedy matching** — Sort all IoU values descending. Greedily assign the best match first, requiring IoU ≥ `iou_threshold` (default: 0.3).

3. **New tracks** — Any unmatched detection gets a new unique ID.

4. **Aging** — Unmatched tracks have their `age` incremented. After `max_age` frames (default: 10) without a match, the track is retired.

5. **ID guarantee** — The `next_id` counter starts at 1 and only increments. **IDs are never reused, even after a track is retired.** This is a hard requirement from the TRIFFID spec.

### Usage

```python
tracker = IoUTracker(iou_threshold=0.3, max_age=10)

# Each frame:
results = tracker.update([
    {'bbox': (x1, y1, x2, y2), 'class_name': 'person', 'confidence': 0.9, ...},
    ...
])
# Each result dict now has an added 'track_id' key
```

### Upgrading

For production, consider replacing this with:
- **ByteTrack** — better for crowded scenes
- **DeepSORT** — uses appearance features for re-identification

The interface is the same: takes a list of detection dicts, returns a list with `track_id` added.

---

## 8. ROS2 Topics & Messages

### Input Topics (from rosbag or live robot)

| Topic | Type | Freq | Description |
|---|---|---|---|
| `/camera_front/raw_image` | `sensor_msgs/Image` | ~15 Hz | Front USB camera, BGR8 |
| `/camera_front/realsense_front/depth/image_rect_raw` | `sensor_msgs/Image` | ~15 Hz | Front RealSense depth, 16UC1 (mm) |
| `/camera_front/realsense_front/depth/camera_info` | `sensor_msgs/CameraInfo` | ~15 Hz | Depth camera intrinsics |
| `/tf` | `tf2_msgs/TFMessage` | ~500 Hz | Dynamic transforms |
| `/tf_static` | `tf2_msgs/TFMessage` | once | Static transforms (camera mounts) |
| `/fix` | `sensor_msgs/NavSatFix` | varies | GPS fix (0 msgs in current bag) |
| `/dog_odom` | `nav_msgs/Odometry` | ~500 Hz | Robot odometry |
| `/dog_imu_raw` | `sensor_msgs/Imu` | ~500 Hz | Robot IMU |

### Output Topics (published by our nodes)

| Topic | Type | Description |
|---|---|---|
| `/ugv/perception/detections_3d` | `vision_msgs/Detection3DArray` | 3D detections in target frame |
| `/uav/perception/detections_2d` | `vision_msgs/Detection2DArray` | 2D detections (UAV, when ready) |
| `/triffid/geojson` | `std_msgs/String` | GeoJSON FeatureCollection as JSON string |
| `/triffid/heartbeat` | `std_msgs/String` | JSON heartbeat with topic health status |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Standard ROS2 diagnostics |

### Message Field Mapping

**Detection3D** (UGV output):
```
detection.header.stamp      = input RGB image timestamp (NEVER generated, always copied)
detection.header.frame_id   = target_frame parameter (default: "map")
detection.id                = persistent tracking ID (string, never reused)
detection.results[0].hypothesis.class_id = class name (e.g., "person")
detection.results[0].hypothesis.score    = confidence (0.0–1.0)
detection.bbox.center.position.x/y/z     = 3D position in target frame (metres)
```

---

## 9. Rosbag Dataset

**Path:** `rosbag2_active_20260220_164455/`
**Format:** SQLite3 (ROS2 default)
**Duration:** ~65 seconds
**Total messages:** 238,080

### Key Topics Summary

| Topic | Messages | Notes |
|---|---|---|
| Front RGB | 935 | USB camera |
| Front Depth | 981 | RealSense D430i |
| Front CameraInfo | 981 | Intrinsics for depth |
| Back RGB | 933 | Second USB camera |
| Back Depth | 980 | Second RealSense |
| TF | 32,678 | Dynamic transforms |
| TF Static | 9 | Camera mounts |
| Dog Odometry | 32,684 | Robot pose |
| Dog IMU | 32,687 | Accelerometer + gyro |
| LiDAR | 654 | RoboSense point cloud |
| GPS Fix | 0 | Not available in this bag |

### Playing the Rosbag

```bash
# Inside Docker container:
ros2 bag play /ws/rosbag

# With rate control:
ros2 bag play /ws/rosbag --rate 0.5    # half speed
ros2 bag play /ws/rosbag -l            # loop forever
ros2 bag play /ws/rosbag --start-offset 10  # skip first 10 seconds
```

---

## 10. Quick Start

### Prerequisites
- Docker with NVIDIA runtime (`nvidia-container-toolkit` installed)
- `sudo` access (or user in `docker` group)

### Step-by-step

```bash
# 1. Navigate to workspace
cd /home/triffid/hua_ws

# 2. Build Docker image (first time only, takes ~5 minutes)
sudo docker compose build

# 3. Enter the container
sudo docker compose run --rm perception bash

# 4. Build ROS2 packages (inside container)
cd /ws && colcon build --symlink-install
source install/setup.bash

# 5. In one terminal — play the rosbag
ros2 bag play /ws/rosbag

# 6. In another terminal — run UGV perception
ros2 run triffid_ugv_perception ugv_node

# 7. In another terminal — run GeoJSON bridge
ros2 run triffid_ugv_perception geojson_bridge \
  --ros-args -p gps_origin_lat:=37.9755 -p gps_origin_lon:=23.7348

# 8. Watch the outputs
ros2 topic echo /ugv/perception/detections_3d
ros2 topic echo /triffid/geojson
```

### Using the Launch File (runs both nodes)

```bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py \
  gps_origin_lat:=37.9755 \
  gps_origin_lon:=23.7348
```

### Quick Test (all-in-one)

```bash
sudo docker compose run --rm perception bash -c "
  cd /ws && colcon build --symlink-install && source install/setup.bash &&
  ros2 launch triffid_ugv_perception ugv_perception.launch.py \
    gps_origin_lat:=37.9755 gps_origin_lon:=23.7348 &
  sleep 8 &&
  timeout 15 ros2 bag play /ws/rosbag --start-offset 5 &
  sleep 10 &&
  ros2 topic echo /triffid/geojson --once
"
```

---

## 11. Configuration & Parameters

### UGV Node Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `yolo11n.pt` | Path to YOLO model weights |
| `confidence_threshold` | float | `0.35` | Minimum detection confidence |
| `target_frame` | string | `map` | TF frame for output coordinates |

### GeoJSON Bridge Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_url` | string | `https://crispres.com/...` | TRIFFID mapping API endpoint |
| `publish_to_api` | bool | `false` | Enable HTTP PUT to API |
| `gps_origin_lat` | float | `0.0` | GPS latitude of local frame origin |
| `gps_origin_lon` | float | `0.0` | GPS longitude of local frame origin |
| `gps_origin_alt` | float | `0.0` | GPS altitude of local frame origin |

### UAV Node Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `yolo11n.pt` | Path to YOLO model weights |
| `confidence_threshold` | float | `0.35` | Minimum detection confidence |
| `rgb_topic` | string | `/uav/camera/image_raw` | RGB image topic (update from rosbag) |
| `gps_topic` | string | `/uav/gps/fix` | GPS topic (update from rosbag) |

---

## 12. Swapping in Your Own Model

The system is designed to make model swapping easy:

### Option A: Drop-in YOLO model

If you have a fine-tuned YOLO model (`.pt` file):

```bash
# Copy your model into the workspace (visible inside Docker)
cp /path/to/my_model.pt /home/triffid/hua_ws/src/

# Run with your model
ros2 run triffid_ugv_perception ugv_node \
  --ros-args -p model_path:=/ws/src/my_model.pt
```

### Option B: Different architecture

If you want to use a completely different model (e.g., Detectron2, custom PyTorch):

1. Edit `ugv_node.py` — replace the `_detect()` method
2. Your method must return a list of dicts:
   ```python
   [
       {
           'bbox': (x1, y1, x2, y2),     # pixel coordinates, float
           'class_id': 0,                  # integer class ID
           'class_name': 'person',         # string class name
           'confidence': 0.95,             # float 0.0–1.0
       },
       ...
   ]
   ```
3. Everything downstream (depth sampling, back-projection, tracking, GeoJSON) works unchanged.

### Updating Target Classes

Edit the `TARGET_CLASSES` dict in `ugv_node.py`:

```python
TARGET_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    # Add your custom classes here
}
```

If using a custom-trained model with different class indices, update the dict to match your model's class mapping.

---

## 13. Diagnostics & Health Monitoring

**File:** `src/triffid_ugv_perception/triffid_ugv_perception/diagnostics.py`
**Executable:** `diagnostics`
**ROS2 Node Name:** `triffid_diagnostics`

Automatically launched by the launch file alongside the perception nodes.

### What It Monitors

| Check | Topic / Source | Condition |
|---|---|---|
| Topic liveness | All input & output topics | Not stale (< 5s since last msg) |
| Message rates | All topics | ≥ 1 Hz for critical inputs |
| Depth-RGB sync | RGB + Depth timestamps | \|t_depth − t_rgb\| < 150 ms |
| TF availability | TF2 buffer | camera_frame → target_frame exists |

### Publications

- `/diagnostics` — `diagnostic_msgs/DiagnosticArray` (standard, works with `rqt_runtime_monitor`)
- `/triffid/heartbeat` — `std_msgs/String` with JSON:
  ```json
  {
    "node": "triffid_diagnostics",
    "stamp": 1740000000,
    "status": "OK",
    "topics": {
      "/camera_front/raw_image": {"hz": 14.9, "alive": true},
      "/ugv/perception/detections_3d": {"hz": 14.8, "alive": true},
      "/triffid/geojson": {"hz": 14.8, "alive": true}
    }
  }
  ```

The heartbeat can be forwarded over the Zenoh bridge to the base station for remote health monitoring.

---

## 14. Integration Test

**File:** `src/triffid_ugv_perception/test/integration_test.py`

Automated test that replays the rosbag and validates all 6 integration requirements:

| Check | Requirement | What It Verifies |
|---|---|---|
| `rosbag` | #3 Replayable dataset | Rosbag directory exists with metadata + db3 |
| `topics` | #4 Run from recorded data | All expected topics receive ≥1 message |
| `types` | #1 Interface definitions | Message types match spec |
| `timestamps` | #5 Timestamp consistency | Detection stamps copied from input (not fabricated), depth-RGB sync < 500ms |
| `fields` | #1 Interface definitions | Every Detection3D has id, class_id, score, frame_id |
| `geojson` | #6 Coordinate frames | GeoJSON is valid RFC-7946, has all SimpleStyle properties |
| `diagnostics` | Health | /diagnostics and /triffid/heartbeat are alive |

### Running

```bash
# Inside Docker container (all nodes must be running):
python3 src/triffid_ugv_perception/test/integration_test.py

# Run a specific check only:
python3 src/triffid_ugv_perception/test/integration_test.py --check geojson

# Skip auto bag playback (if you're already playing it):
python3 src/triffid_ugv_perception/test/integration_test.py --no-bag
```

### Example Output

```
========================================================================
  TRIFFID INTEGRATION TEST RESULTS
========================================================================
  ✓  Req 3: Replayable dataset
     Rosbag found at /ws/rosbag (2 files)
  ✓  Req 4: Topic liveness
     All 7 topics alive
  ✓  Req 1: Message types
     All 7 checked topics have correct types
  ✓  Req 5: Timestamp consistency
     Detection stamps match input epoch (delta 0.066s); depth-RGB sync delta: 76.7 ms
  ✓  Req 1: Detection field validity
     All 6 detections have valid id, class_id, score, frame_id
  ✓  Req 6: GeoJSON / coordinates
     90 GeoJSON msgs, 5 total features, all valid RFC-7946
  ✓  Health: Diagnostics
     Diagnostics active: 6 OK, 1 WARN, 0 ERROR, 1 STALE; heartbeat: STALE
------------------------------------------------------------------------
  All 7 checks passed.
========================================================================
```

---

## 15. Known Limitations & TODOs

### Current Limitations

| Issue | Impact | Fix |
|---|---|---|
| No GPS data in current rosbag (`/fix` has 0 messages) | GeoJSON coordinates require manual GPS origin | Set `gps_origin_lat`/`gps_origin_lon` params |
| TF tree may not contain `map` frame | Back-projection falls back to camera frame | Use `target_frame:=base_link` or check with `ros2 run tf2_tools view_frames` |
| IoU tracker is simple greedy matching | May lose IDs in crowded scenes | Upgrade to ByteTrack or DeepSORT |
| UAV node is a skeleton | Cannot process UAV data yet | Needs UAV rosbag to identify topics |
| UAV geo-projection not implemented | UAV detections in GeoJSON have `[0, 0]` coordinates | Implement camera ray → ground plane intersection |


### TODO List

- [ ] Get UAV rosbag and update topic names
- [ ] Implement UAV geo-projection (pixel → lat/lon via GPS + camera tilt)
- [ ] Deploy fine-tuned model for target domain
- [ ] Check TF frames with `view_frames` and set correct `target_frame`
- [ ] Test API integration with `publish_to_api:=true`
- [ ] Add authentication headers to API calls if required
- [ ] Consider MQTT bridge for live telemetry (alongside HTTP API)
- [ ] Upgrade tracker to ByteTrack for production
- [ ] Add unit tests

---

## Environment Details

| Component | Version |
|---|---|
| Host OS | Ubuntu 24.04.4 LTS |
| Host ROS2 | Jazzy |
| Docker ROS2 | Humble (Jammy) |
| Python (Docker) | 3.10 |
| CUDA | 12.x (via nvidia-container-toolkit) |
| GPU | 2x NVIDIA GeForce RTX 5090 |
| Docker | 29.2.1 |
| Docker Compose | 5.0.2 |
