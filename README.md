# TRIFFID Perception

Perception pipelines for the TRIFFID project — covering both **UGV** (ground robot) and **UAV** (DJI M30T drone) platforms, with a unified MQTT→TELESTO bridge for fusing and publishing merged detections.

| Platform | Package | Runtime | Output |
|----------|---------|---------|--------|
| UGV | `triffid_ugv_perception` | ROS 2 Humble + Docker (GPU) | ROS 2 topics + MQTT + TELESTO |
| UAV | `triffid_uav_perception` | Pure Python + Docker (CPU) | MQTT + GeoJSON files |
| Bridge | `triffid_telesto` | Pure Python (host or container) | TELESTO Map Manager API |

For UAV-specific documentation see [src/triffid_uav_perception/README.md](src/triffid_uav_perception/README.md).

---

## UGV Perception

Real-time 3D object detection pipeline for the TRIFFID UGV platform.
Uses a pixel-aligned RGB-D camera (Intel RealSense with depth aligned to colour, or a live D435i in YUYV mode), runs a fine-tuned YOLOv11l-seg model for 2D detection, samples depth directly at detection pixels and back-projects to 3D, tracks objects across frames, and publishes results as ROS 2 messages and GeoJSON.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hardware Assumptions](#hardware-assumptions)
3. [ROS 2 Topics](#ros-2-topics)
4. [TF Frame Tree](#tf-frame-tree)
5. [Pipeline Steps](#pipeline-steps)
6. [Node Descriptions](#node-descriptions)
7. [Parameters](#parameters)
8. [TELESTO Bridge](#telesto-bridge)
9. [Docker Setup](#docker-setup)
10. [Quick Start (`run.sh`)](#quick-start-runsh)
11. [Running the Integration Test](#running-the-integration-test)
12. [Rosbag Datasets](#rosbag-datasets)
13. [Project Structure](#project-structure)
14. [Interface Specification](#interface-specification)

---

## Architecture Overview

```
 ┌──────────────────────────────┐    ┌──────────────────────────┐
 │  Intel RealSense (RGB-D)     │    │  UAV (DJI M30T)          │
 │  Rosbag: /b2/camera/*        │    │  uav_node.py (Python)    │
 │  Live 435i:                  │    │  YOLO → GPS projection   │
 │  /camera_front_435i/*/       │    │  → GeoJSON               │
 └──────────────┬───────────────┘    └────────────┬─────────────┘
                │                                  │
                ▼                                  │
 ┌─────────────────────────────────┐              │
 │         ugv_node                │              │
 │  1. YOLO on RGB → 2D bboxes     │              │
 │  2. Sample depth at det pixels  │              │
 │  3. Back-project → 3D optical   │              │
 │  4. TF → b2/base_link           │              │
 │  5. ByteTrack → ID              │              │
 │  6. Publish Detection3DArray    │              │
 └──────────┬──────────────────────┘              │
            │                                     │
            ▼                                     │
 ┌──────────────────────────────────┐             │
 │ geojson_bridge                   │             │
 │ Detection3D → GeoJSON (RFC 7946) │             │
 │ GPS + heading rotation           │             │
 │ Spatial dedup (3 m, same class)  │             │
 └──────────┬───────────────────────┘             │
            │ MQTT: ugv/detections/front/geojson  │ MQTT: triffid/uav/geojson
            │                                     │
            ▼                                     ▼
 ┌─────────────────────────────────────────────────────┐
 │  bridge.py  (MQTT → TELESTO)                        │
 │  • normalise class names → lowercase                │
 │  • merge UGV + UAV FeatureCollections               │
 │  • cross-platform dedup (10 m, same class,          │
 │    keep highest confidence)                         │
 │  • sync to TELESTO every 2 s (smart PUT/PATCH/DEL)  │
 └─────────────────────┬───────────────────────────────┘
                       │ HTTPS
                       ▼
 ┌─────────────────────────────────────────────────────┐
 │  TELESTO Map Manager API                            │
 │  crispres.com/wp-json/map-manager/v1                │
 └─────────────────────────────────────────────────────┘
```

---

## Hardware Assumptions

### Rosbag / Default Mode

The default configuration targets an Intel RealSense camera with **depth aligned to colour** (`aligned_depth_to_color`). Both streams share the same pixel grid and intrinsics — no cross-camera projection is needed.

| Property | Value |
|---|---|
| **Camera** | Intel RealSense (RGB-D, depth aligned to colour) |
| **RGB topic** | `/b2/camera/color/image_raw` (configurable) |
| **Depth topic** | `/b2/camera/aligned_depth_to_color/image_raw` (configurable) |
| **CameraInfo topic** | `/b2/camera/color/camera_info` (configurable) |
| **RGB encoding** | `bgr8` |
| **Depth encoding** | `16UC1` (millimetres) |

### Live Mode — Intel RealSense D435i

When the live D435i is connected (via the Jetson, `realsense2_camera` driver v4.57.x), the camera driver publishes YUYV-encoded colour and raw (non-aligned) depth on a different topic namespace.

| Property | Value |
|---|---|
| **Camera** | Intel RealSense D435i (on Jetson) |
| **RGB topic** | `/camera_front_435i/realsense_front_435i/color/image_raw` |
| **Depth topic** | `/camera_front_435i/realsense_front_435i/depth/image_rect_raw` |
| **CameraInfo topic** | `/camera_front_435i/realsense_front_435i/color/camera_info` |
| **RGB encoding** | `yuv422` (YUYV) — converted to BGR in `ugv_node` |
| **Depth encoding** | `16UC1` (millimetres, **not** aligned to colour for PoC) |

Set the three `*_TOPIC` env vars when starting, and the node handles the YUYV→BGR conversion automatically. See [Live Camera Mode](#live-camera-mode--realsense-d435i) below.

---

## ROS 2 Topics

### Subscribed (Input — from rosbag or live robot)

| Topic | Type | Rate | Description |
|---|---|---|---|
| `/b2/camera/color/image_raw` | `sensor_msgs/Image` | ~15 Hz | RGB image (bgr8, or yuv422 for live 435i) |
| `/b2/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | ~15 Hz | Pixel-aligned depth image (16UC1, mm) |
| `/b2/camera/color/camera_info` | `sensor_msgs/CameraInfo` | ~15 Hz | Shared intrinsics matrix K |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | — | Transform tree |
| `/fix` | `sensor_msgs/NavSatFix` | ~0.4 Hz | GPS fix (optional — publishes with `local_frame: true` when absent) |
| `/dog_odom` | `nav_msgs/Odometry` | ~500 Hz | Odometry with heading (optional, for GeoJSON rotation) |

> Topic names are configurable via ROS parameters (`rgb_image_topic`, `depth_image_topic`, `camera_info_topic`). See [Live Camera Mode](#live-camera-mode--realsense-d435i) for 435i-specific values.

### Published (Output)

| Topic | Type | Node | Description |
|---|---|---|---|
| `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` | `ugv_node` | 3D detections in `b2/base_link` frame |
| `/ugv/detections/front/segmentation` | `sensor_msgs/Image` | `ugv_node` | Semantic label map (`mono8`, pixel = class ID) |
| `/ugv/detections/front/debug_image` | `sensor_msgs/Image` | `ugv_node` | Debug overlay: RGB + bboxes, class, track ID (lazy) |
| `/ugv/detections/front/geojson` | `std_msgs/String` | `geojson_bridge` | GeoJSON FeatureCollection (RFC 7946) |

The same GeoJSON payload is also published as MQTT on topic `ugv/detections/front/geojson` (Mosquitto, port `$MQTT_PORT`).

---

## TF Frame Tree

The pipeline requires a single transform from the camera's optical frame to `b2/base_link`. The camera frame ID is **read dynamically** from `CameraInfo.header.frame_id`.

```
b2/base_link
└── b2/camera_optical_frame     (from CameraInfo header)
      Convention: ROS optical (X=right, Y=down, Z=forward)
      Published by robot's driver / localization stack
```

---

## Pipeline Steps

The core processing runs on every RGB frame in `rgb_callback`:

1. **YOLO Detection** — Run the fine-tuned YOLOv11l-seg model (63 disaster-response classes). YUYV input is converted to BGR before inference.

2. **Pixel-aligned Depth Sampling** — For each detection, sample depth at the instance mask pixels (or bbox pixels if no mask). Depth is read directly at detection pixel coordinates — shared intrinsics, no cross-camera TF. Large masks are sub-sampled to ≤500 depth samples for efficiency. Zero-depth pixels are discarded.

3. **Back-project to 3D** — Using shared intrinsics from `CameraInfo`, back-project valid depth samples to 3D points in **camera_optical_frame** (X=right, Y=down, Z=forward).

4. **Median 3D Position** — Take the median of back-projected points. Transform to `b2/base_link` via TF.

5. **3D Extent Estimation** — Compute bounding box extent from point spread. Falls back to 2D bbox corner back-projection at median depth when points cluster too tightly.

6. **ByteTrack Tracking** — Kalman-filter motion prediction, Hungarian assignment (two-pass: high-conf then low-conf), class gate (no cross-class matches), majority-vote class label. Tracks go TENTATIVE → CONFIRMED → LOST; only confirmed tracks are published. Re-ID via 3D distance gate.

7. **Publish** — Emit `Detection3DArray` and per-pixel semantic label map (mono8).

---

## Node Descriptions

### `ugv_node` (main perception)

- Fine-tuned `yolo11l-seg` (63 classes), pixel-aligned RGB-D depth sampling
- YUYV→BGR conversion for live RealSense 435i (`yuv422` encoding)
- ByteTrack-style tracker (Kalman + Hungarian + class gate + majority-vote + 3D gate)
- Camera frame read dynamically from `CameraInfo.header.frame_id`
- Configurable topic names via ROS parameters
- Lazy debug image (only published when subscribed)

### `geojson_bridge`

- Converts `Detection3DArray` → GeoJSON (RFC 7946)
- Publishes with `"local_frame": true` when no GPS fix — never blocks the pipeline
- GPS median-filtered (window 7); heading from `/dog_odom`
- Class-dependent geometry: Point (mobile targets), LineString (fence/wall), Polygon (area targets)
- **Spatial deduplication** before publish: same class within 3 m → keep highest confidence
- MQTT publish to local Mosquitto broker (topic `ugv/detections/front/geojson`)

---

## Parameters

### ugv_node Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `/ws/best.pt` | YOLO model weights |
| `confidence_threshold` | `0.35` | Minimum detection confidence |
| `target_frame` | `b2/base_link` | Output frame for 3D positions |
| `rgb_image_topic` | `/b2/camera/color/image_raw` | RGB image topic |
| `depth_image_topic` | `/b2/camera/aligned_depth_to_color/image_raw` | Depth image topic |
| `camera_info_topic` | `/b2/camera/color/camera_info` | CameraInfo topic |
| `use_dummy_detections` | `false` | Bypass YOLO with a full-image dummy bbox |
| `yolo_imgsz` | `1280` | YOLO input resolution (pixels) |
| `tracker_iou_threshold` | `0.30` | High-confidence IoU threshold |
| `tracker_iou_threshold_low` | `0.15` | Low-confidence IoU threshold |
| `tracker_conf_high` | `0.40` | Confidence split for two-pass matching |
| `tracker_max_age` | `30` | Frames before a lost track is retired |
| `tracker_n_init` | `3` | Consecutive hits before a track is confirmed |
| `tracker_pos_gate` | `2.0` | 3D distance gate (m) for re-ID |

### GeoJSON Bridge Parameters

| Parameter | Default | Description |
|---|---|---|
| `api_url` | `https://crispres.com/...` | TRIFFID mapping API endpoint |
| `publish_to_api` | `false` | Enable HTTP PUT to the API |
| `gps_origin_lat` | `0.0` | GPS origin latitude (or auto from `/fix`) |
| `gps_origin_lon` | `0.0` | GPS origin longitude |
| `mqtt_enabled` | `true` | Enable MQTT publishing |
| `mqtt_host` | `localhost` | MQTT broker hostname |
| `mqtt_port` | `1883` | MQTT broker port |
| `mqtt_topic` | `ugv/detections/front/geojson` | MQTT output topic |
| `dedup_radius_m` | `3.0` | Spatial dedup radius (m); same-class features within this distance are merged, highest confidence kept |

---

## TELESTO Bridge

The `triffid_telesto` package provides a MQTT→TELESTO bridge that:
1. Subscribes to both UGV (`ugv/detections/front/geojson`) and UAV (`triffid/uav/geojson`) MQTT topics
2. Normalises all class names to lowercase on receipt
3. Merges UGV + UAV features into one FeatureCollection
4. Deduplicates cross-platform: same class within **10 m** → keep highest confidence (`local_frame: true` UGV features always pass through)
5. Syncs to TELESTO Map Manager API every 2 s via smart PUT/PATCH/DELETE (only when new data arrives)
6. Notifies the observer endpoint on successful sync

### TELESTO Architecture

```
MQTT broker (localhost:$MQTT_PORT)
  ugv/detections/front/geojson  ──┐
  triffid/uav/geojson            ──┤
                                   ▼
                             bridge._on_message()
                                lowercase 'class' property
                                store as _ugv_latest / _uav_latest
                                   │ (every 2 s, if dirty)
                                   ▼
                             bridge._merge()
                                concatenate UGV + UAV features
                                _deduplicate_features(radius=10 m)
                                   │
                                   ▼
                             TelestoClient.sync_collection()
                                GET current remote features
                                PUT new / PATCH changed / DELETE stale
                                notify_observer(fe_updated=1)
```

### Running the Bridge

The bridge starts automatically with `./run.sh start`. To run it standalone:

```bash
# Inside the container (PYTHONPATH required — no colcon package)
PYTHONPATH=/ws/src python3 -m triffid_telesto.bridge \
    --mqtt-host localhost \
    --mqtt-port 1883

# Dry-run (print merged GeoJSON, don't upload):
PYTHONPATH=/ws/src python3 -m triffid_telesto.bridge --dry-run

# Custom Telesto endpoint:
TELESTO_BASE_URL=https://my-server.com/api \
PYTHONPATH=/ws/src python3 -m triffid_telesto.bridge
```

### Simulating UAV Data

While the UGV container is running, inject mock UAV data directly:

```bash
docker exec triffid_perception bash -c "
python3 -c \"
import paho.mqtt.client as mqtt, json
c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.connect('localhost', 1883)
c.publish('triffid/uav/geojson', json.dumps({
  'type': 'FeatureCollection',
  'features': [{
    'type': 'Feature', 'id': 'a1',
    'geometry': {'type': 'Point', 'coordinates': [23.720, 37.980]},
    'properties': {'class': 'flame', 'confidence': 0.88,
                   'source': 'uav', 'local_frame': False}
  }]
}))
c.disconnect()
\"
"
./run.sh logs    # check bridge merged output
```

### Testing the Bridge

```bash
# Unit tests only (no container, no MQTT, no network):
cd src/triffid_telesto && python3 -m pytest test_telesto.py -v

# End-to-end smoke test (container must be running):
./run.sh bridge-test
```

---

## Docker Setup

### Container Details

- **Image**: `triffid_perception:latest`
- **Base**: ROS 2 Humble on Ubuntu 22.04
- **GPU**: NVIDIA (via `nvidia-container-toolkit`)
- **Network**: `host` mode (shares host DDS network)
- **IPC**: `host` (shared memory for fast DDS transport)
- **ROS_DOMAIN_ID**: `42`
- **DDS**: CycloneDDS with increased fragment buffers for large images
- **MQTT**: Mosquitto broker installed in image, started automatically by `run.sh start`

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
| `./samples/` | `/ws/samples/` | Output samples (camtest PNGs, GeoJSON, etc.) |
| `./output_rosbag/` | `/ws/output_rosbag/` | Recorded output rosbag |

---

## Quick Start (`run.sh`)

The all-in-one `run.sh` wraps Docker and launches the full stack:

```bash
cd ~/hua_ws
./run.sh build           # Build Docker image + colcon workspace
./run.sh start           # Launch ugv_node + geojson_bridge + mosquitto + TELESTO bridge + rosbag
./run.sh stop            # Stop everything
./run.sh restart         # stop + start
./run.sh logs            # Tail all node logs (ugv, geojson, bridge, mosquitto)
./run.sh sample [SEC]    # Collect output samples + merged GeoJSON (default: 15 s)
./run.sh test            # Run integration test (30 s timeout)
./run.sh unit            # Run UGV unit tests (host, no Docker)
./run.sh shell           # Open a bash shell in the container
./run.sh status          # Check running processes + topic list
./run.sh camtest         # Grab one frame from the live RealSense 435i
./run.sh bridge-test     # Smoke-test the TELESTO bridge with mock UGV+UAV data
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BAG_RATE` | `1.0` | Rosbag playback rate |
| `BAG_START` | `0` | Skip this many seconds at the start of the bag |
| `YOLO_IMGSZ` | `1280` | YOLO inference resolution |
| `TIMEOUT` | `30` | Integration test timeout (seconds) |
| `MQTT_PORT` | `1883` | Mosquitto broker port — override if 1883 is taken on this machine |
| `TELESTO_BASE_URL` | *(built-in)* | Override TELESTO Map Manager endpoint |
| `TELESTO_OBSERVER_URL` | *(built-in)* | Override TELESTO Observer endpoint |
| `RGB_TOPIC` | *(empty)* | Override color image topic for `ugv_node` |
| `DEPTH_TOPIC` | *(empty)* | Override depth image topic for `ugv_node` |
| `CAMERA_INFO_TOPIC` | *(empty)* | Override CameraInfo topic for `ugv_node` |
| `CAMTEST_RGB_TOPIC` | `/camera_front_435i/.../color/image_raw` | Color topic for `camtest` only |
| `CAMTEST_DEPTH_TOPIC` | `/camera_front_435i/.../depth/image_rect_raw` | Depth topic for `camtest` only |

> `RGB_TOPIC`/`DEPTH_TOPIC`/`CAMERA_INFO_TOPIC` control what `ugv_node` subscribes to.
> `CAMTEST_RGB_TOPIC`/`CAMTEST_DEPTH_TOPIC` are separate — they default to the live 435i topics and are not affected by rosbag replay overrides in the shell environment.

### Live Camera Mode — RealSense D435i

When the partner connects the D435i (camera topics appear as `/camera_front_435i/...`):

```bash
# Quick sanity check — saves samples/camtest_color.png + camtest_depth.png
./run.sh camtest

# Full pipeline with live camera
RGB_TOPIC=/camera_front_435i/realsense_front_435i/color/image_raw \
DEPTH_TOPIC=/camera_front_435i/realsense_front_435i/depth/image_rect_raw \
CAMERA_INFO_TOPIC=/camera_front_435i/realsense_front_435i/color/camera_info \
./run.sh start
```

### Rosbag Mode (default)

```bash
./run.sh start     # uses /b2/camera/* defaults, plays rosbag automatically
./run.sh sample    # saves frames, detections, GeoJSON, MQTT trace to ./samples/
./run.sh test      # full integration test
```

### `sample` Output Files

`./run.sh sample` saves the following to `./samples/`:

| File | Contents |
|---|---|
| `rgb_frame_0..4.jpg` | 5 raw RGB frames |
| `detections_3d.yaml` | First non-empty `Detection3DArray` message |
| `segmentation.png` | First semantic label map (mono8) |
| `geojson.json` | First non-empty GeoJSON FeatureCollection |
| `geojson_raw.json` | All confirmed tracks (no spatial dedup) |
| `geojson_merged.json` | Spatially deduplicated per class (1 m) |
| `mqtt_trace.jsonl` | Every MQTT GeoJSON message (one JSON per line) |
| `tracking_debug.mp4` | H.264 debug overlay video |
| `track_lifecycle.csv` | Per-frame track presence table |
| `possible_id_switches.csv` | Detected potential ID switches |
| `camtest_color.png` | Live camera colour frame (from `./run.sh camtest`) |
| `camtest_depth.png` | Live camera depth frame, false-colour JET map |

---

## Running the Integration Test

```bash
./run.sh test
```

Or manually inside the container:

```bash
docker exec triffid_perception bash -c "
  source /ws/install/setup.bash &&
  python3 /ws/src/triffid_ugv_perception/test/integration_test.py --timeout 45
"
```

### What it checks

| # | Check | Requirement |
|---|---|---|
| 1 | Output rosbag recorded (metadata + db3) | Replayable dataset |
| 2 | All expected topics receive messages | Topic liveness |
| 3 | Message types match spec | Interface definitions |
| 4 | CameraInfo valid (frame, resolution, intrinsics) | Sensor validation |
| 5 | Required TF transforms available | Coordinate frames |
| 6 | Detection timestamps from rosbag; depth–RGB sync <500 ms | Timestamp consistency |
| 7 | Detection fields populated (id, class, score, frame_id) | Interface definitions |
| 8 | 3D positions finite, non-zero, within 30 m range | Depth pipeline sanity |
| 9 | Persistent tracking IDs, no duplicates per frame | Tracking correctness |
| 10 | GeoJSON RFC-7946 valid, required properties present | GeoJSON schema |
| 11 | Segmentation label map (mono8) published | Segmentation output |

### Options

```bash
python3 integration_test.py --check timestamps   # single check
python3 integration_test.py --no-launch           # skip launching nodes
python3 integration_test.py --no-bag              # skip rosbag playback
python3 integration_test.py --timeout 60          # increase timeout
```

---

## Rosbag Datasets

The rosbag is not included in the repository (17+ GB binary). Mount it via `docker-compose.yml`.

### Required topics

- `/b2/camera/color/image_raw`
- `/b2/camera/color/camera_info`
- `/b2/camera/aligned_depth_to_color/image_raw`
- `/tf_static`

> **Legacy bags** with `/camera_front/raw_image` and `/camera_front/realsense_front/depth/...` topics are compatible by setting the topic env vars when starting.

---

## Project Structure

```
hua_ws/
├── docker-compose.yml          # UGV container (GPU, volumes, network, MQTT_PORT passthrough)
├── docker-compose.uav.yml      # UAV container (lightweight, no ROS2)
├── Dockerfile                  # UGV image: ROS2 Humble + CUDA + ultralytics
├── Dockerfile.uav              # UAV image: Python 3.10 + ultralytics
├── run.sh                      # UGV all-in-one runner (build/start/stop/test/camtest/bridge-test)
├── run_uav.sh                  # UAV all-in-one runner
├── run_telesto.sh              # TELESTO bridge runner (stub)
├── docker_entrypoint.sh
├── cyclonedds.xml              # DDS tuning for large image fragments
├── classes.txt                 # 63-class ontology
│
└── src/
    ├── triffid_ugv_perception/     # UGV perception (ROS2 + MQTT)
    │   ├── package.xml
    │   ├── config/
    │   │   └── bag_qos_overrides.yaml
    │   ├── launch/
    │   │   └── ugv_perception.launch.py
    │   ├── triffid_ugv_perception/
    │   │   ├── ugv_node.py          # Pixel-aligned RGB-D pipeline (+ YUYV support)
    │   │   ├── tracker.py           # ByteTrack-style multi-object tracker
    │   │   └── geojson_bridge.py    # Detection3D → GeoJSON + dedup + MQTT
    │   ├── scripts/
    │   │   ├── camtest.py           # RealSense 435i one-shot frame grab
    │   │   └── collect_samples.py   # Output sample collector
    │   └── test/
    │       ├── integration_test.py  # 11-check end-to-end test
    │       └── test_unit.py         # 164 unit tests
    │
    ├── triffid_uav_perception/     # UAV perception (standalone Python, MQTT output)
    │   ├── requirements.txt
    │   ├── README.md
    │   ├── triffid_uav_perception/
    │   │   ├── metadata.py          # DJI XMP metadata extraction
    │   │   ├── geo.py               # Geo-projection (pixel → GPS)
    │   │   ├── api_client.py        # FUTURISED API client
    │   │   └── uav_node.py          # Pipeline + direct MQTT publish
    │   └── test/
    │       └── test_unit.py         # 50 unit tests
    │
    └── triffid_telesto/            # MQTT→TELESTO bridge (pure Python)
        ├── telesto_client.py        # TelestoClient: GET/PUT/PATCH/DELETE + sync + observer
        ├── bridge.py                # Bridge: subscribe MQTT → normalise → dedup → sync
        ├── smoke_test.py            # End-to-end smoke test (mock UGV+UAV data)
        └── test_telesto.py          # 38 unit tests (client, sync, merge, dedup, normalisation)
```

**Test counts:** 164 UGV + 50 UAV + 38 Telesto = **252 total**

---

## Interface Specification

For integration partners: the full frozen interface document is in **[INTERFACE.md](INTERFACE.md)**.
It specifies every topic, message field, frame convention, GeoJSON schema, class list, and encoding.
