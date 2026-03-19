# TRIFFID UAV Perception

Standalone perception pipeline for the TRIFFID UAV platform (DJI Matrice M30T).  
Processes drone images with embedded XMP metadata, runs segmentation, projects detections to ground GPS coordinates, and publishes GeoJSON via MQTT.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hardware Assumptions](#hardware-assumptions)
3. [Pipeline Steps](#pipeline-steps)
4. [Geo-Projection](#geo-projection)
5. [GeoJSON Schema](#geojson-schema)
6. [Docker Setup](#docker-setup)
7. [Quick Start (`run_uav.sh`)](#quick-start-run_uavsh)
8. [Running Tests](#running-tests)
9. [Project Structure](#project-structure)
10. [Configuration](#configuration)
11. [Future Work](#future-work)

---

## Architecture Overview

Unlike the UGV pipeline (which uses ROS2), the UAV pipeline is a standalone Python application that communicates solely via MQTT. This keeps it lightweight and decoupled from the ROS2 ecosystem.

```
┌─────────────────────────────────┐
│  DJI M30T Image (JPEG/TIFF)    │
│  with embedded XMP metadata     │
│  (GPS, gimbal, LRF, RTK)       │
└──────────┬──────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         uav_node.py                  │
│                                      │
│  1. Extract XMP metadata from image  │
│  2. Validate RTK quality + LRF      │
│  3. Run YOLO-seg on RGB             │
│  4. For each detection:              │
│     a. Get mask contour (pixels)     │
│     b. Project pixels → ground GPS   │
│        via gimbal angles + altitude  │
│     c. Estimate object height        │
│  5. Build GeoJSON FeatureCollection  │
│  6. Publish to MQTT                  │
└──────────┬───────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│  MQTT Broker              │
│  triffid/uav/geojson      │
│  (Mosquitto, port 1883)   │
└──────────────────────────┘
```

---

## Hardware Assumptions

| Property | Value |
|---|---|
| **Drone** | DJI Matrice 30T |
| **Camera** | Wide Camera (84° diagonal FoV, 12MP) |
| **GPS** | RTK-enabled (WGS-84 ellipsoidal altitude) |
| **LRF** | Laser rangefinder (distance + target GPS when status = Normal) |
| **Image format** | JPEG with embedded XMP metadata |

### DJI XMP Metadata Fields Used

| Field | Description | Usage |
|---|---|---|
| `GpsLatitude` / `GpsLongitude` | Drone body position (WGS-84) | Camera origin for ray-casting |
| `AbsoluteAltitude` | Ellipsoidal height (m) | Height above ground calculation |
| `GimbalYawDegree` | Absolute yaw from True North (°, clockwise) | Ray rotation in NED |
| `GimbalPitchDegree` | Pitch angle (°, negative = down) | Ray rotation in NED |
| `GimbalRollDegree` | Roll angle (°, usually ~0) | Ray rotation in NED |
| `RtkFlag` | RTK quality (50 = cm-level fixed) | Quality gating |
| `LRFStatus` | Rangefinder status | Depth reference validation |
| `LRFTargetAbsAlt` | Ground point altitude (m) | Ground plane height |

### RTK Quality Flags

| Flag | Meaning | Accuracy |
|---|---|---|
| 0 | No satellite signal | — |
| 15 | No position solution | — |
| 16 | Single-point solution | Metre-level |
| 34 | Float solution | Decimetre-level |
| 50 | Fixed integer solution | Centimetre-level |

### Gimbal Convention

The DJI gimbal angles are defined in an NED (North-East-Down) frame:
- **Yaw**: 0° = North, 90° = East, clockwise positive
- **Pitch**: 0° = horizontal, negative = looking down, -90° = nadir
- **Roll**: 0° = level (usually ~0 in flight)

`GimbalYawDegree` is **absolute** (relative to True North), not relative to the drone body. This means we don't need `FlightYawDegree` for computing the camera pointing direction.

---

## Pipeline Steps

### 1. Metadata Extraction (`metadata.py`)

Reads the XMP block embedded in JPEG/TIFF files using a regex match on the raw bytes. Parses the `drone-dji` XML namespace attributes into a typed `DJIMetadata` dataclass. No external EXIF library required.

### 2. YOLO Segmentation (`uav_node.py`)

Runs a YOLO segmentation model (default: `yolo11n-seg.pt` for PoC, swap in your trained model via `--model`). The model returns:
- 2D bounding boxes with class and confidence
- Per-instance binary segmentation masks

The same 63-class TRIFFID model used by the UGV works here. The `--model` flag accepts any ultralytics-compatible weights file.

### 3. Geo-Projection (`geo.py`)

Projects 2D image pixels to WGS-84 ground coordinates using:

1. **Pixel → camera ray**: Pinhole model with camera intrinsics
2. **Camera ray → NED ray**: Rotation by gimbal yaw/pitch/roll
3. **NED ray → ground intersection**: Ray-cast to horizontal plane at ground altitude
4. **Ground NED offset → GPS**: Equirectangular approximation

Ground altitude is determined (in priority order):
1. LRF target absolute altitude (when `LRFStatus = Normal`)
2. Drone absolute altitude minus relative altitude (fallback)
3. Explicit `ground_alt` parameter (override)

### 4. GeoJSON Output

Same schema as the UGV pipeline for consistency:
- `"source": "uav"` (vs `"ugv"` for the ground vehicle)
- Same properties: `class`, `id`, `confidence`, `category`, `detection_type`, `gnss_altitude_m`, `height_m`
- Same SimpleStyle: `marker-color`, `stroke`, `fill`, etc.
- Same geometry rules: Polygon for most classes, Point for persons, LineString for fences

### 5. MQTT Publishing

Publishes compact JSON to `triffid/uav/geojson` (QoS 0). Subscribe with:
```bash
mosquitto_sub -h localhost -t 'triffid/uav/geojson'
```

---

## Geo-Projection

### Camera Model

The M30T Wide Camera is modelled as a pinhole camera:

| Resolution | fx | fy | cx | cy |
|---|---|---|---|---|
| 4000×3000 | 2850 | 2850 | 2000 | 1500 |
| 1920×1080 | 1370 | 1370 | 960 | 540 |

These are approximate defaults from the DJI specs (84° diagonal FoV). Override with calibrated values via the `CameraIntrinsics` class.

### Coordinate Frames

```
Camera frame (DJI optical):     NED world frame:
  X = right                       X = North
  Y = down                        Y = East
  Z = forward (out of lens)       Z = Down
```

The rotation matrix converts camera rays to NED, accounting for gimbal yaw, pitch, and roll.

### Projection Method

For each detection mask:
1. Find the mask contour using OpenCV
2. Sample contour points (adaptive step, minimum 4 points)
3. Project each contour point through the camera model → NED ray → ground intersection
4. Convert ground NED offsets to WGS-84 GPS coordinates
5. Close the polygon ring (RFC 7946)
6. Project the mask centroid for the feature centre coordinate

---

## GeoJSON Schema

Identical to the UGV output (see main [INTERFACE.md](../INTERFACE.md)):

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "1",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[ [lon,lat], [lon,lat], ... ]]
      },
      "properties": {
        "class": "Building",
        "id": "1",
        "confidence": 0.85,
        "category": "infrastructure",
        "detection_type": "seg",
        "source": "uav",
        "local_frame": false,
        "gnss_altitude_m": 409.1,
        "height_m": 16.5,
        "marker-color": "#708090",
        "marker-size": "medium",
        "marker-symbol": "building",
        "stroke": "#708090",
        "stroke-width": 2,
        "stroke-opacity": 1.0,
        "fill": "#708090",
        "fill-opacity": 0.25
      }
    }
  ]
}
```

The only difference from UGV output: `"source": "uav"`.

---

## Docker Setup

The UAV pipeline uses a separate, lightweight Docker image (no ROS2, no GPU required for PoC).

### Container Details

- **Image**: `triffid_uav_perception:latest`
- **Base**: Python 3.10 slim
- **Network**: `host` mode (for MQTT access)
- **MQTT**: Mosquitto broker installed in image

### Volumes

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src/triffid_uav_perception/` | `/app/src/triffid_uav_perception/` | Source code (editable) |
| `./uav_images/` | `/app/images/` | Input images |
| `./uav_samples/` | `/app/samples/` | Output GeoJSON |
| `./best.pt` | `/app/best.pt` | YOLO model (read-only) |

---

## Quick Start (`run_uav.sh`)

```bash
cd ~/hua_ws

# Build the Docker image
./run_uav.sh build

# Process a single image (place it in uav_images/ first)
cp /path/to/drone_photo.jpg uav_images/
./run_uav.sh process drone_photo.jpg

# Process all images in a directory
./run_uav.sh batch

# Watch for new images (processes as they appear)
./run_uav.sh watch

# Run unit tests
./run_uav.sh test

# Open a shell in the container
./run_uav.sh shell

# Stop the container
./run_uav.sh stop
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `/app/best.pt` | YOLO model path inside container |
| `CONFIDENCE` | `0.35` | Detection confidence threshold |
| `MQTT_HOST` | `localhost` | MQTT broker hostname |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_TOPIC` | `triffid/uav/geojson` | MQTT output topic |
| `IMGSZ` | `1280` | YOLO input resolution |

### Running Without Docker

```bash
cd ~/hua_ws
pip install -r src/triffid_uav_perception/requirements.txt
PYTHONPATH=src/triffid_uav_perception python3 -m triffid_uav_perception.uav_node \
    --image uav_images/photo.jpg \
    --model best.pt \
    --mqtt-host localhost
```

---

## Running Tests

```bash
# From host (no Docker needed):
cd ~/hua_ws
PYTHONPATH=src/triffid_uav_perception python3 -m pytest src/triffid_uav_perception/test/ -v

# From inside Docker:
./run_uav.sh test
```

---

## Project Structure

```
hua_ws/
├── docker-compose.uav.yml          # UAV container config
├── Dockerfile.uav                   # UAV image (Python 3.10 + deps)
├── run_uav.sh                       # All-in-one runner script
├── uav_images/                      # Input: drone images (host)
├── uav_samples/                     # Output: GeoJSON results (host)
│
├── docker-compose.yml               # UGV container (unchanged)
├── Dockerfile                       # UGV image (unchanged)
├── run.sh                           # UGV runner (unchanged)
│
└── src/
    ├── triffid_uav_perception/      # ← UAV package
    │   ├── requirements.txt
    │   ├── triffid_uav_perception/
    │   │   ├── __init__.py
    │   │   ├── metadata.py          # XMP metadata extraction
    │   │   ├── geo.py               # Geo-projection (pixel → GPS)
    │   │   └── uav_node.py          # Main pipeline + MQTT
    │   └── test/
    │       └── test_unit.py         # 34 unit tests
    │
    └── triffid_ugv_perception/      # UGV package (unchanged)
        └── ...
```

---

## Configuration

### Camera Intrinsics

The default intrinsics are approximate. For better accuracy, calibrate the M30T Wide Camera and pass custom values:

```python
from triffid_uav_perception.geo import CameraIntrinsics

custom = CameraIntrinsics(
    fx=2900, fy=2900, cx=2010, cy=1505,
    width=4000, height=3000,
)
pipeline = UAVPipeline(intrinsics=custom)
```

### Swapping the Model

The pipeline accepts any ultralytics-compatible model file. To use a SegFormer or other model later:

1. Export your model to a format ultralytics can load, or
2. Subclass `UAVPipeline` and override `_detect()` to call your model directly

For PoC, `yolo11n-seg.pt` (pretrained on COCO) works out of the box:
```bash
MODEL=yolo11n-seg.pt ./run_uav.sh batch
```

---

## Future Work

1. **API integration**: When the DJI frame API is available, replace the file-based input with a streaming HTTP/WebSocket client
2. **SegFormer model**: Train and integrate the planned SegFormer model (override `_detect()`)
3. **Tracking**: Add cross-frame tracking (IoU or feature-based) when processing video sequences
4. **Thermal camera**: Extend metadata extraction to handle thermal (`ThermalCamera`) ImageSource
5. **GPU support**: Add NVIDIA runtime to `Dockerfile.uav` when GPU inference is needed for larger models
6. **Camera calibration**: Replace default intrinsics with factory-calibrated or self-calibrated values
