# TRIFFID UGV Perception — Interface Specification

> **Version**: 1.3  
> **Date**: 2026-03-04  
> **Status**: FROZEN — changes require version bump and partner notification
> **ROS 2 Distribution**: Humble  
> **DDS**: CycloneDDS  
> **ROS_DOMAIN_ID**: 42  

This document defines every published and consumed interface of the TRIFFID UGV perception pipeline. Integration partners should treat this as the single source of truth.

---

## Table of Contents

1. [Published Topics (Output)](#1-published-topics-output)
2. [Subscribed Topics (Input)](#2-subscribed-topics-input)
3. [TF Frames](#3-tf-frames)
4. [Detection3DArray Schema](#4-detection3darray-schema)
5. [Segmentation Image Schema](#5-segmentation-image-schema)
6. [GeoJSON Schema](#6-geojson-schema)
   - [MQTT Output](#mqtt-output)
7. [Class List (63 classes)](#7-class-list-63-classes)
8. [Category Mapping](#8-category-mapping)
9. [Coordinate Conventions](#9-coordinate-conventions)
10. [QoS Profiles](#10-qos-profiles)
11. [Current Limitations](#11-current-limitations)

---

## 1. Published Topics (Output)

| Topic | Message Type | Encoding | Rate | Description |
|---|---|---|---|---|
| `/ugv/perception/front/detections_3d` | `vision_msgs/msg/Detection3DArray` | — | ~15 Hz | 3D bounding boxes in `b2/base_link` |
| `/ugv/perception/front/segmentation` | `sensor_msgs/msg/Image` | `mono8` | ~15 Hz | Semantic label map (pixel = class ID) |
| `/triffid/front/geojson` | `std_msgs/msg/String` | JSON (UTF-8) | ~15 Hz | GeoJSON FeatureCollection (RFC 7946) |

The same GeoJSON payload is also published to a local **MQTT** broker (see [§6 MQTT Output](#mqtt-output)).

All output timestamps are copied from the triggering RGB frame (rosbag sim time), **not** wall clock.

---

## 2. Subscribed Topics (Input)

| Topic | Message Type | Encoding | Expected Rate | Required |
|---|---|---|---|---|
| `/camera_front/raw_image` | `sensor_msgs/msg/Image` | `bgr8` | ~15 Hz | Yes |
| `/camera_front/camera_info` | `sensor_msgs/msg/CameraInfo` | — | ~15 Hz | Yes |
| `/camera_front/realsense_front/depth/image_rect_raw` | `sensor_msgs/msg/Image` | `16UC1` (mm) | ~15 Hz | Yes |
| `/camera_front/realsense_front/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | — | ~15 Hz | Yes |
| `/tf` | `tf2_msgs/msg/TFMessage` | — | — | Yes |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | — | Once | Yes |
| `/fix` | `sensor_msgs/msg/NavSatFix` | — | ~0.4 Hz | Optional (GPS) |
| `/dog_odom` | `nav_msgs/msg/Odometry` | — | ~500 Hz | Optional (heading) |

---

## 3. TF Frames

### Required Static Transform Chain

```
b2/base_link
├── f_oc_link                          RGB camera (body convention)
│     Translation: (0.3993, 0.0, -0.0158)
│     Rotation:    (0, 0, 0, 1)         identity
│
└── f_dc_link                          Depth camera mount (~45° pitch)
      Translation: (0.4216, 0.025, 0.0619)
      Rotation:    (0, 0.3827, 0, 0.9239)  ~45° about Y
      │
      └── f_depth_frame                (identity from f_dc_link)
            │
            └── f_depth_optical_frame  ROS optical convention
                  Rotation: (-0.5, 0.5, -0.5, 0.5)
```

### Frame Conventions

| Frame | Convention | Axes |
|---|---|---|
| `b2/base_link` | ROS body | X = forward, Y = left, Z = up |
| `f_oc_link` | ROS body | X = forward, Y = left, Z = up |
| `f_depth_optical_frame` | ROS optical | X = right, Y = down, Z = forward |

### Body ↔ Optical Conversion (used internally)

```
X_optical = −Y_body     (right  = −left)
Y_optical = −Z_body     (down   = −up)
Z_optical =  X_body     (forward = forward)
```

---

## 4. Detection3DArray Schema

**Topic**: `/ugv/perception/front/detections_3d`  
**Type**: `vision_msgs/msg/Detection3DArray`  
**Frame**: `b2/base_link`

### Header

| Field | Type | Value |
|---|---|---|
| `header.stamp` | `builtin_interfaces/Time` | RGB frame timestamp (sim time) |
| `header.frame_id` | `string` | `"b2/base_link"` |

### Per Detection (`Detection3D`)

| Field | Type | Unit | Description |
|---|---|---|---|
| `id` | `string` | — | Persistent track ID (positive integer, never reused within a session) |
| `bbox.center.position.x` | `float64` | metres | Forward distance from base_link |
| `bbox.center.position.y` | `float64` | metres | Lateral offset (positive = left) |
| `bbox.center.position.z` | `float64` | metres | Vertical offset (positive = up) |
| `bbox.size.x` | `float64` | metres | Extent along X (forward) |
| `bbox.size.y` | `float64` | metres | Extent along Y (lateral) |
| `bbox.size.z` | `float64` | metres | Extent along Z (vertical) |
| `results[0].hypothesis.class_id` | `string` | — | Class name (see [Class List](#7-class-list-63-classes)) |
| `results[0].hypothesis.score` | `float64` | 0–1 | YOLO confidence |

### Notes

- `results` always has exactly 1 entry per detection
- `class_id` is the **human-readable class name** (e.g. `"Civilian vehicle"`), not a numeric ID
- `bbox.size` may be `(0, 0, 0)` if depth evidence is insufficient (rare)
- Detections at the same 3D position (within 0.5 m) are deduplicated; highest confidence kept

---

## 5. Segmentation Image Schema

**Topic**: `/ugv/perception/front/segmentation`  
**Type**: `sensor_msgs/msg/Image`

| Field | Value |
|---|---|
| `encoding` | `mono8` |
| `width` | `1280` |
| `height` | `720` |
| `step` | `1280` |
| `header.frame_id` | `f_oc_link` |
| `header.stamp` | RGB frame timestamp (sim time) |

### Pixel Values

| Pixel Value | Meaning |
|---|---|
| `0` | Background (no detection) |
| `1` – `63` | 1-based class ID: `TARGET_CLASSES[pixel - 1]` |

Example: pixel value `15` → class ID `14` → `"Building"`.

When instance masks overlap, the highest-confidence detection's class ID is written.

---

## 6. GeoJSON Schema

**Topic**: `/triffid/front/geojson`  
**Type**: `std_msgs/msg/String`  
**Encoding**: JSON UTF-8  
**Standard**: RFC 7946

### FeatureCollection

```json
{
  "type": "FeatureCollection",
  "features": [ ... ]
}
```

### Feature

```json
{
  "type": "Feature",
  "id": "42",
  "geometry": { ... },
  "properties": { ... }
}
```

### Geometry

Two geometry types are emitted depending on whether the detection has a non-zero 3D extent:

**Point** (when `bbox.size.x == 0` **and** `bbox.size.y == 0`):
```json
{
  "type": "Point",
  "coordinates": [longitude, latitude, altitude]
}
```

**Polygon** (when `bbox.size.x > 0` **or** `bbox.size.y > 0`):
```json
{
  "type": "Polygon",
  "coordinates": [[ [lon,lat,alt], [lon,lat,alt], [lon,lat,alt], [lon,lat,alt], [lon,lat,alt] ]]
}
```
The polygon is a closed axis-aligned rectangle (5 points, first = last) derived from the ground-plane projection of the 3D bounding box. If only one dimension (`size.x` or `size.y`) is non-zero, the zero dimension is clamped to a minimum extent of **0.3 m** so the polygon remains visible on the map.

### Coordinates

- **GPS gating**: GeoJSON is **not published** until at least one valid `/fix` message has been received. This prevents body-frame metre values from being emitted as lon/lat coordinates.
- **With GPS** (`/fix` received or `gps_origin_lat/lon` set): WGS-84 `[longitude, latitude, altitude]` per RFC 7946 §3.1.1. GPS positions are median-filtered over a sliding window of 7 fixes to reduce noise.
- **With heading** (`/dog_odom` received): body-frame detection offsets are rotated by the robot's yaw (ENU convention) into east/north before GPS projection.
- **Without heading**: body X is treated as east, Y as north (heading = 0°).

### Properties (always present)

| Property | Type | Example | Description |
|---|---|---|---|
| `class` | string | `"Civilian vehicle"` | YOLO class name |
| `id` | string | `"42"` | Persistent track ID |
| `confidence` | float | `0.882` | YOLO confidence (0–1) |
| `category` | string | `"vehicle"` | Semantic category (see [§8](#8-category-mapping)) |
| `detection_type` | string | `"seg"` | Detection source model type |
| `source` | string | `"ugv"` | Platform identifier |
| `local_frame` | bool | `false` | `true` if no GPS origin available |
| `gnss_altitude_m` | float | `321.5` | Altitude above WGS-84 ellipsoid (m) |
| `height_m` | float | `4.5` | Object height from `bbox.size.z` (m) |
| `marker-color` | string | `"#0000ff"` | SimpleStyle hex colour |
| `marker-size` | string | `"medium"` | SimpleStyle marker size |
| `marker-symbol` | string | `"car"` | SimpleStyle Maki icon name |

### Properties (Polygon only, additional)

| Property | Type | Example | Description |
|---|---|---|---|
| `stroke` | string | `"#0000ff"` | Border colour (= `marker-color`) |
| `stroke-width` | int | `2` | Border width in pixels |
| `stroke-opacity` | float | `1.0` | Border opacity |
| `fill` | string | `"#0000ff"` | Fill colour (= `marker-color`) |
| `fill-opacity` | float | `0.25` | Fill opacity |

### `detection_type` Values

| Value | Meaning |
|---|---|
| `"seg"` | YOLOv11l-seg (instance segmentation model) |

Additional values (e.g. `"bbox"`) may be added in future versions for bbox-only models.

### MQTT Output

The `geojson_bridge` node also publishes every GeoJSON FeatureCollection to a local **MQTT** broker (Mosquitto, running inside the Docker container, started automatically by `run.sh start`).

| Property | Value |
|---|---|
| **Broker** | `localhost:1883` (default, configurable via `mqtt_host` / `mqtt_port` parameters) |
| **Topic** | `triffid/front/geojson` (configurable via `mqtt_topic` parameter) |
| **QoS** | 0 (at most once) |
| **Payload** | Compact JSON (no indentation) — identical FeatureCollection schema to the ROS 2 `/triffid/front/geojson` topic |
| **Enabled** | `true` by default; disable with `mqtt_enabled:=false` |
| **Client** | `paho-mqtt` ≥ 2.0 (CallbackAPIVersion.VERSION2) |

The MQTT output uses the `paho-mqtt` Python client. If the broker is unreachable at startup, MQTT is silently disabled and the node continues to publish on the ROS 2 topic.

To subscribe from any terminal on the same host:
```bash
mosquitto_sub -h localhost -t 'triffid/front/geojson'
# or from outside the container if using host network:
mosquitto_sub -h <robot-ip> -t 'triffid/front/geojson'
```

The `collect_samples.py` script also captures an MQTT trace during sampling, saving every received message to `mqtt_trace.jsonl` — one complete FeatureCollection JSON object per line (JSONL format), suitable for replaying or post-processing.

---

## 7. Class List (63 classes)

The YOLO model is fine-tuned on the TRIFFID disaster-response dataset. Class IDs are 0-based.

| ID | Class Name | Category |
|---|---|---|
| 0 | Water | obstacle |
| 1 | Fence | obstacle |
| 2 | Green tree | nature |
| 3 | Helmet | equipment |
| 4 | Flame | hazard |
| 5 | Smoke | hazard |
| 6 | First responder | person |
| 7 | Destroyed vehicle | vehicle |
| 8 | Fire hose | equipment |
| 9 | SCBA | equipment |
| 10 | Boot | equipment |
| 11 | Green plant | nature |
| 12 | Mask | equipment |
| 13 | Window | infrastructure |
| 14 | Building | infrastructure |
| 15 | Destroyed building | infrastructure |
| 16 | Debris | obstacle |
| 17 | Ladder | equipment |
| 18 | Dirt road | infrastructure |
| 19 | Dry tree | nature |
| 20 | Wall | infrastructure |
| 21 | Civilian vehicle | vehicle |
| 22 | Road | infrastructure |
| 23 | Citizen | person |
| 24 | Green grass | nature |
| 25 | Pole | infrastructure |
| 26 | Boat | vehicle |
| 27 | Pavement | infrastructure |
| 28 | Dry grass | nature |
| 29 | Animal | nature |
| 30 | Excavator | equipment |
| 31 | Door | infrastructure |
| 32 | Mud | obstacle |
| 33 | Barrier | obstacle |
| 34 | Hole in the ground | obstacle |
| 35 | Bag | equipment |
| 36 | Burnt tree | hazard |
| 37 | Ambulance | vehicle |
| 38 | Fire truck | vehicle |
| 39 | Cone | obstacle |
| 40 | Bicycle | vehicle |
| 41 | Tower | infrastructure |
| 42 | Silo | infrastructure |
| 43 | Military personnel | person |
| 44 | Burnt grass | hazard |
| 45 | Ax | equipment |
| 46 | Glove | equipment |
| 47 | Crane | equipment |
| 48 | Stairs | infrastructure |
| 49 | Dry plant | nature |
| 50 | Furniture | equipment |
| 51 | Tank | equipment |
| 52 | Protective glasses | equipment |
| 53 | Barrel | equipment |
| 54 | Shovel | equipment |
| 55 | Fire hydrant | equipment |
| 56 | Police vehicle | vehicle |
| 57 | Burnt plant | hazard |
| 58 | Army vehicle | vehicle |
| 59 | Chainsaw | equipment |
| 60 | aerial vehicle | vehicle |
| 61 | Lifesaver | equipment |
| 62 | Extinguisher | equipment |

---

## 8. Category Mapping

Each class is assigned to one of 7 semantic categories:

| Category | Classes |
|---|---|
| **hazard** | Flame, Smoke, Burnt tree, Burnt grass, Burnt plant |
| **person** | First responder, Citizen, Military personnel |
| **vehicle** | Civilian vehicle, Destroyed vehicle, Ambulance, Police vehicle, Fire truck, Army vehicle, Boat, Bicycle, aerial vehicle |
| **nature** | Green tree, Green plant, Green grass, Dry tree, Dry grass, Dry plant, Animal |
| **infrastructure** | Building, Destroyed building, Wall, Road, Pavement, Dirt road, Window, Door, Stairs, Pole, Tower, Silo |
| **obstacle** | Debris, Fence, Barrier, Cone, Hole in the ground, Mud, Water |
| **equipment** | Fire hose, Fire hydrant, Extinguisher, Helmet, SCBA, Boot, Mask, Glove, Protective glasses, Ladder, Ax, Shovel, Chainsaw, Bag, Barrel, Furniture, Tank, Crane, Excavator, Lifesaver |

Classes not in the table fall back to `"unknown"`.

---

## 9. Coordinate Conventions

### 3D Positions (`Detection3DArray`)

All positions are in the `b2/base_link` frame:

| Axis | Direction | Positive |
|---|---|---|
| X | Forward | Away from robot |
| Y | Lateral | Left |
| Z | Vertical | Up |

Units: **metres**.

### GPS Coordinates (GeoJSON)

- Order: `[longitude, latitude, altitude]` (RFC 7946 §3.1.1)
- Datum: WGS-84
- Projection: equirectangular approximation (adequate for <1 km range)
- Robot GPS position tracked continuously from `/fix` (median-filtered, window = 7)
- Heading from `/dog_odom` quaternion (magnetometer-fused, ENU yaw)
- Detection body-frame offsets rotated to ENU by robot yaw before GPS projection
- Origin seed available via `gps_origin_lat`/`gps_origin_lon`/`gps_origin_alt` parameters

### Segmentation Pixels

- Pixel value `0` = background
- Pixel value `N` (1–63) = class ID `N - 1` in `TARGET_CLASSES`

---

## 10. QoS Profiles

### Published Topics

| Topic | Reliability | Durability | History | Depth |
|---|---|---|---|---|
| `detections_3d` | RELIABLE | VOLATILE | KEEP_LAST | 10 |
| `segmentation` | RELIABLE | VOLATILE | KEEP_LAST | 10 |
| `geojson` | RELIABLE | VOLATILE | KEEP_LAST | 10 |

### Subscribed Topics

| Topic | Reliability | Durability | History | Depth |
|---|---|---|---|---|
| Camera images | BEST_EFFORT | VOLATILE | KEEP_LAST | 5 |
| CameraInfo | BEST_EFFORT | VOLATILE | KEEP_LAST | 5 |
| `/fix` (GPS) | BEST_EFFORT | VOLATILE | KEEP_LAST | 10 |
| `/dog_odom` | RELIABLE | VOLATILE | KEEP_LAST | 5 |
| `/tf_static` | RELIABLE | TRANSIENT_LOCAL | KEEP_ALL | — |

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-03-01 | Initial frozen interface |
| 1.1 | 2026-03-02 | 3D coordinates, heading rotation, GPS filtering, `/dog_odom` subscription |
| 1.2 | 2026-03-04 | GPS gating (no publish until fix received), polygon emitted when *either* bbox dimension > 0 (was: both), 0.3 m minimum extent for zero-dimension polygons, merged GeoJSON sample output |
| 1.3 | 2026-03-04 | MQTT output: GeoJSON published to local Mosquitto broker (`triffid/front/geojson`, QoS 0), MQTT trace capture in sample collector |

---

## 11. Current Limitations

1. **Equirectangular projection**: GPS coordinate conversion uses a flat-earth approximation. Accurate to ~1 m for ranges < 1 km; use UTM for larger-scale deployments.
2. **`bbox.size` may be zero**: When depth evidence is insufficient, the 3D bbox size degrades to `(0, 0, 0)` and a Point geometry is emitted instead of a Polygon. If only one horizontal dimension is zero, a minimum extent of 0.3 m is applied so a Polygon is still emitted.
3. **Single front camera only**: Current pipeline processes only the front-facing RealSense D435. Rear/side cameras are not integrated yet.
4. **2D segmentation mask**: The segmentation output is a 2D mono8 label map. True 3D volumetric segmentation is not performed.
5. **GPU required**: YOLO inference requires an NVIDIA GPU with CUDA support.
6. **No heading → raw axis mapping**: Without `/dog_odom`, body-frame X is treated as East and Y as North (yaw = 0°). Detections will be misaligned on the map.
