# TRIFFID UGV Perception — Interface Specification

> **Version**: 1.0  
> **Date**: 2026-03-01  
> **Status**: FROZEN — changes require version bump and partner notification  V
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
7. [Class List (63 classes)](#7-class-list-63-classes)
8. [Category Mapping](#8-category-mapping)
9. [Coordinate Conventions](#9-coordinate-conventions)
10. [QoS Profiles](#10-qos-profiles)

---

## 1. Published Topics (Output)

| Topic | Message Type | Encoding | Rate | Description |
|---|---|---|---|---|
| `/ugv/perception/front/detections_3d` | `vision_msgs/msg/Detection3DArray` | — | ~15 Hz | 3D bounding boxes in `b2/base_link` |
| `/ugv/perception/front/segmentation` | `sensor_msgs/msg/Image` | `mono8` | ~15 Hz | Semantic label map (pixel = class ID) |
| `/triffid/front/geojson` | `std_msgs/msg/String` | JSON (UTF-8) | ~15 Hz | GeoJSON FeatureCollection (RFC 7946) |

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
| `/fix` | `sensor_msgs/msg/NavSatFix` | — | ~1 Hz | Optional (GPS) |

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

**Point** (when `bbox.size.x == 0` or `bbox.size.y == 0`):
```json
{
  "type": "Point",
  "coordinates": [longitude, latitude]
}
```

**Polygon** (when `bbox.size.x > 0` and `bbox.size.y > 0`):
```json
{
  "type": "Polygon",
  "coordinates": [[ [lon,lat], [lon,lat], [lon,lat], [lon,lat], [lon,lat] ]]
}
```
The polygon is a closed axis-aligned rectangle (5 points, first = last) derived from the ground-plane projection of the 3D bounding box.

### Coordinates

- **With GPS** (`/fix` received or `gps_origin_lat/lon` set): WGS-84 `[longitude, latitude]` per RFC 7946.
- **Without GPS**: raw local metres `[x_metres, y_metres]` with `"local_frame": true`.

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

- Order: `[longitude, latitude]` (RFC 7946 §3.1.1)
- Datum: WGS-84
- Projection: equirectangular approximation from a local origin (adequate for <1 km range)
- Origin set from `/fix` topic or `gps_origin_lat`/`gps_origin_lon` parameters

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
| `/tf_static` | RELIABLE | TRANSIENT_LOCAL | KEEP_ALL | — |

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-03-01 | Initial frozen interface |
