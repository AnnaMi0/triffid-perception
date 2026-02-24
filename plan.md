# TRIFFID Perception Integration – Setup Guide

Author: <your name>
Platform: ROS2 Humble
Workspace: `/home/triffid/hua_ws`

---

# 0. General Rules (From Partners)

* All code must live in:

  ```
  /home/triffid/hua_ws
  ```
* Use Git (backup everything)
* Prefer Docker if dependencies become heavy
* Do NOT modify ROS2 DDS configuration
* Do NOT upgrade NVIDIA drivers
* Do NOT install random system packages

---

# 1. Create Clean Workspace (Both UAV & UGV)

On GS or UGV machine:

```bash
mkdir -p /home/triffid/hua_ws/src
cd /home/triffid/hua_ws
colcon build
source install/setup.bash
```

Add to bashrc:

```bash
echo "source /home/triffid/hua_ws/install/setup.bash" >> ~/.bashrc
```

---

# =========================

# UAV PIPELINE (RGB ONLY)

# =========================

## UAV Inputs

From UAV:

* RGB image
* UAV global coordinates (lat, lon, alt)
* Camera tilt relative to UAV body
* Timestamp

---

## UAV Required Outputs

For each object:

* 2D coordinates (pixel OR projected geo)
* Class
* Confidence
* Persistent ID

---

## UAV Step-by-Step Implementation

---

## Step 1 — Inspect UAV Rosbag

On GS:

```bash
ros2 bag info <uav_bag>
ros2 bag play <uav_bag>
ros2 topic list
```

Identify:

* RGB topic
* GPS topic
* Tilt/orientation topic

Freeze these topic names.

---

## Step 2 — Create UAV Perception Package

```bash
cd /home/triffid/partnername_ws/src
ros2 pkg create --build-type ament_python triffid_uav_perception
```

---

## Step 3 — Minimal UAV Node Structure

Subscribes:

* RGB image
* UAV pose
* Camera tilt

Publishes:

```
/uav/perception/detections_2d
Type: vision_msgs/Detection2DArray
Frame: uav_camera_frame
```

Tracking requirement:

* ID must persist
* Never reuse IDs

Use:

* ByteTrack or DeepSORT for tracking

---

## Step 4 — Implement Core Logic

Pipeline:

```
RGB image
   ↓
Segmentation / Detection model
   ↓
Bounding boxes or masks
   ↓
Tracking layer
   ↓
Output Detection2DArray
```

Important:

```
output.header.stamp = input.header.stamp
output.header.frame_id = input.header.frame_id
```

---

## Step 5 — Geo Conversion (If Required by TRIFFID)

If UAV must send GeoJSON:

You need:

* UAV GPS
* UAV altitude
* Camera tilt
* Camera intrinsics

Then:

1. Pixel → camera ray
2. Ray intersect ground plane
3. Transform to world frame
4. Convert to lat/lon
5. Send GeoJSON

Do NOT approximate in production.

---

## Step 6 — Validate Offline

```bash
ros2 bag play <uav_bag>
ros2 run triffid_uav_perception uav_node
ros2 topic echo /uav/perception/detections_2d
```

Verify:

* Timestamps match
* Frame ID correct
* Stable output

---

# =========================

# UGV PIPELINE (RGB + DEPTH)

# =========================

## UGV Inputs

From UGV:

* RGB image (USB camera)
* Depth image (RealSense D430i)
* CameraInfo
* IMU
* TF tree (map, odom, base_link)

---

## UGV Required Outputs

For each object:

* 3D coordinates (relative to robot OR map — must confirm)
* Class
* Confidence
* Persistent ID (never reused)

---

# UGV Step-by-Step Implementation

---

## Step 1 — Inspect Rosbag

```bash
ros2 bag play <ugv_bag>
ros2 topic list
```

Identify:

* RGB topic
* Depth topic
* CameraInfo
* TF topics

Likely:

```
/camera_front/usb_cam/image_raw
/camera_front/realsense_front/depth/image_rect_raw
/camera_front/realsense_front/depth/camera_info
/tf
```

Freeze these.

---

## Step 2 — Create UGV Perception Package

```bash
cd /home/triffid/partnername_ws/src
ros2 pkg create --build-type ament_python triffid_ugv_perception
```

---

## Step 3 — Core UGV Pipeline

```
RGB image
Depth image
CameraInfo
TF transforms
        ↓
Detection / segmentation model
        ↓
Mask centroid pixel
        ↓
Get depth at pixel
        ↓
Back-project to camera frame
        ↓
Transform to map frame (TF2)
        ↓
Tracking layer
        ↓
Publish Detection3DArray
```

---

## Step 4 — 3D Back-Projection

From pixel (u,v) and depth Z:

```
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

Where fx, fy, cx, cy come from CameraInfo.

Now you have point in camera frame.

---

## Step 5 — Transform to Map Frame

Use TF2:

```
camera_frame → base_link → odom → map
```

Transform point to `map`.

Publish in:

```
frame_id = map
```

If mapping team builds semantic map → use `map`.

---

## Step 6 — Tracking

You must implement persistent ID logic.

Options:

* ByteTrack
* DeepSORT
* Custom IoU-based tracker

Rules:

* Never reuse IDs
* If object disappears → keep ID stored
* Do not reset counter on node restart unless documented

---

## Step 7 — Publish 3D Detections

Use:

```
vision_msgs/Detection3DArray
```

Each detection:

* ID
* Class
* Confidence
* geometry_msgs/Pose (position only)

---

## Step 8 — Validate Offline

```bash
ros2 bag play <ugv_bag>
ros2 run triffid_ugv_perception ugv_node
ros2 topic echo /ugv/perception/detections_3d
```

Verify:

* Timestamps copied
* frame_id = map
* Coordinates reasonable
* No drift when replayed

---

# REQUIRED VALIDATION CHECKLIST (FOR DEADLINE)

For BOTH UAV & UGV:

✔ Interface frozen
✔ Topic names fixed
✔ Message types fixed
✔ Timestamp rule documented
✔ Frame conventions documented
✔ Rosbag replay works
✔ No live hardware needed
✔ ID tracking works

---
