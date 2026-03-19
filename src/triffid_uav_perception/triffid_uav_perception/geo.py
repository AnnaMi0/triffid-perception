"""
UAV Geo-Projection
===================
Projects 2D image pixel coordinates to WGS-84 GPS coordinates using:

  1. Camera intrinsics (pinhole model)
  2. Gimbal orientation (yaw/pitch/roll from DJI metadata — NED frame)
  3. Drone GPS position + altitude
  4. LRF distance (when available) as the reference depth

Two projection strategies:
  - **LRF-based** (preferred): Use the laser rangefinder distance as a depth
    estimate. The LRF points at the image centre; pixel offsets from centre
    are converted to angular offsets and projected at the same range.
  - **Flat-ground** (fallback): Assume a flat ground plane at the LRF target
    altitude (or a configurable ground elevation). Ray-cast each pixel down
    to that plane.

The DJI gimbal uses NED (North-East-Down) convention:
  - Yaw:   0° = North, 90° = East, clockwise positive
  - Pitch: 0° = horizontal, negative = looking down
  - Roll:  0° = level

M30T Wide Camera intrinsics (from DJI specs, approximate):
  - Sensor: 1/2" CMOS, 12MP (4000×3000 native, often recorded at 4000×3000 or 1920×1080)
  - FoV: 84° (diagonal) → ~70° horizontal at 4:3
  - For 4000×3000: fx ≈ fy ≈ 2850, cx = 2000, cy = 1500
  - For 1920×1080: fx ≈ fy ≈ 1370, cx = 960, cy = 540 (scaled proportionally)
  These are rough defaults; override with calibrated values when available.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

from triffid_uav_perception.metadata import DJIMetadata


# WGS-84 ellipsoid semi-major axis
_R_EARTH = 6378137.0


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# Default intrinsics for M30T WideCamera at common resolutions
_DEFAULT_INTRINSICS = {
    (4000, 3000): CameraIntrinsics(fx=2850, fy=2850, cx=2000, cy=1500,
                                    width=4000, height=3000),
    (1920, 1080): CameraIntrinsics(fx=1370, fy=1370, cx=960, cy=540,
                                    width=1920, height=1080),
}


def get_intrinsics(width: int, height: int,
                   override: Optional[CameraIntrinsics] = None) -> CameraIntrinsics:
    """Return camera intrinsics for the given image size.

    Uses calibrated override if provided, otherwise falls back to
    resolution-matched defaults, otherwise scales from 4000×3000.
    """
    if override is not None:
        return override

    key = (width, height)
    if key in _DEFAULT_INTRINSICS:
        return _DEFAULT_INTRINSICS[key]

    # Scale from the 4000×3000 baseline
    base = _DEFAULT_INTRINSICS[(4000, 3000)]
    sx = width / base.width
    sy = height / base.height
    return CameraIntrinsics(
        fx=base.fx * sx, fy=base.fy * sy,
        cx=base.cx * sx, cy=base.cy * sy,
        width=width, height=height,
    )


def _ned_rotation_matrix(yaw_deg: float, pitch_deg: float,
                         roll_deg: float) -> np.ndarray:
    """Build a 3×3 rotation matrix from NED gimbal angles.

    Converts a ray in camera frame to NED world frame.

    Camera frame convention (DJI optical):
      X = right, Y = down, Z = forward (out of lens)

    NED world frame:
      X = North, Y = East, Z = Down

    Rotation order: Yaw (Z_ned) → Pitch (Y_cam) → Roll (X_cam)
    This matches DJI's intrinsic ZYX Euler convention.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    # Yaw rotation about NED Z (Down) axis
    Rz = np.array([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y),  math.cos(y), 0],
        [0,            0,           1],
    ])

    # Pitch rotation about camera Y axis (in NED: East-ish)
    # DJI pitch: 0 = horizontal, negative = down
    Ry = np.array([
        [ math.cos(p), 0, math.sin(p)],
        [ 0,           1, 0          ],
        [-math.sin(p), 0, math.cos(p)],
    ])

    # Roll rotation about camera Z axis (forward / optical axis)
    Rx = np.array([
        [1, 0,            0           ],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r),  math.cos(r)],
    ])

    # Camera-to-NED: first roll, then pitch, then yaw
    # Camera Z (forward) at zero yaw/pitch/roll should point North horizontal
    # But the camera optical axis is Z-forward, X-right, Y-down
    # At yaw=0, pitch=0: camera looks North horizontally
    # We need to map camera frame → NED

    # Camera-to-NED base transform (no rotation applied):
    # Camera Z (forward) → NED X (North)
    # Camera X (right)   → NED Y (East)
    # Camera Y (down)    → NED Z (Down)
    C_cam_to_ned = np.array([
        [0, 0, 1],  # North = camera forward
        [1, 0, 0],  # East  = camera right
        [0, 1, 0],  # Down  = camera down
    ])

    return Rz @ Ry @ Rx @ C_cam_to_ned


def pixel_to_ray(u: float, v: float,
                 intrinsics: CameraIntrinsics) -> np.ndarray:
    """Convert a pixel coordinate to a unit ray in camera frame.

    Camera frame: X=right, Y=down, Z=forward.
    Returns a normalised 3D direction vector.
    """
    x = (u - intrinsics.cx) / intrinsics.fx
    y = (v - intrinsics.cy) / intrinsics.fy
    z = 1.0
    ray = np.array([x, y, z], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def project_pixel_to_ground(
    u: float, v: float,
    meta: DJIMetadata,
    intrinsics: CameraIntrinsics,
    ground_alt: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """Project a single pixel to GPS coordinates on the ground.

    Uses ray-casting: shoots a ray from the camera through the pixel,
    rotated by gimbal angles, and intersects it with a horizontal plane
    at ``ground_alt`` metres (ellipsoidal).

    If ``ground_alt`` is None, uses the LRF target altitude when available,
    otherwise uses drone altitude minus relative altitude as a rough ground
    estimate.

    Returns (lon, lat, alt) or None if the ray doesn't hit the ground
    (e.g. pointing above horizon).
    """
    # Determine ground plane altitude
    if ground_alt is None:
        if meta.lrf_valid:
            ground_alt = meta.lrf_target_abs_alt
        else:
            ground_alt = meta.abs_alt - meta.rel_alt

    # Camera ray in camera frame
    ray_cam = pixel_to_ray(u, v, intrinsics)

    # Rotate to NED world frame
    R = _ned_rotation_matrix(meta.gimbal_yaw, meta.gimbal_pitch,
                             meta.gimbal_roll)
    ray_ned = R @ ray_cam  # (North, East, Down)

    # The drone is at height (abs_alt - ground_alt) above the ground plane
    h_above_ground = meta.abs_alt - ground_alt
    if h_above_ground < 0.1:
        return None  # drone is at or below ground — no valid projection

    # Ray must be pointing downward (positive Down component)
    if ray_ned[2] <= 1e-6:
        return None  # ray points above horizon, no ground intersection

    # Parameter t where ray hits ground: drone_pos + t * ray_ned, Z component = 0
    # In NED, drone is at Z=0 (its own altitude), ground is at Z = h_above_ground
    t = h_above_ground / ray_ned[2]

    # Ground intersection in NED metres relative to drone
    north = t * ray_ned[0]
    east = t * ray_ned[1]

    # Convert NED offset to GPS
    lat, lon = _ned_offset_to_gps(meta.lat, meta.lon, north, east)

    return (lon, lat, ground_alt)


def project_mask_to_ground(
    mask: np.ndarray,
    meta: DJIMetadata,
    intrinsics: CameraIntrinsics,
    ground_alt: Optional[float] = None,
    sample_step: int = 8,
) -> Optional[Tuple[list, Tuple[float, float, float]]]:
    """Project a binary segmentation mask to a ground polygon.

    Instead of projecting every mask pixel (expensive), we:
      1. Find the mask contour
      2. Sample contour points at ``sample_step`` intervals
      3. Project each sampled contour point to ground GPS
      4. Also project the mask centroid for the centre coordinate

    Returns (polygon_coords, centre_coord) where:
      - polygon_coords: list of [lon, lat] pairs (closed ring)
      - centre_coord: (lon, lat, alt) of the mask centroid
    Or None if projection fails.
    """
    import cv2

    # Find contours of the mask
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 10:
        return None  # too small to be meaningful

    # Sample contour points
    contour_pts = contour.squeeze()
    if contour_pts.ndim == 1:
        return None  # degenerate contour (single point)

    # Adapt sample step so we always get at least 4 points
    n_pts = len(contour_pts)
    effective_step = min(sample_step, max(1, n_pts // 4))
    indices = range(0, n_pts, max(1, effective_step))
    sampled = contour_pts[list(indices)]

    # Project sampled contour points to ground
    ring = []
    for pt in sampled:
        result = project_pixel_to_ground(
            float(pt[0]), float(pt[1]), meta, intrinsics, ground_alt,
        )
        if result is not None:
            ring.append([result[0], result[1]])  # [lon, lat]

    if len(ring) < 3:
        return None  # need at least 3 points for a polygon

    # Close the ring (RFC 7946)
    if ring[0] != ring[-1]:
        ring.append(ring[0])

    # Project mask centroid
    M = cv2.moments(mask_u8)
    if M['m00'] == 0:
        return None
    cu = M['m10'] / M['m00']
    cv_y = M['m01'] / M['m00']
    centre = project_pixel_to_ground(cu, cv_y, meta, intrinsics, ground_alt)
    if centre is None:
        return None

    return (ring, centre)


def project_bbox_to_ground(
    x1: float, y1: float, x2: float, y2: float,
    meta: DJIMetadata,
    intrinsics: CameraIntrinsics,
    ground_alt: Optional[float] = None,
) -> Optional[Tuple[list, Tuple[float, float, float]]]:
    """Project a 2D bounding box to a ground polygon.

    Simpler fallback when no segmentation mask is available.
    Projects the 4 bbox corners + centre.

    Returns (polygon_coords, centre_coord) or None.
    """
    corners_px = [
        (x1, y1), (x2, y1), (x2, y2), (x1, y2),
    ]
    ring = []
    for u, v in corners_px:
        result = project_pixel_to_ground(u, v, meta, intrinsics, ground_alt)
        if result is not None:
            ring.append([result[0], result[1]])

    if len(ring) < 3:
        return None

    ring.append(ring[0])  # close

    # Centre
    cu = (x1 + x2) / 2.0
    cv = (y1 + y2) / 2.0
    centre = project_pixel_to_ground(cu, cv, meta, intrinsics, ground_alt)
    if centre is None:
        return None

    return (ring, centre)


def estimate_object_height(
    mask: np.ndarray,
    meta: DJIMetadata,
    intrinsics: CameraIntrinsics,
    ground_alt: Optional[float] = None,
) -> float:
    """Rough estimate of object height from mask vertical extent.

    Uses the angular extent of the mask in the vertical direction
    combined with the LRF distance to estimate physical height.
    This is approximate — good enough for PoC.
    """
    if ground_alt is None:
        if meta.lrf_valid:
            ground_alt = meta.lrf_target_abs_alt
        else:
            ground_alt = meta.abs_alt - meta.rel_alt

    # Get vertical pixel span of the mask
    rows = np.any(mask, axis=1)
    if not np.any(rows):
        return 0.0
    row_indices = np.where(rows)[0]
    v_top = float(row_indices[0])
    v_bot = float(row_indices[-1])

    # Project top and bottom of the mask's vertical extent at the centroid column
    cols = np.any(mask, axis=0)
    col_indices = np.where(cols)[0]
    u_mid = float(col_indices[len(col_indices) // 2])

    top = project_pixel_to_ground(u_mid, v_top, meta, intrinsics, ground_alt)
    bot = project_pixel_to_ground(u_mid, v_bot, meta, intrinsics, ground_alt)

    if top is None or bot is None:
        # Fallback: use angular extent and distance
        if meta.lrf_valid:
            angle_span = abs(v_bot - v_top) / intrinsics.fy  # radians approx
            return meta.lrf_distance * angle_span
        return 0.0

    # Horizontal distance between top and bottom projections gives
    # a proxy for object height (works well for near-nadir views)
    d_north = _gps_to_ned_offset(top[1], top[0], bot[1], bot[0])
    return math.sqrt(d_north[0]**2 + d_north[1]**2)


# ── Coordinate helpers ──────────────────────────────────────────────

def _ned_offset_to_gps(lat_ref: float, lon_ref: float,
                       north_m: float, east_m: float) -> Tuple[float, float]:
    """Convert NED metre offset to WGS-84 lat/lon.

    Uses equirectangular approximation (accurate to ~1m within 1km).
    """
    d_lat = north_m / _R_EARTH * (180.0 / math.pi)
    d_lon = east_m / (_R_EARTH * math.cos(math.radians(lat_ref))) * (180.0 / math.pi)
    return (lat_ref + d_lat, lon_ref + d_lon)


def _gps_to_ned_offset(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> Tuple[float, float]:
    """Inverse of _ned_offset_to_gps: GPS pair → (north_m, east_m)."""
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    north = d_lat * _R_EARTH
    east = d_lon * _R_EARTH * math.cos(math.radians((lat1 + lat2) / 2))
    return (north, east)
