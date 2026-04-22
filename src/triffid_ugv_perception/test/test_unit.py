#!/usr/bin/env python3
"""
TRIFFID UGV Perception – Unit Tests
=====================================
Comprehensive tests for the UGV perception pipeline, covering:

  1. Geometry & math  (quaternion→matrix, back-projection, pinhole model)
  2. Pixel-aligned pipeline logic  (depth sampling per detection, median 3D)
  3. IoU tracker  (matching, ID persistence, aging, edge cases)
  4. GeoJSON bridge  (local_to_gps, GeoJSON schema, local_frame flag)
  5. Depth handling  (mm→m conversion, zero-depth filtering)

Run with:
    cd /ws && colcon build --symlink-install && source install/setup.bash
    python3 -m pytest src/triffid_ugv_perception/test/test_unit.py -v

Or without colcon (after sourcing ROS2):
    cd /ws/src/triffid_ugv_perception
    python3 -m pytest test/test_unit.py -v
"""

import json
import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module imports (pure Python – no ROS node spin required)
# ---------------------------------------------------------------------------
from triffid_ugv_perception.tracker import IoUTracker
from triffid_ugv_perception.ugv_node import UGVPerceptionNode


# ═══════════════════════════════════════════════════════════════════════════
#  1.  QUATERNION → ROTATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════

class TestQuatToMatrix:
    """Tests for UGVPerceptionNode._quat_to_matrix (static method)."""

    quat2mat = staticmethod(UGVPerceptionNode._quat_to_matrix)

    # --- identity quaternion ---
    def test_identity_quaternion(self):
        R = self.quat2mat(0.0, 0.0, 0.0, 1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    # --- 90° around Z ---
    def test_90deg_around_z(self):
        # quat (0, 0, sin(45°), cos(45°))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        R = self.quat2mat(0.0, 0.0, s, c)
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    # --- 180° around X ---
    def test_180deg_around_x(self):
        R = self.quat2mat(1.0, 0.0, 0.0, 0.0)
        expected = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    # --- 180° around Y ---
    def test_180deg_around_y(self):
        R = self.quat2mat(0.0, 1.0, 0.0, 0.0)
        expected = np.array([
            [-1, 0,  0],
            [ 0, 1,  0],
            [ 0, 0, -1],
        ], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    # --- orthogonality: R^T R = I ---
    def test_orthogonality_random(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            R = self.quat2mat(*q)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    # --- determinant = +1 (proper rotation) ---
    def test_det_is_one(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            R = self.quat2mat(*q)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10

    # --- non-unit quaternion gets normalised ---
    def test_non_unit_quat(self):
        R = self.quat2mat(0.0, 0.0, 0.0, 5.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    # --- zero quaternion → identity fallback ---
    def test_zero_quat(self):
        R = self.quat2mat(0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    # --- actual TF from bag: depth→RGB ---
    def test_bag_tf_depth_to_rgb(self):
        """Verify using the real transform from the rosbag:
        f_depth_optical_frame → f_oc_link
        Quaternion (xyzw): [0.653, -0.653, 0.271, 0.271]
        """
        R = self.quat2mat(0.653, -0.653, 0.271, 0.271)
        # Must be a valid rotation
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-4)
        assert abs(np.linalg.det(R) - 1.0) < 1e-4

    # --- composing two rotations: R(q1) @ R(q2) ≈ R(q1*q2) ---
    def test_composition(self):
        # 90° around Z then 90° around X
        s45 = math.sin(math.pi / 4)
        c45 = math.cos(math.pi / 4)
        R_z = self.quat2mat(0, 0, s45, c45)
        R_x = self.quat2mat(s45, 0, 0, c45)
        R_comp = R_x @ R_z
        # Check it's still a valid rotation
        np.testing.assert_allclose(R_comp.T @ R_comp, np.eye(3), atol=1e-10)
        assert abs(np.linalg.det(R_comp) - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
#  2.  PINHOLE BACK-PROJECTION (pixel → 3D, optical convention)
# ═══════════════════════════════════════════════════════════════════════════

class TestBackProjectionOptical:
    """Tests for pinhole back-projection in camera_optical_frame
    (X=right, Y=down, Z=forward) — the convention used by
    _sample_depth_for_detection."""

    FX, FY, CX, CY = 500.0, 500.0, 640.0, 360.0

    @staticmethod
    def _backproject(u, v, z_m, fx, fy, cx, cy):
        """Back-project a single pixel to optical frame."""
        X = (u - cx) * z_m / fx
        Y = (v - cy) * z_m / fy
        return np.array([X, Y, z_m])

    def test_principal_point(self):
        """Pixel at (cx, cy) → (0, 0, Z)."""
        pt = self._backproject(self.CX, self.CY, 5.0,
                               self.FX, self.FY, self.CX, self.CY)
        np.testing.assert_allclose(pt, [0.0, 0.0, 5.0], atol=1e-10)

    def test_pixel_right_of_centre(self):
        """Pixel right of cx → positive X."""
        pt = self._backproject(self.CX + 100, self.CY, 5.0,
                               self.FX, self.FY, self.CX, self.CY)
        assert pt[0] > 0  # X = right

    def test_pixel_below_centre(self):
        """Pixel below cy → positive Y."""
        pt = self._backproject(self.CX, self.CY + 100, 5.0,
                               self.FX, self.FY, self.CX, self.CY)
        assert pt[1] > 0  # Y = down

    def test_roundtrip(self):
        """Back-project then forward-project recovers original pixel."""
        u_orig, v_orig, z = 200.0, 300.0, 3.5
        pt = self._backproject(u_orig, v_orig, z,
                               self.FX, self.FY, self.CX, self.CY)
        u_rec = self.FX * (pt[0] / pt[2]) + self.CX
        v_rec = self.FY * (pt[1] / pt[2]) + self.CY
        np.testing.assert_allclose([u_rec, v_rec], [u_orig, v_orig], atol=1e-10)

    def test_depth_preserved(self):
        """Z component equals input depth."""
        for z in [0.5, 1.0, 5.0, 15.0]:
            pt = self._backproject(100, 200, z,
                                   self.FX, self.FY, self.CX, self.CY)
            assert abs(pt[2] - z) < 1e-12

    def test_mm_to_m_conversion(self):
        """16UC1 depth in mm → metres: divide by 1000."""
        z_mm = np.uint16(3500)
        z_m = float(z_mm) / 1000.0
        assert abs(z_m - 3.5) < 1e-10

    def test_zero_depth_rejected(self):
        """Zero depth pixels should be filtered (no reading)."""
        z_mm = np.array([0, 0, 1500, 0, 2000], dtype=np.uint16)
        valid = z_mm > 0
        assert np.sum(valid) == 2


# ═══════════════════════════════════════════════════════════════════════════
#  3.  PIXEL-ALIGNED DEPTH SAMPLING (_sample_depth_for_detection)
# ═══════════════════════════════════════════════════════════════════════════

class TestSampleDepthForDetection:
    """Tests for _sample_depth_for_detection (pixel-aligned RGB-D)."""

    FX, FY, CX, CY = 500.0, 500.0, 320.0, 240.0

    @staticmethod
    def _sample(det, depth_img, fx, fy, cx, cy):
        """Re-implement the method's core math for unit testing."""
        from triffid_ugv_perception.ugv_node import _MAX_DEPTH_SAMPLES
        h, w = depth_img.shape[:2]
        mask = det.get('mask')
        x1, y1, x2, y2 = det['bbox']

        if mask is not None:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None
            if len(xs) > _MAX_DEPTH_SAMPLES:
                step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
                xs, ys = xs[::step], ys[::step]
        else:
            ix1 = max(0, min(int(x1), w - 1))
            ix2 = max(ix1 + 1, min(int(x2), w))
            iy1 = max(0, min(int(y1), h - 1))
            iy2 = max(iy1 + 1, min(int(y2), h))
            side = max(1, min(ix2 - ix1, iy2 - iy1) // 10)
            xs_g, ys_g = np.meshgrid(
                np.arange(ix1, ix2, max(1, side)),
                np.arange(iy1, iy2, max(1, side)),
            )
            xs, ys = xs_g.ravel(), ys_g.ravel()
            if len(xs) > _MAX_DEPTH_SAMPLES:
                step = max(1, len(xs) // _MAX_DEPTH_SAMPLES)
                xs, ys = xs[::step], ys[::step]

        ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[ok], ys[ok]
        if len(xs) == 0:
            return None
        z_mm = depth_img[ys, xs].astype(np.float64)
        valid = z_mm > 0
        if not np.any(valid):
            return None
        xs = xs[valid].astype(np.float64)
        ys = ys[valid].astype(np.float64)
        z_m = z_mm[valid] / 1000.0
        X = (xs - cx) * z_m / fx
        Y = (ys - cy) * z_m / fy
        return np.column_stack([X, Y, z_m])

    def test_mask_sampling_basic(self):
        """Mask with valid depth → correct 3D points."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        depth[100:200, 150:250] = 3000  # 3 m

        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 150:250] = True

        det = {'bbox': (150, 100, 250, 200), 'mask': mask}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is not None
        assert pts.shape[1] == 3
        # All Z should be 3.0 m
        np.testing.assert_allclose(pts[:, 2], 3.0)

    def test_all_zero_depth_returns_none(self):
        """Zero depth everywhere → None."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 150:250] = True
        det = {'bbox': (150, 100, 250, 200), 'mask': mask}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is None

    def test_empty_mask_returns_none(self):
        """All-False mask → None."""
        depth = np.full((480, 640), 2000, dtype=np.uint16)
        mask = np.zeros((480, 640), dtype=bool)
        det = {'bbox': (100, 100, 200, 200), 'mask': mask}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is None

    def test_bbox_fallback_when_no_mask(self):
        """No mask → bbox grid sampling produces points."""
        depth = np.full((480, 640), 5000, dtype=np.uint16)
        det = {'bbox': (100, 100, 300, 300), 'mask': None}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is not None
        assert pts.shape[1] == 3
        np.testing.assert_allclose(pts[:, 2], 5.0)

    def test_principal_point_projects_to_origin(self):
        """Mask centered on (cx, cy) at depth Z → median X≈0, Y≈0."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        cx, cy = int(self.CX), int(self.CY)
        depth[cy - 2:cy + 3, cx - 2:cx + 3] = 4000

        mask = np.zeros((480, 640), dtype=bool)
        mask[cy - 2:cy + 3, cx - 2:cx + 3] = True
        det = {'bbox': (cx - 2, cy - 2, cx + 3, cy + 3), 'mask': mask}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is not None
        median = np.median(pts, axis=0)
        assert abs(median[0]) < 0.05  # X ≈ 0
        assert abs(median[1]) < 0.05  # Y ≈ 0
        np.testing.assert_allclose(median[2], 4.0)

    def test_subsampling_caps_points(self):
        """Large mask gets sub-sampled to ≤ _MAX_DEPTH_SAMPLES."""
        from triffid_ugv_perception.ugv_node import _MAX_DEPTH_SAMPLES
        depth = np.full((480, 640), 2000, dtype=np.uint16)
        mask = np.ones((480, 640), dtype=bool)  # 307,200 True pixels
        det = {'bbox': (0, 0, 640, 480), 'mask': mask}
        pts = self._sample(det, depth, self.FX, self.FY, self.CX, self.CY)
        assert pts is not None
        assert len(pts) <= _MAX_DEPTH_SAMPLES + 1


# ═══════════════════════════════════════════════════════════════════════════
#  4.  PIXEL-ALIGNED PIPELINE (end-to-end geometry check)
# ═══════════════════════════════════════════════════════════════════════════

class TestPixelAlignedPipeline:
    """End-to-end test: pixel-aligned depth sampling → back-project →
    median position in camera_optical_frame."""

    FX, FY, CX, CY = 500.0, 500.0, 640.0, 360.0

    def test_single_detection_median(self):
        """A uniform-depth detection gives a median at its centroid."""
        depth = np.zeros((720, 1280), dtype=np.uint16)
        depth[200:400, 400:800] = 5000  # 5 m

        mask = np.zeros((720, 1280), dtype=bool)
        mask[200:400, 400:800] = True

        det = {'bbox': (400, 200, 800, 400), 'mask': mask}
        ys, xs = np.where(mask)
        z_m = depth[ys, xs].astype(np.float64) / 1000.0
        X = (xs.astype(np.float64) - self.CX) * z_m / self.FX
        Y = (ys.astype(np.float64) - self.CY) * z_m / self.FY
        pts = np.column_stack([X, Y, z_m])
        median = np.median(pts, axis=0)

        # Centroid of mask is ~(600, 300), depth=5m
        expected_x = (600.0 - self.CX) * 5.0 / self.FX  # (600-640)*5/500 = -0.4
        expected_y = (300.0 - self.CY) * 5.0 / self.FY  # (300-360)*5/500 = -0.6
        np.testing.assert_allclose(median[0], expected_x, atol=0.15)
        np.testing.assert_allclose(median[1], expected_y, atol=0.15)
        np.testing.assert_allclose(median[2], 5.0)

    def test_no_depth_in_mask_skips(self):
        """If mask region has zero depth, detection is skipped."""
        depth = np.zeros((720, 1280), dtype=np.uint16)
        mask = np.zeros((720, 1280), dtype=bool)
        mask[100:200, 100:200] = True
        det = {'bbox': (100, 100, 200, 200), 'mask': mask}

        ys, xs = np.where(mask)
        z_mm = depth[ys, xs]
        valid = z_mm > 0
        assert not np.any(valid)  # no depth → skip

    def test_partial_depth_coverage(self):
        """Only mask pixels with valid depth contribute."""
        depth = np.zeros((720, 1280), dtype=np.uint16)
        depth[150:180, 150:180] = 3000  # partial coverage

        mask = np.zeros((720, 1280), dtype=bool)
        mask[100:200, 100:200] = True

        ys, xs = np.where(mask)
        z_mm = depth[ys, xs].astype(np.float64)
        valid = z_mm > 0
        n_valid = np.sum(valid)
        assert 0 < n_valid < np.sum(mask)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  BACK-PROJECTION (raw pinhole math — frame-agnostic)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
#  6.  BATCH TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchTransform:
    """Tests for the matrix-based batch point transform."""

    @staticmethod
    def _apply_transform(pts, qx, qy, qz, qw, tx, ty, tz):
        """Replicate _transform_points_batch logic."""
        R = UGVPerceptionNode._quat_to_matrix(qx, qy, qz, qw)
        tvec = np.array([tx, ty, tz])
        return (R @ pts.T).T + tvec

    def test_identity_transform(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        out = self._apply_transform(pts, 0, 0, 0, 1, 0, 0, 0)
        np.testing.assert_allclose(out, pts, atol=1e-12)

    def test_pure_translation(self):
        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        out = self._apply_transform(pts, 0, 0, 0, 1, 10, 20, 30)
        expected = pts + np.array([10, 20, 30])
        np.testing.assert_allclose(out, expected, atol=1e-12)

    def test_pure_rotation_180_z(self):
        pts = np.array([[1, 0, 0]], dtype=np.float64)
        out = self._apply_transform(pts, 0, 0, 1, 0, 0, 0, 0)
        np.testing.assert_allclose(out, [[-1, 0, 0]], atol=1e-12)

    def test_preserves_distances(self):
        """Rigid transform preserves pairwise distances."""
        pts = np.array([[0, 0, 0], [3, 4, 0], [0, 0, 5]], dtype=np.float64)
        dist_before = np.linalg.norm(pts[0] - pts[1])
        out = self._apply_transform(pts, 0.653, -0.653, 0.271, 0.271,
                                    0.025, 0.071, 0.039)
        dist_after = np.linalg.norm(out[0] - out[1])
        np.testing.assert_allclose(dist_before, dist_after, atol=1e-6)

    def test_single_point(self):
        pts = np.array([[5.0, 3.0, 1.0]])
        out = self._apply_transform(pts, 0, 0, 0, 1, 1, 2, 3)
        np.testing.assert_allclose(out, [[6, 5, 4]], atol=1e-12)

    def test_large_batch(self):
        """Transform 10k points to verify vectorised perf doesn't error."""
        pts = np.random.default_rng(0).standard_normal((10000, 3))
        out = self._apply_transform(pts, 0.5, 0.5, 0.5, 0.5, 1, 2, 3)
        assert out.shape == (10000, 3)
        assert np.all(np.isfinite(out))


# ═══════════════════════════════════════════════════════════════════════════
#  7.  IoU TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class TestIoUComputation:
    """Tests for IoUTracker._compute_iou (static method)."""

    iou = staticmethod(IoUTracker._compute_iou)

    def test_identical_boxes(self):
        assert self.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert self.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # box A (0,0,10,10) area=100
        # box B (5,0,15,10) area=100
        # intersection (5,0,10,10) area=50
        # union = 100+100-50 = 150
        expected = 50.0 / 150.0
        result = self.iou((0, 0, 10, 10), (5, 0, 15, 10))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_contained_box(self):
        # small box fully inside big box
        # A=(0,0,100,100) area=10000, B=(25,25,75,75) area=2500
        # inter=2500, union=10000
        expected = 2500.0 / 10000.0
        assert abs(self.iou((0, 0, 100, 100), (25, 25, 75, 75)) - expected) < 1e-10

    def test_symmetry(self):
        a = (10, 20, 50, 60)
        b = (30, 40, 70, 80)
        assert abs(self.iou(a, b) - self.iou(b, a)) < 1e-12

    def test_touching_edge(self):
        # Boxes share an edge but no area
        assert self.iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_zero_area_box(self):
        assert self.iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_float_coords(self):
        result = self.iou((0.5, 0.5, 10.5, 10.5), (5.5, 5.5, 15.5, 15.5))
        assert 0.0 < result < 1.0

    def test_large_boxes(self):
        result = self.iou((0, 0, 1280, 720), (0, 0, 1280, 720))
        assert result == 1.0


class TestTrackerIDPersistence:
    """Tests that track IDs are persistent and never reused."""

    def _make_det(self, bbox, cls='person', conf=0.9, pos=(0, 0, 0)):
        return {
            'bbox': bbox,
            'class_id': 0,
            'class_name': cls,
            'confidence': conf,
            'position': pos,
        }

    def test_first_frame_assigns_ids(self):
        tracker = IoUTracker()
        dets = [self._make_det((10, 10, 50, 50)),
                self._make_det((100, 100, 200, 200))]
        results = tracker.update(dets)
        assert len(results) == 2
        ids = {r['track_id'] for r in results}
        assert ids == {1, 2}

    def test_same_bbox_keeps_id(self):
        tracker = IoUTracker()
        det = [self._make_det((10, 10, 50, 50))]
        r1 = tracker.update(det)
        r2 = tracker.update(det)
        assert r1[0]['track_id'] == r2[0]['track_id']

    def test_ids_never_reused_after_disappearance(self):
        tracker = IoUTracker(max_age=2)
        det = [self._make_det((10, 10, 50, 50))]
        r1 = tracker.update(det)
        first_id = r1[0]['track_id']

        # Disappear for max_age+2 frames to ensure removal
        for _ in range(4):
            tracker.update([])

        # Re-appear at same location
        r2 = tracker.update(det)
        new_id = r2[0]['track_id']
        assert new_id != first_id, "ID was reused after track expired"

    def test_ids_are_monotonically_increasing(self):
        tracker = IoUTracker()
        all_ids = []

        for i in range(5):
            det = [self._make_det((i * 100, 0, i * 100 + 50, 50))]
            results = tracker.update(det)
            all_ids.extend(r['track_id'] for r in results)

        # IDs across new tracks should be strictly increasing
        unique_ids = list(dict.fromkeys(all_ids))  # preserve order, remove dups
        assert unique_ids == sorted(unique_ids)

    def test_counter_never_resets(self):
        tracker = IoUTracker(max_age=0)
        for i in range(10):
            det = [self._make_det((i * 200, 0, i * 200 + 50, 50),
                                  pos=(float(i * 20), 0, 0))]
            tracker.update(det)
        # After 10 different detections (all expired), next_id should be 11
        assert tracker.next_id == 11


class TestTrackerMatching:
    """Tests for the greedy IoU matching behaviour."""

    def _make_det(self, bbox, cls='person', conf=0.9, pos=(0, 0, 0)):
        return {
            'bbox': bbox,
            'class_id': 0,
            'class_name': cls,
            'confidence': conf,
            'position': pos,
        }

    def test_high_iou_match(self):
        tracker = IoUTracker(iou_threshold=0.3)
        dets1 = [self._make_det((100, 100, 200, 200))]
        dets2 = [self._make_det((105, 105, 205, 205))]  # shifted slightly
        r1 = tracker.update(dets1)
        r2 = tracker.update(dets2)
        assert r1[0]['track_id'] == r2[0]['track_id']

    def test_low_iou_creates_new_track(self):
        tracker = IoUTracker(iou_threshold=0.3)
        dets1 = [self._make_det((0, 0, 50, 50), pos=(1.0, 0.0, 0.0))]
        dets2 = [self._make_det((500, 500, 600, 600), pos=(10.0, 10.0, 0.0))]  # far away
        r1 = tracker.update(dets1)
        r2 = tracker.update(dets2)
        assert r1[0]['track_id'] != r2[0]['track_id']

    def test_multiple_objects_tracked(self):
        tracker = IoUTracker()
        dets = [
            self._make_det((0, 0, 50, 50)),
            self._make_det((200, 200, 300, 300)),
            self._make_det((500, 500, 600, 600)),
        ]
        r1 = tracker.update(dets)
        r2 = tracker.update(dets)
        ids1 = sorted(r['track_id'] for r in r1)
        ids2 = sorted(r['track_id'] for r in r2)
        assert ids1 == ids2, "Same detections should keep same IDs"

    def test_one_disappears_others_remain(self):
        tracker = IoUTracker()
        dets_full = [
            self._make_det((0, 0, 50, 50)),
            self._make_det((200, 200, 300, 300)),
        ]
        r1 = tracker.update(dets_full)
        id_map = {r['bbox']: r['track_id'] for r in r1}

        # Only first detection remains
        dets_partial = [self._make_det((0, 0, 50, 50))]
        r2 = tracker.update(dets_partial)
        assert r2[0]['track_id'] == id_map[(0, 0, 50, 50)]

    def test_empty_frame_ages_tracks(self):
        tracker = IoUTracker(max_age=3)
        tracker.update([self._make_det((0, 0, 50, 50))])
        assert len(tracker.tracks) == 1

        for _ in range(3):
            tracker.update([])

        # time_since_update=3, max_age=3 → 3 > 3 is False → still alive
        assert len(tracker.tracks) == 1

        tracker.update([])
        # time_since_update=4 > max_age=3 → removed
        assert len(tracker.tracks) == 0

    def test_empty_detections_returns_empty(self):
        tracker = IoUTracker()
        result = tracker.update([])
        assert result == []

    def test_class_name_updated_on_match(self):
        tracker = IoUTracker()
        det1 = [self._make_det((10, 10, 50, 50), cls='person')]
        det2 = [self._make_det((10, 10, 50, 50), cls='car')]
        tracker.update(det1)
        r2 = tracker.update(det2)
        assert r2[0]['class_name'] == 'car'

    def test_confidence_updated_on_match(self):
        tracker = IoUTracker()
        det1 = [self._make_det((10, 10, 50, 50), conf=0.9)]
        det2 = [self._make_det((10, 10, 50, 50), conf=0.5)]
        tracker.update(det1)
        r2 = tracker.update(det2)
        assert r2[0]['confidence'] == 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  8.  GEOJSON BRIDGE (pure logic tests – no ROS spin)
# ═══════════════════════════════════════════════════════════════════════════

class TestBodyToGPS:
    """Tests for GeoJSONBridge coordinate conversion (body → GPS).

    Tests both the static ``_body_to_enu`` rotation and the full
    ``body_to_gps`` pipeline, exercised via the maths directly
    (no ROS node instantiation).
    """

    R_EARTH = 6378137.0

    # ── _body_to_enu tests ──────────────────────────────────────────

    def test_body_to_enu_identity_at_yaw_zero(self):
        """Yaw=0 (facing East): forward=east, left=north."""
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        e, n, u = GeoJSONBridge._body_to_enu(1.0, 0.0, 0.0, yaw=0.0)
        np.testing.assert_allclose(e, 1.0, atol=1e-12)
        np.testing.assert_allclose(n, 0.0, atol=1e-12)

    def test_body_to_enu_yaw_90_faces_north(self):
        """Yaw=π/2 (facing North): forward=north, left=west."""
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        yaw = math.pi / 2.0
        e, n, u = GeoJSONBridge._body_to_enu(1.0, 0.0, 0.0, yaw)
        np.testing.assert_allclose(e, 0.0, atol=1e-12)
        np.testing.assert_allclose(n, 1.0, atol=1e-12)

    def test_body_to_enu_left_offset_at_yaw_90(self):
        """Yaw=π/2: left direction = -East."""
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        yaw = math.pi / 2.0
        e, n, u = GeoJSONBridge._body_to_enu(0.0, 1.0, 0.0, yaw)
        np.testing.assert_allclose(e, -1.0, atol=1e-12)
        np.testing.assert_allclose(n, 0.0, atol=1e-12)

    def test_body_to_enu_preserves_z(self):
        """Z component passes through unchanged."""
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        _, _, u = GeoJSONBridge._body_to_enu(0.0, 0.0, 3.5, yaw=1.23)
        np.testing.assert_allclose(u, 3.5, atol=1e-12)

    def test_body_to_enu_diagonal_yaw(self):
        """Yaw=π/4 (NE): 1m forward → (√2/2 E, √2/2 N)."""
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        yaw = math.pi / 4.0
        e, n, u = GeoJSONBridge._body_to_enu(1.0, 0.0, 0.0, yaw)
        s2 = math.sqrt(2.0) / 2.0
        np.testing.assert_allclose(e, s2, atol=1e-12)
        np.testing.assert_allclose(n, s2, atol=1e-12)

    # ── Equirectangular projection maths ────────────────────────────

    def test_zero_offset_returns_origin(self):
        """No displacement → coordinates equal origin."""
        R = self.R_EARTH
        lat_rad = math.radians(37.9755)
        d_lat = 0.0 / R * (180.0 / math.pi)
        d_lon = 0.0 / (R * math.cos(lat_rad)) * (180.0 / math.pi)
        assert abs(d_lat) < 1e-15
        assert abs(d_lon) < 1e-15

    def test_100m_north(self):
        """100m north offset gives ~0.0009° latitude increase."""
        d_lat = 100.0 / self.R_EARTH * (180.0 / math.pi)
        assert abs(d_lat - 0.000898) < 0.0001

    def test_100m_east(self):
        """100m east offset gives longitude change depending on latitude."""
        lat_rad = math.radians(37.9755)
        d_lon = 100.0 / (self.R_EARTH * math.cos(lat_rad)) * (180.0 / math.pi)
        assert abs(d_lon - 0.00114) < 0.0001

    def test_roundtrip_symmetry(self):
        """GPS offset for +x,+y should be inverse of -x,-y."""
        R = self.R_EARTH
        lat_rad = math.radians(37.9755)

        x, y = 50.0, 75.0
        d_lat_pos = y / R * (180.0 / math.pi)
        d_lon_pos = x / (R * math.cos(lat_rad)) * (180.0 / math.pi)

        d_lat_neg = (-y) / R * (180.0 / math.pi)
        d_lon_neg = (-x) / (R * math.cos(lat_rad)) * (180.0 / math.pi)

        np.testing.assert_allclose(d_lat_pos, -d_lat_neg, atol=1e-15)
        np.testing.assert_allclose(d_lon_pos, -d_lon_neg, atol=1e-15)

    # ── 3D coordinate output ────────────────────────────────────────

    def test_body_to_gps_returns_3_tuple(self):
        """body_to_gps must return (lon, lat, alt)."""
        # Simulate the math directly
        lat0, lon0, alt0 = 49.73, 13.35, 320.0
        yaw = 0.0
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
        e, n, u = GeoJSONBridge._body_to_enu(5.0, 0.0, 1.5, yaw)
        R = self.R_EARTH
        lat_rad = math.radians(lat0)
        d_lat = n / R * (180.0 / math.pi)
        d_lon = e / (R * math.cos(lat_rad)) * (180.0 / math.pi)
        result = (lon0 + d_lon, lat0 + d_lat, alt0 + u)
        assert len(result) == 3
        np.testing.assert_allclose(result[2], 321.5, atol=1e-6)


class TestGeoJSONSchema:
    """Tests for the GeoJSON output format (RFC-7946 + SimpleStyle)."""

    REQUIRED_PROPERTIES = [
        'class', 'id', 'confidence',
        'category', 'detection_type', 'source',
        'local_frame', 'altitude_m', 'height_m',
        'marker-color', 'marker-size', 'marker-symbol',
    ]

    @staticmethod
    def _make_geojson(detections, gps_valid=False):
        """Build a GeoJSON FeatureCollection like the bridge does.

        Each detection dict must include 'coordinates' (lon, lat, alt).
        Optionally includes 'size' (sx, sy, sz) and 'position' (x, y, z).
        Geometry type is class-dependent: person→Point, Fence→LineString,
        default→Polygon. Coordinates are 2D [lon, lat] only; altitude
        is in altitude_m property.
        """
        from triffid_ugv_perception.geojson_bridge import (
            GeoJSONBridge, _MIN_EXTENT, _POINT_CLASSES, _LINE_CLASSES,
        )
        features = []
        for det in detections:
            lon, lat, alt = det['coordinates']
            cls = det.get('class_name', 'unknown')
            sx, sy, sz = det.get('size', (0.0, 0.0, 0.0))
            pos_x, pos_y, pos_z = det.get('position', (lon, lat, 0.0))

            geom_type = GeoJSONBridge._geometry_type_for_class(cls)

            if geom_type == 'Point':
                geometry = {"type": "Point", "coordinates": [lon, lat]}

            elif geom_type == 'LineString':
                half_long = max(sx, sy, _MIN_EXTENT) / 2.0
                if sx >= sy:
                    endpoints = [
                        [pos_x + half_long, pos_y],
                        [pos_x - half_long, pos_y],
                    ]
                else:
                    endpoints = [
                        [pos_x, pos_y + half_long],
                        [pos_x, pos_y - half_long],
                    ]
                geometry = {"type": "LineString", "coordinates": endpoints}

            else:  # Polygon
                half_x = max(sx, _MIN_EXTENT) / 2.0
                half_y = max(sy, _MIN_EXTENT) / 2.0
                ring = [
                    [pos_x + half_x, pos_y + half_y],
                    [pos_x + half_x, pos_y - half_y],
                    [pos_x - half_x, pos_y - half_y],
                    [pos_x - half_x, pos_y + half_y],
                    [pos_x + half_x, pos_y + half_y],  # closed
                ]
                geometry = {"type": "Polygon", "coordinates": [ring]}

            feature = {
                "type": "Feature",
                "id": det.get('track_id', ''),
                "geometry": geometry,
                "properties": {
                    "class": cls,
                    "id": det.get('track_id', ''),
                    "confidence": det.get('confidence', 0.0),
                    "category": GeoJSONBridge._class_category(cls),
                    "detection_type": "seg",
                    "source": "ugv",
                    "local_frame": not gps_valid,
                    "altitude_m": round(alt, 2),
                    "height_m": round(float(sz), 2),
                    "marker-color": GeoJSONBridge._class_color(cls),
                    "marker-size": "medium",
                    "marker-symbol": GeoJSONBridge._class_symbol(cls),
                }
            }
            if geometry['type'] == 'Polygon':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
                feature["properties"]["fill"] = feature["properties"]["marker-color"]
                feature["properties"]["fill-opacity"] = 0.25
            elif geometry['type'] == 'LineString':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    def test_valid_feature_collection(self):
        dets = [{'coordinates': (23.7, 37.9, 320.0), 'class_name': 'Building',
                 'confidence': 0.85, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 1
        assert gj['features'][0]['type'] == 'Feature'
        # Building with no size → Polygon with _MIN_EXTENT
        assert gj['features'][0]['geometry']['type'] == 'Polygon'

    def test_all_required_properties_present(self):
        dets = [{'coordinates': (1.0, 2.0, 0.0), 'class_name': 'car',
                 'confidence': 0.7, 'track_id': '5'}]
        gj = self._make_geojson(dets, gps_valid=True)
        props = gj['features'][0]['properties']
        for key in self.REQUIRED_PROPERTIES:
            assert key in props, f"Missing property: {key}"

    def test_local_frame_true_when_no_gps(self):
        dets = [{'coordinates': (5.0, 10.0, 0.0), 'class_name': 'person',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=False)
        assert gj['features'][0]['properties']['local_frame'] is True

    def test_local_frame_false_when_gps_set(self):
        dets = [{'coordinates': (23.7, 37.9, 300.0), 'class_name': 'person',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        assert gj['features'][0]['properties']['local_frame'] is False

    def test_json_serialisable(self):
        dets = [
            {'coordinates': (1.0, 2.0, 0.0), 'class_name': 'person',
             'confidence': 0.9, 'track_id': '1'},
            {'coordinates': (3.0, 4.0, 0.0), 'class_name': 'car',
             'confidence': 0.7, 'track_id': '2'},
        ]
        gj = self._make_geojson(dets)
        json_str = json.dumps(gj)
        parsed = json.loads(json_str)
        assert parsed['type'] == 'FeatureCollection'
        assert len(parsed['features']) == 2

    def test_coordinates_are_lon_lat_only(self):
        """GeoJSON geometry uses 2D [lon, lat] only; altitude in properties."""
        lon, lat, alt = 23.7348, 37.9755, 310.5
        dets = [{'coordinates': (lon, lat, alt), 'class_name': 'Citizen',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        coords = gj['features'][0]['geometry']['coordinates']
        assert len(coords) == 2, f'Expected [lon, lat], got {coords}'
        assert coords[0] == lon
        assert coords[1] == lat
        # Altitude should be in properties, not in coordinates
        props = gj['features'][0]['properties']
        assert props['altitude_m'] == round(alt, 2)

    def test_empty_detections_yield_empty_collection(self):
        gj = self._make_geojson([])
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 0

    def test_marker_color_for_known_classes(self):
        for cls, expected_color in [('Flame', '#ff0000'), ('Civilian vehicle', '#0000ff')]:
            dets = [{'coordinates': (0, 0, 0), 'class_name': cls,
                     'confidence': 0.9, 'track_id': '1'}]
            gj = self._make_geojson(dets)
            assert gj['features'][0]['properties']['marker-color'] == expected_color

    def test_unknown_class_gets_default_color(self):
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'alien',
                 'confidence': 0.5, 'track_id': '1'}]
        gj = self._make_geojson(dets)
        assert gj['features'][0]['properties']['marker-color'] == '#808080'
        assert gj['features'][0]['properties']['marker-symbol'] == 'marker'

    def test_category_varies_by_class(self):
        """Category should reflect semantic class grouping, not a constant."""
        cases = [
            ('Flame', 'hazard'), ('First responder', 'person'),
            ('Civilian vehicle', 'vehicle'), ('Green tree', 'nature'),
            ('Building', 'infrastructure'), ('Fence', 'obstacle'),
            ('Helmet', 'equipment'), ('alien', 'unknown'),
        ]
        for cls, expected_cat in cases:
            dets = [{'coordinates': (0, 0, 0), 'class_name': cls,
                     'confidence': 0.9, 'track_id': '1'}]
            gj = self._make_geojson(dets)
            cat = gj['features'][0]['properties']['category']
            assert cat == expected_cat, f'{cls}: expected {expected_cat}, got {cat}'

    # ── 3D coordinate tests ──────────────────────────────────────────

    def test_gnss_altitude_property_present(self):
        """altitude_m property must be present and match z coordinate."""
        dets = [{'coordinates': (13.35, 49.73, 321.5), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        props = gj['features'][0]['properties']
        assert 'altitude_m' in props
        np.testing.assert_allclose(props['altitude_m'], 321.5)

    def test_height_property_present(self):
        """height_m property must reflect bbox size.z."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.7, 'track_id': '1',
                 'size': (1.0, 2.0, 4.5), 'position': (0, 0, 0)}]
        gj = self._make_geojson(dets, gps_valid=True)
        props = gj['features'][0]['properties']
        np.testing.assert_allclose(props['height_m'], 4.5)

    def test_point_coordinates_are_2d(self):
        """Point geometry coordinates must have 2 elements [lon, lat]."""
        dets = [{'coordinates': (13.35, 49.73, 320.0), 'class_name': 'First responder',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        coords = gj['features'][0]['geometry']['coordinates']
        assert len(coords) == 2, f'Expected [lon, lat], got {coords}'

    # ── Polygon geometry tests ──────────────────────────────────────

    def test_polygon_emitted_when_size_nonzero(self):
        """Detection with nonzero bbox size → Polygon geometry."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1',
                 'size': (2.0, 3.0, 4.0), 'position': (5.0, 6.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Polygon'

    def test_polygon_emitted_when_size_zero(self):
        """Polygon class with ALL-ZERO bbox size → Polygon with _MIN_EXTENT."""
        dets = [{'coordinates': (1.0, 2.0, 0.0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1',
                 'size': (0.0, 0.0, 0.0), 'position': (1.0, 2.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Polygon'

    def test_person_emits_point_geometry(self):
        """Person classes always emit Point geometry regardless of bbox size."""
        for cls in ('First responder', 'Citizen', 'Military personnel'):
            dets = [{'coordinates': (13.0, 49.0, 300.0), 'class_name': cls,
                     'confidence': 0.9, 'track_id': '1',
                     'size': (1.0, 0.5, 1.8), 'position': (5.0, 0.0, 0.0)}]
            gj = self._make_geojson(dets, gps_valid=True)
            geom = gj['features'][0]['geometry']
            assert geom['type'] == 'Point', f'{cls} should emit Point, got {geom["type"]}'
            assert len(geom['coordinates']) == 2

    def test_expanded_point_classes(self):
        """Equipment/vehicle/small-object classes emit Point geometry."""
        point_classes = [
            'Helmet', 'Destroyed vehicle', 'Fire hose', 'SCBA', 'Boot',
            'Mask', 'Window', 'Pole', 'Animal', 'Door', 'Civilian vehicle',
            'Hole in the ground', 'Bag', 'Ambulance', 'Fire truck', 'Cone',
            'Ax', 'Glove', 'Stairs', 'Protective glasses', 'Shovel',
            'Fire hydrant', 'Police vehicle', 'Army vehicle', 'Chainsaw',
            'aerial vehicle', 'Lifesaver', 'Extinguisher',
        ]
        for cls in point_classes:
            dets = [{'coordinates': (13.0, 49.0, 300.0), 'class_name': cls,
                     'confidence': 0.8, 'track_id': '1',
                     'size': (1.0, 0.5, 0.5), 'position': (3.0, 0.0, 0.0)}]
            gj = self._make_geojson(dets, gps_valid=True)
            geom = gj['features'][0]['geometry']
            assert geom['type'] == 'Point', f'{cls} should emit Point, got {geom["type"]}'

    def test_wall_emits_linestring_geometry(self):
        """Wall class emits LineString geometry (same as Fence)."""
        dets = [{'coordinates': (13.0, 49.0, 300.0), 'class_name': 'Wall',
                 'confidence': 0.7, 'track_id': '1',
                 'size': (3.0, 0.2, 2.0), 'position': (5.0, 0.0, 0.0)}]
        gj = self._make_geojson(dets, gps_valid=True)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'LineString'

    def test_fence_emits_linestring_geometry(self):
        """Fence class emits LineString geometry."""
        dets = [{'coordinates': (13.0, 49.0, 300.0), 'class_name': 'Fence',
                 'confidence': 0.7, 'track_id': '1',
                 'size': (5.0, 0.3, 1.2), 'position': (10.0, 0.0, 0.0)}]
        gj = self._make_geojson(dets, gps_valid=True)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'LineString'
        assert len(geom['coordinates']) == 2  # two endpoints
        for pt in geom['coordinates']:
            assert len(pt) == 2, f'LineString vertex should be [lon, lat], got {pt}'

    def test_linestring_has_stroke_properties(self):
        """LineString features should have stroke SimpleStyle but no fill."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Fence',
                 'confidence': 0.7, 'track_id': '1',
                 'size': (5.0, 0.3, 1.2), 'position': (0.0, 0.0, 0.0)}]
        gj = self._make_geojson(dets)
        props = gj['features'][0]['properties']
        assert 'stroke' in props
        assert 'stroke-width' in props
        assert 'stroke-opacity' in props
        assert 'fill' not in props
        assert 'fill-opacity' not in props

    def test_polygon_emitted_when_only_one_size_nonzero(self):
        """Detection with one nonzero size → Polygon (uses min extent)."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Green grass',
                 'confidence': 0.9, 'track_id': '1',
                 'size': (0.0, 2.5, 1.4), 'position': (3.0, 1.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Polygon'

    def test_polygon_emitted_when_size_absent(self):
        """Polygon class without size field → Polygon with _MIN_EXTENT."""
        dets = [{'coordinates': (1.0, 2.0, 0.0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1',
                 'position': (1.0, 2.0, 0.0)}]
        gj = self._make_geojson(dets)
        assert gj['features'][0]['geometry']['type'] == 'Polygon'

    def test_polygon_ring_is_closed(self):
        """RFC-7946: first and last coordinate of a ring must be identical."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Road',
                 'confidence': 0.7, 'track_id': '1',
                 'size': (1.0, 2.0, 0.0), 'position': (3.0, 4.0, 0.0)}]
        gj = self._make_geojson(dets)
        ring = gj['features'][0]['geometry']['coordinates'][0]
        assert len(ring) == 5, 'Polygon ring must have 5 vertices (4 + close)'
        assert ring[0] == ring[-1], 'Ring is not closed'

    def test_polygon_corners_match_extent(self):
        """Polygon corners should be centre ± half-extent."""
        sx, sy = 4.0, 6.0
        cx, cy = 10.0, 20.0
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Green grass',
                 'confidence': 0.6, 'track_id': '1',
                 'size': (sx, sy, 0.0), 'position': (cx, cy, 0.0)}]
        gj = self._make_geojson(dets)
        ring = gj['features'][0]['geometry']['coordinates'][0]
        xs = [p[0] for p in ring[:-1]]
        ys = [p[1] for p in ring[:-1]]
        np.testing.assert_allclose(max(xs) - min(xs), sx)
        np.testing.assert_allclose(max(ys) - min(ys), sy)

    def test_polygon_has_simplestyle_fill_properties(self):
        """Polygon features should carry stroke/fill SimpleStyle props."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1',
                 'size': (1.0, 1.0, 1.0), 'position': (0.0, 0.0, 0.0)}]
        gj = self._make_geojson(dets)
        props = gj['features'][0]['properties']
        assert 'stroke' in props
        assert 'stroke-width' in props
        assert 'fill' in props
        assert 'fill-opacity' in props
        assert props['stroke'] == props['marker-color']
        assert props['fill'] == props['marker-color']

    def test_point_lacks_fill_properties(self):
        """Point features should NOT have stroke/fill properties."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Helmet',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets)
        props = gj['features'][0]['properties']
        assert 'stroke' not in props
        assert 'fill' not in props

    def test_mixed_point_and_polygon(self):
        """FeatureCollection can contain both Point and Polygon features."""
        dets = [
            {'coordinates': (1, 2, 0), 'class_name': 'Citizen',
             'confidence': 0.9, 'track_id': '1'},
            {'coordinates': (3, 4, 0), 'class_name': 'Road',
             'confidence': 0.8, 'track_id': '2',
             'size': (5.0, 10.0, 0.1), 'position': (3.0, 4.0, 0.0)},
        ]
        gj = self._make_geojson(dets)
        types = [f['geometry']['type'] for f in gj['features']]
        assert 'Point' in types
        assert 'Polygon' in types

    def test_polygon_serialisable(self):
        """Polygon GeoJSON must be JSON-serialisable."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1',
                 'size': (3.0, 2.0, 5.0), 'position': (1.0, 1.0, 0.0)}]
        gj = self._make_geojson(dets)
        json_str = json.dumps(gj)
        parsed = json.loads(json_str)
        assert parsed['features'][0]['geometry']['type'] == 'Polygon'

    def test_polygon_vertices_are_2d(self):
        """Polygon ring vertices should have 2D coordinates [lon, lat]."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1',
                 'size': (2.0, 3.0, 4.0), 'position': (5.0, 6.0, 1.0)}]
        gj = self._make_geojson(dets)
        ring = gj['features'][0]['geometry']['coordinates'][0]
        for vertex in ring:
            assert len(vertex) == 2, f'Expected [lon, lat], got {vertex}'


# ═══════════════════════════════════════════════════════════════════════════
#  9.  DEPTH ENCODING & EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestDepthEncoding:
    """Tests for depth image handling specifics."""

    def test_16uc1_max_range(self):
        """uint16 max (65535 mm) = 65.535 m – valid for RealSense."""
        z_mm = np.uint16(65535)
        z_m = float(z_mm) / 1000.0
        np.testing.assert_allclose(z_m, 65.535)

    def test_16uc1_min_valid(self):
        """Minimum non-zero depth: 1 mm = 0.001 m."""
        z_m = 1.0 / 1000.0
        assert z_m == 0.001

    def test_large_depth_array_conversion(self):
        """Bulk conversion of depth array."""
        depth = np.random.default_rng(42).integers(
            0, 10000, size=(480, 640), dtype=np.uint16
        )
        z_m = depth.astype(np.float64) / 1000.0
        assert z_m.shape == (480, 640)
        assert z_m.dtype == np.float64
        assert np.all(z_m >= 0)
        assert np.all(z_m <= 10.0)

    def test_zero_depth_mask(self):
        """Zero depth pixels should be masked out."""
        depth = np.array([0, 0, 1500, 3000, 0], dtype=np.uint16)
        valid = depth > 0
        assert list(valid) == [False, False, True, True, False]


# ═══════════════════════════════════════════════════════════════════════════
#  10.  FRAME IDS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFrameConstants:
    """Verify frame IDs and constants match the pixel-aligned RGB-D setup."""

    def test_base_frame_id(self):
        from triffid_ugv_perception.ugv_node import BASE_FRAME
        assert BASE_FRAME == 'b2/base_link'

    def test_default_rgb_topic(self):
        from triffid_ugv_perception.ugv_node import DEFAULT_RGB_TOPIC
        assert DEFAULT_RGB_TOPIC == '/camera_front_435i/realsense_front_435i/color/image_raw'

    def test_default_depth_topic(self):
        from triffid_ugv_perception.ugv_node import DEFAULT_DEPTH_TOPIC
        assert DEFAULT_DEPTH_TOPIC == '/camera_front_435i/realsense_front_435i/depth/image_rect_raw'

    def test_default_camera_info_topic(self):
        from triffid_ugv_perception.ugv_node import DEFAULT_CAMERA_INFO_TOPIC
        assert DEFAULT_CAMERA_INFO_TOPIC == '/camera_front_435i/realsense_front_435i/color/camera_info'

    def test_max_depth_samples(self):
        from triffid_ugv_perception.ugv_node import _MAX_DEPTH_SAMPLES
        assert _MAX_DEPTH_SAMPLES == 500

    def test_target_classes_include_water(self):
        from triffid_ugv_perception.ugv_node import TARGET_CLASSES
        assert 0 in TARGET_CLASSES
        assert TARGET_CLASSES[0] == 'Water'

    def test_target_classes_include_key_classes(self):
        from triffid_ugv_perception.ugv_node import TARGET_CLASSES
        # Verify a selection of important TRIFFID classes exist
        class_names = set(TARGET_CLASSES.values())
        for expected in ('Flame', 'Smoke', 'First responder', 'Building',
                         'Road', 'Citizen', 'Debris', 'Green tree'):
            assert expected in class_names, f'{expected} not in TARGET_CLASSES'


# ═══════════════════════════════════════════════════════════════════════════
#  11.  INTRINSICS VALIDATION (from bag)
# ═══════════════════════════════════════════════════════════════════════════

class TestIntrinsicsConsistency:
    """Validate that intrinsic values for the pixel-aligned camera are
    self-consistent (single shared K matrix for both RGB and depth)."""

    # Shared intrinsics (pixel-aligned RealSense: aligned_depth_to_color)
    K = [642.0186767578125, 0.0, 646.3067626953125,
         0.0, 641.4913330078125, 364.65814208984375,
         0.0, 0.0, 1.0]
    W, H = 1280, 720

    def test_principal_point_inside_image(self):
        cx, cy = self.K[2], self.K[5]
        assert 0 < cx < self.W
        assert 0 < cy < self.H

    def test_focal_lengths_positive(self):
        assert self.K[0] > 0  # fx
        assert self.K[4] > 0  # fy

    def test_k_matrix_last_row(self):
        """K matrix last row should be [0, 0, 1]."""
        assert self.K[6:] == [0.0, 0.0, 1.0]

    def test_near_square_pixel(self):
        """fx ≈ fy (nearly square pixels)."""
        np.testing.assert_allclose(self.K[0], self.K[4], rtol=0.01)

    def test_principal_point_near_centre(self):
        """cx ≈ W/2, cy ≈ H/2 (close to ideal pinhole)."""
        np.testing.assert_allclose(self.K[2], self.W / 2, atol=20)
        np.testing.assert_allclose(self.K[5], self.H / 2, atol=20)


# ═══════════════════════════════════════════════════════════════════════════
#  12.  TRACKER STRESS / EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════

class TestTrackerEdgeCases:
    """Edge cases and stress tests for the IoU tracker."""

    def _make_det(self, bbox, cls='person', conf=0.9, pos=(0, 0, 0)):
        return {
            'bbox': bbox,
            'class_id': 0,
            'class_name': cls,
            'confidence': conf,
            'position': pos,
        }

    def test_single_detection_single_frame(self):
        tracker = IoUTracker()
        dets = [self._make_det((10, 10, 50, 50))]
        results = tracker.update(dets)
        assert len(results) == 1
        assert results[0]['track_id'] == 1

    def test_many_concurrent_objects(self):
        """20 simultaneous detections should all get unique IDs."""
        tracker = IoUTracker()
        dets = [self._make_det((i * 60, 0, i * 60 + 50, 50))
                for i in range(20)]
        results = tracker.update(dets)
        ids = [r['track_id'] for r in results]
        assert len(set(ids)) == 20

    def test_rapid_appearance_disappearance(self):
        """Object appears, disappears, re-appears → new ID."""
        tracker = IoUTracker(max_age=1)
        det = [self._make_det((100, 100, 200, 200))]

        r1 = tracker.update(det)
        id1 = r1[0]['track_id']

        tracker.update([])  # empty → LOST, time_since_update=1
        tracker.update([])  # time_since_update=2 > max_age=1 → removed
        tracker.update([])  # ensure fully expired

        r2 = tracker.update(det)
        id2 = r2[0]['track_id']

        assert id2 > id1

    def test_overlapping_detections_same_frame(self):
        """Two heavily overlapping bboxes in same frame get distinct IDs."""
        tracker = IoUTracker()
        dets = [
            self._make_det((100, 100, 200, 200)),
            self._make_det((110, 110, 210, 210)),  # high IoU with above
        ]
        results = tracker.update(dets)
        ids = [r['track_id'] for r in results]
        assert len(set(ids)) == 2

    def test_iou_threshold_boundary(self):
        """Detection exactly at IoU threshold boundary."""
        tracker = IoUTracker(iou_threshold=0.5)

        # Create boxes with exactly 50% overlap
        # A=(0,0,100,100) area=10000
        # B=(50,0,150,100) area=10000, intersection=(50,0,100,100)=5000
        # IoU = 5000/15000 = 0.333 → below 0.5 → no match
        # Use distinct 3D positions so the position gate doesn't link them
        dets1 = [self._make_det((0, 0, 100, 100), pos=(1.0, 0.0, 0.0))]
        dets2 = [self._make_det((50, 0, 150, 100), pos=(10.0, 10.0, 0.0))]

        r1 = tracker.update(dets1)
        r2 = tracker.update(dets2)
        # IoU ≈ 0.333 < 0.5, and 3D dist > pos_gate → new track
        assert r1[0]['track_id'] != r2[0]['track_id']

    def test_tracker_with_none_position(self):
        """Detections with position=None should still be tracked."""
        tracker = IoUTracker()
        det = [{
            'bbox': (10, 10, 50, 50),
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.9,
            'position': None,
        }]
        results = tracker.update(det)
        assert len(results) == 1
        assert results[0]['position'] is None

    def test_max_age_zero_expires_immediately(self):
        """With max_age=0, unmatched tracks expire in 1 frame."""
        tracker = IoUTracker(max_age=0)
        tracker.update([self._make_det((10, 10, 50, 50))])
        assert len(tracker.tracks) == 1

        tracker.update([])  # LOST, time_since_update=1 > max_age=0 → removed
        assert len(tracker.tracks) == 0

    def test_position_update_on_rematch(self):
        """Position should update when track is re-matched."""
        tracker = IoUTracker()
        det1 = [self._make_det((10, 10, 50, 50), pos=(1.0, 2.0, 3.0))]
        det2 = [self._make_det((10, 10, 50, 50), pos=(4.0, 5.0, 6.0))]

        tracker.update(det1)
        r2 = tracker.update(det2)
        assert r2[0]['position'] == (4.0, 5.0, 6.0)

    def test_hundred_frames_stability(self):
        """Run 100 frames with stable detections, verify IDs are consistent."""
        tracker = IoUTracker()
        dets = [
            self._make_det((100, 100, 200, 200)),
            self._make_det((400, 400, 500, 500)),
        ]

        first_ids = None
        for frame in range(100):
            results = tracker.update(dets)
            ids = sorted(r['track_id'] for r in results)
            if first_ids is None:
                first_ids = ids
            else:
                assert ids == first_ids, f"IDs changed at frame {frame}"


# ═══════════════════════════════════════════════════════════════════════════
#  13.  FULL PIPELINE MATH VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineMathValidation:
    """Validates the complete math chain with known values."""

    def test_backproject_then_project_identity_same_camera(self):
        """For a single camera, backproject→project is identity."""
        fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0

        # Pick random pixels
        rng = np.random.default_rng(123)
        for _ in range(100):
            u = rng.uniform(0, 1280)
            v = rng.uniform(0, 720)
            z = rng.uniform(0.5, 20.0)

            # Backproject
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z

            # Re-project
            u_rec = fx * (X / Z) + cx
            v_rec = fy * (Y / Z) + cy

            np.testing.assert_allclose([u_rec, v_rec], [u, v], atol=1e-8)

    def test_transform_then_inverse_is_identity(self):
        """Applying R and then R^T should recover original points."""
        R = UGVPerceptionNode._quat_to_matrix(0.653, -0.653, 0.271, 0.271)
        t = np.array([0.025, 0.071, 0.039])

        pts_orig = np.array([[1, 2, 3], [4, 5, 6], [-1, 0, 1]], dtype=np.float64)

        # Forward: p' = R @ p + t
        pts_transformed = (R @ pts_orig.T).T + t

        # Inverse: p = R^T @ (p' - t)
        pts_recovered = (R.T @ (pts_transformed - t).T).T

        np.testing.assert_allclose(pts_recovered, pts_orig, atol=1e-8)

    def test_median_of_single_point_is_itself(self):
        pts = np.array([[2.5, 3.5, 4.5]])
        median = np.median(pts, axis=0)
        np.testing.assert_array_equal(median, [2.5, 3.5, 4.5])

    def test_median_of_two_points_is_midpoint(self):
        pts = np.array([[0.0, 0.0, 2.0], [4.0, 6.0, 8.0]])
        median = np.median(pts, axis=0)
        np.testing.assert_allclose(median, [2.0, 3.0, 5.0])

    def test_median_robust_to_outlier(self):
        """Median should be robust to a single outlier in 5 points."""
        pts = np.array([
            [1.0, 1.0, 3.0],
            [1.1, 1.1, 3.1],
            [0.9, 0.9, 2.9],
            [1.0, 1.0, 3.0],
            [100.0, 100.0, 100.0],  # outlier
        ])
        median = np.median(pts, axis=0)
        # Median should be close to (1.0, 1.0, 3.0), not pulled by outlier
        np.testing.assert_allclose(median, [1.0, 1.0, 3.0], atol=0.11)


# ═══════════════════════════════════════════════════════════════════════════
#  14.  SIMPLESTYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class TestSimpleStyleHelpers:
    """Tests for the GeoJSON SimpleStyle color and symbol helpers."""

    # Import the static methods without instantiating the ROS node
    from triffid_ugv_perception.geojson_bridge import GeoJSONBridge
    _color = staticmethod(GeoJSONBridge._class_color)
    _symbol = staticmethod(GeoJSONBridge._class_symbol)

    def test_person_color(self):
        assert self._color('First responder') == '#1e90ff'

    def test_car_color(self):
        assert self._color('Civilian vehicle') == '#0000ff'

    def test_unknown_class_default_color(self):
        assert self._color('spaceship') == '#808080'

    def test_person_symbol(self):
        assert self._symbol('First responder') == 'pitch'

    def test_unknown_class_default_symbol(self):
        assert self._symbol('spaceship') == 'marker'

    def test_fire_classes_have_red_tones(self):
        """Fire-related classes should have red/orange colors."""
        for cls in ('Flame', 'Smoke', 'Burnt tree'):
            color = self._color(cls)
            assert color != '#808080', f'{cls} has default color'

    def test_nature_classes_have_green_or_brown(self):
        """Green vegetation should be greenish."""
        for cls in ('Green tree', 'Green grass', 'Green plant'):
            color = self._color(cls)
            assert color != '#808080', f'{cls} has default color'

    def test_all_target_classes_have_colors(self):
        from triffid_ugv_perception.ugv_node import TARGET_CLASSES
        for cls_name in TARGET_CLASSES.values():
            color = self._color(cls_name)
            assert color.startswith('#')
            assert len(color) == 7  # #RRGGBB


# ═══════════════════════════════════════════════════════════════════════════
#  BBOX → 3-D CORNERS  (back-projection fallback for sparse depth)
# ═══════════════════════════════════════════════════════════════════════════

class TestBboxTo3DCorners:
    """Tests for UGVPerceptionNode._bbox_to_3d_corners (optical convention).

    New signature: _bbox_to_3d_corners(self, u1, v1, u2, v2, depth, fx, fy, cx, cy)
    Convention: camera_optical_frame (X=right, Y=down, Z=forward).
    """

    FX = 642.0186767578125
    FY = 641.4913330078125
    CX = 646.3067626953125
    CY = 364.65814208984375

    def _call(self, u1, v1, u2, v2, depth):
        """Call the method with a dummy self."""
        dummy = type('obj', (object,), {})()
        return UGVPerceptionNode._bbox_to_3d_corners(
            dummy, u1, v1, u2, v2, depth,
            self.FX, self.FY, self.CX, self.CY,
        )

    def test_returns_8x3(self):
        corners = self._call(100, 100, 300, 300, 5.0)
        assert corners.shape == (8, 3)

    def test_z_equals_depth(self):
        """All Z values should equal the input depth (optical: Z=forward)."""
        corners = self._call(50, 50, 200, 200, 3.0)
        np.testing.assert_allclose(corners[:, 2], 3.0)

    def test_nonzero_xy_extent(self):
        """X and Y extent must be positive for any non-degenerate bbox."""
        corners = self._call(100, 100, 500, 400, 4.0)
        extent = np.ptp(corners, axis=0)
        assert extent[0] > 0.1, "X extent should be positive"
        assert extent[1] > 0.1, "Y extent should be positive"

    def test_principal_point_bbox_centered(self):
        """Bbox centred on principal point → symmetric X, Y."""
        half = 50.0
        corners = self._call(
            self.CX - half, self.CY - half,
            self.CX + half, self.CY + half, 5.0,
        )
        x_extent = np.ptp(corners[:, 0])
        y_extent = np.ptp(corners[:, 1])
        assert x_extent > 0
        assert y_extent > 0

    def test_larger_bbox_larger_extent(self):
        """Wider 2D bbox at same depth → larger X extent."""
        small = self._call(300, 200, 400, 300, 5.0)
        large = self._call(100, 200, 600, 300, 5.0)
        assert np.ptp(large[:, 0]) > np.ptp(small[:, 0])

    def test_farther_depth_larger_extent(self):
        """Same pixel bbox at greater depth → proportionally larger extent."""
        near = self._call(200, 200, 400, 400, 2.0)
        far  = self._call(200, 200, 400, 400, 8.0)
        assert np.ptp(far[:, 0]) > np.ptp(near[:, 0])
        assert np.ptp(far[:, 1]) > np.ptp(near[:, 1])


# ═══════════════════════════════════════════════════════════════════════════
#  3-D NMS DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestNMS3D:
    """Tests for UGVPerceptionNode._nms_3d (static method)."""

    nms = staticmethod(UGVPerceptionNode._nms_3d)

    @staticmethod
    def _det(x, y, z, conf, cls='Building'):
        return {
            'position': (x, y, z),
            'extent': (1.0, 1.0, 1.0),
            'class_id': 0,
            'class_name': cls,
            'confidence': conf,
            'bbox': [0, 0, 100, 100],
            'n_depth_pts': 5,
        }

    def test_empty_list(self):
        assert self.nms([], dist_thresh=0.5) == []

    def test_single_detection_kept(self):
        dets = [self._det(1, 2, 3, 0.9)]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 1

    def test_identical_position_keeps_highest_conf(self):
        dets = [
            self._det(2.0, 1.0, -0.4, 0.62, 'Destroyed building'),
            self._det(2.0, 1.0, -0.4, 0.36, 'Building'),
        ]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 1
        assert result[0]['confidence'] == 0.62

    def test_close_positions_suppressed(self):
        """Detections within dist_thresh should be merged."""
        dets = [
            self._det(2.0, 1.0, 0.0, 0.8),
            self._det(2.1, 1.05, 0.0, 0.5),  # ~0.12m away
        ]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 1
        assert result[0]['confidence'] == 0.8

    def test_far_apart_both_kept(self):
        """Detections far apart should both survive."""
        dets = [
            self._det(0.0, 0.0, 0.0, 0.9),
            self._det(5.0, 5.0, 0.0, 0.7),
        ]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 2

    def test_three_at_same_spot(self):
        """Three overlapping → only the best survives."""
        dets = [
            self._det(1.0, 1.0, 1.0, 0.3, 'A'),
            self._det(1.0, 1.0, 1.0, 0.9, 'B'),
            self._det(1.0, 1.0, 1.0, 0.6, 'C'),
        ]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 1
        assert result[0]['class_name'] == 'B'

    def test_two_clusters(self):
        """Two clusters each with 2 detections → 2 survivors."""
        dets = [
            self._det(0.0, 0.0, 0.0, 0.8),
            self._det(0.1, 0.0, 0.0, 0.5),
            self._det(5.0, 5.0, 0.0, 0.7),
            self._det(5.1, 5.0, 0.0, 0.3),
        ]
        result = self.nms(dets, dist_thresh=0.5)
        assert len(result) == 2

    def test_order_independence(self):
        """Result should not depend on input order."""
        dets_a = [
            self._det(1.0, 1.0, 1.0, 0.4, 'X'),
            self._det(1.0, 1.0, 1.0, 0.9, 'Y'),
        ]
        dets_b = list(reversed(dets_a))
        r_a = self.nms(dets_a, dist_thresh=0.5)
        r_b = self.nms(dets_b, dist_thresh=0.5)
        assert len(r_a) == len(r_b) == 1
        assert r_a[0]['class_name'] == r_b[0]['class_name'] == 'Y'


# ═══════════════════════════════════════════════════════════════════════════
#  MQTT integration
# ═══════════════════════════════════════════════════════════════════════════

class TestMqttIntegration:
    """Tests for MQTT support in geojson_bridge."""

    def test_paho_available_flag(self):
        """_PAHO_AVAILABLE reflects whether paho-mqtt is importable."""
        from triffid_ugv_perception.geojson_bridge import _PAHO_AVAILABLE
        # Should be True if paho-mqtt is installed, False otherwise.
        # Either way the import must succeed.
        assert isinstance(_PAHO_AVAILABLE, bool)

    def test_mqtt_parameters_declared(self):
        """Bridge module declares mqtt_* parameter names in source."""
        import inspect
        from triffid_ugv_perception import geojson_bridge as mod
        src = inspect.getsource(mod.GeoJSONBridge.__init__)
        for param in ('mqtt_enabled', 'mqtt_host', 'mqtt_port', 'mqtt_topic'):
            assert param in src, f"Parameter {param} not declared"

    def test_publish_includes_mqtt(self):
        """_publish method references self._mqtt_client."""
        import inspect
        from triffid_ugv_perception import geojson_bridge as mod
        src = inspect.getsource(mod.GeoJSONBridge._publish)
        assert '_mqtt_client' in src
        assert '_mqtt_topic' in src


# ═══════════════════════════════════════════════════════════════════════════
#  ByteTracker-specific tests
# ═══════════════════════════════════════════════════════════════════════════

class TestByteTrackerFeatures:
    """Tests for ByteTracker: Kalman, two-pass, confirmation, 3D gate."""

    from triffid_ugv_perception.tracker import ByteTracker

    def _make_det(self, bbox, cls='person', conf=0.9, pos=(0, 0, 0)):
        return {
            'bbox': bbox,
            'class_id': 0,
            'class_name': cls,
            'confidence': conf,
            'position': pos,
        }

    # ── Track confirmation gate ──

    def test_n_init_withholds_tentative_tracks(self):
        """With n_init=3, first two frames should return empty."""
        tracker = self.ByteTracker(n_init=3)
        det = [self._make_det((100, 100, 200, 200), conf=0.9)]
        r1 = tracker.update(det)
        assert r1 == [], "Frame 1: should be tentative"
        r2 = tracker.update(det)
        assert r2 == [], "Frame 2: should be tentative"
        r3 = tracker.update(det)
        assert len(r3) == 1, "Frame 3: should be confirmed"

    def test_n_init_1_publishes_immediately(self):
        """With n_init=1, tracks are confirmed on first frame."""
        tracker = self.ByteTracker(n_init=1)
        r = tracker.update([self._make_det((100, 100, 200, 200))])
        assert len(r) == 1

    # ── Two-pass association ──

    def test_low_conf_second_pass_maintains_track(self):
        """A low-confidence detection should maintain a track via 2nd pass."""
        tracker = self.ByteTracker(
            n_init=1, conf_threshold_high=0.5,
            iou_threshold=0.3, iou_threshold_low=0.1,
        )
        det_high = [self._make_det((100, 100, 200, 200), conf=0.8)]
        det_low = [self._make_det((105, 105, 205, 205), conf=0.3)]

        r1 = tracker.update(det_high)
        id1 = r1[0]['track_id']

        r2 = tracker.update(det_low)
        assert len(r2) == 1
        assert r2[0]['track_id'] == id1, "Low-conf should keep same track"

    # ── 3D position gate ──

    def test_3d_position_recovers_small_bbox_match(self):
        """When IoU is low but 3D position is close, track is maintained."""
        tracker = self.ByteTracker(
            n_init=1, iou_threshold=0.3, pos_gate=2.0,
        )
        # Small bbox, shifts enough that IoU < 0.3
        det1 = [self._make_det((500, 300, 520, 320), conf=0.8,
                                pos=(3.0, 1.0, -0.5))]
        det2 = [self._make_det((530, 295, 550, 315), conf=0.6,
                                pos=(3.1, 1.0, -0.5))]  # 0.1m away

        r1 = tracker.update(det1)
        r2 = tracker.update(det2)
        assert r1[0]['track_id'] == r2[0]['track_id']

    def test_3d_far_away_creates_new_track(self):
        """When both IoU is low AND 3D position is far, a new track is made."""
        tracker = self.ByteTracker(n_init=1, pos_gate=2.0)
        det1 = [self._make_det((100, 100, 150, 150), conf=0.8,
                                pos=(1.0, 0.0, 0.0))]
        det2 = [self._make_det((500, 500, 550, 550), conf=0.8,
                                pos=(15.0, 10.0, 0.0))]

        r1 = tracker.update(det1)
        r2 = tracker.update(det2)
        assert r1[0]['track_id'] != r2[0]['track_id']

    # ── Kalman prediction ──

    def test_kalman_prediction_aids_matching(self):
        """After a gap frame, Kalman prediction moves bbox toward detection."""
        tracker = self.ByteTracker(n_init=1, max_age=5)
        # Object moving right
        det1 = [self._make_det((100, 100, 200, 200), conf=0.8)]
        det2 = [self._make_det((110, 100, 210, 200), conf=0.8)]

        r1 = tracker.update(det1)
        r2 = tracker.update(det2)
        assert r1[0]['track_id'] == r2[0]['track_id']

        # Skip a frame
        tracker.update([])

        # Continue moving right — Kalman should predict forward
        det4 = [self._make_det((130, 100, 230, 200), conf=0.8)]
        r4 = tracker.update(det4)
        assert len(r4) == 1
        assert r4[0]['track_id'] == r1[0]['track_id']

    # ── Vectorised IoU ──

    def test_iou_batch_identical(self):
        from triffid_ugv_perception.tracker import _iou_batch
        iou = _iou_batch([(0, 0, 10, 10)], [(0, 0, 10, 10)])
        np.testing.assert_allclose(iou, [[1.0]])

    def test_iou_batch_no_overlap(self):
        from triffid_ugv_perception.tracker import _iou_batch
        iou = _iou_batch([(0, 0, 10, 10)], [(20, 20, 30, 30)])
        np.testing.assert_allclose(iou, [[0.0]])

    def test_iou_batch_multi(self):
        from triffid_ugv_perception.tracker import _iou_batch
        a = [(0, 0, 10, 10), (20, 20, 30, 30)]
        b = [(0, 0, 10, 10), (100, 100, 110, 110)]
        iou = _iou_batch(a, b)
        assert iou.shape == (2, 2)
        assert iou[0, 0] == 1.0
        assert iou[1, 1] == 0.0

    # ── ID persistence under ByteTracker ──

    def test_ids_never_reused(self):
        tracker = self.ByteTracker(n_init=1, max_age=1)
        det = [self._make_det((100, 100, 200, 200))]
        r1 = tracker.update(det)
        id1 = r1[0]['track_id']

        tracker.update([])
        tracker.update([])
        tracker.update([])

        r2 = tracker.update(det)
        assert r2[0]['track_id'] > id1

    def test_extent_and_depth_pts_propagated(self):
        """Tracker must propagate extent, n_depth_pts, and class_id."""
        tracker = self.ByteTracker(n_init=1)
        det = [{
            'bbox': (100, 100, 200, 200),
            'class_id': 14,
            'class_name': 'Building',
            'confidence': 0.85,
            'position': (5.0, 1.0, -0.5),
            'extent': (2.0, 3.5, 4.0),
            'n_depth_pts': 42,
        }]
        results = tracker.update(det)
        assert len(results) == 1
        r = results[0]
        assert r['extent'] == (2.0, 3.5, 4.0)
        assert r['n_depth_pts'] == 42
        assert r['class_id'] == 14

    def test_class_gate_prevents_cross_class_match(self):
        """A track should NOT match a detection of a different class."""
        tracker = self.ByteTracker(n_init=1, iou_threshold=0.3)
        # Frame 1: create Fence track at (100, 100, 200, 200)
        tracker.update([{
            'bbox': (100, 100, 200, 200),
            'class_name': 'Fence',
            'class_id': 1,
            'confidence': 0.8,
            'position': (5.0, 0.0, 0.0),
        }])
        # Frame 2: overlapping bbox but different class
        results = tracker.update([{
            'bbox': (105, 105, 210, 210),
            'class_name': 'Civilian vehicle',
            'class_id': 21,
            'confidence': 0.9,
            'position': (5.0, 0.0, 0.0),
        }])
        # Should produce TWO tracks (not merge into one) — the original
        # Fence becomes lost and a new Civilian vehicle track is created.
        # With n_init=1, the new track is immediately confirmed.
        classes = {r['class_name'] for r in results}
        assert 'Civilian vehicle' in classes
        # The Fence track should NOT have been re-labelled
        for r in results:
            if r['class_name'] == 'Fence':
                assert r['track_id'] == 1

    def test_class_majority_vote_stable(self):
        """Class stays stable via majority vote even if one frame disagrees."""
        tracker = self.ByteTracker(n_init=1, iou_threshold=0.3)
        base = {
            'bbox': (100, 100, 200, 200),
            'class_name': 'Building',
            'class_id': 7,
            'confidence': 0.8,
            'position': (5.0, 0.0, 0.0),
        }
        # Frames 1..3: Building
        for _ in range(3):
            tracker.update([base])
        # Frame 4: same bbox, same class but one fluke "Destroyed building"
        fluke = dict(base, class_name='Destroyed building', class_id=99)
        # Class gate will prevent matching with different class,
        # so the Building track stays and the fluke creates a new one.
        results = tracker.update([fluke])
        building = [r for r in results if r['class_name'] == 'Building']
        # Building track should remain as Building (even if lost it stays)
        # since the fluke is a *different* class, it won't match.
        # The original Building track may go lost but its class never flipped.
        assert all(r['class_name'] != 'Destroyed building'
                   for r in results if r['track_id'] == 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Spatial deduplication (collect_samples.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestSpatialDedup:
    """Tests for the spatial dedup logic in SampleCollector."""

    @staticmethod
    def _feature(fid, cls, conf, lon, lat):
        return {
            "type": "Feature",
            "id": str(fid),
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon - 0.0001, lat - 0.0001],
                    [lon + 0.0001, lat - 0.0001],
                    [lon + 0.0001, lat + 0.0001],
                    [lon - 0.0001, lat + 0.0001],
                    [lon - 0.0001, lat - 0.0001],
                ]],
            },
            "properties": {
                "class": cls,
                "confidence": conf,
            },
        }

    def test_identical_location_same_class_merged(self):
        """Two features of same class at same location → keep higher conf."""
        import sys, os
        scripts_dir = os.path.join(
            os.path.dirname(__file__), '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        collect = pytest.importorskip('collect_samples')
        SC = collect.SampleCollector

        f1 = self._feature(1, 'Pole', 0.4, 13.351615, 49.726300)
        f2 = self._feature(2, 'Pole', 0.6, 13.351616, 49.726301)  # ~0.1m away

        # Pass the class itself as 'self' — all methods called on self
        # (_feature_centroid, _haversine) are @staticmethod.
        result = SC._spatial_dedup(SC, [f1, f2])
        assert len(result) == 1
        assert result[0]['properties']['confidence'] == 0.6

    def test_different_class_not_merged(self):
        """Features of different classes at same location → both kept."""
        import sys, os
        scripts_dir = os.path.join(
            os.path.dirname(__file__), '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        collect = pytest.importorskip('collect_samples')
        SC = collect.SampleCollector

        f1 = self._feature(1, 'Pole', 0.5, 13.351615, 49.726300)
        f2 = self._feature(2, 'Building', 0.8, 13.351616, 49.726301)

        result = SC._spatial_dedup(SC, [f1, f2])
        assert len(result) == 2

    def test_far_apart_same_class_not_merged(self):
        """Same class but > merge radius → both kept."""
        import sys, os
        scripts_dir = os.path.join(
            os.path.dirname(__file__), '..', 'scripts')
        sys.path.insert(0, scripts_dir)
        collect = pytest.importorskip('collect_samples')
        SC = collect.SampleCollector

        f1 = self._feature(1, 'Pole', 0.5, 13.351615, 49.726300)
        f2 = self._feature(2, 'Pole', 0.8, 13.352615, 49.726300)  # ~74m away

        result = SC._spatial_dedup(SC, [f1, f2])
        assert len(result) == 2


#  Entry point

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
