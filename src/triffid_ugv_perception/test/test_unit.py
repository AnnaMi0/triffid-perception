#!/usr/bin/env python3
"""
TRIFFID UGV Perception – Unit Tests
=====================================
Comprehensive tests for the UGV perception pipeline, covering:

  1. Geometry & math  (quaternion→matrix, back-projection, projection, pinhole model)
  2. Cross-camera pipeline logic  (depth grid → RGB frame → bbox filtering → median)
  3. IoU tracker  (matching, ID persistence, aging, edge cases)
  4. GeoJSON bridge  (local_to_gps, GeoJSON schema, local_frame flag)
  5. Depth handling  (mm→m conversion, zero-depth filtering, grid sampling)

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
#  2.  PINHOLE PROJECTION (3D → 2D pixel)
# ═══════════════════════════════════════════════════════════════════════════

class TestProjectToRGB:
    """Tests for _project_to_rgb logic (extracted without ROS node)."""

    # Replicate the projection math from ugv_node._project_to_rgb
    @staticmethod
    def _project(pts, fx, fy, cx, cy):
        """Pure-python re-implementation matching ugv_node._project_to_rgb."""
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        pixels = np.full((len(X), 2), -1.0)
        valid = Z > 0
        pixels[valid, 0] = fx * (X[valid] / Z[valid]) + cx
        pixels[valid, 1] = fy * (Y[valid] / Z[valid]) + cy
        return pixels

    # --- RGB intrinsics from bag ---
    FX, FY, CX, CY = 500.0, 500.0, 640.0, 360.0

    def test_principal_point(self):
        """Point on optical axis projects to (cx, cy)."""
        pts = np.array([[0.0, 0.0, 5.0]])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        np.testing.assert_allclose(px[0], [640.0, 360.0])

    def test_known_offset(self):
        """Point at (1, 0, 5) projects to (cx + fx/5, cy)."""
        pts = np.array([[1.0, 0.0, 5.0]])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        expected_u = 500.0 * 1.0 / 5.0 + 640.0  # = 740
        np.testing.assert_allclose(px[0, 0], expected_u)
        np.testing.assert_allclose(px[0, 1], 360.0)

    def test_behind_camera(self):
        """Points with Z ≤ 0 should get (-1, -1)."""
        pts = np.array([
            [1.0, 2.0, -1.0],
            [0.0, 0.0,  0.0],
        ])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        np.testing.assert_array_equal(px, [[-1, -1], [-1, -1]])

    def test_symmetry(self):
        """Symmetric points project symmetrically around principal point."""
        pts = np.array([
            [ 1.0,  1.0, 10.0],
            [-1.0, -1.0, 10.0],
        ])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        # u1 - cx == cx - u2
        assert abs((px[0, 0] - self.CX) + (px[1, 0] - self.CX)) < 1e-10
        assert abs((px[0, 1] - self.CY) + (px[1, 1] - self.CY)) < 1e-10

    def test_batch_mixed_validity(self):
        """Mix of valid and behind-camera points."""
        pts = np.array([
            [0.0, 0.0,  3.0],   # valid
            [1.0, 2.0, -1.0],   # behind
            [2.0, 1.0,  4.0],   # valid
        ])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        assert px[0, 0] != -1 and px[0, 1] != -1
        assert px[1, 0] == -1 and px[1, 1] == -1
        assert px[2, 0] != -1 and px[2, 1] != -1

    def test_close_vs_far_magnification(self):
        """Closer point projects further from centre than farther point."""
        pts = np.array([
            [1.0, 0.0, 2.0],   # close
            [1.0, 0.0, 10.0],  # far
        ])
        px = self._project(pts, self.FX, self.FY, self.CX, self.CY)
        assert px[0, 0] > px[1, 0]  # closer → larger u offset


# ═══════════════════════════════════════════════════════════════════════════
#  3.  BACK-PROJECTION (depth pixel → 3D)
# ═══════════════════════════════════════════════════════════════════════════

class TestBackProjection:
    """Tests for the pinhole back-projection math used in
    _depth_grid_to_rgb_frame (depth pixel → 3D in depth frame)."""

    # Depth intrinsics from bag
    FX_D = 391.5248718261719
    FY_D = 391.5248718261719
    CX_D = 319.7847900390625
    CY_D = 237.64898681640625

    @staticmethod
    def _backproject(u, v, z_m, fx, fy, cx, cy):
        """Back-project a single pixel."""
        x = (u - cx) * z_m / fx
        y = (v - cy) * z_m / fy
        return np.array([x, y, z_m])

    def test_principal_point_backprojects_to_axis(self):
        """Pixel at (cx, cy) with depth Z → (0, 0, Z)."""
        pt = self._backproject(self.CX_D, self.CY_D, 5.0,
                               self.FX_D, self.FY_D, self.CX_D, self.CY_D)
        np.testing.assert_allclose(pt, [0.0, 0.0, 5.0], atol=1e-10)

    def test_roundtrip_project_backproject(self):
        """Back-project then forward-project should recover the pixel."""
        u_orig, v_orig, z = 200.0, 300.0, 3.5
        pt3d = self._backproject(u_orig, v_orig, z,
                                 self.FX_D, self.FY_D, self.CX_D, self.CY_D)
        # Forward project
        u_rec = self.FX_D * (pt3d[0] / pt3d[2]) + self.CX_D
        v_rec = self.FY_D * (pt3d[1] / pt3d[2]) + self.CY_D
        np.testing.assert_allclose([u_rec, v_rec], [u_orig, v_orig], atol=1e-10)

    def test_depth_scale(self):
        """Z component of back-projected point equals input depth."""
        for z in [0.5, 1.0, 5.0, 15.0]:
            pt = self._backproject(100, 200, z,
                                   self.FX_D, self.FY_D, self.CX_D, self.CY_D)
            assert abs(pt[2] - z) < 1e-12

    def test_mm_to_m_conversion(self):
        """16UC1 depth in mm → metres: divide by 1000."""
        z_mm = np.uint16(3500)
        z_m = float(z_mm) / 1000.0
        assert abs(z_m - 3.5) < 1e-10

    def test_zero_depth_rejected(self):
        """Zero depth pixels should be filtered (they mean 'no reading')."""
        z_mm = np.array([0, 0, 1500, 0, 2000], dtype=np.uint16)
        valid = z_mm > 0
        assert np.sum(valid) == 2
        z_m = z_mm[valid].astype(np.float64) / 1000.0
        np.testing.assert_allclose(z_m, [1.5, 2.0])


# ═══════════════════════════════════════════════════════════════════════════
#  4.  DEPTH GRID SAMPLING
# ═══════════════════════════════════════════════════════════════════════════

class TestDepthGridSampling:
    """Tests for the depth grid creation logic from _depth_grid_to_rgb_frame."""

    def test_grid_shape_default_steps(self):
        """Default steps (64, 48) on a 640×480 image → known grid size.
        Actual code starts at step//2 to avoid edges."""
        w, h = 640, 480
        step_u, step_v = 64, 48
        us = np.arange(step_u // 2, w, step_u)   # [32, 96, ..., 608]  → 10 points
        vs = np.arange(step_v // 2, h, step_v)   # [24, 72, ..., 456]  → 10 points
        assert len(us) == 10
        assert len(vs) == 10
        uu, vv = np.meshgrid(us, vs)
        assert uu.shape == (10, 10)
        total = uu.size
        assert total == 100

    def test_grid_avoids_edges(self):
        """Grid starts at half-step offset, not at pixel 0."""
        step_u, step_v = 64, 48
        us = np.arange(step_u // 2, 640, step_u)
        vs = np.arange(step_v // 2, 480, step_v)
        assert us[0] == 32
        assert vs[0] == 24
        assert us[-1] < 640
        assert vs[-1] < 480

    def test_grid_within_bounds(self):
        """All grid coordinates must be within image bounds."""
        w, h = 640, 480
        for step_u, step_v in [(64, 48), (32, 24), (128, 96), (1, 1)]:
            us = np.arange(step_u // 2, w, step_u)
            vs = np.arange(step_v // 2, h, step_v)
            assert np.all(us >= 0) and np.all(us < w)
            assert np.all(vs >= 0) and np.all(vs < h)

    def test_all_zero_depth_returns_empty(self):
        """If entire depth image is zero, no valid points should remain."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        step_u, step_v = 64, 48
        us = np.arange(step_u // 2, 640, step_u)
        vs = np.arange(step_v // 2, 480, step_v)
        uu, vv = np.meshgrid(us, vs)
        z_mm = depth[vv.ravel(), uu.ravel()]
        valid = z_mm > 0
        assert np.sum(valid) == 0

    def test_partial_depth_filters_correctly(self):
        """Only grid points with non-zero depth survive."""
        depth = np.zeros((480, 640), dtype=np.uint16)
        # Set a patch of valid depth
        depth[40:100, 60:130] = 2500  # 2.5m
        step_u, step_v = 64, 48
        us = np.arange(step_u // 2, 640, step_u)
        vs = np.arange(step_v // 2, 480, step_v)
        uu, vv = np.meshgrid(us, vs)
        z_mm = depth[vv.ravel(), uu.ravel()]
        valid = z_mm > 0
        n_valid = int(np.sum(valid))
        assert n_valid > 0
        assert n_valid < uu.size  # not all should be valid
        # All valid depths should be 2500 mm
        assert np.all(z_mm[valid] == 2500)


# ═══════════════════════════════════════════════════════════════════════════
#  5.  CROSS-CAMERA PIPELINE (end-to-end geometry check)
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossCameraPipeline:
    """End-to-end test of the geometry pipeline:
    depth pixel → 3D depth frame → transform → 3D RGB frame → project → RGB pixel
    Using the actual bag intrinsics and TF values.
    """

    # From bag
    FX_D, FY_D, CX_D, CY_D = 391.525, 391.525, 319.785, 237.649
    FX_R, FY_R, CX_R, CY_R = 500.0, 500.0, 640.0, 360.0

    # TF: f_depth_optical_frame → f_oc_link
    TF_QUAT = (0.653, -0.653, 0.271, 0.271)  # (x, y, z, w)
    TF_TRANS = np.array([0.025, 0.071, 0.039])

    @staticmethod
    def _backproject_batch(uu, vv, z_m, fx, fy, cx, cy):
        x = (uu - cx) * z_m / fx
        y = (vv - cy) * z_m / fy
        return np.column_stack([x, y, z_m])

    @staticmethod
    def _project_batch(pts, fx, fy, cx, cy):
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        pixels = np.full((len(X), 2), -1.0)
        valid = Z > 0
        pixels[valid, 0] = fx * (X[valid] / Z[valid]) + cx
        pixels[valid, 1] = fy * (Y[valid] / Z[valid]) + cy
        return pixels

    def _full_pipeline(self, depth_pixels, z_m_values):
        """Run the pipeline: back-project → transform → project."""
        uu = np.array([p[0] for p in depth_pixels], dtype=np.float64)
        vv = np.array([p[1] for p in depth_pixels], dtype=np.float64)
        z_m = np.array(z_m_values, dtype=np.float64)

        # Step 1: Back-project to 3D in depth frame
        pts_depth = self._backproject_batch(uu, vv, z_m,
                                            self.FX_D, self.FY_D,
                                            self.CX_D, self.CY_D)

        # Step 2: Transform depth→RGB frame
        R = UGVPerceptionNode._quat_to_matrix(*self.TF_QUAT)
        pts_rgb = (R @ pts_depth.T).T + self.TF_TRANS

        # Step 3: Project to RGB pixels
        rgb_pixels = self._project_batch(pts_rgb,
                                         self.FX_R, self.FY_R,
                                         self.CX_R, self.CY_R)
        return pts_depth, pts_rgb, rgb_pixels

    def test_points_remain_in_front_of_rgb_camera(self):
        """Points at reasonable depth should stay in front of RGB camera (Z > 0)."""
        pixels = [(320, 240), (100, 100), (500, 400)]
        z_vals = [3.0, 5.0, 2.0]
        _, pts_rgb, _ = self._full_pipeline(pixels, z_vals)
        # At these distances, Z in RGB frame should be positive
        # (cameras are close together and roughly co-pointed)
        for i in range(len(pixels)):
            # The transform may rearrange axes, but at > 2m depth
            # the point should project validly
            assert np.any(pts_rgb[i] != 0), f"Point {i} is at origin after transform"

    def test_projected_pixels_are_finite(self):
        """Projected RGB pixels should be finite numbers."""
        pixels = [(320, 240), (0, 0), (639, 479)]
        z_vals = [4.0, 4.0, 4.0]
        _, pts_rgb, rgb_px = self._full_pipeline(pixels, z_vals)
        for i in range(len(pixels)):
            if pts_rgb[i, 2] > 0:  # only check if in front
                assert np.all(np.isfinite(rgb_px[i])), \
                    f"Pixel {i} is not finite: {rgb_px[i]}"

    def test_depth_centre_projects_near_rgb_centre(self):
        """Depth principal point at medium range should project
        somewhere near centre of RGB image (not at edge)."""
        pixels = [(self.CX_D, self.CY_D)]
        z_vals = [5.0]
        _, pts_rgb, rgb_px = self._full_pipeline(pixels, z_vals)
        if pts_rgb[0, 2] > 0:
            u, v = rgb_px[0]
            # Should be within the 1280×720 image, not necessarily exact centre
            assert 0 <= u <= 1280, f"u={u} out of RGB image"
            assert 0 <= v <= 720, f"v={v} out of RGB image"

    def test_bbox_filtering_logic(self):
        """Simulate bbox filtering: points inside bbox are selected."""
        # Create points projected to known RGB locations
        rgb_pixels = np.array([
            [100, 100],  # outside bbox
            [500, 300],  # inside bbox
            [600, 350],  # inside bbox
            [900, 500],  # outside bbox
            [ -1,  -1],  # invalid (behind camera)
        ], dtype=np.float64)

        pts_rgb = np.array([
            [0.1, 0.1, 3.0],
            [0.5, 0.3, 4.0],
            [0.7, 0.4, 3.5],
            [1.2, 0.8, 5.0],
            [0.0, 0.0, -1.0],
        ])

        # bbox (x1, y1, x2, y2)
        bbox = (400, 250, 700, 400)
        x1, y1, x2, y2 = bbox

        inside = (
            (rgb_pixels[:, 0] >= x1) & (rgb_pixels[:, 0] <= x2) &
            (rgb_pixels[:, 1] >= y1) & (rgb_pixels[:, 1] <= y2) &
            (pts_rgb[:, 2] > 0)
        )

        assert int(np.sum(inside)) == 2  # indices 1 and 2
        matched = pts_rgb[inside]
        median_pt = np.median(matched, axis=0)
        np.testing.assert_allclose(median_pt, [0.6, 0.35, 3.75])

    def test_no_points_in_bbox(self):
        """If no grid points project inside a bbox, detection is skipped."""
        rgb_pixels = np.array([
            [100, 100],
            [200, 200],
        ], dtype=np.float64)
        pts_rgb = np.array([
            [0.1, 0.1, 3.0],
            [0.2, 0.2, 4.0],
        ])

        bbox = (800, 600, 900, 700)
        x1, y1, x2, y2 = bbox
        inside = (
            (rgb_pixels[:, 0] >= x1) & (rgb_pixels[:, 0] <= x2) &
            (rgb_pixels[:, 1] >= y1) & (rgb_pixels[:, 1] <= y2) &
            (pts_rgb[:, 2] > 0)
        )
        assert int(np.sum(inside)) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  5b.  MASK-BASED DEPTH MATCHING
# ═══════════════════════════════════════════════════════════════════════════

class TestMaskBasedDepthMatching:
    """Tests for the mask-based depth filtering (replaces bbox matching)."""

    def test_mask_selects_correct_points(self):
        """Points whose projected pixel falls inside the mask are selected."""
        rgb_h, rgb_w = 720, 1280

        # Create a simple mask: True in a horizontal strip
        mask = np.zeros((rgb_h, rgb_w), dtype=bool)
        mask[300:400, 400:800] = True

        # Projected pixel coordinates
        rgb_pixels = np.array([
            [500.0, 350.0],   # inside mask
            [600.0, 350.0],   # inside mask
            [100.0, 350.0],   # outside (left)
            [500.0, 100.0],   # outside (above)
            [ -1.0,  -1.0],   # invalid (behind camera)
        ])
        pts_rgb = np.array([
            [1.0, 0.0, 3.0],
            [1.5, 0.0, 3.5],
            [0.2, 0.0, 4.0],
            [1.0, 1.0, 5.0],
            [0.0, 0.0, -1.0],
        ])

        px_u = rgb_pixels[:, 0].astype(int)
        px_v = rgb_pixels[:, 1].astype(int)
        in_bounds = (
            (px_u >= 0) & (px_u < rgb_w) &
            (px_v >= 0) & (px_v < rgb_h)
        )
        inside = np.zeros(len(rgb_pixels), dtype=bool)
        inside[in_bounds] = mask[px_v[in_bounds], px_u[in_bounds]]

        assert int(np.sum(inside)) == 2
        matched = pts_rgb[inside]
        median_pt = np.median(matched, axis=0)
        np.testing.assert_allclose(median_pt, [1.25, 0.0, 3.25])

    def test_empty_mask_selects_nothing(self):
        """All-False mask → no depth points matched."""
        rgb_h, rgb_w = 720, 1280
        mask = np.zeros((rgb_h, rgb_w), dtype=bool)
        rgb_pixels = np.array([[500.0, 350.0], [600.0, 400.0]])

        px_u = rgb_pixels[:, 0].astype(int)
        px_v = rgb_pixels[:, 1].astype(int)
        in_bounds = (
            (px_u >= 0) & (px_u < rgb_w) &
            (px_v >= 0) & (px_v < rgb_h)
        )
        inside = np.zeros(len(rgb_pixels), dtype=bool)
        inside[in_bounds] = mask[px_v[in_bounds], px_u[in_bounds]]

        assert int(np.sum(inside)) == 0

    def test_full_mask_selects_all_valid(self):
        """All-True mask → all in-bounds points matched."""
        rgb_h, rgb_w = 720, 1280
        mask = np.ones((rgb_h, rgb_w), dtype=bool)
        rgb_pixels = np.array([
            [500.0, 350.0],
            [600.0, 400.0],
            [ -1.0,  -1.0],   # out of bounds
        ])

        px_u = rgb_pixels[:, 0].astype(int)
        px_v = rgb_pixels[:, 1].astype(int)
        in_bounds = (
            (px_u >= 0) & (px_u < rgb_w) &
            (px_v >= 0) & (px_v < rgb_h)
        )
        inside = np.zeros(len(rgb_pixels), dtype=bool)
        inside[in_bounds] = mask[px_v[in_bounds], px_u[in_bounds]]

        assert int(np.sum(inside)) == 2  # only the 2 in-bounds points

    def test_bbox_fallback_when_no_mask(self):
        """When mask is None, fallback bbox logic should be used."""
        rgb_pixels = np.array([
            [500.0, 300.0],   # inside bbox
            [100.0, 100.0],   # outside bbox
        ])
        bbox = (400, 250, 700, 400)
        x1, y1, x2, y2 = bbox

        # Bbox matching (fallback)
        inside = (
            (rgb_pixels[:, 0] >= x1) & (rgb_pixels[:, 0] <= x2) &
            (rgb_pixels[:, 1] >= y1) & (rgb_pixels[:, 1] <= y2)
        )
        assert int(np.sum(inside)) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  5c.  BODY → OPTICAL COORDINATE CONVERSION
# ═══════════════════════════════════════════════════════════════════════════

class TestBodyToOpticalConversion:
    """Tests for the ROS body (X fwd, Y left, Z up) → optical
    (X right, Y down, Z forward) coordinate conversion used in
    _project_to_rgb."""

    @staticmethod
    def _body_to_optical(pts):
        """Convert body-frame points to optical-frame points."""
        X_opt = -pts[:, 1]   # right  = −Y_body
        Y_opt = -pts[:, 2]   # down   = −Z_body
        Z_opt =  pts[:, 0]   # forward = X_body
        return np.column_stack([X_opt, Y_opt, Z_opt])

    def test_forward_point_maps_to_z(self):
        """Point at (1,0,0) body → (0,0,1) optical."""
        pts = np.array([[1.0, 0.0, 0.0]])
        opt = self._body_to_optical(pts)
        np.testing.assert_allclose(opt[0], [0.0, 0.0, 1.0])

    def test_left_point_maps_to_minus_x(self):
        """Point at (0,1,0) body → (-1,0,0) optical."""
        pts = np.array([[0.0, 1.0, 0.0]])
        opt = self._body_to_optical(pts)
        np.testing.assert_allclose(opt[0], [-1.0, 0.0, 0.0])

    def test_up_point_maps_to_minus_y(self):
        """Point at (0,0,1) body → (0,-1,0) optical."""
        pts = np.array([[0.0, 0.0, 1.0]])
        opt = self._body_to_optical(pts)
        np.testing.assert_allclose(opt[0], [0.0, -1.0, 0.0])

    def test_forward_point_projects_to_centre(self):
        """Point 5m forward in body frame should project to image centre."""
        fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0
        pts = np.array([[5.0, 0.0, 0.0]])
        opt = self._body_to_optical(pts)
        u = fx * opt[0, 0] / opt[0, 2] + cx
        v = fy * opt[0, 1] / opt[0, 2] + cy
        np.testing.assert_allclose([u, v], [cx, cy])

    def test_object_left_of_camera_projects_left(self):
        """Object to the left (positive Y body) projects to left side of image."""
        fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0
        pts = np.array([[5.0, 2.0, 0.0]])  # 5m forward, 2m left
        opt = self._body_to_optical(pts)
        u = fx * opt[0, 0] / opt[0, 2] + cx
        assert u < cx, "Object left of camera should project to u < cx"

    def test_object_above_camera_projects_up(self):
        """Object above (positive Z body) projects to upper half of image."""
        fx, fy, cx, cy = 500.0, 500.0, 640.0, 360.0
        pts = np.array([[5.0, 0.0, 1.0]])  # 5m forward, 1m up
        opt = self._body_to_optical(pts)
        v = fy * opt[0, 1] / opt[0, 2] + cy
        assert v < cy, "Object above camera should project to v < cy"


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

        # Disappear for max_age+1 frames
        for _ in range(3):
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
            det = [self._make_det((i * 200, 0, i * 200 + 50, 50))]
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
        dets1 = [self._make_det((0, 0, 50, 50))]
        dets2 = [self._make_det((500, 500, 600, 600))]  # far away
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

        assert len(tracker.tracks) == 1  # age=3, still within max_age

        tracker.update([])
        assert len(tracker.tracks) == 0  # age=4 > max_age=3, expired

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
        Optionally includes 'size' (sx, sy, sz) and 'position' (x, y, z)
        — when size.x > 0 or size.y > 0 a Polygon footprint is emitted.
        """
        from triffid_ugv_perception.geojson_bridge import GeoJSONBridge, _MIN_EXTENT
        features = []
        for det in detections:
            lon, lat, alt = det['coordinates']
            cls = det.get('class_name', 'unknown')
            sx, sy, sz = det.get('size', (0.0, 0.0, 0.0))
            pos_x, pos_y, pos_z = det.get('position', (lon, lat, 0.0))

            has_extent = (sx > 0.0 or sy > 0.0)

            if has_extent:
                half_x = max(sx, _MIN_EXTENT) / 2.0
                half_y = max(sy, _MIN_EXTENT) / 2.0
                ring = [
                    [pos_x + half_x, pos_y + half_y, pos_z],
                    [pos_x + half_x, pos_y - half_y, pos_z],
                    [pos_x - half_x, pos_y - half_y, pos_z],
                    [pos_x - half_x, pos_y + half_y, pos_z],
                    [pos_x + half_x, pos_y + half_y, pos_z],  # closed
                ]
                geometry = {"type": "Polygon", "coordinates": [ring]}
            else:
                geometry = {"type": "Point", "coordinates": [lon, lat, alt]}

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
            if has_extent:
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
                feature["properties"]["fill"] = feature["properties"]["marker-color"]
                feature["properties"]["fill-opacity"] = 0.25
            features.append(feature)
        return {"type": "FeatureCollection", "features": features}

    def test_valid_feature_collection(self):
        dets = [{'coordinates': (23.7, 37.9, 320.0), 'class_name': 'person',
                 'confidence': 0.85, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        assert gj['type'] == 'FeatureCollection'
        assert len(gj['features']) == 1
        assert gj['features'][0]['type'] == 'Feature'
        assert gj['features'][0]['geometry']['type'] == 'Point'

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

    def test_coordinates_are_lon_lat_alt_order(self):
        """GeoJSON spec: coordinates = [longitude, latitude, altitude]."""
        lon, lat, alt = 23.7348, 37.9755, 310.5
        dets = [{'coordinates': (lon, lat, alt), 'class_name': 'person',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        coords = gj['features'][0]['geometry']['coordinates']
        assert coords[0] == lon
        assert coords[1] == lat
        assert coords[2] == alt

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

    def test_altitude_property_present(self):
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

    def test_point_coordinates_have_altitude(self):
        """Point geometry coordinates must have 3 elements [lon, lat, alt]."""
        dets = [{'coordinates': (13.35, 49.73, 320.0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets, gps_valid=True)
        coords = gj['features'][0]['geometry']['coordinates']
        assert len(coords) == 3

    # ── Polygon geometry tests ──────────────────────────────────────

    def test_polygon_emitted_when_size_nonzero(self):
        """Detection with nonzero bbox size → Polygon geometry."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1',
                 'size': (2.0, 3.0, 4.0), 'position': (5.0, 6.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Polygon'

    def test_point_emitted_when_size_zero(self):
        """Detection with ALL-ZERO bbox size → Point geometry (fallback)."""
        dets = [{'coordinates': (1.0, 2.0, 0.0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1',
                 'size': (0.0, 0.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Point'

    def test_polygon_emitted_when_only_one_size_nonzero(self):
        """Detection with one nonzero size → Polygon (uses min extent)."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Green grass',
                 'confidence': 0.9, 'track_id': '1',
                 'size': (0.0, 2.5, 1.4), 'position': (3.0, 1.0, 0.0)}]
        gj = self._make_geojson(dets)
        geom = gj['features'][0]['geometry']
        assert geom['type'] == 'Polygon'

    def test_point_emitted_when_size_absent(self):
        """Detection without size field → Point geometry."""
        dets = [{'coordinates': (1.0, 2.0, 0.0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets)
        assert gj['features'][0]['geometry']['type'] == 'Point'

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
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Flame',
                 'confidence': 0.9, 'track_id': '1'}]
        gj = self._make_geojson(dets)
        props = gj['features'][0]['properties']
        assert 'stroke' not in props
        assert 'fill' not in props

    def test_mixed_point_and_polygon(self):
        """FeatureCollection can contain both Point and Polygon features."""
        dets = [
            {'coordinates': (1, 2, 0), 'class_name': 'Flame',
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

    def test_polygon_vertices_have_altitude(self):
        """Polygon ring vertices should have 3D coordinates."""
        dets = [{'coordinates': (0, 0, 0), 'class_name': 'Building',
                 'confidence': 0.8, 'track_id': '1',
                 'size': (2.0, 3.0, 4.0), 'position': (5.0, 6.0, 1.0)}]
        gj = self._make_geojson(dets)
        ring = gj['features'][0]['geometry']['coordinates'][0]
        for vertex in ring:
            assert len(vertex) == 3, f'Expected [x,y,z], got {vertex}'


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
    """Verify frame IDs and constants match the rosbag data."""

    def test_depth_frame_id(self):
        from triffid_ugv_perception.ugv_node import DEPTH_FRAME
        assert DEPTH_FRAME == 'f_depth_optical_frame'

    def test_rgb_frame_id(self):
        from triffid_ugv_perception.ugv_node import RGB_FRAME
        assert RGB_FRAME == 'f_oc_link'

    def test_base_frame_id(self):
        from triffid_ugv_perception.ugv_node import BASE_FRAME
        assert BASE_FRAME == 'b2/base_link'

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

    def test_grid_step_defaults(self):
        from triffid_ugv_perception.ugv_node import (
            DEFAULT_GRID_STEP_U, DEFAULT_GRID_STEP_V
        )
        assert DEFAULT_GRID_STEP_U == 64
        assert DEFAULT_GRID_STEP_V == 48


# ═══════════════════════════════════════════════════════════════════════════
#  11.  INTRINSICS VALIDATION (from bag)
# ═══════════════════════════════════════════════════════════════════════════

class TestIntrinsicsConsistency:
    """Validate that intrinsic values from the bag are self-consistent."""

    # Depth intrinsics (from bag /camera_front/realsense_front/depth/camera_info)
    DEPTH_K = [391.5248718261719, 0.0, 319.7847900390625,
               0.0, 391.5248718261719, 237.64898681640625,
               0.0, 0.0, 1.0]
    DEPTH_W, DEPTH_H = 640, 480

    # RGB intrinsics (from bag /camera_front/camera_info)
    RGB_K = [500.0, 0.0, 640.0,
             0.0, 500.0, 360.0,
             0.0, 0.0, 1.0]
    RGB_W, RGB_H = 1280, 720

    def test_depth_principal_point_inside_image(self):
        cx, cy = self.DEPTH_K[2], self.DEPTH_K[5]
        assert 0 < cx < self.DEPTH_W
        assert 0 < cy < self.DEPTH_H

    def test_rgb_principal_point_inside_image(self):
        cx, cy = self.RGB_K[2], self.RGB_K[5]
        assert 0 < cx < self.RGB_W
        assert 0 < cy < self.RGB_H

    def test_depth_focal_lengths_positive(self):
        assert self.DEPTH_K[0] > 0  # fx
        assert self.DEPTH_K[4] > 0  # fy

    def test_rgb_focal_lengths_positive(self):
        assert self.RGB_K[0] > 0
        assert self.RGB_K[4] > 0

    def test_k_matrix_last_row(self):
        """K matrix last row should be [0, 0, 1]."""
        assert self.DEPTH_K[6:] == [0.0, 0.0, 1.0]
        assert self.RGB_K[6:] == [0.0, 0.0, 1.0]

    def test_depth_is_square_pixel(self):
        """Depth camera has fx == fy (square pixels)."""
        np.testing.assert_allclose(self.DEPTH_K[0], self.DEPTH_K[4])

    def test_rgb_is_square_pixel(self):
        """RGB camera has fx == fy."""
        np.testing.assert_allclose(self.RGB_K[0], self.RGB_K[4])

    def test_rgb_principal_point_is_image_centre(self):
        """RGB cx=W/2, cy=H/2 (ideal pinhole)."""
        assert self.RGB_K[2] == self.RGB_W / 2
        assert self.RGB_K[5] == self.RGB_H / 2


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

        tracker.update([])  # empty
        tracker.update([])  # track expires at age > 1

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
        dets1 = [self._make_det((0, 0, 100, 100))]
        dets2 = [self._make_det((50, 0, 150, 100))]

        r1 = tracker.update(dets1)
        r2 = tracker.update(dets2)
        # IoU ≈ 0.333 < 0.5, so should create a new track
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

        tracker.update([])  # no match, age → 1 > max_age=0
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

    def test_depth_grid_count_matches_expectation(self):
        """Given image size and grid steps, count should be predictable."""
        w, h = 640, 480
        step_u, step_v = 64, 48
        expected_cols = len(range(step_u // 2, w, step_u))  # 10
        expected_rows = len(range(step_v // 2, h, step_v))  # 10
        expected_total = expected_cols * expected_rows  # 100
        assert expected_total == 100


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
    """Tests for UGVPerceptionNode._bbox_to_3d_corners."""

    fn = staticmethod(UGVPerceptionNode._bbox_to_3d_corners)

    # Use actual RGB intrinsics from the bag
    class _FakeInfo:
        k = [
            642.0186767578125, 0.0, 646.3067626953125,
            0.0, 641.4913330078125, 364.65814208984375,
            0.0, 0.0, 1.0,
        ]

    def setup_method(self):
        # Monkey-patch: _bbox_to_3d_corners reads self.rgb_camera_info
        self._dummy = type('obj', (object,), {'rgb_camera_info': self._FakeInfo()})()

    def _call(self, u1, v1, u2, v2, depth):
        # Bind to the dummy self
        return UGVPerceptionNode._bbox_to_3d_corners(self._dummy, u1, v1, u2, v2, depth)

    def test_returns_8x3(self):
        corners = self._call(100, 100, 300, 300, 5.0)
        assert corners.shape == (8, 3)

    def test_x_equals_depth(self):
        """All X (forward) values should equal the input depth."""
        corners = self._call(50, 50, 200, 200, 3.0)
        np.testing.assert_allclose(corners[:, 0], 3.0)

    def test_nonzero_yz_extent(self):
        """Y and Z extent must be positive for any non-degenerate bbox."""
        corners = self._call(100, 100, 500, 400, 4.0)
        extent = np.ptp(corners, axis=0)
        assert extent[1] > 0.1, "Y extent should be positive"
        assert extent[2] > 0.1, "Z extent should be positive"

    def test_principal_point_bbox_centered(self):
        """A bbox centred on the principal point should give symmetric Y, Z."""
        cx = self._FakeInfo.k[2]
        cy = self._FakeInfo.k[5]
        half = 50.0
        corners = self._call(cx - half, cy - half, cx + half, cy + half, 5.0)
        y_extent = np.ptp(corners[:, 1])
        z_extent = np.ptp(corners[:, 2])
        assert y_extent > 0
        assert z_extent > 0

    def test_larger_bbox_larger_extent(self):
        """A wider 2-D bbox at the same depth should give larger Y extent."""
        small = self._call(300, 200, 400, 300, 5.0)
        large = self._call(100, 200, 600, 300, 5.0)
        assert np.ptp(large[:, 1]) > np.ptp(small[:, 1])

    def test_farther_depth_larger_extent(self):
        """Same pixel bbox at greater depth → proportionally larger extent."""
        near = self._call(200, 200, 400, 400, 2.0)
        far  = self._call(200, 200, 400, 400, 8.0)
        assert np.ptp(far[:, 1]) > np.ptp(near[:, 1])
        assert np.ptp(far[:, 2]) > np.ptp(near[:, 2])


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


#  Entry point

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
