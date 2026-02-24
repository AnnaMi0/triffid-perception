"""
IoU-based Tracker for TRIFFID Perception
==========================================
Simple multi-object tracker using Intersection-over-Union (IoU) matching
on 2D bounding boxes. Designed for the PoC stage.

Rules (from TRIFFID spec):
  - IDs are persistent and NEVER reused
  - If an object disappears, its ID is retired
  - Counter never resets
"""

import numpy as np


class IoUTracker:
    """Track objects across frames using IoU matching on 2D bboxes."""

    def __init__(self, iou_threshold=0.3, max_age=10):
        """
        Args:
            iou_threshold: minimum IoU to associate a detection with a track.
            max_age: frames without a match before a track is retired.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1          # never resets, never reused
        self.tracks = []          # list of active track dicts

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of dicts with keys:
                'bbox': (x1, y1, x2, y2)
                'class_id': int
                'class_name': str
                'confidence': float
                'position': (x, y, z) or None  – 3D position

        Returns:
            list of dicts with added 'track_id' key.
        """
        if not detections:
            # Age all tracks
            self._age_tracks()
            return []

        if not self.tracks:
            # First frame: create new tracks for all detections
            results = []
            for det in detections:
                track = self._create_track(det)
                results.append({**det, 'track_id': track['id']})
            return results

        # Compute IoU matrix
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        iou_matrix = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(track['bbox'], det['bbox'])

        # Greedy matching (good enough for PoC)
        matched_tracks = set()
        matched_dets = set()
        results = []

        # Sort by IoU descending for greedy assignment
        indices = np.unravel_index(
            np.argsort(iou_matrix, axis=None)[::-1], iou_matrix.shape
        )

        for ti, di in zip(indices[0], indices[1]):
            if ti in matched_tracks or di in matched_dets:
                continue
            if iou_matrix[ti, di] < self.iou_threshold:
                break  # remaining are below threshold

            # Match found
            track = self.tracks[ti]
            det = detections[di]
            track['bbox'] = det['bbox']
            track['position'] = det.get('position')
            track['class_name'] = det['class_name']
            track['confidence'] = det['confidence']
            track['age'] = 0
            matched_tracks.add(ti)
            matched_dets.add(di)
            results.append({**det, 'track_id': track['id']})

        # Create new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track = self._create_track(det)
                results.append({**det, 'track_id': track['id']})

        # Age unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['age'] += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

        return results

    def _create_track(self, det):
        """Create a new track with a unique ID (never reused)."""
        track = {
            'id': self.next_id,
            'bbox': det['bbox'],
            'position': det.get('position'),
            'class_name': det['class_name'],
            'confidence': det['confidence'],
            'age': 0,
        }
        self.next_id += 1
        self.tracks.append(track)
        return track

    def _age_tracks(self):
        """Increment age on all tracks and remove dead ones."""
        for track in self.tracks:
            track['age'] += 1
        self.tracks = [t for t in self.tracks if t['age'] <= self.max_age]

    @staticmethod
    def _compute_iou(bbox_a, bbox_b):
        """Compute IoU between two bounding boxes (x1, y1, x2, y2)."""
        x1 = max(bbox_a[0], bbox_b[0])
        y1 = max(bbox_a[1], bbox_b[1])
        x2 = min(bbox_a[2], bbox_b[2])
        y2 = min(bbox_a[3], bbox_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
