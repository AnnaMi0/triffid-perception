#!/usr/bin/env python3
"""
Collect one sample of every output topic and save raw RGB frames.

Usage (inside Docker):
    python3 /ws/src/triffid_ugv_perception/scripts/collect_samples.py

Output directory: /ws/samples/  (mounted or inside container)

Saves:
  - rgb_frame_0.jpg … rgb_frame_4.jpg   (5 raw RGB frames from the bag)
  - detections_3d.yaml                   (1 Detection3DArray message)
  - segmentation.png                     (1 semantic label map, mono8)
  - geojson.json                         (1 GeoJSON FeatureCollection)
  - geojson_raw.json                     (all confirmed tracks, one per ID,
                                          no spatial deduplication)
  - geojson_merged.json                  (same as raw but spatially
                                          deduplicated by class + proximity)
  - mqtt_trace.jsonl                     (every MQTT GeoJSON message,
                                          one JSON object per line)
    - tracking_debug.mp4                   (RGB debug overlay per frame)
    - track_lifecycle.csv                  (lifetime statistics per track ID)
    - possible_id_switches.csv             (likely ID-switch events)

Exit: automatically stops after all samples are collected, or after --timeout.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

try:
    import paho.mqtt.client as paho_mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False

OUTDIR = '/ws/samples'
NUM_RGB_FRAMES = 5

# Spatial deduplication: merge features of the same class whose
# centroids are closer than this (metres).  1.0 m covers GPS jitter +
# slight tracker re-ID shifts without merging genuinely separate objects.
_MERGE_RADIUS_M = 1.0
_R_EARTH = 6_378_137.0   # WGS-84 semi-major axis

bridge = CvBridge()


class SampleCollector(Node):
    def __init__(self, outdir: str, n_rgb: int):
        super().__init__('sample_collector')
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)

        self.n_rgb_target = n_rgb
        self.n_rgb_saved = 0
        self.got_det = False
        self.got_seg = False
        self.got_geo = False

        # Accumulate all GeoJSON features for the merged output.
        # Keyed by track ID — keeps the highest-confidence version.
        self._merged_features = {}   # id -> feature dict
        self._geo_msg_count = 0

        # MQTT trace capture
        self._mqtt_trace_file = None     # opened lazily
        self._mqtt_msg_count = 0
        self._mqtt_client = None

        # Optional debug video from /ugv/detections/front/debug_image
        self._video_writer = None
        self._video_path = os.path.join(self.outdir, 'tracking_debug.mp4')
        self._video_frames = 0

        # Per-track lifecycle stats inferred from GeoJSON stream
        self._track_stats = {}

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Raw RGB frames from rosbag
        self.create_subscription(
            Image,
            '/camera_front/raw_image',
            self._cb_rgb,
            sensor_qos,
        )

        # Detection3DArray output
        self.create_subscription(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            self._cb_det,
            10,
        )

        # Segmentation label map (Image mono8)
        self.create_subscription(
            Image,
            '/ugv/detections/front/segmentation',
            self._cb_seg,
            sensor_qos,
        )

        # GeoJSON (String)
        self.create_subscription(
            String,
            '/ugv/detections/front/geojson',
            self._cb_geo,
            10,
        )

        # RGB debug image with class + track ID overlay
        self.create_subscription(
            Image,
            '/ugv/detections/front/debug_image',
            self._cb_debug,
            sensor_qos,
        )

        self.get_logger().info(f'Sample collector started — saving to {outdir}')

    def start_mqtt_trace(self, host='localhost', port=1883,
                         topic='ugv/detections/front/geojson'):
        """Connect to local MQTT broker and record every message."""
        if not _PAHO_AVAILABLE:
            self.get_logger().warn('paho-mqtt not installed — MQTT trace disabled')
            return
        path = os.path.join(self.outdir, 'mqtt_trace.jsonl')
        self._mqtt_trace_file = open(path, 'w')
        self._mqtt_topic = topic

        def on_connect(client, userdata, flags, rc, properties=None):
            if rc == 0:
                client.subscribe(topic, qos=0)
                self.get_logger().info(f'MQTT connected — subscribed to {topic}')
            else:
                self.get_logger().warn(f'MQTT connect failed (rc={rc})')

        def on_message(client, userdata, msg, properties=None):
            try:
                payload = msg.payload.decode('utf-8')
                # Write one JSON object per line (JSONL)
                self._mqtt_trace_file.write(payload + '\n')
                self._mqtt_msg_count += 1
            except Exception:
                pass

        self._mqtt_client = paho_mqtt.Client(
            paho_mqtt.CallbackAPIVersion.VERSION2,
            client_id='sample_mqtt_trace',
        )
        self._mqtt_client.on_connect = on_connect
        self._mqtt_client.on_message = on_message
        try:
            self._mqtt_client.connect(host, port, keepalive=60)
            self._mqtt_client.loop_start()
        except Exception as e:
            self.get_logger().warn(f'MQTT broker not reachable ({e}) — trace disabled')
            self._mqtt_trace_file.close()
            self._mqtt_trace_file = None
            self._mqtt_client = None

    # ------------------------------------------------------------------

    def _cb_rgb(self, msg: Image):
        if self.n_rgb_saved >= self.n_rgb_target:
            return
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            path = os.path.join(self.outdir, f'rgb_frame_{self.n_rgb_saved}.jpg')
            cv2.imwrite(path, cv_img)
            self.get_logger().info(f'Saved RGB frame → {path}')
            self.n_rgb_saved += 1
        except Exception as e:
            self.get_logger().error(f'RGB save error: {e}')

    def _cb_det(self, msg: Detection3DArray):
        if self.got_det:
            return
        # Skip empty frames so the sample is meaningful and matches
        # the GeoJSON output (which also skips empty frames).
        if len(msg.detections) == 0:
            return
        path = os.path.join(self.outdir, 'detections_3d.yaml')
        lines = []
        lines.append(f'header:')
        lines.append(f'  stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}')
        lines.append(f'  frame_id: "{msg.header.frame_id}"')
        lines.append(f'detections: ({len(msg.detections)} total)')
        for i, det in enumerate(msg.detections):
            pos = det.bbox.center.position
            lines.append(f'  [{i}]:')
            lines.append(f'    id: "{det.id}"')
            if det.results:
                hyp = det.results[0].hypothesis
                lines.append(f'    class: "{hyp.class_id}"')
                lines.append(f'    score: {hyp.score:.4f}')
            lines.append(f'    position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')
            sz = det.bbox.size
            lines.append(f'    size: ({sz.x:.3f}, {sz.y:.3f}, {sz.z:.3f})')
        with open(path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        self.get_logger().info(f'Saved detections → {path} ({len(msg.detections)} dets)')
        self.got_det = True

    def _cb_seg(self, msg: Image):
        if self.got_seg:
            return
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            path = os.path.join(self.outdir, 'segmentation.png')
            cv2.imwrite(path, cv_img)
            self.get_logger().info(f'Saved segmentation label map → {path}')
            self.got_seg = True
        except Exception as e:
            self.get_logger().error(f'Seg save error: {e}')

    def _cb_geo(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        frame_idx = self._geo_msg_count

        # Accumulate features for the merged output
        for feat in data.get('features', []):
            fid = feat.get('id', '')
            new_conf = feat.get('properties', {}).get('confidence', 0.0)
            existing = self._merged_features.get(fid)
            if existing is None or new_conf > existing.get('properties', {}).get('confidence', 0.0):
                self._merged_features[fid] = feat
            self._update_track_stats(feat, frame_idx)
        self._geo_msg_count += 1

        # Save first non-empty message as the single-frame sample
        if not self.got_geo:
            path = os.path.join(self.outdir, 'geojson.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'Saved GeoJSON → {path}')
            self.got_geo = True

    def _cb_debug(self, msg: Image):
        """Append debug overlay frames into tracking_debug.mp4."""
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        if self._video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(
                self._video_path, fourcc, 10.0, (w, h)
            )
            if not self._video_writer.isOpened():
                self._video_writer = None
                self.get_logger().warn(
                    'Could not open tracking_debug.mp4 for writing'
                )
                return

        self._video_writer.write(frame)
        self._video_frames += 1

    def _update_track_stats(self, feat: dict, frame_idx: int):
        """Update per-track lifecycle stats from one GeoJSON feature."""
        fid = str(feat.get('id', ''))
        if not fid:
            return

        props = feat.get('properties', {})
        cls = props.get('class', 'unknown')
        conf = float(props.get('confidence', 0.0))
        lon, lat = self._feature_centroid(feat.get('geometry', {}))

        st = self._track_stats.get(fid)
        if st is None:
            st = {
                'id': fid,
                'class': cls,
                'first_msg': frame_idx,
                'last_msg': frame_idx,
                'seen_count': 0,
                'max_conf': conf,
                'sum_lon': 0.0,
                'sum_lat': 0.0,
                'n_centroid': 0,
            }
            self._track_stats[fid] = st

        st['last_msg'] = frame_idx
        st['seen_count'] += 1
        st['max_conf'] = max(st['max_conf'], conf)
        st['class'] = cls
        if lon is not None and lat is not None:
            st['sum_lon'] += lon
            st['sum_lat'] += lat
            st['n_centroid'] += 1

    @staticmethod
    def _feature_centroid(geom: dict):
        gtype = geom.get('type')
        if gtype == 'Point':
            coords = geom.get('coordinates', [])
            if len(coords) >= 2:
                return float(coords[0]), float(coords[1])
            return None, None
        if gtype == 'LineString':
            pts = geom.get('coordinates', [])
        elif gtype == 'Polygon':
            rings = geom.get('coordinates', [])
            pts = rings[0] if rings else []
            if len(pts) > 1:
                pts = pts[:-1]
        else:
            pts = []
        if not pts:
            return None, None
        lon = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        return float(lon), float(lat)

    # ------------------------------------------------------------------
    #  Spatial deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        """Haversine distance in metres between two (lon, lat) pairs."""
        rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(rlat1) * math.cos(rlat2)
             * math.sin(dlon / 2) ** 2)
        return _R_EARTH * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _spatial_dedup(self, features):
        """Merge features of the same class that are within *_MERGE_RADIUS_M*.

        For each cluster the feature with the highest confidence wins.
        Uses greedy single-link clustering (O(n^2) per class, fine for
        the typical 10–200 features we see).
        """
        from collections import defaultdict

        by_class = defaultdict(list)
        for f in features:
            by_class[f.get('properties', {}).get('class', '')].append(f)

        kept = []
        for cls, class_feats in by_class.items():
            # Compute centroids
            centroids = []
            for f in class_feats:
                lon, lat = self._feature_centroid(f.get('geometry', {}))
                centroids.append((lon, lat))

            # Union-find style greedy clustering
            n = len(class_feats)
            parent = list(range(n))

            def find(i):
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            for i in range(n):
                if centroids[i][0] is None:
                    continue
                for j in range(i + 1, n):
                    if centroids[j][0] is None:
                        continue
                    if find(i) == find(j):
                        continue
                    d = self._haversine(
                        centroids[i][0], centroids[i][1],
                        centroids[j][0], centroids[j][1],
                    )
                    if d <= _MERGE_RADIUS_M:
                        parent[find(j)] = find(i)

            # Pick the highest-confidence feature per cluster
            clusters = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            for idxs in clusters.values():
                best = max(
                    idxs,
                    key=lambda i: class_feats[i]
                        .get('properties', {})
                        .get('confidence', 0.0),
                )
                kept.append(class_feats[best])

        return kept

    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        return (
            self.n_rgb_saved >= self.n_rgb_target
            and self.got_det
            and self.got_seg
            and self.got_geo
        )

    def save_merged_geojson(self):
        """Write the accumulated merged GeoJSON to disk.

        Writes two files:
        - ``geojson_raw.json``: all confirmed tracks (one feature per
          track ID, highest-confidence snapshot each).  No spatial
          deduplication — every ID the tracker confirmed is present.
        - ``geojson_merged.json``: spatially deduplicated version where
          same-class features within ``_MERGE_RADIUS_M`` are fused
          (the higher-confidence feature survives).
        """
        all_features = list(self._merged_features.values())

        # --- raw: every confirmed track, no spatial dedup ---
        raw_path = os.path.join(self.outdir, 'geojson_raw.json')
        raw_fc = {"type": "FeatureCollection", "features": all_features}
        with open(raw_path, 'w') as f:
            json.dump(raw_fc, f, indent=2)

        # --- merged: after spatial dedup ---
        deduped = self._spatial_dedup(all_features)
        merged_path = os.path.join(self.outdir, 'geojson_merged.json')
        merged_fc = {"type": "FeatureCollection", "features": deduped}
        with open(merged_path, 'w') as f:
            json.dump(merged_fc, f, indent=2)

        n_before = len(self._merged_features)
        self.get_logger().info(
            f'Saved raw GeoJSON → {raw_path} ({n_before} tracks)\n'
            f'Saved merged GeoJSON → {merged_path} '
            f'({len(deduped)} features after spatial dedup '
            f'from {n_before} tracks / {self._geo_msg_count} messages)'
        )

    def stop_mqtt_trace(self):
        """Stop MQTT client and close trace file."""
        if self._mqtt_client is not None:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_client = None
        if self._mqtt_trace_file is not None:
            self._mqtt_trace_file.close()
            path = os.path.join(self.outdir, 'mqtt_trace.jsonl')
            self.get_logger().info(
                f'Saved MQTT trace → {path} ({self._mqtt_msg_count} messages)'
            )
            self._mqtt_trace_file = None

    def stop_debug_video(self):
        """Finalize and close debug video writer."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            self.get_logger().info(
                f'Saved debug video → {self._video_path} '
                f'({self._video_frames} frames)'
            )

    @staticmethod
    def _haversine_m(lat1, lon1, lat2, lon2):
        r = 6378137.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = (
            math.sin(dp / 2.0) ** 2
            + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
        )
        return 2.0 * r * math.asin(math.sqrt(a))

    def save_track_debug_reports(self):
        """Write per-track lifecycle and likely ID-switch CSV reports."""
        if not self._track_stats:
            return

        lifecycle_rows = []
        for st in self._track_stats.values():
            n = st['n_centroid']
            mean_lon = (st['sum_lon'] / n) if n else None
            mean_lat = (st['sum_lat'] / n) if n else None
            lifecycle_rows.append({
                'id': st['id'],
                'class': st['class'],
                'first_msg': st['first_msg'],
                'last_msg': st['last_msg'],
                'lifespan_msgs': st['last_msg'] - st['first_msg'] + 1,
                'seen_count': st['seen_count'],
                'max_conf': round(st['max_conf'], 4),
                'mean_lon': mean_lon,
                'mean_lat': mean_lat,
            })

        lifecycle_rows.sort(key=lambda r: (r['class'], int(r['first_msg'])))
        lifecycle_path = os.path.join(self.outdir, 'track_lifecycle.csv')
        with open(lifecycle_path, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=list(lifecycle_rows[0].keys()))
            wr.writeheader()
            wr.writerows(lifecycle_rows)

        by_class = {}
        for row in lifecycle_rows:
            by_class.setdefault(row['class'], []).append(row)

        switch_rows = []
        for cls, rows in by_class.items():
            rows.sort(key=lambda r: int(r['first_msg']))
            for i in range(len(rows) - 1):
                a = rows[i]
                b = rows[i + 1]
                if a['mean_lon'] is None or b['mean_lon'] is None:
                    continue
                gap = int(b['first_msg']) - int(a['last_msg'])
                if gap < 0:
                    continue
                dist_m = self._haversine_m(
                    float(a['mean_lat']), float(a['mean_lon']),
                    float(b['mean_lat']), float(b['mean_lon']),
                )
                # Heuristic for likely one physical object split across IDs.
                if gap <= 30 and dist_m <= 2.0:
                    switch_rows.append({
                        'class': cls,
                        'old_id': a['id'],
                        'new_id': b['id'],
                        'old_last_msg': a['last_msg'],
                        'new_first_msg': b['first_msg'],
                        'gap_msgs': gap,
                        'distance_m': round(dist_m, 3),
                        'old_seen_count': a['seen_count'],
                        'new_seen_count': b['seen_count'],
                    })

        switch_path = os.path.join(self.outdir, 'possible_id_switches.csv')
        with open(switch_path, 'w', newline='') as f:
            fieldnames = [
                'class', 'old_id', 'new_id',
                'old_last_msg', 'new_first_msg',
                'gap_msgs', 'distance_m',
                'old_seen_count', 'new_seen_count',
            ]
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            if switch_rows:
                wr.writerows(switch_rows)

        self.get_logger().info(
            f'Saved track lifecycle → {lifecycle_path} ({len(lifecycle_rows)} IDs)'
        )
        self.get_logger().info(
            f'Saved switch candidates → {switch_path} ({len(switch_rows)} rows)'
        )

    def status(self) -> str:
        parts = []
        parts.append(f'rgb={self.n_rgb_saved}/{self.n_rgb_target}')
        parts.append(f'det={"✓" if self.got_det else "…"}')
        parts.append(f'seg={"✓" if self.got_seg else "…"}')
        parts.append(f'geo={"✓" if self.got_geo else "…"}')
        parts.append(f'merged={self._geo_msg_count}msgs/{len(self._merged_features)}obj')
        parts.append(f'mqtt={self._mqtt_msg_count}')
        parts.append(f'video={self._video_frames}f')
        return '  '.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Collect pipeline output samples')
    parser.add_argument('--outdir', default=OUTDIR, help='Output directory')
    parser.add_argument('--n-rgb', type=int, default=NUM_RGB_FRAMES,
                        help='Number of raw RGB frames to save')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='Max seconds to wait')
    args = parser.parse_args()

    rclpy.init()
    node = SampleCollector(args.outdir, args.n_rgb)
    node.start_mqtt_trace()  # connects to local broker (best-effort)

    t0 = time.monotonic()
    last_status = 0.0
    samples_done = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            elapsed = time.monotonic() - t0
            if elapsed - last_status > 5.0:
                node.get_logger().info(f'[{elapsed:.0f}s] {node.status()}')
                last_status = elapsed

            if not samples_done and node.all_done():
                node.get_logger().info(
                    'Single-frame samples collected! '
                    'Continuing to accumulate merged GeoJSON until timeout...'
                )
                samples_done = True

            if elapsed > args.timeout:
                if not samples_done:
                    node.get_logger().warn(
                        f'Timeout after {args.timeout}s — {node.status()}')
                break
    except KeyboardInterrupt:
        pass

    # Save merged GeoJSON with all accumulated features
    node.save_merged_geojson()
    node.save_track_debug_reports()
    node.stop_mqtt_trace()
    node.stop_debug_video()

    node.get_logger().info(f'Final: {node.status()}')
    node.destroy_node()
    rclpy.shutdown()

    # Print summary
    print(f'\n  Samples saved to: {args.outdir}/')
    for fname in sorted(os.listdir(args.outdir)):
        fpath = os.path.join(args.outdir, fname)
        sz = os.path.getsize(fpath)
        print(f'    {fname:30s}  {sz:>10,} bytes')
    print()


if __name__ == '__main__':
    main()
