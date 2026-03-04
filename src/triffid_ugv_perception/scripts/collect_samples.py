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
  - geojson_merged.json                  (all GeoJSON features merged,
                                          deduplicated by track ID)
  - mqtt_trace.jsonl                     (every MQTT GeoJSON message,
                                          one JSON object per line)

Exit: automatically stops after all samples are collected, or after --timeout.
"""
import argparse
import json
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
            '/ugv/perception/front/detections_3d',
            self._cb_det,
            10,
        )

        # Segmentation label map (Image mono8)
        self.create_subscription(
            Image,
            '/ugv/perception/front/segmentation',
            self._cb_seg,
            sensor_qos,
        )

        # GeoJSON (String)
        self.create_subscription(
            String,
            '/triffid/front/geojson',
            self._cb_geo,
            10,
        )

        self.get_logger().info(f'Sample collector started — saving to {outdir}')

    def start_mqtt_trace(self, host='localhost', port=1883,
                         topic='triffid/front/geojson'):
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

        # Accumulate features for the merged output
        for feat in data.get('features', []):
            fid = feat.get('id', '')
            new_conf = feat.get('properties', {}).get('confidence', 0.0)
            existing = self._merged_features.get(fid)
            if existing is None or new_conf > existing.get('properties', {}).get('confidence', 0.0):
                self._merged_features[fid] = feat
        self._geo_msg_count += 1

        # Save first non-empty message as the single-frame sample
        if not self.got_geo:
            path = os.path.join(self.outdir, 'geojson.json')
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'Saved GeoJSON → {path}')
            self.got_geo = True

    # ------------------------------------------------------------------

    def all_done(self) -> bool:
        return (
            self.n_rgb_saved >= self.n_rgb_target
            and self.got_det
            and self.got_seg
            and self.got_geo
        )

    def save_merged_geojson(self):
        """Write the accumulated merged GeoJSON to disk."""
        path = os.path.join(self.outdir, 'geojson_merged.json')
        merged = {
            "type": "FeatureCollection",
            "features": list(self._merged_features.values()),
        }
        with open(path, 'w') as f:
            json.dump(merged, f, indent=2)
        self.get_logger().info(
            f'Saved merged GeoJSON → {path} '
            f'({len(merged["features"])} unique features '
            f'from {self._geo_msg_count} messages)'
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

    def status(self) -> str:
        parts = []
        parts.append(f'rgb={self.n_rgb_saved}/{self.n_rgb_target}')
        parts.append(f'det={"✓" if self.got_det else "…"}')
        parts.append(f'seg={"✓" if self.got_seg else "…"}')
        parts.append(f'geo={"✓" if self.got_geo else "…"}')
        parts.append(f'merged={self._geo_msg_count}msgs/{len(self._merged_features)}obj')
        parts.append(f'mqtt={self._mqtt_msg_count}')
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
    node.stop_mqtt_trace()

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
