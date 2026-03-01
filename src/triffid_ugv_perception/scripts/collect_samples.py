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
        if self.got_geo:
            return
        path = os.path.join(self.outdir, 'geojson.json')
        try:
            data = json.loads(msg.data)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except json.JSONDecodeError:
            with open(path, 'w') as f:
                f.write(msg.data)
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

    def status(self) -> str:
        parts = []
        parts.append(f'rgb={self.n_rgb_saved}/{self.n_rgb_target}')
        parts.append(f'det={"✓" if self.got_det else "…"}')
        parts.append(f'seg={"✓" if self.got_seg else "…"}')
        parts.append(f'geo={"✓" if self.got_geo else "…"}')
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

    t0 = time.monotonic()
    last_status = 0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            elapsed = time.monotonic() - t0
            if elapsed - last_status > 5.0:
                node.get_logger().info(f'[{elapsed:.0f}s] {node.status()}')
                last_status = elapsed

            if node.all_done():
                node.get_logger().info('All samples collected!')
                break

            if elapsed > args.timeout:
                node.get_logger().warn(
                    f'Timeout after {args.timeout}s — {node.status()}')
                break
    except KeyboardInterrupt:
        pass

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
