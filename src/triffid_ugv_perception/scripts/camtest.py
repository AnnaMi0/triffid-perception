#!/usr/bin/env python3
"""
RealSense camera sanity check.

Subscribes to color and depth image topics, converts and saves one frame
each to PNG, then exits.  Use this to verify camera access and that the
YUYV→BGR conversion works before starting the full pipeline.

Usage (inside container):
  python3 /ws/src/triffid_ugv_perception/scripts/camtest.py \\
      --rgb-topic /camera_front_435i/realsense_front_435i/color/image_raw \\
      --depth-topic /camera_front_435i/realsense_front_435i/depth/image_rect_raw \\
      --outdir /ws/samples

Output:
  <outdir>/camtest_color.png  — BGR color frame
  <outdir>/camtest_depth.png  — false-colour depth frame
"""

import argparse
import os
import sys
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    _HAS_BRIDGE = True
except ImportError:
    _HAS_BRIDGE = False


class CamTestNode(Node):
    def __init__(self, rgb_topic: str, depth_topic: str, outdir: str,
                 timeout_s: float = 10.0):
        super().__init__('camtest')
        self.outdir = outdir
        self.timeout_s = timeout_s

        self._bridge = CvBridge() if _HAS_BRIDGE else None
        self._color_saved = False
        self._depth_saved = False
        self._color_info: dict = {}
        self._depth_info: dict = {}
        self._done = threading.Event()

        self.sub_color = self.create_subscription(
            Image, rgb_topic, self._color_cb, 5)
        self.sub_depth = self.create_subscription(
            Image, depth_topic, self._depth_cb, 5)

        self.get_logger().info(f'Waiting for frames (timeout {timeout_s}s)...')
        self.get_logger().info(f'  Color : {rgb_topic}')
        self.get_logger().info(f'  Depth : {depth_topic}')

        # Timeout watchdog
        self._timer = self.create_timer(timeout_s, self._on_timeout)

    def _color_cb(self, msg: Image):
        if self._color_saved:
            return
        enc = msg.encoding
        try:
            if enc in ('yuv422', 'yuv422_yuy2', 'YUV422'):
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 2))
                bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUY2)
            elif self._bridge:
                bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            else:
                # Manual fallback for rgb8 without cv_bridge
                raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (msg.height, msg.width, 3))
                bgr = raw[:, :, ::-1]  # RGB → BGR

            path = os.path.join(self.outdir, 'camtest_color.png')
            cv2.imwrite(path, bgr)
            self._color_info = {
                'encoding': enc,
                'size': f'{msg.width}×{msg.height}',
                'stamp': f'{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}',
                'saved': path,
            }
            self._color_saved = True
            self.get_logger().info(
                f'Color frame saved: {path}  [{msg.width}×{msg.height}  {enc}]')
        except Exception as e:
            self.get_logger().error(f'Color conversion failed: {e}')
            return
        self._check_done()

    def _depth_cb(self, msg: Image):
        if self._depth_saved:
            return
        try:
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                (msg.height, msg.width))

            # False-colour: normalise to 0-255, apply COLORMAP_JET
            valid = depth[depth > 0]
            if valid.size == 0:
                self.get_logger().warn('Depth frame is all zeros — saved as black image')
                vis = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
            else:
                d_min, d_max = int(valid.min()), int(valid.max())
                norm = np.clip(
                    (depth.astype(np.float32) - d_min) / max(d_max - d_min, 1),
                    0.0, 1.0,
                )
                grey = (norm * 255).astype(np.uint8)
                vis = cv2.applyColorMap(grey, cv2.COLORMAP_JET)
                # Zero pixels (no measurement) → black
                vis[depth == 0] = 0

            path = os.path.join(self.outdir, 'camtest_depth.png')
            cv2.imwrite(path, vis)
            self._depth_info = {
                'encoding': msg.encoding,
                'size': f'{msg.width}×{msg.height}',
                'depth_range_mm': f'{int(valid.min()) if valid.size else 0}–{int(valid.max()) if valid.size else 0}',
                'saved': path,
            }
            self._depth_saved = True
            self.get_logger().info(
                f'Depth frame saved: {path}  [{msg.width}×{msg.height}  '
                f'range {self._depth_info["depth_range_mm"]} mm]')
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')
            return
        self._check_done()

    def _check_done(self):
        if self._color_saved and self._depth_saved:
            self._done.set()

    def _on_timeout(self):
        if not self._color_saved:
            self.get_logger().error('TIMEOUT: no color frame received')
        if not self._depth_saved:
            self.get_logger().error('TIMEOUT: no depth frame received')
        self._done.set()

    def success(self) -> bool:
        return self._color_saved and self._depth_saved

    def print_summary(self):
        print('\n── Camera Test Results ─────────────────────────')
        if self._color_saved:
            print('Color  ✓')
            for k, v in self._color_info.items():
                print(f'  {k}: {v}')
        else:
            print('Color  ✗  no frame received')
        if self._depth_saved:
            print('Depth  ✓')
            for k, v in self._depth_info.items():
                print(f'  {k}: {v}')
        else:
            print('Depth  ✗  no frame received')
        print('────────────────────────────────────────────────\n')


def main():
    parser = argparse.ArgumentParser(description='RealSense camera sanity check')
    parser.add_argument('--rgb-topic',
                        default='/camera_front_435i/realsense_front_435i/color/image_raw')
    parser.add_argument('--depth-topic',
                        default='/camera_front_435i/realsense_front_435i/depth/image_rect_raw')
    parser.add_argument('--outdir', default='/ws/samples')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Seconds to wait for frames (default: 10)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rclpy.init()
    node = CamTestNode(
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        outdir=args.outdir,
        timeout_s=args.timeout,
    )

    # Spin until both frames received or timeout
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    while not node._done.is_set():
        executor.spin_once(timeout_sec=0.1)

    node.print_summary()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if node.success() else 1)


if __name__ == '__main__':
    main()
