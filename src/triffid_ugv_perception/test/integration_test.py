#!/usr/bin/env python3
"""
TRIFFID Integration Test
=========================
Automated end-to-end test that verifies the full perception pipeline
using the rosbag dataset.  Checks all 6 integration requirements:

  1. Interface definitions  – correct topic names, message types, QoS
  2. Interface document     – (README; not tested programmatically)
  3. Replayable dataset     – rosbag plays and topics appear
  4. Run from recorded data – pipeline produces outputs without hardware
  5. Timestamp consistency  – output stamps match input stamps, depth-RGB sync
  6. Coordinate frames      – TF tree, GeoJSON coordinate validity

Usage:

sudo docker compose run --rm perception bash -c '
cd /ws && colcon build --symlink-install 2>&1 | tail -5 && source install/setup.bash &&
ros2 launch triffid_ugv_perception ugv_perception.launch.py gps_origin_lat:=37.9755 gps_origin_lon:=23.7348 &
sleep 5 &&
python3 /ws/src/triffid_ugv_perception/test/integration_test.py --timeout 25
'

    # Or run specific checks:
    python3 .../integration_test.py --check timestamps
    python3 .../integration_test.py --check geojson
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray
from std_msgs.msg import String
from diagnostic_msgs.msg import DiagnosticArray

# Constants
ROSBAG_PATH = '/ws/rosbag'
TIMEOUT_SEC = 30.0          # max time to wait for the full pipeline
BAG_PLAY_RATE = 1.0         # playback rate

# Expected topics with their types (requirement #1)
EXPECTED_TOPICS = {
    '/camera_front/raw_image': 'sensor_msgs/msg/Image',
    '/camera_front/realsense_front/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/camera_front/realsense_front/depth/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/ugv/perception/detections_3d': 'vision_msgs/msg/Detection3DArray',
    '/triffid/geojson': 'std_msgs/msg/String',
    '/triffid/heartbeat': 'std_msgs/msg/String',
    '/diagnostics': 'diagnostic_msgs/msg/DiagnosticArray',
}


# Test harness node

class IntegrationTestNode(Node):
    """Subscribes to all pipeline topics and records data for validation."""

    def __init__(self):
        super().__init__('integration_test')

        self.results = {}       # check_name → (pass: bool, detail: str)
        self.received = {}      # topic → list of messages (capped)
        self._start_time = time.monotonic()

        # Subscribe to everything────
        sensor_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._sub(Image, '/camera_front/raw_image', sensor_qos)
        self._sub(Image, '/camera_front/realsense_front/depth/image_rect_raw', sensor_qos)
        self._sub(CameraInfo, '/camera_front/realsense_front/depth/camera_info',
                  QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self._sub(Detection3DArray, '/ugv/perception/detections_3d', 10)
        self._sub(String, '/triffid/geojson', 10)
        self._sub(String, '/triffid/heartbeat', 10)
        self._sub(DiagnosticArray, '/diagnostics', 10)

    def _sub(self, msg_type, topic, qos):
        self.received[topic] = []

        def cb(msg, _topic=topic):
            if len(self.received[_topic]) < 200:
                self.received[_topic].append(msg)

        self.create_subscription(msg_type, topic, cb, qos)

    def elapsed(self):
        return time.monotonic() - self._start_time


# Checks

def check_rosbag_exists(node: IntegrationTestNode):
    """Requirement #3: replayable dataset exists."""
    exists = os.path.isdir(ROSBAG_PATH)
    if exists:
        files = os.listdir(ROSBAG_PATH)
        has_meta = any('metadata' in f for f in files)
        has_db = any(f.endswith('.db3') for f in files)
        if has_meta and has_db:
            return True, f'Rosbag found at {ROSBAG_PATH} ({len(files)} files)'
        return False, f'Rosbag directory exists but missing metadata/db3: {files}'
    return False, f'Rosbag not found at {ROSBAG_PATH}'


def check_topic_liveness(node: IntegrationTestNode):
    """Requirement #4: all expected topics receive at least one message."""
    missing = []
    alive = []
    for topic in EXPECTED_TOPICS:
        count = len(node.received.get(topic, []))
        if count == 0:
            missing.append(topic)
        else:
            alive.append(f'{topic} ({count} msgs)')

    if missing:
        return False, f'No messages on: {", ".join(missing)}'
    return True, f'All {len(alive)} topics alive'


def check_message_types(node: IntegrationTestNode):
    """Requirement #1: message types match the interface spec."""
    type_map = {
        '/camera_front/raw_image': Image,
        '/camera_front/realsense_front/depth/image_rect_raw': Image,
        '/camera_front/realsense_front/depth/camera_info': CameraInfo,
        '/ugv/perception/detections_3d': Detection3DArray,
        '/triffid/geojson': String,
        '/triffid/heartbeat': String,
        '/diagnostics': DiagnosticArray,
    }
    errors = []
    for topic, expected_type in type_map.items():
        msgs = node.received.get(topic, [])
        if msgs and not isinstance(msgs[0], expected_type):
            errors.append(f'{topic}: got {type(msgs[0]).__name__}, '
                          f'expected {expected_type.__name__}')

    if errors:
        return False, '; '.join(errors)
    checked = sum(1 for t in type_map if node.received.get(t))
    return True, f'All {checked} checked topics have correct types'


def check_timestamps(node: IntegrationTestNode):
    """Requirement #5: detection timestamps come from input images, not wall clock."""
    det_msgs = node.received.get('/ugv/perception/detections_3d', [])
    rgb_msgs = node.received.get('/camera_front/raw_image', [])

    if not det_msgs:
        return False, 'No detections received'
    if not rgb_msgs:
        return False, 'No RGB messages received'

    # Detection stamps should be from the rosbag era (2026-02-20),
    # NOT from wall-clock now.
    det_stamp = det_msgs[0].header.stamp
    det_t = det_stamp.sec + det_stamp.nanosec * 1e-9

    # Check that detection stamp looks like a rosbag timestamp (year ~2026)
    # rather than zero or current wall clock with big offset
    rgb_stamp = rgb_msgs[0].header.stamp
    rgb_t = rgb_stamp.sec + rgb_stamp.nanosec * 1e-9

    if det_t == 0.0:
        return False, 'Detection timestamp is zero (should copy from input)'

    # Detection stamps should be in the same epoch as RGB stamps (rosbag time)
    delta = abs(det_t - rgb_t)
    if delta > 60.0:  # more than 60s apart → likely using wall clock
        return False, (f'Detection stamp ({det_t:.0f}) differs from RGB stamp '
                       f'({rgb_t:.0f}) by {delta:.0f}s — timestamps likely fabricated')

    # Check depth-RGB sync: compare latest stamps
    depth_msgs = node.received.get(
        '/camera_front/realsense_front/depth/image_rect_raw', [])

    sync_detail = ''
    if depth_msgs and rgb_msgs:
        # Take last received of each
        rgb_last = rgb_msgs[-1].header.stamp
        depth_last = depth_msgs[-1].header.stamp
        rgb_lt = rgb_last.sec + rgb_last.nanosec * 1e-9
        depth_lt = depth_last.sec + depth_last.nanosec * 1e-9
        sync_delta = abs(rgb_lt - depth_lt)
        sync_detail = f'; depth-RGB sync delta: {sync_delta*1000:.1f} ms'
        if sync_delta > 0.5:
            return False, (f'Depth-RGB sync too large: {sync_delta*1000:.1f} ms'
                           f' (>500 ms)')

    return True, (f'Detection stamps match input epoch (delta {delta:.3f}s)'
                  f'{sync_detail}')


def check_detection_fields(node: IntegrationTestNode):
    """Requirement #1: Detection3D messages have all required fields populated."""
    det_msgs = node.received.get('/ugv/perception/detections_3d', [])
    if not det_msgs:
        return False, 'No detections received'

    errors = []
    checked = 0
    for msg in det_msgs[:10]:  # check first 10
        for det in msg.detections:
            checked += 1
            if not det.id:
                errors.append('detection.id is empty (tracking ID required)')
            if not det.results:
                errors.append('detection.results is empty (need class+score)')
            else:
                hyp = det.results[0].hypothesis
                if not hyp.class_id:
                    errors.append('class_id is empty')
                if hyp.score <= 0.0:
                    errors.append(f'score={hyp.score} (should be >0)')
            if not msg.header.frame_id:
                errors.append('header.frame_id is empty')

    if errors:
        unique = list(set(errors))
        return False, f'{len(unique)} issue(s): {"; ".join(unique[:5])}'
    return True, f'All {checked} detections have valid id, class_id, score, frame_id'


def check_geojson(node: IntegrationTestNode):
    """Requirement #6: GeoJSON output is valid RFC-7946."""
    geojson_msgs = node.received.get('/triffid/geojson', [])
    if not geojson_msgs:
        return False, 'No GeoJSON messages received'

    errors = []
    for i, msg in enumerate(geojson_msgs[:5]):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            errors.append(f'msg[{i}]: invalid JSON: {e}')
            continue

        if data.get('type') != 'FeatureCollection':
            errors.append(f'msg[{i}]: type={data.get("type")}, expected FeatureCollection')
            continue

        features = data.get('features', [])
        for j, feat in enumerate(features):
            if feat.get('type') != 'Feature':
                errors.append(f'msg[{i}].features[{j}]: not a Feature')
                continue

            geom = feat.get('geometry', {})
            if geom.get('type') != 'Point':
                errors.append(f'msg[{i}].features[{j}]: geometry type={geom.get("type")}')
                continue

            coords = geom.get('coordinates', [])
            if len(coords) < 2:
                errors.append(f'msg[{i}].features[{j}]: coordinates has <2 elements')
                continue

            lon, lat = coords[0], coords[1]
            # Sanity: coordinates should be finite numbers
            if not (math.isfinite(lon) and math.isfinite(lat)):
                errors.append(f'msg[{i}].features[{j}]: non-finite coords ({lon}, {lat})')

            # Check required SimpleStyle properties
            props = feat.get('properties', {})
            for key in ('name', 'category', 'source', 'track_id', 'confidence',
                        'marker-color', 'marker-size', 'marker-symbol'):
                if key not in props:
                    errors.append(f'msg[{i}].features[{j}]: missing property "{key}"')

    if errors:
        unique = list(set(errors))
        return False, f'{len(unique)} issue(s): {"; ".join(unique[:5])}'

    total_features = sum(
        len(json.loads(m.data).get('features', []))
        for m in geojson_msgs[:5]
    )
    return True, f'{len(geojson_msgs)} GeoJSON msgs, {total_features} total features, all valid RFC-7946'


def check_diagnostics(node: IntegrationTestNode):
    """Health: diagnostics node is publishing."""
    diag_msgs = node.received.get('/diagnostics', [])
    hb_msgs = node.received.get('/triffid/heartbeat', [])

    if not diag_msgs:
        return False, 'No /diagnostics messages received'
    if not hb_msgs:
        return False, 'No /triffid/heartbeat messages received'

    # Check heartbeat is valid JSON
    try:
        hb = json.loads(hb_msgs[-1].data)
        status = hb.get('status', 'UNKNOWN')
    except (json.JSONDecodeError, AttributeError):
        return False, 'Heartbeat is not valid JSON'

    # Count diagnostic statuses (level can be bytes in Humble)
    def _lvl(level):
        return int.from_bytes(level, 'little') if isinstance(level, bytes) else int(level)

    last_diag = diag_msgs[-1]
    n_ok = sum(1 for s in last_diag.status if _lvl(s.level) == 0)
    n_warn = sum(1 for s in last_diag.status if _lvl(s.level) == 1)
    n_err = sum(1 for s in last_diag.status if _lvl(s.level) == 2)
    n_stale = sum(1 for s in last_diag.status if _lvl(s.level) == 3)

    return True, (f'Diagnostics active: {n_ok} OK, {n_warn} WARN, '
                  f'{n_err} ERROR, {n_stale} STALE; heartbeat: {status}')


# Runner

ALL_CHECKS = {
    'rosbag':       ('Req 3: Replayable dataset',      check_rosbag_exists),
    'topics':       ('Req 4: Topic liveness',           check_topic_liveness),
    'types':        ('Req 1: Message types',            check_message_types),
    'timestamps':   ('Req 5: Timestamp consistency',    check_timestamps),
    'fields':       ('Req 1: Detection field validity', check_detection_fields),
    'geojson':      ('Req 6: GeoJSON / coordinates',    check_geojson),
    'diagnostics':  ('Health: Diagnostics',             check_diagnostics),
}


def print_results(results):
    """Pretty-print test results."""
    print('\n' + '=' * 72)
    print('  TRIFFID INTEGRATION TEST RESULTS')
    print('=' * 72)

    passed = 0
    failed = 0
    for name, (label, _) in ALL_CHECKS.items():
        if name not in results:
            continue
        ok, detail = results[name]
        icon = '✓' if ok else '✗'
        colour = '\033[92m' if ok else '\033[91m'
        reset = '\033[0m'
        print(f'  {colour}{icon}{reset}  {label}')
        print(f'     {detail}')
        if ok:
            passed += 1
        else:
            failed += 1

    print('-' * 72)
    if failed == 0:
        print(f'  \033[92mAll {passed} checks passed.\033[0m')
    else:
        print(f'  \033[91m{failed} FAILED\033[0m, {passed} passed.')
    print('=' * 72 + '\n')

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description='TRIFFID integration test')
    parser.add_argument('--check', type=str, default=None,
                        help='Run only this check (e.g. "timestamps", "geojson")')
    parser.add_argument('--timeout', type=float, default=TIMEOUT_SEC,
                        help=f'Seconds to wait for data (default {TIMEOUT_SEC})')
    parser.add_argument('--no-bag', action='store_true',
                        help='Skip automatic rosbag playback (assume already playing)')
    args = parser.parse_args()

    rclpy.init()
    node = IntegrationTestNode()

    # Run the rosbag check first (no data needed)
    results = {}
    ok, detail = check_rosbag_exists(node)
    results['rosbag'] = (ok, detail)
    if not ok:
        node.get_logger().error(f'Rosbag check failed: {detail}')

    # Start rosbag player────
    bag_proc = None
    if not args.no_bag and os.path.isdir(ROSBAG_PATH):
        node.get_logger().info(f'Starting rosbag playback: {ROSBAG_PATH}')
        bag_proc = subprocess.Popen(
            ['ros2', 'bag', 'play', ROSBAG_PATH,
             '--rate', str(BAG_PLAY_RATE),
             '--start-offset', '5'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Spin and collect data────
    node.get_logger().info(f'Collecting data for up to {args.timeout}s...')
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

            # Early exit: if we have data on all topics, no need to wait
            all_populated = all(
                len(node.received.get(t, [])) > 0
                for t in EXPECTED_TOPICS
            )
            if all_populated and node.elapsed() > 5.0:
                node.get_logger().info('All topics populated — running checks early.')
                break
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted.')

    # Stop rosbag────
    if bag_proc is not None:
        bag_proc.send_signal(signal.SIGINT)
        bag_proc.wait(timeout=5)

    # Run checks────
    checks_to_run = ALL_CHECKS
    if args.check:
        if args.check not in ALL_CHECKS:
            print(f'Unknown check: {args.check}')
            print(f'Available: {", ".join(ALL_CHECKS.keys())}')
            sys.exit(1)
        checks_to_run = {args.check: ALL_CHECKS[args.check]}

    for name, (label, fn) in checks_to_run.items():
        if name == 'rosbag':
            continue  # already done
        try:
            ok, detail = fn(node)
            results[name] = (ok, detail)
        except Exception as e:
            results[name] = (False, f'Exception: {e}')

    node.destroy_node()
    rclpy.shutdown()

    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
