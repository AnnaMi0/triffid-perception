#!/usr/bin/env python3

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

from tf2_ros import Buffer, TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# Constants
ROSBAG_PATH = '/ws/rosbag'
OUTPUT_BAG_ROOT = '/ws/output_rosbag'
OUTPUT_BAG_PATH = '/ws/output_rosbag/recording'
TIMEOUT_SEC = 30.0          # max time to wait for the full pipeline
BAG_PLAY_RATE = 1.0         # playback rate

# Output topics that must be recorded into the output rosbag
OUTPUT_TOPICS = [
    '/ugv/detections/front/detections_3d',
    '/ugv/detections/front/geojson',
    '/ugv/detections/front/segmentation',
]

# QoS override file — needed so ros2 bag play publishes /tf_static, with TRANSIENT_LOCAL durability (otherwise TF2 never receives static
# transforms and frames like f_oc_link are invisible).
QOS_OVERRIDES = '/ws/src/triffid_ugv_perception/config/bag_qos_overrides.yaml'

# Frame IDs (must match ugv_node.py constants)
DEPTH_FRAME = 'f_depth_optical_frame'
RGB_FRAME = 'f_oc_link'
BASE_FRAME = 'b2/base_link'

# Expected topics with their types (requirement #1)
EXPECTED_TOPICS = {
    '/camera_front/raw_image': 'sensor_msgs/msg/Image',
    '/camera_front/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/camera_front/realsense_front/depth/image_rect_raw': 'sensor_msgs/msg/Image',
    '/camera_front/realsense_front/depth/camera_info': 'sensor_msgs/msg/CameraInfo',
    '/ugv/detections/front/detections_3d': 'vision_msgs/msg/Detection3DArray',
}


# Test harness node
class IntegrationTestNode(Node):

    def __init__(self):
        super().__init__('integration_test')

        self.results = {}       # check_name → (pass: bool, detail: str)
        self.received = {}      # topic → list of messages (capped)
        self._start_time = time.monotonic()

        # TF2 buffer (for TF tree check)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publish known static transforms from the rosbag so the perception pipeline has TF available regardless of DDS
        # /tf_static delivery race conditions.
        self._publish_static_transforms()

        # Subscribe to everything — BEST_EFFORT + VOLATILE (most permissive)
        sensor_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._sub(Image, '/camera_front/raw_image', sensor_qos)
        self._sub(Image, '/camera_front/realsense_front/depth/image_rect_raw', sensor_qos)
        self._sub(CameraInfo, '/camera_front/camera_info',
                  QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE))
        self._sub(CameraInfo, '/camera_front/realsense_front/depth/camera_info',
                  QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE))
        self._sub(Detection3DArray, '/ugv/detections/front/detections_3d', 10)
        self._sub(String, '/ugv/detections/front/geojson', 10)
        self._sub(Image, '/ugv/detections/front/segmentation', sensor_qos)

    def _sub(self, msg_type, topic, qos):
        self.received[topic] = []

        def cb(msg, _topic=topic):
            if len(self.received[_topic]) < 200:
                self.received[_topic].append(msg)

        self.create_subscription(msg_type, topic, cb, qos)

    def elapsed(self):
        return time.monotonic() - self._start_time

    def _publish_static_transforms(self):
        # Broadcast the required static TFs extracted from the rosbag so the perception pipeline has TF available regardless of DDS
        # QoS race conditions (VOLATILE vs TRANSIENT_LOCAL).

        self.static_broadcaster = StaticTransformBroadcaster(self)

        def _make_tf(parent, child, t, q):
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = parent
            tf.child_frame_id = child
            tf.transform.translation.x = t[0]
            tf.transform.translation.y = t[1]
            tf.transform.translation.z = t[2]
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            return tf

        # Key transforms needed by the perception pipeline:
        # 1. b2/base_link → f_oc_link  (RGB camera)
        # 2. b2/base_link → f_dc_link  (depth camera base)
        # 3. f_dc_link → f_depth_frame → f_depth_optical_frame  (depth chain)
        transforms = [
            _make_tf('b2/base_link', 'f_oc_link',
                     (0.3993, 0.0, -0.0158), (0.0, 0.0, 0.0, 1.0)),
            _make_tf('b2/base_link', 'f_dc_link',
                     (0.4216, 0.025, 0.0619), (0.0, 0.3827, 0.0, 0.9239)),
            _make_tf('f_dc_link', 'f_depth_frame',
                     (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            _make_tf('f_depth_frame', 'f_depth_optical_frame',
                     (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5)),
        ]
        self.static_broadcaster.sendTransform(transforms)
        self.get_logger().info(
            f'Published {len(transforms)} static transforms '
            f'(f_oc_link, f_dc_link, f_depth_frame, f_depth_optical_frame)')

# Checks

def check_output_rosbag(node: IntegrationTestNode):
    """Requirement #3: pipeline records a replayable output rosbag."""
    if not os.path.isdir(OUTPUT_BAG_PATH):
        return False, f'Output rosbag not found at {OUTPUT_BAG_PATH}'

    files = os.listdir(OUTPUT_BAG_PATH)
    has_meta = any('metadata' in f for f in files)
    has_db = any(f.endswith('.db3') for f in files)
    if not (has_meta and has_db):
        return False, f'Output rosbag missing metadata/db3: {files}'

    # Read metadata to verify recorded topics
    meta_file = os.path.join(OUTPUT_BAG_PATH, 'metadata.yaml')
    if os.path.isfile(meta_file):
        import yaml
        with open(meta_file) as f:
            meta = yaml.safe_load(f)
        topics_in_bag = []
        for topic_info in meta.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', []):
            t = topic_info.get('topic_metadata', {}).get('name', '')
            count = topic_info.get('message_count', 0)
            if t:
                topics_in_bag.append((t, count))

        missing = [t for t in OUTPUT_TOPICS if not any(bt == t for bt, _ in topics_in_bag)]
        if missing:
            return False, f'Output bag missing topics: {missing}'

        detail_parts = [f'{t} ({c} msgs)' for t, c in topics_in_bag]
        return True, f'Output rosbag recorded: {" | ".join(detail_parts)}'

    # No metadata file — just check files exist
    db_files = [f for f in files if f.endswith('.db3')]
    total_size = sum(os.path.getsize(os.path.join(OUTPUT_BAG_PATH, f)) for f in files)
    return True, f'Output rosbag at {OUTPUT_BAG_PATH}: {len(db_files)} db3, {total_size/1024:.0f} KB'


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
        '/camera_front/camera_info': CameraInfo,
        '/camera_front/realsense_front/depth/image_rect_raw': Image,
        '/camera_front/realsense_front/depth/camera_info': CameraInfo,
        '/ugv/detections/front/detections_3d': Detection3DArray,
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
    # Requirement #5: detection timestamps come from input images, not wall clock.
    det_msgs = node.received.get('/ugv/detections/front/detections_3d', [])
    rgb_msgs = node.received.get('/camera_front/raw_image', [])

    if not det_msgs:
        return False, 'No detections received'
    if not rgb_msgs:
        return False, 'No RGB messages received'

    # Detection stamps should be from the rosbag era (2026-02-20)
    det_stamp = det_msgs[0].header.stamp
    det_t = det_stamp.sec + det_stamp.nanosec * 1e-9

    # Check that detection stamp looks like a rosbag timestamp, instead of zero or current wall clock with big offset
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
    # Requirement #1: Detection3D messages have all required fields populated, verifies frame_id is b2/base_link (not map)
    det_msgs = node.received.get('/ugv/detections/front/detections_3d', [])
    if not det_msgs:
        return False, 'No detections received'

    errors = []
    checked = 0
    for msg in det_msgs[:10]:  # check first 10
        # Verify frame_id
        if msg.header.frame_id != BASE_FRAME:
            errors.append(
                f'header.frame_id="{msg.header.frame_id}", '
                f'expected "{BASE_FRAME}"')

        for det in msg.detections:
            checked += 1
            if not det.id:
                errors.append('detection.id is empty (tracking ID required)')
            else:
                # Track ID should be a positive integer string
                try:
                    tid = int(det.id)
                    if tid <= 0:
                        errors.append(f'track_id={tid} (should be > 0)')
                except ValueError:
                    errors.append(f'track_id="{det.id}" is not an integer')

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
    # Requirement #6: GeoJSON output is valid RFC-7946
    geojson_msgs = node.received.get('/ugv/detections/front/geojson', [])
    if not geojson_msgs:
        return False, 'No GeoJSON messages received'

    errors = []
    point_count = 0
    polygon_count = 0

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
            geom_type = geom.get('type')

            if geom_type == 'Point':
                coords = geom.get('coordinates', [])
                if len(coords) < 2:
                    errors.append(f'msg[{i}].features[{j}]: Point has <2 coords')
                    continue
                lon, lat = coords[0], coords[1]
                if not (math.isfinite(lon) and math.isfinite(lat)):
                    errors.append(f'msg[{i}].features[{j}]: non-finite Point coords')
                point_count += 1

            elif geom_type == 'Polygon':
                rings = geom.get('coordinates', [])
                if not rings or len(rings[0]) < 4:
                    errors.append(f'msg[{i}].features[{j}]: Polygon ring has <4 vertices')
                    continue
                ring = rings[0]
                # RFC-7946: ring must be closed
                if ring[0] != ring[-1]:
                    errors.append(f'msg[{i}].features[{j}]: Polygon ring not closed')
                # All vertices must be finite
                for k, pt in enumerate(ring):
                    if len(pt) < 2 or not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
                        errors.append(f'msg[{i}].features[{j}]: non-finite Polygon vertex {k}')
                        break
                polygon_count += 1

            else:
                errors.append(f'msg[{i}].features[{j}]: unexpected geometry type={geom_type}')
                continue

            # Check required SimpleStyle properties
            props = feat.get('properties', {})
            for key in ('class', 'id', 'confidence',
                        'category', 'source',
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
    return True, (
        f'{len(geojson_msgs)} GeoJSON msgs, {total_features} features '
        f'({polygon_count} Polygons, {point_count} Points), all valid RFC-7946'
    )


def check_camera_info(node: IntegrationTestNode):
    # verify both CameraInfo topics deliver valid intrinsics
    rgb_infos = node.received.get('/camera_front/camera_info', [])
    depth_infos = node.received.get(
        '/camera_front/realsense_front/depth/camera_info', [])

    errors = []

    for label, msgs, expected_frame, expected_wh in [
        ('RGB', rgb_infos, RGB_FRAME, (1280, 720)),
        ('Depth', depth_infos, DEPTH_FRAME, (640, 480)),
    ]:
        if not msgs:
            errors.append(f'{label} CameraInfo: no messages')
            continue

        info = msgs[-1]

        # Frame ID
        if info.header.frame_id != expected_frame:
            errors.append(
                f'{label} frame_id="{info.header.frame_id}", '
                f'expected "{expected_frame}"')

        # Resolution
        if (info.width, info.height) != expected_wh:
            errors.append(
                f'{label} resolution={info.width}×{info.height}, '
                f'expected {expected_wh[0]}×{expected_wh[1]}')

        # Focal lengths > 0
        fx, fy = info.k[0], info.k[4]
        if fx <= 0 or fy <= 0:
            errors.append(f'{label} focal lengths invalid: fx={fx}, fy={fy}')

    if errors:
        return False, '; '.join(errors)
    return True, 'Both CameraInfo topics valid (frame, resolution, intrinsics)'


def check_tf_tree(node: IntegrationTestNode):
    # Verify that the two required TF transforms are available: f_depth_optical_frame → f_oc_link  and  f_oc_link → b2/base_link
    required = [
        (DEPTH_FRAME, RGB_FRAME),
        (RGB_FRAME, BASE_FRAME),
    ]
    errors = []
    ok_list = []
    for source, target in required:
        try:
            node.tf_buffer.lookup_transform(
                target, source,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            ok_list.append(f'{source} → {target}')
        except Exception as e:
            errors.append(f'{source} → {target}: {e}')

    if errors:
        return False, f'Missing TF: {"; ".join(errors)}'
    return True, f'All {len(ok_list)} required transforms available'


def check_3d_positions(node: IntegrationTestNode):
    # verify that 3D positions in detections are finite, non-zero, within a plausible range from b2/base_link
    det_msgs = node.received.get('/ugv/detections/front/detections_3d', [])
    if not det_msgs:
        return False, 'No detections received'

    errors = []
    checked = 0
    n_zero = 0
    distances = []
    MAX_RANGE = 30.0

    for msg in det_msgs[:20]:
        for det in msg.detections:
            checked += 1
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                errors.append(f'Non-finite position: ({x:.3f}, {y:.3f}, {z:.3f})')
                continue

            dist = math.sqrt(x*x + y*y + z*z)
            distances.append(dist)

            if dist < 1e-6:
                n_zero += 1
            elif dist > MAX_RANGE:
                errors.append(f'Detection at {dist:.1f}m (>{MAX_RANGE}m)')

    if not checked:
        return False, 'Detection messages exist but contain no detections'
    if n_zero == checked:
        return False, f'All {checked} detections have zero position (depth pipeline broken?)'
    if errors:
        unique = list(set(errors))
        return False, f'{len(unique)} issue(s): {"; ".join(unique[:5])}'

    detail = f'{checked} detections checked'
    if distances:
        detail += f', range: {min(distances):.2f}–{max(distances):.2f}m'
    if n_zero > 0:
        detail += f', {n_zero} zero-position'
    return True, detail


def check_tracking(node: IntegrationTestNode):
    # Verify tracking IDs are persistent positive integers with no duplicate IDs within a single frame and consistent class assignment.# 
    det_msgs = node.received.get('/ugv/detections/front/detections_3d', [])
    if not det_msgs:
        return False, 'No detections received'

    all_ids = set()
    id_class = {}  # track_id → class_name
    errors = []
    n_frames = 0

    for msg in det_msgs:
        if not msg.detections:
            continue
        n_frames += 1
        frame_ids = []

        for det in msg.detections:
            tid = det.id
            if not tid:
                errors.append('Empty tracking ID')
                continue
            frame_ids.append(tid)
            all_ids.add(tid)

            cls = det.results[0].hypothesis.class_id if det.results else ''
            if tid in id_class:
                if id_class[tid] != cls:
                    errors.append(f'ID {tid} changed class: {id_class[tid]} → {cls}')
            else:
                id_class[tid] = cls

        # Duplicate IDs in one frame?
        if len(frame_ids) != len(set(frame_ids)):
            dupes = [x for x in frame_ids if frame_ids.count(x) > 1]
            errors.append(f'Duplicate IDs in frame: {set(dupes)}')

    if errors:
        unique = list(set(errors))
        return False, f'{len(unique)} issue(s): {"; ".join(unique[:5])}'

    return True, (f'{len(all_ids)} unique IDs across {n_frames} frames, '
                  f'no duplicates, classes consistent')


def check_segmentation(node: IntegrationTestNode):
    # Verify segmentation label images are published and valid.
    seg_msgs = node.received.get('/ugv/detections/front/segmentation', [])
    if not seg_msgs:
        return False, 'No segmentation messages received (is anything subscribed?)'

    errors = []
    for i, msg in enumerate(seg_msgs[:5]):
        if msg.encoding != 'mono8':
            errors.append(f'msg[{i}]: encoding={msg.encoding}, expected mono8')
        if msg.width == 0 or msg.height == 0:
            errors.append(f'msg[{i}]: zero dimensions {msg.width}x{msg.height}')
        expected_step = msg.width  # 1 byte per pixel for mono8
        if msg.step < expected_step:
            errors.append(f'msg[{i}]: step={msg.step} < expected {expected_step}')
        if len(msg.data) == 0:
            errors.append(f'msg[{i}]: empty data')

    if errors:
        unique = list(set(errors))
        return False, f'{len(unique)} issue(s): {"; ".join(unique[:5])}'

    sample = seg_msgs[0]
    return True, (f'{len(seg_msgs)} segmentation msgs, '
                  f'{sample.width}x{sample.height} {sample.encoding}')


# Runner
ALL_CHECKS = {
    'rosbag':       ('Req 3: Output rosbag recorded',   check_output_rosbag),
    'topics':       ('Req 4: Topic liveness',           check_topic_liveness),
    'types':        ('Req 1: Message types',            check_message_types),
    'camera_info':  ('CameraInfo validation',           check_camera_info),
    'tf_tree':      ('TF: Required transforms',         check_tf_tree),
    'timestamps':   ('Req 5: Timestamp consistency',    check_timestamps),
    'fields':       ('Req 1: Detection field validity', check_detection_fields),
    'positions':    ('Depth: 3D position sanity',       check_3d_positions),
    'tracking':     ('Tracking: Persistent IDs',        check_tracking),
    'geojson':      ('Req 6: GeoJSON RFC-7946',         check_geojson),
    'segmentation': ('Segmentation label map',         check_segmentation),
}


def print_results(results):
    print('\n' + '=' * 72)
    print('  INTEGRATION TEST RESULTS')
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
    parser.add_argument('--no-launch', action='store_true',
                        help='Skip launching perception nodes (assume already running)')
    args = parser.parse_args()

    rclpy.init()
    node = IntegrationTestNode()

    results = {}

    # Launch perception nodes
    launch_proc = None
    if not args.no_launch:
        node.get_logger().info('Launching perception nodes (dummy detection mode)...')
        launch_proc = subprocess.Popen(
            ['ros2', 'launch', 'triffid_ugv_perception', 'ugv_perception.launch.py',
             'use_dummy_detections:=true'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        # Give nodes time to initialise
        time.sleep(8)
        node.get_logger().info('Perception nodes launched.')

    # Start output rosbag recorder
    import shutil
    os.makedirs(OUTPUT_BAG_ROOT, exist_ok=True)
    if os.path.isdir(OUTPUT_BAG_PATH):
        shutil.rmtree(OUTPUT_BAG_PATH, ignore_errors=True)
    record_proc = subprocess.Popen(
        ['ros2', 'bag', 'record', '-o', OUTPUT_BAG_PATH] + OUTPUT_TOPICS,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    node.get_logger().info(f'Recording output rosbag to {OUTPUT_BAG_PATH}')

    # Start rosbag player
    bag_proc = None
    if not args.no_bag and os.path.isdir(ROSBAG_PATH):
        node.get_logger().info(f'Starting rosbag playback: {ROSBAG_PATH}')
        bag_cmd = [
            'ros2', 'bag', 'play', ROSBAG_PATH,
            '--rate', str(BAG_PLAY_RATE),
        ]
        # Add QoS overrides so /tf_static uses TRANSIENT_LOCAL
        if os.path.isfile(QOS_OVERRIDES):
            bag_cmd += ['--qos-profile-overrides-path', QOS_OVERRIDES]
            node.get_logger().info(f'Using QoS overrides: {QOS_OVERRIDES}')
        bag_proc = subprocess.Popen(
            bag_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Spin and collect data
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

    # Stop output recorder
    record_proc.send_signal(signal.SIGINT)
    try:
        record_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        record_proc.kill()
    node.get_logger().info('Output rosbag recording stopped.')

    # Stop rosbag player
    if bag_proc is not None:
        bag_proc.send_signal(signal.SIGINT)
        bag_proc.wait(timeout=5)

    # Stop perception nodes
    if launch_proc is not None:
        try:
            os.killpg(os.getpgid(launch_proc.pid), signal.SIGINT)
            launch_proc.wait(timeout=10)
        except Exception:
            launch_proc.kill()

    # Run checks
    checks_to_run = ALL_CHECKS
    if args.check:
        if args.check not in ALL_CHECKS:
            print(f'Unknown check: {args.check}')
            print(f'Available: {", ".join(ALL_CHECKS.keys())}')
            sys.exit(1)
        checks_to_run = {args.check: ALL_CHECKS[args.check]}

    for name, (label, fn) in checks_to_run.items():
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
