#!/usr/bin/env python3
"""Validate live RealSense topics used by the UGV pipeline.

Topic names are resolved from the running ``ugv_perception_node`` parameters
(which are typically provided via launch), with optional CLI overrides.
"""

import argparse
import time

import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo, Image


FALLBACK_RGB_TOPIC = '/camera_front_435i/realsense_front_435i/color/image_raw'
FALLBACK_DEPTH_TOPIC = '/camera_front_435i/realsense_front_435i/depth/image_rect_raw'
FALLBACK_COLOR_INFO_TOPIC = '/camera_front_435i/realsense_front_435i/color/camera_info'
FALLBACK_DEPTH_INFO_TOPIC = '/camera_front_435i/realsense_front_435i/depth/camera_info'

ALLOWED_COLOR_ENCODINGS = {'bgr8', 'rgb8', 'yuyv', 'yuv422_yuy2'}
ALLOWED_DEPTH_ENCODINGS = {'16UC1'}


class RealSenseStreamValidator(Node):
    """Collects a short sample from required topics and validates assumptions."""

    def __init__(self, color_info_topic, color_image_topic,
                 depth_info_topic, depth_image_topic):
        super().__init__('realsense_stream_validator')

        self.color_info_topic = color_info_topic
        self.color_image_topic = color_image_topic
        self.depth_info_topic = depth_info_topic
        self.depth_image_topic = depth_image_topic

        self.color_info = None
        self.depth_info = None
        self.color_image = None
        self.depth_image = None
        self.color_count = 0
        self.depth_count = 0
        self.color_first_t = None
        self.color_last_t = None
        self.depth_first_t = None
        self.depth_last_t = None

        self.create_subscription(
            CameraInfo,
            self.color_info_topic,
            self._cb_color_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            self.depth_info_topic,
            self._cb_depth_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.color_image_topic,
            self._cb_color_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            self.depth_image_topic,
            self._cb_depth_image,
            qos_profile_sensor_data,
        )

    def _cb_color_info(self, msg: CameraInfo):
        self.color_info = msg

    def _cb_depth_info(self, msg: CameraInfo):
        self.depth_info = msg

    def _cb_color_image(self, msg: Image):
        now = time.monotonic()
        if self.color_first_t is None:
            self.color_first_t = now
        self.color_last_t = now
        self.color_count += 1
        self.color_image = msg

    def _cb_depth_image(self, msg: Image):
        now = time.monotonic()
        if self.depth_first_t is None:
            self.depth_first_t = now
        self.depth_last_t = now
        self.depth_count += 1
        self.depth_image = msg

    def has_core_messages(self) -> bool:
        return all(
            (
                self.color_info is not None,
                self.depth_info is not None,
                self.color_image is not None,
                self.depth_image is not None,
            )
        )

    @staticmethod
    def rate_hz(first_t, last_t, count) -> float:
        if first_t is None or last_t is None or count < 2:
            return 0.0
        dt = max(1e-9, last_t - first_t)
        return (count - 1) / dt


def _derive_topic(base_topic, old_suffix, new_suffix):
    if base_topic.endswith(old_suffix):
        return f'{base_topic[:-len(old_suffix)]}{new_suffix}'
    return ''


def _derive_aux_topics(rgb_topic, depth_topic):
    color_meta = _derive_topic(rgb_topic, '/image_raw', '/metadata')
    depth_meta = _derive_topic(depth_topic, '/image_rect_raw', '/metadata')

    extrinsics = ''
    marker = '/depth/'
    pos = depth_topic.find(marker)
    if pos > 0:
        extrinsics = f'{depth_topic[:pos]}/extrinsics/depth_to_color'

    return color_meta, depth_meta, extrinsics


def _get_parameters(node: Node, target_node: str, names, timeout_sec=1.0):
    svc = f'{target_node}/get_parameters'
    client = node.create_client(GetParameters, svc)
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return {}

    req = GetParameters.Request()
    req.names = list(names)

    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        return {}

    out = {}
    for key, value in zip(req.names, future.result().values):
        if value.string_value:
            out[key] = value.string_value
    return out


def _print_topic_table(node: Node, required_topics):
    print('Observed topic types:')
    graph = dict(node.get_topic_names_and_types())
    for topic in required_topics:
        types = graph.get(topic)
        if types:
            print(f'  OK   {topic} : {", ".join(types)}')
        else:
            print(f'  MISS {topic}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Validate live RealSense streams.')
    parser.add_argument('--timeout', type=float, default=12.0,
                        help='Seconds to wait for required messages (default: 12)')
    parser.add_argument('--topic-node', type=str, default='/ugv_perception_node',
                        help='Node to query for topic params (default: /ugv_perception_node)')
    parser.add_argument('--rgb-image-topic', type=str, default='')
    parser.add_argument('--depth-image-topic', type=str, default='')
    parser.add_argument('--camera-info-topic', type=str, default='')
    parser.add_argument('--depth-camera-info-topic', type=str, default='')
    parser.add_argument('--color-metadata-topic', type=str, default='')
    parser.add_argument('--depth-metadata-topic', type=str, default='')
    parser.add_argument('--extrinsics-topic', type=str, default='')
    parsed = parser.parse_args(args=args)

    rclpy.init()

    resolver = Node('realsense_topic_resolver')
    resolved = _get_parameters(
        resolver,
        parsed.topic_node,
        [
            'rgb_image_topic',
            'depth_image_topic',
            'camera_info_topic',
            'depth_camera_info_topic',
        ],
    )

    rgb_topic = parsed.rgb_image_topic or resolved.get('rgb_image_topic', FALLBACK_RGB_TOPIC)
    depth_topic = parsed.depth_image_topic or resolved.get('depth_image_topic', FALLBACK_DEPTH_TOPIC)
    color_info_topic = (
        parsed.camera_info_topic
        or resolved.get('camera_info_topic')
        or _derive_topic(rgb_topic, '/image_raw', '/camera_info')
        or FALLBACK_COLOR_INFO_TOPIC
    )
    depth_info_topic = (
        parsed.depth_camera_info_topic
        or resolved.get('depth_camera_info_topic')
        or _derive_topic(depth_topic, '/image_rect_raw', '/camera_info')
        or FALLBACK_DEPTH_INFO_TOPIC
    )

    auto_color_meta, auto_depth_meta, auto_extrinsics = _derive_aux_topics(
        rgb_topic, depth_topic
    )
    color_meta_topic = parsed.color_metadata_topic or auto_color_meta
    depth_meta_topic = parsed.depth_metadata_topic or auto_depth_meta
    extrinsics_topic = parsed.extrinsics_topic or auto_extrinsics

    resolver.destroy_node()

    node = RealSenseStreamValidator(
        color_info_topic=color_info_topic,
        color_image_topic=rgb_topic,
        depth_info_topic=depth_info_topic,
        depth_image_topic=depth_topic,
    )

    required_topics = [
        color_info_topic,
        rgb_topic,
        depth_info_topic,
        depth_topic,
    ]
    if color_meta_topic:
        required_topics.append(color_meta_topic)
    if depth_meta_topic:
        required_topics.append(depth_meta_topic)
    if extrinsics_topic:
        required_topics.append(extrinsics_topic)

    deadline = time.monotonic() + parsed.timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.has_core_messages() and node.color_count >= 5 and node.depth_count >= 5:
            break

    errors = []
    warnings = []
    graph = dict(node.get_topic_names_and_types())

    for topic in required_topics:
        if topic not in graph:
            errors.append(f'Missing topic in ROS graph: {topic}')

    if node.color_image is None:
        errors.append(f'No messages on {rgb_topic}')
    if node.depth_image is None:
        errors.append(f'No messages on {depth_topic}')
    if node.color_info is None:
        errors.append(f'No messages on {color_info_topic}')
    if node.depth_info is None:
        errors.append(f'No messages on {depth_info_topic}')

    if node.color_image is not None:
        enc = node.color_image.encoding
        if enc not in ALLOWED_COLOR_ENCODINGS:
            warnings.append(
                f'Unexpected color encoding "{enc}" (expected one of '
                f'{sorted(ALLOWED_COLOR_ENCODINGS)})'
            )

    if node.depth_image is not None:
        enc = node.depth_image.encoding
        if enc not in ALLOWED_DEPTH_ENCODINGS:
            errors.append(
                f'Unexpected depth encoding "{enc}" (expected 16UC1 for UGV depth sampling)'
            )

    if node.color_image is not None and node.color_info is not None:
        if (node.color_image.width, node.color_image.height) != (
            node.color_info.width,
            node.color_info.height,
        ):
            errors.append(
                'Color image resolution does not match color CameraInfo '
                f'({node.color_image.width}x{node.color_image.height} vs '
                f'{node.color_info.width}x{node.color_info.height})'
            )

    if node.depth_image is not None and node.depth_info is not None:
        if (node.depth_image.width, node.depth_image.height) != (
            node.depth_info.width,
            node.depth_info.height,
        ):
            errors.append(
                'Depth image resolution does not match depth CameraInfo '
                f'({node.depth_image.width}x{node.depth_image.height} vs '
                f'{node.depth_info.width}x{node.depth_info.height})'
            )

    # Current ugv_node samples depth directly at RGB pixels; that requires
    # depth aligned to color (same frame + same resolution).
    if node.color_image is not None and node.depth_image is not None:
        if (node.color_image.width, node.color_image.height) != (
            node.depth_image.width,
            node.depth_image.height,
        ):
            errors.append(
                'Color/depth resolutions differ. Current ugv_node expects pixel-aligned depth.'
            )

    if node.color_info is not None and node.depth_info is not None:
        color_frame = node.color_info.header.frame_id
        depth_frame = node.depth_info.header.frame_id
        if color_frame and depth_frame and color_frame != depth_frame:
            warnings.append(
                f'Color and depth frame_ids differ ({color_frame} vs {depth_frame}). '
                'If depth is not aligned-to-color, 3D output can be wrong with current ugv_node.'
            )

    color_rate = node.rate_hz(node.color_first_t, node.color_last_t, node.color_count)
    depth_rate = node.rate_hz(node.depth_first_t, node.depth_last_t, node.depth_count)

    print('\n=== RealSense Stream Validation ===')
    print('Resolved topics:')
    print(f'  RGB image:          {rgb_topic}')
    print(f'  Depth image:        {depth_topic}')
    print(f'  Color CameraInfo:   {color_info_topic}')
    print(f'  Depth CameraInfo:   {depth_info_topic}')
    if color_meta_topic:
        print(f'  Color metadata:     {color_meta_topic}')
    if depth_meta_topic:
        print(f'  Depth metadata:     {depth_meta_topic}')
    if extrinsics_topic:
        print(f'  Depth->color extr.: {extrinsics_topic}')

    _print_topic_table(node, required_topics)

    print('\nMessage stats:')
    print(f'  Color image msgs: {node.color_count}, approx rate: {color_rate:.2f} Hz')
    print(f'  Depth image msgs: {node.depth_count}, approx rate: {depth_rate:.2f} Hz')
    if node.color_image is not None:
        print(
            '  Color stream: '
            f'{node.color_image.width}x{node.color_image.height}, '
            f'encoding={node.color_image.encoding}'
        )
    if node.depth_image is not None:
        print(
            '  Depth stream: '
            f'{node.depth_image.width}x{node.depth_image.height}, '
            f'encoding={node.depth_image.encoding}'
        )
    if node.color_info is not None:
        print(f'  Color frame_id: {node.color_info.header.frame_id}')
    if node.depth_info is not None:
        print(f'  Depth frame_id: {node.depth_info.header.frame_id}')

    if warnings:
        print('\nWarnings:')
        for warning in warnings:
            print(f'  - {warning}')

    if errors:
        print('\nFAILED:')
        for err in errors:
            print(f'  - {err}')
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    print('\nPASS: Required RealSense streams are available and compatible.')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
