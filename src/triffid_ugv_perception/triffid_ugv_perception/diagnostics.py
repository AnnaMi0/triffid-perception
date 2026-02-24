"""
TRIFFID Diagnostics Node
==========================
Publishes diagnostic_msgs/DiagnosticArray with health status for every
perception node.  Monitors:
  - Topic liveness (are messages arriving?)
  - Message rates (Hz)
  - Timestamp freshness (staleness < threshold → OK)
  - Depth-RGB time sync (|t_depth - t_rgb| < threshold)
  - TF availability (can we look up the expected transform?)
  - GeoJSON publish rate
  - Model load status

Publishes to:
  /diagnostics              (diagnostic_msgs/DiagnosticArray)  – standard
  /triffid/heartbeat        (std_msgs/String)                  – simple JSON heartbeat
"""

import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection2DArray

from tf2_ros import Buffer, TransformListener


class DiagnosticsNode(Node):
    """Publish /diagnostics + /triffid/heartbeat with health status."""

    # Tunables
    DIAG_PERIOD = 2.0           # seconds between diagnostic publishes
    STALE_THRESHOLD = 5.0       # seconds before a topic is "stale"
    SYNC_THRESHOLD = 0.15       # max |t_depth - t_rgb| in seconds
    MIN_RATE_HZ = 1.0           # below this, topic is "slow"

    def __init__(self):
        super().__init__('triffid_diagnostics')

        # Parameters
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_frame', 'camera_front_link')
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # Bookkeeping
        self._stamps = {}    # topic_name → last header.stamp  (ROS time)
        self._wall = {}      # topic_name → last wall-clock receipt (float)
        self._counts = {}    # topic_name → message count since last diag
        self._rates = {}     # topic_name → estimated Hz
        self._last_diag_time = time.monotonic()

        # Depth-RGB sync tracking
        self._last_rgb_stamp = None
        self._last_depth_stamp = None
        self._sync_delta = None  # latest |t_depth - t_rgb| in seconds

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS for sensor topics
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribe to everything we want to monitor
        self._make_sub(Image, '/camera_front/raw_image', sensor_qos)
        self._make_sub(Image, '/camera_front/realsense_front/depth/image_rect_raw', sensor_qos)
        self._make_sub(CameraInfo, '/camera_front/realsense_front/depth/camera_info',
                       QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self._make_sub(Detection3DArray, '/ugv/perception/detections_3d', 10)
        self._make_sub(Detection2DArray, '/uav/perception/detections_2d', 10)
        self._make_sub(String, '/triffid/geojson', 10)

        # Extra callbacks for sync tracking
        self.create_subscription(
            Image, '/camera_front/raw_image', self._rgb_stamp_cb, sensor_qos)
        self.create_subscription(
            Image, '/camera_front/realsense_front/depth/image_rect_raw',
            self._depth_stamp_cb, sensor_qos)

        # Publishers
        self.pub_diag = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.pub_heartbeat = self.create_publisher(String, '/triffid/heartbeat', 10)

        # Timer
        self.create_timer(self.DIAG_PERIOD, self._publish_diagnostics)

        self.get_logger().info('Diagnostics node started.')

    # Subscription helpers

    def _make_sub(self, msg_type, topic, qos):
        """Create a lightweight subscription that only records timestamps."""
        self._counts[topic] = 0
        self._wall[topic] = 0.0
        self._rates[topic] = 0.0

        def cb(msg, _topic=topic):
            now = time.monotonic()
            self._wall[_topic] = now
            self._counts[_topic] += 1
            # Try to get the ROS stamp
            stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
            if stamp is not None:
                self._stamps[_topic] = stamp

        self.create_subscription(msg_type, topic, cb, qos)

    # Sync tracking

    def _rgb_stamp_cb(self, msg: Image):
        self._last_rgb_stamp = msg.header.stamp
        self._update_sync()

    def _depth_stamp_cb(self, msg: Image):
        self._last_depth_stamp = msg.header.stamp
        self._update_sync()

    def _update_sync(self):
        if self._last_rgb_stamp is None or self._last_depth_stamp is None:
            return
        t_rgb = self._last_rgb_stamp.sec + self._last_rgb_stamp.nanosec * 1e-9
        t_depth = self._last_depth_stamp.sec + self._last_depth_stamp.nanosec * 1e-9
        self._sync_delta = abs(t_depth - t_rgb)

    # Diagnostics publisher

    def _publish_diagnostics(self):
        now = time.monotonic()
        dt = now - self._last_diag_time
        if dt <= 0:
            dt = self.DIAG_PERIOD

        # Compute rates
        for topic in self._counts:
            self._rates[topic] = self._counts[topic] / dt
            self._counts[topic] = 0
        self._last_diag_time = now

        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()

        # Per-topic status
        critical_topics = {
            '/camera_front/raw_image': 'Front RGB',
            '/camera_front/realsense_front/depth/image_rect_raw': 'Front Depth',
            '/camera_front/realsense_front/depth/camera_info': 'Depth CameraInfo',
        }
        output_topics = {
            '/ugv/perception/detections_3d': 'UGV Detections',
            '/uav/perception/detections_2d': 'UAV Detections',
            '/triffid/geojson': 'GeoJSON Output',
        }

        for topic, label in {**critical_topics, **output_topics}.items():
            status = DiagnosticStatus()
            status.name = f'triffid/{label}'
            status.hardware_id = 'perception'

            age = now - self._wall.get(topic, 0.0) if self._wall.get(topic, 0.0) > 0 else float('inf')
            hz = self._rates.get(topic, 0.0)

            status.values.append(KeyValue(key='rate_hz', value=f'{hz:.1f}'))
            status.values.append(KeyValue(key='age_sec', value=f'{age:.2f}'))

            if age == float('inf'):
                status.level = DiagnosticStatus.STALE
                status.message = 'No messages received'
            elif age > self.STALE_THRESHOLD:
                status.level = DiagnosticStatus.WARN
                status.message = f'Stale ({age:.1f}s since last msg)'
            elif hz < self.MIN_RATE_HZ and topic in critical_topics:
                status.level = DiagnosticStatus.WARN
                status.message = f'Low rate ({hz:.1f} Hz)'
            else:
                status.level = DiagnosticStatus.OK
                status.message = f'OK ({hz:.1f} Hz)'

            diag_msg.status.append(status)

        # Depth-RGB sync check
        sync_status = DiagnosticStatus()
        sync_status.name = 'triffid/Depth-RGB Sync'
        sync_status.hardware_id = 'perception'

        if self._sync_delta is not None:
            sync_status.values.append(
                KeyValue(key='delta_sec', value=f'{self._sync_delta:.4f}'))
            if self._sync_delta > self.SYNC_THRESHOLD:
                sync_status.level = DiagnosticStatus.WARN
                sync_status.message = (
                    f'Depth-RGB desync: {self._sync_delta*1000:.1f} ms '
                    f'(threshold {self.SYNC_THRESHOLD*1000:.0f} ms)')
            else:
                sync_status.level = DiagnosticStatus.OK
                sync_status.message = f'Synced ({self._sync_delta*1000:.1f} ms)'
        else:
            sync_status.level = DiagnosticStatus.STALE
            sync_status.message = 'Waiting for both RGB and Depth'
        diag_msg.status.append(sync_status)

        # TF availability check
        tf_status = DiagnosticStatus()
        tf_status.name = 'triffid/TF Transform'
        tf_status.hardware_id = 'perception'
        tf_status.values.append(
            KeyValue(key='source', value=self.camera_frame))
        tf_status.values.append(
            KeyValue(key='target', value=self.target_frame))

        try:
            self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05))
            tf_status.level = DiagnosticStatus.OK
            tf_status.message = f'{self.camera_frame} → {self.target_frame} available'
        except Exception as e:
            tf_status.level = DiagnosticStatus.WARN
            tf_status.message = f'TF lookup failed: {e}'
        diag_msg.status.append(tf_status)

        # Publish diagnostic array
        self.pub_diag.publish(diag_msg)

        # Heartbeat (simple JSON summary)
        def _lvl(level):
            return int.from_bytes(level, 'little') if isinstance(level, bytes) else int(level)
        overall_level = max(_lvl(s.level) for s in diag_msg.status)
        level_names = {0: 'OK', 1: 'WARN', 2: 'ERROR', 3: 'STALE'}
        heartbeat = {
            'node': 'triffid_diagnostics',
            'stamp': diag_msg.header.stamp.sec,
            'status': level_names.get(overall_level, 'UNKNOWN'),
            'topics': {
                t: {'hz': round(self._rates.get(t, 0.0), 1),
                     'alive': (now - self._wall.get(t, 0.0)) < self.STALE_THRESHOLD
                     if self._wall.get(t, 0.0) > 0 else False}
                for t in list(critical_topics) + list(output_topics)
            },
        }
        hb_msg = String()
        hb_msg.data = json.dumps(heartbeat)
        self.pub_heartbeat.publish(hb_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiagnosticsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
