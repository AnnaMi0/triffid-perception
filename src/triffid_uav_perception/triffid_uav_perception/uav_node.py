"""
TRIFFID UAV Perception Node (Skeleton)
========================================
This is a placeholder

Pipeline (to be implemented):
  1. Subscribe to RGB image from UAV camera
  2. Subscribe to UAV GPS + orientation + camera tilt
  3. Run detection/segmentation model on RGB
  4. Track objects with persistent IDs (ByteTrack / DeepSORT)
  5. Publish vision_msgs/Detection2DArray
  6. (Optional) Geo-project detections to lat/lon

Expected topics (MUST be confirmed with UAV rosbag):
  IN:  /uav/camera/image_raw           (sensor_msgs/Image)        – placeholder
  IN:  /uav/gps/fix                    (sensor_msgs/NavSatFix)    – placeholder
  IN:  /uav/camera/tilt                (geometry_msgs/QuaternionStamped) – placeholder
  OUT: /uav/perception/detections_2d   (vision_msgs/Detection2DArray)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, NavSatFix
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import QuaternionStamped

from cv_bridge import CvBridge

# Reuse tracker from UGV package (same logic applies)
try:
    from triffid_ugv_perception.tracker import IoUTracker
except ImportError:
    # Standalone fallback – copy tracker.py into this package if needed
    from triffid_uav_perception.tracker_stub import IoUTracker  # noqa

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


# Target classes (same subset, adjust as needed)
TARGET_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


class UAVPerceptionNode(Node):
    """UAV perception node – skeleton, ready to fill in once topics are known."""

    def __init__(self):
        super().__init__('uav_perception_node')

        # Parameters
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('rgb_topic', '/uav/camera/image_raw')
        self.declare_parameter('gps_topic', '/uav/gps/fix')

        self.model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        rgb_topic = self.get_parameter('rgb_topic').value
        gps_topic = self.get_parameter('gps_topic').value

        # Load model
        if _HAS_YOLO:
            self.get_logger().info(f'Loading YOLO model: {self.model_path}')
            self.model = YOLO(self.model_path)
        else:
            self.model = None
            self.get_logger().error(
                'ultralytics not installed — no detections will be produced.'
            )

        # State
        self.bridge = CvBridge()
        self.latest_gps = None
        self.tracker = IoUTracker()

        # QoS
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.sub_rgb = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, sensor_qos,
        )
        self.sub_gps = self.create_subscription(
            NavSatFix, gps_topic, self.gps_callback, 10,
        )

        # Publisher
        self.pub_det2d = self.create_publisher(
            Detection2DArray,
            '/uav/perception/detections_2d',
            10,
        )

        self.get_logger().info('UAV Perception node started (skeleton).')
        self.get_logger().info(f'  RGB topic: {rgb_topic}')
        self.get_logger().info(f'  Output:    /uav/perception/detections_2d')

    # Callbacks

    def gps_callback(self, msg: NavSatFix):
        self.latest_gps = msg

    def rgb_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # Detect
        raw_detections = self._detect(cv_image)

        # Track across frames
        tracked = self.tracker.update(raw_detections)

        # Publish Detection2DArray
        det_array = Detection2DArray()
        det_array.header.stamp = msg.header.stamp
        det_array.header.frame_id = msg.header.frame_id or 'uav_camera_frame'

        for t in tracked:
            d2d = Detection2D()
            d2d.header = det_array.header

            x1, y1, x2, y2 = t['bbox']
            d2d.bbox.center.position.x = (x1 + x2) / 2.0
            d2d.bbox.center.position.y = (y1 + y2) / 2.0
            d2d.bbox.size_x = float(x2 - x1)
            d2d.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t['class_name'])
            hyp.hypothesis.score = t['confidence']
            d2d.results.append(hyp)

            d2d.id = str(t['track_id'])

            det_array.detections.append(d2d)

        self.pub_det2d.publish(det_array)

    # Detection

    def _detect(self, cv_image):
        """Run YOLO on a BGR image. Returns list of detection dicts."""
        if self.model is None:
            return []

        results = self.model(cv_image, conf=self.conf_thresh, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': cls_id,
                    'class_name': TARGET_CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                })
        return detections


def main(args=None):
    rclpy.init(args=args)
    node = UAVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
