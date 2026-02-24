"""
TRIFFID UGV Perception Node
============================
Pipeline:
  1. Subscribe to RGB + Depth + CameraInfo + TF
  2. Run detection/segmentation model on RGB
  3. For each detection, sample depth at mask/bbox centroid
  4. Back-project (u,v,Z) → 3D point in camera frame
  5. Transform to map frame via TF2
  6. Track objects with persistent IDs (never reused)
  7. Publish vision_msgs/Detection3DArray

Topic mapping (from rosbag):
  IN:  /camera_front/raw_image                              (sensor_msgs/Image)
  IN:  /camera_front/realsense_front/depth/image_rect_raw   (sensor_msgs/Image)
  IN:  /camera_front/realsense_front/depth/camera_info      (sensor_msgs/CameraInfo)
  IN:  /tf, /tf_static
  OUT: /ugv/perception/detections_3d                        (vision_msgs/Detection3DArray)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped, Point

from cv_bridge import CvBridge
import numpy as np

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa – registers PoseStamped transform

from triffid_ugv_perception.tracker import IoUTracker

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


# test classes
TARGET_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


class UGVPerceptionNode(Node):
    """Main UGV perception node."""

    def __init__(self):
        super().__init__('ugv_perception_node')

        # Parameters
        self.declare_parameter('model_path', 'yolo11n.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('target_frame', 'map')

        self.model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.target_frame = self.get_parameter('target_frame').value

        # Load model
        if _HAS_YOLO:
            self.get_logger().info(f'Loading YOLO model: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('Model loaded.')
        else:
            self.model = None
            self.get_logger().error(
                'ultralytics not installed — no detections will be produced. '
                'Install it with: pip install ultralytics'
            )

        # State
        self.bridge = CvBridge()
        self.camera_info = None          # latest CameraInfo
        self.depth_image = None          # latest depth frame
        self.depth_stamp = None
        self.tracker = IoUTracker()

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS profiles
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.sub_rgb = self.create_subscription(
            Image,
            '/camera_front/raw_image',
            self.rgb_callback,
            sensor_qos,
        )
        self.sub_depth = self.create_subscription(
            Image,
            '/camera_front/realsense_front/depth/image_rect_raw',
            self.depth_callback,
            sensor_qos,
        )
        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            '/camera_front/realsense_front/depth/camera_info',
            self.cam_info_callback,
            reliable_qos,
        )

        # Publisher
        self.pub_det3d = self.create_publisher(
            Detection3DArray,
            '/ugv/perception/detections_3d',
            10,
        )

        self.get_logger().info('UGV Perception node started.')
        self.get_logger().info(f'  RGB topic:   /camera_front/raw_image')
        self.get_logger().info(f'  Depth topic: /camera_front/realsense_front/depth/image_rect_raw')
        self.get_logger().info(f'  Output:      /ugv/perception/detections_3d')

    # Callbacks

    def cam_info_callback(self, msg: CameraInfo):
        """Store latest camera intrinsics."""
        self.camera_info = msg

    def depth_callback(self, msg: Image):
        """Store latest depth image."""
        try:
            # RealSense depth is typically 16UC1 (millimetres)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image):
        """Main processing: runs on every RGB frame."""
        if self.camera_info is None:
            self.get_logger().warn('Waiting for CameraInfo...', throttle_duration_sec=5.0)
            return
        if self.depth_image is None:
            self.get_logger().warn('Waiting for Depth image...', throttle_duration_sec=5.0)
            return

        # Convert RGB
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # Run detection
        raw_detections = self._detect(cv_image)

        # Back-project to 3D and transform to map frame
        detections_3d = []
        camera_frame = msg.header.frame_id or 'camera_front_link'

        for det in raw_detections:
            x1, y1, x2, y2 = det['bbox']
            cls_id = det['class_id']
            cls_name = det['class_name']
            confidence = det['confidence']

            # Centroid pixel
            cu = int((x1 + x2) / 2)
            cv_ = int((y1 + y2) / 2)

            # Sample depth (median of small patch for robustness)
            z = self._sample_depth(cu, cv_)
            if z is None or z <= 0.0:
                continue  # no valid depth

            # Back-project pixel to camera-frame 3D point
            pt_cam = self._backproject(cu, cv_, z)
            if pt_cam is None:
                continue

            # Transform to target frame (map)
            pt_target = self._transform_point(
                pt_cam, camera_frame, self.target_frame, msg.header.stamp
            )

            detections_3d.append({
                'position': pt_target,
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
            })

        # Track across frames
        tracked = self.tracker.update(detections_3d)

        # Publish Detection3DArray
        det_array = Detection3DArray()
        det_array.header.stamp = msg.header.stamp
        det_array.header.frame_id = self.target_frame

        for t in tracked:
            d3d = Detection3D()
            d3d.header = det_array.header

            # Set position
            if t['position'] is not None:
                d3d.bbox.center.position.x = t['position'][0]
                d3d.bbox.center.position.y = t['position'][1]
                d3d.bbox.center.position.z = t['position'][2]

            # Hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t['class_name'])
            hyp.hypothesis.score = t['confidence']
            d3d.results.append(hyp)

            # Tracking ID
            d3d.id = str(t['track_id'])

            det_array.detections.append(d3d)

        self.pub_det3d.publish(det_array)

        if tracked:
            self.get_logger().info(
                f'Published {len(tracked)} detections '
                f'(IDs: {[t["track_id"] for t in tracked]})',
                throttle_duration_sec=1.0,
            )

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

    # Depth sampling and geometry

    def _sample_depth(self, u, v, patch_size=5):
        """Sample depth at pixel (u,v) using median of a small patch.
        Returns depth in metres.
        """
        if self.depth_image is None:
            return None
        h, w = self.depth_image.shape[:2]
        half = patch_size // 2
        u = np.clip(u, half, w - half - 1)
        v = np.clip(v, half, h - half - 1)
        patch = self.depth_image[v - half:v + half + 1, u - half:u + half + 1]
        patch = patch.astype(np.float64)
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        depth_val = float(np.median(valid))
        # RealSense D430 depth is typically in millimetres
        if depth_val > 100:  # likely mm
            depth_val /= 1000.0
        return depth_val

    def _backproject(self, u, v, z):
        """Back-project pixel (u,v) at depth Z to 3D point in camera frame.
        Uses: X = (u - cx) * Z / fx
              Y = (v - cy) * Z / fy
        """
        if self.camera_info is None:
            return None
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        if fx == 0 or fy == 0:
            return None

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return (x, y, z)

    def _transform_point(self, point_cam, source_frame, target_frame, stamp):
        """Transform a 3D point from source_frame to target_frame using TF2.
        Returns (x, y, z) in target frame, or the original point if TF fails.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = source_frame
        pose_msg.pose.position.x = point_cam[0]
        pose_msg.pose.position.y = point_cam[1]
        pose_msg.pose.position.z = point_cam[2]
        pose_msg.pose.orientation.w = 1.0

        try:
            transformed = self.tf_buffer.transform(
                pose_msg, target_frame, timeout=rclpy.duration.Duration(seconds=0.1)
            )
            return (
                transformed.pose.position.x,
                transformed.pose.position.y,
                transformed.pose.position.z,
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF {source_frame}→{target_frame} failed: {e}',
                throttle_duration_sec=5.0,
            )
            # Fall back to camera-frame coordinates
            return point_cam


def main(args=None):
    rclpy.init(args=args)
    node = UGVPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
