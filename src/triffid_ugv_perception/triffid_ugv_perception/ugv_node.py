"""
TRIFFID UGV Perception Node
============================
Cross-camera depth–RGB fusion pipeline (revised plan):

  RGB and depth come from **different sensors** with different resolutions
  and different optical frames.  We cannot simply sample depth at the RGB
  pixel coordinate.  Instead:

  1. Subscribe to RGB + Depth + both CameraInfo topics + TF
  2. Run YOLO on the RGB image → 2D bounding boxes
  3. Sample a coarse grid of depth pixels over the depth image
  4. Back-project each sampled depth pixel → 3D point in f_depth_optical_frame
  5. Transform those 3D points → f_oc_link (RGB camera frame) via TF2
  6. Project 3D points → RGB pixel coordinates using RGB intrinsics
  7. For each detection bbox, keep only projected points that fall inside it
  8. Median of kept 3D points (in f_oc_link) → object position
  9. Transform object position f_oc_link → b2/base_link
  10. Assign persistent tracking ID (IoU tracker on RGB bboxes)
  11. Publish vision_msgs/Detection3DArray in b2/base_link

Topic mapping (from rosbag):
  IN:  /camera_front/raw_image                              (sensor_msgs/Image, bgr8, 1280×720, frame: f_oc_link)
  IN:  /camera_front/camera_info                            (sensor_msgs/CameraInfo, frame: f_oc_link)
  IN:  /camera_front/realsense_front/depth/image_rect_raw   (sensor_msgs/Image, 16UC1 mm, 640×480, frame: f_depth_optical_frame)
  IN:  /camera_front/realsense_front/depth/camera_info      (sensor_msgs/CameraInfo, frame: f_depth_optical_frame)
  IN:  /tf, /tf_static
  OUT: /ugv/detections/front/detections_3d                    (vision_msgs/Detection3DArray, frame: b2/base_link)
  OUT: /ugv/detections/front/segmentation                      (sensor_msgs/Image, bgr8, annotated masks)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
import cv2
import numpy as np

from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa – registers PoseStamped transform

from triffid_ugv_perception.tracker import ByteTracker

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False


# TRIFFID custom model classes (yolo11l-seg fine-tuned)
TARGET_CLASSES = {
    0: 'Water',
    1: 'Fence',
    2: 'Green tree',
    3: 'Helmet',
    4: 'Flame',
    5: 'Smoke',
    6: 'First responder',
    7: 'Destroyed vehicle',
    8: 'Fire hose',
    9: 'SCBA',
    10: 'Boot',
    11: 'Green plant',
    12: 'Mask',
    13: 'Window',
    14: 'Building',
    15: 'Destroyed building',
    16: 'Debris',
    17: 'Ladder',
    18: 'Dirt road',
    19: 'Dry tree',
    20: 'Wall',
    21: 'Civilian vehicle',
    22: 'Road',
    23: 'Citizen',
    24: 'Green grass',
    25: 'Pole',
    26: 'Boat',
    27: 'Pavement',
    28: 'Dry grass',
    29: 'Animal',
    30: 'Excavator',
    31: 'Door',
    32: 'Mud',
    33: 'Barrier',
    34: 'Hole in the ground',
    35: 'Bag',
    36: 'Burnt tree',
    37: 'Ambulance',
    38: 'Fire truck',
    39: 'Cone',
    40: 'Bicycle',
    41: 'Tower',
    42: 'Silo',
    43: 'Military personnel',
    44: 'Burnt grass',
    45: 'Ax',
    46: 'Glove',
    47: 'Crane',
    48: 'Stairs',
    49: 'Dry plant',
    50: 'Furniture',
    51: 'Tank',
    52: 'Protective glasses',
    53: 'Barrel',
    54: 'Shovel',
    55: 'Fire hydrant',
    56: 'Police vehicle',
    57: 'Burnt plant',
    58: 'Army vehicle',
    59: 'Chainsaw',
    60: 'aerial vehicle',
    61: 'Lifesaver',
    62: 'Extinguisher',
}

# Frame IDs (from bag tf_static)
DEPTH_FRAME = 'f_depth_optical_frame'
RGB_FRAME = 'f_oc_link'
BASE_FRAME = 'b2/base_link'

# Depth grid sampling defaults
DEFAULT_GRID_STEP_U = 64   # pixels between horizontal samples (640/64 ≈ 10)
DEFAULT_GRID_STEP_V = 48   # pixels between vertical samples   (480/48 ≈ 10)


class UGVPerceptionNode(Node):
    """Main UGV perception node – cross-camera depth–RGB fusion."""

    def __init__(self):
        super().__init__('ugv_perception_node')

        # ── Parameters ──────────────────────────────────────────────
        self.declare_parameter('model_path', '/ws/best.pt')
        self.declare_parameter('confidence_threshold', 0.35)
        self.declare_parameter('target_frame', BASE_FRAME)
        self.declare_parameter('depth_grid_step_u', DEFAULT_GRID_STEP_U)
        self.declare_parameter('depth_grid_step_v', DEFAULT_GRID_STEP_V)
        self.declare_parameter('use_dummy_detections', False)
        self.declare_parameter('yolo_imgsz', 1280)
        self.declare_parameter('tracker_iou_threshold', 0.30)
        self.declare_parameter('tracker_iou_threshold_low', 0.15)
        self.declare_parameter('tracker_conf_high', 0.40)
        self.declare_parameter('tracker_max_age', 30)
        self.declare_parameter('tracker_n_init', 3)
        self.declare_parameter('tracker_pos_gate', 2.0)
        self.declare_parameter('publish_debug_image', True)

        self.model_path = self.get_parameter('model_path').value
        self.conf_thresh = self.get_parameter('confidence_threshold').value
        self.target_frame = self.get_parameter('target_frame').value
        self.grid_step_u = self.get_parameter('depth_grid_step_u').value
        self.grid_step_v = self.get_parameter('depth_grid_step_v').value
        self.use_dummy = self.get_parameter('use_dummy_detections').value
        self.yolo_imgsz = self.get_parameter('yolo_imgsz').value
        self.tracker_iou = self.get_parameter('tracker_iou_threshold').value
        self.tracker_iou_low = self.get_parameter('tracker_iou_threshold_low').value
        self.tracker_conf_high = self.get_parameter('tracker_conf_high').value
        self.tracker_max_age = self.get_parameter('tracker_max_age').value
        self.tracker_n_init = self.get_parameter('tracker_n_init').value
        self.tracker_pos_gate = self.get_parameter('tracker_pos_gate').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value

        # ── YOLO model ──────────────────────────────────────────────
        if _HAS_YOLO:
            self.get_logger().info(f'Loading YOLO model: {self.model_path}')
            self.model = YOLO(self.model_path)
            self.get_logger().info('Model loaded.')
        else:
            self.model = None
            self.get_logger().error(
                'ultralytics not installed — no detections will be produced. '
                'Install with: pip install ultralytics'
            )

        # ── State ───────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.rgb_camera_info = None      # /camera_front/camera_info
        self.depth_camera_info = None    # /camera_front/realsense_front/depth/camera_info
        self.depth_image = None          # latest depth frame (uint16, mm)
        self.depth_stamp = None
        self.tracker = ByteTracker(
            iou_threshold=float(self.tracker_iou),
            iou_threshold_low=float(self.tracker_iou_low),
            conf_threshold_high=float(self.tracker_conf_high),
            max_age=int(self.tracker_max_age),
            n_init=int(self.tracker_n_init),
            pos_gate=float(self.tracker_pos_gate),
        )

        # ── TF2 ─────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── QoS ─────────────────────────────────────────────────────
        # The rosbag records sensor topics as RELIABLE + VOLATILE.
        # We match that exactly so ros2 bag play data flows correctly.
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ─────────────────────────────────────────────
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
        self.sub_rgb_info = self.create_subscription(
            CameraInfo,
            '/camera_front/camera_info',
            self.rgb_info_callback,
            reliable_qos,
        )
        self.sub_depth_info = self.create_subscription(
            CameraInfo,
            '/camera_front/realsense_front/depth/camera_info',
            self.depth_info_callback,
            reliable_qos,
        )

        # ── Publishers ───────────────────────────────────────────────
        self.pub_det3d = self.create_publisher(
            Detection3DArray,
            '/ugv/detections/front/detections_3d',
            10,
        )
        self.pub_seg = self.create_publisher(
            Image,
            '/ugv/detections/front/segmentation',
            10,
        )
        self.pub_debug = self.create_publisher(
            Image,
            '/ugv/detections/front/debug_image',
            10,
        )

        self.get_logger().info('UGV Perception node started (cross-camera pipeline).')
        self.get_logger().info(f'  RGB topic:       /camera_front/raw_image  (frame: {RGB_FRAME})')
        self.get_logger().info(f'  Depth topic:     /camera_front/realsense_front/depth/image_rect_raw  (frame: {DEPTH_FRAME})')
        self.get_logger().info(f'  Output topic:    /ugv/detections/front/detections_3d  (frame: {self.target_frame})')
        self.get_logger().info(f'  Seg topic:       /ugv/detections/front/segmentation  (mono8 label map)')
        self.get_logger().info(f'  Debug topic:     /ugv/detections/front/debug_image  (RGB+ID overlay)')
        self.get_logger().info(f'  Depth grid step: {self.grid_step_u}×{self.grid_step_v}')
        self.get_logger().info(
            f'  Tracker: IoU={self.tracker_iou:.2f}, max_age={int(self.tracker_max_age)}'
        )
        if self.use_dummy:
            self.get_logger().warn('*** DUMMY DETECTION MODE — bypassing YOLO ***')

    # ─── Callbacks ──────────────────────────────────────────────────

    def rgb_info_callback(self, msg: CameraInfo):
        """Store latest RGB camera intrinsics."""
        if self.rgb_camera_info is None:
            self.get_logger().info(
                f'RGB CameraInfo received: {msg.width}x{msg.height}, '
                f'frame={msg.header.frame_id}')
        self.rgb_camera_info = msg

    def depth_info_callback(self, msg: CameraInfo):
        """Store latest depth camera intrinsics."""
        if self.depth_camera_info is None:
            self.get_logger().info(
                f'Depth CameraInfo received: {msg.width}x{msg.height}, '
                f'frame={msg.header.frame_id}')
        self.depth_camera_info = msg

    def depth_callback(self, msg: Image):
        """Store latest depth image (16UC1, millimetres)."""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            self.depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def rgb_callback(self, msg: Image):
        """Main processing trigger – runs on every RGB frame."""

        # Guard: need both camera infos and a depth image
        if self.rgb_camera_info is None:
            self.get_logger().warn(
                'Waiting for RGB CameraInfo (/camera_front/camera_info)…',
                throttle_duration_sec=5.0,
            )
            return
        if self.depth_camera_info is None:
            self.get_logger().warn(
                'Waiting for Depth CameraInfo…', throttle_duration_sec=5.0,
            )
            return
        if self.depth_image is None:
            self.get_logger().warn(
                'Waiting for Depth image…', throttle_duration_sec=5.0,
            )
            return

        # Convert RGB image
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        # ── Step 1: YOLO detection on RGB ────────────────────────────
        if self.use_dummy:
            raw_detections = self._dummy_detection(cv_image)
        else:
            raw_detections = self._detect(cv_image)
        if not raw_detections:
            self._publish_empty(msg.header.stamp)
            self._publish_debug_overlay(cv_image, [], msg.header)
            return
        
        # ── Step 2: Build 3D candidate cloud from depth grid ────────
        pts_rgb_frame = self._depth_grid_to_rgb_frame(msg.header.stamp)

        if pts_rgb_frame is None or len(pts_rgb_frame) == 0:
            self.get_logger().warn(
                'No valid depth grid points after transform.',
                throttle_duration_sec=5.0,
            )
            self._publish_empty(msg.header.stamp)
            return

        # ── Step 3: Project 3D points to RGB pixel coords ───────────
        rgb_pixels = self._project_to_rgb(pts_rgb_frame)  # (N, 2) u,v

        # ── Step 4: For each detection, filter & compute median 3D ──
        detections_3d = []
        rgb_h, rgb_w = cv_image.shape[:2]

        for det in raw_detections:
            x1, y1, x2, y2 = det['bbox']
            mask = det.get('mask')  # may be None (dummy mode)

            if mask is not None:
                # Use segmentation mask for precise depth matching:
                # find projected depth points whose RGB pixel lands
                # inside the instance mask (pixel-accurate).
                px_u = rgb_pixels[:, 0].astype(int)
                px_v = rgb_pixels[:, 1].astype(int)
                in_bounds = (
                    (px_u >= 0) & (px_u < rgb_w) &
                    (px_v >= 0) & (px_v < rgb_h)
                )
                inside = np.zeros(len(rgb_pixels), dtype=bool)
                inside[in_bounds] = mask[px_v[in_bounds], px_u[in_bounds]]
            else:
                # Fallback to bbox matching (dummy mode / no mask)
                inside = (
                    (rgb_pixels[:, 0] >= x1) & (rgb_pixels[:, 0] <= x2) &
                    (rgb_pixels[:, 1] >= y1) & (rgb_pixels[:, 1] <= y2)
                )

            n_inside = int(np.sum(inside))
            if n_inside == 0:
                continue  # no depth evidence for this detection

            matched_pts = pts_rgb_frame[inside]  # (M, 3) in f_oc_link

            # Median position in RGB camera frame
            median_pt = np.median(matched_pts, axis=0)  # (3,)

            # ── Step 5: Transform f_oc_link → b2/base_link ──────────
            pt_base = self._transform_point(
                tuple(median_pt), RGB_FRAME, self.target_frame,
                msg.header.stamp,
            )

            # ── Compute 3D bbox extent in base_link ─────────────────
            # Try point-cloud extent first; fall back to bbox
            # back-projection when too few depth points.
            pt_min_cam = np.min(matched_pts, axis=0)
            pt_max_cam = np.max(matched_pts, axis=0)
            cloud_extent = pt_max_cam - pt_min_cam

            if np.any(cloud_extent > 1e-3):
                # Enough spread in the matched points
                corners_cam = np.array([
                    [pt_min_cam[0], pt_min_cam[1], pt_min_cam[2]],
                    [pt_min_cam[0], pt_min_cam[1], pt_max_cam[2]],
                    [pt_min_cam[0], pt_max_cam[1], pt_min_cam[2]],
                    [pt_min_cam[0], pt_max_cam[1], pt_max_cam[2]],
                    [pt_max_cam[0], pt_min_cam[1], pt_min_cam[2]],
                    [pt_max_cam[0], pt_min_cam[1], pt_max_cam[2]],
                    [pt_max_cam[0], pt_max_cam[1], pt_min_cam[2]],
                    [pt_max_cam[0], pt_max_cam[1], pt_max_cam[2]],
                ])
            else:
                # All matched points cluster in ~one cell — estimate
                # extent by back-projecting the 2D bbox corners at
                # the median depth (forward distance in f_oc_link).
                corners_cam = self._bbox_to_3d_corners(
                    x1, y1, x2, y2, median_pt[0]
                )

            corners_base = self._transform_points_batch(
                corners_cam, RGB_FRAME, self.target_frame,
                msg.header.stamp,
            )
            if corners_base is not None:
                extent_base = tuple(
                    float(v) for v in np.ptp(corners_base, axis=0)
                )
            else:
                extent_base = (0.0, 0.0, 0.0)

            detections_3d.append({
                'position': pt_base,
                'extent': extent_base,
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'n_depth_pts': n_inside,
            })

        # ── Step 5.5: 3-D NMS — deduplicate overlapping detections ───
        detections_3d = self._nms_3d(detections_3d, dist_thresh=0.5)

        # ── Step 6: Track across frames (IoU on RGB bboxes) ─────────
        tracked = self.tracker.update(detections_3d)

        # ── Step 7: Publish Detection3DArray ─────────────────────────
        det_array = Detection3DArray()
        det_array.header.stamp = msg.header.stamp       # timestamp from RGB
        det_array.header.frame_id = self.target_frame    # b2/base_link

        for t in tracked:
            d3d = Detection3D()
            d3d.header = det_array.header

            if t['position'] is not None:
                d3d.bbox.center.position.x = float(t['position'][0])
                d3d.bbox.center.position.y = float(t['position'][1])
                d3d.bbox.center.position.z = float(t['position'][2])

            ext = t.get('extent', (0.0, 0.0, 0.0))
            d3d.bbox.size.x = float(ext[0])
            d3d.bbox.size.y = float(ext[1])
            d3d.bbox.size.z = float(ext[2])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t['class_name'])
            hyp.hypothesis.score = float(t['confidence'])
            d3d.results.append(hyp)

            d3d.id = str(t['track_id'])
            det_array.detections.append(d3d)

        self.pub_det3d.publish(det_array)

        # ── Step 8: Publish segmentation label map (mono8) ───────────
        self._publish_segmentation(cv_image, raw_detections, msg.header)
        self._publish_debug_overlay(cv_image, tracked, msg.header)

        if tracked:
            self.get_logger().info(
                f'Published {len(tracked)} detections '
                f'(IDs: {[t["track_id"] for t in tracked]})',
                throttle_duration_sec=1.0,
            )

    # ─── YOLO detection ────────────────────────────────────────────

    def _detect(self, cv_image):
        """Run YOLO-seg on a BGR image.

        Returns list of detection dicts, each containing:
          bbox, class_id, class_name, confidence, mask (H×W bool ndarray or None)
        """
        if self.model is None:
            return []

        results = self.model(
            cv_image, conf=self.conf_thresh, imgsz=self.yolo_imgsz, verbose=False,
        )
        detections = []
        for r in results:
            masks_data = r.masks  # may be None if model produced no masks
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Extract per-instance binary mask (resized to input image)
                mask = None
                if masks_data is not None and i < len(masks_data.data):
                    mask = masks_data.data[i].cpu().numpy().astype(bool)
                    # Resize mask to match the input image dimensions
                    h, w = cv_image.shape[:2]
                    if mask.shape != (h, w):
                        mask = cv2.resize(
                            mask.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                detections.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': cls_id,
                    'class_name': TARGET_CLASSES[cls_id],
                    'confidence': float(box.conf[0]),
                    'mask': mask,
                })
        return detections

    def _dummy_detection(self, cv_image):
        """Return a single fake detection at the image centre.
        Useful for testing the depth/TF/tracking pipeline without YOLO."""
        h, w = cv_image.shape[:2]
        # Full-image bbox so projections from the tilted depth camera
        # (which hits the lower portion of the RGB image) are captured.
        return [{
            'bbox': (0.0, 0.0, float(w), float(h)),
            'class_id': 0,
            'class_name': 'Water',
            'confidence': 0.99,
            'mask': None,
        }]

    # ─── Segmentation label-map publisher ──────────────────────────

    def _publish_segmentation(self, cv_image, detections, header):
        """Publish a semantic segmentation label image (mono8).

        Each pixel value is the 1-based YOLO class ID of the detection
        that covers it (0 = background).  With 63 classes this fits
        comfortably in uint8.  When masks overlap, the higher-confidence
        detection wins.
        """
        if self.pub_seg.get_subscription_count() == 0:
            return  # no subscribers, skip

        h, w = cv_image.shape[:2]
        label_img = np.zeros((h, w), dtype=np.uint8)

        # Sort ascending by confidence so higher-conf overwrites lower
        sorted_dets = sorted(detections,
                             key=lambda d: d['confidence'])
        for det in sorted_dets:
            mask = det.get('mask')
            if mask is None:
                continue
            # class_id is 0-based from YOLO; store as 1-based (0 = bg)
            label_img[mask] = det['class_id'] + 1

        seg_msg = self.bridge.cv2_to_imgmsg(label_img, encoding='mono8')
        seg_msg.header = header
        self.pub_seg.publish(seg_msg)

    def _publish_debug_overlay(self, cv_image, tracked, header):
        """Publish RGB debug image with bbox + class + track ID overlay."""
        if not self.publish_debug_image:
            return
        if self.pub_debug.get_subscription_count() == 0:
            return

        dbg = cv_image.copy()
        for t in tracked:
            bbox = t.get('bbox')
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            tid = int(t.get('track_id', -1))
            cls = t.get('class_name', 'unknown')
            conf = float(t.get('confidence', 0.0))
            npts = int(t.get('n_depth_pts', 0))

            # Deterministic per-track color makes ID switches obvious.
            color = (
                int((37 * tid) % 255),
                int((97 * tid) % 255),
                int((173 * tid) % 255),
            )
            cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
            label = f'{cls} #{tid} {conf:.2f} d={npts}'
            ytxt = max(18, y1 - 6)
            cv2.putText(
                dbg, label, (x1, ytxt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
        msg.header = header
        self.pub_debug.publish(msg)

    # ─── Cross-camera depth → RGB-frame geometry ───────────────────

    def _depth_grid_to_rgb_frame(self, stamp):
        """Sample a coarse grid over the depth image, back-project each
        pixel into 3D in ``f_depth_optical_frame``, then batch-transform
        them into ``f_oc_link`` (the RGB camera frame).

        Returns:
            np.ndarray of shape (N, 3) — 3D points in f_oc_link,
            or None on failure.
        """
        if self.depth_image is None or self.depth_camera_info is None:
            return None

        depth_img = self.depth_image
        h, w = depth_img.shape[:2]

        # Depth intrinsics
        fx_d = self.depth_camera_info.k[0]
        fy_d = self.depth_camera_info.k[4]
        cx_d = self.depth_camera_info.k[2]
        cy_d = self.depth_camera_info.k[5]
        if fx_d == 0 or fy_d == 0:
            return None

        # Build grid pixel coordinates (offset by half-step to avoid edges)
        us = np.arange(self.grid_step_u // 2, w, self.grid_step_u)
        vs = np.arange(self.grid_step_v // 2, h, self.grid_step_v)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel()
        vv = vv.ravel()

        # Sample depth at grid points (uint16 mm → float metres)
        z_mm = depth_img[vv, uu].astype(np.float64)
        valid = z_mm > 0
        if not np.any(valid):
            return None

        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        z_m = z_mm[valid] / 1000.0  # mm → metres

        # Back-project to 3D in f_depth_optical_frame (pinhole model)
        x_d = (uu - cx_d) * z_m / fx_d
        y_d = (vv - cy_d) * z_m / fy_d
        pts_depth = np.column_stack([x_d, y_d, z_m])  # (N, 3)

        # Transform depth-frame points → RGB camera frame via TF2
        pts_rgb = self._transform_points_batch(
            pts_depth, DEPTH_FRAME, RGB_FRAME, stamp
        )
        return pts_rgb

    def _transform_points_batch(self, points, source_frame, target_frame, stamp):
        """Transform an (N,3) array of 3D points between frames using TF2.

        Looks up the transform once and applies it as a matrix multiply
        for efficiency (instead of N individual TF calls).

        Returns np.ndarray (N, 3) in the target frame, or None on failure.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF {source_frame}→{target_frame} lookup failed: {e}',
                throttle_duration_sec=5.0,
            )
            return None

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation

        # Quaternion → 3×3 rotation matrix
        rot = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        tvec = np.array([t.x, t.y, t.z])

        # Apply: p_target = R @ p_source + t
        pts_out = (rot @ points.T).T + tvec
        return pts_out

    def _project_to_rgb(self, pts_rgb_frame):
        """Project 3D points (in f_oc_link) onto the RGB image plane.

        f_oc_link uses ROS body convention (X fwd, Y left, Z up).
        The RGB CameraInfo K matrix uses optical convention
        (X right, Y down, Z forward).  We convert before projecting.

        Returns np.ndarray of shape (N, 2) with (u, v) pixel coordinates.
        Points behind the camera (Z_opt ≤ 0) get coordinates of (-1, -1).
        """
        fx = self.rgb_camera_info.k[0]
        fy = self.rgb_camera_info.k[4]
        cx = self.rgb_camera_info.k[2]
        cy = self.rgb_camera_info.k[5]

        # f_oc_link body axes → optical axes
        X_opt = -pts_rgb_frame[:, 1]   # right  = −Y_link
        Y_opt = -pts_rgb_frame[:, 2]   # down   = −Z_link
        Z_opt =  pts_rgb_frame[:, 0]   # forward = X_link

        pixels = np.full((len(X_opt), 2), -1.0)
        valid = Z_opt > 0
        pixels[valid, 0] = fx * (X_opt[valid] / Z_opt[valid]) + cx   # u
        pixels[valid, 1] = fy * (Y_opt[valid] / Z_opt[valid]) + cy   # v
        return pixels

    def _transform_point(self, point_cam, source_frame, target_frame, stamp):
        """Transform a single 3D point between frames using TF2.
        Returns (x, y, z) in target frame, or the original point on failure.
        """
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = source_frame
        pose_msg.pose.position.x = float(point_cam[0])
        pose_msg.pose.position.y = float(point_cam[1])
        pose_msg.pose.position.z = float(point_cam[2])
        pose_msg.pose.orientation.w = 1.0

        try:
            transformed = self.tf_buffer.transform(
                pose_msg, target_frame,
                timeout=rclpy.duration.Duration(seconds=0.1),
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
            return point_cam

    def _bbox_to_3d_corners(self, u1, v1, u2, v2, depth_fwd):
        """Back-project 2D bbox corners to 3D at a given forward depth.

        Uses the RGB camera intrinsics to convert (u, v) pixel corners
        into f_oc_link body-frame 3D points at ``depth_fwd`` (the X
        component in f_oc_link = forward distance).

        Optical-to-body mapping:
            X_body = Z_opt = depth_fwd
            Y_body = −X_opt = −(u − cx) * depth / fx
            Z_body = −Y_opt = −(v − cy) * depth / fy

        Returns np.ndarray (8, 3) – axis-aligned bounding box corners.
        """
        fx = self.rgb_camera_info.k[0]
        fy = self.rgb_camera_info.k[4]
        cx = self.rgb_camera_info.k[2]
        cy = self.rgb_camera_info.k[5]

        # Back-project the four bbox corners
        y_left  = -((u1 - cx) * depth_fwd / fx)
        y_right = -((u2 - cx) * depth_fwd / fx)
        z_top   = -((v1 - cy) * depth_fwd / fy)
        z_bot   = -((v2 - cy) * depth_fwd / fy)

        y_min, y_max = min(y_left, y_right), max(y_left, y_right)
        z_min, z_max = min(z_top, z_bot), max(z_top, z_bot)

        # X (forward) has negligible spread — use a thin slab
        x_fwd = float(depth_fwd)
        return np.array([
            [x_fwd, y_min, z_min],
            [x_fwd, y_min, z_max],
            [x_fwd, y_max, z_min],
            [x_fwd, y_max, z_max],
            [x_fwd, y_min, z_min],
            [x_fwd, y_min, z_max],
            [x_fwd, y_max, z_min],
            [x_fwd, y_max, z_max],
        ])

    # ─── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _nms_3d(detections, dist_thresh=0.5):
        """Suppress duplicate 3-D detections at (nearly) identical positions.

        When several YOLO predictions overlap the same region (e.g.
        "Destroyed building" and "Building"), they match the same sparse
        depth points and produce identical 3-D positions.  This function
        keeps only the highest-confidence detection within *dist_thresh*
        metres (Euclidean in base_link).
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending — greedy NMS
        dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        for det in dets:
            pos = np.array(det['position'])
            suppressed = False
            for kept in keep:
                if np.linalg.norm(pos - np.array(kept['position'])) < dist_thresh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(det)
        return keep

    def _publish_empty(self, stamp):
        """Publish an empty Detection3DArray (keeps downstream aware)."""
        msg = Detection3DArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target_frame
        self.pub_det3d.publish(msg)

    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw):
        """Convert quaternion (x, y, z, w) to a 3×3 rotation matrix."""
        # Normalise
        n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if n == 0:
            return np.eye(3)
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n

        return np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])


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
