"""
TRIFFID GeoJSON Bridge

Subscribes to UGV detection topic (Detection3DArray in b2/base_link)
and converts them to RFC-7946 GeoJSON, then:
  1. Publishes as a ROS2 String topic (for debugging / other nodes)
  2. Optionally PUTs to the TRIFFID mapping API
  3. Optionally publishes to an MQTT broker

Coordinate handling:
  - Detections arrive in ``b2/base_link`` (X=forward, Y=left, Z=up).
  - The robot's current GPS position is tracked from ``/fix``
    (median-filtered over a sliding window to reduce noise).
  - The robot's heading (yaw) is obtained from ``/dog_odom``
    orientation quaternion (Go2 state estimator, magnetometer-fused).
  - Body-frame detection offsets are rotated by the robot's yaw into
    East-North-Up (ENU) and added to the current GPS position.
  - When no GPS is available, raw local (x, y, z) are emitted and a
    ``"local_frame": true`` property is added.
  - 2D coordinates are emitted: [lon, lat] (RFC 7946 §3.1.1).
   - GNSS altitude is stored in properties (``gnss_altitude_m``).

API endpoint (when enabled): https://crispres.com/wp-json/map-manager/v1/features
"""

import collections
import json
import math
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection3DArray
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import String

try:
    import paho.mqtt.client as paho_mqtt
    _PAHO_AVAILABLE = True
except ImportError:
    _PAHO_AVAILABLE = False

# Earth radius (WGS-84 semi-major axis) in metres
_R_EARTH = 6378137.0

# GPS sliding-window size for median filter
_GPS_WINDOW = 7

# Minimum polygon extent (metres) — used when one bbox dimension is 0 so that a visible polygon is still emitted instead of a point 
_MIN_EXTENT = 0.3

# Classes whose geometry is always emitted as a Point (regardless of bbox)
_POINT_CLASSES = frozenset([
    'First responder', 'Citizen', 'Military personnel',
])

# Classes whose geometry is emitted as a LineString (centre-line along longest axis)
_LINE_CLASSES = frozenset([
    'Fence',
])


class GeoJSONBridge(Node):
    """Convert ROS2 detections to GeoJSON and push to TRIFFID API."""

    def __init__(self):
        super().__init__('geojson_bridge')

        # Parameters 
        self.declare_parameter('api_url',
                               'https://crispres.com/wp-json/map-manager/v1/features')
        self.declare_parameter('publish_to_api', False)
        self.declare_parameter('gps_origin_lat', 0.0)
        self.declare_parameter('gps_origin_lon', 0.0)
        self.declare_parameter('gps_origin_alt', 0.0)
        self.declare_parameter('mqtt_enabled', True)
        self.declare_parameter('mqtt_host', 'localhost')
        self.declare_parameter('mqtt_port', 1883)
        self.declare_parameter('mqtt_topic', 'triffid/front/geojson')

        self.api_url = self.get_parameter('api_url').value
        self.publish_to_api = self.get_parameter('publish_to_api').value

        # GPS state
        # Sliding window for median filtering of noisy GPS fixes
        self._gps_lat_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_lon_buf = collections.deque(maxlen=_GPS_WINDOW)
        self._gps_alt_buf = collections.deque(maxlen=_GPS_WINDOW)

        # Current filtered robot GPS position (updated every /fix)
        self.robot_lat = 0.0
        self.robot_lon = 0.0
        self.robot_alt = 0.0
        self.gps_valid = False

        # Seed from parameters if provided
        param_lat = self.get_parameter('gps_origin_lat').value
        param_lon = self.get_parameter('gps_origin_lon').value
        param_alt = self.get_parameter('gps_origin_alt').value
        if param_lat != 0.0 and param_lon != 0.0:
            self.robot_lat = param_lat
            self.robot_lon = param_lon
            self.robot_alt = param_alt
            self.gps_valid = True

        # Heading state 
        # Yaw from /dog_odom quaternion (radians, ENU convention:
        # 0 = East, π/2 = North, counter-clockwise positive)
        self.robot_yaw = 0.0
        self.heading_valid = False

        # Subscribers 
        self.sub_ugv = self.create_subscription(
            Detection3DArray,
            '/ugv/perception/front/detections_3d',
            self.ugv_callback,
            10,
        )
        self.sub_gps = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10,
        )
        self.sub_odom = self.create_subscription(
            Odometry,
            '/dog_odom',
            self.odom_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )

        # Publishers
        self.pub_geojson = self.create_publisher(
            String,
            '/triffid/front/geojson',
            10,
        )

        self.get_logger().info('GeoJSON Bridge started.')
        self.get_logger().info(f'  API URL: {self.api_url}')
        self.get_logger().info(f'  Publish to API: {self.publish_to_api}')
        if self.gps_valid:
            self.get_logger().info(
                f'  GPS seeded from params: ({self.robot_lat}, {self.robot_lon})'
            )
        else:
            self.get_logger().warn(
                'GPS not yet available — emitting local-frame coordinates. '
                'Waiting for /fix topic.'
            )

        # MQTT client
        self._mqtt_enabled = self.get_parameter('mqtt_enabled').value
        self._mqtt_topic = self.get_parameter('mqtt_topic').value
        self._mqtt_client = None     # type: paho_mqtt.Client | None

        if self._mqtt_enabled:
            if not _PAHO_AVAILABLE:
                self.get_logger().warn(
                    'paho-mqtt not installed — MQTT publishing disabled. '
                    'Install with: pip3 install paho-mqtt'
                )
                self._mqtt_enabled = False
            else:
                mqtt_host = self.get_parameter('mqtt_host').value
                mqtt_port = self.get_parameter('mqtt_port').value
                try:
                    self._mqtt_client = paho_mqtt.Client(
                        paho_mqtt.CallbackAPIVersion.VERSION2,
                        client_id='geojson_bridge',
                        protocol=paho_mqtt.MQTTv311,
                    )
                    self._mqtt_client.connect_async(mqtt_host, mqtt_port)
                    self._mqtt_client.loop_start()
                    self.get_logger().info(
                        f'  MQTT: {mqtt_host}:{mqtt_port} → {self._mqtt_topic}'
                    )
                except Exception as e:
                    self.get_logger().warn(f'MQTT connect failed: {e}')
                    self._mqtt_enabled = False

    #  GPS (position tracking with median filter)
    def gps_callback(self, msg: NavSatFix):
        """Update current robot position from every valid GPS fix.

        Uses a sliding-window median filter to suppress GPS noise
        (typical consumer GPS jitter is ±2-5 m).
        """
        if msg.latitude == 0.0 and msg.longitude == 0.0:
            return

        self._gps_lat_buf.append(msg.latitude)
        self._gps_lon_buf.append(msg.longitude)
        self._gps_alt_buf.append(msg.altitude)

        # Median of the sliding window
        self.robot_lat = float(sorted(self._gps_lat_buf)[len(self._gps_lat_buf) // 2])
        self.robot_lon = float(sorted(self._gps_lon_buf)[len(self._gps_lon_buf) // 2])
        self.robot_alt = float(sorted(self._gps_alt_buf)[len(self._gps_alt_buf) // 2])

        if not self.gps_valid:
            self.gps_valid = True
            self.get_logger().info(
                f'GPS acquired: ({self.robot_lat:.7f}, {self.robot_lon:.7f}, '
                f'{self.robot_alt:.1f}m)'
            )


    #  Heading (from /dog_odom orientation quaternion)
    def odom_callback(self, msg: Odometry):
        """Extract yaw from the Go2 state estimator odometry.

        The orientation quaternion from /dog_odom is expected to be in
        an ENU-aligned frame (magnetometer-fused on the Unitree Go2).
        Yaw = angle from East axis, counter-clockwise positive.
        """
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)
        self.robot_yaw = math.atan2(siny, cosy)

        if not self.heading_valid:
            self.heading_valid = True
            self.get_logger().info(
                f'Heading acquired: {math.degrees(self.robot_yaw):.1f}° '
                f'(ENU yaw)'
            )

    #  Coordinate conversion (body-frame → GPS)
    @staticmethod
    def _body_to_enu(x_fwd, y_left, z_up, yaw):
        """Rotate a body-frame offset into East-North-Up.

        Body frame (b2/base_link): X=forward, Y=left, Z=up.
        ENU world frame: X=East, Y=North, Z=Up.

        The robot's heading *yaw* is the angle from East (CCW positive)
        in the ENU frame.

        Robot forward direction in ENU: (cos(yaw), sin(yaw))
        Robot left direction in ENU:    (-sin(yaw), cos(yaw))

        Returns (east, north, up) in metres.
        """
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        east  = x_fwd * cos_y - y_left * sin_y
        north = x_fwd * sin_y + y_left * cos_y
        up    = z_up
        return east, north, up

    def body_to_gps(self, x_fwd, y_left, z_up=0.0):
        """Convert a detection in b2/base_link to [lon, lat, alt].

        1. Rotate body offset by robot yaw → ENU metres
        2. Convert ENU metres → degree offsets (equirectangular)
        3. Add to current robot GPS position

        Returns (lon, lat, alt) tuple.
        When GPS not available, returns raw body-frame coords.
        """
        if not self.gps_valid:
            return (x_fwd, y_left, z_up)

        yaw = self.robot_yaw if self.heading_valid else 0.0

        # Step 1: body → ENU
        east, north, up = self._body_to_enu(x_fwd, y_left, z_up, yaw)

        # Step 2: ENU metres → degree offsets
        lat_rad = math.radians(self.robot_lat)
        d_lat = north / _R_EARTH * (180.0 / math.pi)
        d_lon = east / (_R_EARTH * math.cos(lat_rad)) * (180.0 / math.pi)

        # Step 3: add to current robot position
        lat = self.robot_lat + d_lat
        lon = self.robot_lon + d_lon
        alt = self.robot_alt + up

        return (lon, lat, alt)  # GeoJSON order: [lon, lat, alt]

    # ═══════════════════════════════════════════════════════════════
    #  Detection → GeoJSON conversion
    # ═══════════════════════════════════════════════════════════════

    # Class-dependent geometry type
    @staticmethod
    def _geometry_type_for_class(class_name: str) -> str:
        """Return the GeoJSON geometry type to use for a given class.

        - Person classes → Point (small, mobile targets)
        - Linear structures (Fence) → LineString
        - Everything else → Polygon (footprint)
        """
        if class_name in _POINT_CLASSES:
            return 'Point'
        if class_name in _LINE_CLASSES:
            return 'LineString'
        return 'Polygon'

    def detections_to_geojson(self, detections, source='ugv',
                               detection_type='seg'):
        """Build a GeoJSON FeatureCollection from detection dicts.

        Geometry coordinates are 2D [lon, lat] only (RFC 7946).
        GNSS altitude is stored in ``gnss_altitude_m`` property.
        Geometry type is class-dependent (Point / LineString / Polygon).
        """
        features = []
        for det in detections:
            lon, lat, alt = det['coordinates']
            sx, sy, sz = det.get('size', (0.0, 0.0, 0.0))
            pos_x, pos_y, pos_z = det.get('position', (0.0, 0.0, 0.0))

            class_name = det.get('class_name', 'unknown')
            has_extent = (sx > 0.0 or sy > 0.0)
            geom_type = self._geometry_type_for_class(class_name)

            if geom_type == 'Point':
                # Point: just the centre position in [lon, lat]
                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }

            elif geom_type == 'LineString':
                # LineString: centre-line along the longer bbox axis
                half_long = max(sx, sy, _MIN_EXTENT) / 2.0
                if sx >= sy:
                    # longer along X (forward)
                    endpoints = [
                        list(self.body_to_gps(pos_x + half_long, pos_y, pos_z))[:2],
                        list(self.body_to_gps(pos_x - half_long, pos_y, pos_z))[:2],
                    ]
                else:
                    # longer along Y (left-right)
                    endpoints = [
                        list(self.body_to_gps(pos_x, pos_y + half_long, pos_z))[:2],
                        list(self.body_to_gps(pos_x, pos_y - half_long, pos_z))[:2],
                    ]
                geometry = {
                    "type": "LineString",
                    "coordinates": endpoints,
                }

            else:  # Polygon (default)
                if has_extent:
                    # Build polygon from body-frame corners, each
                    # individually converted to GPS via body_to_gps.
                    # Use minimum extent so a zero dimension still
                    # produces a visible polygon (not a degenerate line).
                    half_x = max(sx, _MIN_EXTENT) / 2.0
                    half_y = max(sy, _MIN_EXTENT) / 2.0
                    body_corners = [
                        (pos_x + half_x, pos_y + half_y, pos_z),
                        (pos_x + half_x, pos_y - half_y, pos_z),
                        (pos_x - half_x, pos_y - half_y, pos_z),
                        (pos_x - half_x, pos_y + half_y, pos_z),
                    ]
                    ring = [list(self.body_to_gps(cx, cy, cz))[:2]
                            for cx, cy, cz in body_corners]
                    ring.append(ring[0])  # close the ring (RFC-7946)
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [ring],
                    }
                else:
                    # Fallback to Point when no bbox extents
                    geometry = {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    }

            feature = {
                "type": "Feature",
                "id": det.get('track_id', ''),
                "geometry": geometry,
                "properties": {
                    "class": class_name,
                    "id": det.get('track_id', ''),
                    "confidence": det.get('confidence', 0.0),
                    "category": self._class_category(class_name),
                    "detection_type": detection_type,
                    "source": source,
                    "local_frame": not self.gps_valid,
                    "gnss_altitude_m": round(alt, 2),
                    "height_m": round(float(sz), 2),
                    "marker-color": self._class_color(class_name),
                    "marker-size": "medium",
                    "marker-symbol": self._class_symbol(class_name),
                }
            }
            # SimpleStyle: stroke/fill for Polygon, stroke for LineString
            if geometry['type'] == 'Polygon':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
                feature["properties"]["fill"] = feature["properties"]["marker-color"]
                feature["properties"]["fill-opacity"] = 0.25
            elif geometry['type'] == 'LineString':
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    #  Detection callback
    def ugv_callback(self, msg: Detection3DArray):
        """Process UGV 3D detections (b2/base_link) → GeoJSON."""
        detections = []
        for det in msg.detections:
            # 3D position in b2/base_link (X=fwd, Y=left, Z=up)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            # 3D bbox extent (metres)
            sx = det.bbox.size.x
            sy = det.bbox.size.y
            sz = det.bbox.size.z

            # Convert body-frame centre to GPS [lon, lat, alt]
            lon, lat, alt = self.body_to_gps(x, y, z)

            # Extract class and confidence
            class_name = 'unknown'
            confidence = 0.0
            if det.results:
                class_name = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score

            detections.append({
                'coordinates': (lon, lat, alt),
                'size': (sx, sy, sz),
                'position': (x, y, z),
                'class_name': class_name,
                'confidence': confidence,
                'track_id': det.id,
            })

        if not detections:
            return

        # Don't publish until GPS is valid — raw body-frame metres
        # look like (lon, lat) near (0°, 0°) and mislead map viewers.
        if not self.gps_valid:
            self.get_logger().warn(
                'Skipping GeoJSON publish — no GPS fix yet.',
                throttle_duration_sec=5.0,
            )
            return

        geojson = self.detections_to_geojson(
            detections, source='ugv', detection_type='seg',
        )
        self._publish(geojson)

    #  Publishing
    def _publish(self, geojson: dict):
        """Publish GeoJSON to ROS2 topic, MQTT, and optionally to the API."""
        json_str = json.dumps(geojson, indent=2)

        msg = String()
        msg.data = json_str
        self.pub_geojson.publish(msg)

        # MQTT publish (compact JSON, no indent, for bandwidth)
        if self._mqtt_enabled and self._mqtt_client is not None:
            try:
                self._mqtt_client.publish(
                    self._mqtt_topic,
                    json.dumps(geojson),
                    qos=0,
                )
            except Exception as e:
                self.get_logger().warn(
                    f'MQTT publish failed: {e}',
                    throttle_duration_sec=10.0,
                )

        self.get_logger().info(
            f'Published GeoJSON with {len(geojson["features"])} features',
            throttle_duration_sec=1.0,
        )

        if self.publish_to_api:
            threading.Thread(
                target=self._send_to_api,
                args=(json_str,),
                daemon=True,
            ).start()

    def _send_to_api(self, json_str: str):
        # PUT GeoJSON to the TRIFFID mapping API.
        try:
            req = Request(
                self.api_url,
                data=json_str.encode('utf-8'),
                method='PUT',
                headers={'Content-Type': 'application/json'},
            )
            with urlopen(req, timeout=5) as resp:
                status = resp.status
                if status in (200, 201):
                    self.get_logger().debug(f'API PUT OK ({status})')
                else:
                    self.get_logger().warn(f'API PUT returned {status}')
        except URLError as e:
            self.get_logger().warn(
                f'API PUT failed: {e}', throttle_duration_sec=10.0
            )
        except Exception as e:
            self.get_logger().error(f'API PUT error: {e}')


    #  SimpleStyle helpers
    @staticmethod
    def _class_color(class_name: str) -> str:

        colors = {
            # Fire / hazard (reds)
            'Flame': '#ff0000',
            'Smoke': '#ff4500',
            'Burnt tree': '#8b0000',
            'Burnt grass': '#a52a2a',
            'Burnt plant': '#b22222',
            'Fire hose': '#dc143c',
            'Fire hydrant': '#ff6347',
            'Fire truck': '#ff0000',
            'Extinguisher': '#ff1493',
            # People (blues)
            'First responder': '#1e90ff',
            'Citizen': '#4169e1',
            'Military personnel': '#000080',
            # Vehicles (dark blues)
            'Civilian vehicle': '#0000ff',
            'Destroyed vehicle': '#00008b',
            'Ambulance': '#4682b4',
            'Police vehicle': '#191970',
            'Army vehicle': '#2f4f4f',
            'Boat': '#5f9ea0',
            'Bicycle': '#00ff00',
            'aerial vehicle': '#87ceeb',
            # Nature (greens)
            'Green tree': '#228b22',
            'Green plant': '#32cd32',
            'Green grass': '#7cfc00',
            'Dry tree': '#daa520',
            'Dry grass': '#bdb76b',
            'Dry plant': '#f0e68c',
            'Animal': '#ff8c00',
            # Infrastructure (greys)
            'Building': '#708090',
            'Destroyed building': '#696969',
            'Wall': '#808080',
            'Road': '#a9a9a9',
            'Pavement': '#c0c0c0',
            'Dirt road': '#d2b48c',
            'Window': '#b0c4de',
            'Door': '#8b4513',
            'Stairs': '#a0522d',
            'Pole': '#778899',
            'Tower': '#556b2f',
            'Silo': '#6b8e23',
            # Obstacles (oranges / yellows)
            'Debris': '#ff8c00',
            'Fence': '#daa520',
            'Barrier': '#ffd700',
            'Cone': '#ff7f50',
            'Hole in the ground': '#8b4513',
            'Mud': '#a0522d',
            'Water': '#00bfff',
            # Equipment (purples)
            'Helmet': '#9370db',
            'SCBA': '#8a2be2',
            'Boot': '#4b0082',
            'Mask': '#9400d3',
            'Glove': '#da70d6',
            'Protective glasses': '#ba55d3',
            'Ladder': '#ff8c00',
            'Ax': '#cd853f',
            'Shovel': '#d2691e',
            'Chainsaw': '#b8860b',
            'Bag': '#bc8f8f',
            'Barrel': '#8b8682',
            'Furniture': '#deb887',
            'Tank': '#2e8b57',
            'Crane': '#b8860b',
            'Excavator': '#daa520',
            'Lifesaver': '#ff4500',
        }
        return colors.get(class_name, '#808080')

    @staticmethod
    def _class_category(class_name: str) -> str:

        categories = {
            # Hazard
            'Flame': 'hazard', 'Smoke': 'hazard',
            'Burnt tree': 'hazard', 'Burnt grass': 'hazard', 'Burnt plant': 'hazard',
            # People
            'First responder': 'person', 'Citizen': 'person',
            'Military personnel': 'person',
            # Vehicles
            'Civilian vehicle': 'vehicle', 'Destroyed vehicle': 'vehicle',
            'Ambulance': 'vehicle', 'Police vehicle': 'vehicle',
            'Fire truck': 'vehicle', 'Army vehicle': 'vehicle',
            'Boat': 'vehicle', 'Bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
            # Nature
            'Green tree': 'nature', 'Green plant': 'nature',
            'Green grass': 'nature', 'Dry tree': 'nature',
            'Dry grass': 'nature', 'Dry plant': 'nature', 'Animal': 'nature',
            # Infrastructure
            'Building': 'infrastructure', 'Destroyed building': 'infrastructure',
            'Wall': 'infrastructure', 'Road': 'infrastructure',
            'Pavement': 'infrastructure', 'Dirt road': 'infrastructure',
            'Window': 'infrastructure', 'Door': 'infrastructure',
            'Stairs': 'infrastructure', 'Pole': 'infrastructure',
            'Tower': 'infrastructure', 'Silo': 'infrastructure',
            # Obstacle
            'Debris': 'obstacle', 'Fence': 'obstacle', 'Barrier': 'obstacle',
            'Cone': 'obstacle', 'Hole in the ground': 'obstacle',
            'Mud': 'obstacle', 'Water': 'obstacle',
            # Equipment
            'Fire hose': 'equipment', 'Fire hydrant': 'equipment',
            'Extinguisher': 'equipment', 'Helmet': 'equipment',
            'SCBA': 'equipment', 'Boot': 'equipment', 'Mask': 'equipment',
            'Glove': 'equipment', 'Protective glasses': 'equipment',
            'Ladder': 'equipment', 'Ax': 'equipment', 'Shovel': 'equipment',
            'Chainsaw': 'equipment', 'Bag': 'equipment', 'Barrel': 'equipment',
            'Furniture': 'equipment', 'Tank': 'equipment', 'Crane': 'equipment',
            'Excavator': 'equipment', 'Lifesaver': 'equipment',
        }
        return categories.get(class_name, 'unknown')

    @staticmethod
    def _class_symbol(class_name: str) -> str:

        symbols = {
            'First responder': 'pitch',
            'Citizen': 'pitch',
            'Military personnel': 'pitch',
            'Civilian vehicle': 'car',
            'Destroyed vehicle': 'car',
            'Ambulance': 'hospital',
            'Police vehicle': 'police',
            'Fire truck': 'fire-station',
            'Army vehicle': 'car',
            'Boat': 'harbor',
            'Bicycle': 'bicycle',
            'aerial vehicle': 'airfield',
            'Flame': 'fire-station',
            'Smoke': 'fire-station',
            'Building': 'building',
            'Destroyed building': 'building',
            'Water': 'water',
            'Animal': 'dog-park',
        }
        return symbols.get(class_name, 'marker')


def main(args=None):
    rclpy.init(args=args)
    node = GeoJSONBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._mqtt_client is not None:
            node._mqtt_client.loop_stop()
            node._mqtt_client.disconnect()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
