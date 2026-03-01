"""
TRIFFID GeoJSON Bridge
=======================
Subscribes to UGV detection topic (Detection3DArray in b2/base_link)
and converts them to RFC-7946 GeoJSON, then:
  1. Publishes as a ROS2 String topic (for debugging / other nodes)
  2. Optionally PUTs to the TRIFFID mapping API

Coordinate handling:
  - Detections arrive in ``b2/base_link`` (local, metres).
  - If a GPS origin is set (via /fix or parameters), local XY are
    converted to lon/lat using an equirectangular approximation.
  - If no GPS is available (current rosbags), the raw local (x, y)
    are emitted as coordinates and a ``"local_frame": true`` property
    is added so downstream consumers know they are **not** WGS-84.

API endpoint (when enabled): https://crispres.com/wp-json/map-manager/v1/features
"""

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
from std_msgs.msg import String


class GeoJSONBridge(Node):
    """Convert ROS2 detections to GeoJSON and push to TRIFFID API."""

    def __init__(self):
        super().__init__('geojson_bridge')

        # Parameters
        self.declare_parameter('api_url', 'https://crispres.com/wp-json/map-manager/v1/features')
        self.declare_parameter('publish_to_api', False)  # disabled until tested
        self.declare_parameter('gps_origin_lat', 0.0)    # set from /fix or param
        self.declare_parameter('gps_origin_lon', 0.0)
        self.declare_parameter('gps_origin_alt', 0.0)

        self.api_url = self.get_parameter('api_url').value
        self.publish_to_api = self.get_parameter('publish_to_api').value
        self.origin_lat = self.get_parameter('gps_origin_lat').value
        self.origin_lon = self.get_parameter('gps_origin_lon').value
        self.origin_alt = self.get_parameter('gps_origin_alt').value
        self.origin_set = (self.origin_lat != 0.0 and self.origin_lon != 0.0)

        # Subscribers
        # UGV 3D detections (in b2/base_link)
        self.sub_ugv = self.create_subscription(
            Detection3DArray,
            '/ugv/perception/detections_3d',
            self.ugv_callback,
            10,
        )
        # GPS fix — to set the local→GPS origin (may not exist in bag)
        self.sub_gps = self.create_subscription(
            NavSatFix,
            '/fix',
            self.gps_callback,
            10,
        )

        # Publishers
        self.pub_geojson = self.create_publisher(
            String,
            '/triffid/geojson',
            10,
        )

        self.get_logger().info('GeoJSON Bridge started.')
        self.get_logger().info(f'  API URL: {self.api_url}')
        self.get_logger().info(f'  Publish to API: {self.publish_to_api}')
        if self.origin_set:
            self.get_logger().info(
                f'  GPS origin: ({self.origin_lat}, {self.origin_lon})'
            )
        else:
            self.get_logger().warn(
                'GPS origin not set — emitting local-frame coordinates. '
                'Set gps_origin_lat / gps_origin_lon params or wait for /fix.'
            )

    # GPS origin

    def gps_callback(self, msg: NavSatFix):
        """Use first valid GPS fix as the local frame origin."""
        if self.origin_set:
            return
        if msg.latitude != 0.0 and msg.longitude != 0.0:
            self.origin_lat = msg.latitude
            self.origin_lon = msg.longitude
            self.origin_alt = msg.altitude
            self.origin_set = True
            self.get_logger().info(
                f'GPS origin set: ({self.origin_lat}, {self.origin_lon}, {self.origin_alt})'
            )

    # Coordinate conversion

    def local_to_gps(self, x, y, z=0.0):
        """Convert local map-frame coordinates (metres) to [lon, lat].

        Uses equirectangular approximation — accurate to ~1m for
        displacements under ~1km from the origin.

        Args:
            x: east offset in metres
            y: north offset in metres
            z: altitude offset (not used in 2D GeoJSON)

        Returns:
            (longitude, latitude) tuple
        """
        if not self.origin_set:
            # Fall back: return raw coordinates (not valid GPS)
            return (x, y)

        # Earth radius in metres
        R = 6378137.0
        lat_origin_rad = math.radians(self.origin_lat)

        # Offset in degrees
        d_lat = y / R * (180.0 / math.pi)
        d_lon = x / (R * math.cos(lat_origin_rad)) * (180.0 / math.pi)

        lat = self.origin_lat + d_lat
        lon = self.origin_lon + d_lon
        return (lon, lat)  # GeoJSON is [lon, lat]

    # Detection to GeoJSON conversion

    def detections_to_geojson(self, detections, source='ugv'):
        """Convert a list of detection dicts to a GeoJSON FeatureCollection.

        Each detection becomes a GeoJSON Point Feature with properties:
          - name: class name
          - category: "detection"
          - source: "ugv"
          - track_id: persistent tracking ID
          - confidence: detection score
          - local_frame: True if coordinates are local (no GPS), False if WGS-84

        Compatible with the TRIFFID API (RFC-7946, SimpleStyle spec).
        """
        features = []
        for det in detections:
            lon, lat = det['coordinates']
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "name": det.get('class_name', 'unknown'),
                    "category": "detection",
                    "source": source,
                    "track_id": det.get('track_id', ''),
                    "confidence": det.get('confidence', 0.0),
                    "local_frame": not self.origin_set,
                    "marker-color": self._class_color(det.get('class_name', '')),
                    "marker-size": "medium",
                    "marker-symbol": self._class_symbol(det.get('class_name', '')),
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    # Callbacks

    def ugv_callback(self, msg: Detection3DArray):
        """Process UGV 3D detections (b2/base_link) → GeoJSON."""
        detections = []
        for det in msg.detections:
            # Extract 3D position (in b2/base_link, metres)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            # Convert to GPS if origin is set, otherwise emit local coords
            lon, lat = self.local_to_gps(x, y, z)

            # Extract class and confidence from results
            class_name = 'unknown'
            confidence = 0.0
            if det.results:
                class_name = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score

            detections.append({
                'coordinates': (lon, lat),
                'class_name': class_name,
                'confidence': confidence,
                'track_id': det.id,
            })

        if not detections:
            return

        geojson = self.detections_to_geojson(detections, source='ugv')
        self._publish(geojson)

    # Publishing

    def _publish(self, geojson: dict):
        """Publish GeoJSON to ROS2 topic and optionally to the TRIFFID API."""
        json_str = json.dumps(geojson, indent=2)

        # ROS2 topic
        msg = String()
        msg.data = json_str
        self.pub_geojson.publish(msg)

        self.get_logger().info(
            f'Published GeoJSON with {len(geojson["features"])} features',
            throttle_duration_sec=1.0,
        )

        # HTTP API
        if self.publish_to_api:
            # Do HTTP in a separate thread to not block the callback
            threading.Thread(
                target=self._send_to_api,
                args=(json_str,),
                daemon=True,
            ).start()

    def _send_to_api(self, json_str: str):
        """PUT GeoJSON to the TRIFFID mapping API."""
        try:
            req = Request(
                self.api_url,
                data=json_str.encode('utf-8'),
                method='PUT',
                headers={
                    'Content-Type': 'application/json',
                },
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

    # SimpleStyle helpers

    @staticmethod
    def _class_color(class_name: str) -> str:
        """Map class name to a SimpleStyle marker color."""
        colors = {
            'person': '#ff0000',
            'car': '#0000ff',
            'truck': '#00008b',
            'bus': '#000080',
            'bicycle': '#00ff00',
            'motorcycle': '#008000',
            'debris': '#ff8c00',
            'chair': '#ff8c00',
            'couch': '#ffd700',
        }
        return colors.get(class_name, '#808080')

    @staticmethod
    def _class_symbol(class_name: str) -> str:
        """Map class name to a Maki icon name (SimpleStyle)."""
        symbols = {
            'person': 'pitch',
            'car': 'car',
            'truck': 'truck',
            'bus': 'bus',
            'bicycle': 'bicycle',
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
