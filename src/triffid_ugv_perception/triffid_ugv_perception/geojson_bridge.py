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
    # Convert ROS2 detections to GeoJSON and push to TRIFFID API.

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
            '/ugv/perception/front/detections_3d',
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
            '/triffid/front/geojson',
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
        # Use first valid GPS fix as the local frame origin.
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
        # Convert local map-frame coordinates (metres) to [lon, lat].

        # Uses equirectangular approximation — accurate to ~1m for
        # displacements under ~1km from the origin.

        # Args:
        #     x: east offset in metres
        #     y: north offset in metres
        #     z: altitude offset (not used in 2D GeoJSON)

        # Returns:
        #     (longitude, latitude) tuple
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

    def detections_to_geojson(self, detections, source='ugv',
                               detection_type='seg'):
        
        features = []
        for det in detections:
            lon, lat = det['coordinates']
            sx, sy, _sz = det.get('size', (0.0, 0.0, 0.0))
            pos_x, pos_y, _pos_z = det.get('position', (0.0, 0.0, 0.0))

            has_extent = (sx > 0.0 and sy > 0.0)

            if has_extent:
                # Build ground-plane polygon from centre ± half-extent
                half_x, half_y = sx / 2.0, sy / 2.0
                corners = [
                    (pos_x + half_x, pos_y + half_y),
                    (pos_x + half_x, pos_y - half_y),
                    (pos_x - half_x, pos_y - half_y),
                    (pos_x - half_x, pos_y + half_y),
                ]
                ring = [list(self.local_to_gps(cx, cy)) for cx, cy in corners]
                ring.append(ring[0])  # close the ring (RFC-7946)
                geometry = {
                    "type": "Polygon",
                    "coordinates": [ring],
                }
            else:
                geometry = {
                    "type": "Point",
                    "coordinates": [lon, lat],
                }

            class_name = det.get('class_name', 'unknown')
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
                    "local_frame": not self.origin_set,
                    "marker-color": self._class_color(class_name),
                    "marker-size": "medium",
                    "marker-symbol": self._class_symbol(class_name),
                }
            }
            # Add SimpleStyle polygon properties when using Polygon geometry
            if has_extent:
                feature["properties"]["stroke"] = feature["properties"]["marker-color"]
                feature["properties"]["stroke-width"] = 2
                feature["properties"]["stroke-opacity"] = 1.0
                feature["properties"]["fill"] = feature["properties"]["marker-color"]
                feature["properties"]["fill-opacity"] = 0.25
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features
        }

    # Callbacks

    def ugv_callback(self, msg: Detection3DArray):
        # Process UGV 3D detections (b2/base_link) → GeoJSON.
        detections = []
        for det in msg.detections:
            # Extract 3D position (in b2/base_link, metres)
            x = det.bbox.center.position.x
            y = det.bbox.center.position.y
            z = det.bbox.center.position.z

            # 3D bbox extent (metres, in base_link axes)
            sx = det.bbox.size.x
            sy = det.bbox.size.y
            sz = det.bbox.size.z

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
                'size': (sx, sy, sz),
                'position': (x, y, z),
                'class_name': class_name,
                'confidence': confidence,
                'track_id': det.id,
            })

        if not detections:
            return

        geojson = self.detections_to_geojson(
            detections, source='ugv', detection_type='seg',
        )
        self._publish(geojson)

    # Publishing

    def _publish(self, geojson: dict):
        # Publish GeoJSON to ROS2 topic and optionally to the TRIFFID API.
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
        # PUT GeoJSON to the TRIFFID mapping API.
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
        # Map TRIFFID class name to a SimpleStyle marker color
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
        """Map TRIFFID class name to a semantic category."""
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
        # Map TRIFFID class name to a Maki icon name - TO CHANGE.
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
