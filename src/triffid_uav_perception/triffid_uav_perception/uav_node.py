"""
TRIFFID UAV Perception Node
=============================
Standalone pipeline (no ROS2) that processes DJI drone images:

  1. Load image + extract embedded XMP metadata
  2. Run YOLO segmentation model on the RGB frame
  3. For each detection: project mask/bbox → ground GPS via gimbal geometry
  4. Build GeoJSON FeatureCollection (same schema as UGV output)
  5. Publish to MQTT broker

Usage:
  # Process a single image:
  python -m triffid_uav_perception.uav_node --image photo.jpg

  # Watch a directory for new images:
  python -m triffid_uav_perception.uav_node --watch /path/to/images/

  # Process all images in a directory (batch):
  python -m triffid_uav_perception.uav_node --batch /path/to/images/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from triffid_uav_perception.metadata import DJIMetadata, extract_metadata
from triffid_uav_perception.geo import (
    CameraIntrinsics,
    get_intrinsics,
    project_mask_to_ground,
    project_bbox_to_ground,
    estimate_object_height,
)
from triffid_uav_perception.api_client import FuturisedClient

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

try:
    import paho.mqtt.client as paho_mqtt
    _HAS_PAHO = True
except ImportError:
    _HAS_PAHO = False

log = logging.getLogger('triffid_uav')


# Same 63 classes as UGV (shared TRIFFID model)
TARGET_CLASSES = {
    0: 'water', 1: 'fence', 2: 'green tree', 3: 'helmet',
    4: 'flame', 5: 'smoke', 6: 'first responder', 7: 'destroyed vehicle',
    8: 'fire hose', 9: 'scba', 10: 'boot', 11: 'green plant',
    12: 'mask', 13: 'window', 14: 'building', 15: 'destroyed building',
    16: 'debris', 17: 'ladder', 18: 'dirt road', 19: 'dry tree',
    20: 'wall', 21: 'civilian vehicle', 22: 'road', 23: 'citizen',
    24: 'green grass', 25: 'pole', 26: 'boat', 27: 'pavement',
    28: 'dry grass', 29: 'animal', 30: 'excavator', 31: 'door',
    32: 'mud', 33: 'barrier', 34: 'hole in the ground', 35: 'bag',
    36: 'burnt tree', 37: 'ambulance', 38: 'fire truck', 39: 'cone',
    40: 'bicycle', 41: 'tower', 42: 'silo', 43: 'military personnel',
    44: 'burnt grass', 45: 'ax', 46: 'glove', 47: 'crane',
    48: 'stairs', 49: 'dry plant', 50: 'furniture', 51: 'tank',
    52: 'protective glasses', 53: 'barrel', 54: 'shovel',
    55: 'fire hydrant', 56: 'police vehicle', 57: 'burnt plant',
    58: 'army vehicle', 59: 'chainsaw', 60: 'aerial vehicle',
    61: 'lifesaver', 62: 'extinguisher',
}

# Same styling helpers as UGV geojson_bridge — kept here to avoid
# importing from the UGV package (they're independent deployments).

_CLASS_COLORS = {
    'flame': '#ff0000', 'smoke': '#ff4500', 'burnt tree': '#8b0000',
    'burnt grass': '#a52a2a', 'burnt plant': '#b22222',
    'fire hose': '#dc143c', 'fire hydrant': '#ff6347',
    'fire truck': '#ff0000', 'extinguisher': '#ff1493',
    'first responder': '#1e90ff', 'citizen': '#4169e1',
    'military personnel': '#000080',
    'civilian vehicle': '#0000ff', 'destroyed vehicle': '#00008b',
    'ambulance': '#4682b4', 'police vehicle': '#191970',
    'army vehicle': '#2f4f4f', 'boat': '#5f9ea0', 'bicycle': '#00ff00',
    'aerial vehicle': '#87ceeb',
    'green tree': '#228b22', 'green plant': '#32cd32',
    'green grass': '#7cfc00', 'dry tree': '#daa520',
    'dry grass': '#bdb76b', 'dry plant': '#f0e68c', 'animal': '#ff8c00',
    'building': '#708090', 'destroyed building': '#696969',
    'wall': '#808080', 'road': '#a9a9a9', 'pavement': '#c0c0c0',
    'dirt road': '#d2b48c', 'window': '#b0c4de', 'door': '#8b4513',
    'stairs': '#a0522d', 'pole': '#778899', 'tower': '#556b2f',
    'silo': '#6b8e23',
    'debris': '#ff8c00', 'fence': '#daa520', 'barrier': '#ffd700',
    'cone': '#ff7f50', 'hole in the ground': '#8b4513',
    'mud': '#a0522d', 'water': '#00bfff',
    'helmet': '#9370db', 'scba': '#8a2be2', 'boot': '#4b0082',
    'mask': '#9400d3', 'glove': '#da70d6', 'protective glasses': '#ba55d3',
    'ladder': '#ff8c00', 'ax': '#cd853f', 'shovel': '#d2691e',
    'chainsaw': '#b8860b', 'bag': '#bc8f8f', 'barrel': '#8b8682',
    'furniture': '#deb887', 'tank': '#2e8b57', 'crane': '#b8860b',
    'excavator': '#daa520', 'lifesaver': '#ff4500',
}

_CLASS_CATEGORIES = {
    'flame': 'hazard', 'smoke': 'hazard', 'burnt tree': 'hazard',
    'burnt grass': 'hazard', 'burnt plant': 'hazard',
    'first responder': 'person', 'citizen': 'person',
    'military personnel': 'person',
    'civilian vehicle': 'vehicle', 'destroyed vehicle': 'vehicle',
    'ambulance': 'vehicle', 'police vehicle': 'vehicle',
    'fire truck': 'vehicle', 'army vehicle': 'vehicle',
    'boat': 'vehicle', 'bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
    'green tree': 'nature', 'green plant': 'nature',
    'green grass': 'nature', 'dry tree': 'nature',
    'dry grass': 'nature', 'dry plant': 'nature', 'animal': 'nature',
    'building': 'infrastructure', 'destroyed building': 'infrastructure',
    'wall': 'infrastructure', 'road': 'infrastructure',
    'pavement': 'infrastructure', 'dirt road': 'infrastructure',
    'window': 'infrastructure', 'door': 'infrastructure',
    'stairs': 'infrastructure', 'pole': 'infrastructure',
    'tower': 'infrastructure', 'silo': 'infrastructure',
    'debris': 'obstacle', 'fence': 'obstacle', 'barrier': 'obstacle',
    'cone': 'obstacle', 'hole in the ground': 'obstacle',
    'mud': 'obstacle', 'water': 'obstacle',
    'fire hose': 'equipment', 'fire hydrant': 'equipment',
    'extinguisher': 'equipment', 'helmet': 'equipment',
    'scba': 'equipment', 'boot': 'equipment', 'mask': 'equipment',
    'glove': 'equipment', 'protective glasses': 'equipment',
    'ladder': 'equipment', 'ax': 'equipment', 'shovel': 'equipment',
    'chainsaw': 'equipment', 'bag': 'equipment', 'barrel': 'equipment',
    'furniture': 'equipment', 'tank': 'equipment', 'crane': 'equipment',
    'excavator': 'equipment', 'lifesaver': 'equipment',
}

_CLASS_SYMBOLS = {
    'first responder': 'pitch', 'citizen': 'pitch',
    'military personnel': 'pitch',
    'civilian vehicle': 'car', 'destroyed vehicle': 'car',
    'ambulance': 'hospital', 'police vehicle': 'police',
    'fire truck': 'fire-station', 'army vehicle': 'car',
    'boat': 'harbor', 'bicycle': 'bicycle', 'aerial vehicle': 'airfield',
    'flame': 'fire-station', 'smoke': 'fire-station',
    'building': 'building', 'destroyed building': 'building',
    'water': 'water', 'animal': 'dog-park',
}

# Person classes get Point geometry (small, mobile targets)
_POINT_CLASSES = frozenset([
    'first responder', 'citizen', 'military personnel',
])

# Linear structures get LineString
_LINE_CLASSES = frozenset([
    'fence',
])


class UAVPipeline:
    """Main UAV perception pipeline.

    Processes images one at a time: extract metadata, run YOLO, project
    to ground GPS, build GeoJSON, publish to MQTT.
    """

    def __init__(
        self,
        model_path: str = 'yolo11n-seg.pt',
        confidence: float = 0.35,
        mqtt_host: str = 'localhost',
        mqtt_port: int = 1883,
        mqtt_topic: str = 'triffid/uav/geojson',
        intrinsics: Optional[CameraIntrinsics] = None,
        yolo_imgsz: int = 1280,
    ):
        self.conf_thresh = confidence
        self.mqtt_topic = mqtt_topic
        self.intrinsics_override = intrinsics
        self.yolo_imgsz = yolo_imgsz
        self._next_id = 1

        # Load YOLO model
        if _HAS_YOLO:
            log.info(f'Loading YOLO model: {model_path}')
            self.model = YOLO(model_path)
            log.info('Model loaded.')
        else:
            self.model = None
            log.error(
                'ultralytics not installed — no detections will be produced. '
                'pip install ultralytics'
            )

        # MQTT client
        self._mqtt = None
        if _HAS_PAHO:
            try:
                self._mqtt = paho_mqtt.Client(
                    paho_mqtt.CallbackAPIVersion.VERSION2,
                    client_id='uav_perception',
                    protocol=paho_mqtt.MQTTv311,
                )
                self._mqtt.connect(mqtt_host, mqtt_port)
                self._mqtt.loop_start()
                log.info(f'MQTT connected: {mqtt_host}:{mqtt_port} → {mqtt_topic}')
            except Exception as e:
                log.warning(f'MQTT connect failed: {e}')
                self._mqtt = None
        else:
            log.warning('paho-mqtt not installed — MQTT disabled.')

    def process_image(self, image_path: str) -> Optional[dict]:
        """Process a single image through the full pipeline.

        Returns the GeoJSON FeatureCollection dict, or None on failure.
        """
        image_path = str(image_path)
        log.info(f'Processing: {image_path}')

        # Step 1: Extract metadata
        try:
            meta = extract_metadata(image_path)
        except ValueError as e:
            log.warning(f'Metadata extraction failed: {e}')
            return None

        # Validate RTK quality
        if meta.rtk_flag < 16:
            log.warning(
                f'Poor GPS quality (rtk_flag={meta.rtk_flag}), '
                f'proceeding with caution'
            )

        log.info(
            f'  Drone: ({meta.lat:.7f}, {meta.lon:.7f}) alt={meta.abs_alt:.1f}m '
            f'gimbal=({meta.gimbal_yaw:.1f}°, {meta.gimbal_pitch:.1f}°) '
            f'LRF={"OK" if meta.lrf_valid else "N/A"}'
        )

        # Step 2: Load image and run YOLO
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            log.error(f'Failed to read image: {image_path}')
            return None

        h, w = cv_image.shape[:2]
        intrinsics = get_intrinsics(w, h, self.intrinsics_override)
        detections = self._detect(cv_image)

        if not detections:
            log.info('  No detections.')
            return None

        log.info(f'  {len(detections)} detections found.')

        # Step 3: Project each detection to ground GPS
        features = []
        for det in detections:
            feature = self._detection_to_feature(det, meta, intrinsics)
            if feature is not None:
                features.append(feature)

        if not features:
            log.info('  No features could be geo-projected.')
            return None

        # Step 4: Build GeoJSON
        geojson = {
            "type": "FeatureCollection",
            "features": features,
        }

        # Step 5: Publish to MQTT
        self._publish_mqtt(geojson)

        log.info(f'  Published {len(features)} features to MQTT.')
        return geojson

    def _detect(self, cv_image: np.ndarray) -> list:
        """Run YOLO segmentation on an image."""
        if self.model is None:
            return []

        results = self.model(
            cv_image, conf=self.conf_thresh, imgsz=self.yolo_imgsz, verbose=False,
        )
        detections = []
        for r in results:
            masks_data = r.masks
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                mask = None
                if masks_data is not None and i < len(masks_data.data):
                    mask = masks_data.data[i].cpu().numpy().astype(bool)
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

    def _detection_to_feature(
        self,
        det: dict,
        meta: DJIMetadata,
        intrinsics: CameraIntrinsics,
    ) -> Optional[dict]:
        """Convert a single detection to a GeoJSON Feature."""
        class_name = det['class_name']
        geom_type = self._geometry_type_for_class(class_name)
        mask = det.get('mask')

        # Try mask-based projection first, fall back to bbox
        if mask is not None:
            result = project_mask_to_ground(mask, meta, intrinsics)
        else:
            result = None

        if result is None:
            x1, y1, x2, y2 = det['bbox']
            result = project_bbox_to_ground(x1, y1, x2, y2, meta, intrinsics)

        if result is None:
            return None

        ring, (centre_lon, centre_lat, centre_alt) = result

        # Estimate object height
        height_m = 0.0
        if mask is not None:
            height_m = estimate_object_height(mask, meta, intrinsics)

        # Build geometry based on class type
        if geom_type == 'Point':
            geometry = {
                "type": "Point",
                "coordinates": [centre_lon, centre_lat],
            }
        elif geom_type == 'LineString':
            # Use first and last ring points as the line endpoints
            if len(ring) >= 3:
                mid = len(ring) // 2
                geometry = {
                    "type": "LineString",
                    "coordinates": [ring[0], ring[mid]],
                }
            else:
                geometry = {
                    "type": "Point",
                    "coordinates": [centre_lon, centre_lat],
                }
        else:
            geometry = {
                "type": "Polygon",
                "coordinates": [ring],
            }

        track_id = str(self._next_id)
        self._next_id += 1

        feature = {
            "type": "Feature",
            "id": track_id,
            "geometry": geometry,
            "properties": {
                "class": class_name,
                "id": track_id,
                "confidence": round(det['confidence'], 4),
                "category": _CLASS_CATEGORIES.get(class_name, 'unknown'),
                "detection_type": "seg",
                "source": "uav",
                "local_frame": False,
                "altitude_m": round(centre_alt, 2),
                "height_m": round(height_m, 2),
                "marker-color": _CLASS_COLORS.get(class_name, '#808080'),
                "marker-size": "medium",
                "marker-symbol": _CLASS_SYMBOLS.get(class_name, 'marker'),
            },
        }

        # SimpleStyle extras for Polygon and LineString
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

        return feature

    def _publish_mqtt(self, geojson: dict):
        """Publish GeoJSON to MQTT broker."""
        if self._mqtt is None:
            return
        try:
            self._mqtt.publish(
                self.mqtt_topic,
                json.dumps(geojson),
                qos=0,
            )
        except Exception as e:
            log.warning(f'MQTT publish failed: {e}')

    def shutdown(self):
        """Clean up MQTT connection."""
        if self._mqtt is not None:
            self._mqtt.loop_stop()
            self._mqtt.disconnect()

    @staticmethod
    def _geometry_type_for_class(class_name: str) -> str:
        if class_name in _POINT_CLASSES:
            return 'Point'
        if class_name in _LINE_CLASSES:
            return 'LineString'
        return 'Polygon'


def _process_batch(pipeline: UAVPipeline, image_dir: str,
                   output_dir: Optional[str] = None):
    """Process all images in a directory."""
    img_dir = Path(image_dir)
    extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.dng'}
    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        log.warning(f'No images found in {image_dir}')
        return

    log.info(f'Found {len(images)} images in {image_dir}')
    all_features = []

    for img_path in images:
        geojson = pipeline.process_image(str(img_path))
        if geojson is not None:
            all_features.extend(geojson['features'])

    # Save merged output
    if output_dir and all_features:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged = {"type": "FeatureCollection", "features": all_features}
        merged_path = out / 'geojson_merged.json'
        merged_path.write_text(json.dumps(merged, indent=2))
        log.info(f'Saved merged GeoJSON: {merged_path} ({len(all_features)} features)')


def _watch_directory(pipeline: UAVPipeline, watch_dir: str,
                     poll_interval: float = 2.0):
    """Watch a directory for new images and process them as they appear."""
    img_dir = Path(watch_dir)
    extensions = {'.jpg', '.jpeg', '.tif', '.tiff', '.png', '.dng'}
    processed = set()

    log.info(f'Watching {watch_dir} for new images (poll every {poll_interval}s)...')

    while True:
        try:
            current = set(
                p for p in img_dir.iterdir()
                if p.suffix.lower() in extensions
            )
            new_files = sorted(current - processed)
            for img_path in new_files:
                # Small delay to ensure file is fully written
                time.sleep(0.5)
                pipeline.process_image(str(img_path))
                processed.add(img_path)

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            break

    log.info('Watch stopped.')


def _poll_api(pipeline: UAVPipeline, client: FuturisedClient,
              poll_interval: float = 10.0, camera: str = 'Wide',
              output_dir: Optional[str] = None):
    """Poll the FUTURISED API for new images and process them.

    Checks for new image uploads at ``poll_interval`` seconds.
    Downloads new images, processes them through the pipeline, and
    optionally saves merged GeoJSON output.

    Parameters
    ----------
    pipeline : UAVPipeline
        The perception pipeline instance.
    client : FuturisedClient
        Configured API client.
    poll_interval : float
        Seconds between API polls.
    camera : str
        Camera filter (e.g. 'Wide'). Empty string for all cameras.
    output_dir : str, optional
        Directory to save merged GeoJSON output.
    """
    log.info(
        f'Polling FUTURISED API every {poll_interval}s '
        f'(camera={camera or "all"}) ...'
    )
    all_features = []

    while True:
        try:
            new_images = client.poll_new_images(camera_filter=camera)
            for img_path in new_images:
                geojson = pipeline.process_image(str(img_path))
                if geojson is not None:
                    all_features.extend(geojson['features'])

            time.sleep(poll_interval)
        except KeyboardInterrupt:
            break

    # Save merged output
    if output_dir and all_features:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        merged = {"type": "FeatureCollection", "features": all_features}
        merged_path = out / 'geojson_merged.json'
        merged_path.write_text(json.dumps(merged, indent=2))
        log.info(f'Saved merged GeoJSON: {merged_path} ({len(all_features)} features)')

    log.info('API polling stopped.')


def main():
    parser = argparse.ArgumentParser(
        description='TRIFFID UAV Perception Pipeline',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str, help='Process a single image')
    mode.add_argument('--batch', type=str, help='Process all images in directory')
    mode.add_argument('--watch', type=str, help='Watch directory for new images')
    mode.add_argument('--poll-api', action='store_true',
                      help='Poll FUTURISED API for new drone images')

    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                        help='YOLO model path (default: yolo11n-seg.pt)')
    parser.add_argument('--confidence', type=float, default=0.35,
                        help='Detection confidence threshold (default: 0.35)')
    parser.add_argument('--mqtt-host', type=str, default='localhost')
    parser.add_argument('--mqtt-port', type=int, default=1883)
    parser.add_argument('--mqtt-topic', type=str, default='triffid/uav/geojson')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='YOLO input size (default: 1280)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saved GeoJSON (batch mode)')
    parser.add_argument('-v', '--verbose', action='store_true')

    # FUTURISED API options
    api_group = parser.add_argument_group('FUTURISED API options')
    api_group.add_argument(
        '--api-media-key', type=str,
        default=os.environ.get('FUTURISED_MEDIA_API_KEY', ''),
        help='Media Files API key (or env FUTURISED_MEDIA_API_KEY)',
    )
    api_group.add_argument(
        '--api-org-id', type=str,
        default=os.environ.get(
            'FUTURISED_ORG_ID',
            '66f9f3ae-cd33-4313-b474-ae24e923a185',
        ),
        help='Organisation UUID for media API (or env FUTURISED_ORG_ID)',
    )
    api_group.add_argument(
        '--api-telemetry-token', type=str,
        default=os.environ.get('FUTURISED_TELEMETRY_TOKEN', ''),
        help='Telemetry API bearer token (or env FUTURISED_TELEMETRY_TOKEN)',
    )
    api_group.add_argument(
        '--api-telemetry-project', type=str, default='Triffid_test',
        help='Telemetry project name (default: Triffid_test)',
    )
    api_group.add_argument(
        '--api-poll-interval', type=float, default=10.0,
        help='Seconds between API polls (default: 10)',
    )
    api_group.add_argument(
        '--api-camera', type=str, default='Wide',
        help='Camera filter: Wide, Zoom, Thermal, or empty for all (default: Wide)',
    )
    api_group.add_argument(
        '--api-download-dir', type=str, default='./uav_images',
        help='Local directory for downloaded images (default: ./uav_images)',
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    pipeline = UAVPipeline(
        model_path=args.model,
        confidence=args.confidence,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_topic=args.mqtt_topic,
        yolo_imgsz=args.imgsz,
    )

    try:
        if args.image:
            result = pipeline.process_image(args.image)
            if result:
                print(json.dumps(result, indent=2))
        elif args.batch:
            _process_batch(pipeline, args.batch, args.output)
        elif args.watch:
            _watch_directory(pipeline, args.watch)
        elif args.poll_api:
            if not args.api_media_key:
                parser.error(
                    '--poll-api requires --api-media-key or '
                    'env FUTURISED_MEDIA_API_KEY'
                )
            client = FuturisedClient(
                media_api_key=args.api_media_key,
                org_id=args.api_org_id,
                telemetry_token=args.api_telemetry_token or None,
                telemetry_project=args.api_telemetry_project,
                download_dir=args.api_download_dir,
            )
            _poll_api(
                pipeline, client,
                poll_interval=args.api_poll_interval,
                camera=args.api_camera,
                output_dir=args.output,
            )
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.shutdown()


if __name__ == '__main__':
    main()
