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
    0: 'Water', 1: 'Fence', 2: 'Green tree', 3: 'Helmet',
    4: 'Flame', 5: 'Smoke', 6: 'First responder', 7: 'Destroyed vehicle',
    8: 'Fire hose', 9: 'SCBA', 10: 'Boot', 11: 'Green plant',
    12: 'Mask', 13: 'Window', 14: 'Building', 15: 'Destroyed building',
    16: 'Debris', 17: 'Ladder', 18: 'Dirt road', 19: 'Dry tree',
    20: 'Wall', 21: 'Civilian vehicle', 22: 'Road', 23: 'Citizen',
    24: 'Green grass', 25: 'Pole', 26: 'Boat', 27: 'Pavement',
    28: 'Dry grass', 29: 'Animal', 30: 'Excavator', 31: 'Door',
    32: 'Mud', 33: 'Barrier', 34: 'Hole in the ground', 35: 'Bag',
    36: 'Burnt tree', 37: 'Ambulance', 38: 'Fire truck', 39: 'Cone',
    40: 'Bicycle', 41: 'Tower', 42: 'Silo', 43: 'Military personnel',
    44: 'Burnt grass', 45: 'Ax', 46: 'Glove', 47: 'Crane',
    48: 'Stairs', 49: 'Dry plant', 50: 'Furniture', 51: 'Tank',
    52: 'Protective glasses', 53: 'Barrel', 54: 'Shovel',
    55: 'Fire hydrant', 56: 'Police vehicle', 57: 'Burnt plant',
    58: 'Army vehicle', 59: 'Chainsaw', 60: 'aerial vehicle',
    61: 'Lifesaver', 62: 'Extinguisher',
}

# Same styling helpers as UGV geojson_bridge — kept here to avoid
# importing from the UGV package (they're independent deployments).

_CLASS_COLORS = {
    'Flame': '#ff0000', 'Smoke': '#ff4500', 'Burnt tree': '#8b0000',
    'Burnt grass': '#a52a2a', 'Burnt plant': '#b22222',
    'Fire hose': '#dc143c', 'Fire hydrant': '#ff6347',
    'Fire truck': '#ff0000', 'Extinguisher': '#ff1493',
    'First responder': '#1e90ff', 'Citizen': '#4169e1',
    'Military personnel': '#000080',
    'Civilian vehicle': '#0000ff', 'Destroyed vehicle': '#00008b',
    'Ambulance': '#4682b4', 'Police vehicle': '#191970',
    'Army vehicle': '#2f4f4f', 'Boat': '#5f9ea0', 'Bicycle': '#00ff00',
    'aerial vehicle': '#87ceeb',
    'Green tree': '#228b22', 'Green plant': '#32cd32',
    'Green grass': '#7cfc00', 'Dry tree': '#daa520',
    'Dry grass': '#bdb76b', 'Dry plant': '#f0e68c', 'Animal': '#ff8c00',
    'Building': '#708090', 'Destroyed building': '#696969',
    'Wall': '#808080', 'Road': '#a9a9a9', 'Pavement': '#c0c0c0',
    'Dirt road': '#d2b48c', 'Window': '#b0c4de', 'Door': '#8b4513',
    'Stairs': '#a0522d', 'Pole': '#778899', 'Tower': '#556b2f',
    'Silo': '#6b8e23',
    'Debris': '#ff8c00', 'Fence': '#daa520', 'Barrier': '#ffd700',
    'Cone': '#ff7f50', 'Hole in the ground': '#8b4513',
    'Mud': '#a0522d', 'Water': '#00bfff',
    'Helmet': '#9370db', 'SCBA': '#8a2be2', 'Boot': '#4b0082',
    'Mask': '#9400d3', 'Glove': '#da70d6', 'Protective glasses': '#ba55d3',
    'Ladder': '#ff8c00', 'Ax': '#cd853f', 'Shovel': '#d2691e',
    'Chainsaw': '#b8860b', 'Bag': '#bc8f8f', 'Barrel': '#8b8682',
    'Furniture': '#deb887', 'Tank': '#2e8b57', 'Crane': '#b8860b',
    'Excavator': '#daa520', 'Lifesaver': '#ff4500',
}

_CLASS_CATEGORIES = {
    'Flame': 'hazard', 'Smoke': 'hazard', 'Burnt tree': 'hazard',
    'Burnt grass': 'hazard', 'Burnt plant': 'hazard',
    'First responder': 'person', 'Citizen': 'person',
    'Military personnel': 'person',
    'Civilian vehicle': 'vehicle', 'Destroyed vehicle': 'vehicle',
    'Ambulance': 'vehicle', 'Police vehicle': 'vehicle',
    'Fire truck': 'vehicle', 'Army vehicle': 'vehicle',
    'Boat': 'vehicle', 'Bicycle': 'vehicle', 'aerial vehicle': 'vehicle',
    'Green tree': 'nature', 'Green plant': 'nature',
    'Green grass': 'nature', 'Dry tree': 'nature',
    'Dry grass': 'nature', 'Dry plant': 'nature', 'Animal': 'nature',
    'Building': 'infrastructure', 'Destroyed building': 'infrastructure',
    'Wall': 'infrastructure', 'Road': 'infrastructure',
    'Pavement': 'infrastructure', 'Dirt road': 'infrastructure',
    'Window': 'infrastructure', 'Door': 'infrastructure',
    'Stairs': 'infrastructure', 'Pole': 'infrastructure',
    'Tower': 'infrastructure', 'Silo': 'infrastructure',
    'Debris': 'obstacle', 'Fence': 'obstacle', 'Barrier': 'obstacle',
    'Cone': 'obstacle', 'Hole in the ground': 'obstacle',
    'Mud': 'obstacle', 'Water': 'obstacle',
    'Fire hose': 'equipment', 'Fire hydrant': 'equipment',
    'Extinguisher': 'equipment', 'Helmet': 'equipment',
    'SCBA': 'equipment', 'Boot': 'equipment', 'Mask': 'equipment',
    'Glove': 'equipment', 'Protective glasses': 'equipment',
    'Ladder': 'equipment', 'Ax': 'equipment', 'Shovel': 'equipment',
    'Chainsaw': 'equipment', 'Bag': 'equipment', 'Barrel': 'equipment',
    'Furniture': 'equipment', 'Tank': 'equipment', 'Crane': 'equipment',
    'Excavator': 'equipment', 'Lifesaver': 'equipment',
}

_CLASS_SYMBOLS = {
    'First responder': 'pitch', 'Citizen': 'pitch',
    'Military personnel': 'pitch',
    'Civilian vehicle': 'car', 'Destroyed vehicle': 'car',
    'Ambulance': 'hospital', 'Police vehicle': 'police',
    'Fire truck': 'fire-station', 'Army vehicle': 'car',
    'Boat': 'harbor', 'Bicycle': 'bicycle', 'aerial vehicle': 'airfield',
    'Flame': 'fire-station', 'Smoke': 'fire-station',
    'Building': 'building', 'Destroyed building': 'building',
    'Water': 'water', 'Animal': 'dog-park',
}

# Person classes get Point geometry (small, mobile targets)
_POINT_CLASSES = frozenset([
    'First responder', 'Citizen', 'Military personnel',
])

# Linear structures get LineString
_LINE_CLASSES = frozenset([
    'Fence',
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
                "gnss_altitude_m": round(centre_alt, 2),
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


def main():
    parser = argparse.ArgumentParser(
        description='TRIFFID UAV Perception Pipeline',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image', type=str, help='Process a single image')
    mode.add_argument('--batch', type=str, help='Process all images in directory')
    mode.add_argument('--watch', type=str, help='Watch directory for new images')

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
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.shutdown()


if __name__ == '__main__':
    main()
