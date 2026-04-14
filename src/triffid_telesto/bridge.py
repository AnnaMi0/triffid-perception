#!/usr/bin/env python3
"""MQTT → TELESTO bridge.

Subscribes to UGV and UAV GeoJSON MQTT topics, merges features from both
sources into a single FeatureCollection, and syncs to the TELESTO backend.

Usage::

    python3 -m triffid_telesto.bridge                             # defaults
    python3 -m triffid_telesto.bridge --mqtt-host 192.168.1.100   # remote broker
    python3 -m triffid_telesto.bridge --dry-run                   # print, don't upload

Environment variables (override CLI defaults):
    TELESTO_BASE_URL        Map Manager endpoint
    TELESTO_OBSERVER_URL    Observer Sync endpoint
    MQTT_HOST               Broker hostname
    MQTT_PORT               Broker port
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional

try:
    import paho.mqtt.client as paho_mqtt
    _PAHO = True
except ImportError:
    _PAHO = False

from triffid_telesto.telesto_client import TelestoClient, TelestoError

log = logging.getLogger('triffid_telesto.bridge')

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_UGV_TOPIC = 'ugv/detections/front/geojson'
_UAV_TOPIC = 'triffid/uav/geojson'
_SYNC_INTERVAL = 2.0  # seconds between TELESTO uploads


class Bridge:
    """MQTT → TELESTO bridge.

    Listens to UGV and UAV GeoJSON MQTT topics, merges features, and
    periodically syncs the merged set to the TELESTO Map Manager API.
    """

    def __init__(
        self,
        mqtt_host: str = 'localhost',
        mqtt_port: int = 1883,
        ugv_topic: str = _UGV_TOPIC,
        uav_topic: str = _UAV_TOPIC,
        telesto_base: Optional[str] = None,
        telesto_observer: Optional[str] = None,
        sync_interval: float = _SYNC_INTERVAL,
        notify_observer: bool = True,
        dry_run: bool = False,
    ):
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.ugv_topic = ugv_topic
        self.uav_topic = uav_topic
        self.sync_interval = sync_interval
        self.notify_observer = notify_observer
        self.dry_run = dry_run

        # Latest FeatureCollections from each source
        self._lock = threading.Lock()
        self._ugv_latest: Optional[dict] = None
        self._uav_latest: Optional[dict] = None
        self._dirty = False  # True when new data arrived since last sync

        # TELESTO client
        kwargs = {}
        if telesto_base:
            kwargs['base_url'] = telesto_base
        if telesto_observer:
            kwargs['observer_url'] = telesto_observer
        self._client = TelestoClient(**kwargs)

        # MQTT client
        self._mqtt: Optional[object] = None

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            log.info(f'MQTT connected to {self.mqtt_host}:{self.mqtt_port}')
            client.subscribe(self.ugv_topic, qos=0)
            client.subscribe(self.uav_topic, qos=0)
            log.info(f'Subscribed: {self.ugv_topic}, {self.uav_topic}')
        else:
            log.error(f'MQTT connect failed: rc={rc}')

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f'Bad payload on {msg.topic}: {e}')
            return

        if payload.get('type') != 'FeatureCollection':
            log.warning(f'Ignoring non-FeatureCollection on {msg.topic}')
            return

        with self._lock:
            if msg.topic == self.ugv_topic:
                self._ugv_latest = payload
                log.debug(
                    f'UGV: {len(payload.get("features", []))} features'
                )
            elif msg.topic == self.uav_topic:
                self._uav_latest = payload
                log.debug(
                    f'UAV: {len(payload.get("features", []))} features'
                )
            self._dirty = True

    def _merge(self) -> dict:
        """Merge latest UGV + UAV features into one FeatureCollection."""
        features = []
        with self._lock:
            if self._ugv_latest:
                features.extend(self._ugv_latest.get('features', []))
            if self._uav_latest:
                features.extend(self._uav_latest.get('features', []))
            self._dirty = False
        return {'type': 'FeatureCollection', 'features': features}

    def _sync_loop(self):
        """Periodically sync merged features to TELESTO."""
        while True:
            time.sleep(self.sync_interval)

            with self._lock:
                dirty = self._dirty

            if not dirty:
                continue

            merged = self._merge()
            n = len(merged['features'])

            if self.dry_run:
                log.info(f'[dry-run] Would sync {n} features to TELESTO')
                print(json.dumps(merged, indent=2))
                continue

            if n == 0:
                log.debug('No features to sync.')
                continue

            try:
                # Upload individual features grouped by source
                ugv_features = [
                    f for f in merged['features']
                    if f.get('properties', {}).get('source') == 'ugv'
                ]
                uav_features = [
                    f for f in merged['features']
                    if f.get('properties', {}).get('source') == 'uav'
                ]

                stats_total = {
                    'created': 0, 'updated': 0,
                    'deleted': 0, 'errors': 0,
                }

                if ugv_features:
                    ugv_coll = {
                        'type': 'FeatureCollection',
                        'features': ugv_features,
                    }
                    s = self._client.sync_collection(ugv_coll, source='ugv')
                    for k in stats_total:
                        stats_total[k] += s[k]

                if uav_features:
                    uav_coll = {
                        'type': 'FeatureCollection',
                        'features': uav_features,
                    }
                    s = self._client.sync_collection(uav_coll, source='uav')
                    for k in stats_total:
                        stats_total[k] += s[k]

                log.info(
                    f'Synced {n} features → '
                    f'created={stats_total["created"]} '
                    f'updated={stats_total["updated"]} '
                    f'deleted={stats_total["deleted"]} '
                    f'errors={stats_total["errors"]}'
                )

                # Notify observer
                if self.notify_observer and stats_total['errors'] == 0:
                    try:
                        self._client.notify_observer(fe_updated=1)
                    except TelestoError as e:
                        log.warning(f'Observer notify failed: {e}')

            except Exception as e:
                log.error(f'Sync failed: {e}')

    def run(self):
        """Start the bridge (blocking)."""
        if not _PAHO:
            raise RuntimeError(
                'paho-mqtt is required.  Install: pip install paho-mqtt'
            )

        # MQTT setup
        self._mqtt = paho_mqtt.Client(
            paho_mqtt.CallbackAPIVersion.VERSION2,
            client_id='triffid-telesto-bridge',
            protocol=paho_mqtt.MQTTv311,
        )
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_message = self._on_message

        log.info(f'Connecting to MQTT {self.mqtt_host}:{self.mqtt_port}...')
        self._mqtt.connect(self.mqtt_host, self.mqtt_port)

        # Start sync thread
        sync_thread = threading.Thread(
            target=self._sync_loop, daemon=True, name='telesto-sync',
        )
        sync_thread.start()

        # MQTT loop (blocking)
        try:
            self._mqtt.loop_forever()
        except KeyboardInterrupt:
            log.info('Shutting down...')
        finally:
            self._mqtt.disconnect()


def main():
    parser = argparse.ArgumentParser(
        description='MQTT → TELESTO GeoJSON bridge',
    )
    parser.add_argument(
        '--mqtt-host',
        default=os.environ.get('MQTT_HOST', 'localhost'),
    )
    parser.add_argument(
        '--mqtt-port', type=int,
        default=int(os.environ.get('MQTT_PORT', '1883')),
    )
    parser.add_argument('--ugv-topic', default=_UGV_TOPIC)
    parser.add_argument('--uav-topic', default=_UAV_TOPIC)
    parser.add_argument(
        '--telesto-base',
        default=os.environ.get('TELESTO_BASE_URL'),
    )
    parser.add_argument(
        '--telesto-observer',
        default=os.environ.get('TELESTO_OBSERVER_URL'),
    )
    parser.add_argument(
        '--sync-interval', type=float, default=_SYNC_INTERVAL,
        help='Seconds between TELESTO uploads (default: 2.0)',
    )
    parser.add_argument(
        '--no-observer', action='store_true',
        help='Disable observer notifications',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print merged GeoJSON instead of uploading',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    bridge = Bridge(
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        ugv_topic=args.ugv_topic,
        uav_topic=args.uav_topic,
        telesto_base=args.telesto_base,
        telesto_observer=args.telesto_observer,
        sync_interval=args.sync_interval,
        notify_observer=not args.no_observer,
        dry_run=args.dry_run,
    )
    bridge.run()


if __name__ == '__main__':
    main()
