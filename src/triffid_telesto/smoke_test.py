#!/usr/bin/env python3
"""Bridge smoke test.

Publishes synthetic UGV and UAV GeoJSON FeatureCollections to the local MQTT
broker, then starts the bridge in --dry-run mode for a few seconds and prints
the merged, deduplicated output.

Verifies:
  1. Bridge receives messages from both sources.
  2. Cross-platform dedup removes features that overlap within 10 m.
  3. local_frame UGV features are passed through even without GPS.
  4. Higher-confidence feature wins when two overlap.

Usage (inside container):
    PYTHONPATH=/ws/src python3 /ws/src/triffid_telesto/smoke_test.py
    PYTHONPATH=/ws/src python3 /ws/src/triffid_telesto/smoke_test.py \\
        --mqtt-host localhost --mqtt-port 1883
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time

try:
    import paho.mqtt.client as paho_mqtt
    _PAHO = True
except ImportError:
    _PAHO = False

_UGV_TOPIC = 'ugv/detections/front/geojson'
_UAV_TOPIC = 'triffid/uav/geojson'


def _feature(source, fid, cls, lon, lat, confidence, local_frame=False):
    return {
        'type': 'Feature',
        'id': fid,
        'geometry': {'type': 'Point', 'coordinates': [lon, lat, 0.0]},
        'properties': {
            'class': cls,
            'id': fid,
            'confidence': confidence,
            'source': source,
            'local_frame': local_frame,
            'category': 'hazard',
            'detection_type': 'seg',
            'altitude_m': 320.0,
            'height_m': 2.0,
            'marker-color': '#ff0000',
            'marker-size': 'medium',
            'marker-symbol': 'circle',
        },
    }


def _collection(*features):
    return {'type': 'FeatureCollection', 'features': list(features)}


def build_test_payloads():
    """Return (ugv_payload, uav_payload) as JSON strings.

    Scenario:
    - UGV detects 'flame' at (23.7200, 37.9800) conf=0.90  — GPS fix
    - UGV detects 'debris' local_frame=True, conf=0.80     — body-frame
    - UAV detects 'flame' at (23.7200, 37.9801) conf=0.70  — ~11 m away, same class
    - UAV detects 'vehicle' at (23.7500, 37.9800) conf=0.85 — far away

    Expected merged result: 3 features
      ugv  flame   0.90  (UAV flame suppressed — same class, within 10 m)
      ugv  debris  0.80  [local_frame — always passes through]
      uav  vehicle 0.85  (unique location)
    """
    ugv = _collection(
        _feature('ugv', 'u1', 'flame',  23.7200, 37.9800, 0.90),
        _feature('ugv', 'u2', 'debris', 0.0,     0.0,     0.80, local_frame=True),
    )
    uav = _collection(
        _feature('uav', 'a1', 'flame',   23.7200, 37.9801, 0.70),
        _feature('uav', 'a2', 'vehicle', 23.7500, 37.9800, 0.85),
    )
    return json.dumps(ugv), json.dumps(uav)


def _drain_stream(stream, q: queue.Queue):
    """Thread target: read stream line-by-line and put each line into q."""
    try:
        for line in iter(stream.readline, ''):
            q.put(line)
    finally:
        q.put(None)  # sentinel


def _wait_for_subscribed(stderr_q: queue.Queue, timeout: float = 10.0) -> list[str]:
    """Block until we see 'Subscribed' in stderr (bridge ready), or timeout."""
    seen = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = stderr_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            break
        seen.append(line.rstrip())
        if 'Subscribed' in line:
            return seen
    return seen


def run_smoke_test(mqtt_host: str, mqtt_port: int) -> int:
    if not _PAHO:
        print('ERROR: paho-mqtt not installed.', file=sys.stderr)
        return 1

    import os

    ugv_payload, uav_payload = build_test_payloads()

    # Run bridge with -u (unbuffered stdout) so dry-run JSON prints immediately.
    bridge_cmd = [
        sys.executable, '-u', '-m', 'triffid_telesto.bridge',
        '--mqtt-host', mqtt_host,
        '--mqtt-port', str(mqtt_port),
        '--dry-run',
        '--sync-interval', '2.0',
    ]
    env = dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH', ''))
    proc = subprocess.Popen(
        bridge_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )

    # Drain stdout and stderr on background threads so we never deadlock.
    stdout_q: queue.Queue = queue.Queue()
    stderr_q: queue.Queue = queue.Queue()
    threading.Thread(target=_drain_stream, args=(proc.stdout, stdout_q), daemon=True).start()
    threading.Thread(target=_drain_stream, args=(proc.stderr, stderr_q), daemon=True).start()

    # Wait until bridge has connected to MQTT and subscribed — then it's safe to publish.
    print('Waiting for bridge to connect to MQTT...')
    stderr_lines = _wait_for_subscribed(stderr_q, timeout=10.0)
    if not any('Subscribed' in l for l in stderr_lines):
        print('ERROR: bridge did not connect to MQTT within 10 s')
        print('stderr so far:', '\n'.join(stderr_lines))
        proc.terminate()
        return 1

    print('Bridge connected. Publishing test messages...')

    client = paho_mqtt.Client(
        paho_mqtt.CallbackAPIVersion.VERSION2,
        client_id='smoke-test-publisher',
    )
    client.connect(mqtt_host, mqtt_port)
    client.loop_start()
    client.publish(_UGV_TOPIC, ugv_payload, qos=1)
    client.publish(_UAV_TOPIC, uav_payload, qos=1)
    print(f'Published UGV ({len(json.loads(ugv_payload)["features"])} features) '
          f'and UAV ({len(json.loads(uav_payload)["features"])} features) to MQTT.')
    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()

    # Collect stdout for one sync cycle (2 s) + a small buffer.
    stdout_lines = []
    deadline = time.monotonic() + 3.5
    while time.monotonic() < deadline:
        try:
            line = stdout_q.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            break
        stdout_lines.append(line)

    proc.terminate()
    proc.wait(timeout=5)

    # Also drain remaining stderr for diagnostics.
    remaining_stderr = []
    while True:
        try:
            line = stderr_q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            break
        remaining_stderr.append(line.rstrip() if line else '')

    stdout_text = ''.join(stdout_lines).strip()

    print('\n── Bridge merged output (dry-run) ──────────────────────────')
    features = []
    if stdout_text:
        try:
            merged = json.loads(stdout_text)
            features = merged.get('features', [])
            print(f'Total features after cross-platform dedup: {len(features)}')
            for f in features:
                p = f['properties']
                lf = ' [local_frame]' if p.get('local_frame') else ''
                print(f"  {p['source']:3s}  {p['class']:20s}  conf={p['confidence']:.2f}{lf}")
        except json.JSONDecodeError:
            print('(could not parse JSON — raw output below)')
            print(stdout_text)
    else:
        print('(no dry-run output received)')
        if remaining_stderr:
            print('bridge log:', '\n  '.join(remaining_stderr[-10:]))
        return 1

    print('────────────────────────────────────────────────────────────\n')

    # Validate expectations
    ok = True
    if features:
        classes_sources = {(f['properties']['source'], f['properties']['class'])
                           for f in features}
        n = len(features)

        if n != 3:
            print(f'FAIL: expected 3 features after dedup, got {n}')
            ok = False
        else:
            print('PASS: dedup produced correct feature count (3)')

        if ('ugv', 'flame') not in classes_sources:
            print("FAIL: UGV 'flame' missing from output")
            ok = False
        if ('uav', 'flame') in classes_sources:
            print("FAIL: UAV 'flame' should have been suppressed (UGV wins on confidence)")
            ok = False
        if ('ugv', 'debris') not in classes_sources:
            print("FAIL: local_frame 'debris' should have passed through")
            ok = False
        if ('uav', 'vehicle') not in classes_sources:
            print("FAIL: UAV 'vehicle' should be present (no overlap)")
            ok = False

        if ok:
            print('PASS: all deduplication assertions correct')
    else:
        print('FAIL: no features in output')
        ok = False

    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description='Bridge smoke test')
    parser.add_argument('--mqtt-host', default='localhost')
    parser.add_argument('--mqtt-port', type=int, default=1883)
    args = parser.parse_args()
    sys.exit(run_smoke_test(args.mqtt_host, args.mqtt_port))


if __name__ == '__main__':
    main()
