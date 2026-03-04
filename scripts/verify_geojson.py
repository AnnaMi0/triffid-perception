#!/usr/bin/env python3
"""
TRIFFID GeoJSON Verification Tool
==================================
Parses MQTT trace files and generates an interactive Leaflet HTML map
overlaying all GeoJSON features on OpenStreetMap satellite imagery.

Usage:
    python3 scripts/verify_geojson.py [trace_file] [--frame N] [--every N]

Options:
    trace_file    Path to mqtt_traces/geojson_trace_*.jsonl (default: latest)
    --frame N     Show only frame N (0-based)
    --every N     Show every Nth frame (default: 50, to avoid browser overload)
    --last N      Show only the last N frames
    --output FILE Output HTML file (default: samples/geojson_map.html)

Open the resulting HTML file in a browser for visual verification.
"""

import json
import math
import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict


def parse_trace(filepath):
    """Parse MQTT trace file → list of GeoJSON dicts."""
    with open(filepath) as f:
        raw = f.read()
    parts = raw.split('triffid/ugv/geojson ')
    msgs = []
    for p in parts[1:]:
        p = p.strip()
        if p:
            try:
                msgs.append(json.loads(p))
            except json.JSONDecodeError:
                pass
    return msgs


def feature_centre(feat):
    """Get [lon, lat] centre of a feature."""
    geom = feat['geometry']
    if geom['type'] == 'Point':
        return geom['coordinates']
    elif geom['type'] == 'Polygon':
        coords = geom['coordinates'][0][:-1]  # exclude closing point
        lon = sum(c[0] for c in coords) / len(coords)
        lat = sum(c[1] for c in coords) / len(coords)
        return [lon, lat]
    return [0, 0]


def generate_html(msgs, frame_indices, output_path):
    """Generate Leaflet HTML map."""
    # Collect all features with frame info
    all_features = []
    centre_lon, centre_lat = 0, 0
    n = 0

    for fi in frame_indices:
        msg = msgs[fi]
        for feat in msg.get('features', []):
            c = feature_centre(feat)
            centre_lon += c[0]
            centre_lat += c[1]
            n += 1
            feat_copy = dict(feat)
            feat_copy['properties'] = dict(feat['properties'])
            feat_copy['properties']['_frame'] = fi
            all_features.append(feat_copy)

    if n > 0:
        centre_lon /= n
        centre_lat /= n

    # Build per-class GeoJSON layers
    by_class = defaultdict(list)
    for f in all_features:
        cls = f['properties'].get('class', 'unknown')
        by_class[cls].append(f)

    layers_js = []
    for cls, feats in sorted(by_class.items()):
        colour = feats[0]['properties'].get('marker-color', '#808080')
        geojson_str = json.dumps({
            "type": "FeatureCollection",
            "features": feats
        })
        layers_js.append(f"""
        (function() {{
            var data = {geojson_str};
            var layer = L.geoJSON(data, {{
                style: function(feature) {{
                    return {{
                        color: '{colour}',
                        weight: 2,
                        fillColor: '{colour}',
                        fillOpacity: 0.3
                    }};
                }},
                pointToLayer: function(feature, latlng) {{
                    return L.circleMarker(latlng, {{
                        radius: 6,
                        fillColor: '{colour}',
                        color: '#000',
                        weight: 1,
                        fillOpacity: 0.8
                    }});
                }},
                onEachFeature: function(feature, layer) {{
                    var p = feature.properties;
                    layer.bindPopup(
                        '<b>' + p['class'] + '</b> (ID: ' + p.id + ')<br>' +
                        'Confidence: ' + (p.confidence * 100).toFixed(1) + '%<br>' +
                        'Category: ' + p.category + '<br>' +
                        'Frame: ' + p._frame
                    );
                }}
            }});
            overlays["{cls} ({len(feats)})"] = layer;
            layer.addTo(map);
        }})();""")

    # Robot path: extract centre of each frame's detections
    path_points = []
    for fi in frame_indices:
        msg = msgs[fi]
        feats = msg.get('features', [])
        if not feats:
            continue
        # The robot position is approximately at the "0,0 offset" detection
        # We can't recover it exactly, but the cluster centre is a proxy
        lons = [feature_centre(f)[0] for f in feats]
        lats = [feature_centre(f)[1] for f in feats]
        path_points.append([sum(lats)/len(lats), sum(lons)/len(lons)])

    path_js = json.dumps(path_points)

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>TRIFFID GeoJSON Verification</title>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
    body {{ margin: 0; padding: 0; }}
    #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
    .info {{ padding: 6px 8px; font: 14px Arial; background: white;
             box-shadow: 0 0 15px rgba(0,0,0,0.2); border-radius: 5px;
             max-width: 350px; }}
    .info h4 {{ margin: 0 0 5px; }}
</style>
</head>
<body>
<div id="map"></div>
<script>
    // Base layers
    var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 22,
        attribution: '&copy; OpenStreetMap'
    }});
    var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        maxZoom: 22,
        attribution: '&copy; Esri'
    }});

    var map = L.map('map', {{
        center: [{centre_lat}, {centre_lon}],
        zoom: 19,
        layers: [satellite]
    }});

    L.control.layers({{
        "Satellite": satellite,
        "OpenStreetMap": osm
    }}).addTo(map);

    var overlays = {{}};

    // Add feature layers
    {''.join(layers_js)}

    // Add layer control for classes
    L.control.layers(null, overlays, {{collapsed: false}}).addTo(map);

    // Robot approximate path
    var path = {path_js};
    if (path.length > 1) {{
        L.polyline(path, {{color: 'yellow', weight: 3, dashArray: '5,10', opacity: 0.7}}).addTo(map);
    }}

    // Info box
    var info = L.control({{position: 'bottomleft'}});
    info.onAdd = function() {{
        var div = L.DomUtil.create('div', 'info');
        div.innerHTML = '<h4>TRIFFID GeoJSON Verify</h4>' +
            'Frames shown: {len(frame_indices)}<br>' +
            'Total features: {len(all_features)}<br>' +
            'Classes: {len(by_class)}<br>' +
            '<small>Yellow dashed = approx robot path</small>';
        return div;
    }};
    info.addTo(map);
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Map saved to: {output_path}")
    print(f"  Frames: {len(frame_indices)}, Features: {len(all_features)}, Classes: {len(by_class)}")
    print(f"  Centre: ({centre_lat:.6f}, {centre_lon:.6f})")
    print(f"\nOpen in browser:  xdg-open {output_path}")


def main():
    parser = argparse.ArgumentParser(description='TRIFFID GeoJSON verification map')
    parser.add_argument('trace_file', nargs='?', help='MQTT trace file path')
    parser.add_argument('--frame', type=int, help='Show only this frame index')
    parser.add_argument('--every', type=int, default=50, help='Show every Nth frame (default: 50)')
    parser.add_argument('--last', type=int, help='Show only the last N frames')
    parser.add_argument('--output', default='samples/geojson_map.html', help='Output HTML file')
    args = parser.parse_args()

    # Find trace file
    if args.trace_file:
        trace_path = args.trace_file
    else:
        trace_dir = Path('mqtt_traces')
        traces = sorted(trace_dir.glob('geojson_trace_*.jsonl'))
        if not traces:
            print("No trace files found in mqtt_traces/")
            sys.exit(1)
        trace_path = str(traces[-1])
        print(f"Using latest trace: {trace_path}")

    msgs = parse_trace(trace_path)
    print(f"Parsed {len(msgs)} GeoJSON messages")

    if not msgs:
        print("No messages found!")
        sys.exit(1)

    # Select frames
    if args.frame is not None:
        frame_indices = [args.frame]
    elif args.last:
        frame_indices = list(range(max(0, len(msgs) - args.last), len(msgs)))
    else:
        frame_indices = list(range(0, len(msgs), args.every))

    print(f"Selecting {len(frame_indices)} frames")
    generate_html(msgs, frame_indices, args.output)


if __name__ == '__main__':
    main()
