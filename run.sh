#!/usr/bin/env bash
# =============================================================
# TRIFFID Perception – All-in-one runner
# =============================================================
#
# Usage:
#   ./run.sh build          # Build Docker image + colcon workspace
#   ./run.sh start [MODE]   # MODE=replay|live (default: replay)
#   ./run.sh stop           # Stop everything
#   ./run.sh restart [MODE] # stop + start with same MODE
#   ./run.sh test           # Run integration test (inside container)
#   ./run.sh unit           # Run unit tests (host, no Docker needed)
#   ./run.sh streams [SEC]  # Validate live RealSense streams (default: 12s)
#   ./run.sh sample [SEC]   # Collect output samples (default: wait 15s)
#   ./run.sh record         # Record output topics to output_rosbag/
#   ./run.sh logs           # Tail node logs
#   ./run.sh shell          # Open bash inside the container
#   ./run.sh status         # Show running processes + topic list
#
# Environment variables:
#   BAG_RATE    Rosbag playback rate     (default: 1.0)
#   BAG_START   Seconds to skip in bag   (default: 0)
#   YOLO_IMGSZ  YOLO inference size      (default: 1280)
#   TIMEOUT     Integration test timeout (default: 30)
#   TOPIC_SOURCE Topic selection mode: auto|manual (default: auto)
#   RGB_TOPIC   RGB image topic override
#   DEPTH_TOPIC Depth image topic override
#   CAMERA_INFO_TOPIC CameraInfo topic override
#   DEPTH_CAMERA_INFO_TOPIC Depth CameraInfo topic override (for test/record checks)
#   SAMPLE_PLAY_BAG Replay rosbag before sampling: true|false (default: true)
#   START_PLAY_BAG Replay rosbag during ./run.sh start: true|false (default: true)
#   MQTT_ENABLED Enable MQTT publish path via launch (default: true)
#   MQTT_HOST  MQTT broker host for GeoJSON publish (default: localhost)
#   MQTT_PORT  MQTT broker port for GeoJSON publish (default: 1883)
#   MQTT_TOPIC MQTT topic for GeoJSON publish/trace (default: ugv/detections/front/geojson)
#   STREAMS_DEBUG_DIR Directory for stream debug snapshots (default: /ws/samples/realsense_debug)
# =============================================================
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER=triffid_perception
IMAGE=triffid_perception:latest
BAG_RATE="${BAG_RATE:-1.0}"
BAG_START="${BAG_START:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
TIMEOUT="${TIMEOUT:-30}"
TOPIC_SOURCE="${TOPIC_SOURCE:-auto}"
DEFAULT_RGB_TOPIC="/camera_front_435i/realsense_front_435i/color/image_raw"
DEFAULT_DEPTH_TOPIC="/camera_front_435i/realsense_front_435i/depth/image_rect_raw"
DEFAULT_CAMERA_INFO_TOPIC="/camera_front_435i/realsense_front_435i/color/camera_info"
DEFAULT_DEPTH_CAMERA_INFO_TOPIC="/camera_front_435i/realsense_front_435i/depth/camera_info"
RGB_TOPIC="${RGB_TOPIC:-$DEFAULT_RGB_TOPIC}"
DEPTH_TOPIC="${DEPTH_TOPIC:-$DEFAULT_DEPTH_TOPIC}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-$DEFAULT_CAMERA_INFO_TOPIC}"
DEPTH_CAMERA_INFO_TOPIC="${DEPTH_CAMERA_INFO_TOPIC:-}"
SAMPLE_PLAY_BAG="${SAMPLE_PLAY_BAG:-true}"
START_PLAY_BAG="${START_PLAY_BAG:-true}"
MQTT_ENABLED="${MQTT_ENABLED:-true}"
MQTT_HOST="${MQTT_HOST:-localhost}"
MQTT_PORT="${MQTT_PORT:-1883}"
MQTT_TOPIC="${MQTT_TOPIC:-ugv/detections/front/geojson}"
STREAMS_DEBUG_DIR="${STREAMS_DEBUG_DIR:-/ws/samples/realsense_debug}"
if [[ -z "$DEPTH_CAMERA_INFO_TOPIC" ]]; then
    if [[ "$DEPTH_TOPIC" == */image_rect_raw ]]; then
        DEPTH_CAMERA_INFO_TOPIC="${DEPTH_TOPIC%/image_rect_raw}/camera_info"
    else
        DEPTH_CAMERA_INFO_TOPIC="$DEFAULT_DEPTH_CAMERA_INFO_TOPIC"
    fi
fi
ACTIVE_RGB_TOPIC="$RGB_TOPIC"
ACTIVE_DEPTH_TOPIC="$DEPTH_TOPIC"
ACTIVE_CAMERA_INFO_TOPIC="$CAMERA_INFO_TOPIC"
ACTIVE_DEPTH_CAMERA_INFO_TOPIC="$DEPTH_CAMERA_INFO_TOPIC"

# Shared ROS env sourcing command (used inside docker exec)
ROS_SETUP='if [ -n "${ROS_DISTRO:-}" ] && [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then source "/opt/ros/${ROS_DISTRO}/setup.bash"; elif [ -f /opt/ros/jazzy/setup.bash ]; then source /opt/ros/jazzy/setup.bash; elif [ -f /opt/ros/humble/setup.bash ]; then source /opt/ros/humble/setup.bash; else echo "No ROS setup.bash found in container." >&2; exit 1; fi'
ROS_ENV="$ROS_SETUP && source /ws/install/setup.bash && export ROS_DOMAIN_ID=42 && export CYCLONEDDS_URI=file:///ws/cyclonedds.xml"
ROS_BUILD_ENV="$ROS_SETUP && export ROS_DOMAIN_ID=42 && export CYCLONEDDS_URI=file:///ws/cyclonedds.xml"

# ── helpers ──────────────────────────────────────────────────

_running() { docker ps -q -f name="$CONTAINER" 2>/dev/null | grep -q .; }

_ensure_running() {
    if ! _running; then
        echo "Container $CONTAINER is not running. Run: ./run.sh start"
        exit 1
    fi
}

_kill_inside() {
    # Kill processes matching a pattern inside the container (best effort)
    docker exec "$CONTAINER" bash -c "pkill -f '$1' 2>/dev/null || true" 2>/dev/null || true
}

_derive_depth_camera_info_topic() {
    local depth_topic="$1"
    if [[ "$depth_topic" == */image_rect_raw ]]; then
        echo "${depth_topic%/image_rect_raw}/camera_info"
    elif [[ "$depth_topic" == */image_raw ]]; then
        echo "${depth_topic%/image_raw}/camera_info"
    else
        echo "$DEFAULT_DEPTH_CAMERA_INFO_TOPIC"
    fi
}

_discover_topics_from_rosbag_metadata() {
    local parsed
    parsed="$(docker exec "$CONTAINER" bash -lc "python3 - <<'PY'
import os
import re

metadata = '/ws/rosbag/metadata.yaml'
if not os.path.isfile(metadata):
    raise SystemExit(2)

topics = []
with open(metadata, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m = re.match(r'\\s*name:\\s*(/\\S+)\\s*$', line)
        if m:
            t = m.group(1)
            if t not in topics:
                topics.append(t)

if not topics:
    raise SystemExit(3)

front = [t for t in topics if t.startswith('/camera_front') or t.startswith('/camera_front_435i')]

def first(items):
    for item in items:
        if item in topics:
            return item
    return ''

def first_front(pred):
    for item in front:
        if pred(item):
            return item
    return ''

rgb = (
    first([
        '/camera_front_435i/realsense_front_435i/color/image_raw',
        '/camera_front/raw_image',
        '/camera_front/realsense_front/color/image_raw',
    ])
    or first_front(lambda t: t.endswith('/color/image_raw'))
    or first_front(lambda t: t.endswith('/raw_image') and '/depth/' not in t and '/infra' not in t)
    or first_front(lambda t: t.endswith('/image_raw') and '/depth/' not in t and '/infra' not in t)
)

depth = (
    first([
        '/camera_front_435i/realsense_front_435i/depth/image_rect_raw',
        '/camera_front/realsense_front/depth/image_rect_raw',
    ])
    or first_front(lambda t: t.endswith('/depth/image_rect_raw'))
    or first_front(lambda t: t.endswith('/aligned_depth_to_color/image_raw'))
)

def replace_last(topic: str, old: str, new: str) -> str:
    if topic.endswith(old):
        return topic[: -len(old)] + new
    return ''

camera_info = ''
for suffix in ['/color/image_raw', '/raw_image', '/image_raw']:
    if rgb.endswith(suffix):
        candidate = replace_last(rgb, suffix, '/camera_info')
        if candidate in topics:
            camera_info = candidate
            break

if not camera_info and rgb.startswith('/camera_front/'):
    if '/camera_front/camera_info' in topics:
        camera_info = '/camera_front/camera_info'

if not camera_info:
    camera_info = (
        first_front(lambda t: t.endswith('/color/camera_info'))
        or first_front(lambda t: t.endswith('/camera_info') and '/depth/' not in t and '/infra' not in t)
    )

depth_camera_info = ''
for suffix in ['/depth/image_rect_raw', '/aligned_depth_to_color/image_raw', '/image_rect_raw', '/image_raw']:
    if depth.endswith(suffix):
        candidate = replace_last(depth, suffix, '/camera_info')
        if candidate in topics:
            depth_camera_info = candidate
            break

if not depth_camera_info:
    depth_camera_info = first_front(lambda t: t.endswith('/depth/camera_info'))

print('|'.join([rgb, depth, camera_info, depth_camera_info]))
PY
")" || return 1
    echo "$parsed"
}

_resolve_active_topics() {
    local topic_source_override="${1:-}"
    local topic_source_effective="$TOPIC_SOURCE"
    if [[ -n "$topic_source_override" ]]; then
        topic_source_effective="$topic_source_override"
    fi

    ACTIVE_RGB_TOPIC="$RGB_TOPIC"
    ACTIVE_DEPTH_TOPIC="$DEPTH_TOPIC"
    ACTIVE_CAMERA_INFO_TOPIC="$CAMERA_INFO_TOPIC"
    ACTIVE_DEPTH_CAMERA_INFO_TOPIC="$DEPTH_CAMERA_INFO_TOPIC"

    local source_label="manual-env/default"
    local parsed=""
    local bag_rgb=""
    local bag_depth=""
    local bag_camera_info=""
    local bag_depth_camera_info=""

    if [[ "$topic_source_effective" == "auto" ]]; then
        parsed="$(_discover_topics_from_rosbag_metadata || true)"
        if [[ -n "$parsed" ]]; then
            IFS='|' read -r bag_rgb bag_depth bag_camera_info bag_depth_camera_info <<<"$parsed"
            [[ -n "$bag_rgb" ]] && ACTIVE_RGB_TOPIC="$bag_rgb"
            [[ -n "$bag_depth" ]] && ACTIVE_DEPTH_TOPIC="$bag_depth"
            [[ -n "$bag_camera_info" ]] && ACTIVE_CAMERA_INFO_TOPIC="$bag_camera_info"
            [[ -n "$bag_depth_camera_info" ]] && ACTIVE_DEPTH_CAMERA_INFO_TOPIC="$bag_depth_camera_info"
            source_label="rosbag metadata (/ws/rosbag/metadata.yaml)"
        else
            source_label="manual-env/default (metadata unavailable)"
        fi
    elif [[ "$topic_source_effective" != "manual" ]]; then
        echo "  Unknown TOPIC_SOURCE='$topic_source_effective' (expected auto|manual); using manual topics."
    fi

    if [[ -z "$ACTIVE_DEPTH_CAMERA_INFO_TOPIC" ]]; then
        ACTIVE_DEPTH_CAMERA_INFO_TOPIC="$(_derive_depth_camera_info_topic "$ACTIVE_DEPTH_TOPIC")"
    fi

    echo "▸ Topic source: $source_label"
    echo "  RGB topic:           $ACTIVE_RGB_TOPIC"
    echo "  Depth topic:         $ACTIVE_DEPTH_TOPIC"
    echo "  Color CameraInfo:    $ACTIVE_CAMERA_INFO_TOPIC"
    echo "  Depth CameraInfo:    $ACTIVE_DEPTH_CAMERA_INFO_TOPIC"
}

# ── commands ─────────────────────────────────────────────────

cmd_build() {
    echo "▸ Building Docker image..."
    docker compose build

    echo "▸ Building colcon workspace inside container..."
    # Start container if not running
    if ! _running; then
        docker rm -f "$CONTAINER" 2>/dev/null || true
        docker compose up -d
        sleep 2
    fi
    docker exec "$CONTAINER" bash -c "cd /ws && rm -rf build/* install/* log/* && $ROS_BUILD_ENV && colcon build --symlink-install"
    echo "✓ Build complete."
}

cmd_start() {
    local mode="${1:-replay}"
    local topic_source_for_start="$TOPIC_SOURCE"
    local start_play_bag_for_start="$START_PLAY_BAG"

    case "$mode" in
        replay)
            topic_source_for_start="auto"
            start_play_bag_for_start="true"
            ;;
        live)
            topic_source_for_start="manual"
            start_play_bag_for_start="false"
            ;;
        "")
            ;;
        *)
            echo "Unknown start mode: $mode (expected: replay|live)"
            exit 1
            ;;
    esac

    echo "▸ Starting $CONTAINER..."
    echo "  Start mode: ${mode} (topic_source=${topic_source_for_start}, play_bag=${start_play_bag_for_start})"

    # Remove stale container if exists
    if docker ps -a -q -f name="$CONTAINER" 2>/dev/null | grep -q .; then
        docker rm -f "$CONTAINER" 2>/dev/null || true
        sleep 1
    fi
    docker compose up -d
    sleep 2

    _resolve_active_topics "$topic_source_for_start"

    # Build workspace (symlink-install so source edits take effect)
    echo "▸ Building workspace..."
    docker exec "$CONTAINER" bash -c "cd /ws && rm -rf build/* install/* log/* && $ROS_BUILD_ENV && colcon build --symlink-install"

    # Launch perception stack via launch file (ugv_node + geojson_bridge)
    echo "▸ Launching perception stack (YOLO imgsz=$YOLO_IMGSZ)..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 launch triffid_ugv_perception ugv_perception.launch.py \
        model_path:=/ws/best.pt yolo_imgsz:=$YOLO_IMGSZ \
        rgb_image_topic:='$ACTIVE_RGB_TOPIC' \
        depth_image_topic:='$ACTIVE_DEPTH_TOPIC' \
        camera_info_topic:='$ACTIVE_CAMERA_INFO_TOPIC' \
        depth_camera_info_topic:='$ACTIVE_DEPTH_CAMERA_INFO_TOPIC' \
        mqtt_enabled:='$MQTT_ENABLED' \
        mqtt_host:='$MQTT_HOST' \
        mqtt_port:='$MQTT_PORT' \
        mqtt_topic:='$MQTT_TOPIC' \
        2>&1 | tee /tmp/perception_launch.log"

    # Start local MQTT broker (mosquitto) only if configured port is free.
    # Use a portable check that works even when `ss` is not installed.
    echo "▸ Starting MQTT broker (mosquitto on ${MQTT_HOST}:${MQTT_PORT})..."
    if [[ "$MQTT_HOST" != "localhost" && "$MQTT_HOST" != "127.0.0.1" ]]; then
        echo "  MQTT_HOST=$MQTT_HOST is remote; not starting local mosquitto in container."
        docker exec "$CONTAINER" bash -lc "echo 'mosquitto skipped: remote MQTT_HOST=$MQTT_HOST' > /tmp/mosquitto.log"
    elif ! [[ "$MQTT_PORT" =~ ^[0-9]+$ ]]; then
        echo "  Invalid MQTT_PORT='$MQTT_PORT' (must be numeric). Skipping local mosquitto start."
        docker exec "$CONTAINER" bash -lc "echo 'mosquitto skipped: invalid MQTT_PORT=$MQTT_PORT' > /tmp/mosquitto.log"
    elif docker exec "$CONTAINER" bash -lc "python3 - <<'PY'
import socket
import sys
port = int('${MQTT_PORT}')

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(('0.0.0.0', port))
except OSError:
    sys.exit(0)  # in use
else:
    sys.exit(1)  # free
finally:
    s.close()
PY
"; then
        echo "  Port ${MQTT_PORT} already in use (host network mode). Skipping local mosquitto start."
        docker exec "$CONTAINER" bash -lc "echo 'mosquitto skipped: port ${MQTT_PORT} already in use' > /tmp/mosquitto.log"
    else
        echo "  Port ${MQTT_PORT} is free. Starting local mosquitto."
        docker exec -d "$CONTAINER" bash -c "mosquitto -c /dev/null -p ${MQTT_PORT} 2>&1 | tee /tmp/mosquitto.log"
    fi
    sleep 1

    # Wait for YOLO model to load
    echo "▸ Waiting for YOLO model to load..."
    sleep 8

    # Optional rosbag playback during start.
    if [[ "$start_play_bag_for_start" == "true" ]]; then
        echo "▸ Playing rosbag (rate=$BAG_RATE, start-offset=${BAG_START}s)..."
        local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
        if [ "$BAG_START" != "0" ]; then
            bag_cmd="$bag_cmd --start-offset $BAG_START"
        fi
        docker exec -d "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"
    else
        echo "▸ Rosbag playback disabled for mode=${mode}."
        docker exec "$CONTAINER" bash -lc "echo 'bag playback skipped (mode=${mode})' > /tmp/bag.log"
    fi

    echo "✓ Pipeline running. Use './run.sh logs' to monitor."
}

cmd_stop() {
    echo "▸ Stopping..."
    _kill_inside "ros2.bag.play"
    _kill_inside "ugv_perception.launch.py"
    _kill_inside "geojson_bridge"
    _kill_inside "ugv_node"
    _kill_inside "mosquitto"
    sleep 1
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "✓ Stopped."
}

cmd_restart() {
    local mode="${1:-replay}"
    cmd_stop
    sleep 1
    cmd_start "$mode"
}

cmd_test() {
    _ensure_running
    _resolve_active_topics
    local rgb_topic
    local depth_topic
    local camera_info_topic
    local depth_camera_info_topic

    rgb_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node rgb_image_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    depth_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node depth_image_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    camera_info_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node camera_info_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    depth_camera_info_topic="$ACTIVE_DEPTH_CAMERA_INFO_TOPIC"

    rgb_topic="${rgb_topic:-$ACTIVE_RGB_TOPIC}"
    depth_topic="${depth_topic:-$ACTIVE_DEPTH_TOPIC}"
    camera_info_topic="${camera_info_topic:-$ACTIVE_CAMERA_INFO_TOPIC}"
    depth_camera_info_topic="${depth_camera_info_topic:-$ACTIVE_DEPTH_CAMERA_INFO_TOPIC}"

    # integration_test.py can start its own rosbag playback; ensure there is
    # no pre-existing player to avoid duplicate /clock/timestamp streams.
    _kill_inside "ros2.bag.play"
    sleep 1

    echo "▸ Running integration test (timeout=${TIMEOUT}s)..."
    docker exec "$CONTAINER" bash -c "$ROS_ENV && python3 \
        /ws/src/triffid_ugv_perception/test/integration_test.py \
        --no-launch --timeout $TIMEOUT \
        --rgb-topic '$rgb_topic' \
        --depth-topic '$depth_topic' \
        --camera-info-topic '$camera_info_topic' \
        --depth-camera-info-topic '$depth_camera_info_topic'"
}

cmd_unit() {
    echo "▸ Running unit tests..."
    cd src/triffid_ugv_perception
    python3 -m pytest test/test_unit.py -v --tb=short
}

cmd_sample() {
    _ensure_running
    _resolve_active_topics
    local wait="${1:-15}"
    local rgb_topic
    local mqtt_host
    local mqtt_port
    local mqtt_topic
    rgb_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node rgb_image_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    rgb_topic="${rgb_topic:-$ACTIVE_RGB_TOPIC}"

    mqtt_host="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /geojson_bridge mqtt_host 2>/dev/null | awk '/value/{print \$NF}'" || true)"
    mqtt_port="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /geojson_bridge mqtt_port 2>/dev/null | awk '/value/{print \$NF}'" || true)"
    mqtt_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /geojson_bridge mqtt_topic 2>/dev/null | awk '/value/{print \$NF}'" || true)"

    mqtt_host="${mqtt_host:-$MQTT_HOST}"
    mqtt_port="${mqtt_port:-$MQTT_PORT}"
    mqtt_topic="${mqtt_topic:-$MQTT_TOPIC}"

    echo "  Sampling MQTT trace from: ${mqtt_host}:${mqtt_port} topic=${mqtt_topic}"

    if [[ "$SAMPLE_PLAY_BAG" == "true" ]]; then
        _kill_inside "ros2.bag.play"
        sleep 1
        local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
        if [ "$BAG_START" != "0" ]; then
            bag_cmd="$bag_cmd --start-offset $BAG_START"
        fi
        echo "▸ Replaying rosbag for sampling (rate=$BAG_RATE, start-offset=${BAG_START}s)..."
        docker exec -d "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"
        sleep 1
    fi

    echo "▸ Collecting samples (timeout=${wait}s)..."
    docker exec "$CONTAINER" bash -c "$ROS_ENV && python3 \
        /ws/src/triffid_ugv_perception/scripts/collect_samples.py \
        --outdir /ws/samples --timeout $wait \
        --rgb-topic '$rgb_topic' \
        --mqtt-host '$mqtt_host' \
        --mqtt-port '$mqtt_port' \
        --mqtt-topic '$mqtt_topic'"

    if [[ "$SAMPLE_PLAY_BAG" == "true" ]]; then
        _kill_inside "ros2.bag.play"
    fi
    echo "✓ Samples saved to ./samples/"
}

cmd_streams() {
    _ensure_running
    local wait="${1:-12}"
    echo "▸ Validating live RealSense streams (timeout=${wait}s)..."
    echo "  Saving debug snapshots to: ${STREAMS_DEBUG_DIR}"
    docker exec "$CONTAINER" bash -c "$ROS_ENV && python3 \
        /ws/src/triffid_ugv_perception/scripts/validate_realsense_streams.py \
        --timeout $wait \
        --save-debug-dir '${STREAMS_DEBUG_DIR}'"
}

cmd_record() {
    _ensure_running
    _resolve_active_topics

    local bag_out="/ws/output_rosbag"
    local output_topics="\
        /ugv/detections/front/detections_3d \
        /ugv/detections/front/segmentation \
        /ugv/detections/front/geojson"
    local rgb_topic
    local depth_topic
    local camera_info_topic
    local depth_camera_info_topic
    local color_meta_topic=""
    local depth_meta_topic=""
    local extrinsics_topic=""

    rgb_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node rgb_image_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    depth_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node depth_image_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    camera_info_topic="$(docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 param get /ugv_perception_node camera_info_topic 2>/dev/null | awk -F': ' '/string value:/{print \$2}'" || true)"
    depth_camera_info_topic="$ACTIVE_DEPTH_CAMERA_INFO_TOPIC"

    rgb_topic="${rgb_topic:-$ACTIVE_RGB_TOPIC}"
    depth_topic="${depth_topic:-$ACTIVE_DEPTH_TOPIC}"
    camera_info_topic="${camera_info_topic:-$ACTIVE_CAMERA_INFO_TOPIC}"
    depth_camera_info_topic="${depth_camera_info_topic:-$ACTIVE_DEPTH_CAMERA_INFO_TOPIC}"

    if [[ "$rgb_topic" == */image_raw ]]; then
        color_meta_topic="${rgb_topic%/image_raw}/metadata"
    fi
    if [[ "$depth_topic" == */image_rect_raw ]]; then
        depth_meta_topic="${depth_topic%/image_rect_raw}/metadata"
    fi
    if [[ "$depth_topic" == */depth/* ]]; then
        extrinsics_topic="${depth_topic%%/depth/*}/extrinsics/depth_to_color"
    fi

    local input_topics="\
        $rgb_topic \
        $camera_info_topic \
        $depth_topic \
        $depth_camera_info_topic \
        $color_meta_topic \
        $depth_meta_topic \
        $extrinsics_topic \
        /fix \
        /dog_odom \
        /tf \
        /tf_static"

    # Clean previous recording
    docker exec "$CONTAINER" bash -c "rm -rf $bag_out/*" 2>/dev/null || true

    # Start recorder in background — output + input topics
    echo "▸ Recording output + input topics to ./output_rosbag/ ..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 bag record \
        $output_topics $input_topics \
        -o $bag_out/recording \
        --max-cache-size 200000000 \
        2>&1 | tee /tmp/record.log"
    sleep 2

    # Restart rosbag playback so it runs from the beginning
    _kill_inside "ros2.bag.play"
    sleep 1
    echo "▸ Playing input rosbag (rate=$BAG_RATE, start-offset=${BAG_START}s)..."
    local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
    if [ "$BAG_START" != "0" ]; then
        bag_cmd="$bag_cmd --start-offset $BAG_START"
    fi
    docker exec "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"

    # Bag playback finished — give nodes a moment to flush, then stop recorder
    echo "▸ Rosbag finished. Flushing..."
    sleep 3
    _kill_inside "ros2.bag.record"
    sleep 1

    echo "✓ Output rosbag saved to ./output_rosbag/recording/"
    echo ""
    docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 bag info $bag_out/recording 2>/dev/null" || true
}

cmd_logs() {
    _ensure_running
    echo "── perception launch (ugv_node + geojson_bridge) ──"
    docker exec "$CONTAINER" bash -c "tail -40 /tmp/perception_launch.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── bag player ──"
    docker exec "$CONTAINER" bash -c "tail -5 /tmp/bag.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── mosquitto ──"
    docker exec "$CONTAINER" bash -c "tail -5 /tmp/mosquitto.log 2>/dev/null" || echo "(no log)"
}

cmd_shell() {
    _ensure_running
    docker exec -it "$CONTAINER" bash -c "$ROS_ENV && exec bash"
}

cmd_status() {
    if _running; then
        echo "Container: RUNNING"
        echo ""
        echo "Processes:"
        docker exec "$CONTAINER" bash -c "ps aux | grep -E 'ugv_perception.launch.py|ugv_node|geojson|bag.play' | grep -v grep" || echo "  (none)"
        echo ""
        echo "ROS2 topics:"
        docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 topic list 2>/dev/null" || echo "  (unavailable)"
    else
        echo "Container: STOPPED"
    fi
}

cmd_help() {
    sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
}

# ── dispatch ─────────────────────────────────────────────────

case "${1:-help}" in
    build)   cmd_build ;;
    start)   cmd_start "${2:-replay}" ;;
    stop)    cmd_stop ;;
    restart) cmd_restart "${2:-replay}" ;;
    test)    cmd_test ;;
    unit)    cmd_unit ;;
    streams) cmd_streams "${2:-}" ;;
    sample)  cmd_sample "${2:-}" ;;
    record)  cmd_record ;;
    logs)    cmd_logs ;;
    shell)   cmd_shell ;;
    status)  cmd_status ;;
    help|-h|--help) cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
