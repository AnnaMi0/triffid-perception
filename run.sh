#!/usr/bin/env bash
# =============================================================
# TRIFFID Perception – All-in-one runner
# =============================================================
#
# Usage:
#   ./run.sh build          # Build Docker image + colcon workspace
#   ./run.sh start          # Start container + launch nodes + play bag
#   ./run.sh stop           # Stop everything
#   ./run.sh restart        # stop + start
#   ./run.sh test           # Run integration test (inside container)
#   ./run.sh unit           # Run unit tests (host, no Docker needed)
#   ./run.sh sample [SEC]   # Collect output samples (default: wait 15s)
#   ./run.sh record         # Record output topics to output_rosbag/
#   ./run.sh logs           # Tail node logs
#   ./run.sh shell          # Open bash inside the container
#   ./run.sh status         # Show running processes + topic list
#   ./run.sh camtest        # Grab one frame from RealSense 435i and save to ./samples/
#   ./run.sh bridge-test    # Smoke-test bridge: publish mock UGV+UAV data, print merged output
#   ./run.sh full-test [SEC]# Real-data bridge test: replay rosbag + UAV pipeline → watch bridge (default: 30s)
#
# Environment variables:
#   BAG_RATE         Rosbag playback rate          (default: 1.0)
#   BAG_START        Seconds to skip in bag        (default: 0)
#   YOLO_IMGSZ       YOLO inference size            (default: 1280)
#   TIMEOUT          Integration test timeout       (default: 30)
#   MQTT_PORT        MQTT broker port               (default: 1883)
#   TELESTO_BASE_URL      TELESTO Map Manager endpoint   (default: built-in)
#   TELESTO_OBSERVER_URL  TELESTO Observer endpoint      (default: built-in)
#   BRIDGE_SAVE_SAMPLES   Set to 1 to save merged GeoJSON to ./samples/ instead of uploading
#   CAMTEST_RGB_TOPIC     Color topic for camtest        (default: 435i topic)
#   CAMTEST_DEPTH_TOPIC   Depth topic for camtest        (default: 435i topic)
#                    Change if port 1883 is already occupied on this machine.
#   RGB_TOPIC        Color image topic override     (default: node built-in)
#   DEPTH_TOPIC      Depth image topic override     (default: node built-in)
#   CAMERA_INFO_TOPIC Camera info topic override   (default: node built-in)
#                    Leave all three empty for rosbag mode.
#                    For live RealSense 435i set to:
#                      RGB_TOPIC=/camera_front_435i/realsense_front_435i/color/image_raw
#                      DEPTH_TOPIC=/camera_front_435i/realsense_front_435i/depth/image_rect_raw
#                      CAMERA_INFO_TOPIC=/camera_front_435i/realsense_front_435i/color/camera_info
# =============================================================
set -euo pipefail
cd "$(dirname "$0")"

# Load .env if present (never committed — see .env.example)
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

CONTAINER=triffid_perception
IMAGE=triffid_perception:latest
BAG_RATE="${BAG_RATE:-1.0}"
BAG_START="${BAG_START:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
TIMEOUT="${TIMEOUT:-30}"
MQTT_PORT="${MQTT_PORT:-1883}"
TELESTO_BASE_URL="${TELESTO_BASE_URL:-}"
TELESTO_OBSERVER_URL="${TELESTO_OBSERVER_URL:-}"
# Replay/default topics (when env overrides are not provided)
DEFAULT_REPLAY_RGB_TOPIC="/camera_front/raw_image"
DEFAULT_REPLAY_DEPTH_TOPIC="/camera_front/realsense_front/depth/image_rect_raw"
DEFAULT_REPLAY_CAMERA_INFO_TOPIC="/camera_front/camera_info"
RGB_TOPIC="${RGB_TOPIC:-$DEFAULT_REPLAY_RGB_TOPIC}"
DEPTH_TOPIC="${DEPTH_TOPIC:-$DEFAULT_REPLAY_DEPTH_TOPIC}"
CAMERA_INFO_TOPIC="${CAMERA_INFO_TOPIC:-$DEFAULT_REPLAY_CAMERA_INFO_TOPIC}"

# Shared ROS env sourcing command (used inside docker exec)
ROS_ENV="source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && export ROS_DOMAIN_ID=42 && export CYCLONEDDS_URI=file:///ws/cyclonedds.xml"
ROS_BUILD_ENV="source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=42 && export CYCLONEDDS_URI=file:///ws/cyclonedds.xml"

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
    echo "▸ Starting $CONTAINER..."

    # Remove stale container if exists
    if docker ps -a -q -f name="$CONTAINER" 2>/dev/null | grep -q .; then
        docker rm -f "$CONTAINER" 2>/dev/null || true
        sleep 1
    fi
    docker compose up -d
    sleep 2

    # Build workspace (symlink-install so source edits take effect)
    echo "▸ Building workspace..."
    docker exec "$CONTAINER" bash -c "cd /ws && rm -rf build/* install/* log/* && $ROS_BUILD_ENV && colcon build --symlink-install"

    # Build optional camera topic args (empty = use node defaults, which match rosbag)
    _CAM_ARGS=""
    [ -n "$RGB_TOPIC" ]          && _CAM_ARGS="$_CAM_ARGS -p rgb_image_topic:=$RGB_TOPIC"
    [ -n "$DEPTH_TOPIC" ]        && _CAM_ARGS="$_CAM_ARGS -p depth_image_topic:=$DEPTH_TOPIC"
    [ -n "$CAMERA_INFO_TOPIC" ]  && _CAM_ARGS="$_CAM_ARGS -p camera_info_topic:=$CAMERA_INFO_TOPIC"

    # Launch ugv_node
    echo "▸ Launching ugv_node (YOLO imgsz=$YOLO_IMGSZ)..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 run triffid_ugv_perception ugv_node \
        --ros-args -p yolo_model:=/ws/best.pt -p yolo_imgsz:=$YOLO_IMGSZ $_CAM_ARGS 2>&1 | tee /tmp/ugv.log"

    # Start local MQTT broker (mosquitto)
    echo "▸ Starting MQTT broker (mosquitto on port $MQTT_PORT)..."
    docker exec -d "$CONTAINER" bash -c "mosquitto -c /dev/null -p $MQTT_PORT 2>&1 | tee /tmp/mosquitto.log"
    sleep 1

    # Launch geojson_bridge
    echo "▸ Launching geojson_bridge..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 run triffid_ugv_perception geojson_bridge \
        --ros-args -p mqtt_port:=$MQTT_PORT 2>&1 | tee /tmp/geojson.log"

    # Launch MQTT→TELESTO bridge
    echo "▸ Launching TELESTO bridge..."
    local _bridge_args="--mqtt-host localhost --mqtt-port $MQTT_PORT"
    [ -n "$TELESTO_BASE_URL" ]     && _bridge_args="$_bridge_args --telesto-base $TELESTO_BASE_URL"
    [ -n "$TELESTO_OBSERVER_URL" ] && _bridge_args="$_bridge_args --telesto-observer $TELESTO_OBSERVER_URL"
    [ "${BRIDGE_SAVE_SAMPLES:-}" = "1" ] && _bridge_args="$_bridge_args --samples-dir /ws/samples"
    docker exec -d "$CONTAINER" bash -c \
        "PYTHONPATH=/ws/src python3 -m triffid_telesto.bridge $_bridge_args 2>&1 | tee /tmp/bridge.log"

    # Wait for YOLO model to load
    echo "▸ Waiting for YOLO model to load..."
    sleep 8

    # Play rosbag
    echo "▸ Playing rosbag (rate=$BAG_RATE, start-offset=${BAG_START}s)..."
    local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
    if [ "$BAG_START" != "0" ]; then
        bag_cmd="$bag_cmd --start-offset $BAG_START"
    fi
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"

    echo "✓ Pipeline running. Use './run.sh logs' to monitor."
}

cmd_stop() {
    echo "▸ Stopping..."
    _kill_inside "ros2.bag.play"
    _kill_inside "triffid_telesto.bridge"
    _kill_inside "geojson_bridge"
    _kill_inside "ugv_node"
    _kill_inside "mosquitto"
    sleep 1
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "✓ Stopped."
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_start
}

cmd_test() {
    _ensure_running

    # Ensure fresh replay input for each test run.
    _kill_inside "ros2.bag.play"
    sleep 1
    local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
    if [ "$BAG_START" != "0" ]; then
        bag_cmd="$bag_cmd --start-offset $BAG_START"
    fi
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"

    echo "▸ Running integration test (timeout=${TIMEOUT}s)..."
    docker exec "$CONTAINER" bash -c "$ROS_ENV && python3 \
        /ws/src/triffid_ugv_perception/test/integration_test.py \
        --no-launch --no-bag --timeout $TIMEOUT"
}

cmd_unit() {
    echo "▸ Running unit tests..."
    cd src/triffid_ugv_perception
    python3 -m pytest test/test_unit.py -v --tb=short
}

cmd_sample() {
    _ensure_running
    local wait="${1:-15}"
    echo "▸ Collecting samples (timeout=${wait}s)..."
    docker exec "$CONTAINER" bash -c "$ROS_ENV && python3 \
        /ws/src/triffid_ugv_perception/scripts/collect_samples.py \
        --outdir /ws/samples --timeout $wait"
    echo "✓ Samples saved to ./samples/"
}

cmd_record() {
    _ensure_running

    local bag_out="/ws/output_rosbag"
    local output_topics="\
        /ugv/detections/front/detections_3d \
        /ugv/detections/front/segmentation \
        /ugv/detections/front/geojson"
    local input_topics="\
        /camera_front/raw_image \
        /camera_front/camera_info \
        /camera_front/realsense_front/depth/image_rect_raw \
        /camera_front/realsense_front/depth/camera_info \
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
    echo "── ugv_node ──"
    docker exec "$CONTAINER" bash -c "tail -20 /tmp/ugv.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── geojson_bridge ──"
    docker exec "$CONTAINER" bash -c "tail -10 /tmp/geojson.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── bag player ──"
    docker exec "$CONTAINER" bash -c "tail -5 /tmp/bag.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── mosquitto ──"
    docker exec "$CONTAINER" bash -c "tail -5 /tmp/mosquitto.log 2>/dev/null" || echo "(no log)"
    echo ""
    echo "── TELESTO bridge ──"
    docker exec "$CONTAINER" bash -c "tail -10 /tmp/bridge.log 2>/dev/null" || echo "(no log)"
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
        docker exec "$CONTAINER" bash -c "ps aux | grep -E 'ugv_node|geojson|bag.play|triffid_telesto' | grep -v grep" || echo "  (none)"
        echo ""
        echo "ROS2 topics:"
        docker exec "$CONTAINER" bash -c "$ROS_ENV && ros2 topic list 2>/dev/null" || echo "  (unavailable)"
    else
        echo "Container: STOPPED"
    fi
}

cmd_camtest() {
    _ensure_running

    # Always default to the live 435i topics — these are separate from RGB_TOPIC/DEPTH_TOPIC
    # which control ugv_node subscriptions and may be set to rosbag replay topics.
    local rgb="${CAMTEST_RGB_TOPIC:-/camera_front_435i/realsense_front_435i/color/image_raw}"
    local depth="${CAMTEST_DEPTH_TOPIC:-/camera_front_435i/realsense_front_435i/depth/image_rect_raw}"

    echo "▸ Camera sanity test"
    echo "  Color topic: $rgb"
    echo "  Depth topic: $depth"
    docker exec "$CONTAINER" bash -c \
        "$ROS_ENV && python3 /ws/src/triffid_ugv_perception/scripts/camtest.py \
            --rgb-topic '$rgb' \
            --depth-topic '$depth' \
            --outdir /ws/samples"
    echo "✓ Saved: ./samples/camtest_color.png  ./samples/camtest_depth.png"
}

cmd_bridge_test() {
    _ensure_running
    echo "▸ Bridge smoke test — publishing mock UGV+UAV data to MQTT and printing merged output..."
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/ws/src python3 -u /ws/src/triffid_telesto/smoke_test.py \
            --mqtt-host localhost --mqtt-port $MQTT_PORT"
}

cmd_full_test() {
    local wait="${1:-30}"
    _ensure_running

    # Restart rosbag from the beginning so the UGV pipeline produces detections immediately.
    echo "▸ Restarting rosbag playback from start..."
    _kill_inside "ros2.bag.play"
    sleep 1
    local bag_cmd="ros2 bag play /ws/rosbag --rate $BAG_RATE --clock"
    [ "$BAG_START" != "0" ] && bag_cmd="$bag_cmd --start-offset $BAG_START"
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && $bag_cmd 2>&1 | tee /tmp/bag.log"

    # Start UAV pipeline — prefer FUTURISED API if key is set, else batch over local images.
    local _uav_pid=""
    if [ -n "${FUTURISED_MEDIA_API_KEY:-}" ]; then
        echo "▸ Starting UAV pipeline (FUTURISED API poll mode)..."
        MQTT_HOST=localhost MQTT_PORT=$MQTT_PORT ./run_uav.sh poll-api &
        _uav_pid=$!
    elif ls uav_images/*.jpg uav_images/*.JPG 2>/dev/null | head -1 > /dev/null; then
        echo "▸ Starting UAV pipeline (batch mode from ./uav_images/)..."
        MQTT_HOST=localhost MQTT_PORT=$MQTT_PORT ./run_uav.sh batch uav_images &
        _uav_pid=$!
    else
        echo "  (no UAV source: set FUTURISED_MEDIA_API_KEY or add images to ./uav_images/)"
        echo "  UGV-only test proceeding."
    fi

    # Stream bridge log live for the requested duration.
    echo "▸ Streaming bridge output for ${wait}s (Ctrl+C to stop early)..."
    echo "  Each line shows: timestamp | features synced | created/updated/deleted"
    echo ""
    docker exec "$CONTAINER" bash -c "tail -f /tmp/bridge.log 2>/dev/null" &
    local _tail_pid=$!

    # Cleanup on Ctrl+C or normal exit.
    _cleanup_full_test() {
        kill "$_tail_pid" 2>/dev/null || true
        [ -n "$_uav_pid" ] && kill "$_uav_pid" 2>/dev/null || true
    }
    trap _cleanup_full_test INT TERM

    sleep "$wait"
    _cleanup_full_test
    trap - INT TERM

    echo ""
    echo "── Summary: last 20 bridge log lines ──────────────────────────"
    docker exec "$CONTAINER" bash -c "tail -20 /tmp/bridge.log 2>/dev/null" || echo "(no log)"
    echo "────────────────────────────────────────────────────────────────"
}

cmd_help() {
    sed -n '2,/^set /{ /^#/s/^# \?//p }' "$0"
}

# ── dispatch ─────────────────────────────────────────────────

case "${1:-help}" in
    build)   cmd_build ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    test)    cmd_test ;;
    unit)    cmd_unit ;;
    sample)  cmd_sample "${2:-}" ;;
    record)  cmd_record ;;
    logs)    cmd_logs ;;
    shell)   cmd_shell ;;
    status)  cmd_status ;;
    camtest) cmd_camtest ;;
    bridge-test) cmd_bridge_test ;;
    full-test)   cmd_full_test "${2:-}" ;;
    help|-h|--help) cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
