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
#
# Environment variables:
#   BAG_RATE    Rosbag playback rate     (default: 1.0)
#   BAG_START   Seconds to skip in bag   (default: 0)
#   YOLO_IMGSZ  YOLO inference size      (default: 1280)
#   TIMEOUT     Integration test timeout (default: 30)
# =============================================================
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER=triffid_perception
IMAGE=triffid_perception:latest
BAG_RATE="${BAG_RATE:-1.0}"
BAG_START="${BAG_START:-0}"
YOLO_IMGSZ="${YOLO_IMGSZ:-1280}"
TIMEOUT="${TIMEOUT:-30}"

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

    # Launch ugv_node
    echo "▸ Launching ugv_node (YOLO imgsz=$YOLO_IMGSZ)..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 run triffid_ugv_perception ugv_node \
        --ros-args -p yolo_model:=/ws/best.pt -p yolo_imgsz:=$YOLO_IMGSZ 2>&1 | tee /tmp/ugv.log"

    # Start local MQTT broker (mosquitto)
    echo "▸ Starting MQTT broker (mosquitto)..."
    docker exec -d "$CONTAINER" bash -c "mosquitto -c /dev/null -p 1883 2>&1 | tee /tmp/mosquitto.log"
    sleep 1

    # Launch geojson_bridge
    echo "▸ Launching geojson_bridge..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 run triffid_ugv_perception geojson_bridge \
        2>&1 | tee /tmp/geojson.log"

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
    local topics="\
        /ugv/perception/front/detections_3d \
        /ugv/perception/front/segmentation \
        /triffid/front/geojson"

    # Clean previous recording
    docker exec "$CONTAINER" bash -c "rm -rf $bag_out/*" 2>/dev/null || true

    # Start recorder in background
    echo "▸ Recording output topics to ./output_rosbag/ ..."
    docker exec -d "$CONTAINER" bash -c "$ROS_ENV && ros2 bag record \
        $topics \
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
        docker exec "$CONTAINER" bash -c "ps aux | grep -E 'ugv_node|geojson|bag.play' | grep -v grep" || echo "  (none)"
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
    help|-h|--help) cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
