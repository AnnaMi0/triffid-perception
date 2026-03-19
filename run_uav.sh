#!/usr/bin/env bash
# =============================================================
# TRIFFID UAV Perception – All-in-one runner
# =============================================================
#
# Usage:
#   ./run_uav.sh build              # Build Docker image
#   ./run_uav.sh process <image>    # Process a single image
#   ./run_uav.sh batch [dir]        # Process all images in directory
#   ./run_uav.sh watch [dir]        # Watch directory for new images
#   ./run_uav.sh stop               # Stop container
#   ./run_uav.sh shell              # Open bash inside the container
#   ./run_uav.sh test               # Run unit tests
#   ./run_uav.sh status             # Show container status
#   ./run_uav.sh help               # Show this help
#
# Environment variables:
#   MODEL       YOLO model path inside container  (default: /app/best.pt)
#   CONFIDENCE  Detection threshold                (default: 0.35)
#   MQTT_HOST   MQTT broker host                   (default: localhost)
#   MQTT_PORT   MQTT broker port                   (default: 1883)
#   MQTT_TOPIC  MQTT topic                         (default: triffid/uav/geojson)
#   IMGSZ       YOLO input size                    (default: 1280)
# =============================================================
set -euo pipefail
cd "$(dirname "$0")"

CONTAINER=triffid_uav_perception
COMPOSE_FILE=docker-compose.uav.yml
MODEL="${MODEL:-/app/best.pt}"
CONFIDENCE="${CONFIDENCE:-0.35}"
MQTT_HOST="${MQTT_HOST:-localhost}"
MQTT_PORT="${MQTT_PORT:-1883}"
MQTT_TOPIC="${MQTT_TOPIC:-triffid/uav/geojson}"
IMGSZ="${IMGSZ:-1280}"

# Common Python command prefix
PY_CMD="python -m triffid_uav_perception.uav_node"
PY_ARGS="--model $MODEL --confidence $CONFIDENCE --mqtt-host $MQTT_HOST --mqtt-port $MQTT_PORT --mqtt-topic $MQTT_TOPIC --imgsz $IMGSZ"

# ── helpers ──────────────────────────────────────────────────

_running() { docker ps -q -f name="$CONTAINER" 2>/dev/null | grep -q .; }

_ensure_running() {
    if ! _running; then
        echo "▸ Starting container..."
        docker compose -f "$COMPOSE_FILE" up -d
        sleep 2
    fi
}

_start_mosquitto() {
    echo "▸ Starting MQTT broker (mosquitto)..."
    docker exec "$CONTAINER" bash -c "mosquitto -c /dev/null -p $MQTT_PORT &" 2>/dev/null || true
    sleep 1
}

# ── commands ─────────────────────────────────────────────────

cmd_build() {
    echo "▸ Building UAV Docker image..."
    docker compose -f "$COMPOSE_FILE" build
    echo "✓ Build complete."
}

cmd_process() {
    local img="${1:?Usage: ./run_uav.sh process <image_path>}"
    _ensure_running
    _start_mosquitto

    # Determine if the image path is absolute or relative
    local container_path
    if [[ "$img" == /* ]]; then
        # Absolute path — check if it's in uav_images/
        container_path="/app/images/$(basename "$img")"
    else
        container_path="/app/images/$img"
    fi

    echo "▸ Processing: $img → $container_path"
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --image $container_path --output /app/samples"
}

cmd_batch() {
    local dir="${1:-/app/images}"
    _ensure_running
    _start_mosquitto

    echo "▸ Processing all images in: $dir"
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --batch $dir --output /app/samples"
    echo "✓ Batch complete. Results in ./uav_samples/"
}

cmd_watch() {
    local dir="${1:-/app/images}"
    _ensure_running
    _start_mosquitto

    echo "▸ Watching for new images in: $dir"
    echo "  (Press Ctrl+C to stop)"
    docker exec -it "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --watch $dir"
}

cmd_stop() {
    echo "▸ Stopping UAV container..."
    docker rm -f "$CONTAINER" 2>/dev/null || true
    echo "✓ Stopped."
}

cmd_shell() {
    _ensure_running
    echo "▸ Opening shell in $CONTAINER..."
    docker exec -it "$CONTAINER" bash -c "PYTHONPATH=/app/src/triffid_uav_perception exec bash"
}

cmd_test() {
    _ensure_running
    echo "▸ Running unit tests..."
    docker exec "$CONTAINER" bash -c \
        "cd /app && PYTHONPATH=/app/src/triffid_uav_perception python -m pytest src/triffid_uav_perception/test/ -v"
}

cmd_status() {
    if _running; then
        echo "✓ Container $CONTAINER is running."
        docker exec "$CONTAINER" bash -c "ps aux | grep -E 'python|mosquitto' | grep -v grep" 2>/dev/null || true
    else
        echo "✗ Container $CONTAINER is not running."
    fi
}

cmd_help() {
    head -21 "$0" | tail -20
}

# ── dispatch ─────────────────────────────────────────────────

case "${1:-help}" in
    build)   cmd_build ;;
    process) cmd_process "${2:-}" ;;
    batch)   cmd_batch "${2:-}" ;;
    watch)   cmd_watch "${2:-}" ;;
    stop)    cmd_stop ;;
    shell)   cmd_shell ;;
    test)    cmd_test ;;
    status)  cmd_status ;;
    help|*)  cmd_help ;;
esac
