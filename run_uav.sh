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
#   ./run_uav.sh poll-api            # Poll FUTURISED API for new images
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
#
# FUTURISED API:
#   FUTURISED_MEDIA_API_KEY   Media Files API key (required for poll-api)
#   FUTURISED_ORG_ID          Organisation UUID (has default)
#   FUTURISED_TELEMETRY_TOKEN Telemetry API bearer token (optional)
#   API_CAMERA                Camera filter (default: Wide)
#   API_POLL_INTERVAL         Seconds between polls (default: 10)
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

_to_container_path() {
    # Map common host paths to mounted container paths.
    # Accepts either container paths (/app/...) or host-relative paths.
    local p="$1"
    if [[ -z "$p" ]]; then
        echo "/app/images"
        return 0
    fi
    if [[ "$p" == /app/* ]]; then
        echo "$p"
        return 0
    fi

    local clean="${p#./}"
    if [[ "$clean" == uav_images* ]]; then
        echo "/app/images${clean#uav_images}"
        return 0
    fi
    if [[ "$clean" == uav_samples* ]]; then
        echo "/app/samples${clean#uav_samples}"
        return 0
    fi
    if [[ "$clean" == uav_data* ]]; then
        echo "/app/uav_data${clean#uav_data}"
        return 0
    fi

    # Fallback: keep original input; caller may intentionally provide
    # a container-visible path.
    echo "$p"
}

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
    # Wait briefly until broker socket accepts connections.
    docker exec "$CONTAINER" bash -c "python3 - <<'PY'
import socket, time
host='127.0.0.1'
port=int('$MQTT_PORT')
ok=False
for _ in range(25):
    s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        ok=True
        break
    except Exception:
        time.sleep(0.1)
    finally:
        s.close()
print('ready' if ok else 'not-ready')
PY" >/tmp/uav_mqtt_wait.log 2>&1 || true
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

    local container_path
    container_path="$(_to_container_path "$img")"

    if ! docker exec "$CONTAINER" test -f "$container_path"; then
        echo "✗ Image not found inside container: $container_path"
        echo "  If this is a host path, place it under ./uav_images or ./uav_data"
        return 1
    fi

    echo "▸ Processing: $img → $container_path"
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --image $container_path --output /app/samples"
}

cmd_batch() {
    local dir="${1:-/app/images}"
    _ensure_running
    _start_mosquitto

    local container_dir
    container_dir="$(_to_container_path "$dir")"

    if ! docker exec "$CONTAINER" test -d "$container_dir"; then
        echo "✗ Directory not found inside container: $container_dir"
        echo "  Use one of: ./uav_images, ./uav_data, /app/images, /app/uav_data"
        return 1
    fi

    echo "▸ Processing all images in: $dir → $container_dir"
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --batch $container_dir --output /app/samples"
    echo "✓ Batch complete. Results in ./uav_samples/"
}

cmd_watch() {
    local dir="${1:-/app/images}"
    _ensure_running
    _start_mosquitto

    local container_dir
    container_dir="$(_to_container_path "$dir")"

    if ! docker exec "$CONTAINER" test -d "$container_dir"; then
        echo "✗ Directory not found inside container: $container_dir"
        echo "  Use one of: ./uav_images, ./uav_data, /app/images, /app/uav_data"
        return 1
    fi

    echo "▸ Watching for new images in: $dir → $container_dir"
    echo "  (Press Ctrl+C to stop)"
    docker exec -it "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --watch $container_dir"
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

cmd_poll_api() {
    if [[ -z "${FUTURISED_MEDIA_API_KEY:-}" ]]; then
        echo "✗ FUTURISED_MEDIA_API_KEY env var is required for poll-api mode."
        echo "  export FUTURISED_MEDIA_API_KEY='your-api-key'"
        return 1
    fi
    _ensure_running
    _start_mosquitto

    local api_args="--api-media-key $FUTURISED_MEDIA_API_KEY"
    api_args="$api_args --api-camera ${API_CAMERA:-Wide}"
    api_args="$api_args --api-poll-interval ${API_POLL_INTERVAL:-10}"
    api_args="$api_args --api-download-dir /app/images"
    [[ -n "${FUTURISED_ORG_ID:-}" ]] && api_args="$api_args --api-org-id $FUTURISED_ORG_ID"
    [[ -n "${FUTURISED_TELEMETRY_TOKEN:-}" ]] && api_args="$api_args --api-telemetry-token $FUTURISED_TELEMETRY_TOKEN"

    echo "▸ Polling FUTURISED API for new images (camera=${API_CAMERA:-Wide})..."
    echo "  (Press Ctrl+C to stop)"
    docker exec "$CONTAINER" bash -c \
        "PYTHONPATH=/app/src/triffid_uav_perception $PY_CMD $PY_ARGS --poll-api $api_args --output /app/samples"
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
    build)    cmd_build ;;
    process)  cmd_process "${2:-}" ;;
    batch)    cmd_batch "${2:-}" ;;
    watch)    cmd_watch "${2:-}" ;;
    poll-api) cmd_poll_api ;;
    stop)     cmd_stop ;;
    shell)    cmd_shell ;;
    test)     cmd_test ;;
    status)   cmd_status ;;
    help|*)   cmd_help ;;
esac
