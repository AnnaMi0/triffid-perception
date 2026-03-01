#!/bin/bash
set -e
cd /ws
source /opt/ros/humble/setup.bash
source install/setup.bash

echo "=== Launching perception nodes ==="
ros2 launch triffid_ugv_perception ugv_perception.launch.py &
LAUNCH_PID=$!
sleep 12

echo "=== Starting bag playback ==="
ros2 bag play /ws/rosbag --rate 1.0 \
  --qos-profile-overrides-path /ws/src/triffid_ugv_perception/config/bag_qos_overrides.yaml &
BAG_PID=$!

sleep 30

echo "=== Stopping ==="
kill $BAG_PID 2>/dev/null || true
sleep 2
kill $LAUNCH_PID 2>/dev/null || true
wait
echo "DONE"
