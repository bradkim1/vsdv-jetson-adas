#!/bin/bash
# Launch CARLA server headless on RTX 5090
# Then start the Kuksa bridge

CARLA_PATH="${CARLA_PATH:-/home/bk/carla}"

echo "[CARLA] Starting headless server..."

# Use existing display :1 or create virtual display
if [ -z "$DISPLAY" ]; then
    echo "[CARLA] Starting Xvfb virtual display..."
    Xvfb :8 -screen 0 1280x720x24 &
    export DISPLAY=:8
    sleep 2
fi

# Launch CARLA server
"$CARLA_PATH/CarlaUE4.sh" \
    -RenderOffScreen \
    -nosound \
    -quality-level=Low \
    -benchmark -fps=20 &

CARLA_PID=$!
echo "[CARLA] Server started (PID: $CARLA_PID), waiting 10s..."
sleep 10

# Start Kuksa bridge
echo "[Bridge] Starting CARLA → Kuksa bridge..."
CARLA_PATH="$CARLA_PATH" python3 "$(dirname "$0")/carla_kuksa_bridge.py"

# Cleanup on exit
kill $CARLA_PID 2>/dev/null
