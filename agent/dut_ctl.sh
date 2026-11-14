#!/bin/bash
# Start/stop the vSDV device-under-test nodes on the Jetson.
# Kept in a script so the pattern match never hits the invoking ssh command line.
cd /home/bk/vsdv-adas || exit 1
ADAS_PAT='adas/adas_inference'
AEB_PAT='vehicle_app/brake_app'

stop() {
  for pat in "$ADAS_PAT" "$AEB_PAT"; do
    pids=$(pgrep -f "python3 $pat")
    [ -n "$pids" ] && kill $pids 2>/dev/null
  done
  sleep 1
}

case "$1" in
  stop) stop; echo "stopped" ;;
  start)
    stop
    KUKSA_HOST=127.0.0.1 setsid nohup python3 vehicle_app/brake_app.py \
        >/tmp/aeb.log 2>&1 </dev/null &
    sleep 3
    KUKSA_HOST=127.0.0.1 ADAS_MODE=carla CARLA_HOST=192.168.55.100 \
      DIST_CONST="${DIST_CONST:-560}" setsid nohup python3 adas/adas_inference.py \
        >/tmp/adas.log 2>&1 </dev/null &
    sleep 2
    echo "started"
    ;;
  status)
    pgrep -af "python3 $ADAS_PAT|python3 $AEB_PAT"
    ;;
esac
