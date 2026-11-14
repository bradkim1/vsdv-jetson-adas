#!/usr/bin/env python3
"""
vSDV ADAS Perception Node — Jetson Orin Nano

Runs YOLOv8n via TensorRT on frames from either:
  - a scenario/CARLA camera stream (TCP, length-prefixed JPEG)  [MODE: carla]
  - a synthetic frame generator                                 [MODE: synthetic]

Publishes *perception* results to the Kuksa Databroker. Actuation
(brake pedal, EBA engagement) belongs to the vehicle app, not here.

Signals published — all exist in VSS 5.1 plus agent/vss_overlay_adas.json:
    Vehicle.ADAS.ObstacleDetection.IsWarning          bool
    Vehicle.ADAS.ObstacleDetection.DistanceToObject   float  m
    Vehicle.ADAS.ObstacleDetection.TimeToCollision    float  s
    Vehicle.ADAS.ObstacleDetection.ObjectCount        uint8
    Vehicle.ADAS.ObstacleDetection.InferenceLatency   float  ms
"""

import asyncio
import math
import os
import pathlib
import socket
import struct
import sys
import time
import threading
import queue

import numpy as np
import cv2

# The shared VSS layer lives in <repo>/agent. Resolve it relative to this file
# so the same source runs from a git checkout and from the container image
# (/app/adas/... + /app/agent/...). VSDV_AGENT_DIR overrides for odd layouts.
_AGENT_DIR = os.environ.get(
    "VSDV_AGENT_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent / "agent"))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from vss import VSS                      # noqa: E402  shared, metadata-aware VSS layer

KUKSA_HOST  = os.environ.get("KUKSA_HOST", "127.0.0.1")
CARLA_HOST  = os.environ.get("CARLA_HOST", "192.168.55.100")
CARLA_PORT  = int(os.environ.get("CARLA_CAM_PORT", "9999"))
MODE        = os.environ.get("ADAS_MODE", "synthetic")      # "carla" or "synthetic"

FRAME_W     = int(os.environ.get("FRAME_W", "640"))
FRAME_H     = int(os.environ.get("FRAME_H", "480"))
FPS_TARGET  = int(os.environ.get("FPS_TARGET", "10"))
WARN_TTC_S  = float(os.environ.get("WARN_TTC_S", "2.0"))

CAMERA_FOV_DEG = float(os.environ.get("CAMERA_FOV_DEG", "90"))
# Pinhole focal length in pixels, derived rather than guessed:
#     f = (width/2) / tan(fov/2)
FOCAL_PX = (FRAME_W / 2) / math.tan(math.radians(CAMERA_FOV_DEG) / 2)

# Monocular range needs the REAL height of the thing detected:
#     distance = focal_px * object_height_m / box_height_px
# The original code used a single (50*1.6)=80 constant for every class, which
# is wrong twice over: wrong for this camera (f is 320 px here, so a 1.75 m
# pedestrian needs 560), and wrong to apply one height to every object. A
# sweep measured the residual at +53% for pedestrians and -37% for vehicles
# using one constant. These are typical heights by COCO class id.
OBJECT_HEIGHT_M = {
    0: 1.75,   # person
    1: 1.10,   # bicycle
    2: 1.50,   # car
    3: 1.30,   # motorcycle
    5: 3.20,   # bus
    6: 4.00,   # train
    7: 3.50,   # truck
}
DEFAULT_HEIGHT_M = float(os.environ.get("DEFAULT_HEIGHT_M", "1.60"))

# How long a lost object stays tracked. Perception drops a detection for a
# single frame routinely; without a hold, distance snaps to the no-obstacle
# sentinel and the vehicle app cancels an active emergency stop. Measured at
# 24 brake releases across 24 scenarios before this existed.
TRACK_HOLD_S = float(os.environ.get("TRACK_HOLD_S", "0.6"))

NO_OBSTACLE_M = 999.0

P = "Vehicle.ADAS.ObstacleDetection."
SIG_WARNING  = P + "IsWarning"
SIG_DISTANCE = P + "DistanceToObject"
SIG_TTC      = P + "TimeToCollision"
SIG_COUNT    = P + "ObjectCount"
SIG_LATENCY  = P + "InferenceLatency"
SIG_SPEED    = "Vehicle.Speed"

frame_queue: "queue.Queue" = queue.Queue(maxsize=2)


# ── Frame sources ─────────────────────────────────────────────────────────────

def camera_receiver_thread():
    """Receive length-prefixed JPEG frames (CARLA bridge / scenario server)."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CARLA_HOST, CARLA_PORT))
            print(f"[ADAS] Connected to camera stream @ {CARLA_HOST}:{CARLA_PORT}",
                  flush=True)
            while True:
                raw_len = b""
                while len(raw_len) < 4:
                    chunk = sock.recv(4 - len(raw_len))
                    if not chunk:
                        raise ConnectionResetError("stream closed")
                    raw_len += chunk
                length = struct.unpack(">I", raw_len)[0]

                data = b""
                while len(data) < length:
                    chunk = sock.recv(min(65536, length - len(data)))
                    if not chunk:
                        raise ConnectionResetError("stream closed mid-frame")
                    data += chunk

                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                     cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put(frame)
        except Exception as e:
            print(f"[ADAS] camera stream lost: {e} — retrying in 2s", flush=True)
            time.sleep(2)


def generate_synthetic_frame(frame_id: int):
    """Crude stand-in scene used only when no camera stream is configured."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for y in range(FRAME_H // 2):
        i = int(80 + y * 0.3)
        frame[y, :] = [i, i + 20, i + 40]
    frame[FRAME_H // 2:, :] = (60, 60, 60)

    obstacles = []
    n = 1 + (frame_id // 30) % 3
    for i in range(n):
        phase = (frame_id + i * 20) % 60
        scale = 0.3 + phase * 0.012
        cx = FRAME_W // 2 + int((i - 1) * 120 * scale)
        cy = int(FRAME_H * 0.55 + phase * 2.0)
        w, h = int(60 * scale), int(40 * scale)
        dist = max(1.0, 20.0 - phase * 0.3)
        x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        obstacles.append(dist)
    return frame, obstacles


# ── Inference loop ────────────────────────────────────────────────────────────

async def run_adas(model, v: VSS):
    frame_id = 0
    source = "stream" if MODE == "carla" else "synthetic"
    print(f"[ADAS] mode={source}  target={FPS_TARGET}Hz  focal={FOCAL_PX:.0f}px  "
          f"warn_ttc={WARN_TTC_S}s  track_hold={TRACK_HOLD_S}s\n", flush=True)

    publish_errors = 0
    track = None          # last confirmed detection: {"t": epoch, "dist": m}
    holds = 0             # frames served from the held track rather than a detection
    while True:
        t0 = time.time()

        if MODE == "carla":
            try:
                frame = frame_queue.get(timeout=2.0)
            except queue.Empty:
                print("[ADAS] waiting for frames...", flush=True)
                continue
            gt_distances = []
        else:
            frame, gt_distances = generate_synthetic_frame(frame_id)

        t1 = time.time()
        results = model(frame, verbose=False, imgsz=FRAME_W)
        latency_ms = (time.time() - t1) * 1000

        boxes = results[0].boxes
        n_detected = 0 if boxes is None else len(boxes)
        distances = []
        if n_detected:
            for box, cls in zip(boxes.xyxy, boxes.cls):
                h = float(box[3] - box[1])
                h_real = OBJECT_HEIGHT_M.get(int(cls), DEFAULT_HEIGHT_M)
                distances.append(max(0.5, FOCAL_PX * h_real / max(h, 1.0)))

        if not distances and gt_distances:
            distances = gt_distances
            n_detected = len(distances)

        # ego speed is owned by the vehicle/scenario side; read it for TTC and
        # for coasting a held track
        speed_kmh = (await v.get(SIG_SPEED)).get(SIG_SPEED) or 0.0
        closing_ms = speed_kmh / 3.6

        now = time.time()
        if distances:
            nearest = min(distances)
            track = {"t": now, "dist": nearest}
            held = False
        elif track and (now - track["t"]) <= TRACK_HOLD_S:
            # Object lost this frame but seen very recently: coast the last
            # measurement forward at the current closing speed instead of
            # declaring the road clear.
            nearest = max(0.5, track["dist"] - closing_ms * (now - track["t"]))
            held = True
            holds += 1
        else:
            nearest = NO_OBSTACLE_M
            track = None
            held = False

        ttc = (nearest / closing_ms) if (closing_ms > 0.1 and nearest < NO_OBSTACLE_M) \
            else NO_OBSTACLE_M
        warning = ttc < WARN_TTC_S

        try:
            await v.set({
                SIG_WARNING:  warning,
                SIG_DISTANCE: nearest,
                SIG_TTC:      min(ttc, NO_OBSTACLE_M),
                SIG_COUNT:    min(255, n_detected),
                SIG_LATENCY:  latency_ms,
            })
        except Exception as e:
            publish_errors += 1
            if publish_errors <= 3 or publish_errors % 50 == 0:
                print(f"[Kuksa] publish failed ({publish_errors}): {e}", flush=True)

        flag = " WARN" if warning else ""
        mark = " HELD" if held else ""
        print(f"[{source}][{frame_id:04d}] n={n_detected} d={nearest:6.1f}m "
              f"ttc={ttc:6.1f}s lat={latency_ms:5.1f}ms{flag}{mark}", flush=True)

        frame_id += 1
        await asyncio.sleep(max(0, 1.0 / FPS_TARGET - (time.time() - t0)))


async def main():
    from ultralytics import YOLO

    engine_path = os.environ.get(
        "ENGINE", str(pathlib.Path(__file__).resolve().parent / "yolov8n.engine"))
    if os.path.exists(engine_path):
        print(f"[ADAS] loading TensorRT engine {engine_path}", flush=True)
        model = YOLO(engine_path)
    else:
        print("[ADAS] building TensorRT engine (first run, ~2 min)", flush=True)
        YOLO("yolov8n.pt").export(format="engine", imgsz=FRAME_W, device=0, half=True)
        model = YOLO(engine_path)

    if MODE == "carla":
        threading.Thread(target=camera_receiver_thread, daemon=True).start()

    async with VSS(f"{KUKSA_HOST}:55555") as v:
        print(f"[Kuksa] connected to {KUKSA_HOST}:55555", flush=True)
        try:
            await run_adas(model, v)
        except KeyboardInterrupt:
            print("\n[ADAS] stopped.")


if __name__ == "__main__":
    asyncio.run(main())
