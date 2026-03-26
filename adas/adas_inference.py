#!/usr/bin/env python3
"""
vSDV ADAS Inference Engine — Jetson Nano
Runs YOLOv8n via TensorRT on frames from either:
  - CARLA camera stream (TCP from host, port 9999)  [MODE: carla]
  - Synthetic frame generator                       [MODE: synthetic]

Publishes detection results to Kuksa Databroker as VSS signals.
"""

import asyncio
import os
import socket
import struct
import time
import threading
import numpy as np
import cv2
import grpc

KUKSA_HOST  = "127.0.0.1"
KUKSA_PORT  = 55555
CARLA_HOST  = os.environ.get("CARLA_HOST", "192.168.55.100")  # Host machine IP
CARLA_PORT  = int(os.environ.get("CARLA_CAM_PORT", "9999"))
MODE        = os.environ.get("ADAS_MODE", "synthetic")         # "carla" or "synthetic"

FRAME_W     = 640
FRAME_H     = 480
FPS_TARGET  = 10
COLLISION_THRESHOLD = 5.0  # metres

ADAS_SIGNALS = {
    "Vehicle.ADAS.ObstacleDetected":        "bool",
    "Vehicle.ADAS.ObstacleCount":           "uint32",
    "Vehicle.ADAS.NearestObstacleDistance": "float",
    "Vehicle.ADAS.CollisionWarning":        "bool",
    "Vehicle.ADAS.InferenceLatencyMs":      "float",
}

# Shared frame queue between receiver thread and inference loop
import queue
frame_queue = queue.Queue(maxsize=2)


# ── Frame sources ─────────────────────────────────────────────────────────────

def carla_receiver_thread():
    """Receives JPEG frames from CARLA bridge over TCP."""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CARLA_HOST, CARLA_PORT))
            print(f"[ADAS] Connected to CARLA camera @ {CARLA_HOST}:{CARLA_PORT}")
            while True:
                # Read 4-byte length prefix
                raw_len = b""
                while len(raw_len) < 4:
                    chunk = sock.recv(4 - len(raw_len))
                    if not chunk:
                        raise ConnectionResetError
                    raw_len += chunk
                length = struct.unpack(">I", raw_len)[0]

                # Read JPEG bytes
                data = b""
                while len(data) < length:
                    chunk = sock.recv(min(65536, length - len(data)))
                    if not chunk:
                        raise ConnectionResetError
                    data += chunk

                # Decode JPEG → numpy frame
                arr   = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put(frame)
        except Exception as e:
            print(f"[ADAS] CARLA stream lost: {e} — retrying in 3s")
            time.sleep(3)


def generate_synthetic_frame(frame_id: int):
    """Synthetic driving scene when CARLA is not connected."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    for y in range(FRAME_H // 2):
        i = int(80 + y * 0.3)
        frame[y, :] = [i, i + 20, i + 40]
    frame[FRAME_H // 2:, :] = (60, 60, 60)

    t = frame_id * 3
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
        cv2.putText(frame, f"{dist:.1f}m", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        obstacles.append(dist)
    return frame, obstacles


# ── Kuksa publisher ───────────────────────────────────────────────────────────

async def publish_to_kuksa(stub, detections: dict):
    from kuksa.val.v1 import val_pb2, types_pb2
    updates = []
    values = {
        "Vehicle.ADAS.ObstacleDetected":        detections["obstacle_detected"],
        "Vehicle.ADAS.ObstacleCount":           detections["obstacle_count"],
        "Vehicle.ADAS.NearestObstacleDistance": detections["nearest_distance"],
        "Vehicle.ADAS.CollisionWarning":        detections["collision_warning"],
        "Vehicle.ADAS.InferenceLatencyMs":      detections["latency_ms"],
    }
    for path, value in values.items():
        dp = types_pb2.Datapoint()
        sig_type = ADAS_SIGNALS[path]
        if sig_type == "bool":
            dp.bool = bool(value)
        elif sig_type == "uint32":
            dp.uint32 = int(value)
        elif sig_type == "float":
            dp.float = float(value)
        updates.append(val_pb2.EntryUpdate(
            entry=types_pb2.DataEntry(path=path, value=dp),
            fields=[types_pb2.Field.FIELD_VALUE],
        ))
    try:
        await stub.Set(val_pb2.SetRequest(updates=updates))
    except Exception as e:
        print(f"[Kuksa] Error: {e}")


# ── Inference loop ────────────────────────────────────────────────────────────

async def run_adas(model, stub):
    frame_id = 0
    source   = "CARLA" if MODE == "carla" else "Synthetic"
    print(f"[ADAS] Mode: {source} | Target: {FPS_TARGET} FPS | "
          f"Collision threshold: {COLLISION_THRESHOLD}m\n")

    while True:
        t0 = time.time()

        # Get frame
        if MODE == "carla":
            try:
                frame = frame_queue.get(timeout=2.0)
                gt_distances = []
            except queue.Empty:
                print("[ADAS] Waiting for CARLA frame...")
                continue
        else:
            frame, gt_distances = generate_synthetic_frame(frame_id)

        # TensorRT inference
        t1 = time.time()
        results = model(frame, verbose=False, imgsz=FRAME_W)
        latency_ms = (time.time() - t1) * 1000

        # Parse detections
        boxes = results[0].boxes
        n_detected = len(boxes) if boxes is not None else 0
        distances = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy:
                h = float(box[3] - box[1])
                distances.append(max(1.0, (50 * 1.6) / max(h, 1)))

        if not distances and gt_distances:
            distances = gt_distances
            n_detected = len(distances)

        nearest  = min(distances) if distances else 99.0
        collision = nearest < COLLISION_THRESHOLD

        await publish_to_kuksa(stub, {
            "obstacle_detected": n_detected > 0,
            "obstacle_count":    n_detected,
            "nearest_distance":  nearest,
            "collision_warning": collision,
            "latency_ms":        latency_ms,
        })

        warn = " ⚠ COLLISION" if collision else ""
        print(f"[{source}][{frame_id:04d}] n={n_detected}  "
              f"dist={nearest:.1f}m  lat={latency_ms:.1f}ms{warn}")

        frame_id += 1
        elapsed = time.time() - t0
        await asyncio.sleep(max(0, 1.0 / FPS_TARGET - elapsed))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    from ultralytics import YOLO

    engine_path = "yolov8n.engine"
    if os.path.exists(engine_path):
        print("[ADAS] Loading TensorRT engine...")
        model = YOLO(engine_path)
    else:
        print("[ADAS] Building TensorRT engine (first run ~2 min)...")
        m = YOLO("yolov8n.pt")
        m.export(format="engine", imgsz=FRAME_W, device=0, half=True)
        model = YOLO(engine_path)

    if MODE == "carla":
        t = threading.Thread(target=carla_receiver_thread, daemon=True)
        t.start()
        print("[ADAS] CARLA receiver thread started")

    channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
    from kuksa.val.v1 import val_pb2_grpc
    stub = val_pb2_grpc.VALStub(channel)

    try:
        await run_adas(model, stub)
    except KeyboardInterrupt:
        print("\n[ADAS] Stopped.")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
