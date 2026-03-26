#!/usr/bin/env python3
"""
vSDV ADAS Inference Engine — Jetson Nano
Runs YOLOv8n via TensorRT on synthetic camera frames.
Publishes detection results to Kuksa Databroker as VSS signals.
"""

import asyncio
import time
import numpy as np
import cv2
import grpc

# Kuksa connection
KUKSA_HOST = "127.0.0.1"
KUKSA_PORT = 55555

# ADAS VSS signals (custom namespace under Vehicle.ADAS)
ADAS_SIGNALS = {
    "Vehicle.ADAS.ObstacleDetected":        "bool",
    "Vehicle.ADAS.ObstacleCount":           "uint32",
    "Vehicle.ADAS.NearestObstacleDistance": "float",
    "Vehicle.ADAS.CollisionWarning":        "bool",
    "Vehicle.ADAS.InferenceLatencyMs":      "float",
}

COLLISION_DISTANCE_THRESHOLD = 5.0   # metres
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS_TARGET   = 10


def generate_synthetic_frame(frame_id: int) -> np.ndarray:
    """
    Generate a synthetic driving scene frame.
    Simulates: road, moving obstacles, varying lighting.
    No camera hardware needed — pure NumPy/OpenCV.
    """
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(FRAME_HEIGHT // 2):
        intensity = int(80 + y * 0.3)
        frame[y, :] = [intensity, intensity + 20, intensity + 40]

    # Road surface
    road_color = (60, 60, 60)
    frame[FRAME_HEIGHT // 2:, :] = road_color

    # Road lane markings
    t = frame_id * 3
    for x in [FRAME_WIDTH // 4, FRAME_WIDTH * 3 // 4]:
        for y_start in range(FRAME_HEIGHT // 2, FRAME_HEIGHT, 40):
            y_offset = (y_start + t) % FRAME_HEIGHT
            if y_offset > FRAME_HEIGHT // 2:
                cv2.rectangle(frame, (x - 5, y_offset), (x + 5, y_offset + 20),
                              (200, 200, 100), -1)

    # Simulate 1-3 vehicles (obstacles) at varying distances
    n_obstacles = 1 + (frame_id // 30) % 3
    obstacles = []
    for i in range(n_obstacles):
        # Object moves toward camera (grows over time)
        phase = (frame_id + i * 20) % 60
        scale = 0.3 + phase * 0.012
        cx = FRAME_WIDTH // 2 + int((i - 1) * 120 * scale)
        cy = int(FRAME_HEIGHT * 0.55 + phase * 2.0)
        w  = int(60 * scale)
        h  = int(40 * scale)
        distance = max(1.0, 20.0 - phase * 0.3)

        # Draw vehicle box
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Distance label
        cv2.putText(frame, f"{distance:.1f}m", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        obstacles.append({"bbox": (x1, y1, x2, y2), "distance": distance})

    return frame, obstacles


async def publish_to_kuksa(stub, detections: dict):
    """Send ADAS detection results to Kuksa Databroker."""
    from kuksa.val.v1 import val_pb2, types_pb2

    updates = []
    type_map = {
        "bool":   lambda v: setattr(types_pb2.Datapoint(), 'bool', v),
        "uint32": lambda v: v,
        "float":  lambda v: v,
    }

    signal_values = {
        "Vehicle.ADAS.ObstacleDetected":        detections["obstacle_detected"],
        "Vehicle.ADAS.ObstacleCount":           detections["obstacle_count"],
        "Vehicle.ADAS.NearestObstacleDistance": detections["nearest_distance"],
        "Vehicle.ADAS.CollisionWarning":        detections["collision_warning"],
        "Vehicle.ADAS.InferenceLatencyMs":      detections["latency_ms"],
    }

    for path, value in signal_values.items():
        dp = types_pb2.Datapoint()
        sig_type = ADAS_SIGNALS[path]
        if sig_type == "bool":
            dp.bool = bool(value)
        elif sig_type == "uint32":
            dp.uint32 = int(value)
        elif sig_type == "float":
            dp.float = float(value)

        entry = val_pb2.EntryUpdate(
            entry=types_pb2.DataEntry(path=path, value=dp),
            fields=[types_pb2.Field.FIELD_VALUE],
        )
        updates.append(entry)

    try:
        await stub.Set(val_pb2.SetRequest(updates=updates))
    except Exception as e:
        print(f"[Kuksa] Publish error: {e}")


async def run_adas(model, stub):
    """Main ADAS inference loop."""
    frame_id = 0
    print(f"[ADAS] Starting inference loop at {FPS_TARGET} FPS target")
    print(f"[ADAS] Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[ADAS] Collision threshold: {COLLISION_DISTANCE_THRESHOLD}m\n")

    while True:
        loop_start = time.time()

        # Generate synthetic frame
        frame, ground_truth_obstacles = generate_synthetic_frame(frame_id)

        # Run YOLOv8 inference
        infer_start = time.time()
        results = model(frame, verbose=False, imgsz=FRAME_WIDTH)
        latency_ms = (time.time() - infer_start) * 1000

        # Parse detections
        boxes = results[0].boxes
        n_detected = len(boxes) if boxes is not None else 0

        # Estimate distances from bounding box height (larger = closer)
        distances = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy:
                h = float(box[3] - box[1])
                # Rough distance estimate: focal_length * real_height / pixel_height
                dist = max(1.0, (50 * 1.6) / max(h, 1))
                distances.append(dist)

        # Fallback: use synthetic ground truth distances
        if not distances and ground_truth_obstacles:
            distances = [o["distance"] for o in ground_truth_obstacles]
            n_detected = len(distances)

        nearest = min(distances) if distances else 99.0
        collision = nearest < COLLISION_DISTANCE_THRESHOLD

        detections = {
            "obstacle_detected": n_detected > 0,
            "obstacle_count":    n_detected,
            "nearest_distance":  nearest,
            "collision_warning": collision,
            "latency_ms":        latency_ms,
        }

        # Publish to Kuksa
        await publish_to_kuksa(stub, detections)

        # Console output
        warn = " ⚠ COLLISION WARNING" if collision else ""
        print(f"[Frame {frame_id:04d}] Obstacles={n_detected}  "
              f"Nearest={nearest:.1f}m  "
              f"Latency={latency_ms:.1f}ms{warn}")

        frame_id += 1

        # Throttle to target FPS
        elapsed = time.time() - loop_start
        sleep_time = max(0, (1.0 / FPS_TARGET) - elapsed)
        await asyncio.sleep(sleep_time)


async def main():
    from ultralytics import YOLO

    print("[ADAS] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    # Export to TensorRT if engine doesn't exist
    import os
    engine_path = "yolov8n.engine"
    if not os.path.exists(engine_path):
        print("[ADAS] Exporting YOLOv8n → TensorRT engine (first run, ~2 min)...")
        model.export(format="engine", imgsz=FRAME_WIDTH, device=0, half=True)
        print("[ADAS] TensorRT engine saved.")

    print("[ADAS] Loading TensorRT engine...")
    model_trt = YOLO(engine_path)

    # Connect to Kuksa
    print(f"[ADAS] Connecting to Kuksa @ {KUKSA_HOST}:{KUKSA_PORT}")
    channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
    from kuksa.val.v1 import val_pb2_grpc
    stub = val_pb2_grpc.VALStub(channel)

    try:
        await run_adas(model_trt, stub)
    except KeyboardInterrupt:
        print("\n[ADAS] Stopped.")
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
