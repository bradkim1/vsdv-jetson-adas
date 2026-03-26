#!/usr/bin/env python3
"""
CARLA → Kuksa Bridge + Camera Streamer
Runs on Host (RTX 5090).

- Connects to CARLA server
- Spawns ego vehicle with camera, lidar, radar sensors
- Pushes vehicle VSS signals to Kuksa on Jetson (192.168.55.1:55555)
- Streams camera frames to Jetson ADAS over TCP (port 9999)
"""

import sys
import os
import time
import asyncio
import socket
import struct
import threading
import numpy as np
import grpc

# Add CARLA Python API to path
CARLA_PATH = os.environ.get("CARLA_PATH", "/home/bk/carla/CARLA_0.9.15")
sys.path.append(os.path.join(CARLA_PATH, "PythonAPI/carla/dist/carla-0.9.15-py3.10-linux-x86_64.egg"))
sys.path.append(os.path.join(CARLA_PATH, "PythonAPI/carla"))

import carla

KUKSA_HOST   = "192.168.55.1"
KUKSA_PORT   = 55555
CAMERA_PORT  = 9999        # TCP port to stream frames to Jetson ADAS
CAMERA_W     = 640
CAMERA_H     = 480
CARLA_HOST   = "127.0.0.1"
CARLA_PORT   = 2000

# Latest camera frame (shared between threads)
latest_frame = None
frame_lock   = threading.Lock()
frame_clients = []
clients_lock  = threading.Lock()


# ── Camera frame server ───────────────────────────────────────────────────────

def frame_server_thread():
    """TCP server: streams JPEG frames to Jetson ADAS."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", CAMERA_PORT))
    server.listen(5)
    print(f"[Camera] Streaming server listening on port {CAMERA_PORT}")

    while True:
        conn, addr = server.accept()
        print(f"[Camera] Jetson ADAS connected from {addr}")
        with clients_lock:
            frame_clients.append(conn)


def push_frame_to_clients(jpeg_bytes: bytes):
    """Send JPEG frame to all connected clients (length-prefixed)."""
    with clients_lock:
        dead = []
        for conn in frame_clients:
            try:
                length = struct.pack(">I", len(jpeg_bytes))
                conn.sendall(length + jpeg_bytes)
            except Exception:
                dead.append(conn)
        for d in dead:
            frame_clients.remove(d)


# ── CARLA sensor callbacks ────────────────────────────────────────────────────

def on_camera_image(image):
    """Convert CARLA RGB image → JPEG and push to Jetson."""
    import cv2
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))   # BGRA
    bgr   = array[:, :, :3]
    _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    push_frame_to_clients(jpeg.tobytes())


# ── VSS signal publisher ──────────────────────────────────────────────────────

async def publish_signals(stub, vehicle):
    """Read CARLA vehicle state and push VSS signals to Kuksa."""
    from kuksa.val.v1 import val_pb2, types_pb2

    while True:
        t = vehicle.get_transform()
        v = vehicle.get_velocity()
        c = vehicle.get_control()
        loc = vehicle.get_location()

        speed_ms  = (v.x**2 + v.y**2 + v.z**2) ** 0.5
        speed_kmh = speed_ms * 3.6

        signal_values = {
            "Vehicle.Speed":                          ("float",  speed_kmh),
            "Vehicle.Chassis.SteeringWheel.Angle":    ("float",  c.steer * 540),
            "Vehicle.Powertrain.Accelerator.PedalPosition": ("float", c.throttle * 100),
            "Vehicle.CurrentLocation.Latitude":       ("double", loc.y * 0.000009),
            "Vehicle.CurrentLocation.Longitude":      ("double", loc.x * 0.000009),
            "Vehicle.CurrentLocation.Speed":          ("double", speed_ms),
        }

        updates = []
        for path, (sig_type, value) in signal_values.items():
            dp = types_pb2.Datapoint()
            if sig_type == "float":
                dp.float = float(value)
            elif sig_type == "double":
                dp.double = float(value)
            updates.append(val_pb2.EntryUpdate(
                entry=types_pb2.DataEntry(path=path, value=dp),
                fields=[types_pb2.Field.FIELD_VALUE],
            ))

        try:
            await stub.Set(val_pb2.SetRequest(updates=updates))
            print(f"[VSS] Speed={speed_kmh:.1f}km/h  Steer={c.steer*540:.1f}°  "
                  f"Throttle={c.throttle*100:.0f}%  GPS=({loc.y:.4f},{loc.x:.4f})")
        except Exception as e:
            print(f"[Kuksa] Error: {e}")

        await asyncio.sleep(0.2)   # 5 Hz VSS updates


# ── Autopilot + main loop ─────────────────────────────────────────────────────

async def main():
    # Connect to CARLA
    print(f"[CARLA] Connecting to {CARLA_HOST}:{CARLA_PORT} ...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"[CARLA] Connected — Map: {world.get_map().name}")

    # Spawn ego vehicle
    bp_lib    = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    spawn_pts  = world.get_map().get_spawn_points()
    vehicle    = world.spawn_actor(vehicle_bp, spawn_pts[0])
    vehicle.set_autopilot(True)
    print(f"[CARLA] Ego vehicle spawned: {vehicle.type_id} — Autopilot ON")

    # Attach RGB camera
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(CAMERA_W))
    cam_bp.set_attribute("image_size_y", str(CAMERA_H))
    cam_bp.set_attribute("fov", "90")
    cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    camera.listen(on_camera_image)
    print(f"[CARLA] RGB camera attached ({CAMERA_W}x{CAMERA_H})")

    # Start frame server thread
    t = threading.Thread(target=frame_server_thread, daemon=True)
    t.start()

    # Connect to Kuksa
    channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
    from kuksa.val.v1 import val_pb2_grpc
    stub = val_pb2_grpc.VALStub(channel)
    print(f"[Kuksa] Connected to {KUKSA_HOST}:{KUKSA_PORT}")

    try:
        await publish_signals(stub, vehicle)
    except KeyboardInterrupt:
        print("\n[Bridge] Stopping...")
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        await channel.close()
        print("[Bridge] Cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
