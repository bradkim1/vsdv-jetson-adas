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
import pathlib
import time
import asyncio
import socket
import struct
import threading
import numpy as np

# Shared VSS layer from <repo>/agent, resolved relative to this file.
_AGENT_DIR = os.environ.get(
    "VSDV_AGENT_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent / "agent"))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from vss import VSS                      # noqa: E402

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

async def publish_signals(v: VSS, vehicle):
    """
    Read CARLA vehicle state and push VSS signals to Kuksa.

    Every path below was checked against the live databroker (VSS 5.1). Two
    paths this used to write do not exist in any VSS release and caused the
    databroker to reject the ENTIRE SetRequest, so none of the ego signals ever
    landed:
        Vehicle.Powertrain.Accelerator.PedalPosition  -> Chassis, not Powertrain
        Vehicle.CurrentLocation.Speed                 -> no such signal; the
                                                         speed is Vehicle.Speed
    SteeringWheel.Angle is also int16, not float — writing a float put the value
    in the wrong protobuf oneof.
    """
    publish_errors = 0
    while True:
        v_vec = vehicle.get_velocity()
        c = vehicle.get_control()
        loc = vehicle.get_location()

        speed_ms = (v_vec.x**2 + v_vec.y**2 + v_vec.z**2) ** 0.5
        speed_kmh = speed_ms * 3.6

        values = {
            "Vehicle.Speed":                             speed_kmh,
            "Vehicle.Chassis.SteeringWheel.Angle":       int(round(c.steer * 540)),
            "Vehicle.Chassis.Accelerator.PedalPosition": max(0, min(100, int(c.throttle * 100))),
            # NOTE: CARLA map coordinates scaled into degrees — a placeholder
            # projection, not a real geodetic transform.
            "Vehicle.CurrentLocation.Latitude":          loc.y * 0.000009,
            "Vehicle.CurrentLocation.Longitude":         loc.x * 0.000009,
        }

        try:
            await v.set(values)
            print(f"[VSS] Speed={speed_kmh:6.1f}km/h  Steer={c.steer*540:7.1f}deg  "
                  f"Throttle={c.throttle*100:3.0f}%  "
                  f"GPS=({loc.y*0.000009:.6f},{loc.x*0.000009:.6f})")
        except Exception as e:
            publish_errors += 1
            if publish_errors <= 3 or publish_errors % 50 == 0:
                print(f"[Kuksa] publish failed ({publish_errors}): {e}")

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
    try:
        async with VSS(f"{KUKSA_HOST}:{KUKSA_PORT}") as v:
            print(f"[Kuksa] Connected to {KUKSA_HOST}:{KUKSA_PORT}")
            await publish_signals(v, vehicle)
    except KeyboardInterrupt:
        print("\n[Bridge] Stopping...")
    finally:
        camera.stop()
        camera.destroy()
        vehicle.destroy()
        print("[Bridge] Cleaned up.")


if __name__ == "__main__":
    asyncio.run(main())
