#!/usr/bin/env python3
"""
vSDV Real-Time Dashboard — Host (RTX 5090)
FastAPI + WebSocket server.

Connects to Kuksa on Jetson (192.168.55.1:55555).
Subscribes to all VSS signals and streams to browser via WebSocket.
Also proxies CARLA camera stream as MJPEG for browser display.
"""

import asyncio
import json
import socket
import struct
import threading
import time
from contextlib import asynccontextmanager

import grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

KUKSA_HOST  = "192.168.55.1"
KUKSA_PORT  = 55555
CARLA_HOST  = "127.0.0.1"
CARLA_PORT  = 9999

# Latest VSS state (shared across WebSocket clients)
vehicle_state = {
    "speed":       0.0,
    "steer":       0.0,
    "throttle":    0.0,
    "lat":         37.7749,
    "lon":        -122.4194,
    "brake":       0,
    "aeb":         False,
    "obstacle":    0,          # ObjectCount — 0 means nothing detected
    "warning":     False,
    "distance":    999.0,      # 999 = no obstacle, matching the ADAS node
    "ttc":         999.0,
    "latency_ms":  0.0,
    "timestamp":   0,
}

# Active WebSocket clients
ws_clients: list[WebSocket] = []

# Latest JPEG frame from CARLA
latest_jpeg: bytes = b""
jpeg_lock = threading.Lock()


# ── CARLA camera receiver thread ──────────────────────────────────────────────

def carla_camera_thread():
    global latest_jpeg
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((CARLA_HOST, CARLA_PORT))
            print(f"[Camera] Connected to CARLA stream @ {CARLA_HOST}:{CARLA_PORT}")
            while True:
                raw = b""
                while len(raw) < 4:
                    chunk = sock.recv(4 - len(raw))
                    if not chunk:
                        raise ConnectionResetError
                    raw += chunk
                length = struct.unpack(">I", raw)[0]
                data = b""
                while len(data) < length:
                    chunk = sock.recv(min(65536, length - len(data)))
                    if not chunk:
                        raise ConnectionResetError
                    data += chunk
                with jpeg_lock:
                    latest_jpeg = data
        except Exception as e:
            print(f"[Camera] Lost connection: {e} — retrying in 3s")
            time.sleep(3)


# ── Kuksa subscriber ──────────────────────────────────────────────────────────

async def kuksa_subscriber():
    """Subscribe to all VSS signals and update vehicle_state."""
    from kuksa.val.v1 import val_pb2, val_pb2_grpc, types_pb2

    SIGNAL_MAP = {
        "Vehicle.Speed":                             ("speed",    "float"),
        "Vehicle.Chassis.SteeringWheel.Angle":             ("steer",      "int32"),
        "Vehicle.Chassis.Brake.PedalPosition":             ("brake",      "uint32"),
        "Vehicle.Chassis.Accelerator.PedalPosition":       ("throttle",   "uint32"),
        "Vehicle.CurrentLocation.Latitude":          ("lat",      "double"),
        "Vehicle.CurrentLocation.Longitude":         ("lon",      "double"),
        "Vehicle.ADAS.ObstacleDetection.IsWarning":        ("warning",    "bool"),
        "Vehicle.ADAS.ObstacleDetection.ObjectCount":      ("obstacle",   "uint32"),
        "Vehicle.ADAS.ObstacleDetection.DistanceToObject": ("distance",   "float"),
        "Vehicle.ADAS.ObstacleDetection.TimeToCollision":  ("ttc",        "float"),
        "Vehicle.ADAS.ObstacleDetection.InferenceLatency": ("latency_ms", "float"),
        "Vehicle.ADAS.EBA.IsEngaged":                      ("aeb",        "bool"),
    }
    # NOTE: IsActive and CruiseControl.SpeedSet used to appear here. IsActive
    # exists in neither VSS 4.0 nor 5.1 — that subscription could never bind —
    # and SpeedSet was carrying inference latency. The ADAS node now publishes
    # ObjectCount / DistanceToObject / TimeToCollision / InferenceLatency from
    # the agent/vss_overlay_adas.json overlay instead.

    while True:
        try:
            channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
            stub = val_pb2_grpc.VALStub(channel)

            # Subscribe to each known-good signal
            tasks = []
            for path, (key, typ) in SIGNAL_MAP.items():
                tasks.append(subscribe_signal(stub, path, key, typ, types_pb2, val_pb2))
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"[Kuksa] Connection error: {e} — retrying in 3s")
            await asyncio.sleep(3)


async def subscribe_signal(stub, path, key, typ, types_pb2, val_pb2):
    while True:
        try:
            req = val_pb2.SubscribeRequest(entries=[
                val_pb2.SubscribeEntry(
                    path=path,
                    view=types_pb2.VIEW_CURRENT_VALUE,
                    fields=[types_pb2.FIELD_VALUE],
                )
            ])
            async for response in stub.Subscribe(req):
                for update in response.updates:
                    v = update.entry.value
                    if typ == "float":
                        vehicle_state[key] = round(v.float, 2)
                    elif typ == "double":
                        vehicle_state[key] = round(v.double, 6)
                    elif typ == "bool":
                        vehicle_state[key] = v.bool
                    elif typ == "uint32":
                        vehicle_state[key] = v.uint32
                    elif typ == "int32":
                        # int8/int16/int32 all ride in the int32 oneof field
                        vehicle_state[key] = v.int32
                    vehicle_state["timestamp"] = int(time.time() * 1000)
        except Exception:
            await asyncio.sleep(2)


# ── WebSocket broadcaster ─────────────────────────────────────────────────────

async def broadcaster():
    """Push vehicle_state to all connected WebSocket clients at 10 Hz."""
    while True:
        if ws_clients:
            msg = json.dumps(vehicle_state)
            dead = []
            for ws in ws_clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                ws_clients.remove(ws)
        await asyncio.sleep(0.1)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start camera receiver thread
    t = threading.Thread(target=carla_camera_thread, daemon=True)
    t.start()
    # Start Kuksa subscriber and broadcaster
    asyncio.create_task(kuksa_subscriber())
    asyncio.create_task(broadcaster())
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()   # keep connection alive
    except WebSocketDisconnect:
        ws_clients.remove(ws)


@app.get("/camera")
async def camera_feed():
    """MJPEG stream for live CARLA camera."""
    def generate():
        while True:
            with jpeg_lock:
                frame = latest_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.05)   # ~20 FPS

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    print("[Dashboard] Starting at http://0.0.0.0:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
