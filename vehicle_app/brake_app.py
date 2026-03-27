#!/usr/bin/env python3
"""
vSDV Vehicle App: Autonomous Emergency Braking (AEB)
Runs on Jetson Nano.

Subscribes to ADAS signals from Kuksa Databroker.
Applies graduated virtual braking based on obstacle proximity.
Publishes brake commands back to Kuksa.

Braking logic:
  dist > 10m  : no action
  5 – 10m     : warning  + light brake  (30%)
  3 – 5m      : caution  + firm brake   (70%)
  < 3m        : EMERGENCY brake         (100%) + AEB engaged
"""

import asyncio
import grpc
import time
from dataclasses import dataclass, field

KUKSA_HOST = "127.0.0.1"
KUKSA_PORT = 55555

# Signals to subscribe
SUB_SIGNALS = [
    "Vehicle.ADAS.ObstacleDetection.IsWarning",  # collision warning
    "Vehicle.Chassis.Brake.PedalPosition",        # distance hint (0-100)
    "Vehicle.Speed",
]

# Signals to publish (actuator responses)
SIG_AEB      = "Vehicle.ADAS.EBA.IsEngaged"      # Emergency Brake Assist
SIG_BRAKE    = "Vehicle.Chassis.Brake.PedalPosition"


@dataclass
class VehicleState:
    speed_kmh:  float = 0.0
    distance:   float = 99.0
    collision:  bool  = False
    n_obstacles: int  = 0
    brake_pct:  float = 0.0
    aeb:        bool  = False
    last_event: str   = "INIT"


# ── Braking logic ─────────────────────────────────────────────────────────────

def compute_brake(state: VehicleState) -> tuple[float, bool, str]:
    """
    Returns (brake_pct, aeb_engaged, event_label).
    Graduated response based on nearest obstacle distance.
    """
    d = state.distance
    spd = state.speed_kmh

    if d > 10.0 or spd < 0.5:
        return 0.0, False, "CLEAR"
    elif d > 5.0:
        return 30.0, False, "WARNING"
    elif d > 3.0:
        return 70.0, False, "CAUTION"
    else:
        return 100.0, True, "EMERGENCY"


# ── Kuksa helpers ─────────────────────────────────────────────────────────────

async def publish_brake(stub, brake_pct: float, aeb: bool):
    """Push brake commands to Kuksa."""
    from kuksa.val.v1 import val_pb2, types_pb2

    updates = []
    for path, sig_type, value in [
        (SIG_BRAKE, "uint8", brake_pct),
        (SIG_AEB,   "bool",  aeb),
    ]:
        dp = types_pb2.Datapoint()
        if sig_type == "uint8":
            dp.uint32 = int(value)
        elif sig_type == "bool":
            dp.bool = bool(value)
        updates.append(val_pb2.EntryUpdate(
            entry=types_pb2.DataEntry(path=path, value=dp),
            fields=[types_pb2.Field.FIELD_VALUE],
        ))
    try:
        await stub.Set(val_pb2.SetRequest(updates=updates))
    except Exception as e:
        print(f"[Brake] Publish error: {e}")


async def subscribe_one(stub, path: str, state: VehicleState):
    """Subscribe to a single signal, retry on failure."""
    from kuksa.val.v1 import val_pb2, types_pb2
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
                    value = update.entry.value
                    if path == "Vehicle.Speed":
                        state.speed_kmh = value.float
                    elif path == "Vehicle.Chassis.Brake.PedalPosition":
                        # Distance hint: 100=closest(0m), 0=clear(10m+)
                        state.distance = max(0.1, 10.0 - value.uint32 / 10.0)
                    elif path == "Vehicle.ADAS.ObstacleDetection.IsWarning":
                        state.collision = value.bool
                    elif path == "Vehicle.ADAS.ObstacleDetection.IsActive":
                        state.n_obstacles = 1 if value.bool else 0
        except Exception as e:
            print(f"[Sub] {path}: {e} — retrying in 2s")
            await asyncio.sleep(2)


async def subscribe_loop(stub, state: VehicleState):
    """Subscribe to all signals concurrently."""
    await asyncio.gather(*[subscribe_one(stub, p, state) for p in SUB_SIGNALS])


# ── Main control loop ─────────────────────────────────────────────────────────

async def control_loop(stub, state: VehicleState):
    """Evaluate braking at 10 Hz and publish commands."""
    prev_event = "INIT"
    prev_brake = -1.0

    print("[AEB]  Autonomous Emergency Braking — ACTIVE")
    print("[AEB]  Monitoring Kuksa signals at 10 Hz\n")
    print(f"{'Time':>8}  {'Speed':>8}  {'Dist':>8}  {'Obstacles':>9}  "
          f"{'Brake':>7}  {'AEB':>5}  Event")
    print("─" * 72)

    while True:
        brake_pct, aeb, event = compute_brake(state)

        # Only publish and log on state change
        if brake_pct != prev_brake or event != prev_event:
            await publish_brake(stub, brake_pct, aeb)

            ts = time.strftime("%H:%M:%S")
            aeb_str = "ON ⚡" if aeb else "off"
            marker  = " ◀ AEB ENGAGED!" if aeb else (" ▲" if event != "CLEAR" else "")

            print(f"{ts:>8}  {state.speed_kmh:>6.1f}km/h  "
                  f"{state.distance:>6.1f}m  "
                  f"{state.n_obstacles:>9}  "
                  f"{brake_pct:>5.0f}%  "
                  f"{aeb_str:>5}  "
                  f"{event}{marker}")

            state.brake_pct = brake_pct
            state.aeb       = aeb
            state.last_event = event
            prev_brake = brake_pct
            prev_event = event

        await asyncio.sleep(0.1)   # 10 Hz


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    print(f"[AEB] Connecting to Kuksa @ {KUKSA_HOST}:{KUKSA_PORT} ...")
    channel = grpc.aio.insecure_channel(f"{KUKSA_HOST}:{KUKSA_PORT}")
    from kuksa.val.v1 import val_pb2_grpc
    stub = val_pb2_grpc.VALStub(channel)

    state = VehicleState()

    print("[AEB] Subscribing to ADAS + vehicle signals...")
    try:
        await asyncio.gather(
            subscribe_loop(stub, state),
            control_loop(stub, state),
        )
    except KeyboardInterrupt:
        print("\n[AEB] Stopped.")
        # Release brakes on exit
        await publish_brake(stub, 0.0, False)
    finally:
        await channel.close()


if __name__ == "__main__":
    asyncio.run(main())
