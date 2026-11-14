#!/usr/bin/env python3
"""
vSDV Vehicle App: Autonomous Emergency Braking (AEB) — Jetson Orin Nano

Consumes *perception* signals from the Kuksa Databroker and owns *actuation*:
this node is the only writer of the brake pedal and the EBA engagement flag.

Subscribes:
    Vehicle.ADAS.ObstacleDetection.DistanceToObject  float  m
    Vehicle.ADAS.ObstacleDetection.TimeToCollision   float  s
    Vehicle.Speed                                    float  km/h
Publishes:
    Vehicle.Chassis.Brake.PedalPosition              uint8  percent
    Vehicle.ADAS.EBA.IsEngaged                       bool

Braking law — graduated on time-to-collision, with a distance backstop:
    TTC > 2.6 s and dist > 12 m : release
    TTC 1.6 – 2.6 s             : 30 %  partial
    TTC 0.9 – 1.6 s             : 70 %  firm
    TTC < 0.9 s  or dist < 4 m  : 100 % emergency, EBA engaged
"""

import asyncio
import os
import pathlib
import sys
import time
from dataclasses import dataclass

# The shared VSS layer lives in <repo>/agent. Resolve it relative to this file
# so the same source runs from a git checkout and from the container image
# (/app/adas/... + /app/agent/...). VSDV_AGENT_DIR overrides for odd layouts.
_AGENT_DIR = os.environ.get(
    "VSDV_AGENT_DIR",
    str(pathlib.Path(__file__).resolve().parent.parent / "agent"))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from vss import VSS                      # noqa: E402

KUKSA_HOST = os.environ.get("KUKSA_HOST", "127.0.0.1")
CONTROL_HZ = float(os.environ.get("CONTROL_HZ", "10"))

P = "Vehicle.ADAS.ObstacleDetection."
SIG_DISTANCE = P + "DistanceToObject"
SIG_TTC      = P + "TimeToCollision"
SIG_SPEED    = "Vehicle.Speed"
SIG_BRAKE    = "Vehicle.Chassis.Brake.PedalPosition"
SIG_AEB      = "Vehicle.ADAS.EBA.IsEngaged"

NO_OBSTACLE_M = 999.0


@dataclass
class VehicleState:
    speed_kmh: float = 0.0
    distance_m: float = NO_OBSTACLE_M
    ttc_s: float = NO_OBSTACLE_M


def compute_brake(s: VehicleState) -> tuple[float, bool, str]:
    """Return (brake_percent, eba_engaged, event_label)."""
    if s.speed_kmh < 0.5:
        return 0.0, False, "STOPPED"
    if s.ttc_s < 0.9 or s.distance_m < 4.0:
        return 100.0, True, "EMERGENCY"
    if s.ttc_s < 1.6:
        return 70.0, False, "FIRM"
    if s.ttc_s < 2.6:
        return 30.0, False, "PARTIAL"
    if s.distance_m < 12.0:
        return 30.0, False, "PROXIMITY"
    return 0.0, False, "CLEAR"


async def subscribe_loop(v: VSS, state: VehicleState):
    paths = [SIG_DISTANCE, SIG_TTC, SIG_SPEED]
    while True:
        try:
            async for delta in v.subscribe(paths):
                if SIG_DISTANCE in delta and delta[SIG_DISTANCE] is not None:
                    state.distance_m = float(delta[SIG_DISTANCE])
                if SIG_TTC in delta and delta[SIG_TTC] is not None:
                    state.ttc_s = float(delta[SIG_TTC])
                if SIG_SPEED in delta and delta[SIG_SPEED] is not None:
                    state.speed_kmh = float(delta[SIG_SPEED])
        except Exception as e:
            print(f"[Sub] subscription lost: {e} — retrying in 2s", flush=True)
            await asyncio.sleep(2)


async def control_loop(v: VSS, state: VehicleState):
    print("[AEB] active — graduated braking on TTC\n", flush=True)
    print(f"{'Time':>8} {'Speed':>9} {'Dist':>9} {'TTC':>7} {'Brake':>6} {'EBA':>4}  Event",
          flush=True)
    print("-" * 68, flush=True)

    prev = None
    while True:
        brake, eba, event = compute_brake(state)
        try:
            await v.set({SIG_BRAKE: int(brake), SIG_AEB: eba})
        except Exception as e:
            print(f"[AEB] publish failed: {e}", flush=True)

        if (brake, event) != prev:
            print(f"{time.strftime('%H:%M:%S'):>8} {state.speed_kmh:>7.1f}kph "
                  f"{state.distance_m:>7.1f}m {state.ttc_s:>6.1f}s "
                  f"{brake:>5.0f}% {'ON' if eba else 'off':>4}  {event}"
                  f"{'  <<< AEB' if eba else ''}", flush=True)
            prev = (brake, event)

        await asyncio.sleep(1.0 / CONTROL_HZ)


async def main():
    async with VSS(f"{KUKSA_HOST}:55555") as v:
        print(f"[AEB] connected to Kuksa @ {KUKSA_HOST}:55555", flush=True)
        state = VehicleState()
        try:
            await asyncio.gather(subscribe_loop(v, state), control_loop(v, state))
        except KeyboardInterrupt:
            print("\n[AEB] stopped — releasing brakes")
            await v.set({SIG_BRAKE: 0, SIG_AEB: False})


if __name__ == "__main__":
    asyncio.run(main())
