#!/usr/bin/env python3
"""
Closed-loop scenario runner for the vSDV scenario-search agent.

Runs on the host. For each scenario it:
  1. renders the scene for the current true distance,
  2. pushes the frame to the Jetson ADAS node over the CARLA wire protocol,
  3. publishes ego speed as VSS,
  4. reads the brake command the Jetson's AEB app produced,
  5. applies that brake to the kinematics and steps time.

The Jetson is therefore a genuine hardware-in-the-loop participant: the real
TensorRT engine and the real vehicle app close the loop; nothing is stubbed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time

from scenario import (ScenarioSpec, EgoState, Renderer, Camera,
                      stopping_distance_m, OBSTACLE_HEIGHT_M)
from vss import VSS

ASSETS = "/home/bk/vsdv-adas/agent/assets"
FRAME_PORT = 9999

P = "Vehicle.ADAS.ObstacleDetection."
SIG_DISTANCE = P + "DistanceToObject"
SIG_COUNT    = P + "ObjectCount"
SIG_LATENCY  = P + "InferenceLatency"
SIG_WARNING  = P + "IsWarning"
SIG_SPEED    = "Vehicle.Speed"
SIG_BRAKE    = "Vehicle.Chassis.Brake.PedalPosition"
SIG_AEB      = "Vehicle.ADAS.EBA.IsEngaged"

READ_SIGNALS = [SIG_DISTANCE, SIG_COUNT, SIG_LATENCY, SIG_BRAKE, SIG_AEB]


class FrameServer:
    """Serves length-prefixed JPEG frames, exactly like carla_kuksa_bridge.py."""

    def __init__(self, port: int = FRAME_PORT):
        self.port = port
        self._writers: set[asyncio.StreamWriter] = set()
        self._server = None
        self.connected = asyncio.Event()

    async def start(self):
        self._server = await asyncio.start_server(
            self._on_client, "0.0.0.0", self.port)
        print(f"[frames] listening on 0.0.0.0:{self.port}")

    async def _on_client(self, reader, writer):
        peer = writer.get_extra_info("peername")
        print(f"[frames] DUT connected from {peer}")
        self._writers.add(writer)
        self.connected.set()
        try:
            await reader.read()          # client never sends; wait for EOF
        except Exception:
            pass
        finally:
            self._writers.discard(writer)
            if not self._writers:
                self.connected.clear()
            print(f"[frames] DUT disconnected {peer}")

    async def broadcast(self, jpeg: bytes):
        dead = []
        for w in list(self._writers):
            try:
                w.write(len(jpeg).to_bytes(4, "big") + jpeg)
                await w.drain()
            except Exception:
                dead.append(w)
        for w in dead:
            self._writers.discard(w)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


async def _reset(v: VSS):
    """Put the signal plane in a known state between scenarios."""
    await v.set({SIG_SPEED: 0.0, SIG_DISTANCE: 999.0, P + "TimeToCollision": 999.0,
                 SIG_COUNT: 0, SIG_WARNING: False})
    await asyncio.sleep(0.4)


async def run_scenario(spec: ScenarioSpec, frames: FrameServer, v: VSS,
                       verbose: bool = False) -> dict:
    """Execute one scenario end to end and return its verdict record."""
    cam = spec.camera()
    renderer = Renderer(ASSETS, cam)
    dt = 1.0 / spec.fps

    await _reset(v)

    ego = EgoState(speed_ms=spec.ego_speed_kmh / 3.6,
                   gap_m=spec.initial_distance_m,
                   lead_speed_ms=spec.lead_speed_kmh / 3.6)

    trace = []
    first_detect_gap = None
    first_brake_gap = None
    eba_gap = None
    dist_errors = []
    latencies = []
    detect_frames = 0
    n_frames = 0

    t_start = time.time()
    while ego.t < spec.max_duration_s:
        # 1. render + stream the current world state
        img = renderer.render(spec.obstacle, ego.gap_m,
                              light_level=spec.light_level,
                              fog_density=spec.fog_density)
        await frames.broadcast(renderer.jpeg(img))

        # 2. publish ego state, 3. read what the DUT decided
        await v.set({SIG_SPEED: ego.speed_ms * 3.6})
        await asyncio.sleep(dt)
        got = await v.get(*READ_SIGNALS)

        brake = float(got.get(SIG_BRAKE) or 0.0)
        count = int(got.get(SIG_COUNT) or 0)
        est_d = got.get(SIG_DISTANCE)
        lat = got.get(SIG_LATENCY)
        aeb = bool(got.get(SIG_AEB))

        n_frames += 1
        if count > 0:
            detect_frames += 1
            if first_detect_gap is None:
                first_detect_gap = ego.gap_m
            if est_d is not None and est_d < 999 and ego.gap_m > 0.5:
                dist_errors.append((est_d - ego.gap_m) / ego.gap_m * 100.0)
        if lat:
            latencies.append(float(lat))
        if brake > 0 and first_brake_gap is None:
            first_brake_gap = ego.gap_m
        if aeb and eba_gap is None:
            eba_gap = ego.gap_m

        trace.append({"t": round(ego.t, 2), "gap": round(ego.gap_m, 2),
                      "speed_kmh": round(ego.speed_ms * 3.6, 2),
                      "brake": brake, "count": count,
                      "est_d": round(est_d, 2) if est_d is not None else None,
                      "aeb": aeb})

        if verbose:
            print(f"  t={ego.t:5.1f}s gap={ego.gap_m:7.2f}m "
                  f"v={ego.speed_ms*3.6:6.1f}kph brake={brake:5.0f}% "
                  f"n={count} est={est_d if est_d is None else round(est_d,1)}")

        # 4. advance the world with the DUT's brake command
        ego.step(dt, brake, spec.friction)

        if ego.gap_m <= 0:
            outcome = "COLLISION"
            break
        if ego.speed_ms <= 0.1:
            outcome = "AVOIDED"
            break
    else:
        outcome = "TIMEOUT"

    impact_kmh = ego.speed_ms * 3.6 if outcome == "COLLISION" else 0.0

    # Brake releases while still closing on the obstacle. A detector with no
    # tracking or hysteresis drops out for a single frame, the distance snaps
    # back to "no obstacle", and the vehicle app cancels an active emergency
    # stop. Each release here is a real cancellation of in-progress braking.
    brake_releases = 0
    for prev, cur in zip(trace, trace[1:]):
        if prev["brake"] > 0 and cur["brake"] == 0 and cur["gap"] > 0 \
                and cur["speed_kmh"] > 1.0:
            brake_releases += 1
    detect_dropouts = sum(
        1 for prev, cur in zip(trace, trace[1:])
        if prev["count"] > 0 and cur["count"] == 0 and cur["gap"] > 0)

    return {
        "scenario_id": spec.scenario_id(),
        "spec": spec.as_dict(),
        "camera": cam.label(),
        "outcome": outcome,
        "impact_speed_kmh": round(impact_kmh, 2),
        "min_gap_m": round(max(ego.min_gap_m, 0.0), 2),
        "final_speed_kmh": round(ego.speed_ms * 3.6, 2),
        "duration_s": round(ego.t, 2),
        "wall_s": round(time.time() - t_start, 2),
        "first_detect_gap_m": None if first_detect_gap is None else round(first_detect_gap, 2),
        "first_brake_gap_m": None if first_brake_gap is None else round(first_brake_gap, 2),
        "eba_gap_m": None if eba_gap is None else round(eba_gap, 2),
        "detection_rate": round(detect_frames / max(1, n_frames), 3),
        "brake_releases": brake_releases,
        "detect_dropouts": detect_dropouts,
        "mean_distance_error_pct": round(sum(dist_errors) / len(dist_errors), 1) if dist_errors else None,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "max_latency_ms": round(max(latencies), 1) if latencies else None,
        "ideal_stopping_distance_m": round(
            stopping_distance_m(spec.ego_speed_kmh, spec.friction), 2),
        "physically_avoidable": spec.initial_distance_m >= stopping_distance_m(
            spec.ego_speed_kmh, spec.friction),
        "trace": trace,
    }


def summarise(rec: dict) -> str:
    v = rec
    det = "never" if v["first_detect_gap_m"] is None else f"{v['first_detect_gap_m']}m"
    brk = "never" if v["first_brake_gap_m"] is None else f"{v['first_brake_gap_m']}m"
    err = "n/a" if v["mean_distance_error_pct"] is None else f"{v['mean_distance_error_pct']:+.0f}%"
    return (f"{v['outcome']:<9} impact={v['impact_speed_kmh']:>5.1f}kph "
            f"detect@{det:<8} brake@{brk:<8} dist_err={err:<7} "
            f"det_rate={v['detection_rate']:.2f} releases={v['brake_releases']} "
            f"lat={v['mean_latency_ms']}ms")


async def main():
    ap = argparse.ArgumentParser(description="run one vSDV scenario against the Jetson DUT")
    ap.add_argument("--speed", type=float, default=50.0)
    ap.add_argument("--distance", type=float, default=80.0)
    ap.add_argument("--obstacle", default="person", choices=list(OBSTACLE_HEIGHT_M))
    ap.add_argument("--lead-speed", type=float, default=0.0)
    ap.add_argument("--light", type=float, default=1.0)
    ap.add_argument("--fog", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="write the verdict JSON here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    spec = ScenarioSpec(ego_speed_kmh=args.speed, obstacle=args.obstacle,
                        initial_distance_m=args.distance,
                        lead_speed_kmh=args.lead_speed, light_level=args.light,
                        fog_density=args.fog, fps=args.fps,
                        max_duration_s=args.duration)

    frames = FrameServer()
    await frames.start()
    print("[runner] waiting for the Jetson ADAS node to connect...")
    await asyncio.wait_for(frames.connected.wait(), timeout=120)

    async with VSS() as v:
        rec = await run_scenario(spec, frames, v, verbose=args.verbose)

    print("\n" + summarise(rec))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"verdict written to {args.out}")
    await frames.stop()


if __name__ == "__main__":
    asyncio.run(main())
