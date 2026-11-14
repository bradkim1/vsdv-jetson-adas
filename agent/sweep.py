#!/usr/bin/env python3
"""
Scenario sweep: run a grid of scenarios against the live Jetson DUT and record
one verdict per run as JSONL.

This is the search half of the agent. The analysis half (envelope.py) reads the
JSONL and reports where the system fails; an agent loop can then pick the next
grid from that report.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import pathlib
import time

from scenario import ScenarioSpec
from runner import FrameServer, run_scenario, summarise
from vss import VSS


def build_grid(args) -> list[ScenarioSpec]:
    combos = itertools.product(args.speeds, args.distances, args.obstacles,
                               args.lights, args.fogs)
    return [ScenarioSpec(ego_speed_kmh=s, initial_distance_m=d, obstacle=o,
                         light_level=l, fog_density=f, fps=args.fps,
                         max_duration_s=args.duration)
            for s, d, o, l, f in combos]


async def main():
    ap = argparse.ArgumentParser(description="sweep the vSDV scenario space")
    ap.add_argument("--speeds", type=float, nargs="+", default=[30, 50, 70])
    ap.add_argument("--distances", type=float, nargs="+", default=[60])
    ap.add_argument("--obstacles", nargs="+", default=["person", "vehicle"])
    ap.add_argument("--lights", type=float, nargs="+", default=[1.0])
    ap.add_argument("--fogs", type=float, nargs="+", default=[0.0])
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--out", default="/home/bk/vsdv-adas/agent/results/sweep.jsonl")
    ap.add_argument("--traces", action="store_true",
                    help="keep the per-tick trace in the JSONL (large)")
    args = ap.parse_args()

    grid = build_grid(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"sweep: {len(grid)} scenarios → {out}")
    frames = FrameServer()
    await frames.start()
    print("[sweep] waiting for the Jetson ADAS node...")
    await asyncio.wait_for(frames.connected.wait(), timeout=120)

    t0 = time.time()
    with out.open("w") as fh:
        async with VSS() as v:
            for i, spec in enumerate(grid, 1):
                print(f"\n[{i:>3}/{len(grid)}] {spec.scenario_id()}")
                rec = await run_scenario(spec, frames, v)
                print(f"        {summarise(rec)}")
                if not args.traces:
                    rec.pop("trace", None)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

    await frames.stop()
    print(f"\ndone in {time.time()-t0:.0f}s — {len(grid)} verdicts in {out}")


if __name__ == "__main__":
    asyncio.run(main())
