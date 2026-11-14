#!/usr/bin/env python3
"""
Failure-envelope analysis over a sweep's verdicts.

Answers the question the sweep exists to answer: under which conditions does
the system fail, and is the failure the perception's, the control law's, or
simply physics?
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

from scenario import stopping_distance_m


def load(path) -> list[dict]:
    return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]


def classify(r: dict) -> str:
    """Attribute each failure to the subsystem responsible."""
    if r["outcome"] != "COLLISION":
        return "ok"
    if not r["physically_avoidable"]:
        return "physics"              # no system could have stopped in time
    if r["first_detect_gap_m"] is None:
        return "perception: never detected"
    if r["first_detect_gap_m"] < r["ideal_stopping_distance_m"]:
        return "perception: detected too late"
    if r["brake_releases"] > 0:
        return "control: brake released while closing"
    if r["first_brake_gap_m"] is None:
        return "control: never braked despite detection"
    return "control: braked too weakly"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?",
                    default="/home/bk/vsdv-adas/agent/results/sweep.jsonl")
    args = ap.parse_args()
    recs = load(args.jsonl)
    if not recs:
        print("no verdicts"); return

    print(f"{len(recs)} scenarios from {args.jsonl}\n")

    # ── outcome grid ─────────────────────────────────────────────────────────
    groups = defaultdict(list)
    for r in recs:
        groups[(r["spec"]["obstacle"], r["spec"]["light_level"],
                r["spec"]["fog_density"])].append(r)

    for (obst, light, fog), rows in sorted(groups.items()):
        print(f"── {obst}  light={light}  fog={fog} " + "─" * 40)
        speeds = sorted({r["spec"]["ego_speed_kmh"] for r in rows})
        dists = sorted({r["spec"]["initial_distance_m"] for r in rows})
        print(f"{'start gap':>10} | " + " | ".join(f"{s:>6.0f} km/h" for s in speeds))
        for d in dists:
            cells = []
            for s in speeds:
                r = next((x for x in rows
                          if x["spec"]["ego_speed_kmh"] == s
                          and x["spec"]["initial_distance_m"] == d), None)
                if r is None:
                    cells.append(" " * 11)
                elif r["outcome"] == "COLLISION":
                    cells.append(f"HIT {r['impact_speed_kmh']:>5.1f}kph")
                else:
                    cells.append(f"{r['outcome'][:4]} {r['min_gap_m']:>4.1f}m ")
            print(f"{d:>9.0f}m | " + " | ".join(cells))
        print()

    # ── blame ────────────────────────────────────────────────────────────────
    blame = defaultdict(int)
    for r in recs:
        blame[classify(r)] += 1
    print("── failure attribution " + "─" * 44)
    for k, n in sorted(blame.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}/{len(recs)}  {k}")

    # ── perception characterisation ──────────────────────────────────────────
    print("\n── perception, by obstacle " + "─" * 40)
    by_obst = defaultdict(list)
    for r in recs:
        by_obst[r["spec"]["obstacle"]].append(r)
    for obst, rows in sorted(by_obst.items()):
        dets = [r["first_detect_gap_m"] for r in rows if r["first_detect_gap_m"]]
        errs = [r["mean_distance_error_pct"] for r in rows
                if r["mean_distance_error_pct"] is not None]
        rates = [r["detection_rate"] for r in rows]
        drops = sum(r["detect_dropouts"] for r in rows)
        rels = sum(r["brake_releases"] for r in rows)
        print(f"  {obst:<8} first detection: "
              f"{'never' if not dets else f'{min(dets):.1f}–{max(dets):.1f}m'}"
              f"   mean det-rate {sum(rates)/len(rates):.2f}"
              f"   dist err {'n/a' if not errs else f'{sum(errs)/len(errs):+.0f}%'}"
              f"   dropouts {drops}   brake releases {rels}")

    # ── the headline ─────────────────────────────────────────────────────────
    NEAR_MISS_M = 2.0
    print("\n── envelope " + "─" * 55)
    for obst, rows in sorted(by_obst.items()):
        speeds = sorted({r["spec"]["ego_speed_kmh"] for r in rows})
        # A speed only counts as clear if EVERY run at it avoided collision.
        clear = [sp for sp in speeds
                 if all(r["outcome"] != "COLLISION"
                        for r in rows if r["spec"]["ego_speed_kmh"] == sp)]
        hit = [sp for sp in speeds
               if any(r["outcome"] == "COLLISION"
                      for r in rows if r["spec"]["ego_speed_kmh"] == sp)]
        if hit:
            print(f"  {obst:<8} all-clear up to "
                  f"{max(clear):.0f} km/h" if clear else f"  {obst:<8} no all-clear speed", end="")
            print(f"; collisions from {min(hit):.0f} km/h "
                  f"(worst impact {max(r['impact_speed_kmh'] for r in rows):.1f} km/h)")
        else:
            print(f"  {obst:<8} all-clear at every speed tested "
                  f"(max {max(speeds):.0f} km/h)")

        near = [r for r in rows
                if r["outcome"] != "COLLISION" and r["min_gap_m"] < NEAR_MISS_M]
        if near:
            worst = min(near, key=lambda r: r["min_gap_m"])
            print(f"  {'':<8} {len(near)} of these 'avoided' runs stopped within "
                  f"{NEAR_MISS_M:.0f} m — worst {worst['min_gap_m']:.1f} m at "
                  f"{worst['spec']['ego_speed_kmh']:.0f} km/h. Not a pass margin.")

    lat = [r["mean_latency_ms"] for r in recs if r["mean_latency_ms"]]
    if lat:
        print(f"\n  inference latency: mean {sum(lat)/len(lat):.1f} ms, "
              f"worst-case frame {max(r['max_latency_ms'] for r in recs if r['max_latency_ms']):.1f} ms")


if __name__ == "__main__":
    main()
