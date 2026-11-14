# [vSDV] AEB Scenario Sweep — 24 scenarios, hardware in the loop

**Date:** 2026-11-14 · **Target:** Jetson Orin Nano 8GB, JetPack 6.2 (L4T R36.4.7)
**Runtime:** 24 scenarios in 169 s · **Repo:** `~/vsdv-adas`, commit `0c62e7e`

## Method

The host renders a pinhole-calibrated road scene and streams it to the board on the
same wire format the CARLA bridge uses (4-byte length prefix + JPEG). The Jetson runs
the real TensorRT YOLOv8n engine and the real AEB vehicle app; the host reads back the
brake command the board produced and applies it to the vehicle kinematics. Nothing in
the loop is stubbed — the board is a genuine hardware-in-the-loop participant.

Camera modelled on the sensor the CARLA bridge configures: **640x480, 90 degree FOV,
2.4 m above the road**, focal length 320 px. Obstacle sprites are real photographs
(a drawn rectangle is not detectable by a COCO-trained network, and using one would
make the device under test look broken when it is not).

Grid: 3 speeds (30/50/70 km/h) x 2 start gaps (40/80 m) x 2 obstacle classes
(pedestrian/vehicle) x 2 light levels (daylight / x0.35).

## Outcome grid

| obstacle | light | start gap | 30 km/h | 50 km/h | 70 km/h |
|---|---|---|---|---|---|
| pedestrian | daylight | 40 m | avoided | **HIT 41.4** | **HIT 61.4** |
| pedestrian | daylight | 80 m | avoided | **HIT 35.6** | **HIT 64.2** |
| pedestrian | dusk | 40 m | avoided | **HIT 38.5** | **HIT 61.4** |
| pedestrian | dusk | 80 m | **HIT 9.0** | **HIT 38.5** | **HIT 64.2** |
| vehicle | daylight | 40 m | avoided | timeout | avoided |
| vehicle | daylight | 80 m | avoided | avoided | avoided |
| vehicle | dusk | 40 m | avoided | avoided | avoided |
| vehicle | dusk | 80 m | avoided | avoided | avoided |

Impact speeds in km/h.

**Failure attribution: 15/24 ok · 8/24 perception detected too late · 1/24 control
released the brake while still closing.** No failure was attributable to physics —
every collision was avoidable with a sensor that saw the obstacle in time.

## Perception

| | pedestrian | vehicle |
|---|---|---|
| First detection | **4.2 – 7.5 m** | 25.6 – 32.8 m |
| Detection rate | **0.13** | 0.62 |
| Distance error | **+53%** | **-37%** |
| Detection dropouts | 8 | 10 |
| Brake releases | 5 | 19 |

Inference latency: mean **20.6 ms**, worst frame 25.5 ms. Compute is not the bottleneck.

## Findings

1. **Pedestrian AEB does not work at any speed tested.** There is no all-clear speed;
   collisions occur from 30 km/h up, worst impact **64.2 km/h** from 70 km/h. Detection
   at 4–7 m against a 23.6 m stopping distance at 70 km/h means the brake is applied far
   too late to matter. This is a **sensor configuration** limit, not a control-tuning
   one: a separate open-loop study measured 640x480 at 90 degrees detecting a pedestrian
   at **no distance between 8 and 60 m**, while 1280x960 or a 50 degree FOV reached 12 m.
   Even 12 m is below the 12.1 m stopping distance at 50 km/h.
2. **The vehicle cases pass on paper only.** All-clear at every speed, yet **4 of those
   "avoided" runs stopped within 2 m, the worst with 0.7 m to spare at 70 km/h.**
3. **The monocular distance estimator is wrong in both directions** — +53% for
   pedestrians, -37% for vehicles. One constant is used for all classes; it is
   calibrated for a 1.75 m pedestrian while a 3.2 m vehicle needs roughly 1024.
   Per-class heights would fix the sign and most of the magnitude.
4. **24 brake releases across 24 runs.** A single frame without a detection snaps the
   distance signal back to the no-obstacle sentinel and the vehicle app cancels an
   active emergency stop. It fires in runs that pass as well as runs that fail. There is
   no tracking or hysteresis between perception and the brake law.
5. **Outcomes near the margin are nondeterministic.** Identical pedestrian conditions at
   30 km/h avoided in one run and collided at 9.0 km/h in another. Detection at 5.8 m
   against a 4.3 m stopping distance leaves ~1.5 m, and frame-timing jitter in the async
   loop is enough to flip the result.

## What this does not show

- **The dusk dimension is weak evidence.** Light level had no systematic effect — dusk
  detection was sometimes better than daylight. The renderer models low light as a
  linear brightness scale, with none of the sensor noise, gain, or motion blur that make
  real night driving hard. Do not read the light rows as a night-time result.
- **Synthetic imagery depresses absolute detection range** relative to CARLA or real
  camera footage. The relative ordering (resolution and FOV dominate) is the robust
  signal; the absolute metre figures are pessimistic.
- **The 50 km/h vehicle timeout did not reproduce.** A re-run of that exact scenario
  avoided cleanly, braking at 30% to a stop with 8.89 m spare. It is nondeterministic
  and currently unexplained.
- **The container image was never built.** Only the Docker build context was verified.
  The base image is l4t r36.2.0 against a board running R36.4.7.

## Prerequisite fixes

Before any of this could run, the committed pipeline had to be repaired: it published
**nothing**. Every node wrote at least one VSS path that exists in neither VSS 4.0 nor
5.1, the databroker rejects the entire SetRequest when one entry is invalid, and no node
read `resp.errors` — while `Set()` does not raise. The dead paths were
`ObstacleDetection.IsActive`, `Powertrain.Accelerator.PedalPosition` (it is under
`Chassis`), and `CurrentLocation.Speed`. The AEB app also read `Chassis.Brake.PedalPosition`
as a distance hint while publishing its own brake command to it — feeding on its output.

## Reproduce

```bash
# on the Jetson: databroker with the custom ADAS signal overlay, then the DUT nodes
docker run -d --name kuksa-databroker --network vsdv-net -p 55555:55555 \
  -v ~/vsdv-adas/agent/vss_overlay_adas.json:/etc/vss/adas_overlay.json:ro \
  ghcr.io/eclipse-kuksa/kuksa-databroker:latest \
  --address 0.0.0.0 --port 55555 --insecure \
  --vss /vss_release_5.1.json,/etc/vss/adas_overlay.json
python3 ~/vsdv-adas/agent/make_sprites.py      # sprites are generated, not committed
~/vsdv-adas/agent/dut_ctl.sh start

# on the host
cd ~/vsdv-adas/agent
python3 sweep.py --speeds 30 50 70 --distances 40 80 \
                 --obstacles person vehicle --lights 1.0 0.35 --duration 16
python3 envelope.py
```

## Next

- Per-class object heights for the distance estimator
- Detection hysteresis or a tracker, so one dropped frame cannot cancel a stop
- Re-run the camera trade study in CARLA, where imagery is photorealistic
- Sweep the camera configuration in closed loop (needs the engine re-exported per resolution)
