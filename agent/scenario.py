#!/usr/bin/env python3
"""
Scenario specification, kinematics and rendering for the vSDV scenario-search agent.

The renderer is deliberately *physically* calibrated rather than tuned to flatter
the device under test: apparent object height follows a pinhole model for the same
camera the CARLA bridge configures (640x480, 90 deg horizontal FOV, 2.4 m above the
road). Any disagreement between the DUT's monocular distance estimate and the
scenario's ground truth is therefore a finding about the DUT, not a rendering artefact.

Frames are emitted on the exact wire format carla_kuksa_bridge.py uses
(4-byte big-endian length prefix + JPEG), so the Jetson ADAS node cannot tell this
apart from CARLA and needs no modification.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field, asdict

from PIL import Image, ImageDraw, ImageFilter

# Default camera — matches the sensor carla_kuksa_bridge.py configures
DEFAULT_CAMERA_W, DEFAULT_CAMERA_H = 640, 480
DEFAULT_CAMERA_FOV_DEG = 90.0
CAMERA_HEIGHT_M = 2.4

# Physical obstacle heights (m) — real-world size of each sprite
OBSTACLE_HEIGHT_M = {"person": 1.75, "vehicle": 3.20}


@dataclass(frozen=True)
class Camera:
    """
    Pinhole camera. Resolution and FOV are first-class scenario parameters
    because they bound the achievable detection range for a given network:
    apparent object height is focal * H_real / distance, and a detector has a
    floor on the pixel height it can resolve.
    """
    width: int = DEFAULT_CAMERA_W
    height: int = DEFAULT_CAMERA_H
    fov_deg: float = DEFAULT_CAMERA_FOV_DEG
    mount_height_m: float = CAMERA_HEIGHT_M

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.fov_deg) / 2)

    @property
    def horizon_y(self) -> int:
        return int(self.height * 0.45)

    def object_height_px(self, obstacle: str, distance_m: float) -> float:
        return self.focal_px * OBSTACLE_HEIGHT_M[obstacle] / max(0.1, distance_m)

    def label(self) -> str:
        return f"{self.width}x{self.height}@{self.fov_deg:.0f}deg"


MAX_DECEL_MS2 = 8.0        # full-brake deceleration on dry asphalt
G = 9.81


@dataclass
class ScenarioSpec:
    """One point in the scenario search space."""
    ego_speed_kmh: float = 50.0
    obstacle: str = "person"          # "person" | "vehicle"
    initial_distance_m: float = 80.0
    lead_speed_kmh: float = 0.0       # 0 = static obstacle
    light_level: float = 1.0          # 1.0 = full daylight, 0.15 = night
    fog_density: float = 0.0          # 0 = clear, 1 = dense
    fps: int = 10
    max_duration_s: float = 20.0
    friction: float = 1.0             # road friction scale on braking authority
    camera_width: int = DEFAULT_CAMERA_W
    camera_height: int = DEFAULT_CAMERA_H
    camera_fov_deg: float = DEFAULT_CAMERA_FOV_DEG

    def camera(self) -> "Camera":
        return Camera(self.camera_width, self.camera_height, self.camera_fov_deg)

    def scenario_id(self) -> str:
        return (f"{self.obstacle}_v{self.ego_speed_kmh:.0f}_d{self.initial_distance_m:.0f}"
                f"_lead{self.lead_speed_kmh:.0f}_light{self.light_level:.2f}"
                f"_fog{self.fog_density:.2f}_mu{self.friction:.2f}")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EgoState:
    """Mutable world state advanced one tick at a time."""
    speed_ms: float
    gap_m: float
    lead_speed_ms: float
    t: float = 0.0
    brake_pct: float = 0.0
    min_gap_m: float = field(default=float("inf"))

    def step(self, dt: float, brake_pct: float, friction: float) -> None:
        """Apply the DUT's brake command, then advance the kinematics."""
        self.brake_pct = brake_pct
        decel = (brake_pct / 100.0) * MAX_DECEL_MS2 * friction
        self.speed_ms = max(0.0, self.speed_ms - decel * dt)
        closing = self.speed_ms - self.lead_speed_ms
        self.gap_m -= closing * dt
        self.min_gap_m = min(self.min_gap_m, self.gap_m)
        self.t += dt

    @property
    def ttc_s(self) -> float:
        closing = self.speed_ms - self.lead_speed_ms
        if closing <= 0.01 or self.gap_m <= 0:
            return 999.0
        return self.gap_m / closing


def stopping_distance_m(speed_kmh: float, friction: float = 1.0,
                        reaction_s: float = 0.0) -> float:
    """Ideal stopping distance — the physical floor no AEB can beat."""
    v = speed_kmh / 3.6
    return v * reaction_s + (v * v) / (2 * MAX_DECEL_MS2 * friction)


# ── rendering ─────────────────────────────────────────────────────────────────

class Renderer:
    """Draws a road scene with one obstacle at a given true distance."""

    def __init__(self, assets_dir, camera: "Camera" | None = None,
                 jpeg_quality: int = 80):
        from pathlib import Path
        self.assets = Path(assets_dir)
        self.camera = camera or Camera()
        self.jpeg_quality = jpeg_quality
        self._sprites: dict[str, Image.Image] = {}

    def sprite(self, name: str) -> Image.Image:
        if name not in self._sprites:
            path = self.assets / f"{name}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"sprite {path} missing — run make_sprites.py first")
            self._sprites[name] = Image.open(path).convert("RGB")
        return self._sprites[name]

    def _background(self, light: float) -> Image.Image:
        W, H, HORIZON = self.camera.width, self.camera.height, self.camera.horizon_y
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        # sky: vertical gradient
        for y in range(HORIZON):
            f = y / max(1, HORIZON)
            d.line([(0, y), (W, y)],
                   fill=(int((110 + 80 * f) * light),
                         int((140 + 70 * f) * light),
                         int((200 + 30 * f) * light)))
        # road: darker toward the horizon
        for y in range(HORIZON, H):
            f = (y - HORIZON) / max(1, H - HORIZON)
            g = int((55 + 45 * f) * light)
            d.line([(0, y), (W, y)], fill=(g, g, g))
        # perspective lane markings
        cx = W // 2
        for side in (-1, 1):
            d.line([(cx + side * 18, H), (cx + side * 2, HORIZON)],
                   fill=(int(225 * light),) * 3, width=3)
            d.line([(cx + side * 300, H), (cx + side * 42, HORIZON)],
                   fill=(int(205 * light),) * 3, width=2)
        return img

    def render(self, obstacle: str, distance_m: float,
               light_level: float = 1.0, fog_density: float = 0.0) -> Image.Image:
        """Scene with `obstacle` at `distance_m` metres ahead."""
        light = max(0.05, min(1.0, light_level))
        img = self._background(light)

        if distance_m > 0.5:
            h_real = OBSTACLE_HEIGHT_M[obstacle]
            h_px = self.camera.focal_px * h_real / distance_m
            if h_px >= 2:
                spr = self.sprite(obstacle)
                aspect = spr.width / spr.height
                w_px = max(2, int(h_px * aspect))
                h_px_i = max(2, int(h_px))
                # ground contact point for a camera self.camera.mount_height_m above the road
                y_ground = self.camera.horizon_y + self.camera.focal_px * self.camera.mount_height_m / distance_m
                y_top = int(y_ground - h_px_i)
                x_left = int(self.camera.width / 2 - w_px / 2)
                if y_top < self.camera.height and y_ground > 0:
                    resized = spr.resize((w_px, h_px_i), Image.LANCZOS)
                    if light < 0.999:
                        resized = Image.eval(resized, lambda p: int(p * light))
                    if fog_density > 0:
                        # distance-dependent contrast loss (Koschmieder)
                        vis = max(1.0, 120.0 * (1.0 - fog_density))
                        t = math.exp(-distance_m / vis)
                        grey = Image.new("RGB", resized.size,
                                         (int(190 * light),) * 3)
                        resized = Image.blend(grey, resized, max(0.03, t))
                        if fog_density > 0.4:
                            resized = resized.filter(
                                ImageFilter.GaussianBlur(fog_density * 1.2))
                    img.paste(resized, (x_left, y_top))

        if fog_density > 0:
            veil = Image.new("RGB", img.size, (int(185 * light),) * 3)
            img = Image.blend(img, veil, min(0.75, fog_density * 0.55))
        return img

    def jpeg(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return buf.getvalue()


def dut_estimate_m(obstacle: str, true_distance_m: float,
                   camera: "Camera" | None = None) -> float:
    """
    What adas_inference.py will estimate for this true distance.

    The DUT uses dist = (50 * 1.6) / box_height_px for every class, i.e. it
    assumes focal*height == 80. Reproduced here so the harness can report the
    estimator's bias without needing the DUT in the loop.
    """
    cam = camera or Camera()
    h_px = cam.object_height_px(obstacle, true_distance_m)
    return (50 * 1.6) / max(h_px, 1.0)


if __name__ == "__main__":
    import sys
    print(f"focal = {FOCAL_PX:.1f}px  horizon_y = {HORIZON_Y}  "
          f"frame = {FRAME_W}x{FRAME_H} fov = {CAMERA_FOV_DEG}deg\n")
    print(f"{'true d':>8} {'person h':>9} {'veh h':>8} "
          f"{'DUT est (person)':>17} {'DUT est (veh)':>14}")
    for d in (100, 60, 40, 30, 20, 15, 10, 7, 5, 3):
        hp = FOCAL_PX * OBSTACLE_HEIGHT_M['person'] / d
        hv = FOCAL_PX * OBSTACLE_HEIGHT_M['vehicle'] / d
        print(f"{d:>7}m {hp:>8.1f}p {hv:>7.1f}p "
              f"{dut_estimate_m('person', d):>16.2f}m {dut_estimate_m('vehicle', d):>13.2f}m")
    print(f"\nideal stopping distance @50km/h = {stopping_distance_m(50):.1f}m, "
          f"@100km/h = {stopping_distance_m(100):.1f}m")

    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        r = Renderer("/home/bk/vsdv-adas/agent/assets")
        img = r.render("person", 25.0)
        img.save(out)
        print(f"\nsample frame written to {out}")
