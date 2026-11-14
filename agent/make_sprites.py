#!/usr/bin/env python3
"""
Build the obstacle sprite library used by the scenario renderer.

Run this on the Jetson (needs cv2 + ultralytics):
    python3 agent/make_sprites.py

The sprites are GENERATED, not committed. They are crops of third-party sample
imagery that ships with the local toolchain — Ultralytics' COCO8 sample set and
its bundled bus.jpg, both under GPL-3.0 — so redistributing them from this
repository would drag that licence along with it. Regenerating locally keeps
the harness reproducible without redistributing anything.

Why real photographs at all: a drawn rectangle is not detectable by a
COCO-trained network. Rendering one would make the device under test look
broken when it is not, so every sprite is verified to be detected, as the right
class, before it is written.
"""

import pathlib
import sys

import cv2
from ultralytics import YOLO

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else pathlib.Path(__file__).resolve().parent / "assets")
COCO8 = pathlib.Path("/home/bk/datasets/coco8")
ASSETS = pathlib.Path("/home/bk/.local/lib/python3.10/site-packages/ultralytics/assets")
WEIGHTS = pathlib.Path("/home/bk/vsdv-adas/adas/yolov8n.pt")

VEHICLE_CLASSES = {2, 5, 7}          # car, bus, truck
PERSON_CLASS = 0
DEVICE = "cpu"                        # one-off job; keeps the 8 GB GPU free


def crop_from_label(img_path, label_path, want_cls, pad=0.04):
    """Crop the largest instance of want_cls using its YOLO-format label."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    best = None
    for line in label_path.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != want_cls:
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:5])
        if best is None or bw * bh > best[2] * best[3]:
            best = (cx, cy, bw, bh)
    if best is None:
        return None
    cx, cy, bw, bh = best
    x1, x2 = int(max(0, (cx - bw / 2 - pad) * W)), int(min(W, (cx + bw / 2 + pad) * W))
    y1, y2 = int(max(0, (cy - bh / 2 - pad) * H)), int(min(H, (cy + bh / 2 + pad) * H))
    return img[y1:y2, x1:x2]


def crop_vehicle(model, img_path):
    """
    Crop a vehicle from bus.jpg.

    The bus's own box also contains the pedestrians standing in front of it, so
    the full box verifies as 'person'. Walk the bottom edge up until a vehicle
    class wins the crop.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    res = model(img, verbose=False, device=DEVICE)[0]
    cands = [(b, c, f) for b, c, f in
             zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf)
             if int(c) in VEHICLE_CLASSES]
    if not cands:
        return None
    box = max(cands, key=lambda t: float(t[2]))[0]
    x1, y1, x2, y2 = (int(v) for v in box)
    for keep in (1.0, 0.85, 0.75, 0.65, 0.55, 0.45):
        crop = img[y1:y1 + int((y2 - y1) * keep), x1:x2]
        r = model(crop, verbose=False, imgsz=640, device=DEVICE)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        i = int(r.boxes.conf.argmax())
        if int(r.boxes.cls[i]) in VEHICLE_CLASSES:
            return crop
    return None


def verify(model, path, want):
    """A sprite is only useful if the detector fires on it, as the right class."""
    res = model(cv2.imread(str(path)), verbose=False, imgsz=640, device=DEVICE)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return False, "NO DETECTION"
    i = int(res.boxes.conf.argmax())
    cls, conf = int(res.boxes.cls[i]), float(res.boxes.conf[i])
    ok = (cls == PERSON_CLASS) if want == "person" else (cls in VEHICLE_CLASSES)
    return ok, f"{model.names[cls]} conf={conf:.2f}"


def main():
    for p in (COCO8, ASSETS, WEIGHTS):
        if not p.exists():
            sys.exit(f"missing prerequisite: {p}")
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(WEIGHTS))

    made = []
    person = crop_from_label(COCO8 / "images/val/000000000036.jpg",
                             COCO8 / "labels/val/000000000036.txt", PERSON_CLASS)
    if person is not None:
        cv2.imwrite(str(OUT / "person.png"), person)
        made.append(("person.png", "person"))

    vehicle = crop_vehicle(model, ASSETS / "bus.jpg")
    if vehicle is not None:
        cv2.imwrite(str(OUT / "vehicle.png"), vehicle)
        made.append(("vehicle.png", "vehicle"))

    print(f"wrote {len(made)} sprites to {OUT}\n")
    failed = 0
    for name, want in made:
        img = cv2.imread(str(OUT / name))
        ok, what = verify(model, OUT / name, want)
        failed += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'BAD '} {name:<14} "
              f"{img.shape[1]}x{img.shape[0]}px  detected as: {what}")
    if failed or len(made) < 2:
        sys.exit("sprite generation failed — the renderer needs person.png and vehicle.png")


if __name__ == "__main__":
    main()
