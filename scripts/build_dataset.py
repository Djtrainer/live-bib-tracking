#!/usr/bin/env python3
"""Rebuild the YOLO dataset with an honest, domain-aware train/val split.

Why this exists
---------------
The dataset under ``data/processed`` cannot train or evaluate anything
meaningful as it stands:

* ``images/train`` holds 18 images while ``labels/train`` holds 220 labels, and
  only **5** match by stem. YOLO pairs an image to its label by filename, so
  training against this layout *today* would see 5 images. (The deployed model
  scores 0.917 on held-out data, so it plainly trained on far more than that --
  the directories were reorganised after it was trained. The layout is
  unusable going forward either way.)
* 307 label files have no image; 266 images have no label.
* Labels contain class ids 2 and 3 (25 boxes) that ``nc: 2`` cannot accept.
* The bulk of the finish-line footage sat in ``images/val``, so the best data
  was being measured against rather than learned from.

Note on what was *not* wrong: the original split was temporally disjoint
(``images/train`` covers t=753-2285s, ``images/val`` t=8-597s, with no pair of
frames within 2s of each other), so its ``mAP50 = 0.941`` was not the
memorisation score it first appeared to be. Re-measuring the deployed model on
the rebuilt split gives 0.917, which is consistent with that. The value here is
recovering usable pairs from a broken layout, dropping invalid classes, and
making val domain-aware -- not fixing leakage that turned out not to exist.

What "ideal" means here
-----------------------
The three sources are not interchangeable, and treating them as one pool is
what makes a score meaningless:

* ``finish_line`` -- frames from the actual finish-line camera. This is the
  deployment domain, and the **only** one whose score predicts race day.
* ``race_photos`` -- DSLR shots of the same race. Real bibs, real runners, but
  a different camera, distance and angle.
* ``public``      -- public bib datasets, added to make detection robust.

So validation is drawn from ``finish_line`` only. The other two exist to teach
the model what a bib looks like, and measuring on them would answer a question
nobody is asking -- a great score on stock photos says nothing about whether
bib 121 gets read at 10am on race day.

Within the finish-line footage, frames arrive in bursts a quarter-second apart
with long gaps between them. The script cuts a new segment at every gap over
``--gap`` seconds and assigns whole segments, so near-duplicate frames can
never straddle the split while footage minutes apart still can.

    python scripts/build_dataset.py --dry-run
    python scripts/build_dataset.py --out data/dataset --val-fraction 0.25
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Only person and bib are trainable; 2 and 3 are a few stray dog / stroller
# boxes that nc=2 would reject outright.
KEEP_CLASSES = {0, 1}
CLASS_NAMES = ["person", "bib"]

TIMESTAMPED = re.compile(r"^frame_(\d+)_t([\d.]+)s$")
PLAIN_FRAME = re.compile(r"^frame_(\d+)$")

FINISH_LINE = "finish_line"   # the deployment domain
RACE_PHOTOS = "race_photos"   # same race, different camera
PUBLIC = "public"             # public bib datasets, for robustness

PUBLIC_DIRS = {"annotations_set2", "annotations_set3"}
RACE_PHOTO_DIRS = {"race_2022", "race_2023"}


def domain_of(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if PUBLIC_DIRS & parts:
        return PUBLIC
    if RACE_PHOTO_DIRS & parts:
        return RACE_PHOTOS
    return FINISH_LINE


@dataclass
class Sample:
    stem: str
    image: Path
    label_source: Path
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    domain: str = FINISH_LINE
    group: str = "unknown"

    @property
    def bib_count(self) -> int:
        return sum(1 for b in self.boxes if b[0] == 1)


def yolo_from_txt(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls not in KEEP_CLASSES:
            continue
        values = [float(v) for v in parts[1:]]
        if any(v < 0 or v > 1 for v in values):
            continue
        boxes.append((cls, *values))
    return boxes


def yolo_from_labelme(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Convert LabelMe rectangles to normalized YOLO boxes.

    Clamps to the image: a hand-drawn rectangle routinely runs a few pixels off
    the edge, and YOLO rejects out-of-range coordinates.
    """
    data = json.loads(path.read_text())
    width = float(data.get("imageWidth") or 0)
    height = float(data.get("imageHeight") or 0)
    if width <= 0 or height <= 0:
        return []
    boxes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        try:
            cls = int(float(shape.get("label")))
        except (TypeError, ValueError):
            continue
        if cls not in KEEP_CLASSES:
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        (x1, y1), (x2, y2) = points[0], points[1]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append(
            (
                cls,
                ((x1 + x2) / 2) / width,
                ((y1 + y2) / 2) / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    return boxes


def collect(source: Path) -> tuple[list[Sample], Counter]:
    """Every image that has a usable label, preferring .txt over .json.

    Labels are matched by stem across the whole tree, because the annotations
    for an image frequently live in a different directory than the image.
    """
    images: dict[str, Path] = {}
    txts: dict[str, Path] = {}
    jsons: dict[str, Path] = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            # Prefer the copy that sits next to its own annotations.
            if path.stem not in images or path.parent.name in PUBLIC_DIRS | RACE_PHOTO_DIRS:
                images.setdefault(path.stem, path)
        elif suffix == ".txt":
            txts.setdefault(path.stem, path)
        elif suffix == ".json":
            jsons.setdefault(path.stem, path)

    skipped = Counter()
    samples = []
    for stem, image in images.items():
        if stem in txts:
            boxes, origin = yolo_from_txt(txts[stem]), txts[stem]
        elif stem in jsons:
            boxes, origin = yolo_from_labelme(jsons[stem]), jsons[stem]
        else:
            skipped["no label"] += 1
            continue
        if not boxes:
            skipped["label had no usable boxes"] += 1
            continue
        samples.append(
            Sample(
                stem=stem,
                image=image,
                label_source=origin,
                boxes=boxes,
                domain=domain_of(image),
            )
        )
    return samples, skipped


def assign_groups(samples: list[Sample], gap_seconds: float, frame_gap: int) -> None:
    """Tag each sample with the source it must not be split away from."""
    timestamped: list[tuple[float, Sample]] = []
    plain: list[tuple[int, Sample]] = []

    for sample in samples:
        if sample.domain != FINISH_LINE:
            # Non-deployment domains are never split, so one group each is
            # enough to keep the bookkeeping uniform.
            sample.group = f"{sample.domain}_{sample.image.parent.name}"
            continue
        if (m := TIMESTAMPED.match(sample.stem)):
            timestamped.append((float(m.group(2)), sample))
        elif (m := PLAIN_FRAME.match(sample.stem)):
            plain.append((int(m.group(1)), sample))
        else:
            sample.group = f"finish_other_{sample.image.parent.name}"

    for series, gap, prefix in (
        (timestamped, gap_seconds, "video_seg"),
        (plain, frame_gap, "frames_seg"),
    ):
        series.sort(key=lambda pair: pair[0])
        segment = 0
        previous = None
        for value, sample in series:
            if previous is not None and (value - previous) > gap:
                segment += 1
            sample.group = f"{prefix}{segment:02d}"
            previous = value


def split_groups(
    samples: list[Sample], val_fraction: float
) -> tuple[set[str], set[str]]:
    """Choose val groups from the finish-line domain only.

    Everything else trains. Validation has to answer "will this work on race
    day", and only footage from the finish-line camera answers it.

    Groups are taken smallest-first (above a minimum size) rather than
    largest-first, which fills the target share with several distinct stretches
    of footage instead of one big one. With a dataset this small, a val set
    that is a single segment measures whether the model generalises to that one
    stretch -- a much narrower and noisier question than the one being asked.
    Segments below ``min_group`` are skipped as too small to be informative on
    their own.
    """
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group].append(sample)

    finish_groups = [
        g for g, members in by_group.items() if members[0].domain == FINISH_LINE
    ]
    finish_total = sum(len(by_group[g]) for g in finish_groups)
    target = finish_total * val_fraction
    min_group = 4

    val: set[str] = set()
    val_n = 0
    for group in sorted(finish_groups, key=lambda g: len(by_group[g])):
        size = len(by_group[group])
        if size < min_group or val_n >= target:
            continue
        if val_n + size > target * 1.15 and val_n > 0:
            continue  # would overshoot; try a smaller one
        val.add(group)
        val_n += size
    if not val and finish_groups:  # degenerate: take the largest rather than none
        val.add(max(finish_groups, key=lambda g: len(by_group[g])))
    train = set(by_group) - val
    return train, val


def write_split(samples: list[Sample], out: Path, split: str) -> None:
    image_dir = out / "images" / split
    label_dir = out / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        shutil.copy2(sample.image, image_dir / f"{sample.stem}{sample.image.suffix}")
        lines = [
            f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in sample.boxes
        ]
        (label_dir / f"{sample.stem}.txt").write_text("\n".join(lines) + "\n")


def summarize(name: str, samples: list[Sample]) -> None:
    by_domain = Counter(s.domain for s in samples)
    boxes = Counter(c for s in samples for c, *_ in s.boxes)
    print(
        f"  {name:<6} images={len(samples):<5} person={boxes.get(0, 0):<6} "
        f"bib={boxes.get(1, 0):<6} with_bib={sum(1 for s in samples if s.bib_count):<5} "
        f"{dict(by_domain)}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the YOLO dataset")
    parser.add_argument("--source", default="data/processed")
    parser.add_argument("--out", default="data/dataset")
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument(
        "--gap", type=float, default=30.0,
        help="Seconds between finish-line frames that starts a new segment",
    )
    parser.add_argument("--frame-gap", type=int, default=200)
    parser.add_argument("--config-out", default="config/yolo_dataset.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    samples, skipped = collect(Path(args.source))
    if not samples:
        print(f"No labelled images found under {args.source}", file=sys.stderr)
        return 1
    assign_groups(samples, args.gap, args.frame_gap)
    train_groups, val_groups = split_groups(samples, args.val_fraction)

    train = [s for s in samples if s.group in train_groups]
    val = [s for s in samples if s.group in val_groups]

    print(f"labelled images: {len(samples)}   (skipped: {dict(skipped)})")
    print("\nby domain:")
    for domain in (FINISH_LINE, RACE_PHOTOS, PUBLIC):
        members = [s for s in samples if s.domain == domain]
        boxes = Counter(c for s in members for c, *_ in s.boxes)
        print(
            f"  {domain:<12} images={len(members):<5} person={boxes.get(0, 0):<6} "
            f"bib={boxes.get(1, 0):<6} groups={len({s.group for s in members})}"
        )

    print("\nsplit:")
    summarize("train", train)
    summarize("val", val)

    assert not (train_groups & val_groups), "a group leaked across the split"
    assert all(s.domain == FINISH_LINE for s in val), "val must be finish-line only"
    print("\nchecks: no group spans both splits; val is finish-line only")
    if val and not any(s.bib_count for s in val):
        print("WARNING: val has no bib boxes -- bib metrics will be meaningless")

    if args.dry_run:
        print("\n--dry-run: nothing written\n")
        by_group = defaultdict(list)
        for s in samples:
            by_group[s.group].append(s)
        for group in sorted(by_group):
            side = "VAL  " if group in val_groups else "train"
            print(f"    {side} {group:<26}{len(by_group[group]):>4}  ({by_group[group][0].domain})")
        return 0

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    write_split(train, out, "train")
    write_split(val, out, "val")

    Path(args.config_out).write_text(
        "# Generated by scripts/build_dataset.py -- do not hand-edit.\n"
        "# val is drawn only from finish-line camera footage, split by time\n"
        "# segment, so the score predicts race day rather than measuring\n"
        "# memorisation of adjacent video frames or of stock bib photos.\n"
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"\nwrote {out}/  and  {args.config_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
