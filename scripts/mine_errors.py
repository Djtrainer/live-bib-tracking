#!/usr/bin/env python3
"""Diff the detector against human ground truth to find what it actually misses.

Every other measurement in this project is indirect. Val mAP compresses
everything into one number, and the smoke test only sees whichever frames a
racer happens to appear in. This compares predictions to hand-drawn boxes
one at a time, so the failures can be characterised exactly: which objects are
missed, at what size, in which domain, and on which images.

That matters because the obvious next investment -- label more data -- is only
worth making if the failures cluster somewhere labelling would help. If the
detector misses small bibs specifically, more small-bib labels fix it. If it
misses them uniformly at all sizes, the problem is capacity or training, and
more labels of the same kind will not move it.

Matching is greedy by IoU within a class, which is the standard detection
convention and means a prediction only counts if it lands on the right object,
not merely somewhere in the right image.

    python scripts/mine_errors.py
    python scripts/mine_errors.py --model models/gpu_runs/yolo11n_1280/weights/best.mlpackage --imgsz 1280
    python scripts/mine_errors.py --two-stage --worst 30
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from race_cv.config import ModelConfig, RoiConfig  # noqa: E402
from race_cv.detect import Detector  # noqa: E402

PERSON, BIB = 0, 1
CLASS_NAME = {PERSON: "person", BIB: "bib"}


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_truth(label_path: Path, width: int, height: int):
    """YOLO-normalized label file -> absolute xyxy boxes."""
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:])
        boxes.append(
            (cls, (cx - w / 2) * width, (cy - h / 2) * height,
             (cx + w / 2) * width, (cy + h / 2) * height)
        )
    return boxes


def match(truth, predicted, threshold: float):
    """Greedy IoU matching within a class. Returns (hits, misses, spurious)."""
    unmatched = list(predicted)
    hits, misses = [], []
    for cls, *box in truth:
        best, best_iou = None, 0.0
        for candidate in unmatched:
            if candidate.cls != cls:
                continue
            score = iou(box, candidate.xyxy)
            if score > best_iou:
                best, best_iou = candidate, score
        if best is not None and best_iou >= threshold:
            unmatched.remove(best)
            hits.append((cls, box, best, best_iou))
        else:
            misses.append((cls, box, best_iou))
    return hits, misses, unmatched


def bucket(width_px: float) -> str:
    for edge, name in ((25, "tiny <25px"), (50, "small 25-50"),
                       (100, "medium 50-100"), (200, "large 100-200")):
        if width_px < edge:
            return name
    return "huge >200px"


BUCKET_ORDER = ["tiny <25px", "small 25-50", "medium 50-100",
                "large 100-200", "huge >200px"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Find what the detector misses")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument(
        "--model", default="models/gpu_runs/yolo11n_1280/weights/best.mlpackage")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU for a prediction to count as matching a truth box")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--worst", type=int, default=20,
                        help="How many worst images to list for relabelling")
    parser.add_argument("--json", default="data/error_analysis.json")
    args = parser.parse_args(argv)

    root = Path(args.dataset)
    pairs = []
    for split in ("train", "val"):
        for image in sorted((root / "images" / split).iterdir()):
            label = root / "labels" / split / f"{image.stem}.txt"
            if label.exists():
                pairs.append((split, image, label))
    if not pairs:
        print(f"no images under {root} -- run scripts/build_dataset.py", file=sys.stderr)
        return 1

    print(f"{len(pairs)} labelled images | model={args.model} @ {args.imgsz}px "
          f"conf={args.conf} two_stage={args.two_stage}")

    detector = None
    frame_size = None
    per_image = []
    hits_by = Counter()
    miss_by = Counter()
    fp_by = Counter()
    miss_sizes = defaultdict(list)
    hit_sizes = defaultdict(list)
    miss_by_split = Counter()
    total_truth = Counter()

    for i, (split, image_path, label_path) in enumerate(pairs, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        if detector is None or frame_size != (width, height):
            detector = Detector(
                ModelConfig(path=args.model, imgsz=args.imgsz, conf=args.conf,
                            device="cpu", two_stage=args.two_stage,
                            two_stage_imgsz=args.imgsz),
                RoiConfig(), width, height)
            frame_size = (width, height)

        predicted = detector.detect(image)
        if args.two_stage:
            people = [d for d in predicted if d.cls == PERSON]
            predicted = detector.merge(
                predicted, detector.bibs_in_people(image, people))

        truth = load_truth(label_path, width, height)
        hits, misses, spurious = match(truth, predicted, args.iou)

        for cls, *_ in truth:
            total_truth[cls] += 1
        for cls, box, _, _ in hits:
            hits_by[cls] += 1
            hit_sizes[cls].append(box[2] - box[0])
        for cls, box, best_iou in misses:
            miss_by[cls] += 1
            miss_sizes[cls].append(box[2] - box[0])
            miss_by_split[(split, cls)] += 1
        for d in spurious:
            fp_by[d.cls] += 1

        bib_missed = sum(1 for cls, *_ in misses if cls == BIB)
        per_image.append({
            "image": str(image_path), "split": split,
            "truth": len(truth), "hits": len(hits),
            "missed": len(misses), "missed_bibs": bib_missed,
            "false_positives": len(spurious),
        })
        if i % 50 == 0:
            print(f"  {i}/{len(pairs)}...", flush=True)

    print(f"\n{'=' * 68}\nDETECTION vs GROUND TRUTH  (IoU >= {args.iou})\n{'=' * 68}")
    print(f"{'class':<10}{'truth':>8}{'found':>8}{'missed':>9}{'recall':>9}"
          f"{'false pos':>11}")
    for cls in (PERSON, BIB):
        t = total_truth[cls]
        if not t:
            continue
        print(f"{CLASS_NAME[cls]:<10}{t:>8}{hits_by[cls]:>8}{miss_by[cls]:>9}"
              f"{hits_by[cls] / t:>8.1%}{fp_by[cls]:>11}")

    # The question that decides whether more labelling helps.
    print(f"\n{'=' * 68}\nBIB RECALL BY SIZE -- does it fail on small ones specifically?"
          f"\n{'=' * 68}")
    found_b = Counter(bucket(w) for w in hit_sizes[BIB])
    missed_b = Counter(bucket(w) for w in miss_sizes[BIB])
    print(f"{'bib width':<18}{'truth':>8}{'found':>8}{'missed':>9}{'recall':>9}")
    for name in BUCKET_ORDER:
        total = found_b[name] + missed_b[name]
        if not total:
            continue
        print(f"{name:<18}{total:>8}{found_b[name]:>8}{missed_b[name]:>9}"
              f"{found_b[name] / total:>8.1%}")
    if miss_sizes[BIB]:
        print(f"\nmissed bib width : median {stats.median(miss_sizes[BIB]):.0f}px")
    if hit_sizes[BIB]:
        print(f"found  bib width : median {stats.median(hit_sizes[BIB]):.0f}px")

    print(f"\nmisses by split: "
          f"{ {f'{s}/{CLASS_NAME[c]}': n for (s, c), n in sorted(miss_by_split.items())} }")

    worst = sorted(per_image, key=lambda r: (-r["missed_bibs"], -r["missed"]))[:args.worst]
    print(f"\n{'=' * 68}\nWORST IMAGES -- relabel or inspect these first\n{'=' * 68}")
    print(f"{'missed bibs':>12}{'missed':>8}{'FP':>5}  image")
    for r in worst:
        if r["missed"] == 0 and r["false_positives"] == 0:
            continue
        print(f"{r['missed_bibs']:>12}{r['missed']:>8}{r['false_positives']:>5}  "
              f"{Path(r['image']).name}")

    Path(args.json).write_text(json.dumps({
        "model": args.model, "imgsz": args.imgsz, "conf": args.conf,
        "iou": args.iou, "two_stage": args.two_stage,
        "totals": {CLASS_NAME[c]: {"truth": total_truth[c], "found": hits_by[c],
                                   "missed": miss_by[c], "false_positives": fp_by[c]}
                   for c in (PERSON, BIB)},
        "bib_recall_by_size": {n: {"found": found_b[n], "missed": missed_b[n]}
                               for n in BUCKET_ORDER},
        "per_image": per_image,
    }, indent=2))
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
