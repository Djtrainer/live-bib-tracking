#!/usr/bin/env python3
"""Pre-label unlabelled images with the detector, for human review in LabelMe.

Read this before trusting the output
------------------------------------
Labelling with the model you are trying to improve is circular. Boxes the
detector finds get written out fine; boxes it **misses** produce a file that
asserts "nothing here", and training on that actively teaches the model to keep
missing them. Bib recall is exactly this project's weak spot, so blanket
pseudo-labelling could easily make the model worse rather than better.

This script is therefore built to produce a *review queue*, not ground truth:

* It labels with the strongest configuration available -- highest resolution,
  a low confidence floor, and the two-stage person-crop pass -- so the teacher
  is meaningfully stronger than the model that will be trained on its output.
* It triages every image. Only images where the detections are coherent and
  confident are marked ``auto_ok``. Anything ambiguous is marked
  ``needs_review`` with the reason.
* The single most dangerous case gets its own flag: a person detected with no
  bib. That is either a racer genuinely without a visible number (common -- a
  third of the finishers in the smoke set are "No bib") or a missed bib. Only a
  human can tell, and guessing wrong in the second case poisons training.
* Output is LabelMe 5.5.0 JSON written beside each image, so the existing
  workflow opens it directly and review is correction rather than drawing.
* Existing labels are never overwritten.

    python scripts/autolabel.py --dry-run
    python scripts/autolabel.py --model models/gpu_runs/yolo11n_1280/weights/best.mlpackage --imgsz 1280
    python scripts/autolabel.py --only-domain finish_line --previews
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from race_cv.config import ModelConfig, RoiConfig  # noqa: E402
from race_cv.detect import Detector  # noqa: E402

PERSON, BIB = 0, 1
LABELME_VERSION = "5.5.0"


def find_unlabelled(source: Path) -> list[Path]:
    images: dict[str, Path] = {}
    labelled: set[str] = set()
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png"):
            images.setdefault(path.stem, path)
        elif suffix in (".txt", ".json"):
            labelled.add(path.stem)
    return sorted(p for stem, p in images.items() if stem not in labelled)


def domain_of(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if {"annotations_set2", "annotations_set3"} & parts:
        return "public"
    if {"race_2022", "race_2023"} & parts:
        return "race_photos"
    return "finish_line"


def to_labelme(image_path: Path, width: int, height: int, boxes) -> dict:
    """LabelMe 5.5.0 JSON matching the hand-labelled files in this repo.

    imageData is left null: LabelMe resolves the image via imagePath, and
    embedding base64 would multiply the file size by ~2.5MB per image for no
    benefit here.
    """
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": [
            {
                "label": str(cls),
                "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
            for cls, x1, y1, x2, y2 in boxes
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def triage(people, bibs, min_conf: float) -> tuple[str, list[str]]:
    """Decide whether this image is safe to train on unreviewed."""
    reasons = []
    if not people and not bibs:
        reasons.append("nothing detected -- an empty label claims 'background'")
    if people and not bibs:
        reasons.append(
            f"{len(people)} person(s), no bib -- genuinely bibless, or a missed bib?"
        )
    weak_people = [p for p in people if p.conf < min_conf]
    weak_bibs = [b for b in bibs if b.conf < min_conf]
    if weak_people:
        reasons.append(f"{len(weak_people)} person box(es) below {min_conf}")
    if weak_bibs:
        reasons.append(f"{len(weak_bibs)} bib box(es) below {min_conf}")
    if len(bibs) > len(people) and people:
        reasons.append("more bibs than people -- likely a duplicate or false box")
    return ("needs_review" if reasons else "auto_ok"), reasons


def draw_preview(image, people, bibs, status: str) -> np.ndarray:
    out = image.copy()
    for b in bibs:
        x1, y1, x2, y2 = (int(v) for v in b.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(out, f"bib {b.conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(out, f"person {p.conf:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 128, 0), 2)
    colour = (0, 200, 0) if status == "auto_ok" else (0, 165, 255)
    cv2.putText(out, status, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, colour, 4)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pre-label images for LabelMe review")
    parser.add_argument("--source", default="data/processed")
    parser.add_argument(
        "--model", default="models/gpu_runs/yolo11n_1280/weights/best.mlpackage"
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--conf", type=float, default=0.15,
        help="Low on purpose: a marginal box a human can delete beats a missed "
             "box that silently becomes a false negative",
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.50,
        help="Below this, an image is flagged for review rather than auto_ok",
    )
    parser.add_argument("--two-stage", action="store_true", default=True)
    parser.add_argument("--no-two-stage", dest="two_stage", action="store_false")
    parser.add_argument("--only-domain", default=None,
                        choices=[None, "finish_line", "race_photos", "public"])
    parser.add_argument("--previews", action="store_true",
                        help="Also write annotated JPEGs for fast visual triage")
    parser.add_argument("--preview-dir", default="data/autolabel_preview")
    parser.add_argument("--report", default="data/autolabel_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    targets = find_unlabelled(Path(args.source))
    if args.only_domain:
        targets = [p for p in targets if domain_of(p) == args.only_domain]
    if not targets:
        print("No unlabelled images found.")
        return 0

    print(f"unlabelled images: {len(targets)}")
    print(f"  by domain: {dict(Counter(domain_of(p) for p in targets))}")
    print(f"teacher: {args.model} @ {args.imgsz}px, conf={args.conf}, "
          f"two_stage={args.two_stage}")
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    if not Path(args.model).exists():
        print(f"model not found: {args.model}", file=sys.stderr)
        return 1

    detector = None
    rows, counts = [], Counter()
    preview_dir = Path(args.preview_dir)
    if args.previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for i, image_path in enumerate(targets, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable"] += 1
            rows.append({"image": str(image_path), "status": "unreadable"})
            continue
        height, width = image.shape[:2]

        # The detector is bound to a frame size, so rebuild it when that changes
        # -- these images come from several cameras at different resolutions.
        # Detection itself is stateless here (see .detect below): these are
        # unrelated stills, not video.
        if detector is None or detector._frame_size != (width, height):
            detector = Detector(
                ModelConfig(path=args.model, imgsz=args.imgsz, conf=args.conf,
                            device="cpu", two_stage=args.two_stage,
                            two_stage_imgsz=args.imgsz),
                RoiConfig(), width, height,
            )
            detector._frame_size = (width, height)

        # detect(), not track(): track() carries ByteTrack state across calls,
        # which for a folder of unrelated images means each photo's detections
        # get matched against the previous photo's tracks and silently dropped.
        detections = detector.detect(image)
        people = [d for d in detections if d.cls == PERSON]
        bibs = [d for d in detections if d.cls == BIB]
        if args.two_stage and people:
            bibs = detector.merge(bibs, detector.bibs_in_people(image, people))

        status, reasons = triage(people, bibs, args.min_conf)
        counts[status] += 1

        boxes = [(PERSON, *d.xyxy) for d in people] + [(BIB, *d.xyxy) for d in bibs]
        out_json = image_path.with_suffix(".json")
        if out_json.exists():
            counts["skipped_existing"] += 1
            continue
        out_json.write_text(
            json.dumps(to_labelme(image_path, width, height, boxes), indent=2)
        )

        if args.previews:
            tag = "review" if status == "needs_review" else "ok"
            cv2.imwrite(
                str(preview_dir / f"{tag}__{image_path.stem}.jpg"),
                draw_preview(image, people, bibs, status),
            )

        rows.append({
            "image": str(image_path),
            "json": str(out_json),
            "domain": domain_of(image_path),
            "status": status,
            "reasons": reasons,
            "people": len(people),
            "bibs": len(bibs),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(targets)}...", flush=True)

    Path(args.report).write_text(json.dumps(rows, indent=2))

    print(f"\n{'=' * 60}")
    print(f"auto_ok       {counts['auto_ok']:>5}   coherent + confident")
    print(f"needs_review  {counts['needs_review']:>5}   open these in LabelMe first")
    if counts["unreadable"]:
        print(f"unreadable    {counts['unreadable']:>5}")
    print(f"{'=' * 60}")
    reason_counts = Counter(r for row in rows for r in row.get("reasons", []))
    if reason_counts:
        print("why review was flagged:")
        for reason, n in reason_counts.most_common():
            print(f"  {n:>4}  {reason}")
    print(f"\nwrote LabelMe JSON beside each image; report at {args.report}")
    if args.previews:
        print(f"previews in {preview_dir}/ (prefixed review__ / ok__)")
    print(
        "\nThese are a review queue, not ground truth. In particular, every\n"
        "'person but no bib' image needs a human decision: leaving a missed bib\n"
        "unlabelled trains the model to keep missing it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
