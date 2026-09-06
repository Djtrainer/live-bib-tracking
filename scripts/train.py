#!/usr/bin/env python3
"""Train and evaluate bib-detection models across a small experiment matrix.

Replaces ``src/yolo_utils/train.py``, which hardcoded one run against
``config/yolo_config.yaml`` -- the config whose train split contained 5 usable
images. This trains against the rebuilt dataset from ``build_dataset.py``,
whose val split is held-out finish-line footage, and reports every run against
that same val so the numbers are comparable to each other and to the deployed
model.

Two things this deliberately does that the old script did not:

* **Trains at the resolution it will be deployed at.** The deployed CoreML
  model has a fixed 640x640 input, and finish-line bibs are a median 46px wide
  in a 1920px frame -- 15px once squeezed into 640. Running a 640-trained model
  at 1280 helps in the field but is out-of-distribution; training at the target
  size is the fix.
* **Exports at the trained size.** An export at a different imgsz than training
  throws away the point of training larger.

    python scripts/train.py --imgsz 960 --epochs 150
    python scripts/train.py --models yolo11n,yolo11s --imgsz 640,960,1280 --epochs 120
    python scripts/train.py --imgsz 960 --export coreml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_one(model_name, imgsz, args) -> dict:
    from ultralytics import YOLO

    tag = f"{model_name}_{imgsz}"
    started = time.time()
    model = YOLO(f"{model_name}.pt")
    model.train(
        data=args.data,
        imgsz=imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=0,
        project=args.project,
        name=tag,
        exist_ok=True,
        patience=args.patience,
        # Finish-line bibs are ~2.4x smaller than the training median, because
        # most labelled bibs come from DSLR and stock close-ups. Extra scale
        # jitter is the cheapest way to make the model see them small.
        scale=args.scale,
        mosaic=1.0,
        close_mosaic=15,
        plots=True,
        val=True,
    )
    train_seconds = time.time() - started

    # Re-validate explicitly so every row is measured the same way.
    metrics = model.val(
        data=args.data, imgsz=imgsz, device=args.device, split="val",
        verbose=False, plots=False,
    )
    row = {
        "model": model_name,
        "imgsz": imgsz,
        "epochs": args.epochs,
        "train_seconds": round(train_seconds, 1),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "person_mAP50": round(float(metrics.box.ap50[0]), 4),
        "bib_mAP50": round(float(metrics.box.ap50[1]), 4),
        "bib_precision": round(float(metrics.box.p[1]), 4),
        "bib_recall": round(float(metrics.box.r[1]), 4),
        "weights": str(Path(args.project) / tag / "weights" / "best.pt"),
    }

    if args.export:
        # Export at the trained size -- exporting at a different imgsz would
        # discard exactly what training larger bought.
        exported = YOLO(row["weights"]).export(
            format=args.export, imgsz=imgsz, nms=False, half=False
        )
        row["export"] = str(exported)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train bib detection models")
    parser.add_argument("--data", default="config/yolo_dataset.yaml")
    parser.add_argument("--models", default="yolo11n")
    parser.add_argument("--imgsz", default="960")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu', 'mps'")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--scale", type=float, default=0.7)
    parser.add_argument("--project", default="models")
    parser.add_argument("--export", default=None, choices=[None, "coreml", "onnx"])
    parser.add_argument("--json", default="training_results.json")
    args = parser.parse_args(argv)

    if not Path(args.data).exists():
        print(f"{args.data} not found -- run scripts/build_dataset.py first", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.imgsz.split(",") if s.strip()]

    rows = []
    for model_name in models:
        for imgsz in sizes:
            print(f"\n{'=' * 70}\n=== {model_name} @ {imgsz}px, {args.epochs} epochs\n{'=' * 70}")
            try:
                rows.append(run_one(model_name, imgsz, args))
            except Exception as exc:
                print(f"FAILED {model_name}@{imgsz}: {type(exc).__name__}: {exc}")
                rows.append({"model": model_name, "imgsz": imgsz, "error": str(exc)})
            Path(args.json).write_text(json.dumps(rows, indent=2))

    print(f"\n{'=' * 78}\nRESULTS (val = held-out finish-line footage)\n{'=' * 78}")
    header = f"{'model':<10}{'imgsz':>6}{'mAP50':>8}{'mAP50-95':>10}{'bib mAP50':>11}{'bib R':>8}{'train':>9}"
    print(header)
    for r in sorted(
        [r for r in rows if "error" not in r], key=lambda r: -r["bib_mAP50"]
    ):
        print(
            f"{r['model']:<10}{r['imgsz']:>6}{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}"
            f"{r['bib_mAP50']:>11.3f}{r['bib_recall']:>8.3f}{r['train_seconds'] / 60:>8.1f}m"
        )
    for r in [r for r in rows if "error" in r]:
        print(f"{r['model']:<10}{r['imgsz']:>6}   FAILED: {r['error'][:60]}")
    print(f"\nwrote {args.json}")
    print("\nBaseline to beat (deployed best.pt @640 on this same val):")
    print("  mAP50 0.917   bib mAP50 0.870   bib recall 0.833")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
