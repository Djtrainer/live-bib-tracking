#!/usr/bin/env python3
"""Validate and export every trained run found on disk.

Salvage path for runs whose training succeeded but whose bookkeeping failed.
Finds best.pt under the runs tree, infers model+imgsz from the directory name,
validates each against the same held-out finish-line split, and exports at the
size it was trained at.
"""
import json, re, sys
from pathlib import Path
from ultralytics import YOLO

DATA = sys.argv[1] if len(sys.argv) > 1 else "config/yolo_dataset.yaml"
EXPORT = "coreml" if "--export" in sys.argv else None
rows = []
weights = sorted(Path.home().rglob("runs/detect/models/*/weights/best.pt"))
print(f"found {len(weights)} trained runs")

for w in weights:
    tag = w.parent.parent.name                     # e.g. yolo11n_960
    m = re.match(r"(?P<model>.+)_(?P<imgsz>\d+)$", tag)
    if not m:
        print(f"  skip {tag}: cannot infer imgsz"); continue
    imgsz = int(m.group("imgsz"))
    try:
        model = YOLO(str(w))
        r = model.val(data=DATA, imgsz=imgsz, device=0, split="val",
                      verbose=False, plots=False)
        row = {
            "model": m.group("model"), "imgsz": imgsz,
            "mAP50": round(float(r.box.map50), 4),
            "mAP50_95": round(float(r.box.map), 4),
            "person_mAP50": round(float(r.box.ap50[0]), 4),
            "bib_mAP50": round(float(r.box.ap50[1]), 4),
            "bib_precision": round(float(r.box.p[1]), 4),
            "bib_recall": round(float(r.box.r[1]), 4),
            "weights": str(w),
        }
        if EXPORT:
            row["export"] = str(YOLO(str(w)).export(
                format=EXPORT, imgsz=imgsz, nms=False, half=False))
        rows.append(row)
        print(f"  {tag}: mAP50={row['mAP50']:.3f} bib={row['bib_mAP50']:.3f} R={row['bib_recall']:.3f}")
    except Exception as e:
        print(f"  {tag} FAILED: {type(e).__name__}: {e}")

Path("evaluation_results.json").write_text(json.dumps(rows, indent=2))
print(f"\n{'model':<10}{'imgsz':>6}{'mAP50':>8}{'mAP50-95':>10}{'bib mAP50':>11}{'bib P':>8}{'bib R':>8}")
for r in sorted(rows, key=lambda r: -r["bib_mAP50"]):
    print(f"{r['model']:<10}{r['imgsz']:>6}{r['mAP50']:>8.3f}{r['mAP50_95']:>10.3f}"
          f"{r['bib_mAP50']:>11.3f}{r['bib_precision']:>8.3f}{r['bib_recall']:>8.3f}")
print("\nbaseline (deployed best.pt @640, same val): mAP50 0.917  bib 0.870  bib R 0.833")
