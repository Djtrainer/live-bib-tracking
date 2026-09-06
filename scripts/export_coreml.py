#!/usr/bin/env python3
"""Export the trained weights to CoreML at the input size the pipeline runs.

Why this exists as a script rather than a one-liner in a README:

* ``models/`` is gitignored, so the deployed ``.mlpackage`` is not in the
  repo. Anyone setting up a new machine, or recovering from a lost export,
  needs a reproducible way to make it from ``best.pt``.

* ultralytics writes ``<name>.mlpackage`` NEXT TO the source ``.pt``. Exporting
  ``best.pt`` in place therefore overwrites ``best.mlpackage`` in the same
  directory -- which is how the deployed model was destroyed once during
  development. Every export here runs from a throwaway copy in a scratch
  directory, and only the finished result is copied to its destination.

* The size is a [height, width] pair, height first, because that is
  ultralytics' convention and ``config/race_cv.yaml`` uses the same one.
  Getting it backwards produces a model CoreML rejects on every frame.

The local ultralytics (8.1.x) predates YOLO11 and cannot load these weights.
Run this with a newer one in an isolated environment:

    python -m venv --system-site-packages /tmp/exportenv
    /tmp/exportenv/bin/pip install "ultralytics>=8.3"
    /tmp/exportenv/bin/python scripts/export_coreml.py --size 512 928

The resulting .mlpackage loads fine under the older ultralytics that runs
the race; CoreML models are self-contained.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export weights to CoreML")
    parser.add_argument(
        "--weights", default="models/gpu_runs/yolo11n_1280/weights/best.pt")
    parser.add_argument(
        "--size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=[512, 928],
        help="Input size, HEIGHT then WIDTH (ultralytics' ordering). "
             "Each must be a multiple of 32.")
    parser.add_argument(
        "--out", default=None,
        help="Destination .mlpackage (default: models/exports/rect_<W>x<H>.mlpackage)")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination")
    args = parser.parse_args(argv)

    height, width = args.size
    for v in (height, width):
        if v % 32:
            print(f"size {v} is not a multiple of 32", file=sys.stderr)
            return 2
    weights = Path(args.weights)
    if not weights.exists():
        print(f"weights not found: {weights}", file=sys.stderr)
        return 1
    out = Path(args.out or f"models/exports/rect_{width}x{height}.mlpackage")
    if out.exists() and not args.force:
        print(f"{out} exists; pass --force to replace it", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    # Never export in place: see the module docstring.
    with tempfile.TemporaryDirectory(prefix="coreml_export_") as scratch:
        staged = Path(scratch) / "m.pt"
        shutil.copy2(weights, staged)
        print(f"exporting {weights} at {width}x{height} (h={height}, w={width}) ...")
        produced = YOLO(str(staged)).export(
            format="coreml", imgsz=[height, width], verbose=False)
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(produced), str(out))

    # Confirm the artifact says what we asked for; a silent h/w swap here
    # would fail at the first frame on race day.
    try:
        import coremltools as ct

        spec = ct.models.MLModel(str(out), compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
        image = spec.description.input[0].type.imageType
        got = (int(image.width), int(image.height))
        if got != (width, height):
            print(f"export input is {got[0]}x{got[1]}, expected {width}x{height}",
                  file=sys.stderr)
            return 1
        print(f"wrote {out}  (input {got[0]}x{got[1]})")
    except ImportError:
        print(f"wrote {out}  (coremltools not available to verify input size)")
    print(f"\nconfig/race_cv.yaml:\n  model:\n    path: {out}\n    imgsz: [{height}, {width}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
