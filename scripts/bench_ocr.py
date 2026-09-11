#!/usr/bin/env python3
"""Time EasyOCR on real bib crops, the way the pipeline feeds them.

Synthetic crops lie: on the Mac a rendered "123" read in 21 ms and a real
bib crop in 48 ms, and the per-track OCR rate limit was tuned to the real
number. This takes crops from footage exactly as ``Pipeline._read_bib``
does -- the detector's bib boxes from a replay's ``trace.jsonl``, cut from
the full-resolution frame with ``crop_with_padding``, then
``BibReader.preprocess`` (grayscale, upscale to ``target_height``, CLAHE,
Otsu, padded to a width bucket) -- and times ``BibReader.read`` on each
after the same warm-up the race performs.

Also reported, because they decide the config levers:

  * the warm-up sweep's first-call cost per width bucket (on MPS these
    were 150-1000 ms kernel compiles, the reason ``width_buckets_px`` exists);
  * the same crops read *without* bucketing, so a new width mid-race can be
    costed on this backend;
  * the preprocess cost, which is CPU and on the OCR worker thread.

    python scripts/bench_ocr.py --config config/race_cv.jetson.yaml \
        --video "data/raw/<clip>.mov" --trace runs/<replay>/trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from race_cv.config import Config, OcrConfig  # noqa: E402
from race_cv.ocr import BibReader, crop_with_padding  # noqa: E402


def collect_crops(video: str, trace: Path, min_conf: float, padding: int,
                  max_crops: int) -> list[tuple[int, np.ndarray, float]]:
    """(frame index, crop copy, yolo conf) for bib boxes in the trace."""
    wanted: dict[int, list[tuple[list[float], float]]] = {}
    for line in trace.read_text().splitlines():
        row = json.loads(line)
        boxes = [(b["xyxy"], b["conf"]) for b in row.get("bibs", []) if b["conf"] >= min_conf]
        if boxes:
            wanted[row["frame"]] = boxes
    if not wanted:
        return []
    frames = sorted(wanted)
    # Spread the sample over the whole clip rather than the first N frames,
    # so it covers far and near, small and large, not one approach.
    step = max(1, len(frames) // max_crops)
    chosen = set(frames[::step][:max_crops])
    cap = cv2.VideoCapture(video)
    crops = []
    index = 0
    last = max(chosen)
    while index <= last:
        ok = cap.grab()
        if not ok:
            break
        if index in chosen:
            ok, image = cap.retrieve()
            if ok:
                for xyxy, conf in wanted[index]:
                    crop = crop_with_padding(image, tuple(xyxy), padding)
                    if crop.size:
                        crops.append((index, crop.copy(), conf))
        index += 1
    cap.release()
    return crops


def timed(fn, *args, **kwargs):
    started = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - started


def report(name: str, seconds: list[float]) -> dict:
    arr = np.asarray(seconds) * 1000.0
    stats = {"n": int(arr.size), "median_ms": float(np.median(arr)),
             "p90_ms": float(np.percentile(arr, 90)), "max_ms": float(arr.max()),
             "mean_ms": float(arr.mean())}
    print(f"  {name:<34} n={stats['n']:<4} median {stats['median_ms']:6.1f} ms  "
          f"p90 {stats['p90_ms']:6.1f}  max {stats['max_ms']:7.1f}  mean {stats['mean_ms']:6.1f}"
          f"  -> {1000.0 / stats['mean_ms']:.0f}/s sustained")
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Time EasyOCR on real bib crops")
    parser.add_argument("--video", required=True)
    parser.add_argument("--trace", required=True, help="trace.jsonl from scripts/replay.py")
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--max-crops", type=int, default=300)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    ocr = config.ocr
    crops = collect_crops(args.video, Path(args.trace), ocr.min_bib_yolo_conf,
                          ocr.crop_padding, args.max_crops)
    if not crops:
        print("no bib crops in the trace above min_bib_yolo_conf", file=sys.stderr)
        return 1
    widths = [c.shape[1] for _, c, _ in crops]
    heights = [c.shape[0] for _, c, _ in crops]
    print(f"{len(crops)} real bib crops from {Path(args.video).name}: "
          f"width {min(widths)}-{max(widths)} px (median {int(np.median(widths))}), "
          f"height {min(heights)}-{max(heights)} px (median {int(np.median(heights))})")

    reader = BibReader(ocr)
    started = time.perf_counter()
    easy = reader._ensure_reader()
    print(f"  easyocr.Reader construction {time.perf_counter() - started:.1f}s "
          f"(device {getattr(easy, 'device', '?')})")

    # The warm-up sweep, timed per width so first-call costs are visible.
    height = ocr.target_height
    print("  warm-up sweep, first call at each bucket width:")
    first_calls = {}
    for width in list(ocr.width_buckets_px):
        probe = np.full((height, width), 255, dtype=np.uint8)
        cv2.putText(probe, "123", (10, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 5)
        _, dt = timed(easy.readtext, probe, allowlist="0123456789")
        _, dt2 = timed(easy.readtext, probe, allowlist="0123456789")
        first_calls[width] = (dt * 1000, dt2 * 1000)
        print(f"    width {width:>4}: first {dt * 1000:7.1f} ms, second {dt2 * 1000:6.1f} ms")

    pre_times, read_times, texts = [], [], []
    for _, crop, _ in crops:
        binary, dt_pre = timed(reader.preprocess, crop)
        (text, conf), dt_read = timed(reader.read, binary)
        pre_times.append(dt_pre)
        read_times.append(dt_read)
        texts.append((text, conf))
    print("bucketed (as the race runs):")
    pre = report("preprocess (CPU, worker thread)", pre_times)
    read = report("read, bucketed width", read_times)
    got = sum(1 for t, _ in texts if t)
    from collections import Counter
    top = Counter(t for t, _ in texts if t).most_common(5)
    print(f"  {got}/{len(texts)} crops produced digits; most common {top}")

    # Same crops, no bucketing: every unseen width is a fresh shape.
    loose = BibReader(OcrConfig(**{**vars(ocr), "width_buckets_px": []}))
    loose._reader = easy
    loose_times = []
    seen_widths = set()
    new_width_times = []
    for _, crop, _ in crops:
        binary = loose.preprocess(crop)
        _, dt = timed(loose.read, binary)
        loose_times.append(dt)
        if binary.shape[1] not in seen_widths:
            seen_widths.add(binary.shape[1])
            new_width_times.append(dt)
    print("unbucketed (arbitrary widths):")
    loose_stats = report("read, unbucketed width", loose_times)
    new_stats = report(f"  of which first-at-width ({len(seen_widths)} widths)", new_width_times)

    if args.json:
        Path(args.json).write_text(json.dumps({
            "video": Path(args.video).name, "config": args.config, "crops": len(crops),
            "preprocess": pre, "read_bucketed": read, "read_unbucketed": loose_stats,
            "first_at_new_width": new_stats, "warmup_first_calls_ms": first_calls,
            "digits_found": got,
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
