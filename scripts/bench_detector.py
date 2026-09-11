#!/usr/bin/env python3
"""Per-frame cost of the detector and of the whole frame loop, on one clip.

Neither existing number isolates the detector. ``replay.py``'s fps includes
decode, and the health line's ``processed (N fps)`` on a *file* source is
the file's own frame rate whenever pacing is off: capture timestamps are
synthesised from the frame index, so ``frames_processed / wall_span`` is
30 fps no matter how fast the loop ran. This measures, over a fixed window
of one clip, with the race configuration and the OCR worker running:

  * ``decode``   -- ``cap.read()`` alone (HEVC through OpenCV/FFmpeg on the CPU)
  * ``detector`` -- ``Detector.track()``: crop, letterbox, inference, ByteTrack
  * ``loop``     -- ``Pipeline.process()``: detector + boundary + OCR submit +
                    crossing logic, i.e. what a frame costs the frame loop

and prints median / p90 / mean milliseconds with the fps each implies.
``loop`` is the number to hold against the camera's frame interval.

    python scripts/bench_detector.py --config config/race_cv.jetson.yaml \
        --video "data/raw/Camo Recording 2025-10-04 14-48-12.mov" --start-frame 800 --limit 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from race_cv.capture import Frame, VideoFileSource  # noqa: E402
from race_cv.config import Config  # noqa: E402
from race_cv.detect import Detector  # noqa: E402
from race_cv.ocr import BibReader  # noqa: E402
from race_cv.pipeline import Pipeline  # noqa: E402
from race_cv.run import load_roster  # noqa: E402


def summarize(name: str, samples: list[float]) -> dict:
    arr = np.asarray(samples, dtype=np.float64) * 1000.0
    stats = {
        "n": int(arr.size),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "mean_ms": float(arr.mean()),
        "max_ms": float(arr.max()),
    }
    stats["fps_at_median"] = 1000.0 / stats["median_ms"] if stats["median_ms"] else 0.0
    stats["fps_at_mean"] = 1000.0 / stats["mean_ms"] if stats["mean_ms"] else 0.0
    print(
        f"  {name:<9} median {stats['median_ms']:6.1f} ms  p90 {stats['p90_ms']:6.1f}  "
        f"mean {stats['mean_ms']:6.1f}  max {stats['max_ms']:7.1f}   "
        f"-> {stats['fps_at_median']:5.1f} fps at median, {stats['fps_at_mean']:5.1f} at mean"
    )
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Per-frame detector and loop cost on one clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--roster", default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--limit", type=int, default=600, help="Frames to time")
    parser.add_argument("--warmup-frames", type=int, default=30,
                        help="Frames run before timing starts, on top of Detector.warmup()")
    parser.add_argument("--json", default=None, help="Append a JSON line here")
    parser.add_argument("--label", default=None, help="Free-text tag for the JSON line")
    args = parser.parse_args(argv)

    config = Config.load(args.config)
    config.pipeline.target_fps = 0.0
    config.sink.api_url = ""
    roster = load_roster(args.roster)

    source = VideoFileSource(args.video, start_epoch=0.0)
    width, height = source.frame_width, source.frame_height
    detector = Detector(config.model, config.roi, width, height)
    warm = detector.warmup(width, height)
    for warning in detector.warnings:
        print(f"CONFIG MISMATCH: {warning}")
    reader = BibReader(config.ocr) if config.ocr.enabled else None
    ocr_warm = reader.warmup() if reader is not None else 0.0
    pipeline = Pipeline(
        config=config, detector=detector, frame_width=width, frame_height=height,
        run_id="bench", bib_reader=reader, roster=roster, emit=lambda e: None,
    )

    # Time the detector from inside the loop, so "loop" and "detector" are
    # measured on the same frames under the same conditions.
    detector_times: list[float] = []
    inner_track = detector.track

    def timed_track(image):
        started = time.perf_counter()
        try:
            return inner_track(image)
        finally:
            detector_times.append(time.perf_counter() - started)

    detector.track = timed_track

    print(
        f"{args.video}: {width}x{height} @ {source.fps:.0f} fps | model {config.model.path} "
        f"imgsz {detector.imgsz} device {config.model.device} half {config.model.half} "
        f"| roi {'on' if detector.roi.enabled else 'off'} two_stage {config.model.two_stage} "
        f"| warm-up detector {warm:.1f}s ocr {ocr_warm:.1f}s"
    )

    decode_times: list[float] = []
    loop_times: list[float] = []
    cap = source.cap
    if args.start_frame:
        # Sequential decode keeps this honest for a codec with long GOPs.
        for _ in range(args.start_frame):
            if not cap.grab():
                break
    index = args.start_frame
    timed = 0
    wall_started = None
    while timed < args.limit + args.warmup_frames:
        t0 = time.perf_counter()
        ok, image = cap.read()
        t1 = time.perf_counter()
        if not ok or image is None:
            break
        frame = Frame(image=image, capture_ts=index / source.fps, index=index)
        pipeline.stats.frames_seen += 1
        pipeline.process(frame)
        t2 = time.perf_counter()
        index += 1
        timed += 1
        if timed <= args.warmup_frames:
            detector_times.clear()
            continue
        if wall_started is None:
            wall_started = t0
        decode_times.append(t1 - t0)
        loop_times.append(t2 - t1)
    wall = time.perf_counter() - wall_started if wall_started else 0.0
    pipeline.flush()
    pipeline.close()
    source.release()

    n = len(loop_times)
    print(f"timed {n} frames from {args.start_frame + args.warmup_frames}: "
          f"wall {wall:.1f}s -> {n / wall if wall else 0:.1f} fps end to end (decode + loop)")
    result = {
        "label": args.label,
        "config": args.config,
        "video": Path(args.video).name,
        "model": config.model.path,
        "imgsz": list(detector.imgsz_wh),
        "roi": detector.roi.enabled,
        "two_stage": config.model.two_stage,
        "frames": n,
        "end_to_end_fps": n / wall if wall else 0.0,
        "decode": summarize("decode", decode_times),
        "detector": summarize("detector", detector_times[-n:]),
        "loop": summarize("loop", loop_times),
        "people_detections": pipeline.stats.people_detections,
        "bib_detections": pipeline.stats.bib_detections,
        "ocr_reads": pipeline.stats.ocr_reads,
        "ocr_dropped": pipeline.stats.ocr_dropped,
        "second_stage_bibs": pipeline.stats.second_stage_bibs,
        "two_stage_errors": pipeline.stats.two_stage_errors,
    }
    print(f"  people {result['people_detections']}  bibs {result['bib_detections']}  "
          f"ocr reads {result['ocr_reads']} dropped {result['ocr_dropped']}"
          + (f"  second-stage bibs {result['second_stage_bibs']}" if config.model.two_stage else "")
          + (f"  TWO-STAGE ERRORS {result['two_stage_errors']}" if result["two_stage_errors"] else ""))
    if args.json:
        with open(args.json, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
