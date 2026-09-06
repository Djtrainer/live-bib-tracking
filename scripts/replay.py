#!/usr/bin/env python3
"""Replay a video through the pipeline offline and write results.

This is the measurement tool the project never had. It runs headless, with no
server and no browser, processes every frame by default, and produces:

  * ``<out>/results.csv``  -- one row per finisher, the thing you score
  * ``<out>/trace.jsonl``  -- per-frame detections, OCR reads and crossings
  * ``<out>/summary.json`` -- run statistics
  * ``<out>/debug.mp4``    -- annotated video, with --video

Determinism matters: timestamps come from the frame index, not the wall clock,
so two runs over the same file produce identical output and a change in results
is a change in behaviour rather than noise.

    python scripts/replay.py --video data/raw/race.mp4 --out data/results/baseline
    python scripts/replay.py --video data/raw/race.mp4 --out runs/tuned \
        --config config/race_cv.yaml --limit 5000 --video-out
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from race_cv.capture import VideoFileSource  # noqa: E402
from race_cv.config import Config  # noqa: E402
from race_cv.pipeline import FrameResult  # noqa: E402
from race_cv.run import build_pipeline, load_roster, setup_logging  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline replay of a race video through the CV pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the race video")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--roster", default=None, help="Start-list CSV")
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N frames (0 = whole file)"
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Skip this many frames first"
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Override pacing; 0 processes every frame for determinism",
    )
    parser.add_argument(
        "--video-out", action="store_true", help="Write an annotated debug.mp4"
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.load(args.config)
    config.pipeline.target_fps = args.target_fps
    config.sink.api_url = ""  # never post from a replay

    source = VideoFileSource(args.video, start_epoch=0.0)
    print(
        f"Replaying {args.video}: {source.frame_width}x{source.frame_height} "
        f"@ {source.fps:.2f} fps, {source.frame_count} frames"
    )

    events = []
    pipeline = build_pipeline(
        config, source, run_id="replay", roster=load_roster(args.roster),
        emit=events.append,
    )

    writer = None
    if args.video_out:
        writer = cv2.VideoWriter(
            str(out_dir / "debug.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, source.fps if args.target_fps == 0 else args.target_fps),
            (source.frame_width, source.frame_height),
        )

    trace_path = out_dir / "trace.jsonl"
    started = time.time()
    processed = 0

    from race_cv.overlay import annotate

    with trace_path.open("w", encoding="utf-8") as trace:

        def on_result(result: FrameResult) -> None:
            nonlocal processed
            processed += 1
            trace.write(
                json.dumps(
                    {
                        "frame": result.frame.index,
                        "capture_ts": round(result.frame.capture_ts, 4),
                        "people": [
                            {
                                "id": p.track_id,
                                "conf": round(p.conf, 3),
                                "xyxy": [round(c, 1) for c in p.xyxy],
                            }
                            for p in result.people
                        ],
                        "bibs": [
                            {"conf": round(b.conf, 3), "xyxy": [round(c, 1) for c in b.xyxy]}
                            for b in result.bibs
                        ],
                        "crossings": [c.track_id for c in result.crossings],
                        "events": [e.event_id for e in result.events],
                    }
                )
                + "\n"
            )
            if writer is not None:
                labels = {
                    p.track_id: (pipeline.voter.resolve(p.track_id).text or "?")
                    for p in result.people
                }
                writer.write(
                    annotate(
                        result,
                        pipeline.line,
                        pipeline.detector.roi,
                        pipeline.stats,
                        labels,
                        boundary=pipeline.boundary,
                    )
                )
            if processed % 200 == 0:
                elapsed = time.time() - started
                print(
                    f"  frame {result.frame.index} | processed {processed} | "
                    f"finishers {pipeline.stats.events_emitted} | "
                    f"{processed / max(elapsed, 1e-6):.1f} fps",
                    flush=True,
                )

        for frame in source.frames():
            if args.start_frame and frame.index < args.start_frame:
                continue
            if args.limit and processed >= args.limit:
                break
            pipeline.stats.frames_seen += 1
            if not pipeline.should_process(frame):
                pipeline.stats.frames_paced_out += 1
                continue
            on_result(pipeline.process(frame))

        pipeline.flush()
        pipeline.close()

    source.release()
    if writer is not None:
        writer.release()

    write_results_csv(out_dir / "results.csv", events)
    summary = {
        "video": str(args.video),
        "config": str(args.config),
        "stats": asdict(pipeline.stats),
        "processed_fps_wall": processed / max(time.time() - started, 1e-6),
        "finishers": len(events),
        "unknown_bibs": sum(1 for e in events if e.bib_number is None),
        "suppressed_first_seen_past": pipeline.stats.suppressed_first_seen_past,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nFinishers detected: {len(events)}")
    print(f"  unknown bib: {summary['unknown_bibs']}")
    print(f"  suppressed (first seen past line): {summary['suppressed_first_seen_past']}")
    print(f"  wrote {out_dir}/results.csv, trace.jsonl, summary.json")
    return 0


def write_results_csv(path: Path, events) -> None:
    """One row per finisher, ordered by crossing time."""
    ordered = sorted(events, key=lambda e: e.capture_ts)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "place",
                "bib",
                "capture_ts",
                "elapsed_s",
                "track_id",
                "ocr_votes",
                "ocr_score",
                "bib_locked",
                "in_roster",
                "interpolated",
            ]
        )
        first = ordered[0].capture_ts if ordered else 0.0
        for place, event in enumerate(ordered, 1):
            writer.writerow(
                [
                    place,
                    event.bib_number or "",
                    f"{event.capture_ts:.3f}",
                    f"{event.capture_ts - first:.3f}",
                    event.track_id,
                    event.ocr_votes,
                    f"{event.ocr_score:.3f}",
                    int(event.bib_locked),
                    int(event.in_roster),
                    int(event.interpolated),
                ]
            )


if __name__ == "__main__":
    raise SystemExit(main())
