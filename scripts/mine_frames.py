#!/usr/bin/env python3
"""Pull the frames the detector gets wrong out of a recording, for labelling.

The replay trace already says where the detector fails: a runner's track
that has no detection on some frames of its approach, and frames where a
runner is tracked but no bib box lies inside them. Those frames are the
hard examples worth a human's time; the smoke test never sees them because
it only scores finishers. This runs the race detector over a clip (every
frame, deterministic), picks frames by why they matter, and writes them as
JPEGs with a manifest, into a directory ``scripts/autolabel.py`` and
``scripts/build_dataset.py`` will read like any other.

Selection, per clip (all rates are configurable):

* ``gap``      -- frames inside a tracked runner's span with no detection of
                  that runner at all: the detector missed the person. Every
                  ``--gap-stride``-th frame of each gap, so a 16-frame gap
                  yields ~5 frames rather than 16 near-duplicates.
* ``lowconf``  -- the runner was found, but below ``--lowconf`` (the band
                  ByteTrack lives on and the model is unsure in).
* ``nobib``    -- the runner was found and no bib box lay inside the person
                  box. Either the bib is hidden or the detector missed it;
                  autolabel flags exactly this case for review.
* ``easy``     -- one frame every ``--easy-every`` seconds while a runner is
                  tracked: the new domain's normal case, so training does
                  not only see the failures.
* ``empty``    -- one frame every ``--empty-every`` seconds with nobody
                  tracked: background for the detector to learn to ignore
                  (spectators, the treeline).

Run it with the *strongest* configuration available, not the race one: the
point is to find what even the best detector misses, and the labels a
human corrects afterwards do not depend on which model proposed them.

    python scripts/mine_frames.py --video "data/raw/<clip>.mov" --config config/race_cv.jetson.yaml
    python scripts/mine_frames.py --video "data/raw/<clip>.mov" --config <cfg> --out data/processed/mined_<name>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cv2  # noqa: E402

from race_cv.capture import Frame, VideoFileSource  # noqa: E402
from race_cv.config import Config  # noqa: E402
from race_cv.detect import Detector  # noqa: E402
from race_cv.pipeline import Pipeline  # noqa: E402


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def bib_inside(person, bibs) -> bool:
    x1, y1, x2, y2 = person.xyxy
    return any(x1 <= b.center[0] <= x2 and y1 <= b.center[1] <= y2 for b in bibs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mine hard frames from a clip for labelling")
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: data/processed/mined_<clip slug>)")
    parser.add_argument("--gap-stride", type=int, default=3, help="Keep every Nth frame of a gap")
    parser.add_argument("--lowconf", type=float, default=0.3)
    parser.add_argument("--lowconf-stride", type=int, default=5)
    parser.add_argument("--nobib-stride", type=int, default=6)
    parser.add_argument("--easy-every", type=float, default=2.0, help="seconds")
    parser.add_argument("--empty-every", type=float, default=15.0, help="seconds")
    parser.add_argument("--min-track-frames", type=int, default=30,
                        help="Ignore tracks shorter than this (fragments, spectators)")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N frames (0 = all)")
    args = parser.parse_args(argv)

    video = Path(args.video)
    out = Path(args.out or f"data/processed/mined_{slug(video.stem)}")
    config = Config.load(args.config)
    config.pipeline.target_fps = 0.0
    config.sink.api_url = ""
    config.ocr.enabled = False  # detection only: no OCR worker, no votes

    # Pass 1: run the detector over the clip and remember, per frame, who was
    # tracked at what confidence and whether a bib box sat inside them.
    source = VideoFileSource(video, start_epoch=0.0)
    fps = source.fps
    detector = Detector(config.model, config.roi, source.frame_width, source.frame_height)
    detector.warmup(source.frame_width, source.frame_height)
    for warning in detector.warnings:
        print(f"CONFIG MISMATCH: {warning}")
    pipeline = Pipeline(config=config, detector=detector, frame_width=source.frame_width,
                        frame_height=source.frame_height, run_id="mine", bib_reader=None,
                        roster=None, emit=lambda e: None)
    per_frame: dict[int, list[tuple[int, float, bool]]] = {}   # frame -> [(track, conf, bib_inside)]
    tracks: dict[int, list[int]] = defaultdict(list)
    started = time.time()
    n = 0
    for frame in source.frames():
        result = pipeline.process(frame)
        rows = [(p.track_id, p.conf, bib_inside(p, result.bibs)) for p in result.people]
        per_frame[frame.index] = rows
        for tid, _, _ in rows:
            tracks[tid].append(frame.index)
        n += 1
        if args.limit and n >= args.limit:
            break
    pipeline.close()
    source.release()
    print(f"pass 1: {n} frames in {time.time() - started:.0f}s, {len(tracks)} tracks")

    # Decide which frames to keep and why.
    reasons: dict[int, set[str]] = defaultdict(set)
    long_tracks = {t: fr for t, fr in tracks.items() if len(fr) >= args.min_track_frames}
    for tid, fr in long_tracks.items():
        prev = fr[0]
        for x in fr[1:]:
            if x - prev > 1:
                for g in range(prev + 1, x, args.gap_stride):
                    reasons[g].add("gap")
            prev = x
        for k, f in enumerate(fr):
            rows = [r for r in per_frame[f] if r[0] == tid]
            if not rows:
                continue
            _, conf, has_bib = rows[0]
            if conf < args.lowconf and k % args.lowconf_stride == 0:
                reasons[f].add("lowconf")
            if not has_bib and k % args.nobib_stride == 0:
                reasons[f].add("nobib")
    tracked_frames = sorted({f for fr in long_tracks.values() for f in fr})
    last_easy = -1e9
    for f in tracked_frames:
        if f - last_easy >= args.easy_every * fps:
            reasons[f].add("easy"); last_easy = f
    last_empty = -1e9
    tracked_set = set(tracked_frames)
    for f in range(n):
        if f in tracked_set:
            continue
        if f - last_empty >= args.empty_every * fps:
            reasons[f].add("empty"); last_empty = f
    chosen = sorted(reasons)
    counts = defaultdict(int)
    for f in chosen:
        for r in reasons[f]:
            counts[r] += 1
    print(f"selected {len(chosen)} frames: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    # Pass 2: write them. Sequential read again: seeking in a long-GOP HEVC
    # file lands on the wrong frame on some decoders.
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    wanted = set(chosen)
    manifest = []
    index = 0
    stem = slug(video.stem)
    while wanted:
        ok = cap.grab()
        if not ok:
            break
        if index in wanted:
            ok, image = cap.retrieve()
            if ok:
                name = f"{stem}_f{index:06d}.jpg"
                cv2.imwrite(str(out / name), image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                manifest.append({"file": name, "frame": index, "t": round(index / fps, 2),
                                 "why": sorted(reasons[index]),
                                 "people": [{"track": t, "conf": round(c, 3), "bib": b} for t, c, b in per_frame.get(index, [])]})
            wanted.discard(index)
        index += 1
    cap.release()
    (out / "mined_manifest.json").write_text(json.dumps({
        "video": str(video), "config": args.config, "fps": fps, "frames_scanned": n,
        "model": config.model.path, "imgsz": list(detector.imgsz_wh), "conf": config.model.conf,
        "counts": dict(counts), "frames": manifest,
    }, indent=1))
    print(f"wrote {len(manifest)} JPEGs and mined_manifest.json to {out}")
    print(f"next: python scripts/autolabel.py --source {out}   (then review in LabelMe; build_dataset.py picks the dir up)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
