#!/usr/bin/env python3
"""Run every clip in an expectations file and score the results.

This is the batch version of replay.py + score.py: one command that answers
"across all the footage we have, what does the pipeline actually get right?"

The expectations file lists one clip per line, with the finishers that should
be found in it:

    Camo Recording 2025-10-04 10-25-15: [(121, 29s)]
    Camo Recording 2025-10-04 14-48-12: [(120, 36s), (No bib, 122s), (120, 172s)]

`No bib` means a racer crossed but had no readable number -- the pipeline is
expected to report a finisher with an unresolved bib, not to invent one.

Times are seconds into the clip, which is what replay's capture_ts measures
(playback is anchored at 0.0), so they compare directly.

    python scripts/smoke_test.py --expected smoke_test.yaml
    python scripts/smoke_test.py --expected smoke_test.yaml --target-fps 0 --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from race_cv.capture import VideoFileSource  # noqa: E402
from race_cv.config import Config  # noqa: E402
from race_cv.detect import Detector  # noqa: E402
from race_cv.ocr import BibReader  # noqa: E402
from race_cv.pipeline import Pipeline  # noqa: E402
from race_cv.run import load_roster  # noqa: E402

NO_BIB = "No bib"
LINE = re.compile(r"^(?P<name>.+?):\s*\[(?P<pairs>.*)\]\s*$")
PAIR = re.compile(r"\(\s*(?P<bib>[^,()]+?)\s*,\s*(?P<seconds>[0-9.]+)\s*s\s*\)")


@dataclass
class Expected:
    bib: str | None  # None means "a finisher with no readable bib"
    seconds: float

    @property
    def label(self) -> str:
        return self.bib if self.bib else NO_BIB


@dataclass
class Detected:
    bib: str | None
    seconds: float


@dataclass
class Match:
    expected: Expected | None
    detected: Detected | None

    @property
    def is_missed(self) -> bool:
        return self.detected is None

    @property
    def is_ghost(self) -> bool:
        return self.expected is None

    @property
    def bib_correct(self) -> bool:
        if self.expected is None or self.detected is None:
            return False
        return (self.expected.bib or None) == (self.detected.bib or None)

    @property
    def drift(self) -> float | None:
        if self.expected is None or self.detected is None:
            return None
        return self.detected.seconds - self.expected.seconds


@dataclass
class ClipResult:
    name: str
    matches: list[Match] = field(default_factory=list)
    error: str | None = None
    wall_seconds: float = 0.0
    stats: dict = field(default_factory=dict)


def parse_expectations(path: Path) -> dict[str, list[Expected]]:
    """Parse the expectations file.

    Deliberately regex-based rather than yaml.safe_load: the file's `[(a, b)]`
    form is not valid YAML (bare parens, unquoted `No bib`), and YAML would
    silently mangle it into a flat list of strings rather than failing.
    """
    expectations: dict[str, list[Expected]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            raise ValueError(f"Could not parse expectations line: {raw!r}")
        finishers = []
        for pair in PAIR.finditer(match.group("pairs")):
            bib = pair.group("bib").strip()
            finishers.append(
                Expected(
                    bib=None if bib.lower().replace(" ", "") == "nobib" else bib,
                    seconds=float(pair.group("seconds")),
                )
            )
        if not PAIR.search(match.group("pairs")) and match.group("pairs").strip():
            raise ValueError(f"Could not parse any (bib, time) pairs in: {raw!r}")
        expectations[match.group("name").strip()] = finishers
    return expectations


def resolve_video(name: str, video_dir: Path) -> Path | None:
    for suffix in (".mov", ".MOV", ".mp4", ".MP4", ""):
        candidate = video_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def match_finishers(
    expected: list[Expected], detected: list[Detected], tolerance: float
) -> list[Match]:
    """Pair expected finishers with detected ones by closest time.

    Greedy nearest-in-time, one-to-one. Matching on time rather than bib is
    deliberate: a finisher detected at the right moment with the wrong number
    is a different (and much less bad) failure than one never detected at all,
    and the report needs to tell those apart.
    """
    remaining = sorted(detected, key=lambda d: d.seconds)
    matches: list[Match] = []
    for exp in sorted(expected, key=lambda e: e.seconds):
        best, best_gap = None, None
        for candidate in remaining:
            gap = abs(candidate.seconds - exp.seconds)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = candidate, gap
        if best is not None:
            remaining.remove(best)
        matches.append(Match(expected=exp, detected=best))
    matches.extend(Match(expected=None, detected=d) for d in remaining)
    return matches


def run_clip(
    video: Path, config: Config, reader: BibReader | None, roster: set[str]
) -> tuple[list[Detected], dict, float]:
    """Run one clip and return its finishers, stats and wall time.

    A fresh Detector per clip on purpose: model.track(persist=True) carries
    tracker state between calls, so reusing one across clips would leak track
    IDs and stale tracks from the previous video into the next.
    """
    started = time.time()
    source = VideoFileSource(video, start_epoch=0.0)
    detector = Detector(config.model, config.roi, source.frame_width, source.frame_height)
    detector.warmup(source.frame_width, source.frame_height)

    events = []
    pipeline = Pipeline(
        config=config,
        detector=detector,
        frame_width=source.frame_width,
        frame_height=source.frame_height,
        run_id="smoke",
        bib_reader=reader,
        roster=roster,
        emit=events.append,
    )
    pipeline.run(source.frames())
    source.release()

    detected = [Detected(bib=e.bib_number, seconds=e.capture_ts) for e in events]
    stats = {
        "frames_processed": pipeline.stats.frames_processed,
        "people_detections": pipeline.stats.people_detections,
        "people_outside_boundary": pipeline.stats.people_outside_boundary,
        "bib_detections": pipeline.stats.bib_detections,
        "ocr_reads": pipeline.stats.ocr_reads,
        "suppressed_first_seen_past": pipeline.stats.suppressed_first_seen_past,
    }
    return detected, stats, time.time() - started


def print_clip(result: ClipResult) -> None:
    print(f"\n=== {result.name} ===")
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    expected_n = sum(1 for m in result.matches if m.expected)
    detected_n = sum(1 for m in result.matches if m.detected)
    print(f"  expected {expected_n}, detected {detected_n}   ({result.wall_seconds:.0f}s)")
    for m in result.matches:
        if m.is_ghost:
            bib = m.detected.bib or NO_BIB
            print(f"  + GHOST                    -> {bib:>7} @ {m.detected.seconds:6.1f}s")
        elif m.is_missed:
            print(f"  - MISSED  {m.expected.label:>7} @ {m.expected.seconds:5.0f}s")
        elif m.bib_correct:
            print(
                f"  ok        {m.expected.label:>7} @ {m.expected.seconds:5.0f}s"
                f"  -> {(m.detected.bib or NO_BIB):>7} @ {m.detected.seconds:6.1f}s"
                f"  ({m.drift:+.1f}s)"
            )
        else:
            print(
                f"  ! BIB     {m.expected.label:>7} @ {m.expected.seconds:5.0f}s"
                f"  -> {(m.detected.bib or NO_BIB):>7} @ {m.detected.seconds:6.1f}s"
                f"  ({m.drift:+.1f}s)"
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Batch-score clips against expectations")
    parser.add_argument("--expected", default="smoke_test.yaml")
    parser.add_argument("--videos", default="data/raw", help="Directory holding the clips")
    parser.add_argument("--config", default="config/race_cv.yaml")
    parser.add_argument("--roster", default=None)
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Override pipeline.target_fps (0 = every frame). Default: use the config",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=3.0,
        help="Seconds a detection may differ from the expected time and still match",
    )
    parser.add_argument("--only", default=None, help="Substring filter on clip name")
    parser.add_argument("--json", default=None, help="Write full results as JSON")
    args = parser.parse_args(argv)

    expectations = parse_expectations(Path(args.expected))
    if args.only:
        expectations = {k: v for k, v in expectations.items() if args.only in k}

    config = Config.load(args.config)
    if args.target_fps is not None:
        config.pipeline.target_fps = args.target_fps
    config.sink.api_url = ""
    roster = load_roster(args.roster)

    # One reader for the whole batch: EasyOCR holds no per-clip state, so the
    # several-second warm-up is paid once rather than 13 times.
    reader = BibReader(config.ocr) if config.ocr.enabled else None
    if reader is not None:
        print(f"OCR warm-up: {reader.warmup():.1f}s")

    print(
        f"config={args.config}  target_fps={config.pipeline.target_fps}  "
        f"tolerance=±{args.tolerance}s  clips={len(expectations)}"
    )

    video_dir = Path(args.videos)
    results: list[ClipResult] = []
    for name, expected in expectations.items():
        video = resolve_video(name, video_dir)
        result = ClipResult(name=name)
        if video is None:
            result.error = f"video not found in {video_dir}"
            result.matches = [Match(expected=e, detected=None) for e in expected]
        else:
            try:
                detected, stats, wall = run_clip(video, config, reader, roster)
                result.matches = match_finishers(expected, detected, args.tolerance)
                result.stats = stats
                result.wall_seconds = wall
            except Exception as exc:  # keep going: one bad clip shouldn't end the run
                result.error = f"{type(exc).__name__}: {exc}"
                result.matches = [Match(expected=e, detected=None) for e in expected]
        results.append(result)
        print_clip(result)

    return report(results, args)


def report(results: list[ClipResult], args) -> int:
    all_matches = [m for r in results for m in r.matches]
    expected_total = sum(1 for m in all_matches if m.expected)
    detected_total = sum(1 for m in all_matches if m.detected)
    found = [m for m in all_matches if m.expected and m.detected]
    missed = [m for m in all_matches if m.is_missed]
    ghosts = [m for m in all_matches if m.is_ghost]
    bib_right = [m for m in found if m.bib_correct]
    bib_wrong = [m for m in found if not m.bib_correct]

    # Bib accuracy is reported separately for racers who wore a readable
    # number, because "correctly reported no bib" and "correctly read 121" are
    # different capabilities and averaging them hides which one is failing.
    with_bib = [m for m in found if m.expected.bib]
    with_bib_right = [m for m in with_bib if m.bib_correct]
    no_bib = [m for m in found if not m.expected.bib]
    no_bib_right = [m for m in no_bib if m.bib_correct]

    drifts = sorted(abs(m.drift) for m in found)

    width = 52
    print("\n" + "=" * width)
    print("SMOKE TEST SUMMARY")
    print("=" * width)
    print(f"{'clips':<34}{len(results):>8}")
    print(f"{'expected finishers':<34}{expected_total:>8}")
    print(f"{'detected finishers':<34}{detected_total:>8}")
    print("-" * width)
    recall = len(found) / expected_total if expected_total else 0.0
    print(f"{'RECALL (found / expected)':<34}{recall:>7.1%}   <- the race-day metric")
    if found:
        print(f"{'bib exactly right':<34}{len(bib_right)}/{len(found):<7}")
    if with_bib:
        print(f"{'  of racers wearing a bib':<34}{len(with_bib_right)}/{len(with_bib):<7}")
    if no_bib:
        print(f"{'  of racers with no bib':<34}{len(no_bib_right)}/{len(no_bib):<7}")
    print(f"{'missed (never detected)':<34}{len(missed):>8}")
    print(f"{'ghosts (detected, not expected)':<34}{len(ghosts):>8}")
    if drifts:
        median = drifts[len(drifts) // 2]
        print(f"{'median |time drift|':<34}{median:>7.1f}s")
        print(f"{'worst |time drift|':<34}{drifts[-1]:>7.1f}s")
    print("=" * width)

    if missed:
        print("\nMISSED:")
        for m in missed:
            clip = next(r.name for r in results if m in r.matches)
            print(f"  {m.expected.label:>7} @ {m.expected.seconds:5.0f}s   {clip}")
    if bib_wrong:
        print("\nWRONG BIB:")
        for m in bib_wrong:
            clip = next(r.name for r in results if m in r.matches)
            print(
                f"  expected {m.expected.label:>7} @ {m.expected.seconds:5.0f}s"
                f"  got {(m.detected.bib or NO_BIB):>7}   {clip}"
            )
    if ghosts:
        print("\nGHOSTS:")
        for m in ghosts:
            clip = next(r.name for r in results if m in r.matches)
            print(f"  {(m.detected.bib or NO_BIB):>7} @ {m.detected.seconds:6.1f}s   {clip}")

    if args.json:
        payload = {
            "config": args.config,
            "tolerance": args.tolerance,
            "summary": {
                "clips": len(results),
                "expected": expected_total,
                "detected": detected_total,
                "found": len(found),
                "recall": round(recall, 4),
                "bib_right": len(bib_right),
                "missed": len(missed),
                "ghosts": len(ghosts),
            },
            "clips": [
                {
                    "name": r.name,
                    "error": r.error,
                    "wall_seconds": round(r.wall_seconds, 1),
                    "stats": r.stats,
                    "matches": [
                        {
                            "expected_bib": m.expected.label if m.expected else None,
                            "expected_s": m.expected.seconds if m.expected else None,
                            "detected_bib": (m.detected.bib or NO_BIB) if m.detected else None,
                            "detected_s": round(m.detected.seconds, 2) if m.detected else None,
                            "bib_correct": m.bib_correct,
                        }
                        for m in r.matches
                    ],
                }
                for r in results
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0 if (not missed and not ghosts and not bib_wrong) else 1


if __name__ == "__main__":
    raise SystemExit(main())
