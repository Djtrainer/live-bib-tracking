#!/usr/bin/env python3
"""Score a replay against hand-labelled ground truth.

Recall is the number that matters. The race-day failure was racers missed
entirely, so "of the runners who actually crossed, how many did we report?" is
the metric every change is judged against. Everything else -- bib accuracy,
ordering, time error -- is secondary to not losing people.

Ground truth is a CSV with a ``bib`` column and, optionally, ``elapsed_s``
(seconds from the first finisher) so timing can be scored too:

    bib,elapsed_s
    322,0.0
    10,4.6
    7,11.2

    python scripts/score.py --truth data/results/truth.csv \
        --results data/results/baseline/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pick(row: dict[str, str], *names: str) -> str | None:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return None


def load(path: Path) -> list[tuple[str, float | None]]:
    """Return (bib, elapsed_seconds or None) in file order."""
    entries = []
    for row in read_rows(path):
        bib = pick(row, "bib", "Bib", "bibNumber", "Bib Number")
        if not bib:
            continue
        raw_time = pick(row, "elapsed_s", "elapsed", "capture_ts", "Time")
        try:
            elapsed = float(raw_time) if raw_time is not None else None
        except ValueError:
            elapsed = None
        entries.append((bib, elapsed))
    return entries


def inversions(sequence: list[int]) -> int:
    """Count out-of-order pairs -- how scrambled the finishing order is."""
    return sum(
        1
        for i in range(len(sequence))
        for j in range(i + 1, len(sequence))
        if sequence[i] > sequence[j]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Score replay results against ground truth"
    )
    parser.add_argument("--truth", required=True, help="Ground-truth CSV")
    parser.add_argument("--results", required=True, help="replay.py results.csv")
    parser.add_argument("--json", default=None, help="Also write metrics as JSON")
    args = parser.parse_args(argv)

    truth = load(Path(args.truth))
    results = load(Path(args.results))

    truth_bibs = [b for b, _ in truth]
    result_bibs = [b for b, _ in results]
    truth_set, result_set = set(truth_bibs), set(result_bibs)

    found = truth_set & result_set
    missed = [b for b in truth_bibs if b not in result_set]
    ghosts = [b for b in result_bibs if b not in truth_set]

    recall = len(found) / len(truth_set) if truth_set else 0.0
    precision = len(found) / len(result_set) if result_set else 0.0

    duplicates = sorted({b for b in result_bibs if result_bibs.count(b) > 1})

    # Ordering: positions in our results of the truth finishers, in truth order.
    result_position = {}
    for index, bib in enumerate(result_bibs):
        result_position.setdefault(bib, index)
    ordered = [result_position[b] for b in truth_bibs if b in result_position]
    order_inversions = inversions(ordered)
    max_pairs = len(ordered) * (len(ordered) - 1) // 2
    order_accuracy = 1.0 - (order_inversions / max_pairs) if max_pairs else 1.0

    # Timing: align on the median offset, then report residuals.
    truth_times = {b: t for b, t in truth if t is not None}
    result_times = {b: t for b, t in results if t is not None}
    shared = [b for b in truth_times if b in result_times]
    time_errors = []
    if shared:
        offsets = [result_times[b] - truth_times[b] for b in shared]
        offset = statistics.median(offsets)
        time_errors = [abs(result_times[b] - truth_times[b] - offset) for b in shared]

    metrics = {
        "truth_finishers": len(truth_set),
        "reported_finishers": len(result_set),
        "matched": len(found),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "missed_bibs": missed,
        "ghost_bibs": ghosts,
        "duplicate_bibs": duplicates,
        "order_accuracy": round(order_accuracy, 4),
        "order_inversions": order_inversions,
        "median_abs_time_error_s": (
            round(statistics.median(time_errors), 3) if time_errors else None
        ),
        "p95_abs_time_error_s": (
            round(sorted(time_errors)[int(len(time_errors) * 0.95)], 3)
            if len(time_errors) >= 20
            else None
        ),
    }

    width = 34
    print("=" * width)
    print("REPLAY SCORE")
    print("=" * width)
    print(f"{'ground-truth finishers':<26}{metrics['truth_finishers']:>8}")
    print(f"{'reported finishers':<26}{metrics['reported_finishers']:>8}")
    print(f"{'matched':<26}{metrics['matched']:>8}")
    print("-" * width)
    print(f"{'RECALL':<26}{recall:>8.1%}   <- the race-day metric")
    print(f"{'precision':<26}{precision:>8.1%}")
    print(f"{'order accuracy':<26}{order_accuracy:>8.1%}")
    if metrics["median_abs_time_error_s"] is not None:
        print(f"{'median |time error|':<26}{metrics['median_abs_time_error_s']:>7.3f}s")
    print("-" * width)
    if missed:
        print(f"MISSED ({len(missed)}): {', '.join(missed[:25])}"
              + (" ..." if len(missed) > 25 else ""))
    if ghosts:
        print(f"GHOSTS ({len(ghosts)}): {', '.join(ghosts[:25])}"
              + (" ..." if len(ghosts) > 25 else ""))
    if duplicates:
        print(f"DUPLICATES ({len(duplicates)}): {', '.join(duplicates[:25])}")
    if not missed and not ghosts and not duplicates:
        print("no missed, ghost or duplicate finishers")
    print("=" * width)

    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
