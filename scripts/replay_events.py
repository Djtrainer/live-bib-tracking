#!/usr/bin/env python3
"""Re-deliver finish events from the durable log into the results API.

sink.py appends every finish to ``events.jsonl`` *before* attempting delivery,
specifically so nothing is lost when the API is unreachable -- that was the bug
that cost real finishers in 2025. But the durability half was only useful with
a way to get the events back in, and until now there wasn't one: a run that
ended with events undelivered left you a file and manual re-entry.

This reads the log, works out which events the API never confirmed, and posts
them. Each carries the same ``eventId`` it had originally, so re-running this
is safe against a server that honours idempotency, and the ``--dry-run`` shows
exactly what would be sent against one that doesn't.

    python scripts/replay_events.py --dry-run
    python scripts/replay_events.py --api-url http://localhost:8001
    python scripts/replay_events.py --all        # re-post delivered ones too
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def load_events(path: Path):
    """Return (pending, delivered) event dicts keyed by event_id.

    The log is append-only with a status per line, so an event appears once as
    "pending" and again as "delivered" if it landed. Anything without a
    delivered line is what needs re-sending.
    """
    latest: dict[str, dict] = {}
    delivered: set[str] = set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"  skipping unparseable line {line_no}", file=sys.stderr)
            continue
        event_id = record.get("event_id")
        if not event_id:
            continue
        latest.setdefault(event_id, record)
        if record.get("status") == "delivered":
            delivered.add(event_id)
    pending = {k: v for k, v in latest.items() if k not in delivered}
    return pending, delivered


def to_payload(record: dict) -> dict:
    """Rebuild the POST body, matching sink.FinishEvent.to_payload."""
    payload = {
        "eventId": record["event_id"],
        "bibNumber": record.get("bib_number")
                     or f"Unknown-{record.get('track_id')}",
        "captureTime": record["capture_ts"],
        "wallClockTime": record["capture_ts"],
        "source": "race_cv_replay",
    }
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Re-deliver events from the log")
    parser.add_argument("--log", default="data/results/events.jsonl")
    parser.add_argument("--api-url", default="http://localhost:8001")
    parser.add_argument("--all", action="store_true",
                        help="Re-post every event, not only undelivered ones")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    log = Path(args.log)
    if not log.exists():
        print(f"no event log at {log}", file=sys.stderr)
        return 1

    pending, delivered = load_events(log)
    targets = dict(pending)
    if args.all:
        all_events, _ = load_events(log)
        # load_events drops delivered ones; re-read for the full set
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("event_id"):
                targets.setdefault(record["event_id"], record)

    print(f"log: {log}")
    print(f"  delivered previously : {len(delivered)}")
    print(f"  never confirmed      : {len(pending)}")
    print(f"  will send            : {len(targets)}")
    if not targets:
        print("\nnothing to re-deliver")
        return 0

    for event_id, record in sorted(targets.items(), key=lambda kv: kv[1]["capture_ts"]):
        bib = record.get("bib_number") or f"Unknown-{record.get('track_id')}"
        print(f"    {bib:>10}  capture_ts={record['capture_ts']:.3f}  {event_id}")

    if args.dry_run:
        print("\n--dry-run: nothing sent")
        return 0

    import requests

    url = args.api_url.rstrip("/") + "/api/results"
    session = requests.Session()
    sent = failed = 0
    for event_id, record in sorted(targets.items(), key=lambda kv: kv[1]["capture_ts"]):
        payload = to_payload(record)
        for attempt in range(1, args.retries + 1):
            try:
                response = session.post(url, json=payload, timeout=args.timeout)
                body = response.json() if response.content else {}
                if response.status_code < 400 and body.get("success"):
                    sent += 1
                    print(f"  ok   {payload['bibNumber']}")
                    break
                reason = body.get("message", f"HTTP {response.status_code}")
                # The clock not running is the classic refusal; it is worth
                # saying so plainly rather than burying it in a retry loop.
                print(f"  retry {attempt}/{args.retries} {payload['bibNumber']}: {reason}")
            except Exception as exc:
                print(f"  retry {attempt}/{args.retries} {payload['bibNumber']}: "
                      f"{type(exc).__name__}: {exc}")
            if attempt < args.retries:
                time.sleep(1.0 * attempt)
        else:
            failed += 1
            print(f"  FAILED {payload['bibNumber']} -- still in the log, safe to retry")

    print(f"\ndelivered {sent}, failed {failed}")
    if failed:
        print("Is the race clock running? A stopped clock refuses finishers.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
