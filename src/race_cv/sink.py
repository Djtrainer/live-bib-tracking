"""Durable delivery of finish events.

This module exists because of a proven race-day data loss. The legacy pipeline
did this::

    self.result_callback(payload)
    history["result_sent"] = True          # before knowing the outcome
    logger.info("Successfully sent finisher data ...")

The callback was fire-and-forget, and ``POST /api/results`` rejects a finisher
whenever the race clock is not running. In ``backend.log`` Bib #10 was detected
correctly, logged as a success, refused by the API, and never retried -- seven
seconds before the operator started the clock.

The contract here is the opposite:

* Every event is appended to a local JSONL log **before** any delivery attempt,
  so results survive a crash or a total API outage.
* An event is only marked delivered when the API confirms it.
* A refusal is retried with backoff, not discarded. "Race clock is not running"
  becomes a delay, not a lost racer.
* Every event carries a stable ``event_id`` so retries cannot duplicate.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from .config import SinkConfig


@dataclass
class FinishEvent:
    """A racer finishing, as published by the pipeline."""

    event_id: str
    track_id: int
    bib_number: str | None
    capture_ts: float
    frame_index: int
    ocr_votes: int = 0
    ocr_score: float = 0.0
    bib_locked: bool = False
    in_roster: bool = False
    interpolated: bool = False
    track_observations: int = 0

    def to_payload(self) -> dict[str, Any]:
        """The request body for ``POST /api/results``.

        ``wallClockTime`` is kept for compatibility with the current server,
        which converts it against the race clock. ``captureTime`` and ``eventId``
        are the fields the hardened server should prefer.

        ``racerName`` is omitted rather than sent as ``None``: the server
        resolves a display name with
        ``roster_data.get("racerName", finish_data.get("racerName", default))``,
        and a *present* key with value ``None`` short-circuits that chain and
        overrides the roster lookup and the "Racer #{bib}" fallback with a
        literal null. Omitting the key lets the server's own fallback apply.
        """
        payload: dict[str, Any] = {
            "eventId": self.event_id,
            "bibNumber": self.bib_number or f"Unknown-{self.track_id}",
            "captureTime": self.capture_ts,
            "wallClockTime": self.capture_ts,
            "source": "race_cv",
        }
        return payload


@dataclass
class SinkStats:
    """Operator-visible health of result delivery."""

    submitted: int = 0
    delivered: int = 0
    attempts: int = 0
    failures: int = 0
    pending: int = 0
    last_success_ts: float | None = None
    last_error: str | None = None
    log_failures: int = 0      # durable-log writes that failed (delivery still attempted)
    recovered: int = 0         # events re-queued from a previous run's log at startup

    @property
    def undelivered(self) -> int:
        return self.submitted - self.delivered


class ResultSink:
    """Append-then-deliver sink with an unbounded retry worker."""

    # Unconfirmed events older than this are not re-queued on start; see
    # recover_pending(). Class attribute so a test (or an operator) can lower it.
    recover_max_age_s: float = 12 * 3600

    def __init__(self, config: SinkConfig, session=None, clock=time.time):
        self.config = config
        self._clock = clock
        self._queue: queue.Queue[FinishEvent] = queue.Queue()
        self._stats = SinkStats()
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._session = session
        self._delivered: set[str] = set()

        self.event_log = Path(config.event_log)
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        # True while the worker holds an event it has dequeued but not yet
        # delivered -- the retry-in-progress state the queue size cannot see.
        self._inflight = False

    @property
    def stats(self) -> SinkStats:
        with self._lock:
            snapshot = SinkStats(**asdict(self._stats))
            # The queue plus the event the worker is retrying right now. That
            # one has already been taken off the queue, so counting the queue
            # alone reported "pending 0 | delivered 0" on the health line
            # while a finisher sat in a 30s backoff loop against a dead API
            # -- the one state an operator most needs to see.
            snapshot.pending = self._queue.qsize() + (1 if self._inflight else 0)
            return snapshot

    def _ensure_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
        return self._session

    def submit(self, event: FinishEvent) -> None:
        """Persist an event, then queue it for delivery.

        Persisting first is what makes this survivable: if the process dies
        before the API ever accepts the event, the JSONL still has it and the
        result can be recovered.
        """
        self._append(event, status="pending")
        with self._lock:
            self._stats.submitted += 1
        self._queue.put(event)

    def _append(self, event: FinishEvent, status: str) -> None:
        """Append to the durable log. Never raises.

        The log is the backup channel; delivery is the primary one. A full
        disk or a bad permission must not turn a durability write into the
        thing that kills the frame loop -- which it did, because submit()
        runs on the pipeline's thread. Log it, count it, deliver anyway.
        """
        record = {"status": status, "logged_at": self._clock(), **asdict(event)}
        try:
            with self.event_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:
            with self._lock:
                self._stats.log_failures += 1
                self._stats.last_error = f"event log write failed: {type(exc).__name__}: {exc}"
            logging.getLogger(__name__).error(
                "Could not append to %s (%s). The event is still queued for "
                "delivery, but it will NOT be recoverable from the log.",
                self.event_log, exc,
            )

    def recover_pending(self) -> int:
        """Re-queue events a previous run logged but never got confirmed.

        race_cv dying with finishes still retrying used to require someone
        to remember scripts/replay_events.py after the race. The API is
        idempotent on event_id, so re-sending on the next start is safe.
        Returns the number of events re-queued.
        """
        if not self.event_log.is_file():
            return 0
        latest: dict[str, dict] = {}
        delivered: set[str] = set()
        try:
            for line in self.event_log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event_id = record.get("event_id")
                if not event_id:
                    continue
                latest.setdefault(event_id, record)
                if record.get("status") == "delivered":
                    delivered.add(event_id)
        except Exception as exc:
            logging.getLogger(__name__).warning("Could not read %s for recovery: %s", self.event_log, exc)
            return 0
        fields = FinishEvent.__dataclass_fields__
        count = 0
        stale = 0
        now = time.time()
        for event_id, record in latest.items():
            if event_id in delivered or event_id in self._delivered:
                continue
            try:
                event = FinishEvent(**{k: v for k, v in record.items() if k in fields})
            except TypeError:
                continue
            # An unconfirmed event from yesterday's rehearsal is not a
            # finisher in today's race. Judge only epoch-like timestamps:
            # offline replays stamp frames relative to the file start.
            if event.capture_ts > 1e9 and now - event.capture_ts > self.recover_max_age_s:
                stale += 1
                continue
            self._queue.put(event)
            count += 1
        if stale:
            logging.getLogger(__name__).warning(
                "Ignored %d unconfirmed event(s) in %s older than %.0fh -- a "
                "previous race or rehearsal. Start with --fresh to archive the log.",
                stale, self.event_log, self.recover_max_age_s / 3600,
            )
        if count:
            with self._lock:
                self._stats.recovered += count
            logging.getLogger(__name__).warning(
                "Re-queued %d finish event(s) from %s that a previous run never "
                "got confirmed", count, self.event_log,
            )
        return count

    def deliver_once(self, event: FinishEvent) -> tuple[bool, str | None]:
        """Attempt a single delivery. Returns ``(delivered, error)``."""
        if not self.config.api_url:
            return True, None  # replay / offline mode: the JSONL is the result
        url = self.config.api_url.rstrip("/") + "/api/results"
        with self._lock:
            self._stats.attempts += 1
        try:
            response = self._ensure_session().post(
                url, json=event.to_payload(), timeout=self.config.timeout_seconds
            )
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

        if response.status_code >= 400:
            return False, f"HTTP {response.status_code}: {response.text[:200]}"
        try:
            body = response.json()
        except ValueError:
            return False, f"Non-JSON response: {response.text[:200]}"

        # The server signals a refusal in the body, not the status code. This is
        # exactly how Bib #10 was lost: a 200 OK carrying success=False.
        if not body.get("success", False):
            return False, str(body.get("message", "server reported success=False"))
        return True, None

    def _worker(self) -> None:
        backoff = self.config.retry_seconds
        pending: FinishEvent | None = None
        while self._running or pending is not None or not self._queue.empty():
            if pending is None:
                try:
                    pending = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                with self._lock:
                    self._inflight = True

            if pending.event_id in self._delivered:
                pending = None
                backoff = self.config.retry_seconds
                with self._lock:
                    self._inflight = False
                continue

            delivered, error = self.deliver_once(pending)
            if delivered:
                self._delivered.add(pending.event_id)
                self._append(pending, status="delivered")
                with self._lock:
                    self._stats.delivered += 1
                    self._stats.last_success_ts = self._clock()
                    self._stats.last_error = None
                    self._inflight = False
                pending = None
                backoff = self.config.retry_seconds
                continue

            with self._lock:
                self._stats.failures += 1
                self._stats.last_error = error
            # Retry the same event rather than dropping it. A refused finisher
            # is a delayed finisher, never a lost one.
            time.sleep(min(backoff, self.config.max_retry_seconds))
            backoff = min(backoff * 2, self.config.max_retry_seconds)

    def start(self) -> None:
        if self._running:
            return
        self.recover_pending()
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, drain_timeout: float = 30.0) -> SinkStats:
        """Stop accepting work and give the worker a chance to drain."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=drain_timeout)
        return self.stats

    def undelivered_events(self) -> list[FinishEvent]:
        """Events still queued -- what an operator needs to enter by hand."""
        remaining = []
        while True:
            try:
                remaining.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return remaining


def make_event_id(track_id: int, capture_ts: float, run_id: str) -> str:
    """A stable idempotency key for a finish.

    Deterministic in a replay (same run_id, same track, same timestamp) so
    re-running a replay does not create duplicate records downstream.
    """
    return f"{run_id}-t{track_id}-{capture_ts:.3f}"
