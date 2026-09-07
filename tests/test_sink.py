"""Tests for durable finish-event delivery.

The central case is a regression test for the race-day data loss found in
backend.log: a finisher refused because the race clock had not been started
must be retried until it lands, never marked sent and dropped.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import SinkConfig
from race_cv.sink import FinishEvent, ResultSink, make_event_id


class FakeResponse:
    def __init__(self, body, status_code=200):
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self):
        return self._body


class FakeSession:
    """Records posts and replays a scripted sequence of responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.posts = []

    def post(self, url, json=None, timeout=None):
        self.posts.append(json)
        response = self._responses.pop(0) if self._responses else FakeResponse(
            {"success": True}
        )
        if isinstance(response, Exception):
            raise response
        return response


def event(track_id=1, bib="10", ts=1000.0) -> FinishEvent:
    return FinishEvent(
        event_id=make_event_id(track_id, ts, "test"),
        track_id=track_id,
        bib_number=bib,
        capture_ts=ts,
        frame_index=42,
    )


def sink_for(tmp_path, session, **overrides) -> ResultSink:
    config = SinkConfig(
        api_url="http://localhost:8001",
        event_log=str(tmp_path / "events.jsonl"),
        retry_seconds=0.01,
        max_retry_seconds=0.02,
        **overrides,
    )
    return ResultSink(config, session=session)


class TestDelivery:
    def test_successful_delivery_marks_delivered(self, tmp_path):
        session = FakeSession([FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.submit(event())
        sink.start()
        deadline = time.time() + 2
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()
        assert sink.stats.delivered == 1
        assert sink.stats.undelivered == 0

    def test_race_clock_refusal_is_retried_not_dropped(self, tmp_path):
        """The Bib #10 regression.

        A 200 OK carrying success=False must not count as delivery. The event
        is retried until the operator starts the clock, and then lands.
        """
        refusal = FakeResponse(
            {"success": False, "message": "Race clock is not running. Please start the race clock first."}
        )
        session = FakeSession([refusal, refusal, FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.submit(event(bib="10"))
        sink.start()
        deadline = time.time() + 3
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()

        assert sink.stats.delivered == 1, "refused finisher must eventually land"
        assert sink.stats.failures >= 2, "refusals must be counted as failures"
        assert len(session.posts) == 3
        assert all(p["bibNumber"] == "10" for p in session.posts)

    def test_retries_reuse_one_event_id(self, tmp_path):
        """Idempotency: the server must be able to dedupe retries."""
        refusal = FakeResponse({"success": False, "message": "nope"})
        session = FakeSession([refusal, refusal, FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.submit(event())
        sink.start()
        deadline = time.time() + 3
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()
        assert len({p["eventId"] for p in session.posts}) == 1

    def test_network_error_is_retried(self, tmp_path):
        session = FakeSession(
            [ConnectionError("connection refused"), FakeResponse({"success": True})]
        )
        sink = sink_for(tmp_path, session)
        sink.submit(event())
        sink.start()
        deadline = time.time() + 3
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()
        assert sink.stats.delivered == 1

    def test_http_error_is_retried(self, tmp_path):
        session = FakeSession(
            [FakeResponse({"detail": "boom"}, status_code=500), FakeResponse({"success": True})]
        )
        sink = sink_for(tmp_path, session)
        sink.submit(event())
        sink.start()
        deadline = time.time() + 3
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()
        assert sink.stats.delivered == 1


class TestDurability:
    def test_event_is_persisted_before_any_delivery_attempt(self, tmp_path):
        """A crash before delivery must still leave the result on disk."""
        session = FakeSession([])
        sink = sink_for(tmp_path, session)
        sink.submit(event(bib="322"))
        # Worker never started: nothing has been sent anywhere.
        assert session.posts == []
        lines = Path(sink.event_log).read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["status"] == "pending"
        assert record["bib_number"] == "322"

    def test_delivery_appends_a_second_record(self, tmp_path):
        session = FakeSession([FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.submit(event())
        sink.start()
        deadline = time.time() + 2
        while sink.stats.delivered < 1 and time.time() < deadline:
            time.sleep(0.01)
        sink.stop()
        statuses = [
            json.loads(line)["status"]
            for line in Path(sink.event_log).read_text().strip().splitlines()
        ]
        assert statuses == ["pending", "delivered"]

    def test_offline_mode_needs_no_api(self, tmp_path):
        config = SinkConfig(api_url="", event_log=str(tmp_path / "e.jsonl"))
        sink = ResultSink(config)
        assert sink.deliver_once(event()) == (True, None)

    def test_undelivered_events_are_recoverable(self, tmp_path):
        session = FakeSession([])
        sink = sink_for(tmp_path, session)
        sink.submit(event(track_id=1))
        sink.submit(event(track_id=2))
        remaining = sink.undelivered_events()
        assert [e.track_id for e in remaining] == [1, 2]


def test_event_ids_are_stable_and_unique():
    assert make_event_id(1, 100.0, "r") == make_event_id(1, 100.0, "r")
    assert make_event_id(1, 100.0, "r") != make_event_id(2, 100.0, "r")
    assert make_event_id(1, 100.0, "r") != make_event_id(1, 100.5, "r")


class TestPendingCountsTheEventBeingRetried:
    """The health line must not read "pending 0 | delivered 0" while a
    finisher is stuck in a backoff loop against a dead API.

    Measured on a profiling run: the worker had dequeued the event and was
    retrying it, the queue was therefore empty, and the health line reported
    pending 0 -- the one state an operator most needs to see.
    """

    def test_in_flight_retry_is_reported_as_pending(self, tmp_path):
        session = FakeSession([ConnectionError("API down")] * 200)
        sink = sink_for(tmp_path, session)
        sink.start()
        sink.submit(event())
        time.sleep(0.3)                     # several failed attempts in
        stats = sink.stats
        try:
            assert stats.delivered == 0
            assert stats.pending == 1, stats
            assert stats.last_error and "API down" in stats.last_error
        finally:
            sink.stop(drain_timeout=0.2)

    def test_pending_returns_to_zero_once_delivered(self, tmp_path):
        session = FakeSession([ConnectionError("blip")] + [FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.start()
        sink.submit(event())
        time.sleep(0.3)
        try:
            assert sink.stats.delivered == 1
            assert sink.stats.pending == 0
        finally:
            sink.stop(drain_timeout=0.2)


class TestLogWriteFailureCannotKillTheLoop:
    def test_append_failure_is_counted_and_delivery_still_happens(self, tmp_path):
        session = FakeSession([FakeResponse({"success": True})])
        sink = sink_for(tmp_path, session)
        sink.event_log = tmp_path / "no_such_dir" / "events.jsonl"   # every append fails
        sink.start()
        sink.submit(event())                                        # must not raise
        time.sleep(0.3)
        try:
            stats = sink.stats
            assert stats.delivered == 1
            assert stats.log_failures >= 1
            # last_error is delivery state (it clears on success); the log
            # failure is surfaced through the counter and an ERROR log line.
            assert stats.last_error is None
        finally:
            sink.stop(drain_timeout=0.2)


class TestRecoverPendingFromAPreviousRun:
    def test_pending_events_from_the_log_are_requeued_and_delivered(self, tmp_path):
        # A previous run logged two events; one was confirmed, one never was.
        session = FakeSession([FakeResponse({"success": True})] * 5)
        earlier = sink_for(tmp_path, session)
        confirmed, lost = event(track_id=1, bib="10"), event(track_id=2, bib="20", ts=1001.0)
        earlier._append(confirmed, "pending"); earlier._append(confirmed, "delivered")
        earlier._append(lost, "pending")
        # New run, same log directory.
        sink = sink_for(tmp_path, session)
        assert sink.recover_pending() == 1
        sink.start()
        time.sleep(0.3)
        try:
            assert sink.stats.delivered == 1
            assert session.posts[-1]["eventId"] == lost.event_id
        finally:
            sink.stop(drain_timeout=0.2)

    def test_nothing_to_recover_is_zero(self, tmp_path):
        assert sink_for(tmp_path, FakeSession([])).recover_pending() == 0

    def test_start_recovers_automatically(self, tmp_path):
        session = FakeSession([FakeResponse({"success": True})])
        sink_for(tmp_path, session)._append(event(bib="30"), "pending")
        sink = sink_for(tmp_path, session)
        sink.start(); time.sleep(0.3)
        try:
            assert sink.stats.recovered == 1 and sink.stats.delivered == 1
        finally:
            sink.stop(drain_timeout=0.2)

    def test_yesterdays_rehearsal_is_not_todays_finisher(self, tmp_path):
        # A --fresh start rotates the log, but if someone forgets, an
        # unconfirmed event from a previous day must not be re-delivered
        # into today's race. In testing, 16 of them were.
        now = time.time()
        earlier = sink_for(tmp_path, FakeSession([]))
        earlier._append(event(track_id=1, bib="10", ts=now - 2 * 24 * 3600), "pending")  # old
        earlier._append(event(track_id=2, bib="20", ts=now - 600), "pending")            # this race
        earlier._append(event(track_id=3, bib="30", ts=1000.0), "pending")               # file-relative
        sink = sink_for(tmp_path, FakeSession([]))
        assert sink.recover_pending() == 2
        queued = {sink._queue.get_nowait().bib_number for _ in range(2)}
        assert queued == {"20", "30"}
