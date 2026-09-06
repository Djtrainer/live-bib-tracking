"""Tests for running OCR off the frame loop.

The property that makes this safe is narrow and worth stating: a bib read may
arrive late, but it must never arrive *after the finish event that needs it*.
Everything here is ultimately guarding that one invariant, plus the two ways
the mechanism could fail dangerously -- blocking the frame loop it was built
to unblock, or dying silently and taking bib reading offline for the rest of
the race.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import OcrConfig
from race_cv.ocr import AsyncBibReader, BibVoter


class FakeReader:
    """Stands in for EasyOCR: configurable latency, recorded calls."""

    def __init__(self, text="121", conf=0.9, latency=0.0, raises=False):
        self.text, self.conf = text, conf
        self.latency, self.raises = latency, raises
        self.calls = 0
        self._lock = threading.Lock()

    def preprocess(self, crop):
        return crop

    def read(self, crop):
        if self.latency:
            time.sleep(self.latency)
        with self._lock:
            self.calls += 1
        if self.raises:
            raise RuntimeError("bad crop")
        return self.text, self.conf


def crop():
    return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def voter():
    return BibVoter(OcrConfig(lock_conf=0.99))


class TestResultsArrive:
    def test_read_reaches_the_voter(self, voter):
        worker = AsyncBibReader(FakeReader("121"), voter)
        worker.submit(1, crop(), 0.8)
        assert worker.wait_for(1, timeout=2.0)
        worker.stop()
        assert voter.resolve(1).text == "121"

    def test_votes_accumulate_across_submits(self, voter):
        worker = AsyncBibReader(FakeReader("121"), voter)
        for _ in range(3):
            worker.submit(1, crop(), 0.8)
            worker.wait_for(1, timeout=2.0)
        worker.stop()
        assert voter.resolve(1).votes == 3

    def test_tracks_stay_independent(self, voter):
        worker = AsyncBibReader(FakeReader("7"), voter)
        worker.submit(1, crop(), 0.8)
        worker.submit(2, crop(), 0.8)
        worker.drain(timeout=2.0)
        worker.stop()
        assert voter.resolve(1).text == "7"
        assert voter.resolve(2).text == "7"


class TestNeverBlocksTheFrameLoop:
    def test_submit_returns_immediately_under_slow_ocr(self, voter):
        """The whole point. A 200ms read must not cost the caller 200ms."""
        worker = AsyncBibReader(FakeReader(latency=0.2), voter)
        started = time.monotonic()
        for i in range(5):
            worker.submit(i, crop(), 0.8)
        elapsed = time.monotonic() - started
        worker.stop()
        assert elapsed < 0.05, f"submit blocked for {elapsed * 1000:.0f}ms"

    def test_backlog_is_bounded_not_unbounded(self, voter):
        """A wedged worker must cost memory in constant space, not linear."""
        worker = AsyncBibReader(
            FakeReader(latency=5.0), voter, max_queue=4, max_inflight_per_track=99
        )
        for _ in range(50):
            worker.submit(1, crop(), 0.8)
        stats = worker.stats
        worker.stop()
        assert stats.dropped_backlog >= 40

    def test_one_track_cannot_crowd_out_the_others(self, voter):
        """A long approach by one runner must not starve a late arrival."""
        worker = AsyncBibReader(
            FakeReader(latency=5.0), voter, max_queue=64, max_inflight_per_track=2
        )
        for _ in range(20):
            worker.submit(1, crop(), 0.8)
        assert worker.stats.skipped_inflight == 18
        assert worker.submit(2, crop(), 0.8) is True
        worker.stop()


class TestWaitForIsBounded:
    def test_wait_times_out_rather_than_hanging(self, voter):
        worker = AsyncBibReader(FakeReader(latency=5.0), voter)
        worker.submit(1, crop(), 0.8)
        started = time.monotonic()
        assert worker.wait_for(1, timeout=0.1) is False
        elapsed = time.monotonic() - started
        worker.stop()
        assert 0.1 <= elapsed < 0.5, f"waited {elapsed:.2f}s for a 0.1s timeout"
        assert worker.stats.wait_timeouts == 1

    def test_wait_returns_at_once_when_nothing_is_pending(self, voter):
        worker = AsyncBibReader(FakeReader(), voter)
        started = time.monotonic()
        assert worker.wait_for(99, timeout=5.0) is True
        worker.stop()
        assert time.monotonic() - started < 0.05

    def test_stop_releases_a_blocked_waiter(self, voter):
        """Shutdown must not make a caller sit out its full timeout."""
        worker = AsyncBibReader(FakeReader(latency=5.0), voter)
        worker.submit(1, crop(), 0.8)
        worker.submit(1, crop(), 0.8)
        worker.stop(timeout=0.2)
        started = time.monotonic()
        worker.wait_for(1, timeout=3.0)
        assert time.monotonic() - started < 1.0


class TestFailureIsContained:
    def test_a_raising_read_does_not_kill_the_worker(self, voter):
        reader = FakeReader(raises=True)
        worker = AsyncBibReader(reader, voter)
        worker.submit(1, crop(), 0.8)
        worker.wait_for(1, timeout=2.0)

        reader.raises = False
        worker.submit(2, crop(), 0.8)
        assert worker.wait_for(2, timeout=2.0)
        worker.stop()
        assert voter.resolve(2).text == "121"
        assert worker.stats.errors == 1

    def test_stop_is_idempotent(self, voter):
        worker = AsyncBibReader(FakeReader(), voter)
        worker.submit(1, crop(), 0.8)
        worker.stop()
        worker.stop()


class TestVoterIsThreadSafe:
    def test_concurrent_adds_lose_no_votes(self):
        """Under the GIL this would probably pass anyway; the lock makes it
        a guarantee rather than a coincidence."""
        voter = BibVoter(OcrConfig(lock_conf=0.99))
        workers = [
            AsyncBibReader(FakeReader("42", conf=0.9), voter, max_inflight_per_track=99)
            for _ in range(4)
        ]
        for worker in workers:
            for _ in range(25):
                worker.submit(1, crop(), 0.8)
        for worker in workers:
            worker.drain(timeout=5.0)
            worker.stop()
        assert voter.resolve(1).votes == 100
