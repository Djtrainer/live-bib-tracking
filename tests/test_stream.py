"""Tests for the preview frame relay.

The one property that must hold: submitting a frame never blocks, and a slow
consumer only ever drops frames, never backs up. If this queue could grow or
block, a stalled network POST would leak back into detection/finish timing --
the same coupling sink.py and pipeline.py were built to avoid.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import StreamConfig
from race_cv.stream import FrameStreamer


def frame(value: int = 0) -> np.ndarray:
    return np.full((4, 4, 3), value, dtype=np.uint8)


class FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code
        self.text = "{}"


class FakeSession:
    def __init__(self, fail_times: int = 0, delay: float = 0.0):
        self.posts = []
        self.fail_times = fail_times
        self.delay = delay

    def post(self, url, data=None, headers=None, timeout=None):
        if self.delay:
            time.sleep(self.delay)
        self.posts.append({"url": url, "size": len(data)})
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("simulated network failure")
        return FakeResponse()


class TestSubmitNeverBlocks:
    def test_submit_returns_immediately_with_no_worker_running(self):
        streamer = FrameStreamer("http://x", StreamConfig(), session=FakeSession())
        start = time.time()
        for _ in range(50):
            streamer.submit(frame())
        assert time.time() - start < 0.5

    def test_flooding_submit_only_drops_never_queues(self):
        streamer = FrameStreamer("http://x", StreamConfig(), session=FakeSession())
        for _ in range(100):
            streamer.submit(frame())
        assert streamer._queue.qsize() <= 1


class TestPublishing:
    def test_published_frame_reaches_the_session(self):
        session = FakeSession()
        streamer = FrameStreamer(
            "http://localhost:8001", StreamConfig(target_fps=0), session=session
        )
        streamer.start()
        streamer.submit(frame(255))
        deadline = time.time() + 2
        while streamer.stats.sent < 1 and time.time() < deadline:
            time.sleep(0.01)
        streamer.stop()
        assert streamer.stats.sent == 1
        assert session.posts[0]["url"] == "http://localhost:8001/api/frame"
        assert session.posts[0]["size"] > 0

    def test_url_is_built_from_api_url(self):
        streamer = FrameStreamer("http://localhost:8001/", StreamConfig())
        assert streamer.url == "http://localhost:8001/api/frame"

    def test_target_fps_throttles_publish_rate(self):
        session = FakeSession()
        streamer = FrameStreamer(
            "http://x", StreamConfig(target_fps=5.0), session=session
        )
        streamer.start()
        start = time.time()
        while time.time() - start < 1.0:
            streamer.submit(frame())
            time.sleep(0.01)
        streamer.stop()
        # ~5/s for ~1s: allow generous slack for scheduling jitter, but this
        # must be far below the ~100/s submit rate above.
        assert 2 <= streamer.stats.sent <= 8

    def test_network_errors_are_counted_not_raised(self):
        session = FakeSession(fail_times=3)
        streamer = FrameStreamer(
            "http://x", StreamConfig(target_fps=0), session=session
        )
        streamer.start()
        for _ in range(3):
            streamer.submit(frame())
            time.sleep(0.05)
        deadline = time.time() + 2
        while streamer.stats.errors < 3 and time.time() < deadline:
            streamer.submit(frame())
            time.sleep(0.05)
        streamer.stop()
        assert streamer.stats.errors >= 1
        assert streamer.stats.last_error is not None

    def test_a_slow_post_does_not_block_submit(self):
        """The core guarantee: a hung network call must never propagate back."""
        session = FakeSession(delay=0.3)
        streamer = FrameStreamer(
            "http://x", StreamConfig(target_fps=0), session=session
        )
        streamer.start()
        streamer.submit(frame())
        time.sleep(0.05)  # let the worker pick it up and start "sending"
        start = time.time()
        for _ in range(20):
            streamer.submit(frame())  # must not block despite the in-flight POST
        elapsed = time.time() - start
        streamer.stop(timeout=2.0)
        assert elapsed < 0.2
