"""Tests for frame sources, focused on real-time rehearsal pacing.

The point of realtime mode is that a rehearsal against a recorded file should
put the same time pressure on the pipeline that a live camera does. That means
two things, and the second is the one that's easy to get wrong: frames arrive
on the wall clock, AND frames the consumer was too slow to collect are
dropped. A version that only paced (never dropped) would let a slow pipeline
fall arbitrarily behind while still seeing every frame -- hiding exactly the
coverage gaps a rehearsal is supposed to expose.

Clock and sleep are injected so these run instantly and deterministically.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2

from race_cv.capture import VideoFileSource, is_gstreamer_pipeline, open_source

TINY = str(Path(__file__).parent / "fixtures_tiny.mp4")  # 30 frames @ 30fps


class FakeClock:
    """A clock the test drives, optionally with work time per frame."""

    def __init__(self, start=1000.0, work_per_frame=0.0):
        self.t = start
        self.work_per_frame = work_per_frame
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds

    def do_work(self):
        self.t += self.work_per_frame


class TestDeterministicMode:
    def test_delivers_every_frame(self):
        source = VideoFileSource(TINY)
        frames = list(source.frames())
        source.release()
        assert len(frames) == 30
        assert source.dropped == 0

    def test_timestamps_are_synthetic_and_reproducible(self):
        first = VideoFileSource(TINY, start_epoch=500.0)
        a = [f.capture_ts for f in first.frames()]
        first.release()
        second = VideoFileSource(TINY, start_epoch=500.0)
        b = [f.capture_ts for f in second.frames()]
        second.release()
        assert a == b
        assert a[0] == 500.0
        assert a[1] == pytest.approx(500.0 + 1 / 30)

    def test_never_sleeps(self):
        clock = FakeClock()
        source = VideoFileSource(TINY, now=clock.now, sleep=clock.sleep)
        list(source.frames())
        source.release()
        assert clock.slept == []


class TestRealtimeMode:
    def test_paces_frames_on_the_wall_clock(self):
        clock = FakeClock()
        source = VideoFileSource(TINY, realtime=True, now=clock.now, sleep=clock.sleep)
        frames = list(source.frames())
        source.release()
        assert len(frames) == 30
        # A fast consumer sleeps between frames rather than racing ahead.
        assert len(clock.slept) > 0
        # 30 frames at 30fps spans ~1 second of wall clock.
        assert sum(clock.slept) == pytest.approx(29 / 30, abs=0.05)

    def test_anchors_to_playback_start_not_construction(self):
        """Model warm-up happens between construction and the first frame."""
        clock = FakeClock()
        source = VideoFileSource(
            TINY, start_epoch=0.0, realtime=True, now=clock.now, sleep=clock.sleep
        )
        clock.t += 7.0  # 7 seconds of warm-up before the loop starts
        frames = list(source.frames())
        source.release()
        # Anchoring at construction would make every frame "late" and drop the
        # whole opening of the video.
        assert len(frames) == 30
        assert source.dropped == 0

    def test_drops_frames_when_the_consumer_is_slow(self):
        """A slow pipeline must lose frames, the way it would on a camera."""
        clock = FakeClock(work_per_frame=0.2)  # 200ms per frame vs 33ms budget
        source = VideoFileSource(TINY, realtime=True, now=clock.now, sleep=clock.sleep)
        collected = []
        for frame in source.frames():
            collected.append(frame)
            clock.do_work()
        source.release()
        assert source.dropped > 0
        assert len(collected) + source.dropped == 30
        # At 200ms of work per delivered frame, only ~1 in 6 survives.
        assert len(collected) < 12

    def test_a_fast_consumer_drops_nothing(self):
        clock = FakeClock(work_per_frame=0.001)
        source = VideoFileSource(TINY, realtime=True, now=clock.now, sleep=clock.sleep)
        for _ in source.frames():
            clock.do_work()
        source.release()
        assert source.dropped == 0


class TestOpenSource:
    def test_file_spec_honours_realtime(self):
        source = open_source(TINY, realtime=True)
        assert isinstance(source, VideoFileSource)
        assert source.realtime is True
        source.release()

    def test_file_spec_defaults_to_deterministic(self):
        source = open_source(TINY)
        assert source.realtime is False
        source.release()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            VideoFileSource("does/not/exist.mp4")


class FakeCapture:
    """Stands in for cv2.VideoCapture: records what was asked of the device."""

    instances: list = []

    def __init__(self, spec, backend=None):
        self.spec, self.backend = spec, backend
        self.props: dict = {}
        FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def set(self, prop, value):
        self.props[prop] = value

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 30.0}.get(prop, 0)

    def read(self):
        return False, None

    def release(self):
        pass


class TestCameraSpecs:
    """A camera index behaves as it always has; a pipeline and a rate are new."""

    def setup_method(self):
        FakeCapture.instances.clear()

    def test_an_index_requests_nothing_extra(self, monkeypatch):
        monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
        source = open_source("1")
        cap = FakeCapture.instances[-1]
        assert cap.spec == 1 and cap.backend is None
        assert cv2.CAP_PROP_FPS not in cap.props      # the Mac path: no rate request
        assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1
        assert source.is_live
        source.release()

    def test_a_rate_request_reaches_the_driver(self, monkeypatch):
        monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
        source = open_source("1", camera_fps=60)
        assert FakeCapture.instances[-1].props[cv2.CAP_PROP_FPS] == 60.0
        source.release()

    def test_a_gstreamer_pipeline_opens_with_the_gstreamer_backend(self, monkeypatch):
        monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
        pipeline = "v4l2src device=/dev/video0 ! image/jpeg ! nvv4l2decoder mjpeg=1 ! nvvidconv ! appsink"
        source = open_source(pipeline)
        cap = FakeCapture.instances[-1]
        assert cap.spec == pipeline and cap.backend == cv2.CAP_GSTREAMER
        assert source.is_live
        source.release()

    def test_reopen_reapplies_the_same_requests(self, monkeypatch):
        monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
        source = open_source("0", camera_fps=60)
        assert source._reopen()
        assert len(FakeCapture.instances) == 2
        assert FakeCapture.instances[-1].props[cv2.CAP_PROP_FPS] == 60.0
        assert source.reopens == 1
        source.release()

    def test_a_path_with_a_bang_in_it_is_still_a_file(self):
        assert not is_gstreamer_pipeline("data/raw/race!take2.mov")
        assert is_gstreamer_pipeline("videotestsrc ! video/x-raw ! appsink drop=true")


@pytest.mark.skipif(
    "GStreamer:                   YES" not in cv2.getBuildInformation(),
    reason="OpenCV built without GStreamer",
)
def test_a_real_gstreamer_pipeline_delivers_live_frames():
    source = open_source(
        "videotestsrc is-live=true ! video/x-raw,width=320,height=240,framerate=30/1,format=BGR "
        "! appsink drop=true max-buffers=1"
    )
    try:
        frames = source.frames()
        first = next(frames)
        assert first.image.shape == (240, 320, 3)
        assert source.is_live and source.frame_width == 320
    finally:
        source.release()
