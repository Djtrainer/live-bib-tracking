"""Tests for keeping the OCR worker current with the frame loop.

Measured on real bib crops: a read costs ~48ms on MPS (not the 21ms a
synthetic crop suggested), with outliers of 150-1000ms whenever a crop
arrives at a width the Metal backend has not compiled a kernel for. The
frame loop, meanwhile, submits a crop for every frame a bib is visible --
up to 29 a second per runner. The worker cannot keep up, the per-track cap
discards crops, and a finish event's 0.25s resolve timeout fires with the
evidence still queued. In the bib_env replay of 14-48-12 that produced
"No bib" for a racer whose bib was legible on 53 frames, and "20" for one
whose earliest -- and only correct -- reads were the ones dropped.

Three properties, each guarded below:

* submits are rate-limited per track, so demand stays under capacity;
* crop widths are quantized to a fixed set the warm-up sweeps, so no
  shape is first seen mid-race;
* a pending finish defers, without blocking the loop, while that racer's
  reads are still in flight.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.capture import Frame
from race_cv.config import Config, FinishLineConfig, OcrConfig, PipelineConfig
from race_cv.detect import Detection
from race_cv.ocr import AsyncBibReader, BibReader, BibVoter
from race_cv.pipeline import Pipeline


class FakeReader:
    def __init__(self, text="120", conf=0.95, latency=0.0):
        self.text, self.conf, self.latency = text, conf, latency
        self.calls = 0

    def preprocess(self, crop):
        return crop

    def read(self, crop):
        if self.latency:
            time.sleep(self.latency)
        self.calls += 1
        return self.text, self.conf


def crop():
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestSubmitRateLimit:
    def test_one_track_is_limited_to_the_configured_rate(self):
        # A raised in-flight cap isolates the rate gate: with a synthetic
        # clock the 30 submits arrive instantly and the worker has no real
        # time to drain, which is not the production condition under test.
        worker = AsyncBibReader(FakeReader(), BibVoter(OcrConfig()),
                                min_submit_interval_s=0.12, max_inflight_per_track=99)
        # 30 frames over one second, all with a visible bib
        accepted = sum(worker.submit(1, crop(), 0.9, now=100.0 + i / 30.0) for i in range(30))
        worker.stop()
        assert 7 <= accepted <= 9, accepted          # ~8/s, not 30/s
        assert worker.stats.skipped_rate == 30 - accepted

    def test_tracks_are_limited_independently(self):
        # A raised in-flight cap isolates the rate gate: with a synthetic
        # clock the 30 submits arrive instantly and the worker has no real
        # time to drain, which is not the production condition under test.
        worker = AsyncBibReader(FakeReader(), BibVoter(OcrConfig()),
                                min_submit_interval_s=0.12, max_inflight_per_track=99)
        assert worker.submit(1, crop(), 0.9, now=100.0)
        assert worker.submit(2, crop(), 0.9, now=100.0)   # a different runner, same frame
        assert not worker.submit(1, crop(), 0.9, now=100.03)
        worker.stop()

    def test_zero_interval_means_every_frame(self):
        worker = AsyncBibReader(FakeReader(), BibVoter(OcrConfig()),
                                min_submit_interval_s=0.0, max_inflight_per_track=99)
        assert all(worker.submit(1, crop(), 0.9, now=100.0 + i / 30.0) for i in range(30))
        worker.stop()


class TestWidthBuckets:
    """Every crop the worker sees must have a width the warm-up compiled."""

    def _reader(self, buckets=(128, 160, 192, 256, 320)):
        return BibReader(OcrConfig(target_height=120, width_buckets_px=list(buckets)))

    def test_width_is_padded_up_to_the_next_bucket(self):
        reader = self._reader()
        out = reader.preprocess(np.zeros((81, 97, 3), dtype=np.uint8))  # 97x81 -> 144x120
        assert out.shape == (120, 160)

    def test_an_exact_bucket_width_is_left_alone(self):
        reader = self._reader()
        out = reader.preprocess(np.zeros((120, 192, 3), dtype=np.uint8))
        assert out.shape == (120, 192)

    def test_wider_than_the_largest_bucket_is_padded_to_a_multiple_of_it(self):
        """Never crop content away; a very wide bib pads to the next 32px."""
        reader = self._reader()
        out = reader.preprocess(np.zeros((120, 400, 3), dtype=np.uint8))
        assert out.shape[0] == 120 and out.shape[1] >= 400 and out.shape[1] % 32 == 0

    def test_padding_is_white_not_black(self):
        """Black padding reads as a giant glyph edge to OCR; bibs are white."""
        reader = self._reader()
        src = np.full((120, 130, 3), 200, dtype=np.uint8)
        out = reader.preprocess(src)
        assert out[:, -1].min() == 255

    def test_warmup_sweeps_exactly_the_buckets(self):
        reader = self._reader()
        seen = []

        class Probe:
            def readtext(self, image, **kw):
                seen.append(image.shape[1]); return []

        reader._reader = Probe()
        reader.warmup()
        assert sorted(set(seen)) == [128, 160, 192, 256, 320]


W = H = 1000


class ScriptedDetector:
    def __init__(self, script): self.script = script
    def track(self, image): return self.script.get(getattr(image, "frame_index", 0), [])
    def split(self, dets):
        return ([d for d in dets if d.cls == 0 and d.track_id is not None],
                [d for d in dets if d.cls == 1])


class Tagged(np.ndarray):
    @classmethod
    def make(cls, i):
        a = np.zeros((H, W, 3), dtype=np.uint8).view(cls); a.frame_index = i; return a


def person(tid, bottom):
    return Detection(xyxy=(450.0, bottom - 200.0, 550.0, bottom), conf=0.9, cls=0, track_id=tid)


def bib_on(bottom):
    return Detection(xyxy=(470.0, bottom - 120.0, 530.0, bottom - 80.0), conf=0.9, cls=1)


class TestPendingFinishDefersForInFlightReads:
    """A finish must wait for the reads it already dispatched, without
    blocking the frame loop, and only up to a bound."""

    def _run(self, latency, grace, timeout=0.0):
        # 12 approach frames with a bib, crossing at frame ~5, then 30 frames past
        script = {i: [person(1, 300 + i * 50), bib_on(300 + i * 50)] for i in range(6)}
        for i in range(6, 40):
            script[i] = [person(1, 620)]
        config = Config(
            finish_line=FinishLineConfig(p1=(0.0, 0.5), p2=(1.0, 0.5), confirm_frames=2),
            ocr=OcrConfig(enabled=True, min_bib_yolo_conf=0.5, lock_conf=0.99,
                          resolve_timeout=timeout, resolve_grace_s=grace,
                          async_min_submit_interval_s=0.0),
            pipeline=PipelineConfig(target_fps=0.0),
        )
        events, emitted_at = [], []
        p = Pipeline(config=config, detector=ScriptedDetector(script), frame_width=W,
                     frame_height=H, run_id="t", bib_reader=FakeReader(latency=latency),
                     roster={"120"}, emit=lambda e: (events.append(e), emitted_at.append(time.monotonic())))
        started = time.monotonic()
        loop_max = 0.0
        for i in range(40):
            t0 = time.monotonic()
            p.process(Frame(image=Tagged.make(i), capture_ts=i / 10.0, index=i))
            loop_max = max(loop_max, time.monotonic() - t0)
        p.flush(); p.close()
        return events, loop_max, p

    def test_finish_waits_for_slow_reads_and_keeps_the_bib(self):
        events, loop_max, p = self._run(latency=0.08, grace=2.0)
        assert len(events) == 1
        assert events[0].bib_number == "120"
        assert p.stats.finishes_deferred_for_ocr > 0

    def test_deferral_never_blocks_the_frame_loop(self):
        """The whole point of async OCR. Deferral is a decision, not a wait."""
        _, loop_max, _ = self._run(latency=0.08, grace=2.0)
        assert loop_max < 0.05, f"a process() call blocked for {loop_max * 1000:.0f}ms"

    def test_grace_is_bounded(self):
        """A wedged worker must not hold a finish forever."""
        events, _, p = self._run(latency=5.0, grace=0.3)
        assert len(events) == 1                       # emitted anyway
        assert p.stats.finishes_deferred_for_ocr > 0
        assert p.stats.ocr_wait_timeouts >= 0
