"""End-to-end pipeline tests using a scripted detector.

These run without the YOLO model or EasyOCR, so they exercise the finish logic
deterministically: one racer produces exactly one event, pacing never blinds the
line for a whole second, and a racer crossing in the final frames is still
reported.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.capture import Frame
from race_cv.config import (
    Config,
    CourseBoundaryConfig,
    FinishLineConfig,
    OcrConfig,
    PipelineConfig,
)
from race_cv.detect import Detection
from race_cv.pipeline import Pipeline, associate_bib

W = H = 1000


class ScriptedDetector:
    """Returns a pre-baked list of detections per frame index."""

    def __init__(self, script):
        self.script = script

    def track(self, image):
        return self.script.get(getattr(image, "frame_index", 0), [])

    def split(self, detections):
        people = [d for d in detections if d.cls == 0 and d.track_id is not None]
        bibs = [d for d in detections if d.cls == 1]
        return people, bibs


class TaggedImage(np.ndarray):
    """A tiny ndarray that remembers which frame it came from."""

    @classmethod
    def make(cls, index):
        arr = np.zeros((4, 4, 3), dtype=np.uint8).view(cls)
        arr.frame_index = index
        return arr


def person(track_id, bottom_y, x1=450.0, x2=550.0):
    return Detection(
        xyxy=(x1, bottom_y - 200.0, x2, bottom_y), conf=0.9, cls=0, track_id=track_id
    )


def base_config(target_fps: float = 0.0) -> Config:
    return Config(
        finish_line=FinishLineConfig(p1=(0.0, 0.5), p2=(1.0, 0.5), confirm_frames=2),
        ocr=OcrConfig(enabled=False),
        pipeline=PipelineConfig(target_fps=target_fps),
    )


def build(script, config=None, roster=None):
    config = config or base_config()
    events = []
    pipeline = Pipeline(
        config=config,
        detector=ScriptedDetector(script),
        frame_width=W,
        frame_height=H,
        run_id="test",
        bib_reader=None,
        roster=roster,
        emit=events.append,
    )
    return pipeline, events


def frames(count, fps=10.0):
    return [
        Frame(image=TaggedImage.make(i), capture_ts=i / fps, index=i)
        for i in range(count)
    ]


class TestFinishFlow:
    def test_one_racer_produces_exactly_one_event(self):
        script = {i: [person(1, 300.0 + i * 50)] for i in range(10)}
        pipeline, events = build(script)
        pipeline.run(frames(10))
        assert len(events) == 1
        assert events[0].track_id == 1
        assert events[0].interpolated

    def test_finish_time_is_capture_time_not_processing_time(self):
        # Bottom edge 480 -> 520 between frames 0 and 1 at 10 fps.
        script = {0: [person(1, 480.0)], 1: [person(1, 520.0)]}
        pipeline, events = build(script)
        pipeline.run(frames(2))
        assert len(events) == 1
        # Crossed halfway between t=0.0 and t=0.1.
        assert events[0].capture_ts == pytest.approx(0.05)

    def test_racer_crossing_in_final_frames_is_still_emitted(self):
        """confirm_frames must never swallow a finisher at end of stream."""
        script = {0: [person(1, 400.0)], 1: [person(1, 600.0)]}
        pipeline, events = build(script)  # confirm_frames=2, only 2 frames exist
        pipeline.run(frames(2))
        assert len(events) == 1

    def test_track_lost_after_crossing_emits_immediately(self):
        script = {0: [person(1, 400.0)], 1: [person(1, 600.0)], 2: [], 3: []}
        pipeline, events = build(script)
        pipeline.run(frames(4))
        assert len(events) == 1

    def test_two_racers_produce_two_events(self):
        script = {
            i: [person(1, 300.0 + i * 50), person(2, 250.0 + i * 50, 700.0, 800.0)]
            for i in range(12)
        }
        pipeline, events = build(script)
        pipeline.run(frames(12))
        assert sorted(e.track_id for e in events) == [1, 2]

    def test_reacquired_track_past_the_line_does_not_invent_a_finisher(self):
        script = {0: [person(1, 400.0)], 1: [person(1, 600.0)], 2: [person(9, 800.0)]}
        pipeline, events = build(script)
        pipeline.run(frames(3))
        assert [e.track_id for e in events] == [1]
        assert pipeline.stats.suppressed_first_seen_past == 1

    def test_no_detections_produces_no_events(self):
        pipeline, events = build({})
        pipeline.run(frames(20))
        assert events == []
        assert pipeline.stats.frames_processed == 20


class TestPacing:
    def test_target_fps_zero_processes_every_frame(self):
        pipeline, _ = build({}, config=base_config(target_fps=0.0))
        pipeline.run(frames(30, fps=30.0))
        assert pipeline.stats.frames_processed == 30
        assert pipeline.stats.frames_paced_out == 0

    def test_pacing_is_uniform_never_a_burst(self):
        """The legacy cooldown blinded the line for a full second at a time."""
        config = base_config(target_fps=10.0)
        pipeline, _ = build({}, config=config)
        pipeline.run(frames(60, fps=30.0))
        # 60 frames over 2s at a 10 fps target: roughly 20 processed, and the
        # gap between any two processed frames stays near 0.1s.
        assert 18 <= pipeline.stats.frames_processed <= 21
        assert pipeline.stats.processed_fps == pytest.approx(10.0, abs=1.5)

    def test_never_blind_for_longer_than_the_pacing_interval(self):
        config = base_config(target_fps=10.0)
        processed = []
        pipeline, _ = build({}, config=config)
        pipeline.run(frames(90, fps=30.0), on_result=lambda r: processed.append(r.frame.capture_ts))
        gaps = [b - a for a, b in zip(processed, processed[1:])]
        assert max(gaps) < 0.15, "a gap this large is a blind window at the line"


class TestBibAssociation:
    def test_picks_bib_inside_the_person_box(self):
        p = person(1, 600.0)
        inside = Detection(xyxy=(480.0, 450.0, 520.0, 480.0), conf=0.9, cls=1)
        outside = Detection(xyxy=(10.0, 10.0, 40.0, 40.0), conf=0.99, cls=1)
        assert associate_bib(p, [outside, inside]) is inside

    def test_highest_confidence_wins_when_boxes_overlap(self):
        p = person(1, 600.0)
        weak = Detection(xyxy=(480.0, 450.0, 520.0, 480.0), conf=0.5, cls=1)
        strong = Detection(xyxy=(490.0, 460.0, 530.0, 490.0), conf=0.95, cls=1)
        assert associate_bib(p, [weak, strong]) is strong

    def test_no_bib_returns_none(self):
        assert associate_bib(person(1, 600.0), []) is None


class TestMinObservations:
    """A track must be seen enough times before it may finish.

    Guards against track fragmentation: a sharper detector invents more
    short-lived tracks, and on real footage one seen 6 times fired a duplicate
    finish 0.44s before the real racer's track (seen 44 times).
    """

    def _config(self, minimum: int) -> Config:
        config = base_config()
        config.finish_line.min_observations = minimum
        return config

    def test_disabled_by_default(self):
        assert base_config().finish_line.min_observations == 0

    def test_short_lived_track_cannot_finish(self):
        # Seen twice, then crosses on the second frame.
        script = {0: [person(1, 400.0)], 1: [person(1, 600.0)]}
        pipeline, events = build(script, config=self._config(5))
        pipeline.run(frames(2))
        assert events == []
        assert pipeline.stats.finishes_below_min_observations == 1

    def test_well_observed_track_finishes_normally(self):
        script = {i: [person(1, 300.0 + i * 30)] for i in range(12)}
        pipeline, events = build(script, config=self._config(5))
        pipeline.run(frames(12))
        assert len(events) == 1
        assert pipeline.stats.finishes_below_min_observations == 0

    def test_threshold_of_zero_admits_everything(self):
        script = {0: [person(1, 400.0)], 1: [person(1, 600.0)]}
        pipeline, events = build(script, config=self._config(0))
        pipeline.run(frames(2))
        assert len(events) == 1

    def test_event_records_how_often_the_track_was_seen(self):
        script = {i: [person(1, 300.0 + i * 30)] for i in range(12)}
        pipeline, events = build(script, config=self._config(0))
        pipeline.run(frames(12))
        # Counting stops when the confirmation window elapses and the event is
        # built, so this is under the 12 frames the track was actually in.
        assert events[0].track_observations >= 5


class TestCourseBoundary:
    """Confirms the boundary gate is actually wired into the frame loop:
    someone outside it must never reach OCR, crossing detection, or a
    finish event -- exactly the 2025 behavior this reproduces.
    """

    def _corridor_config(self) -> Config:
        config = base_config()
        config.course_boundary = CourseBoundaryConfig(
            enabled=True,
            left_p1=(0.3, 0.0),
            left_p2=(0.3, 1.0),
            right_p1=(0.7, 0.0),
            right_p2=(0.7, 1.0),
        )
        return config

    def test_person_outside_boundary_never_finishes(self):
        config = self._corridor_config()
        # Well to the left of the [0.3, 0.7] corridor (x in [10, 50] of 1000).
        script = {
            i: [person(1, 300.0 + i * 50, x1=10.0, x2=50.0)] for i in range(10)
        }
        pipeline, events = build(script, config=config)
        pipeline.run(frames(10))
        assert events == []
        assert pipeline.stats.people_outside_boundary == 10
        assert pipeline.stats.people_detections == 10  # still detected, just gated

    def test_person_inside_boundary_finishes_normally(self):
        config = self._corridor_config()
        # Default person() box (x1=450, x2=550) sits inside [300, 700].
        script = {i: [person(1, 300.0 + i * 50)] for i in range(10)}
        pipeline, events = build(script, config=config)
        pipeline.run(frames(10))
        assert len(events) == 1
        assert pipeline.stats.people_outside_boundary == 0

    def test_boundary_disabled_by_default_keeps_everyone(self):
        """The regression guard: nothing changes unless a venue opts in."""
        script = {
            i: [person(1, 300.0 + i * 50, x1=10.0, x2=50.0)] for i in range(10)
        }
        pipeline, events = build(script)  # base_config(): boundary not set -> disabled
        pipeline.run(frames(10))
        assert len(events) == 1
        assert pipeline.stats.people_outside_boundary == 0
