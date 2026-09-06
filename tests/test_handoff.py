"""Tests for recovering a crossing across a track break.

Both genuine misses on the 13-clip set had one signature: a racer
approaches the line for ~40 frames, the track dies a few frames short of
it as she becomes large and clipped by the frame edge, and a new track is
born a few frames past it. No track saw the sign change, and the newborn
was suppressed as "first seen past the line". These tests pin down when a
newborn past the line is a continuation and, just as importantly, when it
is not -- the suppression exists because reacquired tracks invent ghosts.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.capture import Frame
from race_cv.config import Config, FinishLineConfig, OcrConfig, PipelineConfig
from race_cv.detect import Detection
from race_cv.finish import CrossingDetector, FinishLine
from race_cv.ocr import BibRead, BibVoter
from race_cv.pipeline import Pipeline

W = H = 1000
FPS = 10.0


def line_config(**over) -> FinishLineConfig:
    base = dict(p1=(0.0, 0.5), p2=(1.0, 0.5), side="below", confirm_frames=2,
                require_approach=True, handoff_window_s=1.0,
                handoff_max_distance=0.35, handoff_min_gap_s=0.05)
    base.update(over)
    return FinishLineConfig(**base)


def box(bottom_y, x=500.0):
    return (x - 50.0, bottom_y - 200.0, x + 50.0, bottom_y)


def approach(det, track, bottoms, t0=0.0):
    """Feed a track walking toward the line (y=500). Returns the last ts."""
    ts = t0
    for b in bottoms:
        assert det.update(track, box(b), ts, int(ts * FPS)) is None
        ts += 1.0 / FPS
    return ts - 1.0 / FPS


class TestHandoff:
    def test_newborn_past_the_line_continues_a_vanished_approacher(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [300, 350, 400, 440, 470])     # closing, 30px short
        # track 1 vanishes; track 2 appears 0.3s later, 40px past the line
        crossing = det.update(2, box(540), last + 0.3, 99)
        assert crossing is not None
        assert crossing.track_id == 2
        assert crossing.predecessor_track_id == 1
        assert crossing.interpolated
        assert det.handoffs == 1
        assert 2 not in det.suppressed_first_seen_past

    def test_crossing_time_is_interpolated_across_the_gap(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [300, 400, 470])                # ends 30px short at t=last
        crossing = det.update(2, box(530), last + 0.3, 99)      # 30px past, 0.3s later
        # symmetric distances -> the crossing is halfway through the gap
        assert crossing.capture_ts == pytest.approx(last + 0.15, abs=1e-6)

    def test_predecessor_may_only_hand_off_once(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [300, 400, 470])
        assert det.update(2, box(540), last + 0.2, 99) is not None
        # a second newborn past the line must NOT also claim track 1
        assert det.update(3, box(560), last + 0.4, 100) is None
        assert 3 in det.suppressed_first_seen_past


class TestNotAHandoff:
    """Every case here is a ghost the old guard was written to stop."""

    def test_no_predecessor_at_all_is_still_suppressed(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        assert det.update(7, box(600), 10.0, 100) is None
        assert 7 in det.suppressed_first_seen_past

    def test_a_predecessor_that_vanished_too_long_ago_does_not_count(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [300, 400, 470])
        assert det.update(2, box(540), last + 5.0, 99) is None   # 5s > 1s window
        assert 2 in det.suppressed_first_seen_past

    def test_a_predecessor_far_from_the_line_does_not_count(self):
        """Someone who wandered off 400px short of the line did not finish."""
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [50, 80, 100])                    # 400px short
        assert det.update(2, box(540), last + 0.3, 99) is None
        assert 2 in det.suppressed_first_seen_past

    def test_a_predecessor_moving_away_does_not_count(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [470, 440, 400])                  # retreating
        assert det.update(2, box(540), last + 0.3, 99) is None

    def test_a_still_alive_neighbour_is_not_a_predecessor(self):
        """Two people: one approaching, one born past the line the SAME frame."""
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = approach(det, 1, [300, 400, 470])
        # track 1 is observed again on this very frame (still alive) ...
        assert det.update(1, box(480), last + 0.1, 50) is None
        # ... so a newborn past the line on the same frame is someone else
        assert det.update(2, box(540), last + 0.1, 50) is None
        assert 2 in det.suppressed_first_seen_past
        # and track 1 still gets its own crossing when it actually crosses
        assert det.update(1, box(520), last + 0.2, 51) is not None

    def test_disabled_by_zero_window(self):
        cfg = line_config(handoff_window_s=0.0)
        det = CrossingDetector(FinishLine(cfg, W, H), cfg)
        last = approach(det, 1, [300, 400, 470])
        assert det.update(2, box(540), last + 0.3, 99) is None


class TestVoteTransfer:
    def test_reads_and_lock_move_to_the_new_track(self):
        voter = BibVoter(OcrConfig(lock_conf=0.99), roster={"120"})
        voter.add(1, BibRead("120", 0.995, 0.9))      # locks
        voter.add(1, BibRead("120", 0.80, 0.9))
        voter.transfer(1, 2)
        verdict = voter.resolve(2)
        assert verdict.text == "120" and verdict.locked and verdict.votes == 2
        assert voter.resolve(1).text is None


class ScriptedDetector:
    def __init__(self, script):
        self.script = script

    def track(self, image):
        return self.script.get(getattr(image, "frame_index", 0), [])

    def split(self, detections):
        return ([d for d in detections if d.cls == 0 and d.track_id is not None],
                [d for d in detections if d.cls == 1])


class Tagged(np.ndarray):
    @classmethod
    def make(cls, i):
        a = np.zeros((4, 4, 3), dtype=np.uint8).view(cls)
        a.frame_index = i
        return a


def person(tid, bottom):
    return Detection(xyxy=box(bottom), conf=0.9, cls=0, track_id=tid)


class TestPipelineInheritsHistory:
    """The recovery is worthless if the newborn then fails min_observations
    or resolves with no bib. Both must come from the track that died."""

    def _run(self):
        # track 1 approaches for 6 frames, dies; track 2 born past the line
        script = {i: [person(1, 300 + i * 30)] for i in range(6)}    # ends at 450
        script[8] = [person(2, 560)]
        for i in range(9, 14):
            script[i] = [person(2, 560)]
        config = Config(
            finish_line=line_config(min_observations=5),
            ocr=OcrConfig(enabled=False),
            pipeline=PipelineConfig(target_fps=0.0),
        )
        events = []
        p = Pipeline(config=config, detector=ScriptedDetector(script), frame_width=W,
                     frame_height=H, run_id="t", bib_reader=None, roster={"120"},
                     emit=events.append)
        p.voter.add(1, BibRead("120", 0.995, 0.9))   # a read that happened on track 1
        p.run([Frame(image=Tagged.make(i), capture_ts=i / FPS, index=i) for i in range(14)])
        return p, events

    def test_finisher_is_emitted_with_inherited_observations_and_bib(self):
        p, events = self._run()
        assert len(events) == 1
        assert events[0].track_id == 2
        assert events[0].bib_number == "120"
        assert events[0].track_observations >= 6
        assert p.stats.handoffs == 1
        assert p.stats.finishes_below_min_observations == 0


class TestDirectionIsJudgedOverAWindow:
    """Reproduces the 14-48-12 trace: 60px of net approach, then a jittery
    plateau whose final step is 0.0 or slightly negative, then the track
    dies and a newborn appears past the line."""

    def _jittery_approach(self, det):
        # 0.6s of approach (+60px) then 0.4s of plateau ending in -0.0
        bottoms = [180, 195, 206, 210, 207, 205, 204, 228, 236, 239, 239,
                   238, 237, 236, 239, 240, 240, 240]
        ts = 0.0
        for b in bottoms:
            assert det.update(1, box(b), ts, int(ts * 30)) is None
            ts += 1.0 / 30.0
        return ts - 1.0 / 30.0

    def test_a_jittery_but_closing_predecessor_hands_off(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        last = self._jittery_approach(det)          # line is at y=500; ends 260px short
        crossing = det.update(2, box(530), last + 0.43, 99)
        assert crossing is not None
        assert crossing.predecessor_track_id == 1

    def test_a_genuinely_retreating_predecessor_still_does_not(self):
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        ts = 0.0
        for b in [300, 290, 280, 270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160]:
            assert det.update(1, box(b), ts, int(ts * 30)) is None
            ts += 1.0 / 30.0
        assert det.update(2, box(530), ts + 0.3, 99) is None
        assert 2 in det.suppressed_first_seen_past

    def test_a_stationary_predecessor_does_not(self):
        """Net progress must be strictly positive."""
        det = CrossingDetector(FinishLine(line_config(), W, H), line_config())
        ts = 0.0
        for _ in range(20):
            assert det.update(1, box(400), ts, int(ts * 30)) is None
            ts += 1.0 / 30.0
        assert det.update(2, box(530), ts + 0.3, 99) is None
