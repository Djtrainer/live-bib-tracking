"""Tests for finish-line geometry and crossing detection.

These cover the failure modes that cost racers on race day: phantom finishers
from re-acquired tracks, order-dependent geometry, and quantised timing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import FinishLineConfig
from race_cv.finish import CrossingDetector, FinishLine, reference_point

W = H = 1000


def horizontal_line(**kwargs) -> tuple[FinishLine, FinishLineConfig]:
    cfg = FinishLineConfig(p1=(0.0, 0.5), p2=(1.0, 0.5), **kwargs)
    return FinishLine(cfg, W, H), cfg


def box_at(cy: float) -> tuple[float, float, float, float]:
    """A 100x200 box whose bottom edge (the reference point) sits at cy."""
    return (450.0, cy - 200.0, 550.0, cy)


class TestGeometry:
    def test_sign_convention_below_is_finished(self):
        line, _ = horizontal_line()
        assert line.signed_distance((500, 600)) > 0
        assert line.signed_distance((500, 400)) < 0

    def test_endpoint_order_does_not_change_sign(self):
        forward = FinishLine(FinishLineConfig(p1=(0.0, 0.5), p2=(1.0, 0.5)), W, H)
        reverse = FinishLine(FinishLineConfig(p1=(1.0, 0.5), p2=(0.0, 0.5)), W, H)
        for point in [(500, 600), (500, 400), (10, 900), (990, 100)]:
            assert forward.signed_distance(point) == pytest.approx(
                reverse.signed_distance(point)
            )

    def test_side_above_inverts(self):
        line = FinishLine(
            FinishLineConfig(p1=(0.0, 0.5), p2=(1.0, 0.5), side="above"), W, H
        )
        assert line.signed_distance((500, 400)) > 0
        assert line.signed_distance((500, 600)) < 0

    def test_sloped_line_matches_hand_computation(self):
        # The legacy race-day geometry: off-frame bottom-left to 78% height right.
        line = FinishLine(FinishLineConfig(p1=(0.0, 1.09), p2=(1.0, 0.78)), W, H)
        # At x=0 the line sits at y=1090, so y=1000 is still short of it.
        assert line.signed_distance((0, 1000)) < 0
        # At x=1000 the line sits at y=780, so y=900 is past it.
        assert line.signed_distance((1000, 900)) > 0

    def test_near_vertical_line_is_supported(self):
        line = FinishLine(FinishLineConfig(p1=(0.5, 0.0), p2=(0.5, 1.0)), W, H)
        assert line.signed_distance((400, 500)) != pytest.approx(0)
        assert line.signed_distance((400, 500)) == -line.signed_distance((600, 500))

    def test_degenerate_line_rejected(self):
        with pytest.raises(ValueError, match="same point"):
            FinishLine(FinishLineConfig(p1=(0.5, 0.5), p2=(0.5, 0.5)), W, H)

    def test_unknown_side_rejected(self):
        with pytest.raises(ValueError, match="side"):
            FinishLine(FinishLineConfig(side="sideways"), W, H)

    def test_reference_point_modes(self):
        bbox = (100.0, 200.0, 300.0, 600.0)
        assert reference_point(bbox, "bottom_center") == (200.0, 600.0)
        assert reference_point(bbox, "center") == (200.0, 400.0)
        assert reference_point(bbox, "top_center") == (200.0, 200.0)
        with pytest.raises(ValueError, match="Unknown reference_point"):
            reference_point(bbox, "elbow")


class TestCrossingDetector:
    def test_fires_once_on_transition(self):
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        assert det.update(1, box_at(400), capture_ts=0.0, frame_index=0) is None
        crossing = det.update(1, box_at(600), capture_ts=1.0, frame_index=1)
        assert crossing is not None
        assert crossing.track_id == 1
        # Further observations past the line must not fire again.
        assert det.update(1, box_at(700), capture_ts=2.0, frame_index=2) is None
        assert det.has_finished(1)

    def test_crossing_time_is_interpolated(self):
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        # Bottom edge goes 480 -> 520, so it touched 500 exactly halfway.
        det.update(1, box_at(480), capture_ts=10.0, frame_index=0)
        crossing = det.update(1, box_at(520), capture_ts=11.0, frame_index=1)
        assert crossing.interpolated
        assert crossing.capture_ts == pytest.approx(10.5)

    def test_interpolation_is_weighted_not_midpoint(self):
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        # 490 -> 590: the line is crossed 10% of the way through the interval.
        det.update(1, box_at(490), capture_ts=0.0, frame_index=0)
        crossing = det.update(1, box_at(590), capture_ts=1.0, frame_index=1)
        assert crossing.capture_ts == pytest.approx(0.1)

    def test_track_first_seen_past_line_is_suppressed(self):
        """A re-acquired track must not invent a finisher."""
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        assert det.update(7, box_at(800), capture_ts=0.0, frame_index=0) is None
        assert 7 in det.suppressed_first_seen_past
        assert not det.has_finished(7)

    def test_suppression_can_be_disabled(self):
        line, cfg = horizontal_line(require_approach=False)
        det = CrossingDetector(line, cfg)
        crossing = det.update(7, box_at(800), capture_ts=0.0, frame_index=0)
        assert crossing is not None
        assert not crossing.interpolated

    def test_approaching_but_not_reaching_does_not_fire(self):
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        for i, y in enumerate([300, 350, 400, 450, 490]):
            assert det.update(1, box_at(y), capture_ts=float(i), frame_index=i) is None

    def test_tracks_are_independent(self):
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        det.update(1, box_at(400), capture_ts=0.0, frame_index=0)
        det.update(2, box_at(400), capture_ts=0.0, frame_index=0)
        assert det.update(1, box_at(600), capture_ts=1.0, frame_index=1) is not None
        assert not det.has_finished(2)
        assert det.update(2, box_at(600), capture_ts=2.0, frame_index=2) is not None

    def test_wide_box_uses_feet_not_leading_corner(self):
        """A box that grows sideways must not trigger an early finish."""
        line, cfg = horizontal_line()
        det = CrossingDetector(line, cfg)
        det.update(1, (450.0, 200.0, 550.0, 480.0), capture_ts=0.0, frame_index=0)
        # Arms fly out: the box widens a lot but the feet have not reached the line.
        assert det.update(1, (100.0, 200.0, 900.0, 495.0), 1.0, 1) is None
