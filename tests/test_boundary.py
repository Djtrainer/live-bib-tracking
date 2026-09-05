"""Tests for the course-boundary gate.

This reproduces the 2025 setup's guide_line_left / guide_line_right gate,
which kept spectators and passersby off the leaderboard by only tracking
people between two lines that narrow toward the horizon like a driveway does
in perspective. Several tests below hand-compute against the *actual* 2025
numbers (frame_width * 0.31 etc.) rather than arbitrary values, so a
transcription error in porting them would be caught here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.boundary import CourseBoundary, _interpolate_x_at_y
from race_cv.config import CourseBoundaryConfig

W, H = 1920, 1080


def boundary_2025(**overrides) -> CourseBoundary:
    """The exact 2025 guide_line_left / guide_line_right geometry."""
    config = CourseBoundaryConfig(
        enabled=True,
        left_p1=(0.31, 1.0),
        left_p2=(0.285, 0.49),
        right_p1=(1.0, 0.78),
        right_p2=(0.32, 0.49),
        **overrides,
    )
    return CourseBoundary(config, W, H)


class TestInterpolation:
    def test_matches_hand_computation(self):
        # Vertical-ish line: x goes 100 -> 200 as y goes 0 -> 100.
        assert _interpolate_x_at_y((100, 0), (200, 100), 50) == pytest.approx(150)
        assert _interpolate_x_at_y((100, 0), (200, 100), 0) == pytest.approx(100)
        assert _interpolate_x_at_y((100, 0), (200, 100), 100) == pytest.approx(200)

    def test_extrapolates_beyond_the_segment(self):
        # A person's corner can sit above/below the line's own p1/p2 span;
        # the gate still needs an x to compare against there.
        assert _interpolate_x_at_y((100, 0), (200, 100), 200) == pytest.approx(300)

    def test_horizontal_line_falls_back_to_p1_x(self):
        assert _interpolate_x_at_y((50, 10), (999, 10), 500) == 50


class Test2025Geometry:
    """Cross-checks against the exact 2025 numbers, hand-computed."""

    def test_left_line_at_bottom_of_frame(self):
        boundary = boundary_2025()
        # guide_line_left p1 = (0.31*W, 1.0*H) -- exactly the bottom of frame.
        assert boundary.left_p1 == pytest.approx((0.31 * W, 1.0 * H))

    def test_right_line_pixel_endpoints(self):
        boundary = boundary_2025()
        assert boundary.right_p1 == pytest.approx((1.0 * W, 0.78 * H))
        assert boundary.right_p2 == pytest.approx((0.32 * W, 0.49 * H))

    def test_point_at_bottom_center_is_inside(self):
        # At y=H (bottom), left boundary is at x=0.31*W, right boundary's
        # line (right_p1=(W,0.78H), right_p2=(0.32W,0.49H)) extrapolated to
        # y=H sits past the right edge, so the bottom-center of a 1920-wide
        # frame is comfortably inside.
        boundary = boundary_2025()
        assert boundary.contains_point((W * 0.6, H))

    def test_point_far_left_at_bottom_is_outside(self):
        boundary = boundary_2025()
        assert not boundary.contains_point((10, H))

    def test_point_right_of_the_right_line_is_outside(self):
        # right_p1=(W, 0.78H) sits above the very bottom of frame, so the
        # right line only meaningfully restricts anything above that y --
        # pick a y within the segment itself (not below right_p1, where the
        # line is simply never evaluated against anything in the original
        # geometry) so this isn't relying on extrapolation past the frame.
        boundary = boundary_2025()
        y = H * 0.6
        assert boundary.contains_point((1050, y))  # just inside x_right≈1109.6
        assert not boundary.contains_point((1150, y))  # just outside it


class TestBoxGating:
    def test_box_fully_inside_is_kept(self):
        boundary = boundary_2025()
        # A box in the middle-lower part of frame, well within both lines.
        assert boundary.contains_box((W * 0.5, H * 0.9, W * 0.6, H * 0.98))

    def test_box_fully_outside_left_is_excluded(self):
        boundary = boundary_2025()
        assert not boundary.contains_box((0, H * 0.9, W * 0.05, H * 0.98))

    def test_box_straddling_edge_counts_as_inside(self):
        """Matches 2025 semantics: any corner inside keeps the whole box."""
        boundary = boundary_2025()
        # Left edge of the box sits outside, right edge sits inside.
        left_edge_x = boundary.left_p1[0]
        box = (left_edge_x - 50, H * 0.95, left_edge_x + 50, H)
        assert boundary.contains_box(box)

    def test_disabled_boundary_flag_is_off(self):
        config = CourseBoundaryConfig()  # enabled defaults to False
        boundary = CourseBoundary(config, W, H)
        assert boundary.enabled is False

    def test_pixel_lines_returns_both_segments(self):
        boundary = boundary_2025()
        left, right = boundary.pixel_lines()
        assert left == (boundary.left_p1, boundary.left_p2)
        assert right == (boundary.right_p1, boundary.right_p2)
