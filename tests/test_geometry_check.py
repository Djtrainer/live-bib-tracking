"""Tests for the ROI-vs-course cross-check.

Cropping is the only setting in this system that can lose a racer with
nothing to show for it: someone outside the crop is absent from the input,
so no counter, log line or overlay downstream can report them missing. The
2025 pipeline cropped the left ~28% of every frame and said nothing. These
tests exist so a crop that disagrees with the course boundary is loud.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import Config, CourseBoundaryConfig, FinishLineConfig, RoiConfig
from race_cv.geometry_check import check_roi_covers_course, describe_roi

W, H = 1920, 1080


def config_with(roi_left: float | None, boundary_left=(0.31, 0.285)) -> Config:
    """A config whose course runs between boundary_left and the right edge."""
    p1x, p2x = boundary_left
    return Config(
        roi=RoiConfig(
            enabled=roi_left is not None,
            polygon=(
                []
                if roi_left is None
                else [[roi_left, 0.0], [1.0, 0.0], [1.0, 1.0], [roi_left, 1.0]]
            ),
        ),
        course_boundary=CourseBoundaryConfig(
            enabled=True,
            left_p1=(p1x, 1.0), left_p2=(p2x, 0.49),
            right_p1=(1.0, 0.78), right_p2=(0.32, 0.49),
        ),
        finish_line=FinishLineConfig(p1=(0.0, 1.09), p2=(1.0, 0.78)),
    )


class TestCropVersusCourse:
    def test_disabled_roi_is_always_fine(self):
        assert check_roi_covers_course(config_with(None), W, H) == []

    def test_a_crop_left_of_the_boundary_passes(self):
        assert check_roi_covers_course(config_with(0.26), W, H) == []

    def test_a_crop_inside_the_boundary_is_reported(self):
        """The exact 2025 failure: a crop eating live course, silently."""
        warnings = check_roi_covers_course(config_with(0.333), W, H)
        assert len(warnings) == 1
        assert "course_boundary extends to" in warnings[0]

    def test_the_warning_names_a_value_that_would_fix_it(self):
        warnings = check_roi_covers_course(config_with(0.333), W, H)
        assert "<= 0.2" in warnings[0], warnings[0]

    def test_a_crop_exactly_on_the_boundary_is_accepted(self):
        assert check_roi_covers_course(config_with(0.285), W, H) == []

    def test_severity_does_not_depend_on_frame_resolution(self):
        """Geometry is normalized, so a 4K camera must reach the same verdict."""
        assert len(check_roi_covers_course(config_with(0.333), 3840, 2160)) == 1
        assert check_roi_covers_course(config_with(0.26), 3840, 2160) == []


class TestDescribeRoi:
    def test_returns_nothing_when_disabled(self):
        assert describe_roi(config_with(None), W, H) is None

    def test_reports_the_resolution_actually_gained(self):
        """The reason to crop at all. Full frame squeezes 1920 into 1280
        (x0.667); cropping to 1421 wide raises that to x0.901."""
        summary = describe_roi(config_with(0.26), W, H)
        assert "1421x1080" in summary
        assert "x1.35" in summary, summary

    def test_a_tighter_crop_reports_a_larger_gain(self):
        assert "x1.50" in describe_roi(config_with(0.333), W, H)
