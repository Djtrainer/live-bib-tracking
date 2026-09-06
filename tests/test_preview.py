"""Tests for the preview rate gate.

The gate exists because the preview window was costing a third of the
detector's throughput (29.9 -> 19.4 fps on the real service). The property
that matters is the one the old pacing bug taught: falling behind must never
be followed by a catch-up burst, because a burst of imshow calls is exactly
the stall it was meant to prevent.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.preview import RateGate, downscale


class TestRateGate:
    def test_first_call_is_always_due(self):
        assert RateGate(10.0).due(100.0)

    def test_admits_at_most_fps_per_second(self):
        gate = RateGate(10.0)
        # 30 frames over one second, 1/30 s apart
        admitted = sum(gate.due(100.0 + i / 30.0) for i in range(30))
        assert admitted == 10

    def test_rejects_inside_the_interval(self):
        gate = RateGate(10.0)
        assert gate.due(100.0)
        assert not gate.due(100.05)
        assert gate.due(100.10)

    def test_a_stall_yields_one_event_not_a_burst(self):
        """Fall 1s behind at 10fps; the next call is due exactly once."""
        gate = RateGate(10.0)
        assert gate.due(100.0)
        assert gate.due(101.0)          # the stall ends
        assert not gate.due(101.01)     # NOT nine more to "catch up"
        assert not gate.due(101.05)
        assert gate.due(101.10)

    def test_zero_means_every_frame(self):
        gate = RateGate(0.0)
        assert all(gate.due(100.0 + i / 30.0) for i in range(30))


class TestDownscale:
    def test_halves_each_dimension(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert downscale(image, 0.5).shape == (540, 960, 3)

    def test_scale_one_is_a_no_op(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        assert downscale(image, 1.0) is image

    def test_nonsense_scale_is_a_no_op_not_a_crash(self):
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        assert downscale(image, 0.0) is image
        assert downscale(image, -1.0) is image
