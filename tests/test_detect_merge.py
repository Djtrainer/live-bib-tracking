"""Tests for merging two-stage detections into the full-frame set.

The two-stage pass re-examines a region the full-frame pass already covered,
so a bib visible at both scales is found twice. Concatenating produces stacked
near-identical boxes -- noise for a human reviewing auto-labels, duplicated
supervision if trained on, and a double vote during bib association.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.detect import Detection, Detector

BIB, PERSON = 1, 0


def bib(x1, y1, x2, y2, conf=0.9) -> Detection:
    return Detection(xyxy=(x1, y1, x2, y2), conf=conf, cls=BIB)


class TestMerge:
    def test_duplicate_of_an_existing_box_is_dropped(self):
        primary = [bib(100, 100, 140, 130)]
        extra = [bib(101, 101, 139, 129)]  # same bib, found again at crop scale
        assert len(Detector.merge(primary, extra)) == 1

    def test_a_genuinely_new_box_is_kept(self):
        primary = [bib(100, 100, 140, 130)]
        extra = [bib(600, 400, 650, 440)]
        assert len(Detector.merge(primary, extra)) == 2

    def test_partial_overlap_below_threshold_is_kept(self):
        primary = [bib(100, 100, 200, 200)]
        extra = [bib(180, 180, 280, 280)]  # small corner overlap only
        assert len(Detector.merge(primary, extra)) == 2

    def test_boxes_of_different_classes_never_suppress_each_other(self):
        """A bib sits inside its person box; that is not a duplicate."""
        person = Detection(xyxy=(100, 100, 140, 130), conf=0.9, cls=PERSON)
        assert len(Detector.merge([person], [bib(100, 100, 140, 130)])) == 2

    def test_empty_inputs(self):
        assert Detector.merge([], []) == []
        assert len(Detector.merge([], [bib(1, 1, 5, 5)])) == 1
        assert len(Detector.merge([bib(1, 1, 5, 5)], [])) == 1

    def test_extra_boxes_dedupe_against_each_other_too(self):
        primary = []
        extra = [bib(10, 10, 50, 50), bib(11, 11, 49, 49), bib(300, 300, 340, 340)]
        assert len(Detector.merge(primary, extra)) == 2

    def test_zero_area_box_does_not_divide_by_zero(self):
        assert len(Detector.merge([bib(10, 10, 10, 10)], [bib(10, 10, 10, 10)])) == 2

    def test_threshold_is_respected(self):
        primary = [bib(100, 100, 200, 200)]
        extra = [bib(150, 100, 250, 200)]  # IoU = 1/3
        assert len(Detector.merge(primary, extra, iou=0.9)) == 2
        assert len(Detector.merge(primary, extra, iou=0.3)) == 1
