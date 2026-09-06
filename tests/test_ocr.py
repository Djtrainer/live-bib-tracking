"""Tests for bib vote aggregation.

The locking tests are a regression guard for a failure found on real footage:
a racer wearing 120 was read as "20" at 0.999 confidence, which locked the
wrong number in and stopped the pipeline from ever looking again. A lock ends
the search, so it must not be grantable to a number that cannot be correct.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.config import OcrConfig
from race_cv.ocr import BibRead, BibVoter


def read(text, ocr=0.9, yolo=0.9) -> BibRead:
    return BibRead(text=text, ocr_conf=ocr, yolo_conf=yolo)


class TestPlausibility:
    def test_single_digit_bibs_are_allowed(self):
        voter = BibVoter(OcrConfig())
        voter.add(1, read("7"))
        assert voter.resolve(1).text == "7"

    def test_non_numeric_is_rejected(self):
        voter = BibVoter(OcrConfig())
        voter.add(1, read("12A"))
        assert voter.resolve(1).text is None

    def test_too_long_is_rejected(self):
        voter = BibVoter(OcrConfig(max_len=3))
        voter.add(1, read("12345"))
        assert voter.resolve(1).text is None


class TestVoting:
    def test_confidence_weighted_winner(self):
        voter = BibVoter(OcrConfig())
        voter.add(1, read("121", ocr=0.5, yolo=0.5))
        voter.add(1, read("999", ocr=0.9, yolo=0.9))
        assert voter.resolve(1).text == "999"

    def test_repeated_reads_accumulate(self):
        """Enough weak agreeing reads outweigh one strong outlier.

        Each 121 read scores 0.5*0.5 = 0.25; the lone 999 scores 0.9*0.9 =
        0.81, so it takes four of them to overtake it.
        """
        voter = BibVoter(OcrConfig())
        for _ in range(4):
            voter.add(1, read("121", ocr=0.5, yolo=0.5))
        voter.add(1, read("999", ocr=0.9, yolo=0.9))
        assert voter.resolve(1).text == "121"

    def test_low_confidence_reads_are_filtered(self):
        voter = BibVoter(OcrConfig(min_ocr_conf=0.4))
        voter.add(1, read("121", ocr=0.1))
        assert voter.resolve(1).text is None

    def test_tracks_are_independent(self):
        voter = BibVoter(OcrConfig())
        voter.add(1, read("121"))
        voter.add(2, read("225"))
        assert voter.resolve(1).text == "121"
        assert voter.resolve(2).text == "225"


class TestRoster:
    def test_roster_candidate_beats_higher_scoring_stranger(self):
        voter = BibVoter(OcrConfig(), roster={"121"})
        voter.add(1, read("7", ocr=0.95))
        voter.add(1, read("121", ocr=0.5))
        verdict = voter.resolve(1)
        assert verdict.text == "121"
        assert verdict.in_roster

    def test_falls_back_when_nothing_is_in_the_roster(self):
        """Reporting an off-roster number beats reporting nothing at all."""
        voter = BibVoter(OcrConfig(), roster={"121"})
        voter.add(1, read("20", ocr=0.8))
        verdict = voter.resolve(1)
        assert verdict.text == "20"
        assert not verdict.in_roster


class TestLocking:
    def test_high_confidence_read_locks(self):
        voter = BibVoter(OcrConfig(lock_conf=0.99))
        voter.add(1, read("121", ocr=0.999))
        assert voter.is_locked(1)
        assert voter.resolve(1).locked

    def test_lock_short_circuits_later_votes(self):
        voter = BibVoter(OcrConfig(lock_conf=0.99))
        voter.add(1, read("121", ocr=0.999))
        for _ in range(10):
            voter.add(1, read("999", ocr=0.9))
        assert voter.resolve(1).text == "121"

    def test_off_roster_read_cannot_lock(self):
        """The 120-read-as-20 regression.

        A lock ends the search, so a number nobody is wearing must not get
        one no matter how confident the read.
        """
        voter = BibVoter(OcrConfig(lock_conf=0.99), roster={"120", "121", "225"})
        voter.add(1, read("20", ocr=0.999))
        assert not voter.is_locked(1)

    def test_search_continues_and_can_still_be_corrected(self):
        voter = BibVoter(OcrConfig(lock_conf=0.99), roster={"120", "121", "225"})
        voter.add(1, read("20", ocr=0.999))   # confident but impossible
        voter.add(1, read("120", ocr=0.55))   # weaker, but a real bib
        verdict = voter.resolve(1)
        assert verdict.text == "120"
        assert verdict.in_roster

    def test_roster_read_still_locks_normally(self):
        voter = BibVoter(OcrConfig(lock_conf=0.99), roster={"120"})
        voter.add(1, read("120", ocr=0.999))
        assert voter.is_locked(1)

    def test_without_a_roster_any_confident_read_locks(self):
        """No roster means no way to know a number is impossible."""
        voter = BibVoter(OcrConfig(lock_conf=0.99))
        voter.add(1, read("20", ocr=0.999))
        assert voter.is_locked(1)
