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
        """An off-roster number is reported once it is corroborated.

        Reporting it beats reporting nothing -- rosters are incomplete more
        often than never -- but only when there is agreement. A single read
        is what a bibless racer looks like after a few hundred frames of OCR
        on a logo, and it credits their finish to a stranger; see
        TestOffRosterNeedsAgreement for that case.
        """
        voter = BibVoter(OcrConfig(), roster={"121"})
        voter.add(1, read("20", ocr=0.8))
        voter.add(1, read("20", ocr=0.7))
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


class TestOffRosterNeedsAgreement:
    """A single off-roster read must not win a verdict when a roster is loaded.

    Found on the 13-clip set: a racer with no bib, tracked for 346 frames,
    collected one OCR read of "1" and was credited with bib 1. Real bibs on
    the same footage resolve with 3+ reads at 0.99+. One read is noise; two
    agreeing off-roster reads are kept, because an incomplete roster is a
    thing that happens.
    """

    def _voter(self, min_votes=2):
        return BibVoter(OcrConfig(lock_conf=0.99, min_votes_off_roster=min_votes),
                        roster={"120", "225"})

    def test_a_single_off_roster_read_resolves_to_unknown(self):
        v = self._voter()
        v.add(1, read("1", ocr=0.85))
        verdict = v.resolve(1)
        assert verdict.text is None
        assert verdict.votes == 1          # the evidence is reported, not hidden

    def test_two_agreeing_off_roster_reads_are_accepted(self):
        v = self._voter()
        v.add(1, read("77", ocr=0.8)); v.add(1, read("77", ocr=0.7))
        assert v.resolve(1).text == "77"

    def test_two_disagreeing_off_roster_reads_are_still_noise(self):
        v = self._voter()
        v.add(1, read("1", ocr=0.9)); v.add(1, read("7", ocr=0.9))
        assert v.resolve(1).text is None

    def test_on_roster_bibs_are_untouched_by_the_rule(self):
        """One good read of a real bib still resolves, as before."""
        v = self._voter()
        v.add(1, read("120", ocr=0.85))
        assert v.resolve(1).text == "120"

    def test_a_roster_read_still_beats_repeated_off_roster_reads(self):
        v = self._voter()
        for _ in range(5): v.add(1, read("1", ocr=0.9))
        v.add(1, read("120", ocr=0.6))
        assert v.resolve(1).text == "120"

    def test_without_a_roster_the_rule_does_not_apply(self):
        v = BibVoter(OcrConfig(lock_conf=0.99, min_votes_off_roster=2))
        v.add(1, read("1", ocr=0.85))
        assert v.resolve(1).text == "1"

    def test_setting_one_restores_the_old_behaviour(self):
        v = self._voter(min_votes=1)
        v.add(1, read("1", ocr=0.85))
        assert v.resolve(1).text == "1"
