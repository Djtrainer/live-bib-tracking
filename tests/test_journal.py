"""Tests for the API's crash-recoverable race journal.

The property under test is the one the operator needs: after any number
of changes, killing the process at any instant loses at most the change in
flight, and a restart puts everything else back. Plus the two files a human
reads without tooling: the running leaderboard and the append-only log.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_backend.journal import RaceJournal, _format_finish


def result(bib, name="", finish_ms=None, source="race_cv", **extra):
    r = {"id": f"id-{bib}-{finish_ms}", "bibNumber": bib, "racerName": name,
         "finishTime": finish_ms, "source": source}
    r.update(extra)
    return r


RUNNING = {"raceStartTime": 1_700_000_000.0, "status": "running", "offset": 0}
STOPPED = {"raceStartTime": None, "status": "stopped", "offset": 0}


class TestRestoreRoundTrip:
    def test_everything_recorded_comes_back(self, tmp_path):
        j = RaceJournal(tmp_path)
        results = [result("120", "Jane", 1_294_500), result("225", "Ann", 1_310_000)]
        j.record("ADD", results, RUNNING, "bib 225 Ann")
        restored = RaceJournal(tmp_path).restore()
        assert restored is not None
        got_results, got_clock = restored
        assert got_results == results
        assert got_clock == RUNNING

    def test_nothing_to_restore_is_none_not_an_error(self, tmp_path):
        assert RaceJournal(tmp_path).restore() is None

    def test_a_corrupt_state_file_is_none_not_an_error(self, tmp_path):
        j = RaceJournal(tmp_path)
        j.state_path.write_text("{not json", encoding="utf-8")
        assert j.restore() is None

    def test_last_write_wins(self, tmp_path):
        j = RaceJournal(tmp_path)
        j.record("ADD", [result("120", finish_ms=1000)], RUNNING)
        j.record("DELETE", [], RUNNING)
        assert j.restore() == ([], RUNNING)


class TestAtomicity:
    def test_no_temp_file_is_left_behind(self, tmp_path):
        j = RaceJournal(tmp_path)
        j.record("ADD", [result("1", finish_ms=1)], RUNNING)
        leftovers = [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_state_is_valid_json_after_every_write(self, tmp_path):
        j = RaceJournal(tmp_path)
        for i in range(25):
            j.record("ADD", [result(str(n), finish_ms=n * 1000) for n in range(i)], RUNNING)
            json.loads(j.state_path.read_text())   # would raise on a half-written file


class TestHumanReadableFiles:
    def test_results_txt_lists_finishers_in_order_with_times(self, tmp_path):
        j = RaceJournal(tmp_path)
        j.record("ADD", [
            result("225", "Ann", 1_310_000),
            result("120", "Jane", 1_294_500),
            result("7", "Not finished", None),
        ], RUNNING)
        text = j.results_path.read_text()
        lines = [l for l in text.splitlines() if l.strip() and l[0].isdigit() or l.lstrip()[:1].isdigit()]
        assert "120" in lines[0] and "21:34.5" in lines[0] and "Jane" in lines[0]
        assert "225" in lines[1] and "21:50.0" in lines[1]
        assert "Not finished" not in text
        assert "clock running" in text

    def test_log_is_append_only_and_timestamped(self, tmp_path):
        # record() reads the clock more than once per call (log line, state
        # file, results header), so the fake must be a settable value, not
        # an iterator that runs dry mid-record.
        now = [1_700_000_000]
        j = RaceJournal(tmp_path, clock=lambda: now[0])
        j.record("CLOCK", [], RUNNING, "start")
        now[0] += 61
        j.record("ADD", [result("120", "Jane", 1000)], RUNNING, "bib 120 Jane 00:01.0")
        lines = j.log_path.read_text().splitlines()
        assert len(lines) == 2
        assert lines[0].split()[1] == "CLOCK" and lines[0].endswith("start")
        assert lines[1].split()[1] == "ADD" and "bib 120 Jane" in lines[1]
        # the first line was not rewritten by the second record
        assert lines[0].split()[0] != lines[1].split()[0]

    def test_a_failing_write_never_raises_into_the_api(self, tmp_path):
        j = RaceJournal(tmp_path)
        j.directory.chmod(0o500)          # read/execute only: writes must fail
        try:
            j.record("ADD", [result("1", finish_ms=1)], RUNNING)   # must not raise
        finally:
            j.directory.chmod(0o700)


class TestArchive:
    def test_fresh_moves_the_old_race_aside_instead_of_deleting_it(self, tmp_path):
        j = RaceJournal(tmp_path, clock=lambda: 1_700_000_000)
        j.record("ADD", [result("120", finish_ms=1000)], RUNNING)
        archived = j.archive()
        assert archived is not None and archived.exists()
        assert j.restore() is None                       # a fresh race starts empty
        assert json.loads(archived.read_text())["results"][0]["bibNumber"] == "120"
        assert not j.log_path.exists() and not j.results_path.exists()

    def test_archive_with_nothing_saved_is_a_no_op(self, tmp_path):
        assert RaceJournal(tmp_path).archive() is None


def test_format_finish():
    assert _format_finish(1_294_500) == "21:34.5"
    assert _format_finish(0) == "00:00.0"
    assert _format_finish(None) == "--:--.-"
