"""Tests for the frame loop's survival mechanisms."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from race_cv.resilience import ErrorBudget, StallWatchdog


class TestErrorBudget:
    def test_isolated_errors_are_tolerated(self):
        budget = ErrorBudget(max_consecutive=3)
        for _ in range(10):
            assert budget.record_error(ValueError("bad frame")) is False
            budget.record_ok()
        assert budget.total == 10 and budget.consecutive == 0

    def test_a_run_of_errors_exhausts_the_budget(self):
        budget = ErrorBudget(max_consecutive=3)
        assert budget.record_error(RuntimeError("1")) is False
        assert budget.record_error(RuntimeError("2")) is False
        assert budget.record_error(RuntimeError("3")) is True
        assert budget.exhausted
        assert budget.last_error == "RuntimeError: 3"

    def test_a_success_resets_the_run(self):
        budget = ErrorBudget(max_consecutive=2)
        budget.record_error(RuntimeError("a"))
        budget.record_ok()
        assert budget.record_error(RuntimeError("b")) is False


class TestStallWatchdog:
    def _make(self, last, now, fired):
        return StallWatchdog(
            heartbeat=lambda: last[0], on_stall=lambda s: fired.append(s),
            threshold_s=5.0, interval_s=1.0, clock=lambda: now[0], sleep=lambda _: None,
        )

    def test_quiet_before_the_first_frame(self):
        fired = []
        assert self._make([None], [100.0], fired).check() is None
        assert fired == []

    def test_fresh_heartbeat_does_not_fire(self):
        fired = []
        assert self._make([99.0], [100.0], fired).check() is None
        assert fired == []

    def test_stale_heartbeat_fires_with_the_stall_length(self):
        fired = []
        w = self._make([90.0], [100.0], fired)
        assert w.check() == 10.0
        assert fired == [10.0] and w.stalls_reported == 1

    def test_keeps_firing_while_stalled_and_stops_when_frames_resume(self):
        fired = []
        last, now = [90.0], [100.0]
        w = self._make(last, now, fired)
        w.check(); now[0] = 105.0; w.check()
        assert len(fired) == 2
        last[0] = 105.0                      # frames resumed
        assert w.check() is None
        assert len(fired) == 2

    def test_thread_lifecycle(self):
        fired = []
        w = StallWatchdog(heartbeat=lambda: None, on_stall=fired.append,
                          threshold_s=5.0, interval_s=0.01)
        w.start(); w.start()                 # idempotent
        w.stop(); w.stop()
        assert fired == []
