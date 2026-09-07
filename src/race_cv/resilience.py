"""Keep the frame loop alive, and make it obvious when it is not.

Two failure modes found in review that the loop did not survive:

* An exception anywhere in ``Pipeline.process`` -- a transient CoreML error,
  a malformed frame, a numpy edge case -- propagated to the top of the loop
  and ended race timing. One bad frame in ten thousand is not a reason to
  stop timing a race. It is a reason to skip a frame and say so.

* A camera that stops producing frames stalled the loop silently. The
  health line was printed *from inside* the loop, so a stall produced no
  output at all -- the one condition an operator most needs to see looked
  identical to a quiet stretch with nobody finishing.

:class:`ErrorBudget` decides when an error is a skipped frame and when it
is structural (too many in a row) and the process should stop and say why.
:class:`StallWatchdog` runs on its own thread, so it reports precisely when
the loop cannot.
"""

from __future__ import annotations

import threading
import time
from typing import Callable


class ErrorBudget:
    """Tolerate isolated per-frame errors; refuse to hide a persistent one."""

    def __init__(self, max_consecutive: int = 30):
        self.max_consecutive = max_consecutive
        self.total = 0
        self.consecutive = 0
        self.last_error: str | None = None

    def record_ok(self) -> None:
        self.consecutive = 0

    def record_error(self, exc: BaseException) -> bool:
        """Count an error. Returns True when the budget is exhausted."""
        self.total += 1
        self.consecutive += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        return self.exhausted

    @property
    def exhausted(self) -> bool:
        return self.consecutive >= self.max_consecutive


class StallWatchdog:
    """Call ``on_stall(seconds)`` whenever the heartbeat goes stale.

    ``heartbeat`` returns the wall-clock time of the last processed frame (or
    None before the first). Once it is older than ``threshold_s`` the callback
    fires, and keeps firing every ``interval_s`` until frames resume -- a
    stall that lasts a minute should be logged all minute, not once.
    """

    def __init__(
        self,
        heartbeat: Callable[[], float | None],
        on_stall: Callable[[float], None],
        threshold_s: float = 5.0,
        interval_s: float = 5.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._heartbeat = heartbeat
        self._on_stall = on_stall
        self._threshold = threshold_s
        self._interval = interval_s
        self._clock = clock
        self._sleep = sleep
        self._running = False
        self._thread: threading.Thread | None = None
        self.stalls_reported = 0

    def check(self) -> float | None:
        """One evaluation. Returns the stall length if stalled, else None."""
        last = self._heartbeat()
        if last is None:
            return None
        stale = self._clock() - last
        if stale >= self._threshold:
            self.stalls_reported += 1
            self._on_stall(stale)
            return stale
        return None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="stall-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while self._running:
            try:
                self.check()
            except Exception:
                pass  # the watchdog must never be the thing that dies
            self._sleep(self._interval)
