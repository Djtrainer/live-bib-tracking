"""A running, crash-recoverable record of the race, kept by the API.

Why this exists
---------------
On race day the API runs with ``--no-processor``, and in that mode its
leaderboard snapshot returned early: every result and the race clock lived
only in memory. ``race_cv`` keeps its own durable log of detections
(``events.jsonl``), but that never sees what the operator does in Live
Management -- the manual add for a racer the camera missed, the bib fixed
from "20" to "120", the deletion of a ghost -- nor when the clock started.
If the API process died, all of that was gone and there was nothing to
restore from.

Three files, all under the results directory, written on every change:

``race_state.json``
    The complete state: every result and the clock. Written atomically
    (temp file + rename), so a crash mid-write leaves the previous good
    copy, never a half file. This is what a restart restores from.

``race_results.txt``
    The current leaderboard as a person would read it -- place, bib, name,
    time, where it came from. Rewritten on every change. Exists so that
    with the API down and a runner at the booth, the answer is one file
    away, no tooling required.

``race_log.txt``
    One line per change, appended, never rewritten: what happened, when,
    to whom. Append-only means a crash cannot lose what was already
    written, and a dispute afterwards has the sequence, not just the
    final state.

The API calls :meth:`RaceJournal.record` at every mutation and
:meth:`RaceJournal.restore` at startup. ``--fresh`` archives an existing
state file instead of restoring it.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _format_finish(finish_ms: float | int | None) -> str:
    """MM:SS.t from milliseconds; blank when the racer has not finished."""
    if finish_ms is None:
        return "--:--.-"
    total = float(finish_ms) / 1000.0
    minutes, seconds = divmod(max(total, 0.0), 60.0)
    return f"{int(minutes):02d}:{seconds:04.1f}"


class RaceJournal:
    def __init__(self, directory: str | Path, clock=time.time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "race_state.json"
        self.results_path = self.directory / "race_results.txt"
        self.log_path = self.directory / "race_log.txt"
        self._clock = clock

    # ----------------------------------------------------------------- write

    def record(
        self,
        action: str,
        results: list[dict[str, Any]],
        clock_state: dict[str, Any],
        detail: str = "",
    ) -> None:
        """Persist everything after one change. Never raises into the API.

        Order matters: the log line first, because it is the cheapest and
        the one that must survive; then the state, atomically; then the
        human-readable leaderboard, which is derived and can be rebuilt.
        """
        try:
            self._append_log(action, detail)
        except Exception:
            pass
        try:
            self._write_state(results, clock_state)
        except Exception:
            pass
        try:
            self._write_results_txt(results, clock_state)
        except Exception:
            pass

    def _append_log(self, action: str, detail: str) -> None:
        stamp = datetime.fromtimestamp(self._clock()).strftime("%H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{stamp}  {action:<8} {detail}\n")

    def _write_state(self, results: list[dict[str, Any]], clock_state: dict[str, Any]) -> None:
        payload = {
            "saved_at": self._clock(),
            "clock": clock_state,
            "results": results,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.state_path)  # atomic on POSIX: old or new, never half

    def _write_results_txt(self, results: list[dict[str, Any]], clock_state: dict[str, Any]) -> None:
        finished = [r for r in results if r.get("finishTime") is not None]
        finished.sort(key=lambda r: r.get("finishTime"))
        start = clock_state.get("raceStartTime")
        started = (
            datetime.fromtimestamp(start).strftime("%H:%M:%S") if start else "not started"
        )
        lines = [
            f"Race results  (clock {clock_state.get('status', '?')}, started {started}; "
            f"written {datetime.fromtimestamp(self._clock()).strftime('%H:%M:%S')})",
            f"{'place':>5}  {'bib':>6}  {'time':>8}  {'name':<28} source",
            "-" * 64,
        ]
        for place, r in enumerate(finished, 1):
            lines.append(
                f"{place:>5}  {str(r.get('bibNumber') or '?'):>6}  "
                f"{_format_finish(r.get('finishTime')):>8}  "
                f"{str(r.get('racerName') or ''):<28} {r.get('source') or ''}"
            )
        if not finished:
            lines.append("(no finishers yet)")
        tmp = self.results_path.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, self.results_path)

    # --------------------------------------------------------------- restore

    def restore(self) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
        """The last saved state, or None if there is none or it is unreadable."""
        if not self.state_path.is_file():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return list(payload["results"]), dict(payload["clock"])
        except Exception:
            return None

    def archive(self) -> Path | None:
        """Move an existing state aside (for --fresh) rather than delete it."""
        if not self.state_path.is_file():
            return None
        stamp = datetime.fromtimestamp(self._clock()).strftime("%Y%m%d_%H%M%S")
        for path in (self.state_path, self.results_path, self.log_path):
            if path.is_file():
                path.rename(path.with_name(f"{path.stem}_{stamp}{path.suffix}"))
        return self.directory / f"race_state_{stamp}.json"
