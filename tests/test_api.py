"""Tests for the results API on the paths race day actually exercises.

Until now the API had no tests at all. These pin the behaviours that lose
racers or corrupt times when they go wrong:

* a finish posted while the clock is stopped is REFUSED, not stored with a
  nonsense time -- and the sink retries, so nothing is lost;
* a re-delivered finish (same eventId) does not become a second finisher;
* Start Clock pressed twice does not silently re-base every time already
  recorded; Reset with results present is refused unless forced;
* every change is journaled and a restart restores it;
* the built site is served, with the SPA fallback /admin needs.

The journal is redirected to a temp directory for every test, and the
module-level race state is reset, so tests neither touch data/results/
nor leak into each other.
"""

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_backend import local_server as server  # noqa: E402
from api_backend.journal import RaceJournal  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Fresh in-memory race and a throwaway journal for each test."""
    monkeypatch.setattr(server, "journal", RaceJournal(tmp_path))
    server.race_results.clear()
    server.race_clock_state.clear()
    server.race_clock_state.update({"raceStartTime": None, "status": "stopped", "offset": 0})
    server.app_state.pop("fresh", None)
    yield tmp_path


@pytest.fixture
def client():
    with TestClient(server.app) as c:
        yield c


def finish(bib="120", event_id=None, source="race_cv", **extra):
    now = time.time()
    body = {"bibNumber": bib, "captureTime": now, "wallClockTime": now, "source": source}
    if event_id:
        body["eventId"] = event_id
    body.update(extra)
    return body


class TestClockGuards:
    def test_start_then_status(self, client):
        assert client.post("/api/clock/start").json()["success"]
        assert client.get("/api/clock/status").json()["data"]["status"] == "running"

    def test_starting_a_running_clock_is_refused(self, client):
        """The 2025-class failure: a second press re-bases every recorded time."""
        client.post("/api/clock/start")
        started = client.get("/api/clock/status").json()["data"]["raceStartTime"]
        time.sleep(0.01)
        second = client.post("/api/clock/start").json()
        assert second["success"] is False
        assert client.get("/api/clock/status").json()["data"]["raceStartTime"] == started

    def test_reset_with_results_is_refused_without_force(self, client):
        client.post("/api/clock/start")
        assert client.post("/api/results", json=finish()).json()["success"]
        assert client.post("/api/clock/reset").json()["success"] is False
        assert len(client.get("/api/results").json()["data"]) == 1
        assert client.post("/api/clock/reset", params={"force": "true"}).json()["success"]

    def test_reset_with_no_results_is_fine(self, client):
        client.post("/api/clock/start")
        assert client.post("/api/clock/reset").json()["success"]
        assert client.get("/api/clock/status").json()["data"]["status"] == "stopped"


class TestFinishDelivery:
    def test_finish_with_clock_stopped_is_refused_not_mangled(self, client):
        r = client.post("/api/results", json=finish()).json()
        assert r["success"] is False
        assert "clock" in r["message"].lower()
        assert client.get("/api/results").json()["data"] == []

    def test_finish_with_clock_running_is_stored(self, client):
        client.post("/api/clock/start")
        r = client.post("/api/results", json=finish("120", "evt-1")).json()
        assert r["success"] and r["data"]["bibNumber"] == "120"
        assert r["data"]["eventId"] == "evt-1" and r["data"]["source"] == "race_cv"

    def test_same_event_id_twice_is_one_finisher(self, client):
        client.post("/api/clock/start")
        first = client.post("/api/results", json=finish("120", "evt-1")).json()["data"]
        again = client.post("/api/results", json=finish("120", "evt-1")).json()
        assert again["success"] and again.get("duplicate") is True
        assert again["data"]["id"] == first["id"]
        assert len(client.get("/api/results").json()["data"]) == 1

    def test_manual_adds_never_dedupe(self, client):
        """Two people can wear the same bib; a manual add is always new."""
        client.post("/api/clock/start")
        client.post("/api/results", json={"bibNumber": "7", "finishTime": "20:00.0"})
        client.post("/api/results", json={"bibNumber": "7", "finishTime": "21:00.0"})
        assert len(client.get("/api/results").json()["data"]) == 2

    def test_edit_and_delete(self, client):
        client.post("/api/clock/start")
        rid = client.post("/api/results", json=finish("120", "evt-1")).json()["data"]["id"]
        assert client.put(f"/api/results/{rid}", json={"bibNumber": "120", "racerName": "Jane"}).json()["success"]
        assert client.get("/api/results").json()["data"][0]["racerName"] == "Jane"
        assert client.delete(f"/api/results/{rid}").json()["success"]
        assert client.get("/api/results").json()["data"] == []


class TestJournalAndRestore:
    def test_every_change_is_journaled(self, client, isolated_state):
        client.post("/api/clock/start")
        rid = client.post("/api/results", json=finish("120", "evt-1")).json()["data"]["id"]
        client.put(f"/api/results/{rid}", json={"bibNumber": "120", "racerName": "Jane"})
        client.delete(f"/api/results/{rid}")
        log = (isolated_state / "race_log.txt").read_text()
        for action in ("CLOCK", "ADD", "EDIT", "DELETE"):
            assert action in log, log
        assert (isolated_state / "race_state.json").exists()
        assert (isolated_state / "race_results.txt").exists()

    def test_restart_restores_results_and_clock(self, client, isolated_state):
        client.post("/api/clock/start")
        client.post("/api/results", json=finish("120", "evt-1"))
        # simulate the process dying: wipe memory, keep the files
        server.race_results.clear()
        server.race_clock_state.update({"raceStartTime": None, "status": "stopped", "offset": 0})
        with TestClient(server.app):            # lifespan startup runs the restore
            pass
        assert [r["bibNumber"] for r in server.race_results] == ["120"]
        assert server.race_clock_state["status"] == "running"

    def test_fresh_archives_instead_of_restoring(self, client, isolated_state):
        client.post("/api/clock/start")
        client.post("/api/results", json=finish("120", "evt-1"))
        server.race_results.clear()
        server.app_state["fresh"] = True
        with TestClient(server.app):
            pass
        assert server.race_results == []
        assert list(isolated_state.glob("race_state_*.json"))

    def test_redelivery_after_restore_does_not_duplicate(self, client, isolated_state):
        client.post("/api/clock/start")
        client.post("/api/results", json=finish("120", "evt-1"))
        server.race_results.clear()
        with TestClient(server.app) as c2:
            again = c2.post("/api/results", json=finish("120", "evt-1")).json()
            assert again.get("duplicate") is True
            assert len(c2.get("/api/results").json()["data"]) == 1


DIST = Path(__file__).resolve().parents[1] / "src" / "frontend" / "dist" / "index.html"


@pytest.mark.skipif(not DIST.exists(), reason="frontend not built")
class TestSiteServing:
    def test_root_and_admin_serve_the_app(self, client):
        for path in ("/", "/admin"):
            r = client.get(path)
            assert r.status_code == 200 and '<div id="root"' in r.text, path

    def test_api_routes_win_over_the_catch_all(self, client):
        assert client.get("/api/does-not-exist").status_code == 404
        assert client.get("/api/clock/status").status_code == 200

    def test_no_path_traversal_through_the_catch_all(self, client):
        r = client.get("/../../pyproject.toml")
        assert "[project]" not in r.text
