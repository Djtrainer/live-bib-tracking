# Live Bib Tracking

Times a running race from a camera at the finish line. A YOLO11 detector
finds runners and their bibs, ByteTrack follows each runner to the line, an
OCR pass reads the bib number, and every finish is posted to a results API
that drives a live leaderboard and a Live Management page for corrections.

> **Running a race?** Everything operational — start/stop commands, the
> two-hotspot network setup for the pavilion TV, what the health line means,
> the config levers, and how to recover from a crash — is in
> **[RACE_DAY_RUNBOOK.md](RACE_DAY_RUNBOOK.md)**. For *why* the system is
> shaped the way it is, including what went wrong in 2025, see
> [RACE_DAY_ANALYSIS.md](RACE_DAY_ANALYSIS.md).

## How it fits together

```
 camera ──► race_cv (detect · track · OCR · finish-line logic)
                │  POST /api/results, one durable event per finish
                ▼
            results API (FastAPI) ── serves the site ──► leaderboard  /
                │  journal: every change on disk         Live Management /admin
                ▼
         data/results/  (events.jsonl, race_state.json, race_results.txt, race_log.txt)
```

Two processes, deliberately. `race_cv` owns the camera and the model and
nothing else; it keeps running if the API, the network, or the browser go
away, and logs every finish to disk *before* trying to deliver it. If it
crashes or wedges, the launcher restarts it and it re-queues whatever was
never confirmed. The API owns results, the clock, and persistence; a
restart restores the race.

## Layout

| path | what |
|---|---|
| `src/race_cv/` | the CV service: `capture` (camera/file), `detect` (YOLO + ByteTrack), `ocr` (EasyOCR, async), `finish` (crossing detection, track hand-off), `boundary`, `pipeline`, `sink` (durable delivery), `run` (CLI) |
| `src/api_backend/` | `local_server.py` (results API, WebSocket, serves the site), `journal.py` (crash-recoverable race record) |
| `src/frontend/` | the React leaderboard (`/`) and Live Management (`/admin`) |
| `config/race_cv.yaml` | every threshold, geometry point and model path — nothing is hardcoded |
| `scripts/` | offline tooling: `replay.py`, `smoke_test.py`, `score.py`, `calibrate.py`, `export_coreml.py`, `replay_events.py`, plus dataset/training helpers |
| `tests/` | 200+ tests; scripted detectors and fake OCR, no model needed |
| `start-race-cv.sh` / `stop-race-cv.sh` | the race-day launcher and its counterpart |

## Setup

The race runs on macOS with the model exported to CoreML. Python 3.11:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-race.txt     # the exact versions the race has been validated on
```

`pyproject.toml` carries the version ranges; `requirements-race.txt` is the
pinned set. `start-race-cv.sh` finds a suitable interpreter on its own
(`.venv`, then the `bib_env` conda environment) — see the runbook.

**Models are not in git.** `models/` is ignored. On a new machine, copy it
from the race Mac, or export from the trained weights:

```bash
python scripts/export_coreml.py --size 512 928
```

The frontend builds once and the API serves it:

```bash
(cd src/frontend && npm ci && npm run build)
```

## Run

Race day, in one line (details and the network setup in the runbook):

```bash
./start-race-cv.sh -c 0 -r roster.csv
```

A rehearsal against a recording, paced in real time with frames dropped
exactly as a camera would drop them:

```bash
./start-race-cv.sh -v "data/raw/<clip>.mov" -r roster.csv
```

Score every clip against `smoke_test.yaml`:

```bash
python scripts/smoke_test.py --expected smoke_test.yaml --roster roster.csv --realtime
```

Tests:

```bash
python -m pytest tests/ -q
```

## Development

For UI work, run the API and then Vite's dev server, which hot-reloads and
proxies `/api` and `/ws` to the API:

```bash
python src/api_backend/local_server.py --port 8001
```

```bash
cd src/frontend && npm run dev
```

Nothing is containerised. The previous Docker frontend served a static
build from a Linux VM holding 1–2 GB of an 8 GB machine, so it was retired;
the launcher builds `dist/` when it is stale and the API serves it.

## What the numbers are

Measured on the race machine (M2, 8 GB) against 13 recorded clips with 23
finishers: **22/23 found, 0 genuine ghosts, 22/22 bibs right**. The one miss
is an expectation written past the end of its clip. The tracking loop runs in
~20 ms against a 33 ms budget at 30 fps, and the race-day path processes
~99.5% of frames. `RACE_DAY_ANALYSIS.md` has the measurements behind each
decision.
