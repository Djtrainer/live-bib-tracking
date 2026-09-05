# race_cv

Standalone computer-vision service for finish-line detection and bib reading.

It owns the camera, detects finishes, and posts them to the results API. It has
no HTTP dependency and no browser coupling — the previous pipeline ran inside an
MJPEG response generator, so closing a tab stopped the race. See
[RACE_DAY_ANALYSIS.md](../../RACE_DAY_ANALYSIS.md) for what went wrong and why
each piece here is shaped the way it is.

## Requirements

Use an environment with `ultralytics >= 8.3`. The base conda environment on this
machine has 8.1.43, which **cannot load YOLO11 weights**:

```bash
/Users/dantrainer/miniconda3/envs/bib_env/bin/python -m pytest tests/ -q
```

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | All parameters, loaded from YAML. Unknown keys raise. |
| `capture.py` | Frame sources. Stamps every frame at grab time. |
| `detect.py` | YOLO tracking with explicit parameters; ROI translation. |
| `ocr.py` | Bib reading, confidence-weighted voting, roster snapping. |
| `finish.py` | Finish-line geometry; crossings as interpolated transitions. |
| `pipeline.py` | The frame loop. Uniform pacing, no burst skipping. |
| `sink.py` | Durable delivery: persist, then retry until confirmed. |
| `overlay.py` | Annotation for preview and debug video. |
| `run.py` | CLI entrypoint. |

## Before the race: calibrate

Geometry is normalized to the frame, but it is still specific to where the
camera points. Recalibrate against this year's camera and confirm the overlay:

```bash
python scripts/calibrate.py --source 1 --config config/race_cv.yaml
```

Click the two ends of the finish line, press `f` if the shaded "finished" side
is wrong, then `s` to save.

### Keeping non-runners off the course

`course_boundary` (disabled by default) restricts tracking to people between
two lines -- e.g. the edges of a driveway -- so someone walking past on the
sidewalk is never counted as a racer at all. A person is kept if at least one
corner of their box falls between the lines, interpolated at that corner's
own y, so the region narrows or widens with distance the way a real course
edge does in perspective.

This reproduces the 2025 setup's `guide_line_left` / `guide_line_right` gate
(`config/race_cv.yaml` ships the exact 2025 numbers). The difference:
`PipelineStats.people_outside_boundary` counts everyone it excludes every run,
and the overlay (`--preview`, or `--video-out` in replay) draws the lines
themselves plus anyone gated out in gray, labeled "outside course" -- so a
boundary that no longer matches this year's camera position is something you
notice, not a silent drop. `scripts/calibrate.py` doesn't set this yet; edit
`course_boundary` in the config directly, using the overlay to check it.

## Measure before you change anything

```bash
python scripts/replay.py --video data/raw/race.mp4 --out runs/baseline
python scripts/score.py --truth data/results/truth.csv --results runs/baseline/results.csv
```

Replay is deterministic: timestamps come from the frame index, not the wall
clock, so two runs over the same file produce identical output. A difference in
results is a difference in behaviour, never noise.

Ground truth is a CSV of `bib,elapsed_s`. **Recall is the metric that matters** —
the race-day failure was losing racers, not mislabelling them.

Add `--video-out` to write an annotated `debug.mp4` when you need to see what the
pipeline saw.

## Race day

```bash
python -m race_cv.run --source 1 --config config/race_cv.yaml \
    --roster data/raw/start_list.csv --preview
```

- `--source` takes a camera index or a video path.
- `--roster` breaks OCR ties toward bib numbers that exist in the race. Cheapest
  accuracy win available; supply it.
- `--no-api` runs offline, writing only the event log.

Every finish is appended to `sink.event_log` (default
`data/results/events.jsonl`) **before** any delivery attempt, so results survive
a crash or a total API outage. On shutdown the service prints any event it could
not deliver, with bib and timestamp, so nothing is lost quietly.

A refused finish — including "race clock is not running" — is retried with
backoff, never dropped. Each event carries a stable `eventId` so retries cannot
duplicate.

### Health output

Every 10 seconds:

```
health | processed 1834 (9.8 fps) | paced out 3502 | camera dropped 12 |
        finishers 27 (unknown bib 2) | delivered 27 | pending 0
```

`pending` climbing, or a `last error`, means results are not reaching the API —
the failure is visible within seconds instead of after the race.

## Browser preview

`localhost:8001/` (and `/video_feed`) work again under `--no-processor`:
`race_cv` publishes annotated frames to `POST /api/frame`, and the server
relays the latest one to any browser watching. This is presentation only --
see `stream.py` and `config.StreamConfig`. It uses the same drop-oldest,
never-block discipline as camera capture: a slow network POST only ever costs
a stale preview frame, never latency on detection or finish-line timing.
Disable with `--no-stream` or `stream.enabled: false` if bandwidth is tight.

## Known gaps

- The current server computes `finishTime = wallClockTime - raceStartTime`. A
  finish genuinely detected before the clock started will now land with a
  negative elapsed time rather than being discarded. That is recoverable
  information; discarding it was not.
- The server still needs to honour `eventId` for server-side deduplication.
