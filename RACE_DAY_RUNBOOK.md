# Race day runbook

Operational reference. For *why* the system is shaped this way, see
[RACE_DAY_ANALYSIS.md](RACE_DAY_ANALYSIS.md).

## Before anything else

```bash
# Start the race clock BEFORE racers arrive.
# A finisher detected while the clock is stopped is retried, not lost -- but it
# lands with a negative elapsed time, which someone then has to reconcile.
curl -X POST http://localhost:8001/api/clock/start
```

Always pass `--roster`. It is not a nicety any more: it is what takes bib
accuracy from 11/13 to 13/13, by refusing to lock a read like "20" that is not
a number anyone in the race is wearing.

## Live, against the camera

```bash
./start-race-cv.sh -c 0 -r data/raw/roster_example.csv --preview
```

- `-c 0` built-in camera, `-c 1` external/iPhone
- `-r` start-list CSV — needs a `Bib` column, see `data/raw/roster_example.csv`
- `--preview` opens an annotated window and runs in the foreground

Starts three things: the frontend in Docker (port 5173), a results API with no
video pipeline inside it (port 8001), and `race_cv` owning the camera. Closing
a browser tab cannot affect race timing.

Watch at `http://localhost:5173` (leaderboard) or `http://localhost:8001`
(raw annotated stream).

## Stopping

```bash
./stop-race-cv.sh
```

Stops `race_cv` first and gives it a graceful drain window, so a finish event
still retrying gets a chance to land before the API it is retrying against goes
away. `--keep-frontend` leaves Docker up; `-f` skips the graceful wait.

## A "live" test against a recording

This is the honest rehearsal: real-time pacing, and frames dropped when the
pipeline falls behind, exactly as a camera behaves.

```bash
./start-race-cv.sh -v "data/raw/Camo Recording 2025-10-04 10-25-15.mov" \
  -r data/raw/roster_example.csv --preview
```

Real-time is the default for `-v`. Add `--fast` to process as fast as possible
instead — useful for a quick answer, useless for judging whether the machine
can keep up.

### What to watch in the health line

```
health | processed 739 (14.7 fps) | paced out 576 | source dropped 202 | finishers 1 | delivered 1 | pending 0
```

| field | meaning | worry when |
|---|---|---|
| `processed (N fps)` | achieved rate | well below `target_fps` |
| `paced out` | deliberately skipped by pacing | never — this is by design |
| `source dropped` | **frames lost because the pipeline was busy** | climbing fast during a crossing |
| `pending` | finish events not yet accepted by the API | anything above 0 for long |
| `delivered` | events the API confirmed | should track `finishers` |
| `ocr read N` | bibs read off the frame loop | far below the number of finishers |
| `ocr … skipped N` | crops dropped because the reader was backed up | climbing steadily |
| `ocr … late N` | finishes resolved before their reads landed | anything above 0 |
| `TWO-STAGE FAILING` | second-stage inference is erroring | **ever** — the config is wrong |

`source dropped` is the number that matters. It is coverage you lost at the
line, and it is the thing a fast replay cannot show you.

## Scoring a run

```bash
python scripts/smoke_test.py --expected smoke_test.yaml \
  --roster data/raw/roster_example.csv
```

Runs all 13 clips and scores against `smoke_test.yaml`. Recall is the race-day
metric — a missed racer is unrecoverable, a wrong bib is a correction.

For one clip with an annotated video to scrub through afterwards:

```bash
python scripts/replay.py --video "data/raw/<clip>.mov" --out runs/latest --video-out
```

Note `replay.py` writes `debug.mp4` at the *source* frame rate, so it re-times
away every stall and can never look choppy no matter how badly the pipeline
struggled. Judge performance from the health line, never from the debug video.

## Config levers

All in `config/race_cv.yaml`.

| lever | current | effect |
|---|---|---|
| `model.path` / `model.imgsz` | 928×512 rect export, `[512, 928]` | **Must match**, and `imgsz` is `[height, width]`. A rectangular export sized to the crop below; the square 1280 spent 44% of each forward pass on padding. Regenerate with `python scripts/export_coreml.py`. |
| `roi.polygon` | x ≥ 0.28, y ≥ 0.30 | The racer-only region. Checked against `course_boundary` at startup — a crop that cuts into declared course prints `CONFIG MISMATCH`. Cropping only saves compute when `imgsz` is sized to match; on its own it changes nothing. |
| `pipeline.target_fps` | 30.0 | Loop is ~20ms against a 33ms budget. 36 frames dropped of 9338 at 30fps on the longest clip. If `source dropped` climbs, go to 15 first. |
| `model.two_stage` | false | Finds bibs by re-running the detector on each person's crop. **Only worth enabling with `two_stage_model` pointing at a smaller export.** A CoreML `.mlpackage` accepts exactly one input size, so without that every crop costs a full forward pass of the deployed model, *per person* — measured at 54.9ms/crop when the 1280 square export was deployed, roughly doubling frame cost. The 928×512 export makes each crop cheaper but the arithmetic is the same: one extra inference per runner. Against a dedicated small export it was 12.2ms per crop. Not needed at 30fps with every frame processed. |
| `ocr.async_reads` | true | Reads bibs on a background thread. Inline, a ~27ms read fired exactly at the line and pushed crossing frames over budget. Turn off only to reproduce old behaviour. |
| `finish_line.p1/p2` | 2025 geometry | **Recalibrate per camera.** `python scripts/calibrate.py --source 0 --config config/race_cv.yaml` |
| `course_boundary.enabled` | true | Keeps people outside the driveway off the leaderboard. Excluded people are drawn in grey on the overlay and counted in `people_outside_boundary`. |
| `finish_line.min_observations` | 0 | Requires a track to be seen N times before it may finish. Cuts duplicate-track ghosts. 5–10 was free on the smoke set; raise if you see two finishes a fraction of a second apart. |
| `ocr.crop_padding` | 15 | Tested 15/30/50; all read correctly with a roster loaded. Not a lever worth pulling. |
| `stream.enabled` | true | Browser preview. Disable if bandwidth or CPU is tight; it cannot slow detection either way. |

## If something goes wrong

**Finishers not appearing on the leaderboard** — check `pending` in the health
line and the race clock. Events are never dropped: they are appended to
`data/results/events.jsonl` *before* any delivery attempt, and retried until
the API confirms. On shutdown anything undelivered is printed with bib and
timestamp.

**Pipeline falling behind** (`source dropped` climbing) — lower `target_fps`
to 15 first; it was 26 dropped there against 36 at 30. Do **not** reach for
`two_stage` without a matching small export: with the deployed model it runs
every crop through the full network and roughly doubles frame cost. And check
the startup log for `CONFIG MISMATCH` — an `imgsz` that doesn't match the
export, or a crop the model wasn't sized for, costs far more than any fps knob
recovers.

**Racers missed entirely** — check `people_outside_boundary`. If it is large,
the course boundary no longer matches the camera; recalibrate or set
`course_boundary.enabled: false`.

**Wrong bib numbers** — confirm `--roster` is actually loaded (it logs the
count at startup). Without it, a confident misread can lock.
