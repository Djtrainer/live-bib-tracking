# Race day runbook

Operational reference. For *why* the system is shaped this way, see
[RACE_DAY_ANALYSIS.md](RACE_DAY_ANALYSIS.md).

## Which Python runs the race

`start-race-cv.sh` resolves an interpreter via `find_python()`: `.venv`, then
the `bib_env` conda environment, then `python3` — taking the first one that
has `ultralytics >= 8.3`. On this machine that is **`bib_env`** (Python 3.11,
ultralytics 8.3.x, torch 2.7, FastAPI/Starlette compatible). The base
miniconda `python3` is skipped on purpose: its ultralytics 8.1.x cannot load
YOLO11 weights, and its Starlette is incompatible with its FastAPI so the
results API will not even boot there.

**Measure with the interpreter that runs the race.** Anything timed or
scored under a different Python — a different ultralytics, torch, or thread
defaults — describes a machine you are not racing on. To use the race
interpreter explicitly:

```bash
RACE_CV_PYTHON=~/miniconda3/envs/bib_env/bin/python ./start-race-cv.sh -c 0 -r roster.csv
```

```bash
~/miniconda3/envs/bib_env/bin/python scripts/smoke_test.py --expected smoke_test.yaml --roster data/raw/roster_example.csv
```

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
./start-race-cv.sh -c 0 -r data/raw/roster_example.csv --native-frontend
```

- `-c 0` built-in camera, `-c 1` external/iPhone
- `-r` start-list CSV — needs a `Bib` column, see `data/raw/roster_example.csv`
- `--native-frontend` skips Docker; the API serves the leaderboard and Live
  Management itself on port 8001 (see the next section for who opens what)
- `--preview` opens an annotated window and runs in the foreground. It is
  redrawn at 10fps at half resolution (`--preview-fps`, `--preview-scale` on
  `race_cv.run`). This is not cosmetic: a full-rate, full-res preview cost
  19ms per frame *on the frame loop* and took the service from 29.9fps to
  19.4fps — a third of its throughput, and the difference between reading
  3/3 bibs and 2/3. Measured after gating: 28.3fps. Watch the browser stream
  instead if you want smoother video; it cannot slow detection.
  **Pressing `q` in that window stops the race service.** It logs a warning
  when it happens, but a stray keypress with the window focused ends timing —
  one more reason to run without it on race day.

Starts two things: a results API with no video pipeline inside it, which
also serves the site (port 8001), and `race_cv` owning the camera. Closing a
browser tab cannot affect race timing.

## Who opens what: the line, the pavilion, and the network between

The leaderboard and Live Management are two pages of **one site served by
the API**: `/` and `/admin` on port 8001. The pavilion is ~300 yards from the
finish line — too far for one Wi-Fi hotspot to bridge — so the setup is two
hotspots and a tunnel, which is the arrangement that worked last year minus
the screen share:

```
 FINISH LINE  (your phone's hotspot)            PAVILION  (its own hotspot)
 ┌───────────────────────────────────┐          ┌──────────────────────────┐
 │ race Mac: race_cv + API :8001     │  ngrok   │ any computer, wired to   │
 │ tablet:  http://<mac-ip>:8001/admin│ ───────► │ the TV, full screen:     │
 │          (local Wi-Fi, no password)│ cellular │ https://<dev-domain>/    │
 └───────────────────────────────────┘          └──────────────────────────┘
```

**At the line** everything is local. The tablet joins the Mac's hotspot and
opens `http://<mac-ip>:8001/admin` — ten metres of Wi-Fi, no cellular, no
password. Manual adds, bib corrections and the clock all live there. The
Mac's address on the hotspot is *not* its home-Wi-Fi address; read it off the
launcher's startup output or run `ipconfig getifaddr en0`. The `.local` name
usually does not resolve on a hotspot — use the IP.

**At the pavilion** the computer opens the tunnel URL over its own hotspot.
The tunnel is [ngrok](https://ngrok.com): the Mac makes an *outbound*
connection to ngrok's relay, ngrok forwards the public URL down it to port
8001. Nothing inbound ever reaches the Mac, and neither hotspot needs to be
reachable from the other. The free plan's one assigned dev domain is
permanent, so the pavilion kiosk can be set up in advance.

```bash
ngrok http 8001 --url https://bonanza-overbite-sprawl.ngrok-free.dev --traffic-policy-file config/ngrok-policy.yml
```

- First visit per browser session shows ngrok's interstitial — click through
  before the first finisher.
- The **leaderboard, its reads, and the WebSocket need no password**; the TV
  never sees a prompt.
- **`/admin` and every non-GET API call require a password** (the policy
  file). Gating only the page would leave the endpoints it calls open;
  gating the writes is what protects the results. The admin browser is
  challenged once — starting the clock is the first write, so expect it
  there — and re-sends credentials to every later call.
- **Set a real password.** `config/ngrok-policy.yml` is committed with
  `race:change-me`. Copy it to `config/ngrok-policy.local.yml` (gitignored),
  put the real password there, and pass that file instead. The policy is
  read when ngrok starts; restart it after changing the file.
- ngrok reconnects on its own after a cellular blip, and so does the page
  (it re-fetches everything it missed). The Mac never depended on the tunnel
  for timing in the first place.

**Do not screen-share from the race Mac.** A Google Meet share is a browser
tab encoding the screen at 30 fps — a large slice of a core and hundreds of
MB on an 8 GB machine that already pages under the pipeline — for a
blurrier, laggier picture than the pavilion opening the page itself. The
tunnel carries kilobytes where Meet carried video.

**Test in the driveway before race day:** Mac on your phone's hotspot,
another laptop on a second phone's hotspot, both URLs. If it works there it
works in the park; the only variable is signal, and last year proved that.

## Race-day command sheet

Everything in order, from the race Mac's terminal unless stated.

**The night before**

```bash
(cd src/frontend && npm run build)     # only after frontend changes; the launcher rebuilds if stale
```
```bash
cp config/ngrok-policy.yml config/ngrok-policy.local.yml   # then edit the password in the copy
```
- Roster CSV ready (`Bib` column). Mac charged, charger packed. Camera
  position rehearsed; `python scripts/calibrate.py --source 0 --config config/race_cv.yaml`
  if it moved.

**At the booth**

1. Mac on your phone's hotspot; tablet on the same hotspot. Plug the Mac in.
2. Start the stack:
   ```bash
   ./start-race-cv.sh -c 0 -r roster.csv --native-frontend
   ```
   It prints the local addresses. Note the IP.
3. Start the tunnel in a second terminal and leave it open:
   ```bash
   ngrok http 8001 --url https://bonanza-overbite-sprawl.ngrok-free.dev --traffic-policy-file config/ngrok-policy.local.yml
   ```
4. Tablet: `http://<mac-ip>:8001/admin`. Pavilion: the tunnel URL, click
   through the interstitial, full screen.
5. Start the clock at the gun — the button on Live Management, or:
   ```bash
   curl -X POST http://localhost:8001/api/clock/start
   ```

**During the race**

```bash
tail -f race_cv.log | grep --line-buffered "health |"
```
Watch `source dropped`, `pending`, and `handed off` — see *What to watch in
the health line* below.

```bash
curl -s http://127.0.0.1:4040/api/tunnels | python3 -c "import json,sys; [print(t['public_url'], '->', t['config']['addr']) for t in json.load(sys.stdin)['tunnels']]"
```
Confirms the tunnel is up and where it points.

```bash
ipconfig getifaddr en0
```
The Mac's current address, if the tablet needs it again.

**Stopping**

```bash
./stop-race-cv.sh
```
Then Ctrl-C the ngrok terminal.

**After the race**

```bash
python scripts/replay_events.py --dry-run
```
Lists any finish event the API never confirmed; drop `--dry-run` to re-send.

```bash
python scripts/replay.py --video "<recording>.mov" --out runs/race --roster roster.csv --video-out
```
Re-runs the recording with every frame, for reconciling anything the live
run missed. Finishers and times from a replay are trustworthy; see the
offline-OCR caveat under *Scoring a run* before trusting its bibs over the
live ones.

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
| `finishers N (…, handed off M)` | M finishes recovered across a track break | never a worry on its own — each one was a miss before. A large M means the tracker is fragmenting at the line; consider whether the camera is too close. |
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

**Offline runs under-report bib accuracy.** Without `--realtime` the frame loop
runs flat out with no pacing sleep, and the OCR worker thread gets almost no
idle CPU; live at 30fps it does. Measured on the same clip and code: realtime
read 15 bibs with 0 skipped, the offline replay read 7 with 7 dropped and one
finish resolved before its reads landed. Finishers and times are trustworthy
offline; a "No bib" that only appears offline usually is not. When bib
accuracy is the question, score with `--realtime`.

## Config levers

All in `config/race_cv.yaml`.

| lever | current | effect |
|---|---|---|
| `model.path` / `model.imgsz` | 928×512 rect export, `[512, 928]` | **Must match**, and `imgsz` is `[height, width]`. A rectangular export sized to the crop below; the square 1280 spent 44% of each forward pass on padding. Regenerate with `python scripts/export_coreml.py`. |
| `roi.polygon` | x ≥ 0.28, y ≥ 0.30 | The racer-only region. Checked against `course_boundary` at startup — a crop that cuts into declared course prints `CONFIG MISMATCH`. Cropping only saves compute when `imgsz` is sized to match; on its own it changes nothing. |
| `pipeline.target_fps` | 30.0 | Loop is ~20ms against a 33ms budget. 36 frames dropped of 9338 at 30fps on the longest clip. If `source dropped` climbs, go to 15 first. |
| `model.two_stage` | false | Finds bibs by re-running the detector on each person's crop. **Only worth enabling with `two_stage_model` pointing at a smaller export.** A CoreML `.mlpackage` accepts exactly one input size, so without that every crop costs a full forward pass of the deployed model, *per person* — measured at 54.9ms/crop when the 1280 square export was deployed, roughly doubling frame cost. The 928×512 export makes each crop cheaper but the arithmetic is the same: one extra inference per runner. Against a dedicated small export it was 12.2ms per crop. Not needed at 30fps with every frame processed. |
| `ocr.async_reads` | true | Reads bibs on a background thread. Inline, a read fired exactly at the line and pushed crossing frames over budget. Turn off only to reproduce old behaviour. |
| `ocr.async_min_submit_interval_s` | 0.12 | Per-runner spacing between crops sent to OCR (~8/s). A real read is ~48ms, so unshaped demand (up to 29/s) backs the worker up and finishes resolve before their reads land. Lower only if `ocr read` is far below finishers *and* `skipped` is 0. |
| `ocr.width_buckets_px` | 128…448 | Crop widths the warm-up compiles. Any width outside the set costs a 150–1000ms kernel compile on first sight, mid-race. Leave alone. |
| `ocr.resolve_grace_s` | 1.0 | How long a finish may wait, without blocking the loop, for that racer's reads still in flight. Costs event latency only; the crossing time is fixed. Watch `late N` in the health line — it should stay 0. |
| `finish_line.p1/p2` | 2025 geometry | **Recalibrate per camera.** `python scripts/calibrate.py --source 0 --config config/race_cv.yaml` |
| `course_boundary.enabled` | true | Keeps people outside the driveway off the leaderboard. Excluded people are drawn in grey on the overlay and counted in `people_outside_boundary`. |
| `finish_line.min_observations` | 5 | Requires a track to be seen N times before it may finish. Cuts duplicate-track ghosts. Raise if you see two finishes a fraction of a second apart. A hand-off (below) inherits the dead track's count, so it does not fight this. |
| `finish_line.handoff_window_s` | 1.0 | Recovers a crossing when the tracker issues a new id right at the line — a racer reaching a close camera gets large and clipped, and ByteTrack breaks the track a few frames short. Both genuine misses on the 13-clip set were this; both are found now. A track born past the line within this window of an approaching track vanishing nearby is treated as its continuation. Widen only if `handed off` stays 0 while racers are still being missed at the line; 0 disables it. |
| `ocr.crop_padding` | 15 | Tested 15/30/50; all read correctly with a roster loaded. Not a lever worth pulling. |
| `stream.enabled` | true | Browser preview. Disable if bandwidth or CPU is tight; it cannot slow detection either way. |

## Race-day lean mode: where the compute actually goes

Measured on the race machine (M2, 8 GB, `bib_env`), realtime, full stack.
The pipeline is not the problem: `race_cv` runs at **~0.75 of one core and
~240 MB** with every worker thread asleep. What competes with it is
everything else on the Mac, and on 8 GB the constraint is memory before CPU
— `vm_stat` showed 1,000–2,000 pageouts a minute under the full stack, and a
swap stall at the line looks exactly like a slow model.

| cost | measured | race day |
|---|---|---|
| `race_cv` (detector + tracker + async OCR) | ~73% CPU, 239 MB | keep; it is already lean |
| `--preview` window | 19 ms/frame *on the frame loop* before gating; now 10 fps, half-res | **leave it off**; watch the browser stream, which cannot slow detection |
| frame streamer + API relay | no measurable CPU difference on vs off | keep on for the operator's browser; `stream.target_fps` 8 is fine |
| Camo Studio + its extension | 20–27% CPU | needed for the iPhone camera; close Camo's own preview window |
| WindowServer | 18–39% CPU, driven by on-screen windows | no preview window, no Camo preview, no browser on the race Mac |
| Docker Desktop (frontend container) | a Linux VM holding 1–2 GB to serve a static folder | **`--native-frontend`** — the API serves the same `dist/` itself on 8001, reachable from the pavilion |
| editors, chat apps, other projects | ~40% CPU and hundreds of MB, measured | close them; they were the load in every profile |

Concretely:

```bash
./start-race-cv.sh -c 0 -r roster.csv --native-frontend
```

- Plug in. On battery macOS throttles; Low Power Mode throttles harder.
- Watch the leaderboard and the stream from a **phone or second laptop**, not
  the race Mac — every browser tab there is CPU and RAM taken from detection.
- The preflight prints reclaimable memory and warns under 1.5 GB. Take the
  warning seriously; it is the one thing here that can lose a racer.
- Two things that are *not* worth changing: OpenMP/torch thread caps (no
  effect under `bib_env`'s torch 2.7; the spinning-pool problem exists only
  under base miniconda's torch 2.2) and the CoreML compute-unit setting
  (`ALL` is already ultralytics 8.3's default; the override is a guard
  against a future upgrade, not a race-day lever).

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

**Pavilion TV not updating** — results at the line are unaffected; this is
display only. In order: is the ngrok terminal still open and showing the
tunnel (`curl -s http://127.0.0.1:4040/api/tunnels`)? Does the Mac's hotspot
have signal? Reload the page on the pavilion machine — it reconnects on its
own after a blip, but a reload is the fastest confirmation. If ngrok itself
died, restart the command from the sheet; the page recovers by itself.

**Tablet can't reach `/admin`** — it must be on the *Mac's* hotspot, and the
address is the Mac's IP on that hotspot (`ipconfig getifaddr en0` on the
Mac), not its home-Wi-Fi address and not the `.local` name. Fallback: the
tunnel URL works from anywhere, with the password.
