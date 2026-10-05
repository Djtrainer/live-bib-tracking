# Race day runbook

**On the morning itself, run [RACE_MORNING.md](RACE_MORNING.md):** the commands in order, nothing else.

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

Start the race clock **before** racers arrive: the Start Race button in Live
Management, signed in. A finisher detected while the clock is stopped is
retried, not lost -- but it lands with a negative elapsed time, which someone
then has to reconcile. (Starting the clock is an operator action; a bare
`curl -X POST .../api/clock/start` is answered 401 once a login is set.)

Always pass `--roster`. It is not a nicety any more: it is what takes bib
accuracy from 11/13 to 13/13, by refusing to lock a read like "20" that is not
a number anyone in the race is wearing.

## Live, against the camera

```bash
./start-race-cv.sh -c 0 -r data/raw/roster_example.csv
```

- `-c 0` built-in camera, `-c 1` external/iPhone
- `-r` start-list CSV — needs a `Bib` column, see `data/raw/roster_example.csv`
- The API serves the leaderboard and Live Management itself on port 8001
  (see the next section for who opens what); nothing runs in Docker
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
 │          (local Wi-Fi, admin login)│ cellular │ https://<dev-domain>/    │
 └───────────────────────────────────┘          └──────────────────────────┘
```

**At the line** everything is local. The tablet joins the Mac's hotspot and
opens `http://<mac-ip>:8001/admin` — ten metres of Wi-Fi, no cellular. It asks
for the admin login (below). Manual adds, bib corrections and the clock all live there. The
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
./broadcast.sh leaderboard
```

which runs `ngrok http 8001 --url https://slaysarcoma.ngrok.app
--traffic-policy-file config/ngrok-policy.local.yml` in the background
(`.broadcast/ngrok.log`). The same script points the **same URL** at one
of two holding pages instead of the leaderboard, so spectators never meet
an empty or broken site:

| command | what the public URL shows |
|---|---|
| `./broadcast.sh pre-race --time "around 8:30 am"` | "The race hasn't started yet! Check back around 8:30 am" (port 8002) |
| `./broadcast.sh leaderboard` | the live leaderboard (port 8001); refuses if nothing answers there |
| `./broadcast.sh thank-you` | "Thank you for coming", plus the sponsors in `config/sponsors.txt` if that file exists (port 8003) |
| `./broadcast.sh status` | which of the three the tunnel points at, and which servers are up |
| `./broadcast.sh offline` | nothing: the URL goes dark, exactly as Ctrl-C in the ngrok window did; the leaderboard and page servers keep running locally |
| `./broadcast.sh stop` | offline, plus stop the two page servers |

A switch is a restart of the ngrok agent against another local port and
takes a few seconds; the holding pages reload themselves (every 60 s /
120 s), so tabs left open on the pavilion TV follow the switch on their
own. The pages are in `src/static_pages/` and are self-contained (no
external assets to fail through the tunnel); they answer on every path, so
a bookmarked `/admin` shows the holding page too.

- First visit per browser session shows ngrok's interstitial — click through
  before the first finisher.
- The **leaderboard, its reads, and the WebSocket need no password**; the TV
  never sees a prompt.
- **`/admin` and every non-GET API call require a password** (the policy
  file). Gating only the page would leave the endpoints it calls open;
  gating the writes is what protects the results. The admin browser is
  challenged once — on opening `/admin`, before the app's own login
  appears — and re-sends credentials to every later call.
- **Set a real password.** `config/ngrok-policy.yml` is committed with
  `race:change-me`. Copy it to `config/ngrok-policy.local.yml` (gitignored),
  put the real password there, and pass that file instead. The policy is
  read when ngrok starts; restart it after changing the file.
- **Live Management has its own login, checked by the API.** The tunnel
  password is the outer door and only exists on the public URL; on the
  hotspot there was no door at all, and the old login page compared the
  password in the browser (and `/admin/dashboard` skipped it). Now the
  credentials live in `config/admin.local.yml` on the Mac
  (`cp config/admin.example.yml config/admin.local.yml`, set the password;
  gitignored), the API checks them, and the clock, edits, deletes, roster
  upload and manual adds are refused without a signed-in session. The
  detector's finishes and preview frames need no login when they arrive
  from where a detector can be: this Mac itself or the USB link to the
  Jetson (192.168.55.x), and never through the tunnel. So nothing changes
  on the Jetson over the cable. If it ever posts over the hotspot instead,
  list its hotspot address under `detector_addresses` in the same file, or
  its finishes are refused: the Jetson's health line shows `pending`
  climbing with a 401, and `api_server.log` names the refused address. A session lasts 18 hours from its last use (it is renewed
  while the dashboard is open, so it does not end under you; 72 hours after
  the sign-in it ends regardless) and survives an
  API restart. Sessions are per address: the tablet on the hotspot, the
  Mac on localhost and the public URL each sign in once. Sign Out only
  drops the cookie in that browser; to end every session, change the
  password and restart the API. The username is not case-sensitive, the
  password is; keep both in quotes in the file. With no such file the API
  runs open and the launcher says so in red; with the placeholder still in
  the file the API refuses to start.
- ngrok reconnects on its own after a cellular blip, and so does the page
  (it re-fetches everything it missed). The Mac never depended on the tunnel
  for timing in the first place.

**Between races the URL comes from a free VM, not the laptop.** The same
holding page and the same ngrok agent run on Google's always-free
`e2-micro` (`deploy/broadcast-vm/`), so the public URL shows "Check back on
October 24th" from now until race day with every laptop closed. One
agent can hold the URL at a time, so race day is a hand-off, and the
race-day commands do it for you:

| when | command | what happens |
|---|---|---|
| once | `deploy/broadcast-vm/deploy.sh create` | creates the VM, pushes the page, sponsors and agent config, starts both services, checks the URL |
| sponsors or page changed | `deploy/broadcast-vm/deploy.sh sync` | pushes `config/sponsors.txt` and the pages again |
| race morning | `./broadcast.sh pre-race` (or `leaderboard`) | stops **and disables** the VM's agent first, then claims the URL here -- a VM reboot mid-race cannot take it back |
| packing up | `./broadcast.sh cloud-on` | stops the local agent, re-enables the VM's, confirms the URL answers |
| any time | `./broadcast.sh status` / `deploy/broadcast-vm/deploy.sh status` | who holds the URL; VM and service state |

What it costs: the VM, its 10 GB disk and the page's traffic are inside the
free tier; the external IPv4 address is not, at $0.005 an hour, about $3.72
a month. Visitors on the free plan click through ngrok's interstitial once
per browser session, as they do for the leaderboard. The
page lists sponsors when `config/sponsors.txt` exists (one name per
line, gitignored) and omits the section when it does not; add the file and
run `sync`. The VM needs the Mac's `gcloud` login for the hand-off, so the
race-morning command needs internet on the Mac (the hotspot) before it can
release the VM; if gcloud cannot reach it, it says so and stops rather than
fight the VM for the URL (`BROADCAST_SKIP_VM=1 ./broadcast.sh pre-race`
overrides that when you know the VM is not holding it). Once released, the
later switches that morning skip the VM check entirely.

**Do not screen-share from the race Mac.** A Google Meet share is a browser
tab encoding the screen at 30 fps — a large slice of a core and hundreds of
MB on an 8 GB machine that already pages under the pipeline — for a
blurrier, laggier picture than the pavilion opening the page itself. The
tunnel carries kilobytes where Meet carried video.

**Test in the driveway before race day:** Mac on your phone's hotspot,
another laptop on a second phone's hotspot, both URLs. If it works there it
works in the park; the only variable is signal, and last year proved that.

## Two machines: the Jetson times, the Mac is the console

The race-day shape once the Jetson exists. The Jetson owns the camera and
runs nothing but the detector on its GPU; the Mac at the finish line runs
the results API, the leaderboard and Live Management, the journal and the
ngrok tunnel, and is what you look at and type into. A dead Mac loses no
finish (the Jetson keeps timing and logging to its own disk); a dead
Jetson loses no result (the Mac restores the race from its journal and the
Jetson re-queues whatever was never confirmed when it comes back).

```
 USB-C cable (L4T device mode)          phone hotspot            cellular
 Jetson 192.168.55.1  ◄────────►  Mac 192.168.55.100 / <hotspot-ip>  ────►  pavilion
   camera, race_cv                 API :8001, site, journal, ngrok           (tunnel URL)
   posts finishes + preview        tablet: http://<hotspot-ip>:8001/admin
```

The cable is the only path the timing depends on. The Jetson has no Wi-Fi
and no internet in this arrangement and needs neither. The Mac's hotspot
carries the tablet and the tunnel exactly as in the one-machine setup.

**Once, on a new board:** JetPack 7.2 ships the USB gadget with RNDIS
first and macOS never brings it up -- see `JETSON_NOTES.md`, "Mac to
Jetson over USB-C" for the one-line fix. Install your key so nothing
prompts for a password: `ssh-copy-id dantrainer@192.168.55.1`.

**In order, from the Mac's terminal:**

1. The console. `-r` is not needed here; racer names come from the roster
   upload in Live Management.
   ```bash
   ./start-race-cv.sh --api-only --fresh
   ```
   It prints the tablet and pavilion addresses and the line the Jetson
   needs: `--api-url http://192.168.55.100:8001`.
2. The Jetson's code, if the Mac has newer commits (the Jetson has no
   other route to them; this pushes over the cable and fast-forwards):
   ```bash
   git push dantrainer@192.168.55.1:live-bib-tracking jetson-eval:refs/heads/from-mac && ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && git merge --ff-only from-mac && git branch -D from-mac'
   ```
   Then the Jetson's clock. The board has no RTC battery and no NTP without
   Wi-Fi; at the first split rehearsal it was 9.5 days behind. The Jetson
   launcher measures the skew against the console's `/api/time` to the
   millisecond. A few seconds only earns a warning, because the detector
   corrects every finish time by the measured offset; over a minute it
   refuses to start and prints this command. It needs the Jetson's sudo
   password, and it must run while **no detector is running** (a clock
   jump under a running detector corrupts its timestamps). The script
   authenticates sudo first and reads the console's time afterwards; an
   earlier one-liner computed the time before the password prompt and left
   the board behind by however long the typing took:
   ```bash
   ssh -t dantrainer@192.168.55.1 live-bib-tracking/scripts/set_clock_from_console.sh
   ```
   Whatever is left, the detector measures against `/api/time` at startup
   and corrects every finish timestamp by it (the log line `Clock offset
   vs the API`). To never type the password: once, on the board, allow
   `date` without one, and the launcher sets the clock by itself whenever
   it is more than 2 s off:
   ```bash
   ssh -t dantrainer@192.168.55.1 "echo 'dantrainer ALL=(root) NOPASSWD: /usr/bin/date' | sudo tee /etc/sudoers.d/race-clock && sudo chmod 440 /etc/sudoers.d/race-clock"
   ``` If the clock ever has to be corrected mid-race:
   stop the detector, set the date, start it again **without** `--fresh`
   (so nothing is archived) -- its unconfirmed events are re-queued.
3. The detector, on the Jetson. On the board the launcher picks
   `config/race_cv.jetson.yaml` (the yolo11m-1280 TensorRT engine) and
   the camera source by itself: the raw-YUY2 GStreamer string for
   `/dev/video0` at 1080p30, measured live at 30.0 fps. Do **not** pass
   `-c 0`: a bare index negotiates 1280x720, and `-c 1` is the camera's
   metadata node and fails to open. No `--preview` (headless -- the
   browser stream is the preview). `--api-url` starts nothing but the
   supervised detector, after checking that the API answers and the
   clock agrees.
   ```bash
   ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./start-race-cv.sh --api-url http://192.168.55.100:8001 -r data/results/roster_2026.csv --fresh'
   ```
   The real start list goes to the Jetson beforehand (the detector's
   `-r` needs a `Bib` column; the launcher refuses to start if the file is
   not there and lists the CSVs that are) and into Live Management on the
   Mac (names):
   ```bash
   scp roster_2026.csv dantrainer@192.168.55.1:live-bib-tracking/data/results/
   ```
   The launcher exits once the detector is up; the detector keeps running
   under its supervisor after the ssh session ends.
4. The holding page, the tablet, the pavilion and the clock: steps 3-6 of
   the command sheet below, unchanged.

**Watching it.** The health line is in the Jetson's `race_cv.log`, not on
the Mac:
```bash
ssh dantrainer@192.168.55.1 tail -f live-bib-tracking/race_cv.log
```
The camera, with boxes, is `http://localhost:8001/video_feed` on the Mac
(the Jetson posts annotated frames to the API at 8 fps; those requests
are kept out of `api_server.log`, which would otherwise grow by 29,000
lines an hour). `finishers`, `delivered` and `pending` mean the same as
ever; `pending` climbing with `last error` naming the API means the cable
or the Mac, not the camera. A `SUPERVISOR ... RESTART` line means the
detector died and came back; look at the traceback above it.

**Before the camera is bolted down** (once, at setup): the finish line in
`config/race_cv.jetson.yaml` is a camera position, and the Jetson has no
screen for `calibrate.py`. Grab a still on the Jetson with no detector
running, calibrate on the Mac, copy the config back:
```bash
ssh dantrainer@192.168.55.1 "cd ~/live-bib-tracking && PYTHONPATH=src .venv/bin/python scripts/focus.py --source 'v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! appsink drop=true max-buffers=1' --save still.jpg"
```
(`focus.py` also prints a sharpness score -- twist the lens until distant
edges peak; the B0497 shipped out of focus. Ctrl-C writes the still.)
```bash
scp dantrainer@192.168.55.1:live-bib-tracking/still.jpg /tmp/still.jpg && python scripts/calibrate.py --source /tmp/still.jpg --config config/race_cv.jetson.yaml && scp config/race_cv.jetson.yaml dantrainer@192.168.55.1:live-bib-tracking/config/
```
Geometry is stored normalised, so a still calibrated on the Mac is the
Jetson's geometry. Focus can be re-checked with the stack running:
`python scripts/focus.py --source http://localhost:8001/video_feed`.

**Stopping**, in this order so the Jetson can drain against a live API:
```bash
ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./stop-race-cv.sh'
```
```bash
./stop-race-cv.sh
```

**What can go wrong that the one-machine setup could not:**

- *The Mac launcher run without `--api-only` while the Jetson is timing*
  starts a second detector on the Mac's camera; every finisher would be
  reported twice. The single-instance guard only sees its own machine.
  On race day the Mac's command is `--api-only`, nothing else; if in
  doubt, `pgrep -fa 'python -m race_cv.run'` on the Mac must print
  nothing.
- *The Mac falls asleep.* On battery macOS sleeps after one idle minute,
  which drops the USB link and the API. The launcher now holds the Mac
  awake with `caffeinate` for as long as the API runs, but nothing stops
  lid-close sleep: lid open, and on power.
- *The cable drops.* The Jetson keeps timing; finishes retry with backoff
  (up to 30 s) and sit in its `data/results/events.jsonl` until the link
  is back, then land. Preview frames are simply dropped meanwhile.
- *The Mac's API restarts.* It restores the race from the journal; the
  Jetson's retries land on the new process (`eventId` makes them
  idempotent).
- *The Jetson reboots.* Its clock may be wrong again (step 2); the
  supervisor is not a boot service yet, so start the detector again by
  hand (step 3). Unconfirmed events from before the reboot are re-queued
  on start.
- *Recovery after the race* needs both machines' files: the Mac's
  `data/results/` (journal) and the Jetson's `data/results/events.jsonl`.

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
2. Start the stack. **One machine** (the Mac on the camera):
   ```bash
   ./start-race-cv.sh -c 0 -r roster.csv
   ```
   **Two machines** (the Jetson on the camera), in this order, all from
   the Mac's terminal -- details in "Two machines" above:
   ```bash
   ./start-race-cv.sh --api-only --fresh
   ```
   ```bash
   ssh -t dantrainer@192.168.55.1 live-bib-tracking/scripts/set_clock_from_console.sh
   ```
   ```bash
   ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./start-race-cv.sh --api-url http://192.168.55.100:8001 -r data/results/roster_2026.csv --fresh'
   ```
   ```bash
   ssh dantrainer@192.168.55.1 tail -f live-bib-tracking/race_cv.log
   ```
   (console; the Jetson's clock, sudo password, no detector running yet;
   the detector, which the launcher checks against the API and the clock
   before starting; the health line, in a terminal you keep open.)
   Either way it prints the local addresses. Note the IP.
3. Put the holding page on the public URL (first thing, before the
   pavilion opens its browser):
   ```bash
   ./broadcast.sh pre-race --time "8:30 am"
   ```
4. Tablet: `http://<mac-ip>:8001/admin`. Pavilion: the tunnel URL, click
   through the interstitial, full screen — it shows the holding page and
   switches by itself later.
5. When results should be public:
   ```bash
   ./broadcast.sh leaderboard
   ```
6. Start the clock at the gun: the Start Race button in Live Management (signed in).

**During the race**

```bash
tail -f race_cv.log | grep --line-buffered "health |"
```
Watch `source dropped`, `pending`, and `handed off` — see *What to watch in
the health line* below.

```bash
./broadcast.sh status
```
Confirms the tunnel is up and which of the three sites it points at.

```bash
./broadcast.sh thank-you
```
**If the leaderboard must not be seen** — a crash, a bad restore, results
that need fixing — this puts the thank-you page on the public URL within
seconds; timing at the line is unaffected. `./broadcast.sh leaderboard`
brings the results back.

```bash
ipconfig getifaddr en0
```
The Mac's current address, if the tablet needs it again.

**Stopping**

```bash
./stop-race-cv.sh
```
Then `./broadcast.sh thank-you` if the pavilion should keep showing
something, or `./broadcast.sh offline` to take the URL down.

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

With the Jetson on the camera, stop it **first** so it can drain against a
live API, then the Mac:

```bash
ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./stop-race-cv.sh'
```
```bash
./stop-race-cv.sh
```

Stops `race_cv` first and gives it a graceful drain window, so a finish event
still retrying gets a chance to land before the API it is retrying against goes
away. `race_cv` drains for up to 10 s, then prints its final health line and
anything still undelivered (bib and timestamp) before exiting; the stop
script waits 20 s before escalating, so that report always lands in
`race_cv.log`. `-f` skips the graceful wait.

`race_cv` runs under a small supervisor: a deliberate stop (this script, `q`
in the preview, the end of a file) is final, but any other exit — a crash, an
exhausted error budget, a wedged frame loop — is restarted after 5 s,
indefinitely, and the restarted process re-queues every finish the API never
confirmed. The consequence: `kill <pid>` on the detector alone just gets it
restarted. Use the stop script, or `pkill -TERM -f race_cv.run`, which takes
the supervisor with it.

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
| `frame errors N` | frames skipped because processing raised | a few is a bad frame; 30 in a row stops the process and the supervisor restarts it |
| `CAMERA SILENT Ns` | the device has produced nothing for N seconds | 2 s+ during the race — check Camo/cable; the capture thread reopens it every 5 s |
| `reopens N` | times the camera was reopened after a stall | a climbing count is a bad cable or hub |
| `recovered N` | finishes re-queued from the event log at start | only after a restart; with `--fresh` it should be absent |
| `EVENT LOG WRITE FAILURES N` | `events.jsonl` cannot be written | **ever** — delivery continues but crash recovery is gone; free disk space |

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
| `ocr.min_votes_off_roster` | 2 | With a roster loaded, a number not on it needs two agreeing reads to win; one stray read is what a bibless racer looks like. **Load the full roster on race day** — on the 14-clip set, bib 76 resolved only once it was on the list. |
| `ocr.snap_fragments_to_roster` | true | An off-roster winner of 2+ digits that is one digit short of exactly one roster bib is a clipped bib and resolves to it (bib 120 read as "20" twice with an arm over the first digit). Fragments several roster bibs could own are left as read; single digits never snap. |

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
| Docker Desktop | a Linux VM holding 1–2 GB even when idle | quit it; the site is served by the API itself on 8001, nothing needs Docker any more |
| editors, chat apps, other projects | ~40% CPU and hundreds of MB, measured | close them; they were the load in every profile |

Concretely:

```bash
./start-race-cv.sh -c 0 -r roster.csv
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

## What is recorded, and how to recover

Everything lands in `data/results/`, written as it happens, so a crash at any
moment loses at most the change in flight.

| file | written by | what it is |
|---|---|---|
| `events.jsonl` | `race_cv` | every detected finish, appended *before* delivery is attempted, and again when the API confirms it. The camera's own record; survives the API being down. |
| `race_state.json` | the API | the complete race — every result and the clock — rewritten atomically on every change. **This is what a restart restores from.** |
| `race_results.txt` | the API | the current leaderboard as a person reads it: place, bib, time, name, source. Rewritten on every change. With the API down and a runner at the booth, the answer is in this file. |
| `race_log.txt` | the API | one line per change, appended, never rewritten: `14:02:11  ADD  bib 120 Jane 21:34.5 [race_cv]`, `EDIT`, `DELETE`, `CLOCK start`. The sequence, for disputes. |

**If the API dies mid-race:** restart it exactly as before —

```bash
./start-race-cv.sh -c 0 -r roster.csv
```

— and it restores the previous results and clock on its own, logging
`RESTORED previous race state … N results … clock running`. Manual adds, bib
corrections and the clock start all come back. A detection `race_cv` was
still retrying when the API died is re-sent and **does not duplicate**: every
finish carries an `eventId`, and the API returns the existing record for one
it has already seen. Manual adds carry no `eventId` and are always new, so two
people with the same bib remain possible on purpose.

**Starting a new race** on a machine that has a saved one:

```bash
./start-race-cv.sh -c 0 -r roster.csv --fresh
```

`--fresh` archives the old `race_state.json`, `race_results.txt` and
`race_log.txt` with a timestamp instead of deleting them, and moves
`events.jsonl` aside the same way. That last part matters: `race_cv`
re-queues every unconfirmed event it finds in that log on start, so without
the rotation the leftovers of the last rehearsal would be delivered into the
new race as finishers (16 of them were, in testing). If `--fresh` is
forgotten, unconfirmed events older than 12 hours are ignored with a warning
rather than re-delivered. Without `--fresh`, a restart *restores*, which is
the right default on race day and the wrong one at the next event — the
startup log says which happened, loudly.

**If the race Mac itself dies:** the three files above plus `events.jsonl`
are on its disk. Copy `data/results/` to another machine, start the API
there, and it restores the same way. Anything the camera saw but never
delivered is in `events.jsonl`:

```bash
python scripts/replay_events.py --dry-run
```

## If something goes wrong

**Finishers not appearing on the leaderboard** — check `pending` in the health
line and the race clock. Events are never dropped: they are appended to
`data/results/events.jsonl` *before* any delivery attempt, and retried until
the API confirms. On shutdown anything undelivered is printed with bib and
timestamp.

**`race_cv` restarted on its own** — `race_cv.log` has a `SUPERVISOR ...
RESTART #n` line with the exit status, and the next health line shows
`recovered N` for the finishes re-queued from the event log. One restart is
worth a look at the traceback above it; a restart every minute means
something structural (a camera that opens and dies, a model that no longer
loads) and the log will say which.

**`NO FRAMES PROCESSED for Ns`** — the frame loop has stopped while the
process is alive. If the same line says `camera silent`, the device stopped
delivering: the capture thread reopens it every 5 s on its own, so check
Camo, the cable and the phone's lock screen. If the camera is *not* silent,
the loop is wedged; after 30 s it exits with status 3 and the supervisor
starts a fresh process.

**`CAMERA SILENT Ns` / `reopens N` in the health line** — same thing, seen
from the health line. A few reopens after a cable wiggle are fine; a
climbing count is a bad cable or hub.

**`EVENT LOG WRITE FAILURES N`** — `events.jsonl` cannot be written (disk
full, permissions). Delivery continues and finishers still reach the
leaderboard, but a crash would now lose anything undelivered. Free disk space
before assuming it is safe.

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
display only. In order: is the tunnel up and pointing at the leaderboard
(`./broadcast.sh status`)? Does the Mac's hotspot
have signal? Reload the page on the pavilion machine — it reconnects on its
own after a blip, but a reload is the fastest confirmation. If ngrok itself
died, `./broadcast.sh leaderboard` restarts it; the page recovers by itself.

**Tablet can't reach `/admin`** — it must be on the *Mac's* hotspot, and the
address is the Mac's IP on that hotspot (`ipconfig getifaddr en0` on the
Mac), not its home-Wi-Fi address and not the `.local` name. Fallback: the
tunnel URL works from anywhere, with the password.
