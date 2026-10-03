# Race morning

The commands, in order. Everything runs in the Mac's terminal, from this
repo. Explanations and what-ifs: [RACE_DAY_RUNBOOK.md](RACE_DAY_RUNBOOK.md).

## The night before

```bash
scp roster_2026.csv dantrainer@192.168.55.1:live-bib-tracking/data/results/
```
```bash
git push dantrainer@192.168.55.1:live-bib-tracking jetson-eval:refs/heads/from-mac && ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && git merge --ff-only from-mac && git branch -D from-mac'
```
```bash
cp config/ngrok-policy.yml config/ngrok-policy.local.yml   # then put a real password in the copy
```

Charge the Mac and the Jetson's power bank. Pack: USB-C cable (Mac to
Jetson), camera and its USB 3 cable, tablet.

## At the booth

Mac on the phone's hotspot and on power, lid open. Jetson powered, on the
USB-C cable, camera plugged into a blue USB 3 port.

**1. Console (Mac).** Note the tablet address it prints.
```bash
./start-race-cv.sh --api-only --fresh
```

**2. Public URL to the holding page.**
```bash
./broadcast.sh pre-race --time "around 8:30 am"
```

**3. Jetson clock.** Asks for the Jetson's sudo password.
```bash
ssh -t dantrainer@192.168.55.1 live-bib-tracking/scripts/set_clock_from_console.sh
```

**4. Detector (Jetson).**
```bash
ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./start-race-cv.sh --api-url http://192.168.55.100:8001 -r data/results/roster_2026.csv --fresh'
```

**5. Health line.** Leave this terminal open.
```bash
ssh dantrainer@192.168.55.1 tail -f live-bib-tracking/race_cv.log
```
Good: `30.0 fps`, `pending 0`, `source dropped` barely moving.
Bad: `SUPERVISOR ... RESTART`, `CAMERA SILENT`, `pending` climbing.

**6. Tablet.** Open the address from step 1 with `/admin` on the end.
Upload the roster there (names). Check the camera picture and the finish
line: `http://localhost:8001/video_feed` on the Mac.

**7. Results public,** when the pavilion should see them.
```bash
./broadcast.sh leaderboard
```

**8. At the gun:** Start the clock in Live Management.

## During the race

```bash
./broadcast.sh status          # what the public URL shows right now
```
```bash
./broadcast.sh thank-you       # emergency: hide the leaderboard, keep timing
```
Missed or wrong bib: fix it in Live Management. Nothing is lost if the
Mac or the Jetson restarts; start the same command again **without**
`--fresh`.

## Packing up

Jetson first, so it drains; then the Mac; then the URL back to the VM.

```bash
ssh dantrainer@192.168.55.1 'cd ~/live-bib-tracking && ./stop-race-cv.sh'
```
```bash
./stop-race-cv.sh
```
```bash
./broadcast.sh cloud-on
```

Results are in `data/results/race_results.txt` on the Mac.
