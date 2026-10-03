#!/bin/bash
# Set this machine's clock from the console's results API. Run it ON the
# detector machine (the Jetson); from the console that is:
#
#   ssh -t dantrainer@192.168.55.1 live-bib-tracking/scripts/set_clock_from_console.sh
#
# sudo is authenticated FIRST and the time is fetched AFTER. A timestamp
# computed on the console before the password prompt is stale by however
# long the typing took: the first version of this command left the board
# 3.4 s behind every time, which the launcher then refused.
#
#   set_clock_from_console.sh [API_URL]     default http://192.168.55.100:8001
set -u
API=${1:-http://192.168.55.100:8001}
DATE=$(command -v date)

console_now() {  # the console's clock right now, to the millisecond
    python3 - "$API" <<'PY'
import json, sys, time, urllib.request
url = sys.argv[1].rstrip("/") + "/api/time"
best = None
for _ in range(5):
    t0 = time.monotonic()
    epoch = float(json.load(urllib.request.urlopen(url, timeout=4))["epoch"])
    t3 = time.monotonic()
    if best is None or (t3 - t0) < best[1]:
        best = (epoch + (t3 - t0) / 2, t3 - t0, t3)   # server time at the moment of t3
print(f"{best[0] + (time.monotonic() - best[2]):.3f}")
PY
}
skew() {
    python3 - "$API" <<'PY'
import json, sys, time, urllib.request
t0 = time.time(); e = float(json.load(urllib.request.urlopen(sys.argv[1].rstrip("/") + "/api/time", timeout=4))["epoch"]); t3 = time.time()
print(f"{(t0 + t3) / 2 - e:+.3f}")
PY
}

# No prompt if sudoers allows date without a password; otherwise ask now,
# before any timestamp exists.
if ! sudo -n "$DATE" -u >/dev/null 2>&1; then
    sudo -v || { echo "sudo failed; clock not set"; exit 1; }
fi
before=$(skew 2>/dev/null) || { echo "console API not reachable at $API -- start it first (./start-race-cv.sh --api-only)"; exit 1; }
now=$(console_now) || { echo "could not read $API/api/time"; exit 1; }
sudo "$DATE" -u -s "@$now" >/dev/null || { echo "date -s failed"; exit 1; }
echo "clock set from $API: was ${before}s off, now $(skew)s; it is $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
