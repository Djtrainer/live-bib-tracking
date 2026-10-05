#!/bin/bash

# Live Bib Tracking - Race-day launcher
#
# Everything runs natively on macOS; nothing is containerised. (The 2025
# pipeline ran the detector inside the API server's video endpoint and served
# the site from Docker Desktop -- RACE_DAY_ANALYSIS.md covers why neither
# survived race day.)
#
# This script starts:
#   1. local_server.py --no-processor  -- just the results API, WebSocket,
#      and static frontend. No video pipeline runs inside it.
#   2. race_cv.run                     -- the standalone CV service, owning
#      the camera/video directly and POSTing finish events to that API.
#
# Closing a browser tab can no longer affect race timing, because nothing
# about timing lives in the HTTP process anymore.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
CAMERA_INDEX=${CAMERA_INDEX:-1}
VIDEO_PATH=""
PORT=${PORT:-8001}
CONFIG=${RACE_CV_CONFIG:-}
# On a Jetson the model is a TensorRT engine and the config that names it is
# config/race_cv.jetson.yaml (JETSON_NOTES.md). The Mac config's CoreML
# .mlpackage cannot run here at all, so defaulting to it only produces a
# misleading "install coremltools" failure. --config / RACE_CV_CONFIG still win.
if [[ -z "$CONFIG" ]]; then
    if [[ -f /etc/nv_tegra_release && -f config/race_cv.jetson.yaml ]]; then
        CONFIG="config/race_cv.jetson.yaml"
    else
        CONFIG="config/race_cv.yaml"
    fi
fi
# Never let ultralytics pip-install a missing package into the race
# interpreter mid-run; a race is not the moment for a surprise download.
export YOLO_AUTOINSTALL=false
MODEL_PATH=""
ROSTER=""
PREVIEW=0
API_ONLY=0
API_URL=""        # set by --api-url: the results API runs on another machine
# A video file is a rehearsal for a live camera, so pace it like one by
# default. --fast opts out when you just want the answer quickly.
REALTIME=1
# Override with: RACE_CV_PYTHON=/path/to/python ./start-race-cv.sh
RACE_CV_PYTHON=${RACE_CV_PYTHON:-}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --camera     Camera index (0=built-in, 1=external/iPhone), or a GStreamer"
    echo "                   launch string ending in appsink (Jetson: see JETSON_NOTES.md)"
    echo "  -v, --video      Path to video file for testing (overrides camera)"
    echo "  -p, --port       Backend port (default: 8001)"
    echo "  -m, --model      Override model path from $CONFIG"
    echo "  -r, --roster     Start-list CSV, for OCR bib snapping"
    echo "  --config         Path to race_cv config (default: config/race_cv.yaml)"
    echo "  --preview        Show an annotated OpenCV window (runs in foreground)"
    echo "  --fast           With -v, process the file as fast as possible instead"
    echo "                   of at its real frame rate (default is real time, so a"
    echo "                   rehearsal stresses the pipeline the way race day will)"
    echo "  --fresh         Start a NEW race: archive the saved results and clock"
    echo "                   instead of restoring them. Without it, a restart puts"
    echo "                   the previous race back (right mid-race, wrong next event)."
    echo "  --api-url URL    The results API runs on ANOTHER machine (the console): do"
    echo "                   not start an API or the site here, point the detector at"
    echo "                   URL and refuse to start until it answers. This is the"
    echo "                   Jetson side of a split run; the Mac runs --api-only."
    echo "  --api-only       Start only the results API and the site: this machine is"
    echo "                   the console, the detector runs elsewhere (the Jetson) and"
    echo "                   is pointed here with --api-url. Prints that URL."
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  RACE_CV_PYTHON   Python interpreter to use (must have ultralytics>=8.3)"
    echo ""
    echo "Examples:"
    echo "  $0                    # External camera (default), background"
    echo "  $0 -c 0               # Built-in camera"
    echo "  $0 -v data/raw/race.mp4 --preview   # Replay a file with a preview window"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--camera)
            CAMERA_INDEX="$2"
            CAMERA_GIVEN=1
            shift 2
            ;;
        -v|--video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -r|--roster)
            ROSTER="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --preview)
            PREVIEW=1
            shift
            ;;
        --fast)
            REALTIME=0
            shift
            ;;
        --fresh)
            # New race: archive the saved results/clock instead of restoring
            # them. The default -- restore -- is right for a mid-race restart
            # and wrong for the next event.
            FRESH=1
            shift
            ;;
        --api-url)
            API_URL="${2%/}"
            shift 2
            ;;
        --api-only)
            API_ONLY=1
            shift
            ;;
        --native-frontend)
            # Accepted for muscle memory; this is now the only mode. The
            # Docker frontend container is gone -- it was a Linux VM holding
            # 1-2 GB of an 8 GB machine to serve a static folder.
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# On a Jetson the USB camera is /dev/video0 (video1 is its metadata node, and
# the Mac default of index 1 fails to open), and a bare index negotiates
# 1280x720 because nothing asks for a size. The raw-YUY2 appsink string is
# what JETSON_NOTES.md measured live: 1080p30 at 30.0 fps with one dropped
# frame in 40 s. -c still overrides (a 60 fps camera wants framerate=60/1
# and the doubled frame-counted levers in the config).
if [[ -z "$VIDEO_PATH" && "${CAMERA_GIVEN:-0}" != "1" && -f /etc/nv_tegra_release && -e /dev/video0 ]]; then
    CAMERA_INDEX='v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=1920,height=1080,framerate=30/1 ! appsink drop=true max-buffers=1'
fi
SOURCE="$CAMERA_INDEX"
if [[ -n "$VIDEO_PATH" ]]; then
    SOURCE="$VIDEO_PATH"
fi
# A GStreamer launch string is a live camera, not a file: never pace it.
if [[ "$SOURCE" == *"!"* && "$SOURCE" == *appsink* ]]; then
    VIDEO_PATH=""
    CAMERA_INDEX="$SOURCE"
fi

echo -e "${BLUE}🚀 Live Bib Tracking - race_cv Development Setup${NC}"
echo "=============================================================="
echo -e "${YELLOW}Frontend:  built once, served by the API (port $PORT)${NC}"
echo -e "${YELLOW}API/WS:    Native macOS, no video pipeline (port $PORT)${NC}"
echo -e "${YELLOW}Pipeline:  race_cv, native macOS${NC}"

if [[ -n "$VIDEO_PATH" ]]; then
    echo -e "${YELLOW}Input:     Video file: $VIDEO_PATH${NC}"
    if [[ $REALTIME -eq 1 ]]; then
        echo -e "${YELLOW}Pacing:    Real time, dropping frames when slow (like a camera)${NC}"
    else
        echo -e "${YELLOW}Pacing:    As fast as possible (--fast)${NC}"
    fi
else
    echo -e "${YELLOW}Input:     Live camera (Index $CAMERA_INDEX)${NC}"
fi
echo ""


# Function to check if ports are available
check_ports() {
    echo -e "${YELLOW}🔍 Checking port availability...${NC}"
    # A second race_cv is the one thing worse than none: two detectors post
    # every finish twice under different event ids, and the API cannot tell.
    # The API port check below does not catch this when the API has died
    # and only race_cv is left running.
    local running
    # Anchored to the interpreter or the supervisor wrapper: a plain
    # "race_cv.run" matched an ssh session or a grep that mentioned the
    # module and refused to start (and the stop script killed them).
    running=$(pgrep -f '^[^ ]*python[0-9.]* -m race_cv\.run( |$)|^[^ ]*bash -c .* race_cv_supervisor ' 2>/dev/null | tr '\n' ' ')
    if [[ -n "$running" ]]; then
        echo -e "${RED}❌ race_cv is already running (PID $running).${NC}"
        echo -e "${BLUE}   Two detectors would report every finisher twice.${NC}"
        echo -e "${BLUE}   Stop it first:  ./stop-race-cv.sh${NC}"
        exit 1
    fi
    # The API port is the only one that has to be free: the site is served
    # from it too, and race_cv is a client, not a listener.
    if [[ -n "$API_URL" ]]; then
        echo -e "${GREEN}✅ API is remote ($API_URL); no local port needed${NC}"
        return 0
    fi
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $PORT is already in use${NC}"
        echo -e "${BLUE}💡 If this is a previous session, stop it with ./stop-race-cv.sh,${NC}"
        echo -e "${BLUE}   or find it with: lsof -Pi :$PORT -sTCP:LISTEN${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Port $PORT is available${NC}"
}

# Function to find a Python interpreter with the packages race_cv needs.
#
# ultralytics>=8.3 is required to load YOLO11 weights at all -- the base conda
# environment on this machine has 8.1.43, which fails with
# "AttributeError: Can't get attribute 'C3k2'" on this exact model. See
# RACE_DAY_ANALYSIS.md.
find_python() {
    local candidates=()
    if [[ -n "$RACE_CV_PYTHON" ]]; then
        candidates+=("$RACE_CV_PYTHON")
    fi
    candidates+=(
        "$(pwd)/.venv/bin/python"
        "$HOME/miniconda3/envs/bib_env/bin/python"
        "python3"
    )

    for candidate in "${candidates[@]}"; do
        if ! command -v "$candidate" &>/dev/null; then
            continue
        fi
        if "$candidate" -c "
import sys
try:
    import cv2, ultralytics, easyocr, requests, yaml
    major, minor, *_ = (int(p) for p in ultralytics.__version__.split('.')[:2])
    sys.exit(0 if (major, minor) >= (8, 3) else 1)
except ImportError:
    sys.exit(1)
" 2>/dev/null; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

check_python_env() {
    echo -e "${YELLOW}🐍 Checking for a race_cv-compatible Python...${NC}"
    RACE_CV_PYTHON=$(find_python) || true
    if [[ -z "$RACE_CV_PYTHON" ]]; then
        echo -e "${RED}❌ No Python with ultralytics>=8.3, opencv, easyocr, requests and pyyaml found${NC}"
        echo -e "${BLUE}💡 Point RACE_CV_PYTHON at one, e.g.:${NC}"
        echo -e "${BLUE}     RACE_CV_PYTHON=/path/to/python $0${NC}"
        echo -e "${BLUE}   Or install into .venv:${NC}"
        echo -e "${BLUE}     pip install -e .${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Using $RACE_CV_PYTHON${NC}"
    "$RACE_CV_PYTHON" -c "import ultralytics; print('   ultralytics', ultralytics.__version__)"
}

check_model_backend() {
    echo -e "${YELLOW}🤖 Checking model backend...${NC}"
    local effective_model
    effective_model=$(MODEL_PATH_OVERRIDE="$MODEL_PATH" "$RACE_CV_PYTHON" -c "
import os, sys
sys.path.insert(0, 'src')
from race_cv.config import Config
config = Config.load('$CONFIG')
print(os.environ.get('MODEL_PATH_OVERRIDE') or config.model.path)
" 2>/dev/null)

    if [[ -z "$effective_model" ]]; then
        echo -e "${RED}❌ Could not resolve model path from $CONFIG${NC}"
        exit 1
    fi
    if [[ ! -e "$effective_model" ]]; then
        echo -e "${RED}❌ Model not found: $effective_model${NC}"
        echo -e "${BLUE}   models/ is not in git. On a new machine, copy models/ from the race Mac,${NC}"
        echo -e "${BLUE}   or export from the trained weights:${NC}"
        echo -e "${BLUE}     python scripts/export_coreml.py --size 512 928${NC}"
        exit 1
    fi

    if [[ "$effective_model" == *.mlpackage && "$(uname -s)" == "Linux" ]]; then
        echo -e "${RED}❌ $effective_model is a CoreML export; CoreML does not run on Linux.${NC}"
        echo -e "${BLUE}💡 Use the Jetson config, which names a TensorRT engine built on this board:${NC}"
        echo -e "${BLUE}     $0 --config config/race_cv.jetson.yaml ...   (see JETSON_NOTES.md)${NC}"
        exit 1
    fi
    if [[ "$effective_model" == *.mlpackage ]]; then
        # ultralytics imports coremltools lazily, only when loading a .mlpackage,
        # so a missing install doesn't surface until race_cv is already running --
        # exactly the silent-until-it-isn't failure mode this project keeps
        # tripping over. Check for it now, before anything starts.
        if ! "$RACE_CV_PYTHON" -c "import coremltools" 2>/dev/null; then
            echo -e "${RED}❌ $effective_model needs coremltools, not installed for $RACE_CV_PYTHON${NC}"
            echo -e "${BLUE}💡 Install it:${NC}"
            echo -e "${BLUE}     $RACE_CV_PYTHON -m pip install coremltools${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✅ Model $effective_model${NC}"
}

check_config() {
    echo -e "${YELLOW}⚙️  Checking config...${NC}"
    if [[ ! -f "$CONFIG" ]]; then
        echo -e "${RED}❌ Config not found: $CONFIG${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Using $CONFIG${NC}"
}

check_memory() {
    # This is an 8 GB machine and vm_stat has shown it paging out under the
    # full stack. Swapping during a crossing looks exactly like a slow model:
    # dropped frames at the line. Say so before the race, not after.
    echo -e "${YELLOW}🧠 Checking memory...${NC}"
    if [[ "$(uname -s)" == "Linux" ]]; then
        local avail_kb total_kb swap_kb avail_gb total_gb
        avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        swap_kb=$(awk '/SwapTotal/ {print $2}' /proc/meminfo)
        avail_gb=$(awk -v k="$avail_kb" 'BEGIN {printf "%.1f", k/1048576}')
        total_gb=$(awk -v k="$total_kb" 'BEGIN {printf "%.1f", k/1048576}')
        echo -e "   ${total_gb} GB total, ~${avail_gb} GB available, swap $(( swap_kb / 1024 )) MB"
        if awk -v f="$avail_gb" 'BEGIN {exit !(f < 1.5)}'; then
            echo -e "${RED}⚠️  Under 1.5 GB available. On a Jetson the desktop session holds 3-4 GB;"
            echo -e "   close editors and browsers, or boot to multi-user.target for the race.${NC}"
        fi
        return
    fi
    local page_bytes free_pages total_bytes free_gb total_gb swap
    page_bytes=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
    total_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    free_pages=$(vm_stat 2>/dev/null | awk '/Pages free|Pages inactive|Pages speculative/ {gsub(/\./,"",$NF); s+=$NF} END {print s+0}')
    free_gb=$(awk -v p="$free_pages" -v b="$page_bytes" 'BEGIN {printf "%.1f", p*b/1073741824}')
    total_gb=$(awk -v t="$total_bytes" 'BEGIN {printf "%.0f", t/1073741824}')
    swap=$(sysctl -n vm.swapusage 2>/dev/null | sed -E 's/.*used = ([0-9.]+M).*/\1/')
    echo -e "   ${total_gb} GB total, ~${free_gb} GB reclaimable, swap used ${swap:-?}"
    if awk -v f="$free_gb" 'BEGIN {exit !(f < 1.5)}'; then
        echo -e "${RED}⚠️  Under 1.5 GB reclaimable. Close browsers, editors, Camo's preview"
        echo -e "   window, Docker Desktop and any other apps before racing.${NC}"
    fi
}

start_frontend() {
    # `vite build` -> dist/, which the API then serves itself. Nothing runs
    # here beyond the build, and only when dist/ is missing or stale.
    echo -e "${YELLOW}🎨 Building frontend if needed...${NC}"
    if ! command -v npm >/dev/null 2>&1; then
        if [[ -f "$(pwd)/src/frontend/dist/index.html" ]]; then
            echo -e "${YELLOW}   npm not found; serving the existing dist/ as is${NC}"
            return
        fi
        echo -e "${RED}❌ npm not found and no src/frontend/dist/ to serve; install Node${NC}"
        exit 1
    fi
    local fe="$(pwd)/src/frontend"
    if [[ ! -d "$fe/node_modules" ]]; then
        echo -e "${RED}❌ $fe/node_modules missing. Run: (cd src/frontend && npm ci)${NC}"
        exit 1
    fi
    # Rebuild when dist is missing or older than the sources.
    if [[ ! -f "$fe/dist/index.html" ]] || [[ -n "$(find "$fe/src" "$fe/index.html" -newer "$fe/dist/index.html" 2>/dev/null | head -1)" ]]; then
        echo -e "   building dist/ (API at http://localhost:$PORT)..."
        (cd "$fe" && VITE_API_BASE_URL="http://localhost:$PORT" VITE_WS_BASE_URL="ws://localhost:$PORT" \
            npm run build --silent > "$(pwd)/../../frontend_build.log" 2>&1) \
            || { echo -e "${RED}❌ frontend build failed; see frontend_build.log${NC}"; exit 1; }
    fi
    # No separate server. The API serves dist/ itself on $PORT, which is the
    # only arrangement that works from other machines: both pages build
    # their /api and /ws URLs from window.location.host, so a site served
    # from any other port would need a proxy back to the API. (An earlier
    # version of this started `vite preview` on 5173 -- it loaded, and every
    # API call from it failed.)
    echo -e "${GREEN}✅ Frontend built; the API serves it on port $PORT${NC}"
}

lan_urls() {
    # What to type into the pavilion TV's browser and the tablet at the line.
    local ip host
    # `|| true` throughout: this runs under set -e after every service is
    # already up, and a missing LAN address or ngrok agent must not abort
    # the script (it did -- the management commands never printed and the
    # launcher exited 1 with everything running).
    if [[ "$(uname -s)" == "Linux" ]]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
        host="$(hostname 2>/dev/null).local"
    else
        ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
        host="$(scutil --get LocalHostName 2>/dev/null).local"
    fi
    echo -e "${YELLOW}📺 From other machines on the same network:${NC}"
    [[ -n "$ip" ]] && echo -e "   leaderboard (pavilion TV):   ${BLUE}http://$ip:$PORT/${NC}"
    [[ -n "$ip" ]] && echo -e "   Live Management (tablet):    ${BLUE}http://$ip:$PORT/admin${NC}"
    echo -e "   or by name:                  ${BLUE}http://$host:$PORT/${NC}  (mDNS; some hotspots block it -- use the IP)"
    [[ -z "$ip" ]] && echo -e "${RED}   no LAN address -- is this machine on the hotspot?${NC}"
    # The pavilion reaches us through ngrok. If a tunnel is already running,
    # its public URL is on the agent's local API; print it so the operator
    # never has to hunt for it. See RACE_DAY_RUNBOOK.md, "Who opens what".
    local tunnel
    tunnel=$(curl -s --max-time 1 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | python3 -c "import json,sys; ts=json.load(sys.stdin).get('tunnels',[]); print(ts[0]['public_url'] if ts else '')" 2>/dev/null || true)
    if [[ -n "$tunnel" ]]; then
        echo -e "   pavilion TV (via ngrok):     ${BLUE}$tunnel/${NC}"
    else
        echo -e "   pavilion TV (via ngrok):     not running -- in another terminal:"
        echo -e "     ngrok http $PORT --url https://<your-dev-domain> --traffic-policy-file config/ngrok-policy.local.yml"
    fi
}

start_api_server() {
    echo -e "${YELLOW}🐍 Starting results API (no video pipeline) natively...${NC}"
    echo -e "${BLUE}🌐 API will be available at: http://localhost:$PORT${NC}"

    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        nohup "$RACE_CV_PYTHON" src/api_backend/local_server.py \
        --no-processor --host 0.0.0.0 --port "$PORT" ${FRESH:+--fresh} \
        > api_server.log 2>&1 &
    API_PID=$!

    for _ in $(seq 1 20); do
        if curl -s -o /dev/null "http://localhost:$PORT/"; then
            echo -e "${GREEN}✅ API server is up (PID: $API_PID)${NC}"
            local admin
            admin=$(curl -s -m 3 "http://localhost:$PORT/api/admin/session" 2>/dev/null || true)
            if [[ "$admin" == *'"configured":true'* ]]; then
                # Say where the API took the login from: exported RACE_ADMIN_*
                # variables win over the file, silently, and the operator would
                # type the file's password on the tablet and be refused.
                local admin_src
                admin_src=$(sed -nE "s/.*requires the admin login (\\(user .*\\))\$/\\1/p" api_server.log 2>/dev/null | tail -1 || true)
                echo -e "${GREEN}🔐 Live Management requires the admin login ${admin_src:-(config/admin.local.yml)}${NC}"
                if [[ "$admin_src" == *"from environment"* && -f config/admin.local.yml ]]; then
                    echo -e "${YELLOW}⚠️  RACE_ADMIN_USER / RACE_ADMIN_PASSWORD are set in this terminal and override config/admin.local.yml.${NC}"
                    echo -e "${YELLOW}   The file's password will NOT work. unset them and restart to use the file.${NC}"
                fi
                # Which addresses may post finishes without the login, as the
                # API read them: a list that re-opens that door to a whole
                # network must not look the same as a correct one.
                local detector_line
                detector_line=$(sed -nE 's/.*(Detector posts accepted without a login from: .*)$/\1/p' api_server.log 2>/dev/null | tail -1 || true)
                if [[ "$detector_line" == *"WARNING"* ]]; then
                    echo -e "${YELLOW}⚠️  ${detector_line}${NC}"
                elif [[ -n "$detector_line" ]]; then
                    echo -e "${GREEN}   ${detector_line}${NC}"
                fi
                # Every one, not the last: a stray key must not hide a misspelt detector_addresses.
                sed -nE "s/.*(Admin config: unknown key .*)\$/\1/p" api_server.log 2>/dev/null \
                    | while IFS= read -r unknown_key; do echo -e "${YELLOW}⚠️  ${unknown_key}${NC}"; done || true
            else
                echo -e "${RED}⚠️  NO ADMIN LOGIN: anyone on this network can start, stop or reset the clock and edit results.${NC}"
                echo -e "${BLUE}   cp config/admin.example.yml config/admin.local.yml, set a password in the copy, restart.${NC}"
            fi
            # Say whether this is a restored race or a fresh one. A restore
            # at the start of a NEW event is the mistake this line prevents.
            local record
            record=$(grep -E "RESTORED previous race state|Starting a fresh race" api_server.log | tail -1 | sed 's/.* - //')
            if [[ "$record" == RESTORED* ]]; then
                echo -e "${YELLOW}⚠️  $record${NC}"
            elif [[ -n "$record" ]]; then
                echo -e "   $record"
            fi
            return 0
        fi
        kill -0 "$API_PID" 2>/dev/null || break     # it exited: do not wait out the loop
        sleep 0.5
    done

    echo -e "${RED}❌ API server did not come up. Check api_server.log${NC}"
    local why
    why=$(grep -E "ADMIN LOGIN CONFIG UNUSABLE" api_server.log 2>/dev/null | tail -1 | sed -E 's/^.*ADMIN LOGIN CONFIG UNUSABLE: //' || true)
    if [[ -n "$why" ]]; then
        echo -e "${RED}   The admin login config cannot be used: $why${NC}"
    else
        tail -n 30 api_server.log
    fi
    exit 1
}

build_race_cv_args() {
    RACE_CV_ARGS=(--source "$SOURCE" --config "$CONFIG" --api-url "${API_URL:-http://localhost:$PORT}")
    if [[ -n "$MODEL_PATH" ]]; then
        RACE_CV_ARGS+=(--model "$MODEL_PATH")
    fi
    if [[ -n "$ROSTER" ]]; then
        RACE_CV_ARGS+=(--roster "$ROSTER")
    fi
    if [[ $PREVIEW -eq 1 ]]; then
        RACE_CV_ARGS+=(--preview)
    fi
    # Only meaningful for a file; a camera is already real time.
    if [[ $REALTIME -eq 1 && -n "$VIDEO_PATH" ]]; then
        RACE_CV_ARGS+=(--realtime)
    fi
}

# A roster path that does not exist used to cost one WARNING line in the log
# and a whole race without bib snapping. Refuse instead, and say what is there.
check_roster() {
    [[ $API_ONLY -eq 1 ]] && return 0      # the console does not read it; names come from the roster upload
    if [[ -z "$ROSTER" ]]; then
        echo -e "${YELLOW}⚠️  No roster (-r): bib reads will not be checked against the start list${NC}"
        return 0
    fi
    if [[ ! -f "$ROSTER" ]]; then
        echo -e "${RED}❌ Roster not found: $ROSTER${NC}"
        local found
        found=$(ls data/results/*.csv data/raw/*.csv 2>/dev/null | head -8)
        [[ -n "$found" ]] && echo -e "${BLUE}   CSV files here:${NC}" && echo "$found" | sed 's/^/     /'
        exit 1
    fi
    echo -e "${GREEN}✅ Roster $ROSTER ($(($(wc -l < "$ROSTER") - 1)) rows)${NC}"
}

measure_skew() {  # this machine's clock minus the console's, seconds (+ = ahead); empty if no /api/time
    python3 - "$API_URL" <<'PY' 2>/dev/null
import json, sys, time, urllib.request
url = sys.argv[1] + "/api/time"
direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy
best = None
for _ in range(5):
    t0 = time.time()
    try:
        epoch = float(json.load(direct.open(url, timeout=4))["epoch"])
    except Exception:
        break
    t3 = time.time()
    rtt = t3 - t0
    if best is None or rtt < best[1]:
        best = ((t0 + t3) / 2 - epoch, rtt)
print(f"{best[0]:+.3f}" if best else "")
PY
}

# The console answers but refuses this machine's posts (a login is set and
# this address is not a detector address). Said in red here and again as the
# launcher's last lines -- but NOT a reason to exit: the detector still times
# every finish into its event log and retries, and they all land, with their
# real times, once the console lists this address. Exiting here left the line
# untimed for as long as the console took to fix.
DETECTOR_REFUSED=0
DETECTOR_ADDR=""
say_detector_refused() {
    echo -e "${RED}❌ The console has an admin login and does not accept detector posts from this machine${DETECTOR_ADDR:+ ($DETECTOR_ADDR)}.${NC}"
    echo -e "${RED}   Every finish and every preview frame is REFUSED until that is fixed. The detector runs anyway:${NC}"
    echo -e "${RED}   finishes are timed into its event log and land once the console accepts this address.${NC}"
    echo -e "${BLUE}💡 Over the USB link use  --api-url http://192.168.55.100:$PORT  (accepted with no setup).${NC}"
    echo -e "${BLUE}   Over a hotspot, on the console add this machine's address to config/admin.local.yml:${NC}"
    echo -e "${BLUE}     detector_addresses: [\"${DETECTOR_ADDR:-JETSON_ADDRESS}\"]${NC}"
    echo -e "${BLUE}   and restart the console (./stop-race-cv.sh, then ./start-race-cv.sh --api-only; no --fresh mid-race).${NC}"
}

# --api-url: the API lives on the console machine. Refuse to start a detector
# that has nowhere to deliver; a finish logged only to the event log is a
# recovery job, not a result.
check_remote_api() {
    echo -e "${YELLOW}🛰  Checking the results API at $API_URL...${NC}"
    local code
    # --noproxy: an HTTP_PROXY exported on this machine must not sit between
    # the detector and its console (the detector itself ignores it too).
    code=$(curl -s --noproxy '*' -o /dev/null -m 4 -w '%{http_code}' "$API_URL/api/clock/status" 2>/dev/null || true)
    if [[ "$code" != "200" ]]; then
        echo -e "${RED}❌ No results API at $API_URL (HTTP ${code:-none}).${NC}"
        echo -e "${BLUE}💡 On the console machine run:  ./start-race-cv.sh --api-only --fresh -r roster.csv${NC}"
        echo -e "${BLUE}   and use the --api-url it prints (USB link: http://192.168.55.100:$PORT).${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ API answers at $API_URL${NC}"
    # With a login set on the console, the detector is known by its address.
    # Ask now, with a body the console will not show (not a JPEG): a 401
    # means every finish and every preview frame from here is refused. Say so
    # and go on (see say_detector_refused): a refused finish is a delayed one.
    local probe
    probe=$(curl -s --noproxy '*' -o /dev/null -m 4 -w '%{http_code}' -X POST -H 'Content-Type: image/jpeg' --data-binary 'probe' "$API_URL/api/frame" 2>/dev/null || true)
    if [[ "$probe" == "401" ]]; then
        DETECTOR_ADDR=$(ip -4 route get "$(echo "$API_URL" | sed -E 's#^[a-z]+://##; s#[:/].*##')" 2>/dev/null | sed -nE 's/.* src ([0-9.]+).*/\1/p' | head -1 || true)
        DETECTOR_REFUSED=1
        say_detector_refused
    elif [[ "$probe" == "200" ]]; then
        echo -e "${GREEN}✅ The console accepts detector posts from this machine${NC}"
    fi
    # The API converts every finish against ITS clock; a detector on another
    # machine with a different idea of the time shifts every result by the
    # difference. The API's Date header is whole seconds, which is enough to
    # catch the case that matters: a board that booted with no internet.
    # GET /api/time gives the console's clock to the millisecond; five
    # samples, the one with the shortest round trip wins (the same
    # measurement race_cv repeats at startup to correct its timestamps).
    # An older console without /api/time falls back to the Date header of
    # a GET (whole seconds; a HEAD is answered 405 by FastAPI).
    local skew
    skew=$(measure_skew)
    if [[ -z "$skew" ]]; then
        local remote_epoch local_epoch
        remote_epoch=$(curl -s --noproxy '*' -m 4 -D - -o /dev/null "$API_URL/api/clock/status" 2>/dev/null | awk -F': ' 'tolower($1)=="date"{print $2}' | tr -d '\r')
        if [[ -n "$remote_epoch" ]]; then
            remote_epoch=$(date -u -d "$remote_epoch" +%s 2>/dev/null || date -u -j -f "%a, %d %b %Y %T %Z" "$remote_epoch" +%s 2>/dev/null || true)
            local_epoch=$(date -u +%s)
            [[ -n "$remote_epoch" ]] && skew=$(printf '%+d' $((local_epoch - remote_epoch)))
        fi
    fi
    if [[ -z "$skew" ]]; then
        echo -e "${YELLOW}⚠️  Could not compare clocks with the console; make sure they agree before racing${NC}"
        return 0
    fi
    # What the skew costs. race_cv measures this same offset at startup and
    # corrects every finish time by it, so a few seconds is harmless. Over
    # a minute the detector's log, its event log and the 12 h recovery
    # cut-off all run on the wrong time, and that is refused.
    skew_over() { python3 -c "import sys; sys.exit(0 if abs(float('$1')) > $2 else 1)"; }
    local me set_cmd
    me=$(ip -4 route get "$(echo "$API_URL" | sed -E 's#^[a-z]+://##; s#[:/].*##')" 2>/dev/null | sed -nE 's/.* src ([0-9.]+).*/\1/p' | head -1)
    [[ -z "$me" ]] && me=$(hostname -I 2>/dev/null | awk '{print $1}')
    [[ -z "$me" ]] && me=$(ipconfig getifaddr en0 2>/dev/null || true)
    set_cmd="ssh -t $(whoami)@${me:-JETSON_ADDRESS} $(pwd)/scripts/set_clock_from_console.sh $API_URL"
    if skew_over "$skew" 2.0 && sudo -n "$(command -v date)" -u >/dev/null 2>&1 && [[ -x scripts/set_clock_from_console.sh ]]; then
        # sudoers lets this user run date without a password: fix it here.
        echo -e "${YELLOW}⏱  Clock is ${skew}s from the console's; setting it...${NC}"
        scripts/set_clock_from_console.sh "$API_URL" 2>&1 | sed 's/^/   /' || true
        skew=$(measure_skew)
    fi
    if skew_over "$skew" 60; then
        echo -e "${RED}❌ This machine's clock is ${skew}s from the console's. Results would still be corrected, but the${NC}"
        echo -e "${RED}   detector's log, its event log and the recovery cut-off would all run on the wrong time.${NC}"
        echo -e "${BLUE}💡 From the console, with no detector running here (asks for this machine's sudo password):${NC}"
        echo -e "${BLUE}     $set_cmd${NC}"
        exit 1
    elif skew_over "$skew" 0.25; then
        echo -e "${YELLOW}⚠️  Clock is ${skew}s from the console's. The detector measures this at startup and corrects every${NC}"
        echo -e "${YELLOW}   finish time by it, so results are right. To set the clock itself:  $set_cmd${NC}"
    else
        echo -e "${GREEN}✅ Clock ${skew}s from the console${NC}"
    fi
}

# macOS sleeps a laptop on battery after a minute idle, which drops the USB
# link and the API mid-race. Hold it awake for as long as the given process
# lives. (Nothing prevents lid-close sleep: keep the lid open.)
hold_awake() {
    local pid="$1" what="$2"
    if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null 2>&1; then
        nohup caffeinate -dimsu -w "$pid" >/dev/null 2>&1 &
        echo -e "${GREEN}☕ caffeinate holds this Mac awake while $what runs; keep the lid open${NC}"
    fi
}

start_race_cv_background() {
    echo -e "${YELLOW}📹 Starting race_cv pipeline natively (background, supervised)...${NC}"
    build_race_cv_args
    # --fresh archives the API's saved race; the same must happen to
    # race_cv's event log, or every unconfirmed event from the last
    # rehearsal is re-queued on start and lands in the new race as a
    # finisher (16 of them did, in testing).
    if [[ "${FRESH:-0}" == "1" ]]; then
        local event_log rotated
        event_log=$(grep -A1 '^sink:' "$CONFIG" | grep event_log | awk '{print $2}' || true)
        event_log=${event_log:-data/results/events.jsonl}
        if [[ -s "$event_log" ]]; then
            rotated="${event_log%.jsonl}.$(date +%Y%m%d-%H%M%S).jsonl"
            mv "$event_log" "$rotated"
            echo -e "   --fresh: previous event log moved to $rotated"
        fi
    fi
    # Crash-only supervisor. race_cv exits 0 when it was told to stop
    # (stop-race-cv.sh, 'q' in the preview, end of a file) and that is final.
    # Any other exit -- error budget exhausted, a wedged frame loop the
    # watchdog bailed out of, an uncaught crash, the OS killing it -- is
    # restarted after a short pause, indefinitely: a dead detector at the
    # finish line is worse than a noisy log. Restarting is safe because the
    # sink re-queues every unconfirmed event from the event log on start and
    # the API is idempotent on eventId.
    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        nohup bash -c '
            py="$1"; shift
            attempt=0
            while :; do
                "$py" -m race_cv.run "$@"; status=$?
                [ "$status" -eq 0 ] && exit 0
                attempt=$((attempt + 1))
                echo "$(date "+%Y-%m-%d %H:%M:%S") SUPERVISOR race_cv exited with status $status; RESTART #$attempt in 5s" >&2
                sleep 5
            done' race_cv_supervisor "$RACE_CV_PYTHON" "${RACE_CV_ARGS[@]}" \
        > race_cv.log 2>&1 &
    RACE_CV_PID=$!
    sleep 2
    if kill -0 "$RACE_CV_PID" 2>/dev/null; then
        echo -e "${GREEN}✅ race_cv started (PID: $RACE_CV_PID)${NC}"
    else
        echo -e "${RED}❌ race_cv failed to start. Check race_cv.log${NC}"
        tail -n 30 race_cv.log
        exit 1
    fi
}

# --- Main execution ---
echo -e "${BLUE}🔍 Running pre-flight checks...${NC}"
echo ""
check_memory
echo ""
check_ports
echo ""
check_python_env
echo ""
check_config
echo ""
if [[ $API_ONLY -eq 1 ]]; then
    echo -e "${BLUE}🛰  --api-only: the detector runs on another machine; skipping the model check${NC}"
else
    check_model_backend
fi
echo ""

check_roster
echo ""
if [[ -n "$API_URL" ]]; then
    check_remote_api
else
    start_frontend
    echo ""
    start_api_server
fi
echo ""

if [[ $API_ONLY -eq 1 ]]; then
    hold_awake "$API_PID" "the API"
    echo -e "${GREEN}🎉 Results API and site are running (--api-only; no detector here).${NC}"
    echo ""
    echo -e "${BLUE}📱 Frontend:${NC}  http://localhost:$PORT (served by the API)"
    echo -e "${BLUE}🔧 API/WS:${NC}    http://localhost:$PORT (native, PID: $API_PID)"
    echo ""
    lan_urls
    echo ""
    echo -e "${YELLOW}📹 The detector on the Jetson posts to this machine. Start it there with:${NC}"
    usb_link=$(ifconfig 2>/dev/null | awk '/inet 192\.168\.55\./{print $2; exit}' || true)
    lan_ip=$(ipconfig getifaddr en0 2>/dev/null || true)
    [[ -n "$usb_link" ]] && echo -e "   over the USB link:   ${BLUE}--api-url http://$usb_link:$PORT${NC}   (L4T device mode; the Jetson is 192.168.55.1)"
    [[ -n "$lan_ip" ]]   && echo -e "   over the hotspot:    ${BLUE}--api-url http://$lan_ip:$PORT${NC}"
    [[ -n "$lan_ip" ]]   && echo -e "                        (with a login set, also list the Jetson's hotspot address under detector_addresses in config/admin.local.yml)"
    [[ -z "$usb_link" && -z "$lan_ip" ]] && echo -e "${RED}   no USB-link or LAN address found on this machine${NC}"
    echo ""
    echo -e "${YELLOW}📋 Management Commands:${NC}"
    echo -e "${BLUE}  Tail API logs:${NC}   tail -f api_server.log"
    echo -e "${BLUE}  Stop:${NC}            ./stop-race-cv.sh   (stop the detector on the Jetson first, so it can drain)"
    exit 0
fi

if [[ $PREVIEW -eq 1 ]]; then
    # A preview window wants to be interactive: run race_cv in the foreground so
    # 'q' in the window or Ctrl+C in this terminal stops it directly.
    echo -e "${YELLOW}📹 Starting race_cv in the foreground (--preview)...${NC}"
    echo -e "${BLUE}Press 'q' in the preview window or Ctrl+C here to stop.${NC}"
    echo ""
    build_race_cv_args
    if [[ -n "$API_PID" ]]; then
        trap 'echo -e "\n${YELLOW}Stopping API server (PID: $API_PID)...${NC}"; kill "$API_PID" 2>/dev/null || true' EXIT
    fi
    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$RACE_CV_PYTHON" -m race_cv.run "${RACE_CV_ARGS[@]}"
    exit 0
fi

start_race_cv_background
hold_awake "$RACE_CV_PID" "the detector"

echo ""
echo -e "${GREEN}🎉 All services are running!${NC}"
echo ""
if [[ -n "$API_URL" ]]; then
    echo -e "${BLUE}🔧 API/WS:${NC}    $API_URL (on the console machine; the site is there too)"
    echo -e "${BLUE}📹 race_cv:${NC}   native, PID: $RACE_CV_PID (supervised)"
    echo ""
    echo -e "${YELLOW}👀 Watch it from the console:${NC}  ssh $(whoami)@$(hostname -I 2>/dev/null | awk '{print $1}') tail -f $(pwd)/race_cv.log"
else
    echo -e "${BLUE}📱 Frontend:${NC}  http://localhost:$PORT (served by the API)"
    echo -e "${BLUE}🔧 API/WS:${NC}    http://localhost:$PORT (native, PID: $API_PID)"
    echo -e "${BLUE}📹 race_cv:${NC}   native, PID: $RACE_CV_PID"
    echo ""
    lan_urls
fi
echo ""
echo -e "${YELLOW}📋 Management Commands:${NC}"
echo -e "${BLUE}  Tail API logs:${NC}         tail -f api_server.log"
echo -e "${BLUE}  Tail pipeline logs:${NC}    tail -f race_cv.log"
echo -e "${BLUE}  Stop everything:${NC}       ./stop-race-cv.sh   (race_cv first, so it can drain)"
echo -e "${BLUE}  Stop race_cv only:${NC}     pkill -TERM -f race_cv.run   (kills the supervisor too)"
echo ""
echo -e "${YELLOW}💡 Undelivered finish events, if any, are preserved in $(grep -A1 '^sink:' "$CONFIG" | grep event_log | awk '{print $2}')${NC}"
# Last, so it cannot scroll away. An if, not `[[ ]] &&`: as the script's final
# command a false test would make a healthy start exit 1.
if [[ "$DETECTOR_REFUSED" == "1" ]]; then
    echo ""
    say_detector_refused
fi
