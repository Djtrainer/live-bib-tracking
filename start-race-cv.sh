#!/bin/bash

# Live Bib Tracking - Hybrid Development Launcher (race_cv pipeline)
#
# Same shape as start-dev.sh: frontend in Docker, everything else native on
# macOS. The difference is what runs the pipeline. start-dev.sh launches
# run_live_native.sh, which runs the legacy VideoInferenceProcessor in-process
# inside local_server.py -- the pipeline documented in RACE_DAY_ANALYSIS.md as
# the source of the race-day failures (silently dropped finishers, frame-skip
# bursts, hardcoded finish-line geometry).
#
# This script instead starts:
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
CONFIG=${RACE_CV_CONFIG:-"config/race_cv.yaml"}
MODEL_PATH=""
ROSTER=""
PREVIEW=0
# A video file is a rehearsal for a live camera, so pace it like one by
# default. --fast opts out when you just want the answer quickly.
REALTIME=1
# Override with: RACE_CV_PYTHON=/path/to/python ./start-race-cv.sh
RACE_CV_PYTHON=${RACE_CV_PYTHON:-}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --camera     Camera index (0=built-in, 1=external/iPhone)"
    echo "  -v, --video      Path to video file for testing (overrides camera)"
    echo "  -p, --port       Backend port (default: 8001)"
    echo "  -m, --model      Override model path from $CONFIG"
    echo "  -r, --roster     Start-list CSV, for OCR bib snapping"
    echo "  --config         Path to race_cv config (default: config/race_cv.yaml)"
    echo "  --preview        Show an annotated OpenCV window (runs in foreground)"
    echo "  --fast           With -v, process the file as fast as possible instead"
    echo "                   of at its real frame rate (default is real time, so a"
    echo "                   rehearsal stresses the pipeline the way race day will)"
    echo "  --native-frontend"
    echo "                   Skip Docker; the API serves the leaderboard and Live"
    echo "                   Management itself on the API port, reachable from the"
    echo "                   pavilion. Frees the 1-2 GB Docker Desktop holds."
    echo "  --fresh          Start a NEW race: archive the saved results and clock"
    echo "                   instead of restoring them. Without it, a restart puts"
    echo "                   the previous race back (right mid-race, wrong next event)."
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
        --native-frontend)
            # Serve the built leaderboard with Vite's preview server instead of
            # a Docker container. Docker Desktop on macOS is a Linux VM that
            # holds 1-2 GB of an 8 GB machine to serve a static folder; on race
            # day that memory is worth more to the detector than to a VM.
            NATIVE_FRONTEND=1
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

SOURCE="$CAMERA_INDEX"
if [[ -n "$VIDEO_PATH" ]]; then
    SOURCE="$VIDEO_PATH"
fi

echo -e "${BLUE}🚀 Live Bib Tracking - race_cv Development Setup${NC}"
echo "=============================================================="
echo -e "${YELLOW}Frontend:  Docker Container (port 5173)${NC}"
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

# Function to check if Docker is running
check_docker() {
    echo -e "${YELLOW}🐳 Checking Docker...${NC}"
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running${NC}"
        echo -e "${BLUE}💡 Please start Docker Desktop and try again${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker is running${NC}"
}

# Function to check if ports are available
check_ports() {
    echo -e "${YELLOW}🔍 Checking port availability...${NC}"
    # 5173 is not checked here: docker compose owns that port, and
    # `docker compose up -d --build` is idempotent -- reusing an already-running
    # frontend container is the normal case, not a conflict. Only the backend
    # port needs to be free, since that's a plain native process this script
    # is about to start.
    if lsof -Pi :"$PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}❌ Port $PORT is already in use${NC}"
        echo -e "${BLUE}💡 If this is a previous run_live_native.sh or race_cv session,${NC}"
        echo -e "${BLUE}   find it with: lsof -Pi :$PORT -sTCP:LISTEN${NC}"
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
        echo -e "   window and any other apps before racing; pass --native-frontend to"
        echo -e "   avoid Docker Desktop's VM entirely.${NC}"
    fi
}

start_frontend() {
    if [[ "${NATIVE_FRONTEND:-0}" == "1" ]]; then
        start_frontend_native
        return
    fi
    echo -e "${YELLOW}🎨 Starting frontend container...${NC}"
    docker compose up -d --build
    sleep 3
    if docker compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ Frontend container started successfully${NC}"
        echo -e "${BLUE}🌐 Frontend available at: http://localhost:5173${NC}"
    else
        echo -e "${RED}❌ Failed to start frontend container${NC}"
        docker compose logs
        exit 1
    fi
}

start_frontend_native() {
    # Same artefact the container serves (`vite build` -> dist/), served by
    # Vite's own preview server, which does SPA fallback and needs nothing
    # installed beyond the existing node_modules. The API base URL is baked
    # in at build time, so it is built here with localhost rather than the
    # container's host.docker.internal.
    echo -e "${YELLOW}🎨 Starting frontend natively (no Docker)...${NC}"
    if ! command -v npm >/dev/null 2>&1; then
        echo -e "${RED}❌ npm not found; install Node or drop --native-frontend${NC}"
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
    ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
    host="$(scutil --get LocalHostName 2>/dev/null).local"
    echo -e "${YELLOW}📺 From other machines on the same network:${NC}"
    [[ -n "$ip" ]] && echo -e "   leaderboard (pavilion TV):   ${BLUE}http://$ip:$PORT/${NC}"
    [[ -n "$ip" ]] && echo -e "   Live Management (tablet):    ${BLUE}http://$ip:$PORT/admin${NC}"
    echo -e "   or by name:                  ${BLUE}http://$host:$PORT/${NC}  (mDNS; some hotspots block it -- use the IP)"
    [[ -z "$ip" ]] && echo -e "${RED}   no LAN address on en0/en1 -- is the Mac on the hotspot?${NC}"
    # The pavilion reaches us through ngrok. If a tunnel is already running,
    # its public URL is on the agent's local API; print it so the operator
    # never has to hunt for it. See RACE_DAY_RUNBOOK.md, "Who opens what".
    local tunnel
    tunnel=$(curl -s --max-time 1 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | python3 -c "import json,sys; ts=json.load(sys.stdin).get('tunnels',[]); print(ts[0]['public_url'] if ts else '')" 2>/dev/null)
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
        sleep 0.5
    done

    echo -e "${RED}❌ API server did not come up. Check api_server.log${NC}"
    tail -n 30 api_server.log
    exit 1
}

build_race_cv_args() {
    RACE_CV_ARGS=(--source "$SOURCE" --config "$CONFIG" --api-url "http://localhost:$PORT")
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

start_race_cv_background() {
    echo -e "${YELLOW}📹 Starting race_cv pipeline natively (background)...${NC}"
    build_race_cv_args
    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        nohup "$RACE_CV_PYTHON" -m race_cv.run "${RACE_CV_ARGS[@]}" \
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
if [[ "${NATIVE_FRONTEND:-0}" == "1" ]]; then
    echo -e "${GREEN}✅ Docker not needed (--native-frontend)${NC}"
else
    check_docker
fi
echo ""
check_memory
echo ""
check_ports
echo ""
check_python_env
echo ""
check_config
echo ""
check_model_backend
echo ""

start_frontend
echo ""
start_api_server
echo ""

if [[ $PREVIEW -eq 1 ]]; then
    # A preview window wants to be interactive: run race_cv in the foreground so
    # 'q' in the window or Ctrl+C in this terminal stops it directly.
    echo -e "${YELLOW}📹 Starting race_cv in the foreground (--preview)...${NC}"
    echo -e "${BLUE}Press 'q' in the preview window or Ctrl+C here to stop.${NC}"
    echo ""
    build_race_cv_args
    trap 'echo -e "\n${YELLOW}Stopping API server (PID: $API_PID)...${NC}"; kill "$API_PID" 2>/dev/null || true' EXIT
    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        "$RACE_CV_PYTHON" -m race_cv.run "${RACE_CV_ARGS[@]}"
    exit 0
fi

start_race_cv_background

echo ""
echo -e "${GREEN}🎉 All services are running!${NC}"
echo ""
if [[ "${NATIVE_FRONTEND:-0}" == "1" ]]; then
    echo -e "${BLUE}📱 Frontend:${NC}  http://localhost:$PORT (served by the API)"
else
    echo -e "${BLUE}📱 Frontend:${NC}  http://localhost:5173 (Docker container)"
fi
echo -e "${BLUE}🔧 API/WS:${NC}    http://localhost:$PORT (native, PID: $API_PID)"
echo -e "${BLUE}📹 race_cv:${NC}   native, PID: $RACE_CV_PID"
echo ""
lan_urls
echo ""
echo -e "${YELLOW}📋 Management Commands:${NC}"
echo -e "${BLUE}  Check frontend status:${NC} docker compose ps"
echo -e "${BLUE}  Stop frontend:${NC}         docker compose down"
echo -e "${BLUE}  Tail API logs:${NC}         tail -f api_server.log"
echo -e "${BLUE}  Tail pipeline logs:${NC}    tail -f race_cv.log"
echo -e "${BLUE}  Stop API server:${NC}       kill $API_PID"
echo -e "${BLUE}  Stop race_cv:${NC}          kill $RACE_CV_PID"
echo -e "${BLUE}  Stop everything:${NC}       docker compose down && kill $API_PID $RACE_CV_PID"
echo ""
echo -e "${YELLOW}💡 Undelivered finish events, if any, are preserved in $(grep -A1 '^sink:' "$CONFIG" | grep event_log | awk '{print $2}')${NC}"
