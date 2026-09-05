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

check_config() {
    echo -e "${YELLOW}⚙️  Checking config...${NC}"
    if [[ ! -f "$CONFIG" ]]; then
        echo -e "${RED}❌ Config not found: $CONFIG${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Using $CONFIG${NC}"
}

start_frontend() {
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

start_api_server() {
    echo -e "${YELLOW}🐍 Starting results API (no video pipeline) natively...${NC}"
    echo -e "${BLUE}🌐 API will be available at: http://localhost:$PORT${NC}"

    PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}" \
        nohup "$RACE_CV_PYTHON" src/api_backend/local_server.py \
        --no-processor --host 0.0.0.0 --port "$PORT" \
        > api_server.log 2>&1 &
    API_PID=$!

    for _ in $(seq 1 20); do
        if curl -s -o /dev/null "http://localhost:$PORT/"; then
            echo -e "${GREEN}✅ API server is up (PID: $API_PID)${NC}"
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
check_docker
echo ""
check_ports
echo ""
check_python_env
echo ""
check_config
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
echo -e "${BLUE}📱 Frontend:${NC}  http://localhost:5173 (Docker container)"
echo -e "${BLUE}🔧 API/WS:${NC}    http://localhost:$PORT (native, PID: $API_PID)"
echo -e "${BLUE}📹 race_cv:${NC}   native, PID: $RACE_CV_PID"
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
