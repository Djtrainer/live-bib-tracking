#!/bin/bash

# Live Bib Tracking - Stop everything started by start-race-cv.sh (or start-dev.sh)
#
# Stops, in order:
#   1. race_cv.run           -- SIGTERM, then wait for it to drain and exit
#   2. local_server.py       -- API/WS/static server (--no-processor or legacy)
#   3. run_live_native.sh    -- legacy wrapper, if that's what's running instead
#   4. the frontend -- the native Vite preview server if start-race-cv.sh
#      was run with --native-frontend (PID in .frontend.pid), and/or the
#      Docker container (docker compose down)
#
# The ordering matters: race_cv is stopped *before* the API server so any
# finish event still retrying (e.g. because the race clock wasn't running yet)
# gets a chance to actually deliver during its graceful shutdown, instead of
# failing every retry against a server that's already gone. See sink.py and
# RACE_DAY_ANALYSIS.md for why silently losing that event would be exactly the
# bug this project exists to fix -- SIGTERM here is never SIGKILL by default,
# specifically so that drain can happen.
#
# PIDs aren't tracked in a file; this finds processes by matching their command
# line, the same way you'd find them by hand with `pgrep -fl`.

# Deliberately no `set -u`: macOS ships bash 3.2, where `set -u` treats a
# legitimately-empty array's `${arr[@]}` expansion as an unbound variable
# (fixed in bash 4.4+, but this script has to run on the stock shell).

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

GRACE_SECONDS=${GRACE_SECONDS:-20}
FORCE=0
KEEP_FRONTEND=0

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -f, --force          Skip the graceful wait; SIGKILL immediately"
    echo "  --grace SECONDS      How long to wait for graceful shutdown (default: 20)"
    echo "  --keep-frontend      Leave the Docker frontend container running"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Stops race_cv, the results API server (local_server.py, either mode),"
    echo "the legacy run_live_native.sh wrapper if that's what's running, and the"
    echo "frontend container -- whichever of start-race-cv.sh / start-dev.sh you used."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=1
            shift
            ;;
        --grace)
            GRACE_SECONDS="$2"
            shift 2
            ;;
        --keep-frontend)
            KEEP_FRONTEND=1
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

echo -e "${BLUE}🛑 Live Bib Tracking - Stopping services${NC}"
echo "=============================================================="

# Find PIDs matching a command-line pattern, excluding this script's own PID
# and its parent shell (pgrep -f can otherwise match the very pipeline that's
# invoking it, e.g. if this script's own path contains a matching substring).
#
# Populates the array named by $1 (bash 3.2, macOS's default, has no mapfile).
find_pids() {
    local __outvar="$1"
    local pattern="$2"
    local pid
    eval "$__outvar=()"
    while IFS= read -r pid; do
        [[ -n "$pid" ]] && eval "$__outvar+=(\"\$pid\")"
    done < <(pgrep -f "$pattern" 2>/dev/null | grep -vx -e "$$" -e "$PPID")
}

# Send SIGTERM to a set of PIDs, then wait up to $GRACE_SECONDS for all of them
# to exit before escalating to SIGKILL. Skips the wait entirely with --force.
stop_pids() {
    local label="$1"
    shift
    local pids=("$@")

    if [[ ${#pids[@]} -eq 0 ]]; then
        echo -e "${BLUE}  no $label process running${NC}"
        return 0
    fi

    for pid in "${pids[@]}"; do
        echo -e "${YELLOW}  stopping $label (PID $pid)...${NC}"
        kill "$pid" 2>/dev/null || true
    done

    if [[ $FORCE -eq 1 ]]; then
        for pid in "${pids[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
        return 0
    fi

    local waited=0
    while [[ $waited -lt $GRACE_SECONDS ]]; do
        local still_alive=0
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && still_alive=1
        done
        [[ $still_alive -eq 0 ]] && break
        sleep 1
        waited=$((waited + 1))
    done

    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${RED}  $label (PID $pid) did not stop within ${GRACE_SECONDS}s, sending SIGKILL${NC}"
            kill -9 "$pid" 2>/dev/null || true
        else
            echo -e "${GREEN}  $label (PID $pid) stopped${NC}"
        fi
    done
}

# --- 1. race_cv first, so it can drain against a still-live API ---
echo -e "${YELLOW}📹 Stopping race_cv...${NC}"
find_pids RACE_CV_PIDS "race_cv[/.]run"
stop_pids "race_cv" "${RACE_CV_PIDS[@]}"
echo ""

# --- 2. the API/results server, whichever mode it's running in ---
echo -e "${YELLOW}🐍 Stopping backend (local_server.py)...${NC}"
find_pids BACKEND_PIDS "api_backend/local_server\.py"
stop_pids "backend" "${BACKEND_PIDS[@]}"
echo ""

# --- 3. the legacy wrapper script, if start-dev.sh was used instead ---
echo -e "${YELLOW}🎬 Stopping run_live_native.sh (if present)...${NC}"
find_pids NATIVE_PIDS "run_live_native\.sh"
stop_pids "run_live_native.sh" "${NATIVE_PIDS[@]}"
echo ""

# --- 4. the frontend: native preview server and/or Docker container ---
if [[ $KEEP_FRONTEND -eq 1 ]]; then
    echo -e "${BLUE}Leaving frontend running (--keep-frontend)${NC}"
else
    # With --native-frontend the API serves the site itself, so stopping the
    # API (step 2) already stopped it; only a Docker container is left here.
    echo -e "${YELLOW}🎨 Stopping frontend container (if any)...${NC}"
    if docker info >/dev/null 2>&1; then
        docker compose down
        echo -e "${GREEN}✅ Frontend container stopped${NC}"
    else
        echo -e "${BLUE}  Docker is not running; nothing to stop${NC}"
    fi
fi

echo ""
echo -e "${GREEN}✅ Done.${NC}"

# Undelivered finish events, if any, live in whatever event_log the config
# pointed at (default: data/results/events.jsonl) -- check there if a run
# ended before every finisher was confirmed delivered.
if [[ -f "data/results/events.jsonl" ]]; then
    PENDING=$(python3 -c "
import json
pending = set()
delivered = set()
try:
    with open('data/results/events.jsonl') as f:
        for line in f:
            rec = json.loads(line)
            (delivered if rec['status'] == 'delivered' else pending).add(rec['event_id'])
except FileNotFoundError:
    pass
print(len(pending - delivered))
" 2>/dev/null || echo 0)
    if [[ "$PENDING" != "0" ]]; then
        echo -e "${RED}⚠️  $PENDING finish event(s) in data/results/events.jsonl were never confirmed delivered.${NC}"
        echo -e "${BLUE}   Check them before assuming the leaderboard is complete.${NC}"
    fi
fi
