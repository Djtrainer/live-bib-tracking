#!/bin/bash
# Which site the public URL shows: the live leaderboard, the pre-race
# holding page, or the thank-you page.
#
# One ngrok tunnel, one URL, three possible upstreams:
#   8001  the results API serving the leaderboard (start-race-cv.sh)
#   8002  "The race hasn't started yet, check back around 8:30"
#   8003  "Thank you for coming" -- for when the leaderboard must not be seen
#
# Switching is a restart of the ngrok agent against a different local port
# (the URL, the traffic policy and the interstitial are unchanged); it takes
# a few seconds, and the holding pages reload themselves so open tabs follow.
#
#   ./broadcast.sh pre-race [--time "8:30 am"]   before the stack is up
#   ./broadcast.sh leaderboard                   when results should be public
#   ./broadcast.sh thank-you                     if something goes wrong
#   ./broadcast.sh status                        what the URL points at now
#   ./broadcast.sh stop                          stop ngrok and the page servers
#
# The page servers keep running once started (they are tiny), so the switch
# only ever touches ngrok. Nothing here touches race_cv or the API.

set -u
cd "$(dirname "$0")"

URL=${NGROK_URL:-https://bonanza-overbite-sprawl.ngrok-free.dev}
POLICY=${NGROK_POLICY:-}
if [[ -z "$POLICY" ]]; then
    # The committed policy carries the placeholder password; the gitignored
    # .local copy carries the real one (RACE_DAY_RUNBOOK.md).
    if [[ -f config/ngrok-policy.local.yml ]]; then POLICY=config/ngrok-policy.local.yml
    else POLICY=config/ngrok-policy.yml; fi
fi
PORT_LEADERBOARD=${PORT:-8001}
PORT_PRE_RACE=8002
PORT_THANK_YOU=8003
RUN_DIR=.broadcast
mkdir -p "$RUN_DIR"

PY=${RACE_CV_PYTHON:-}
if [[ -z "$PY" ]]; then
    if [[ -x .venv/bin/python ]]; then PY=.venv/bin/python; else PY=python3; fi
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

port_open() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/"; }

start_page() {  # $1 page name, $2 port, rest: extra args for holding_page.py
    local page=$1 port=$2; shift 2
    if port_open "$port"; then
        echo -e "   $page page already up on port $port"
        return 0
    fi
    nohup "$PY" scripts/holding_page.py "$page" --port "$port" "$@" > "$RUN_DIR/$page.log" 2>&1 &
    echo $! > "$RUN_DIR/$page.pid"
    for _ in $(seq 1 20); do port_open "$port" && break; sleep 0.25; done
    if port_open "$port"; then
        echo -e "${GREEN}✅ $page page serving on port $port${NC}"
    else
        echo -e "${RED}❌ $page page did not come up; see $RUN_DIR/$page.log${NC}"
        return 1
    fi
}

stop_pages() {
    for page in pre-race thank-you; do
        if [[ -f "$RUN_DIR/$page.pid" ]]; then
            kill "$(cat "$RUN_DIR/$page.pid")" 2>/dev/null && echo "   stopped $page page"
            rm -f "$RUN_DIR/$page.pid"
        fi
    done
}

current_tunnel() {  # prints "public_url -> addr" for the running agent, or nothing
    curl -s --max-time 1 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | "$PY" -c "import json,sys
try:
    ts = json.load(sys.stdin).get('tunnels', [])
except Exception:
    ts = []
for t in ts:
    print(t.get('public_url'), '->', t.get('config', {}).get('addr'))" 2>/dev/null
}

point_tunnel() {  # $1 local port
    local port=$1
    if ! command -v ngrok >/dev/null 2>&1; then
        echo -e "${RED}❌ ngrok is not installed on this machine (https://ngrok.com/download)${NC}"
        return 1
    fi
    if [[ ! -f "$POLICY" ]]; then
        echo -e "${RED}❌ traffic policy not found: $POLICY${NC}"
        return 1
    fi
    if grep -q "change-me" "$POLICY"; then
        echo -e "${YELLOW}⚠️  $POLICY still has the placeholder password; /admin is protected by 'change-me'.${NC}"
        echo -e "${YELLOW}   cp config/ngrok-policy.yml config/ngrok-policy.local.yml and set a real one.${NC}"
    fi
    local now
    now=$(current_tunnel)
    if [[ "$now" == *"-> http://localhost:$port"* ]]; then
        echo -e "${GREEN}✅ tunnel already points at port $port${NC}"
        return 0
    fi
    pkill -x ngrok 2>/dev/null && sleep 1
    nohup ngrok http "$port" --url "$URL" --traffic-policy-file "$POLICY" --log stdout \
        > "$RUN_DIR/ngrok.log" 2>&1 &
    echo $! > "$RUN_DIR/ngrok.pid"
    for _ in $(seq 1 40); do
        now=$(current_tunnel)
        [[ "$now" == *"-> http://localhost:$port"* ]] && break
        sleep 0.5
    done
    if [[ "$now" == *"-> http://localhost:$port"* ]]; then
        echo -e "${GREEN}✅ $now${NC}"
    else
        echo -e "${RED}❌ ngrok did not report a tunnel to port $port; see $RUN_DIR/ngrok.log${NC}"
        tail -5 "$RUN_DIR/ngrok.log"
        return 1
    fi
}

status() {
    echo -e "${BLUE}Public URL: $URL${NC}"
    local now
    now=$(current_tunnel)
    if [[ -z "$now" ]]; then
        echo "   ngrok: not running"
    else
        case "$now" in
            *":$PORT_LEADERBOARD"*) echo "   ngrok: $now   (LIVE LEADERBOARD)";;
            *":$PORT_PRE_RACE"*)    echo "   ngrok: $now   (pre-race holding page)";;
            *":$PORT_THANK_YOU"*)   echo "   ngrok: $now   (thank-you page)";;
            *)                      echo "   ngrok: $now";;
        esac
    fi
    port_open "$PORT_LEADERBOARD" && echo "   leaderboard API: up on $PORT_LEADERBOARD" || echo "   leaderboard API: not answering on $PORT_LEADERBOARD"
    port_open "$PORT_PRE_RACE" && echo "   pre-race page:   up on $PORT_PRE_RACE" || echo "   pre-race page:   not running"
    port_open "$PORT_THANK_YOU" && echo "   thank-you page:  up on $PORT_THANK_YOU" || echo "   thank-you page:  not running"
}

cmd=${1:-status}; shift || true
case "$cmd" in
    pre-race)
        start_page pre-race "$PORT_PRE_RACE" "$@" && point_tunnel "$PORT_PRE_RACE" ;;
    thank-you)
        start_page thank-you "$PORT_THANK_YOU" "$@" && point_tunnel "$PORT_THANK_YOU" ;;
    leaderboard)
        if ! port_open "$PORT_LEADERBOARD"; then
            echo -e "${YELLOW}⚠️  nothing is answering on port $PORT_LEADERBOARD -- start the stack first (./start-race-cv.sh).${NC}"
            echo -e "${YELLOW}   Pointing the tunnel there now would show spectators an error page.${NC}"
            exit 1
        fi
        point_tunnel "$PORT_LEADERBOARD" ;;
    status) status ;;
    stop)
        if [[ -f "$RUN_DIR/ngrok.pid" ]] || pgrep -x ngrok >/dev/null; then pkill -x ngrok && echo "   stopped ngrok"; fi
        rm -f "$RUN_DIR/ngrok.pid"
        stop_pages ;;
    *)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
