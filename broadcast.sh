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
#   ./broadcast.sh pre-race [--time "around 8:30 am"]   before the stack is up
#   ./broadcast.sh leaderboard                   when results should be public
#   ./broadcast.sh thank-you                     if something goes wrong
#   ./broadcast.sh status                        what the URL points at now
#   ./broadcast.sh offline                       take the URL down (= Ctrl-C in the ngrok window);
#                                                the leaderboard and page servers keep running locally
#   ./broadcast.sh stop                          offline, plus stop the page servers
#   ./broadcast.sh cloud-on                      hand the URL to the free VM (deploy/broadcast-vm):
#                                                the pre-race page, with this laptop off
#   ./broadcast.sh cloud-off                     take it back from the VM (pre-race/leaderboard/
#                                                thank-you do this by themselves first)
#
# The page servers keep running once started (they are tiny), so the switch
# only ever touches ngrok. Nothing here touches race_cv or the API.
#
# One agent can hold the URL at a time. Between races the VM's agent holds
# it; on race day every local mode releases the VM's agent (and disables it,
# so a VM reboot cannot snatch the URL back) before claiming the URL here.

set -u
cd "$(dirname "$0")"

URL=${NGROK_URL:-https://slaysarcoma.ngrok.app}
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
VM=${BROADCAST_VM:-race-broadcast}
VM_ZONE=${BROADCAST_ZONE:-us-central1-a}

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

# --- the free VM (deploy/broadcast-vm) --------------------------------------
vm_state() {  # RUNNING / TERMINATED / ABSENT / UNREACHABLE / NOGCLOUD
    command -v gcloud >/dev/null 2>&1 || { echo NOGCLOUD; return; }
    local out
    if out=$(gcloud compute instances describe "$VM" --zone="$VM_ZONE" --format="value(status)" 2>&1); then
        echo "$out"
    elif [[ "$out" == *"was not found"* ]]; then
        echo ABSENT
    else
        # Not logged in, no internet yet, wrong project: NOT the same as "no VM".
        echo "$out" >> "$RUN_DIR/gcloud.log"
        echo UNREACHABLE
    fi
}
vm_ssh() { gcloud compute ssh "$VM" --zone="$VM_ZONE" --quiet --ssh-flag=-oConnectTimeout=10 --command "$1" 2>>"$RUN_DIR/gcloud.log"; }
vm_agent() {  # active / activating / inactive / failed / unknown
    local out
    out=$(vm_ssh "systemctl is-active race-ngrok.service" || true)
    echo "${out:-unknown}"
}
RELEASED_MARK="$RUN_DIR/vm-released"
cloud_release() {  # the VM lets go of the URL, and will not take it back on a reboot
    # Once released, later switches (leaderboard, an emergency thank-you) must
    # not pay two gcloud round trips each or hang on a dead hotspot: the marker
    # says the VM is already out of the picture. cloud-on removes it.
    [[ -f "$RELEASED_MARK" ]] && return 0
    if [[ "${BROADCAST_SKIP_VM:-0}" == "1" ]]; then
        echo -e "${YELLOW}⚠️  BROADCAST_SKIP_VM=1: not checking the VM; if it still holds $URL this agent will fail with ERR_NGROK_334${NC}"
        return 0
    fi
    local state
    state=$(vm_state)
    case "$state" in
        ABSENT|NOGCLOUD) return 0 ;;
        UNREACHABLE)
            echo -e "${RED}❌ cannot check the VM ($VM): gcloud failed (not logged in? no internet yet?). See $RUN_DIR/gcloud.log.${NC}"
            echo -e "${BLUE}   Fix gcloud (gcloud auth login), or if you KNOW the VM is not holding the URL: BROADCAST_SKIP_VM=1 $0 ...${NC}"
            return 1 ;;
        RUNNING) ;;
        *) echo -e "   VM $VM is $state; nothing to release"; touch "$RELEASED_MARK"; return 0 ;;
    esac
    local was
    was=$(vm_agent)
    echo -e "${YELLOW}☁️  releasing $URL from the VM ($VM; agent was $was)...${NC}"
    # Unconditional: "activating" (retry loop) and "inactive but enabled" both
    # come back after a reboot; disable --now on a stopped unit is a no-op.
    if ! vm_ssh "sudo systemctl disable --now race-ngrok.service"; then
        echo -e "${RED}❌ could not stop the VM's agent; the URL may still be held there. See $RUN_DIR/gcloud.log.${NC}"
        return 1
    fi
    touch "$RELEASED_MARK"
    sleep 2
    echo -e "${GREEN}✅ VM agent stopped and disabled${NC}"
}
cloud_claim() {  # the VM takes the URL; local ngrok goes offline first
    local state
    state=$(vm_state)
    if [[ "$state" != "RUNNING" ]]; then
        echo -e "${RED}❌ VM $VM is $state; create it with deploy/broadcast-vm/deploy.sh create${NC}"
        return 1
    fi
    if pgrep -x ngrok >/dev/null; then
        pkill -x ngrok && echo "   local ngrok stopped"
        rm -f "$RUN_DIR/ngrok.pid"
        sleep 1
    fi
    echo -e "${YELLOW}☁️  handing $URL to the VM ($VM)...${NC}"
    vm_ssh "sudo systemctl enable --now race-ngrok.service && sudo systemctl restart race-ngrok.service" || { echo -e "${RED}❌ could not start the VM's agent; see $RUN_DIR/gcloud.log${NC}"; return 1; }
    rm -f "$RELEASED_MARK"
    local body=""
    for _ in $(seq 1 12); do
        sleep 2
        body=$(curl -s -m 8 -H 'ngrok-skip-browser-warning: 1' "$URL/" || true)
        [[ "$body" == *"Check back "* ]] && break
    done
    if [[ "$body" == *"Check back "* ]]; then
        echo -e "${GREEN}✅ $URL shows the holding page from the VM; this laptop can go to sleep${NC}"
    else
        echo -e "${RED}❌ $URL is not showing the holding page yet; check: deploy/broadcast-vm/deploy.sh status${NC}"
        return 1
    fi
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
    cloud_release || return 1
    pkill -x ngrok 2>/dev/null && sleep 1
    nohup ngrok http "$port" --url "$URL" --traffic-policy-file "$POLICY" --log stdout \
        > "$RUN_DIR/ngrok.log" 2>&1 &
    echo $! > "$RUN_DIR/ngrok.pid"
    for _ in $(seq 1 40); do
        now=$(current_tunnel)
        [[ "$now" == *"-> http://localhost:$port"* ]] && break
        kill -0 "$(cat "$RUN_DIR/ngrok.pid")" 2>/dev/null || break   # it died; do not wait 20 s
        sleep 0.5
    done
    if [[ "$now" == *"-> http://localhost:$port"* ]]; then
        echo -e "${GREEN}✅ $now${NC}"
    elif grep -q "ERR_NGROK_334" "$RUN_DIR/ngrok.log" 2>/dev/null; then
        echo -e "${RED}❌ another agent already holds $URL (the VM?). Release it: ./broadcast.sh cloud-off${NC}"
        echo -e "${BLUE}   or: deploy/broadcast-vm/deploy.sh ssh  ->  sudo systemctl disable --now race-ngrok.service${NC}"
        return 1
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
    local state agent
    state=$(vm_state)
    case "$state" in
        NOGCLOUD) ;;
        ABSENT)   echo "   free VM:         none (deploy/broadcast-vm/deploy.sh create)";;
        RUNNING)  agent=$(vm_agent)
                  echo "   free VM:         $VM running, agent $agent$( [[ "$agent" == active ]] && echo '   (VM HOLDS THE URL)')$( [[ -f "$RELEASED_MARK" ]] && echo '   [released for this race; cloud-on hands it back]')";;
        *)        echo "   free VM:         $VM $state";;
    esac
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
    cloud-on)  cloud_claim ;;
    cloud-off) cloud_release ;;
    offline|down)
        # Exactly what Ctrl-C in the ngrok terminal did: the public URL stops
        # resolving to anything of ours; nothing local is touched.
        if pgrep -x ngrok >/dev/null; then
            pkill -x ngrok && echo -e "${GREEN}✅ ngrok stopped; $URL is offline${NC}"
        else
            echo "   ngrok was not running; $URL is already offline"
        fi
        rm -f "$RUN_DIR/ngrok.pid"
        echo "   leaderboard, pre-race and thank-you servers are untouched (./broadcast.sh status)"
        [[ -f "$RELEASED_MARK" ]] && echo "   packing up? ./broadcast.sh cloud-on hands $URL back to the VM" ;;
    stop)
        if pgrep -x ngrok >/dev/null; then pkill -x ngrok && echo "   stopped ngrok"; fi
        rm -f "$RUN_DIR/ngrok.pid"
        stop_pages
        [[ -f "$RELEASED_MARK" ]] && echo "   packing up? ./broadcast.sh cloud-on hands $URL back to the VM" ;;
    *)
        # The usage is the header comment, up to its first blank line.
        awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$0"; exit 1 ;;
esac
