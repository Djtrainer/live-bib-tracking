#!/bin/bash
# The public URL between races, from Google's always-free e2-micro instead of
# a laptop that has to stay on. The VM runs the holding page and the ngrok
# agent (deploy/broadcast-vm/setup-vm.sh); broadcast.sh cloud-on / cloud-off
# and the race-day commands hand the URL between the VM and the Mac.
#
#   deploy/broadcast-vm/deploy.sh create    create the VM (free tier: e2-micro,
#                                           us-central1, 10 GB standard disk) and deploy
#   deploy/broadcast-vm/deploy.sh sync      push the page, sponsors.txt and agent
#                                           config again; restart the services
#   deploy/broadcast-vm/deploy.sh status    VM state, services, what the URL returns
#   deploy/broadcast-vm/deploy.sh ssh       a shell on the VM
#   deploy/broadcast-vm/deploy.sh delete    delete the VM (asks first)
#
# Costs: the e2-micro, its 10 GB standard disk and ~1 GB/month of egress are in
# the free tier (one such VM per billing account, in us-west1/us-central1/
# us-east1). Its external IPv4 address is not: $0.005/h, about $3.72/month.
# Nothing else is created.
set -euo pipefail
cd "$(dirname "$0")/../.."

VM=${BROADCAST_VM:-race-broadcast}
ZONE=${BROADCAST_ZONE:-us-central1-a}
URL=${NGROK_URL:-https://slaysarcoma.ngrok.app}
PAGE=${BROADCAST_PAGE:-pre-race}
TIME_TEXT=${BROADCAST_TIME:-on October 24th}
NGROK_CFG=${NGROK_CONFIG:-"$HOME/Library/Application Support/ngrok/ngrok.yml"}
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

vm_ssh() { gcloud compute ssh "$VM" --zone="$ZONE" --quiet --command "$1"; }

create_vm() {
    if gcloud compute instances describe "$VM" --zone="$ZONE" >/dev/null 2>&1; then
        echo -e "   VM $VM already exists in $ZONE"
        return 0
    fi
    echo -e "${YELLOW}☁️  Creating $VM (e2-micro, $ZONE, Debian 12, 10 GB pd-standard)...${NC}"
    gcloud compute instances create "$VM" \
        --zone="$ZONE" \
        --machine-type=e2-micro \
        --image-family=debian-12 --image-project=debian-cloud \
        --boot-disk-size=10GB --boot-disk-type=pd-standard \
        --shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring \
        --no-service-account --no-scopes \
        --format="table(name,zone.basename(),machineType.basename(),status,networkInterfaces[0].accessConfigs[0].natIP)"
}

wait_for_ssh() {
    echo -e "${YELLOW}⏳ Waiting for SSH...${NC}"
    for _ in $(seq 1 40); do
        vm_ssh "true" >/dev/null 2>&1 && return 0
        sleep 5
    done
    echo -e "${RED}❌ SSH to $VM never came up${NC}"; return 1
}

sync_vm() {
    if [[ ! -f "$NGROK_CFG" ]]; then
        echo -e "${RED}❌ ngrok config not found at $NGROK_CFG (the agent's authtoken lives there)${NC}"; return 1
    fi
    if ! grep -q authtoken "$NGROK_CFG"; then
        echo -e "${RED}❌ $NGROK_CFG has no authtoken; run: ngrok config add-authtoken <token>${NC}"; return 1
    fi
    local bundle
    bundle=$(mktemp -d)
    # The bundle holds a copy of the authtoken until it is on the VM; remove
    # it whatever happens.
    trap 'rm -rf "$bundle"' EXIT
    mkdir -p "$bundle/static_pages"
    cp scripts/holding_page.py "$bundle/"
    cp src/static_pages/*.html "$bundle/static_pages/"
    [[ -f config/sponsors.txt ]] && cp config/sponsors.txt "$bundle/"
    cp "$NGROK_CFG" "$bundle/ngrok.yml"
    cp deploy/broadcast-vm/setup-vm.sh "$bundle/"
    echo -e "${YELLOW}📦 Pushing the page, sponsors and agent config to $VM...${NC}"
    vm_ssh "rm -rf ~/bundle"
    gcloud compute scp --recurse --zone="$ZONE" --quiet "$bundle" "$VM:~/bundle" >/dev/null
    rm -rf "$bundle"; trap - EXIT
    echo -e "${YELLOW}🔧 Setting up services on the VM...${NC}"
    local cmd
    printf -v cmd 'bash ~/bundle/setup-vm.sh %q %q %q' "$URL" "$PAGE" "$TIME_TEXT"
    vm_ssh "$cmd"
}

check_url() {
    # The free plan's interstitial answers browsers first; the header skips it.
    # Right after a restart the agent needs a few seconds to re-register
    # the URL; ngrok answers 404 ERR_NGROK_3200 until then.
    local code
    for _ in 1 2 3 4 5 6; do
        code=$(curl -s -o /dev/null -m 10 -H 'ngrok-skip-browser-warning: 1' -w '%{http_code}' "$URL/" || true)
        [[ "$code" == "200" ]] && break
        sleep 4
    done
    local text
    text=$(curl -s -m 10 -H 'ngrok-skip-browser-warning: 1' "$URL/" | sed -E 's/<[^>]+>/ /g' | tr -s ' \n' ' ' | grep -oE "Check back [^.]{0,28}|Thank you for coming|ERR_NGROK_[0-9]+" | head -1 || true)
    echo -e "   $URL/  ->  HTTP ${code:-none}  ${text:+\"$text\"}"
}

status_vm() {
    local state
    state=$(gcloud compute instances describe "$VM" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "ABSENT")
    echo -e "${BLUE}VM $VM ($ZONE): $state${NC}"
    if [[ "$state" == "RUNNING" ]]; then
        vm_ssh "echo \"   page:  \$(systemctl is-active race-holding-page.service)   agent: \$(systemctl is-active race-ngrok.service) (\$(systemctl is-enabled race-ngrok.service 2>/dev/null))\"; sudo journalctl -u race-ngrok.service -n 2 --no-pager -o cat | cut -c1-140 | sed 's/^/   ngrok: /'" 2>/dev/null || echo "   (ssh failed)"
    fi
    check_url
}

case "${1:-help}" in
    create)  create_vm && wait_for_ssh && sync_vm && echo && status_vm ;;
    sync)    wait_for_ssh && sync_vm && echo && status_vm ;;
    status)  status_vm ;;
    ssh)     gcloud compute ssh "$VM" --zone="$ZONE" ;;
    delete)  gcloud compute instances delete "$VM" --zone="$ZONE" ;;
    *)       sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
