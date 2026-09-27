#!/bin/bash
# Runs ON the broadcast VM (Debian 12, e2-micro). Idempotent: re-run to
# update the page, the sponsors, or the agent. Called by deploy.sh with the
# bundle already in ~/bundle: holding_page.py, static_pages/, sponsors.txt
# (optional), ngrok.yml (the account's agent config, with the authtoken).
#
#   bash ~/bundle/setup-vm.sh <public-url> <page> <time-text>
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8   # quiet apt on a headless ssh
URL="${1:?public ngrok URL}"
PAGE="${2:-pre-race}"
TIME_TEXT="${3:-October 24}"
APP=/opt/race-broadcast
PORT=8002

# 0. Tools the rest assumes (present on debian-cloud images; a no-op then).
sudo apt-get install -y -qq --no-install-recommends curl ca-certificates python3 >/dev/null 2>&1 || true
command -v python3 >/dev/null && command -v curl >/dev/null || { echo "python3/curl missing"; exit 1; }

# 1. The ngrok agent, from ngrok's apt repository, with the tarball as the
#    fallback if the repository is unreachable.
if ! command -v ngrok >/dev/null 2>&1; then
    if curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
        && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" | sudo tee /etc/apt/sources.list.d/ngrok.list >/dev/null \
        && sudo apt-get update -qq && sudo apt-get install -y -qq ngrok >/dev/null; then
        :
    else
        curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | sudo tar -xz -C /usr/local/bin ngrok
    fi
fi
NGROK=$(command -v ngrok)
echo "ngrok: $($NGROK version)"

# 2. The page bundle, laid out like the repo so holding_page.py's own
#    REPO_ROOT-relative paths (src/static_pages, config/sponsors.txt) hold.
sudo mkdir -p "$APP/scripts" "$APP/src/static_pages" "$APP/config"
sudo cp ~/bundle/holding_page.py "$APP/scripts/holding_page.py"
sudo cp ~/bundle/static_pages/*.html "$APP/src/static_pages/"
# The pages reload themselves every 60 s so pavilion tabs follow a switch
# on race day. Served from here for a month that is a problem: ngrok's free
# plan caps HTTP requests at 20,000 a month, and one tab left open on a 60 s
# refresh makes 43,000. Fifteen minutes is plenty between races.
sudo sed -i -E 's/(http-equiv="refresh" content=")60"/\1900"/' "$APP"/src/static_pages/*.html
grep -c 'content="900"' "$APP/src/static_pages/pre-race.html" | sed 's/^/refresh set to 900 s in pre-race.html: /'
if [[ -f ~/bundle/sponsors.txt ]]; then
    sudo cp ~/bundle/sponsors.txt "$APP/config/sponsors.txt"
    echo "sponsors: $(wc -l < ~/bundle/sponsors.txt) names"
else
    sudo rm -f "$APP/config/sponsors.txt"
    echo "sponsors: none (config/sponsors.txt absent on the Mac; the page omits the section)"
fi
sudo chmod -R a+rX "$APP"

# 3. The agent's config (authtoken), readable only by the ngrok user.
sudo useradd -r -s /usr/sbin/nologin -d /nonexistent ngrok 2>/dev/null || true
sudo install -d -m 750 -o root -g ngrok /etc/ngrok
sudo install -m 640 -o root -g ngrok ~/bundle/ngrok.yml /etc/ngrok/ngrok.yml
shred -u ~/bundle/ngrok.yml 2>/dev/null || rm -f ~/bundle/ngrok.yml

# 4. Two services. The page server binds loopback only: nothing on this VM
#    is reachable from the internet except through the tunnel.
sudo tee /etc/systemd/system/race-holding-page.service >/dev/null <<UNIT
[Unit]
Description=Race holding page ($PAGE) on 127.0.0.1:$PORT
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/python3 $APP/scripts/holding_page.py $PAGE --host 127.0.0.1 --port $PORT --time "$TIME_TEXT"
Restart=always
RestartSec=3
DynamicUser=yes

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/race-ngrok.service >/dev/null <<UNIT
[Unit]
Description=ngrok agent: $URL -> the holding page
Wants=network-online.target race-holding-page.service
After=network-online.target race-holding-page.service

[Service]
ExecStart=$NGROK http $PORT --url $URL --config /etc/ngrok/ngrok.yml --log stdout
Restart=always
RestartSec=10
User=ngrok
Group=ngrok

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now race-holding-page.service
sudo systemctl restart race-holding-page.service
# The agent is enabled so the page comes back after a reboot -- but only
# when the VM is meant to hold the URL. broadcast.sh cloud-off disables it
# for race day so a reboot cannot snatch the URL from the Mac.
sudo systemctl enable --now race-ngrok.service
sudo systemctl restart race-ngrok.service
sleep 6
echo "page:  $(systemctl is-active race-holding-page.service)  local HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/)"
echo "agent: $(systemctl is-active race-ngrok.service)"
sudo journalctl -u race-ngrok.service -n 4 --no-pager -o cat | cut -c1-160
