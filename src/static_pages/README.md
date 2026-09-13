# Holding pages

Two self-contained pages (inline CSS, the brand mark inlined) that stand in
for the leaderboard on the public URL:

* `pre-race.html` -- "The race hasn't started yet, check back around
  {{TIME}}". Refreshes itself every minute so open tabs follow when the
  tunnel is switched to the live leaderboard.
* `thank-you.html` -- "Thank you for coming" with an optional sponsor list,
  for when the leaderboard is broken and nobody should be looking at an
  error. Refreshes every two minutes.

`scripts/holding_page.py` serves one of them on a port and answers *every*
path with it (so `/admin` bookmarks show it too). `broadcast.sh` at the
repo root is the race-day switch: it runs the page servers and points the
ngrok tunnel at the leaderboard (8001), the pre-race page (8002) or the
thank-you page (8003).

Placeholders filled by the server: `{{TIME}}` (`--time`), `{{YEAR}}`
(this year), `{{SPONSORS}}` (a "Thank you to our sponsors" section built
from `config/sponsors.txt`, one name per line; omitted entirely if that
file is missing or empty, so no placeholder text can reach the public).
