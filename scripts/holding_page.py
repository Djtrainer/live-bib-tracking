#!/usr/bin/env python3
"""Serve one holding page on a port, for every path.

The public URL is one ngrok tunnel. Before the race it should show "check
back at 8:30", and if the leaderboard breaks it should show a thank-you
page rather than an error -- without touching the leaderboard's process on
8001. So each page gets its own port and its own tiny server, and
``broadcast.sh`` points the tunnel at whichever one should be public.

Every GET/HEAD path returns the page: spectators arrive on ``/``, on
``/admin``, on whatever they bookmarked, and all of them should see it.
Placeholders are filled per request, so ``config/sponsors.txt`` can be
edited while the page is live.

    python scripts/holding_page.py pre-race --port 8002 --time "8:30 am"
    python scripts/holding_page.py thank-you --port 8003
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = REPO_ROOT / "src" / "static_pages"
SPONSORS_FILE = REPO_ROOT / "config" / "sponsors.txt"
PAGES = {"pre-race": "pre-race.html", "thank-you": "thank-you.html"}
DEFAULT_PORTS = {"pre-race": 8002, "thank-you": 8003}


def sponsors_section(path: Path = SPONSORS_FILE) -> str:
    """A "thank you to our sponsors" block from one name per line, or nothing.

    Nothing rather than placeholder names: a page that says "Sponsor A"
    to the pavilion TV is worse than one that thanks sponsors in general.
    """
    if not path.exists():
        return ""
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [n for n in names if n and not n.startswith("#")]
    if not names:
        return ""
    items = "".join(f"<li>{html.escape(n)}</li>" for n in names)
    return f'<section class="sponsors"><h2>Thank you to our sponsors</h2><ul>{items}</ul></section>'


def render(page: str, time_text: str = "8:30 am", year: int | None = None,
           sponsors_path: Path = SPONSORS_FILE) -> str:
    template = (PAGES_DIR / PAGES[page]).read_text(encoding="utf-8")
    return (
        template.replace("{{TIME}}", html.escape(time_text))
        .replace("{{YEAR}}", str(year or _dt.date.today().year))
        .replace("{{SPONSORS}}", sponsors_section(sponsors_path))
    )


def make_handler(page: str, time_text: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "holding-page/1"

        def _send(self, with_body: bool) -> None:
            body = render(page, time_text).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if with_body:
                self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server naming)
            self._send(with_body=True)

        def do_HEAD(self) -> None:  # noqa: N802
            self._send(with_body=False)

        def log_message(self, fmt, *args) -> None:
            logging.getLogger("holding_page").info("%s " + fmt, self.address_string(), *args)

    return Handler


def serve(page: str, port: int, time_text: str, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(page, time_text))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Serve a holding page on every path")
    parser.add_argument("page", choices=sorted(PAGES))
    parser.add_argument("--port", type=int, default=None, help="default: 8002 pre-race, 8003 thank-you")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--time", default="8:30 am", help="fills {{TIME}} on the pre-race page")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stderr)
    port = args.port or DEFAULT_PORTS[args.page]
    render(args.page, args.time)  # fail now, not on the first visitor, if the template is broken
    httpd = serve(args.page, port, args.time, args.host)
    logging.getLogger("holding_page").info("serving %s on http://%s:%d/ (every path)", args.page, args.host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
