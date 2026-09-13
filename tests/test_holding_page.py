"""The holding pages answer every path, fill their placeholders, and never leak one."""

import sys
import threading
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import holding_page  # noqa: E402


@pytest.fixture
def server():
    httpd = holding_page.serve("pre-race", 0, "8:30 am", host="127.0.0.1")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


class TestEveryPath:
    @pytest.mark.parametrize("path", ["/", "/admin", "/results/2025", "/anything?x=1"])
    def test_any_path_gets_the_page(self, server, path):
        with urllib.request.urlopen(server + path) as r:
            body = r.read().decode()
            assert r.status == 200
            assert "text/html" in r.headers["Content-Type"]
        assert "hasn't started yet" in body
        assert "Check back around 8:30 am" in body

    def test_head_has_headers_and_no_body(self, server):
        req = urllib.request.Request(server + "/", method="HEAD")
        with urllib.request.urlopen(req) as r:
            assert r.status == 200
            assert int(r.headers["Content-Length"]) > 1000
            assert r.read() == b""


class TestRendering:
    def test_no_placeholder_survives(self, tmp_path):
        for page in holding_page.PAGES:
            out = holding_page.render(page, "9:15 am", year=2026, sponsors_path=tmp_path / "none.txt")
            assert "{{" not in out and "}}" not in out
            assert "2026 Slay Sarcoma Race" in out

    def test_time_is_escaped(self, tmp_path):
        out = holding_page.render("pre-race", "<b>8:30</b>", sponsors_path=tmp_path / "none.txt")
        assert "<b>8:30</b>" not in out and "&lt;b&gt;8:30&lt;/b&gt;" in out

    def test_sponsors_come_from_the_file_or_not_at_all(self, tmp_path):
        missing = tmp_path / "sponsors.txt"
        assert "sponsors" not in holding_page.render("thank-you", sponsors_path=missing).lower().split("thank you for coming")[1][:200]
        missing.write_text("# comment\n\nAcme Running Co\nRiver <City> Bank\n")
        out = holding_page.render("thank-you", sponsors_path=missing)
        assert "Thank you to our sponsors" in out
        assert "<li>Acme Running Co</li>" in out
        assert "River &lt;City&gt; Bank" in out and "<City>" not in out

    def test_pages_are_self_contained(self):
        for name in holding_page.PAGES.values():
            text = (holding_page.PAGES_DIR / name).read_text()
            assert "http://" not in text and "https://" not in text  # no external assets to fail through the tunnel
            assert 'src="data:image/png;base64,' in text
