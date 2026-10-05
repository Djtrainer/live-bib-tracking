"""The operator login.

Before this, "admin" was a page: the password was compared in the browser
and /admin/dashboard opened without it; the API asked nobody anything.
These tests pin what the server now refuses, what it deliberately leaves
open (the detector at its own address, the public reads), and the three
states of the config: absent (open, loudly), usable, and broken (the API
must not start, and must not print the password while saying why).
"""

import asyncio
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api_backend import admin_auth as auth_module  # noqa: E402
from api_backend import local_server as server  # noqa: E402
from api_backend.admin_auth import AdminAuth, AdminConfigError, COOKIE_NAME  # noqa: E402
from api_backend.journal import RaceJournal, _format_finish  # noqa: E402

USER, PASSWORD = "timer", "test-only-not-a-real-password"
CONFIG = f'username: "{USER}"\npassword: "{PASSWORD}"\n'
JETSON = ("192.168.55.1", 50001)        # the detector, over the USB link
HOTSPOT = ("172.20.10.9", 50002)        # a phone on the hotspot


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "journal", RaceJournal(tmp_path))
    server.race_results.clear()
    server.race_clock_state.clear()
    server.race_clock_state.update({"raceStartTime": None, "status": "stopped", "offset": 0})
    server.app_state.pop("fresh", None)
    server._refused_detector_hosts.clear()
    for name in ("RACE_ADMIN_USER", "RACE_ADMIN_PASSWORD", "RACE_DETECTOR_ADDRESSES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RACE_ADMIN_CONFIG", str(tmp_path / "admin.yml"))
    yield tmp_path


def write_config(tmp_path, text=CONFIG):
    (tmp_path / "admin.yml").write_text(text)


def load(tmp_path):
    return AdminAuth.from_environment({"RACE_ADMIN_CONFIG": str(tmp_path / "admin.yml")})


@pytest.fixture
def client(isolated_state):
    """The tablet: an ordinary address, signs in to act."""
    write_config(isolated_state)
    with TestClient(server.app) as c:
        yield c


def at(address):
    return TestClient(server.app, client=address)


def login(c, username=USER, password=PASSWORD):
    return c.post("/api/admin/login", json={"username": username, "password": password})


def detector_finish(event_id="e1", bib="120"):
    now = time.time()
    return {"eventId": event_id, "bibNumber": bib, "captureTime": now, "wallClockTime": now, "source": "race_cv"}


def jpeg() -> bytes:
    return cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()


OPERATOR_CALLS = [
    ("post", "/api/clock/start", None),
    ("post", "/api/clock/stop", None),
    ("post", "/api/clock/edit", {"elapsedMs": 0}),
    ("post", "/api/clock/reset", None),
    ("put", "/api/results/abc", {"bibNumber": "1"}),
    ("delete", "/api/results/abc", None),
    ("post", "/api/reorder", {"order": []}),
    ("post", "/api/results", {"bibNumber": "7", "finishTime": 1000}),     # a manual add
]


def call(c, method, path, body):
    return getattr(c, method)(path, json=body) if body is not None else getattr(c, method)(path)


class TestTokens:
    def test_round_trip_and_tamper(self):
        auth = AdminAuth(USER, PASSWORD)
        token = auth.issue()
        assert auth.verify(token)
        version, login_at, expiry, sig = token.split(".")
        assert not auth.verify(f"{version}.{login_at}.{int(expiry) + 60}.{sig}")     # expiry extended
        assert not auth.verify(f"{version}.{int(login_at) + 60}.{expiry}.{sig}")     # sign-in moved
        assert not auth.verify(f"{version}.{login_at}.{expiry}.{'0' * len(sig)}")
        assert not auth.verify(f"v1.{expiry}.{sig}")                                 # the old format
        for junk in ("", None, "v2.x.y.z", token + " ", " " + token, token + ".x", token.upper()):
            assert not auth.verify(junk)

    def test_strange_input_is_a_refusal_not_a_crash(self):
        auth = AdminAuth(USER, PASSWORD)
        assert not auth.verify("v2.1.9999999999.éé")
        assert not auth.verify("v2.١.9999999999." + "0" * 64)              # a non-ASCII digit
        assert not auth.check_login("é", "\udce9")                          # lone surrogate
        assert not auth.check_login(None, ["x"])

    def test_expiry(self):
        now = [1000.0]
        auth = AdminAuth(USER, PASSWORD, ttl_s=60, clock=lambda: now[0])
        token = auth.issue()
        now[0] = 1059.0
        assert auth.verify(token)
        now[0] = 1061.0
        assert not auth.verify(token)

    def test_sessions_slide_past_half_life(self):
        now = [1000.0]
        auth = AdminAuth(USER, PASSWORD, ttl_s=100, clock=lambda: now[0])
        token = auth.issue()
        assert not auth.needs_refresh(token)
        now[0] = 1051.0
        assert auth.needs_refresh(token)                                       # under half its life left
        renewed = auth.issue(renew=token)
        assert renewed.split(".")[1] == token.split(".")[1]                    # same sign-in time
        assert int(renewed.split(".")[2]) == 1151
        now[0] = 1101.0
        assert not auth.needs_refresh(token) and not auth.verify(token)        # expired: a login, not a refresh
        assert auth.issue(renew=token) is None                                 # an expired token renews nothing
        assert auth.issue(renew="v2.1.2." + "0" * 64) is None                  # nor does a forgery
        assert auth.ends_in(token) is None

    def test_renewal_stops_at_the_cap_after_sign_in(self, monkeypatch):
        """A copied cookie cannot be kept alive forever."""
        monkeypatch.setattr(auth_module, "MAX_SESSION_S", 250)
        now = [1000.0]
        auth = AdminAuth(USER, PASSWORD, ttl_s=100, clock=lambda: now[0])
        token = auth.issue()
        for t in (1060.0, 1130.0, 1200.0):
            now[0] = t
            assert auth.verify(token)
            token = auth.issue(renew=token)
        assert token.split(".")[1:3] == ["1000", "1250"]                       # capped at sign-in + 250
        now[0] = 1240.0
        assert auth.verify(token) and not auth.needs_refresh(token)
        now[0] = 1251.0
        assert not auth.verify(token)

    def test_session_survives_a_restart_but_not_a_password_change(self):
        token = AdminAuth(USER, PASSWORD).issue()
        assert AdminAuth(USER, PASSWORD).verify(token)            # a new process, same credentials
        assert not AdminAuth(USER, PASSWORD + "x").verify(token)
        assert not AdminAuth(USER + "x", PASSWORD).verify(token)

    def test_nothing_configured_verifies_nothing(self):
        assert not AdminAuth().enabled
        assert not AdminAuth().verify(AdminAuth(USER, PASSWORD).issue())
        assert not AdminAuth().check_login("", "")

    def test_username_is_not_case_sensitive_but_the_password_is(self):
        """A tablet keyboard capitalises the first letter of a text field."""
        auth = AdminAuth(USER, PASSWORD)
        assert auth.check_login(USER.capitalize(), PASSWORD)
        assert auth.check_login(f"  {USER.upper()} ", PASSWORD)
        assert not auth.check_login(USER, PASSWORD.upper())
        assert not auth.check_login(USER, PASSWORD + " ")

    def test_a_huge_login_field_is_refused_before_it_is_normalised(self, monkeypatch):
        """NFKC is quadratic on combining marks in the wrong order: one 250 KB wrong
        login held the event loop -- the clock, the detector's posts, a correct
        login -- for 8 s, and the body limit is 8 MB."""
        auth = AdminAuth(USER, PASSWORD)
        normalised = []
        monkeypatch.setattr(auth_module, "_normal", lambda text: normalised.append(len(text)) or text)
        marks = "a" + "\u0301" * 60_000 + "\u0316" * 60_000
        assert not auth.check_login(USER, marks)
        assert not auth.check_login(marks, PASSWORD)
        assert not auth.check_login(USER, "\ufdfa" * 100_000)
        assert not auth.check_login([USER], PASSWORD) and not auth.check_login(USER, {"a": 1})
        assert normalised == []
        # Nothing a person types is affected.
        assert auth.check_login(f" {USER.upper()} ", PASSWORD)
        assert not auth.check_login(USER, PASSWORD + "x" * 900)
        assert normalised

    def test_composed_and_decomposed_accents_are_the_same_password(self, tmp_path):
        write_config(tmp_path, 'username: "timer"\npassword: "café-au-lait-26"\n')
        assert load(tmp_path).check_login("timer", "café-au-lait-26")


class TestConfigLoading:
    def test_absent_file_is_open(self, tmp_path):
        auth = AdminAuth.from_environment({"RACE_ADMIN_CONFIG": str(tmp_path / "nope.yml")})
        assert not auth.enabled and "absent" in auth.source

    def test_file_and_environment(self, tmp_path):
        write_config(tmp_path)
        auth = load(tmp_path)
        assert auth.enabled and auth.check_login(USER, PASSWORD) and not auth.check_login(USER, "wrong")
        assert auth.password_length == len(PASSWORD)
        env = AdminAuth.from_environment({"RACE_ADMIN_USER": "a", "RACE_ADMIN_PASSWORD": "b-long-enough"})
        assert env.enabled and env.source == "environment"

    @pytest.mark.parametrize("text", [
        'username: "admin"\npassword: "change-me"\n',   # the example file, copied and not edited
        'username: "admin"\npassword: "admin"\n',        # what the old page shipped publicly
        'username: "timer"\npassword: "Timer"\n',        # the username again
        'username: "timer"\npassword: "short"\n',        # under 8 characters
        'username: "timer"\npassword: "race-day-2026 "\n',     # a space the operator cannot see
        'username: "timer"\npassword: "race\\tday-2026"\n',    # a real TAB after YAML unescapes it
        'username: "admin"\n',                             # no password
        'password: "something-real"\n',                    # no username
        "just a string\n",
        "username: [unclosed\n",
    ])
    def test_a_broken_weak_or_placeholder_config_refuses_to_load(self, tmp_path, text):
        write_config(tmp_path, text)
        with pytest.raises(AdminConfigError):
            load(tmp_path)

    @pytest.mark.parametrize("text", [
        'username: "admin"\npassword: 01234567\n',       # YAML: an octal-looking int
        'username: "admin"\npassword: 12345678\n',       # an int
        'username: "admin"\npassword: yes\n',            # True
        'username: "admin"\npassword: 10:24\n',          # sexagesimal: 624
        'username: 007\npassword: "long-enough-pw"\n',   # the username retyped to 7
    ])
    def test_values_yaml_would_retype_are_refused_with_the_reason(self, tmp_path, text):
        """Otherwise the API starts, says the login is on, and refuses the
        password exactly as it is written in the file."""
        write_config(tmp_path, text)
        with pytest.raises(AdminConfigError, match="quotes"):
            load(tmp_path)

    @pytest.mark.parametrize("text", [
        'username: "admin"\npassword: "sekret-hunter2-value\n',     # an unclosed quote
        'username: "admin"\npassword: *sekret-hunter2-value\n',     # YAML reads * as an alias
        'username: "admin"\npassword: !sekret-hunter2-value x\n',   # ...and ! as a tag
        'username: "admin"\n\tpassword: "sekret-hunter2-value"\n',  # a tab
    ])
    def test_a_malformed_file_never_puts_the_password_in_the_error(self, tmp_path, text):
        """PyYAML's message quotes the offending line -- the password line --
        and this message is logged and shown by the launcher."""
        write_config(tmp_path, text)
        with pytest.raises(AdminConfigError) as caught:
            load(tmp_path)
        assert "sekret" not in str(caught.value) and "hunter2" not in str(caught.value)
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        assert "not valid YAML" in str(caught.value)

    def test_the_environment_gets_the_same_rules(self):
        for env in ({"RACE_ADMIN_USER": "a"},
                    {"RACE_ADMIN_USER": "admin", "RACE_ADMIN_PASSWORD": "admin"},
                    {"RACE_ADMIN_USER": "admin", "RACE_ADMIN_PASSWORD": "change-me"},
                    {"RACE_ADMIN_USER": "admin", "RACE_ADMIN_PASSWORD": " padded-password "}):
            with pytest.raises(AdminConfigError):
                AdminAuth.from_environment(env)

    def test_the_example_file_is_refused_as_shipped(self):
        example = Path(__file__).resolve().parents[1] / "config" / "admin.example.yml"
        with pytest.raises(AdminConfigError, match="placeholder"):
            AdminAuth.from_environment({"RACE_ADMIN_CONFIG": str(example)})

    def test_the_api_does_not_start_on_a_broken_config(self, isolated_state):
        write_config(isolated_state, 'username: "admin"\npassword: "change-me"\n')
        with pytest.raises(AdminConfigError):
            with TestClient(server.app):
                pass

    def test_detector_addresses(self, tmp_path):
        auth = AdminAuth(USER, PASSWORD)
        for host in ("127.0.0.1", "::1", "192.168.55.1", "192.168.55.100", "::ffff:192.168.55.1"):
            assert auth.is_detector_address(host), host
        for host in ("172.20.10.9", "10.0.0.5", "192.168.56.1", "testclient", "", None):
            assert not auth.is_detector_address(host), host
        write_config(tmp_path, CONFIG + 'detector_addresses: ["172.20.10.9"]\n')
        assert load(tmp_path).is_detector_address("172.20.10.9")
        env = AdminAuth.from_environment({"RACE_ADMIN_CONFIG": str(tmp_path / "admin.yml"),
                                          "RACE_DETECTOR_ADDRESSES": "10.1.2.0/24, 10.9.9.9"})
        assert env.is_detector_address("10.1.2.77") and env.is_detector_address("172.20.10.9")
        write_config(tmp_path, CONFIG + 'detector_addresses: ["not-an-address"]\n')
        with pytest.raises(AdminConfigError, match="detector_addresses"):
            load(tmp_path)


class TestDetectorAddressEntries:
    def test_an_address_with_a_prefix_length_is_that_one_machine(self, tmp_path):
        """`ip addr` prints 172.20.10.7/28. Read loosely that was the whole
        hotspot, and every phone on it could post finishes with no login."""
        write_config(tmp_path, CONFIG + 'detector_addresses: ["172.20.10.7/28"]\n')
        auth = load(tmp_path)
        assert auth.is_detector_address("172.20.10.7")
        assert not auth.is_detector_address("172.20.10.9")
        assert [str(n) for n in auth.extra_detector_networks] == ["172.20.10.7/32"]

    def test_a_network_is_still_accepted_and_is_visible_as_one(self, tmp_path):
        write_config(tmp_path, CONFIG + 'detector_addresses: ["172.20.10.0/28", "10.0.0.5"]\n')
        auth = load(tmp_path)
        assert auth.is_detector_address("172.20.10.9")                          # never refused: no lockout
        assert [n.num_addresses for n in auth.extra_detector_networks] == [16, 1]

    def test_a_misspelt_key_is_reported_not_silently_ignored(self, tmp_path):
        write_config(tmp_path, CONFIG + 'detector_adresses: ["172.20.10.7"]\n')
        auth = load(tmp_path)
        assert auth.unknown_keys == ["detector_adresses"]
        assert not auth.is_detector_address("172.20.10.7")

    def test_the_environment_list_adds_to_the_file(self, tmp_path):
        write_config(tmp_path, CONFIG + 'detector_addresses: "172.20.10.7"\n')
        auth = AdminAuth.from_environment({
            "RACE_ADMIN_CONFIG": str(tmp_path / "admin.yml"), "RACE_DETECTOR_ADDRESSES": "10.0.0.5, 10.0.0.6",
        })
        assert all(auth.is_detector_address(a) for a in ("172.20.10.7", "10.0.0.5", "10.0.0.6"))

    def test_startup_says_which_addresses_are_exempt(self, isolated_state, caplog):
        write_config(isolated_state, CONFIG + 'detector_addresses: ["172.20.10.0/28"]\nspeling: 1\n')
        with caplog.at_level("INFO"), TestClient(server.app):
            pass
        text = " ".join(record.getMessage() for record in caplog.records)
        assert "Detector posts accepted without a login from" in text and "172.20.10.0/28 covers many addresses" in text
        assert "unknown key 'speling'" in text


class TestOperatorEndpointsNeedTheLogin:
    def test_session_reports_configured_and_not_authenticated(self, client):
        assert client.get("/api/admin/session").json() == {"success": True, "configured": True, "authenticated": False}

    @pytest.mark.parametrize("method,path,body", OPERATOR_CALLS)
    def test_refused_without_a_session_and_reaches_the_handler_with_one(self, client, method, path, body):
        r = call(client, method, path, body)
        assert r.status_code == 401, (path, r.text)
        assert login(client).status_code == 200
        r = call(client, method, path, body)
        assert r.status_code not in (401, 403) and r.status_code < 500, (path, r.status_code, r.text)

    def test_the_detectors_address_is_not_an_operator(self, client):
        """Being at the Jetson's address opens the detector's two doors only."""
        jetson = at(JETSON)
        for method, path, body in OPERATOR_CALLS:
            assert call(jetson, method, path, body).status_code == 401, path

    def test_roster_upload_needs_the_session(self, client):
        files = {"file": ("r.csv", b"bibNumber,racerName\n1,A\n", "text/csv")}
        assert client.post("/api/roster/upload", files=files).status_code == 401
        login(client)
        r = client.post("/api/roster/upload", files=files)
        assert r.status_code == 200 and r.json()["success"] is True

    def test_wrong_password_is_401_and_sets_no_cookie(self, client):
        r = login(client, password="nope")
        assert r.status_code == 401 and COOKIE_NAME not in r.cookies
        assert login(client, username="nobody").status_code == 401
        assert client.post("/api/admin/login", json=["not", "a", "dict"]).status_code in (401, 422)

    def test_login_then_the_clock_starts(self, client):
        r = login(client, username=USER.capitalize())            # as a tablet keyboard types it
        assert r.status_code == 200 and r.json()["configured"] is True
        cookie = r.headers["set-cookie"]
        assert COOKIE_NAME in cookie and "HttpOnly" in cookie and "samesite=strict" in cookie.lower()
        assert "secure" not in cookie.lower()                    # plain http: the tablet on the hotspot
        assert client.get("/api/admin/session").json()["authenticated"] is True
        assert client.post("/api/clock/start").json()["success"] is True
        assert client.post("/api/results", json={"bibNumber": "7", "finishTime": 1000}).status_code == 200

    def test_cookie_is_secure_behind_the_https_tunnel(self, client):
        r = client.post("/api/admin/login", json={"username": USER, "password": PASSWORD},
                        headers={"X-Forwarded-Proto": "https"})
        assert "secure" in r.headers["set-cookie"].lower()

    def test_logout_drops_the_cookie_in_that_browser(self, client):
        login(client)
        assert client.post("/api/clock/start").status_code == 200
        client.post("/api/admin/logout")                         # the Set-Cookie deletion empties the jar
        assert COOKIE_NAME not in client.cookies
        assert client.post("/api/clock/stop").status_code == 401

    def test_a_copied_cookie_outlives_logout_until_the_password_changes(self, client, monkeypatch):
        """Sessions are signed, not stored: Sign Out cannot recall a copy.
        Documented in the runbook; revoking means changing the password."""
        token = login(client).cookies[COOKIE_NAME]
        client.post("/api/admin/logout")
        thief = at(HOTSPOT)
        thief.cookies.set(COOKIE_NAME, token)
        assert thief.post("/api/clock/start").status_code == 200
        monkeypatch.setattr(server, "admin_auth", AdminAuth(USER, PASSWORD + "-changed"))
        assert thief.post("/api/clock/stop").status_code == 401

    def test_a_forged_or_stale_cookie_is_refused(self, client):
        client.cookies.set(COOKIE_NAME, AdminAuth(USER, "some-other-password").issue())
        assert client.post("/api/clock/start").status_code == 401

    def test_the_session_check_renews_a_valid_session_and_nothing_else(self, client):
        assert "set-cookie" not in client.get("/api/admin/session").headers     # nothing to renew
        client.cookies.set(COOKIE_NAME, "v2.1.9999999999." + "0" * 64)
        assert "set-cookie" not in client.get("/api/admin/session").headers     # a forgery earns nothing
        client.cookies.clear()
        first = login(client).cookies[COOKIE_NAME]
        r = client.get("/api/admin/session")
        assert COOKIE_NAME in r.headers.get("set-cookie", "")
        assert r.cookies[COOKIE_NAME].split(".")[1] == first.split(".")[1]      # same sign-in time

    def test_a_renewal_that_lapsed_in_between_sets_no_cookie(self, client, monkeypatch):
        """verify() and issue() read the clock separately. A token that lapsed
        between them came back as a new sign-in: 72 fresh hours, no password."""
        login(client)
        monkeypatch.setattr(server.admin_auth, "issue", lambda renew=None: None)
        r = client.get("/api/admin/session")
        assert r.json()["authenticated"] is True and "set-cookie" not in r.headers

    def test_the_session_says_when_it_ends_and_ends_then(self, client, monkeypatch):
        """72 h after the sign-in, whatever was renewed in between."""
        login(client)
        start = time.time()
        assert abs(client.get("/api/admin/session").json()["endsIn"] - auth_module.MAX_SESSION_S) < 5
        for hours in (10, 20, 30, 40, 50, 60, 70):                               # a dashboard left open
            monkeypatch.setattr(server.admin_auth, "_clock", lambda h=hours: start + h * 3600)
            body = client.get("/api/admin/session").json()
            assert body["authenticated"] is True, hours
            assert abs(body["endsIn"] - (72 - hours) * 3600) < 5
        monkeypatch.setattr(server.admin_auth, "_clock", lambda: start + 72 * 3600 + 5)
        body = client.get("/api/admin/session").json()
        assert body["authenticated"] is False and "endsIn" not in body
        assert client.post("/api/clock/start").status_code == 401

    def test_an_operator_action_renews_a_session_past_half_life(self, client, monkeypatch):
        login(client)
        assert "set-cookie" not in client.post("/api/clock/start").headers      # fresh: left alone
        ten_hours_on = time.time() + 10 * 3600                                   # of an 18 h session
        monkeypatch.setattr(server.admin_auth, "_clock", lambda: ten_hours_on)
        r = client.post("/api/clock/stop")
        assert r.status_code == 200 and COOKIE_NAME in r.headers.get("set-cookie", "")


class TestTheDetectorIsKnownByItsAddress:
    @pytest.fixture
    def running(self, client):
        login(client)
        assert client.post("/api/clock/start").json()["success"] is True
        client.post("/api/admin/logout")
        return client

    def test_the_jetson_posts_finishes_and_frames_without_a_login(self, running):
        jetson = at(JETSON)
        r = jetson.post("/api/results", json=detector_finish())
        assert r.status_code == 200 and r.json()["success"] is True
        replay = dict(detector_finish("e2"), source="race_cv_replay")           # scripts/replay_events.py
        assert jetson.post("/api/results", json=replay).json()["success"] is True
        r = jetson.post("/api/frame", content=jpeg(), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 200 and r.json()["success"] is True
        assert at(("127.0.0.1", 50003)).post("/api/results", json=detector_finish("e3")).json()["success"] is True

    def test_the_same_payload_from_the_hotspot_needs_the_login(self, running):
        """It used to take first place with a negative time under a roster name."""
        phone = at(HOTSPOT)
        assert phone.post("/api/results", json=detector_finish("forged")).status_code == 401
        assert phone.post("/api/frame", content=jpeg(), headers={"Content-Type": "image/jpeg"}).status_code == 401
        assert running.get("/api/results").json()["data"] == []

    def test_through_the_tunnel_nothing_is_the_detector(self, running):
        """The ngrok agent connects from localhost; X-Forwarded-For is how a
        tunnel visitor is told apart from the detector on this machine."""
        local = at(("127.0.0.1", 50004))
        tunnelled = {"X-Forwarded-For": "203.0.113.9"}
        assert local.post("/api/results", json=detector_finish("t1"), headers=tunnelled).status_code == 401
        assert local.post("/api/frame", content=jpeg(), headers={**tunnelled, "Content-Type": "image/jpeg"}).status_code == 401

    @pytest.mark.parametrize("header", [
        {"X-Forwarded-For": ""},                   # present and empty is still a proxy
        {"X-Forwarded-Host": "slaysarcoma.ngrok.app"},
        {"X-Forwarded-Proto": "https"},
        {"Forwarded": "for=203.0.113.9"},
        {"X-Real-IP": "203.0.113.9"},
        {"Via": "1.1 proxy"},
    ])
    def test_any_sign_of_a_proxy_ends_the_exemption(self, running, header):
        for address in (("127.0.0.1", 50005), JETSON):
            caller = at(address)
            assert caller.post("/api/results", json=detector_finish("p1"), headers=header).status_code == 401
            assert caller.post("/api/frame", content=jpeg(), headers={**header, "Content-Type": "image/jpeg"}).status_code == 401
        assert running.get("/api/results").json()["data"] == []

    @pytest.mark.parametrize("header", [{"Origin": "https://example.org"}, {"Sec-Fetch-Site": "cross-site"}])
    def test_a_browser_on_this_machine_is_not_the_detector(self, running, header):
        """Any web page open on the Mac can post to 127.0.0.1. Browsers say
        so; the detector's client never does."""
        local = at(("127.0.0.1", 50006))
        assert local.post("/api/results", json=detector_finish("b1"), headers=header).status_code == 401
        assert local.post("/api/results", json=detector_finish("b2")).json()["success"] is True

    def test_a_usb_link_address_counts_only_on_the_usb_link(self, running):
        """192.168.55.1 as a source means the Jetson only on a connection
        that arrived at this machine's own 192.168.55.x address."""
        def jetson_arriving_at(local):
            return TestClient(server.app, base_url=f"http://{local}:8001", client=JETSON)
        assert jetson_arriving_at("192.168.55.100").post("/api/results", json=detector_finish("u1")).json()["success"] is True
        assert jetson_arriving_at("172.20.10.5").post("/api/results", json=detector_finish("u2")).status_code == 401
        assert jetson_arriving_at("172.20.10.5").post(
            "/api/frame", content=jpeg(), headers={"Content-Type": "image/jpeg"}).status_code == 401
        # Unknown local address (a name, a unix socket): as before, accepted.
        assert at(JETSON).post("/api/results", json=detector_finish("u3")).json()["success"] is True
        # This machine is not subject to it.
        local = TestClient(server.app, base_url="http://172.20.10.5:8001", client=("127.0.0.1", 50007))
        assert local.post("/api/results", json=detector_finish("u4")).json()["success"] is True

    def test_a_listed_usb_address_is_taken_at_the_operators_word(self, isolated_state):
        write_config(isolated_state, CONFIG + f'detector_addresses: ["{JETSON[0]}"]\n')
        with TestClient(server.app) as tablet:
            login(tablet)
            tablet.post("/api/clock/start")
            elsewhere = TestClient(server.app, base_url="http://172.20.10.5:8001", client=JETSON)
            assert elsewhere.post("/api/results", json=detector_finish("l1")).json()["success"] is True

    @pytest.mark.parametrize("kind", ["frame", "finish"])
    def test_a_refused_detector_is_named_in_the_log_once(self, running, caplog, kind):
        with caplog.at_level("WARNING"):
            for _ in range(3):
                if kind == "frame":
                    at(HOTSPOT).post("/api/frame", content=jpeg(), headers={"Content-Type": "image/jpeg"})
                else:
                    at(HOTSPOT).post("/api/results", json=detector_finish("r1"))
        named = [r.getMessage() for r in caplog.records if HOTSPOT[0] in r.getMessage() and "detector_addresses" in r.getMessage()]
        assert len(named) == 1

    def test_a_configured_hotspot_address_is_a_detector(self, isolated_state):
        write_config(isolated_state, CONFIG + f'detector_addresses: ["{HOTSPOT[0]}"]\n')
        with TestClient(server.app) as tablet:
            login(tablet)
            tablet.post("/api/clock/start")
            assert at(HOTSPOT).post("/api/results", json=detector_finish("h1")).json()["success"] is True

    def test_only_a_detector_shaped_finish_is_exempt_even_from_the_jetson(self, running):
        jetson = at(JETSON)
        no_id = detector_finish(); del no_id["eventId"]
        assert jetson.post("/api/results", json=no_id).status_code == 401
        assert jetson.post("/api/results", json={"bibNumber": "9", "finishTime": 1, "source": "race_cv"}).status_code == 401
        assert jetson.post("/api/results", json=dict(detector_finish("e9"), finishTime=1)).status_code == 401
        for junk in ([1, 2, 3], "text"):
            assert jetson.post("/api/results", json=junk).status_code in (401, 422)

    def test_public_reads_need_nothing(self, client):
        for path in ("/api/results", "/api/clock/status", "/api/time", "/api/admin/session"):
            assert at(HOTSPOT).get(path).status_code == 200, path


class TestBounds:
    @pytest.fixture
    def jetson(self, client):
        login(client)
        client.post("/api/clock/start")
        return at(JETSON)

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", '"soon"', "null"])
    def test_a_time_that_is_not_a_finite_number_is_never_stored(self, client, jetson, bad):
        """One NaN row froze race_results.txt and made its own DELETE answer 500."""
        body = ('{"eventId":"n1","source":"race_cv","bibNumber":"1","captureTime":1,"wallClockTime":%s}' % bad).encode()
        r = jetson.post("/api/results", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 200 and r.json()["success"] is False
        assert client.get("/api/results").json()["data"] == []

    def test_the_results_file_survives_an_unformattable_time(self):
        for bad in (float("nan"), float("inf"), "soon", None, [1]):
            assert _format_finish(bad) == "--:--.-"
        assert _format_finish(61500) == "01:01.5"

    def test_an_oversized_body_is_refused_before_it_is_read(self, jetson):
        huge = b"\xff\xd8" + b"0" * (server.MAX_BODY_BYTES + 1)
        assert jetson.post("/api/frame", content=huge, headers={"Content-Type": "image/jpeg"}).status_code == 413
        assert jetson.post("/api/frame", content=jpeg(), headers={"Content-Type": "image/jpeg"}).status_code == 200

    def test_long_text_fields_are_cut(self, client, jetson):
        jetson.post("/api/results", json=dict(detector_finish("long"), bibNumber="9" * 500))
        rows = client.get("/api/results").json()["data"]
        assert len(rows) == 1 and len(rows[0]["bibNumber"]) <= server.RESULT_FIELD_LIMITS["bibNumber"]

    def test_a_body_that_declares_no_length_is_counted_as_it_arrives(self, client, jetson):
        """Chunked transfer has no Content-Length; it went straight past the first limit."""
        def chunks(total):
            yield b"\xff\xd8"
            for _ in range(total // 65536 + 1):
                yield b"0" * 65536
        big = jetson.post("/api/frame", content=chunks(server.MAX_BODY_BYTES), headers={"Content-Type": "image/jpeg"})
        assert big.status_code == 413
        assert "content-length" not in {k.lower() for k in big.request.headers}        # it really was chunked
        ok = jetson.post("/api/frame", content=chunks(65536), headers={"Content-Type": "image/jpeg"})
        assert ok.status_code == 200 and ok.json()["success"] is True
        def json_chunks():
            yield b'{"eventId":"c1","source":"race_cv","bibNumber":"1","captureTime":1,"wallClockTime":1,"racerName":"'
            for _ in range(server.MAX_BODY_BYTES // 65536 + 2):
                yield b"a" * 65536
            yield b'"}'
        r = jetson.post("/api/results", content=json_chunks(), headers={"Content-Type": "application/json"})
        assert r.status_code == 413
        assert client.get("/api/results").json()["data"] == []

    def test_every_text_field_is_cut_whatever_its_type(self):
        data = {"racerName": ["x" * 500], "age": "9" * 40, "team": {"a": "b" * 500}, "bibNumber": 12, "gender": None}
        server._bound_result_fields(data)
        assert len(data["racerName"]) == server.RESULT_FIELD_LIMITS["racerName"]
        assert data["age"] == "9" * 8 and len(data["team"]) == server.RESULT_FIELD_LIMITS["team"]
        assert data["bibNumber"] == 12 and data["gender"] is None                # scalars are left alone
        edit = {"racerName": "n" * 500, "junk": "j" * 10_000_000, "finishTime": "01:00.0"}
        server._bound_result_fields(edit, server.RESULT_EDITABLE)
        assert set(edit) == {"racerName", "finishTime"} and len(edit["racerName"]) == 120

    def test_an_edit_is_cut_too(self, client, jetson):
        jetson.post("/api/results", json=detector_finish("edit-me"))
        row = client.get("/api/results").json()["data"][0]
        r = client.put(f"/api/results/{row['id']}", json={"racerName": "n" * 5000, "age": "4" * 100, "surprise": "s" * 5000})
        assert r.status_code == 200 and r.json()["success"] is True
        row = client.get("/api/results").json()["data"][0]
        assert len(row["racerName"]) == 120 and len(str(row.get("age", ""))) <= 8 and "surprise" not in row

    @pytest.mark.parametrize("value, expected", [
        (1500, 1500.0), ("1500", 1500.0), (0, 0.0), (-3.5, -3.5),
        (float("nan"), None), (float("inf"), None), ("NaN", None), ("1e999", None), (True, None),
        (None, None), ([1], None), ({"a": 1}, None), ("soon", None), (10 ** 400, None),
    ])
    def test_finite_ms(self, value, expected):
        assert server._finite_ms(value) == expected

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-1", "null", "[1]", "true", "1e999"])
    def test_the_clock_is_never_set_to_something_that_is_not_a_time(self, client, jetson, bad):
        """NaN became the clock's offset; every finish after it was refused, across restarts."""
        before = dict(server.race_clock_state)
        r = client.post("/api/clock/edit", content=('{"time": %s}' % bad).encode(), headers={"Content-Type": "application/json"})
        assert r.status_code == 200 and r.json()["success"] is False
        assert server.race_clock_state == before
        assert jetson.post("/api/results", json=detector_finish("after-" + bad)).json()["success"] is True

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "-5", "[1]", "true", '{"a":1}'])
    def test_an_edit_never_stores_a_time_that_is_not_one(self, client, jetson, bad):
        jetson.post("/api/results", json=detector_finish("t-edit"))
        row = client.get("/api/results").json()["data"][0]
        r = client.put(f"/api/results/{row['id']}", content=('{"finishTime": %s}' % bad).encode(),
                       headers={"Content-Type": "application/json"})
        assert r.status_code == 200 and r.json()["success"] is False
        assert client.get("/api/results").json()["data"][0]["finishTime"] == row["finishTime"]
        assert client.put(f"/api/results/{row['id']}", json={"finishTime": 61500}).json()["success"] is True
        assert client.get("/api/results").json()["data"][0]["finishTime"] == 61500

    def test_null_is_a_deliberate_clear_not_a_mistyped_time(self, client, jetson):
        """null is the one not-a-time an edit accepts: it means "has not finished"
        (an emptied time cell), and the row leaves /api/results. The server cannot
        tell that from a mistyped time, because JSON has no NaN: the dashboard
        must never send null for one (handleCellBlur checks Number.isFinite)."""
        jetson.post("/api/results", json=detector_finish("t-clear"))
        row = client.get("/api/results").json()["data"][0]
        r = client.put(f"/api/results/{row['id']}", json={"finishTime": None})
        assert r.status_code == 200 and r.json()["success"] is True and r.json()["data"]["finishTime"] is None
        assert client.get("/api/results").json()["data"] == []
        # Not lost: the same row takes a time again.
        assert client.put(f"/api/results/{row['id']}", json={"finishTime": 61500}).json()["success"] is True
        assert client.get("/api/results").json()["data"][0]["finishTime"] == 61500

    @staticmethod
    def drive_limit(pieces, respond_first=False):
        """BodyLimitMiddleware(max 10 bytes) around an app that reads the whole
        body, fed one ASGI message per piece. Returns the status it answered."""
        sent = []

        async def inner(scope, receive, send):
            if respond_first:
                await send({"type": "http.response.start", "status": 200, "headers": []})
            try:
                more = True
                while more:
                    message = await receive()
                    more = message.get("more_body", False)
            except server.StarletteHTTPException as refused:
                await send({"type": "http.response.start", "status": refused.status_code, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
            if not respond_first:
                await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        queue = [{"type": "http.request", "body": piece, "more_body": i < len(pieces) - 1}
                 for i, piece in enumerate(pieces)]

        async def receive():
            return queue.pop(0)

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/x", "headers": []}
        asyncio.run(server.BodyLimitMiddleware(inner, max_bytes=10)(scope, receive, send))
        return [m["status"] for m in sent if m["type"] == "http.response.start"]

    def test_the_body_count_runs_across_messages(self):
        """A real server delivers a body in many pieces; no single one is over the limit."""
        assert self.drive_limit([b"123456", b"123456", b"123456"]) == [413]
        assert self.drive_limit([b"12345", b"12345"]) == [200]

    def test_nothing_is_refused_once_the_response_has_started(self):
        """/video_feed answers first and then only listens for a disconnect: a
        413 can no longer be sent, and raising then was a traceback per request."""
        assert self.drive_limit([b"123456", b"123456", b"123456"], respond_first=True) == [200]

    def test_a_number_with_thousands_of_digits_is_cut_like_text(self):
        data = {"bibNumber": int("9" * 4000), "age": 10 ** 50, "racerName": 7}
        server._bound_result_fields(data)
        assert data["bibNumber"] == "9" * server.RESULT_FIELD_LIMITS["bibNumber"]
        assert data["age"] == ("1" + "0" * 50)[:8] and data["racerName"] == 7
        small = {"bibNumber": 120, "age": 34}
        assert server._bound_result_fields(small) == {"bibNumber": 120, "age": 34}

    def test_there_is_a_ceiling_on_rows(self, client, jetson, monkeypatch):
        monkeypatch.setattr(server, "MAX_RESULTS", 2)
        assert jetson.post("/api/results", json=detector_finish("a")).json()["success"] is True
        assert jetson.post("/api/results", json=detector_finish("b")).json()["success"] is True
        r = jetson.post("/api/results", json=detector_finish("c")).json()
        assert r["success"] is False and "limit" in r["message"]
        assert len(client.get("/api/results").json()["data"]) == 2


class TestGuessingIsSlow:
    """Failed logins wait their turn, four at a time, however many are sent."""

    @pytest.fixture
    def gates(self, client, monkeypatch):
        # Fresh semaphores: an asyncio one binds to the first loop it waits in.
        monkeypatch.setattr(server, "_failed_login_gate", asyncio.Semaphore(4))
        monkeypatch.setattr(server, "_failed_login_gate_tunnel", asyncio.Semaphore(4))
        return client

    @staticmethod
    def browser():
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://testserver")

    @staticmethod
    def guess(c, headers=None):
        return c.post("/api/admin/login", json={"username": USER, "password": "a-wrong-guess"}, headers=headers or {})

    def test_a_failed_login_pauses_and_a_good_one_does_not(self, gates, monkeypatch):
        pauses = []
        async def pause():
            pauses.append(1)
        monkeypatch.setattr(server, "_login_failure_pause", pause)
        assert login(gates, password="a-wrong-guess").status_code == 401 and len(pauses) == 1
        assert login(gates).status_code == 200 and len(pauses) == 1
        assert server.FAILED_LOGIN_DELAY_S >= 0.5

    def test_no_more_than_four_guesses_are_answered_at_once(self, gates, monkeypatch):
        active, peak = 0, 0
        async def pause():
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
        monkeypatch.setattr(server, "_login_failure_pause", pause)
        async def flood():
            async with self.browser() as c:
                return await asyncio.gather(*[self.guess(c) for _ in range(12)])
        answers = asyncio.run(flood())
        assert [r.status_code for r in answers] == [401] * 12
        assert peak == 4

    def test_a_flood_through_the_tunnel_does_not_delay_the_tablet(self, gates, monkeypatch):
        async def pause():
            await asyncio.sleep(0)
        monkeypatch.setattr(server, "_login_failure_pause", pause)
        tunnelled = {"X-Forwarded-For": "203.0.113.9"}
        async def scenario():
            for _ in range(4):                               # the tunnel's queue is full
                await server._failed_login_gate_tunnel.acquire()
            async with self.browser() as c:
                direct = await asyncio.wait_for(self.guess(c), 2)            # the tablet's typo is answered
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(self.guess(c, tunnelled), 0.3)    # the tunnel's guess waits
                good = await asyncio.wait_for(
                    c.post("/api/admin/login", json={"username": USER, "password": PASSWORD}, headers=tunnelled), 2)
            return direct, good
        direct, good = asyncio.run(scenario())
        assert direct.status_code == 401
        assert good.status_code == 200                       # a correct password never queues


class TestNoLoginConfigured:
    def test_everything_works_as_before_and_the_session_says_so(self, isolated_state):
        with TestClient(server.app) as c:                        # no admin.yml written
            assert c.get("/api/admin/session").json() == {"success": True, "configured": False, "authenticated": True}
            assert c.post("/api/admin/login", json={"username": "x", "password": "y"}).json() == {"success": True, "configured": False}
            assert c.post("/api/clock/start").json()["success"] is True
            assert at(HOTSPOT).post("/api/results", json=detector_finish()).json()["success"] is True
