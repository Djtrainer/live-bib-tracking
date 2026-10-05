"""The operator login.

Before this, "admin" was a page: the password was compared in the browser
and /admin/dashboard opened without it; the API asked nobody anything.
These tests pin what the server now refuses, what it deliberately leaves
open (the detector at its own address, the public reads), and the three
states of the config: absent (open, loudly), usable, and broken (the API
must not start, and must not print the password while saying why).
"""

import sys
import time
from pathlib import Path

import cv2
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
        assert auth.issue(renew=token).split(".")[1] == "1101"                 # an expired token renews nothing

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

    def test_there_is_a_ceiling_on_rows(self, client, jetson, monkeypatch):
        monkeypatch.setattr(server, "MAX_RESULTS", 2)
        assert jetson.post("/api/results", json=detector_finish("a")).json()["success"] is True
        assert jetson.post("/api/results", json=detector_finish("b")).json()["success"] is True
        r = jetson.post("/api/results", json=detector_finish("c")).json()
        assert r["success"] is False and "limit" in r["message"]
        assert len(client.get("/api/results").json()["data"]) == 2


class TestNoLoginConfigured:
    def test_everything_works_as_before_and_the_session_says_so(self, isolated_state):
        with TestClient(server.app) as c:                        # no admin.yml written
            assert c.get("/api/admin/session").json() == {"success": True, "configured": False, "authenticated": True}
            assert c.post("/api/admin/login", json={"username": "x", "password": "y"}).json() == {"success": True, "configured": False}
            assert c.post("/api/clock/start").json()["success"] is True
            assert at(HOTSPOT).post("/api/results", json=detector_finish()).json()["success"] is True
