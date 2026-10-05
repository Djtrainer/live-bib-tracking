"""Operator login for the results API.

Until this existed the "admin login" was a page: the username and password
were compared in the browser against values shipped in the JavaScript (and
committed to a public repository), and `/admin/dashboard` opened without
them. Nothing on the server asked who was calling, so anyone who could
reach port 8001 could reset the race clock.

Here the credentials live on the machine running the API -- a gitignored
`config/admin.local.yml`, or `RACE_ADMIN_USER` / `RACE_ADMIN_PASSWORD` --
and never reach the browser. A successful login gets a signed session
cookie; the operator endpoints (clock, edits, deletes, reorder, roster
upload, manual adds) refuse requests without it.

The choices that are deliberate:

* **The session is a signed value, not server memory.** The signing key is
  derived from the credentials, so an API restart mid-race does not log
  the tablet out, and changing the password invalidates every session.
* **Sessions slide, up to a cap.** A valid session is renewed whenever it
  is used, so one opened the evening before does not end at the gun; but
  72 hours after the sign-in it ends regardless, so a copied cookie cannot
  be kept alive forever.
* **A cookie, not an Authorization header.** Through the ngrok tunnel the
  browser already uses `Authorization` for the tunnel's own basic-auth
  password; a second value there would break that outer door.
* **No config means open, loudly; a broken config means no start.** A
  race must not be lost to a missing file, so with nothing configured the
  API runs as it always did and says so at startup and on the login page.
  A file that exists but is unreadable, incomplete, weak or still the
  placeholder stops the API with the reason: someone meant to set a
  password, and silently running open would be the worst outcome.
* **The detector is recognised by where it is, not by what it says.** The
  detector has no credentials. Its finishes and preview frames are
  accepted without a login only from an address a detector can be at:
  this machine, or the USB link to the Jetson (192.168.55.0/24), plus any
  `detector_addresses` in the config -- and never through the tunnel. A
  phone on the hotspot that imitates the detector's payload gets a 401.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Callable, Mapping

import yaml

COOKIE_NAME = "race_admin"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "admin.local.yml"
DEFAULT_TTL_S = 18 * 3600          # one race day, start to pack-up; renewed on use
MAX_SESSION_S = 72 * 3600          # ...but never past this long after the sign-in
# Longest username or password check_login() will look at, in characters.
# Normalising is not linear in the worst case: 250 KB of combining marks in
# the wrong order held the event loop for 8 s, and the body limit is 8 MB.
LOGIN_FIELD_MAX = 1024
# "admin" is on the list because admin/admin is what the old client-side
# login shipped in its JavaScript and in a public repository.
PLACEHOLDER_PASSWORDS = {"change-me", "changeme", "password", "admin", "letmein", ""}
MIN_PASSWORD_LENGTH = 8
KEY_ITERATIONS = 200_000
# Where a detector can post from without a login: this machine (one-machine
# mode) and the L4T USB device-mode link to the Jetson.
DEFAULT_DETECTOR_ADDRESSES = ("127.0.0.0/8", "::1/128", "192.168.55.0/24")
_TOKEN = re.compile(r"v2\.([0-9]{1,12})\.([0-9]{1,12})\.([0-9a-f]{64})")


class AdminConfigError(RuntimeError):
    """The admin config exists but cannot be used. The API must not start."""


def _normal(text: str) -> str:
    """One Unicode form on both sides, so an accented letter typed on a
    tablet matches the same letter saved by an editor."""
    return unicodedata.normalize("NFKC", text)


def _validated(username: object, password: object, where: str) -> tuple[str, str]:
    """One set of rules for the file and the environment.

    Each refusal is a way the operator gets locked out, or the lock is left
    open, while the launcher prints a green line:

    * YAML reads an unquoted ``0123`` as 83, ``yes`` as True, ``10:24`` as
      624. The API would start and then refuse the password exactly as
      written in the file. The same goes for a space at either end of the
      password, or a tab or line break inside it.
    * admin/admin, the placeholder, a password equal to the username, or a
      one-character password would "configure" a lock anyone can open.

    No message here may contain the password.
    """
    if username is None or password is None or username == "" or password == "":
        raise AdminConfigError(f"{where} must set both 'username' and 'password'")
    if not isinstance(username, str) or not isinstance(password, str):
        raise AdminConfigError(
            f'{where}: put the username and the password in quotes. YAML reads an unquoted '
            f'0123, yes or 10:24 as a number or a boolean, e.g. write  password: "0123-race"'
        )
    username = _normal(username.strip())
    password = _normal(password)
    if not username or not password.strip():
        raise AdminConfigError(f"{where} must set both 'username' and 'password'")
    if password != password.strip():
        raise AdminConfigError(
            f"{where}: the password starts or ends with a space or a line break; remove it"
        )
    if any(ord(c) < 32 or c == "\x7f" for c in username + password):
        raise AdminConfigError(
            f"{where}: the username or password contains a tab or another character that cannot be typed"
        )
    if password.lower() in PLACEHOLDER_PASSWORDS:
        raise AdminConfigError(f"{where} still has a placeholder password; set a real one")
    if password.casefold() == username.casefold():
        raise AdminConfigError(f"{where}: the password must not be the username")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AdminConfigError(f"{where}: the password must be at least {MIN_PASSWORD_LENGTH} characters")
    return username, password


def _as_list(value: object) -> object:
    """None, "a, b" or ["a", "b"] as a list; anything else is returned as is
    for the caller to refuse."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[,\s]+", value) if part]
    if isinstance(value, tuple):
        return list(value)
    return value


def _networks(extra: object, where: str) -> tuple:
    """The built-in detector addresses plus any configured ones."""
    extra = _as_list(extra)
    if not isinstance(extra, (list, tuple)):
        raise AdminConfigError(f"{where}: detector_addresses must be a list of addresses")
    networks = []
    for value in list(DEFAULT_DETECTOR_ADDRESSES) + [str(item) for item in extra]:
        try:
            interface = ipaddress.ip_interface(value)
        except ValueError:
            raise AdminConfigError(
                f"{where}: detector_addresses entry '{value}' is not an address or a network"
            ) from None
        # "172.20.10.7/28", exactly as `ip addr` prints it on the Jetson, means
        # that one machine. Read loosely it was the whole hotspot subnet, and
        # every phone on it was a detector again.
        if interface.ip == interface.network.network_address:
            networks.append(interface.network)
        else:
            networks.append(ipaddress.ip_network(interface.ip))
    return tuple(networks)


class AdminAuth:
    def __init__(self, username: str | None = None, password: str | None = None,
                 ttl_s: float = DEFAULT_TTL_S, clock: Callable[[], float] = time.time,
                 source: str = "", detector_addresses: object = None) -> None:
        self.username = username or None
        self._password = password or None
        self.ttl_s = ttl_s
        self._clock = clock
        self.source = source
        self.detector_networks = _networks(detector_addresses, source or "detector_addresses")
        self.extra_detector_networks = self.detector_networks[len(DEFAULT_DETECTOR_ADDRESSES):]
        self.unknown_keys: list = []      # misspelt keys in the config file, for the startup log
        # The tablet at the finish line speaks plain http on the hotspot, so a
        # session cookie can be captured there. The cookie is an HMAC under
        # this key; a key that is one SHA-256 of the password would let a
        # captured cookie be used to guess the password offline at billions
        # of tries a second. PBKDF2 makes each guess cost what it costs us
        # once at startup (~0.1 s).
        self._key = (
            hashlib.pbkdf2_hmac(
                "sha256", self._password.encode(),
                b"race-admin-session|" + self.username.encode(), KEY_ITERATIONS,
            )
            if self.enabled else b""
        )

    @property
    def enabled(self) -> bool:
        return bool(self.username and self._password)

    @property
    def password_length(self) -> int:
        """For the startup line: a password cut short by an unquoted ' #' in
        the file shows up as a length the operator does not recognise."""
        return len(self._password or "")

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_environment(cls, env: Mapping[str, str] | None = None,
                         clock: Callable[[], float] = time.time) -> "AdminAuth":
        env = os.environ if env is None else env
        env_addresses = env.get("RACE_DETECTOR_ADDRESSES")
        user, password = env.get("RACE_ADMIN_USER"), env.get("RACE_ADMIN_PASSWORD")
        if user or password:
            if not (user and password):
                raise AdminConfigError("RACE_ADMIN_USER and RACE_ADMIN_PASSWORD must both be set")
            user, password = _validated(user, password, "RACE_ADMIN_USER / RACE_ADMIN_PASSWORD")
            return cls(user, password, clock=clock, source="environment", detector_addresses=env_addresses)
        path = Path(env.get("RACE_ADMIN_CONFIG") or DEFAULT_CONFIG)
        if not path.is_file():
            return cls(clock=clock, source=f"none ({path} absent)", detector_addresses=env_addresses)
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.MarkedYAMLError as exc:
            # str(exc) quotes the offending source line -- which is the
            # password line -- and this message is logged. Positions only,
            # and no exception chaining: a traceback would print it too.
            lines = sorted({
                mark.line + 1
                for mark in (getattr(exc, "context_mark", None), getattr(exc, "problem_mark", None))
                if mark is not None
            })
            near = f" near line {' / '.join(str(n) for n in lines)}" if lines else ""
            raise AdminConfigError(
                f"{path} is not valid YAML{near}; keep both values in double quotes"
            ) from None
        except OSError as exc:
            raise AdminConfigError(f"{path} cannot be read ({type(exc).__name__})") from None
        except Exception as exc:
            raise AdminConfigError(f"{path} is not valid YAML ({type(exc).__name__})") from None
        if not isinstance(data, dict):
            raise AdminConfigError(f"{path} must contain 'username:' and 'password:'")
        user, password = _validated(data.get("username"), data.get("password"), str(path))
        addresses = _as_list(data.get("detector_addresses"))
        if env_addresses and isinstance(addresses, list):
            addresses = addresses + _as_list(env_addresses)
        auth = cls(user, password, clock=clock, source=str(path), detector_addresses=addresses)
        auth.unknown_keys = sorted(
            str(key) for key in data if key not in ("username", "password", "detector_addresses")
        )
        return auth

    # -- the detector --------------------------------------------------------
    def is_detector_address(self, host: str | None) -> bool:
        try:
            address = ipaddress.ip_address(host or "")
        except ValueError:
            return False
        mapped = getattr(address, "ipv4_mapped", None)
        if mapped is not None:
            address = mapped
        return any(address in network for network in self.detector_networks
                   if network.version == address.version)

    # -- login and sessions ------------------------------------------------
    def check_login(self, username: object, password: object) -> bool:
        if not self.enabled:
            return False
        # The username is not the secret: a tablet keyboard capitalises the
        # first letter of a text field, and "Admin" must not be a lockout.
        # The password is compared exactly. Bytes, not str: compare_digest
        # raises on non-ASCII text, which would turn a strange login into a 500.
        # Bounded BEFORE normalising, and only text: this runs on the event
        # loop, ahead of the failed-login gate, for anyone who can reach the
        # port. The login form sends two strings and nothing else does.
        if not isinstance(username, str) or not isinstance(password, str):
            return False
        if (len(username) > max(LOGIN_FIELD_MAX, 8 * len(self.username))
                or len(password) > max(LOGIN_FIELD_MAX, 8 * len(self._password))):
            return False
        try:
            given_user = _normal(username).strip().casefold()
            given_password = _normal(password)
        except Exception:
            return False
        user_ok = hmac.compare_digest(
            given_user.encode("utf-8", "surrogatepass"),
            self.username.casefold().encode("utf-8", "surrogatepass"),
        )
        pass_ok = hmac.compare_digest(
            given_password.encode("utf-8", "surrogatepass"),
            self._password.encode("utf-8", "surrogatepass"),
        )
        return user_ok and pass_ok

    def _sign(self, login_at: int, expiry: int) -> str:
        message = f"{self.username}|{login_at}|{expiry}".encode()
        return hmac.new(self._key, message, hashlib.sha256).hexdigest()

    def _parse(self, token: object) -> tuple[int, int] | None:
        """(sign-in time, expiry) of a token that is valid right now."""
        if not self.enabled or not isinstance(token, str):
            return None
        match = _TOKEN.fullmatch(token)
        if not match:
            return None
        login_at, expiry = int(match.group(1)), int(match.group(2))
        if not hmac.compare_digest(match.group(3).encode(), self._sign(login_at, expiry).encode()):
            return None
        now = self._clock()
        if expiry < now or now - login_at > MAX_SESSION_S:
            return None
        return login_at, expiry

    def issue(self, renew: str | None = None) -> str | None:
        """A new session; or, given a valid token, that session extended.

        None when the token to renew is not valid (any more). A renewal must
        never turn into a new sign-in: the check and this call read the clock
        separately, and a token that lapsed in between used to come back with
        a fresh sign-in time and a fresh 72 hours, with no password.
        """
        now = self._clock()
        if renew is None:
            login_at = int(now)
        else:
            parsed = self._parse(renew)
            if parsed is None:
                return None
            login_at = parsed[0]
        expiry = int(min(now + self.ttl_s, login_at + MAX_SESSION_S))
        return f"v2.{login_at}.{expiry}.{self._sign(login_at, expiry)}"

    def ends_in(self, token: str | None) -> int | None:
        """Seconds until this session's hard end (the cap after sign-in)."""
        parsed = self._parse(token)
        if not parsed:
            return None
        return max(0, int(parsed[0] + MAX_SESSION_S - self._clock()))

    def verify(self, token: str | None) -> bool:
        return self._parse(token) is not None

    def needs_refresh(self, token: str | None) -> bool:
        """A valid session with less than half its life left (and room to grow).

        Sessions slide: an operator who signed in the evening before must
        not be signed out at the gun. The API re-issues the cookie on the
        session check and on operator actions once it is past half-life.
        """
        parsed = self._parse(token)
        if not parsed:
            return False
        login_at, expiry = parsed
        if expiry >= login_at + MAX_SESSION_S:
            return False                      # already at the cap; renewing gains nothing
        return expiry - self._clock() < self.ttl_s / 2
