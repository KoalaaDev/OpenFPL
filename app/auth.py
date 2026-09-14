"""Sessions and Google sign-in.

Every visitor gets a signed session cookie. Anonymous sessions carry only a
random id; signing in with Google binds a user id to it. The cookie is the
only client-side state — the payload is HMAC-signed with a server secret, so
it cannot be forged or edited, and it carries no personal data beyond the
opaque ids.

Google OAuth is the plain server-side authorization-code flow:

    /api/auth/google/start     -> 302 to Google with a signed state cookie
    /api/auth/google/callback  -> code -> tokens; the id_token is verified by
                                  Google's own tokeninfo endpoint (signature,
                                  expiry, issuer) and then checked here for
                                  audience, nonce and a verified email

No third-party library: the flow is four HTTPS requests and one HMAC.

Configuration (environment):

    GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET   from Google Cloud Console
    FPLABS_BASE_URL       public origin, e.g. https://fplabs.example.com — the
                          OAuth redirect URI is <base>/api/auth/google/callback
                          and must be registered on the Google client
    FPLABS_SECRET_KEY     cookie-signing secret; generated once into
                          data/web_cache/secret.key when unset
    FPLABS_ADMIN_EMAILS   comma-separated Google emails allowed to trigger a
                          data pull / projection rebuild by hand
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
import urllib.request

from fpl_engine import config

SESSION_COOKIE = "fplabs_session"
OAUTH_COOKIE = "fplabs_oauth"
SESSION_DAYS = 30
OAUTH_STATE_SECONDS = 600

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

_secret_cache: bytes | None = None


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def base_url() -> str:
    return (os.environ.get("FPLABS_BASE_URL") or "").rstrip("/")


def secure_cookies() -> bool:
    """`Secure` cookies are only deliverable over HTTPS; on a plain-HTTP dev
    box they would silently never be set. Follow the public URL's scheme."""
    return base_url().lower().startswith("https://")


def google_client() -> tuple[str, str] | None:
    cid = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    sec = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    if cid and sec:
        return cid, sec
    return None


def google_enabled() -> bool:
    return google_client() is not None and bool(base_url())


def admin_emails() -> set[str]:
    raw = os.environ.get("FPLABS_ADMIN_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def is_admin(user: dict | None) -> bool:
    if not user:
        return False
    return (user.get("email") or "").lower() in admin_emails()


def secret_key() -> bytes:
    """The cookie-signing key. From the environment, else generated once and
    kept in a file only the server user can read — losing it merely signs
    everyone out."""
    global _secret_cache
    if _secret_cache:
        return _secret_cache
    env = os.environ.get("FPLABS_SECRET_KEY")
    if env:
        _secret_cache = env.encode()
        return _secret_cache
    path = os.path.join(config.DATA_DIR, "web_cache", "secret.key")
    try:
        with open(path, "rb") as f:
            key = f.read().strip()
        if len(key) >= 32:
            _secret_cache = key
            return key
    except OSError:
        pass
    key = secrets.token_urlsafe(48).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    _secret_cache = key
    return key


# --------------------------------------------------------------------------
# signed tokens
# --------------------------------------------------------------------------

def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign(payload: dict, key: bytes | None = None) -> str:
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    mac = hmac.new(key or secret_key(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64e(mac)}"


def verify(token: str | None, key: bytes | None = None) -> dict | None:
    """The payload if the signature checks out and it has not expired."""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    want = hmac.new(key or secret_key(), body.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(want, _b64d(sig)):
            return None
        payload = json.loads(_b64d(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    exp = payload.get("exp")
    if exp is not None and float(exp) < time.time():
        return None
    return payload


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

def new_session(uid: str | None = None) -> dict:
    now = time.time()
    return {"sid": secrets.token_urlsafe(18), "uid": uid,
            "iat": now, "exp": now + SESSION_DAYS * 86400}


def principal_of(session: dict) -> str:
    if session.get("uid"):
        return f"user:{session['uid']}"
    return f"anon:{session['sid']}"


def cookie_header(name: str, value: str, max_age: int, *, path: str = "/") -> str:
    parts = [f"{name}={value}", f"Max-Age={max_age}", f"Path={path}",
             "HttpOnly", "SameSite=Lax"]
    if secure_cookies():
        parts.append("Secure")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Google OAuth
# --------------------------------------------------------------------------

def redirect_uri() -> str:
    return f"{base_url()}/api/auth/google/callback"


def start_login(next_path: str = "/") -> tuple[str, str]:
    """(authorization URL, signed state cookie value)."""
    client = google_client()
    if not client or not base_url():
        raise RuntimeError("Google sign-in is not configured")
    cid, _ = client
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    if not next_path.startswith("/") or next_path.startswith("//"):
        next_path = "/"                      # never an open redirect
    params = {
        "client_id": cid, "redirect_uri": redirect_uri(),
        "response_type": "code", "scope": "openid email profile",
        "state": state, "nonce": nonce, "prompt": "select_account",
        "access_type": "online",
    }
    url = f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"
    cookie = sign({"state": state, "nonce": nonce, "next": next_path,
                   "exp": time.time() + OAUTH_STATE_SECONDS})
    return url, cookie


def _post_form(url: str, data: dict, timeout: float = 15.0) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get_json(url: str, timeout: float = 15.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def exchange_code(code: str) -> dict:
    cid, sec = google_client() or (None, None)
    return _post_form(GOOGLE_TOKEN_URL, {
        "code": code, "client_id": cid, "client_secret": sec,
        "redirect_uri": redirect_uri(), "grant_type": "authorization_code"})


def verify_id_token(id_token: str, nonce: str, *, fetch=_get_json) -> dict:
    """Google validates the signature and expiry; we check that the token was
    minted for THIS client, for THIS login attempt, and names a verified
    email. Returns the claims."""
    info = fetch(f"{GOOGLE_TOKENINFO_URL}?{urllib.parse.urlencode({'id_token': id_token})}")
    cid = (google_client() or ("", ""))[0]
    if info.get("aud") != cid:
        raise ValueError("id_token audience mismatch")
    if info.get("iss") not in GOOGLE_ISSUERS:
        raise ValueError("id_token issuer mismatch")
    if nonce and info.get("nonce") != nonce:
        raise ValueError("id_token nonce mismatch")
    ev = info.get("email_verified")
    if ev not in (True, "true", "True", 1, "1"):
        raise ValueError("Google email is not verified")
    if not info.get("sub") or not info.get("email"):
        raise ValueError("id_token is missing sub/email")
    return info


def finish_login(code: str, state_cookie: str | None, state_param: str | None,
                 *, exchange=exchange_code, verify_token=verify_id_token) -> tuple[dict, str]:
    """Complete the callback: returns (claims, next_path) or raises."""
    st = verify(state_cookie)
    if not st or not state_param or not hmac.compare_digest(
            str(st.get("state")), str(state_param)):
        raise ValueError("login state mismatch — start again")
    tokens = exchange(code)
    idt = tokens.get("id_token")
    if not idt:
        raise ValueError("Google returned no id_token")
    claims = verify_token(idt, st.get("nonce") or "")
    return claims, st.get("next") or "/"


def public_user(user: dict | None) -> dict | None:
    if not user:
        return None
    return {"id": user["id"], "email": user.get("email"),
            "name": user.get("name"), "picture": user.get("picture"),
            "plan": user.get("plan") or "free", "is_admin": is_admin(user)}
