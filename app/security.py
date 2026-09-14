"""Hardening for a planner that is reachable from the internet.

Pure-ASGI middleware (no BaseHTTPMiddleware: it buffers bodies and has known
problems with streaming responses) plus a small in-memory rate limiter.

    SessionMiddleware      resolves the signed session cookie into
                           `scope["state"]["session"]`, minting an anonymous
                           one — and setting the cookie — when there is none
    SecurityHeaders        CSP, frame denial, nosniff, referrer policy, HSTS
    BodyLimit              413 on request bodies over the cap
    MutationGuard          state-changing /api calls must be JSON with the
                           `X-Requested-With` header — a cross-site form post
                           cannot send either, so together with SameSite=Lax
                           cookies this closes CSRF without a token dance
    RateLimiter            token buckets per client IP and route class

Client IP: `scope["client"]` unless `FPLABS_TRUSTED_PROXY=1`, in which case
the first `X-Forwarded-For` hop is used (set it ONLY behind your own reverse
proxy — otherwise anyone can spoof the header and dodge the limiter).
"""
from __future__ import annotations

import os
import threading
import time

from . import auth

MAX_BODY_BYTES = int(os.environ.get("FPLABS_MAX_BODY", 1_000_000))

# route class -> (capacity, refill per second)
LIMITS = {
    "cheap": (240, 4.0),        # reads
    "write": (60, 0.5),         # saving drafts / squads
    "league": (12, 0.05),       # fans out to FPL's API per rival
    "heavy": (6, 1 / 120),      # a solve: minutes of CPU each
    "auth": (20, 0.1),
}


def _client_ip(scope) -> str:
    if os.environ.get("FPLABS_TRUSTED_PROXY") == "1":
        for k, v in scope.get("headers") or []:
            if k == b"x-forwarded-for":
                return v.decode("latin-1").split(",")[0].strip() or "?"
    client = scope.get("client")
    return client[0] if client else "?"


def _header(scope, name: bytes) -> str | None:
    for k, v in scope.get("headers") or []:
        if k == name:
            return v.decode("latin-1")
    return None


def _cookies(scope) -> dict[str, str]:
    raw = _header(scope, b"cookie") or ""
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            out[k] = v
    return out


async def _send_plain(send, status: int, text: str) -> None:
    body = text.encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------

class SessionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cookies = _cookies(scope)
        session = auth.verify(cookies.get(auth.SESSION_COOKIE))
        fresh = False
        if not session or not session.get("sid"):
            session = auth.new_session()
            fresh = True
        state = scope.setdefault("state", {})
        state["session"] = session
        state["session_fresh"] = fresh
        state["client_ip"] = _client_ip(scope)

        async def send_wrapped(message):
            if message["type"] == "http.response.start":
                # a handler may have replaced the session (login / logout);
                # whatever is in state at response time is what gets set
                sess = state.get("session")
                if state.get("session_fresh") or state.get("session_dirty"):
                    hdr = auth.cookie_header(
                        auth.SESSION_COOKIE, auth.sign(sess),
                        auth.SESSION_DAYS * 86400)
                    message.setdefault("headers", [])
                    message["headers"] = list(message["headers"]) + [
                        (b"set-cookie", hdr.encode("latin-1"))]
            await send(message)

        return await self.app(scope, receive, send_wrapped)


# --------------------------------------------------------------------------
# headers
# --------------------------------------------------------------------------

def csp() -> str:
    # Inline styles are used throughout the React tree (style={...}), so
    # style-src needs 'unsafe-inline'; scripts are bundled and never inline.
    return "; ".join([
        "default-src 'self'",
        # Cloudflare injects its analytics beacon into HTML it proxies; the
        # site runs behind Cloudflare, so let that one origin through
        "script-src 'self' https://static.cloudflareinsights.com",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "img-src 'self' data: https://lh3.googleusercontent.com",
        "connect-src 'self' https://cloudflareinsights.com",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self' https://accounts.google.com",
        "object-src 'none'",
    ])


class SecurityHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_wrapped(message):
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                hdrs += [
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy",
                     b"camera=(), microphone=(), geolocation=(), payment=()"),
                    (b"content-security-policy", csp().encode()),
                    (b"cross-origin-opener-policy", b"same-origin-allow-popups"),
                ]
                if auth.secure_cookies():
                    hdrs.append((b"strict-transport-security",
                                 b"max-age=31536000; includeSubDomains"))
                path = scope.get("path", "")
                if path.startswith("/api/"):
                    hdrs.append((b"cache-control", b"no-store"))
                elif not path.startswith("/assets/"):
                    # index.html and the legal pages: always revalidate, so a
                    # redeploy is seen on the next load (assets are hashed)
                    hdrs.append((b"cache-control", b"no-cache"))
                message["headers"] = hdrs
            await send(message)

        return await self.app(scope, receive, send_wrapped)


# --------------------------------------------------------------------------
# body size
# --------------------------------------------------------------------------

class BodyLimit:
    def __init__(self, app, max_bytes: int = MAX_BODY_BYTES):
        self.app = app
        self.max = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cl = _header(scope, b"content-length")
        if cl and cl.isdigit() and int(cl) > self.max:
            return await _send_plain(send, 413, '{"detail":"request body too large"}')
        seen = 0
        tripped = False

        async def receive_wrapped():
            nonlocal seen, tripped
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body") or b"")
                if seen > self.max:
                    tripped = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return msg

        # a chunked body that grows past the cap is truncated to nothing,
        # which the JSON parser then rejects with a 4xx rather than a solve
        # being handed a multi-megabyte parameter set
        return await self.app(scope, receive_wrapped, send)


# --------------------------------------------------------------------------
# CSRF: state-changing calls must look like our own fetch()
# --------------------------------------------------------------------------

MUTATING = {"POST", "PUT", "DELETE", "PATCH"}


class MutationGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (scope["type"] == "http" and scope.get("method") in MUTATING
                and scope.get("path", "").startswith("/api/")):
            ct = (_header(scope, b"content-type") or "").lower()
            xrw = (_header(scope, b"x-requested-with") or "")
            if not ct.startswith("application/json") or xrw != "fetch":
                return await _send_plain(
                    send, 403, '{"detail":"cross-site request blocked"}')
        return await self.app(scope, receive, send)


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------

class RateLimiter:
    """Token bucket per (client, class). Memory-bounded by evicting idle
    buckets; good enough for one process, which is what this app is."""

    def __init__(self, limits: dict[str, tuple[int, float]] | None = None):
        self.limits = limits or LIMITS
        self._b: dict[tuple[str, str], list[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def allow(self, client: str, cls: str, now: float | None = None) -> bool:
        cap, rate = self.limits.get(cls) or self.limits.get("cheap") or LIMITS["cheap"]
        now = time.monotonic() if now is None else now
        with self._lock:
            if now - self._last_sweep > 300:
                self._sweep(now)
            b = self._b.get((client, cls))
            if b is None:
                b = [float(cap), now]
                self._b[(client, cls)] = b
            tokens, last = b
            tokens = min(float(cap), tokens + (now - last) * rate)
            if tokens < 1.0:
                b[0], b[1] = tokens, now
                return False
            b[0], b[1] = tokens - 1.0, now
            return True

    def _sweep(self, now: float) -> None:
        self._last_sweep = now
        dead = [k for k, (tok, last) in self._b.items()
                if now - last > 3600]
        for k in dead:
            self._b.pop(k, None)
        if len(self._b) > 50_000:            # somebody is spraying IPs
            self._b.clear()


limiter = RateLimiter()
