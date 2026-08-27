"""Polite HTTP with provenance.

Every response carries where it came from, when, and a hash of its content, so
a parser can be improved later and re-run over stored responses instead of
re-downloading. Requests are paced per host and identify themselves honestly.
"""
from __future__ import annotations

import hashlib
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

USER_AGENT = "fpl-engine-acquire/0.1 (personal research; not for redistribution)"
DEFAULT_DELAY_S = 2.0

_LAST: dict[str, float] = {}
_LOCK = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


@dataclass
class Response:
    url: str
    status: int | None
    text: str
    retrieved_utc: str
    content_type: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    @property
    def hash(self) -> str:
        return content_hash(self.text)


def _throttle(host: str, delay: float) -> None:
    while True:
        with _LOCK:
            now = time.monotonic()
            wait = _LAST.get(host, 0.0) + delay - now
            if wait <= 0:
                _LAST[host] = now
                return
        time.sleep(wait)


def get(url: str, *, delay: float = DEFAULT_DELAY_S, timeout: int = 30,
        headers: dict | None = None, retries: int = 3) -> Response:
    host = urllib.parse.urlparse(url).netloc
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    last = ""
    for attempt in range(retries):
        _throttle(host, delay)
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return Response(url, r.status, r.read().decode("utf-8", "replace"),
                                utcnow(), r.headers.get("Content-Type", ""))
        except urllib.error.HTTPError as e:
            # 4xx are decisions, not glitches: do not hammer them
            if 400 <= e.code < 500 and e.code != 429:
                return Response(url, e.code, "", utcnow(), error=f"HTTP {e.code}")
            last = f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    return Response(url, None, "", utcnow(), error=last)
