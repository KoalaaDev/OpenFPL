"""Local disk cache for club shirts and badges.

The UI was pointing every <img> straight at fantasy.premierleague.com and
resources.premierleague.com. A pitch view is 15 shirts, the Projections table
is a badge per row, and each one was a separate cross-origin request on a cold
cache — which is why the Planner took a few seconds to look finished.

These are ~2KB files for 20 clubs that change once a season, so they are
fetched once, written to `data/web_cache/img/`, and then served from disk with
an immutable cache header. After the first request nothing leaves the machine,
and the app works offline.

A miss degrades to a 404 and the UI already hides broken images, so a CDN
change or no network makes the page plainer, never broken.
"""
from __future__ import annotations

import os
import threading
import urllib.request

from fpl_engine import config

IMG_DIR = os.path.join(config.DATA_DIR, "web_cache", "img")
TIMEOUT = 8
UA = "OpenFPL-planner (local cache)"

SOURCES = {
    "shirt": "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}-66.webp",
    "shirt_gk": "https://fantasy.premierleague.com/dist/img/shirts/standard/shirt_{code}_1-66.webp",
    "badge": "https://resources.premierleague.com/premierleague/badges/70/t{code}.png",
    # Player cut-outs, keyed by player.code (not player_id). 250x250 is the
    # smallest size that still looks clean behind a card header; the 110x140
    # variant visibly softens once it is scaled up to fill one.
    "photo": "https://resources.premierleague.com/premierleague/photos/players/250x250/p{code}.png",
}
EXT = {"shirt": "webp", "shirt_gk": "webp", "badge": "png", "photo": "png"}
MEDIA = {"webp": "image/webp", "png": "image/png"}

_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _lock_guard:
        return _locks.setdefault(key, threading.Lock())


def path_for(kind: str, code: int) -> str | None:
    """Local path for one image, downloading it once if needed.

    Returns None when the kind is unknown or the source cannot be reached —
    never a placeholder, because a wrong badge is worse than a missing one.
    """
    if kind not in SOURCES:
        return None
    try:
        code = int(code)
    except (TypeError, ValueError):
        return None
    os.makedirs(IMG_DIR, exist_ok=True)
    name = f"{kind}_{code}.{EXT[kind]}"
    dest = os.path.join(IMG_DIR, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    with _lock_for(name):
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest          # another thread won the race
        url = SOURCES[kind].format(code=code)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                blob = r.read()
            if not blob:
                return None
            tmp = dest + ".part"  # never leave a truncated file to be cached
            with open(tmp, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, dest)
            return dest
        except Exception:
            return None


def media_type(kind: str) -> str:
    return MEDIA.get(EXT.get(kind, ""), "application/octet-stream")


def prewarm(codes: list[int]) -> None:
    """Fetch every club's art in the background, once, at startup.

    On-demand caching alone still makes the FIRST view slow, which is the
    complaint. Twenty clubs is sixty small files and it happens while the user
    is still reading the page.

    Player photos are deliberately NOT prewarmed: they are keyed per player at
    ~340KB each, so the whole league would be ~200MB fetched to show the
    handful of cards anyone actually opens. They cache on first view instead.
    """
    def run():
        for code in codes:
            for kind in ("shirt", "shirt_gk", "badge"):
                path_for(kind, code)
    threading.Thread(target=run, daemon=True, name="img-prewarm").start()
