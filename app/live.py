"""The Live desk: everything that moves between now and the deadline.

Unlike the admin deadline desk (``app/deadline.py``), which is the operator's
view of the machinery, this is for everyone and it exists for one window:

    24 h before a deadline  ->  the tab appears and updates
    the deadline passes     ->  it stays 6 h more, marked over
    otherwise               ->  it is not there at all

That schedule is the whole design. A permanent "deadline" tab is furniture
nobody reads; a tab that shows up when there is something to do is a signal.
Everything on it is already archived by the pipeline — team news, Friday's
manager quotes, the three predicted-lineup feeds, price pressure and the
model's own biggest changes of mind — and none of it changes a projection.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd

from fpl_engine import config, db

from . import deadline as dd
from . import services

_cache: dict = {"t": 0.0, "v": None}
TTL = 60.0

OPEN_BEFORE = 24 * 3600.0     # the tab appears this long before a deadline
STAY_AFTER = 6 * 3600.0       # ...and lingers this long after it, marked over


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def window(deadlines: dict, now: float | None = None) -> dict:
    """Which gameweek the live desk is about, and what phase it is in.

    ``deadlines`` is ``status.deadlines`` — gameweek -> ISO timestamp. Phases:
    ``open`` (inside the 24 h run-up), ``closed`` (the deadline has gone but
    within 6 h, so the gameweek is being played), ``idle`` (no tab).
    """
    now = time.time() if now is None else now
    stamped = sorted((int(g), t) for g, t in
                     ((g, _ts(v)) for g, v in (deadlines or {}).items())
                     if t is not None)
    nxt = next(((g, t) for g, t in stamped if t > now), None)
    prev = next(((g, t) for g, t in reversed(stamped) if t <= now), None)
    if prev and now - prev[1] <= STAY_AFTER:
        return {"phase": "closed", "gw": prev[0], "deadline": prev[1],
                "seconds": now - prev[1]}
    if nxt and nxt[1] - now <= OPEN_BEFORE:
        return {"phase": "open", "gw": nxt[0], "deadline": nxt[1],
                "seconds": nxt[1] - now}
    return {"phase": "idle", "gw": nxt[0] if nxt else None,
            "deadline": nxt[1] if nxt else None,
            "seconds": (nxt[1] - now) if nxt else None}


def _fixtures(conn, season: str, gw: int, tm: dict) -> list[dict]:
    rows = conn.execute(
        "SELECT fixture_id, kickoff_utc, team_h, team_a, team_h_score, "
        "team_a_score, finished FROM fixture WHERE season=? AND gw=? "
        "ORDER BY kickoff_utc", (season, gw)).fetchall()
    out = []
    for r in rows:
        out.append({
            "fixture_id": int(r["fixture_id"]),
            "kickoff": r["kickoff_utc"],
            "home": tm.get(int(r["team_h"])), "away": tm.get(int(r["team_a"])),
            "home_id": int(r["team_h"]), "away_id": int(r["team_a"]),
            "score": (None if r["team_h_score"] is None else
                      [int(r["team_h_score"]), int(r["team_a_score"] or 0)]),
            "finished": bool(r["finished"]),
        })
    return out


def _top_picks(gw: int, pl: dict, tm: dict, limit: int = 12) -> list[dict]:
    """The model's own board for this gameweek — what it would captain."""
    cache = services._load_proj_cache()
    rows = []
    for rec in cache.get("players", {}).values():
        ep = (rec.get("ep") or {}).get(str(gw))
        if ep is None:
            continue
        rows.append({"player_id": int(rec["player_id"]), "name": rec["player"],
                     "team": rec.get("team"), "pos": rec.get("position"),
                     "price": rec.get("price"), "ep": round(float(ep), 2)})
    rows.sort(key=lambda r: -r["ep"])
    return rows[:limit]


def _price_pressure(limit: int = 8) -> dict:
    """Who is about to move before the deadline, at the price the Round-7
    conversion puts on it — a tie-breaker, said out loud."""
    try:
        p = services.prices_payload(limit=limit)
    except Exception as exc:            # noqa: BLE001
        return {"ok": False, "error": str(exc), "risers": [], "fallers": []}
    return p


def payload(force: bool = False) -> dict:
    now = time.time()
    if not force and _cache["v"] is not None and now - _cache["t"] < TTL:
        return _cache["v"]
    season = config.CURRENT_SEASON
    status = services.status_payload()
    win = window(status.get("deadlines") or {}, now)
    gw = win["gw"]
    if gw is None:
        out = {"season": season, "window": win, "built_at": now}
        _cache["t"], _cache["v"] = now, out
        return out
    conn = db.connect(config.DB_PATH)
    try:
        pl, tm = dd._names(conn, season)
        start_p = services._model_start_probs(season)
        out = {
            "season": season,
            "window": win,
            "gw": gw,
            "fixtures": _fixtures(conn, season, gw, tm),
            # FPL's own change log: the only team news that is both free and
            # timestamped by the source, and the one signal the live model
            # actually consumes
            "news": dd._news(conn, season, pl, tm, days=7.0)[:40],
            "pressers": dd._pressers(conn, season, gw, pl, tm)[:40],
            "lineups": dd._lineups(conn, season, gw, pl, tm, start_p),
            "movers": dd._movers(gw, pl, tm),
            "picks": _top_picks(gw, pl, tm),
            "prices": _price_pressure(),
            "proj_updated_at": status.get("proj_updated_at"),
            "built_at": now,
        }
    finally:
        conn.close()
    _cache["t"], _cache["v"] = now, out
    return out
