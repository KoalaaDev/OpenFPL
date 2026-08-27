"""FPL availability: status, chance of playing, and the news text behind it.

Why this source first. The modelling work established that expected minutes is
the largest remaining source of error, that the start model is already
calibrated at AUC ~0.95, and that the one genuinely unexploited lever is team
news — which FPL publishes itself, before the deadline, in `status`,
`chance_of_playing_next_round`, `news` and `news_added`. Backtests currently
have to switch it off, because the database only ever held *today's* status,
overwritten every pull. Nothing was archiving it.

This collector turns that snapshot into a change log. It needs no scraping, no
third party and no terms-of-service question: it is the same endpoint the
pipeline already uses.

Two timestamps, and the distinction matters:

  observed_utc          when we saw the state. Guaranteed safe for a backtest.
  source_published_utc  FPL's own `news_added`. Because it is preserved, a
                        snapshot taken later still tells us when a standing
                        item began — partial reconstruction of the past
                        without having been there.

What it cannot do is recover states that were overwritten before collection
started. Availability for seasons already gone is not recoverable by anyone;
this becomes backtestable roughly a season after it starts running.
"""
from __future__ import annotations

import json

from ..core import http
from .. import storage

SOURCE_ID = "fpl_api"
PARSER_VERSION = "1"
URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def register(conn) -> None:
    storage.register_source(
        conn, SOURCE_ID,
        name="Fantasy Premier League official API",
        base_url="https://fantasy.premierleague.com",
        source_type="api",
        enabled=1,
        robots_policy="robots.txt present, no Disallow rules for generic agents",
        terms_note="Public JSON endpoint already used by the modelling pipeline; "
                   "read-only, no authentication, polite rate limiting applied",
        request_delay=1.0,
        parser_version=PARSER_VERSION)


def _chance(e: dict):
    v = e.get("chance_of_playing_next_round")
    return None if v is None else float(v) / 100.0


def parse(payload: str) -> list[dict]:
    """Normalise a bootstrap payload into availability observations."""
    boot = json.loads(payload)
    out = []
    for e in boot.get("elements", []):
        out.append({
            "player_id": int(e["id"]),
            "status": e.get("status"),
            "chance_next": _chance(e),
            # verbatim: "75% chance of playing" is the record, not "available"
            "news": (e.get("news") or "").strip() or None,
            "source_published_utc": e.get("news_added"),
        })
    return out


def _season_of(boot_payload: str, fallback: str) -> str:
    """Season label, from the events if the payload carries them."""
    try:
        boot = json.loads(boot_payload)
        for ev in boot.get("events", []):
            dl = ev.get("deadline_time") or ""
            if len(dl) >= 4:
                y = int(dl[:4])
                # a season starting in August spans y..y+1; events before July
                # belong to the season that started the previous year
                start = y if int(dl[5:7]) >= 7 else y - 1
                return f"{start}-{str(start + 1)[-2:]}"
    except Exception:  # noqa: BLE001
        pass
    return fallback


def ingest_payload(conn, payload: str, *, season: str, observed_utc: str,
                   raw_id: int | None) -> int:
    """Write observations for states that have CHANGED since the last one.

    A change log rather than a snapshot table: re-ingesting the same payload
    writes nothing, and "state at time T" stays a single indexed lookup.
    """
    prev = storage.last_availability(conn, season)
    rows = []
    for obs in parse(payload):
        pid = obs["player_id"]
        cur = (obs["status"], obs["chance_next"], obs["news"])
        if prev.get(pid) == cur:
            continue
        rows.append((season, pid, observed_utc, SOURCE_ID,
                     obs["source_published_utc"], obs["status"],
                     obs["chance_next"], obs["news"], raw_id))
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO acq_player_availability (season, player_id, "
            "observed_utc, source_id, source_published_utc, status, "
            "chance_next, news, raw_id) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def pull(conn, *, season: str, dry_run: bool = False) -> dict:
    """Fetch the current bootstrap and record any availability changes."""
    register(conn)
    resp = http.get(URL, delay=1.0)
    if not resp.ok:
        storage.mark(conn, SOURCE_ID, False, resp.retrieved_utc)
        return {"source": SOURCE_ID, "error": resp.error or f"HTTP {resp.status}"}
    season = _season_of(resp.text, season)
    if dry_run:
        return {"source": SOURCE_ID, "season": season, "dry_run": True,
                "players": len(parse(resp.text))}
    raw_id = storage.store_raw(conn, SOURCE_ID, resp,
                               parser_version=PARSER_VERSION)
    n = ingest_payload(conn, resp.text, season=season,
                       observed_utc=resp.retrieved_utc, raw_id=raw_id)
    storage.mark(conn, SOURCE_ID, True, resp.retrieved_utc)
    conn.execute(
        "INSERT OR REPLACE INTO acq_sync_state (source_id, dataset, season, "
        "last_cursor, last_hash, last_success_utc) VALUES (?,?,?,?,?,?)",
        (SOURCE_ID, "availability", season, resp.retrieved_utc, resp.hash,
         resp.retrieved_utc))
    return {"source": SOURCE_ID, "season": season, "changes": n,
            "raw_id": raw_id}


def backfill_from_snapshots(conn, *, season: str) -> dict:
    """Replay bootstrap payloads the pipeline already archived.

    ``raw_snapshot`` has been storing these all along without anyone parsing
    them, so the change log can start earlier than this collector did.
    """
    rows = conn.execute(
        "SELECT retrieved_utc, payload FROM raw_snapshot "
        "WHERE endpoint='bootstrap-static' AND payload LIKE '{%' "
        "ORDER BY retrieved_utc ASC").fetchall()
    total, used = 0, 0
    for r in rows:
        payload = r["payload"]
        try:
            s = _season_of(payload, season)
            n = ingest_payload(conn, payload, season=s,
                               observed_utc=r["retrieved_utc"], raw_id=None)
        except Exception:  # noqa: BLE001 - a bad archived row must not abort
            continue
        used += 1
        total += n
    return {"snapshots_replayed": used, "changes": total}
