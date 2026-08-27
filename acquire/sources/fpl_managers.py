"""Elite-manager picks: what proven managers own, before each deadline.

The hypothesis is that skilled managers collectively hold forward-looking
information the model cannot derive from history — team news, the eye test,
whatever a beat reporter said on Friday. It is the same shape as the betting
market, which was measured to *encompass* our team model entirely, so it is
worth testing rather than assuming either way.

Two things had to be settled first, and both shape the design.

**It is not backtestable.** `entry/{id}/event/{gw}/picks/` has no season
parameter and serves only the season in progress; previous seasons 404.
`entry/{id}/history/` keeps past seasons as totals and ranks, never squads. So
2024-25 and 2025-26 picks are gone, and the only route is forward collection —
the same wall as availability and expected lineups.

**The panel must not be "the current top 1000".** Sampled after one gameweek,
the top 250 of the Overall league had a *median past-season rank of 2.6
million*; 78% had a career median worse than a million, only 19% had ever
finished inside the top 100k, and 15% had no history at all. That table is a
lottery draw, not a ranking of skill, and it stays partly luck all season.

So the panel is selected on **past seasons only** — membership never looks at
how a manager is doing now.

**One caveat, and it matters.** The *filter* is clean but the *enumeration* is
not: candidates are found by walking the Overall table, and after one gameweek
you only sit near the top of it if that gameweek went well. The 59 managers
found this way scored 103-116 in GW1 against an FPL average of 50. Their GW1
picks therefore cannot look bad, and any statistic computed on them is an
artefact of the search, not a finding.

The resolution is that the panel is **fixed once built**. From the next
gameweek onward these managers are followed regardless of how they do, so their
picks are recorded before the outcome and are genuinely out of sample. Rebuild
the panel only in pre-season, when the table carries no current-season
information at all; rebuilding it mid-season would re-introduce exactly the
conditioning described above.
"""
from __future__ import annotations

import json

from ..core import http
from .. import storage

SOURCE_ID = "fpl_managers"
PARSER_VERSION = "1"
API = "https://fantasy.premierleague.com/api"

# "proven" = repeatedly good, not once lucky
ELITE_RANK = 100_000
ELITE_SEASONS = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS acq_manager (
    entry_id      INTEGER PRIMARY KEY,
    observed_utc  TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    seasons_played INTEGER,
    best_rank     INTEGER,
    median_rank   INTEGER,
    elite_seasons INTEGER,      -- seasons finished inside ELITE_RANK
    is_panel      INTEGER,      -- selected on PAST seasons only
    discovered_rank INTEGER     -- where it sat when found; never a selector
);

CREATE TABLE IF NOT EXISTS acq_manager_pick (
    season        TEXT NOT NULL,
    gw            INTEGER NOT NULL,
    entry_id      INTEGER NOT NULL,
    player_id     INTEGER NOT NULL,
    slot          INTEGER,
    multiplier    INTEGER,      -- 0 bench, 1 starting, 2 captain, 3 triple
    is_captain    INTEGER,
    is_vice       INTEGER,
    active_chip   TEXT,
    observed_utc  TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    PRIMARY KEY (season, gw, entry_id, player_id)
);
CREATE INDEX IF NOT EXISTS idx_acq_pick_gw
    ON acq_manager_pick (season, gw, player_id);
"""


def init(conn) -> None:
    conn.executescript(SCHEMA)


def register(conn) -> None:
    storage.register_source(
        conn, SOURCE_ID,
        name="FPL manager picks (proven-manager panel)",
        base_url="https://fantasy.premierleague.com",
        source_type="api", enabled=1,
        robots_policy="robots.txt present, no Disallow rules for generic agents",
        terms_note="Public read-only endpoints; panel selected on past seasons "
                   "only so no current-season information enters selection",
        request_delay=0.3, parser_version=PARSER_VERSION)


def _j(url, delay=0.3):
    r = http.get(url, delay=delay)
    if not r.ok:
        return None
    try:
        return json.loads(r.text)
    except ValueError:
        return None


def _classify(past: list[dict]) -> dict:
    ranks = sorted(p["rank"] for p in past if p.get("rank"))
    if not ranks:
        return {"seasons_played": len(past), "best_rank": None,
                "median_rank": None, "elite_seasons": 0, "is_panel": 0}
    elite = sum(1 for r in ranks if r <= ELITE_RANK)
    return {"seasons_played": len(past), "best_rank": ranks[0],
            "median_rank": ranks[len(ranks) // 2], "elite_seasons": elite,
            "is_panel": int(elite >= ELITE_SEASONS)}


def build_panel(conn, *, pages: int = 20, progress=None) -> dict:
    """Find managers and grade them on their PAST seasons.

    The Overall table is only a way to enumerate entry ids. Where a manager
    currently sits never decides membership — that would import this season's
    luck into the panel and make everything downstream circular.
    """
    register(conn)
    init(conn)
    now = http.utcnow()
    found = []
    for page in range(1, pages + 1):
        d = _j(f"{API}/leagues-classic/314/standings/?page_standings={page}")
        if not d:
            break
        res = d["standings"]["results"]
        found += [(r["entry"], r["rank"]) for r in res]
        if not d["standings"].get("has_next"):
            break
    rows, panel = [], 0
    for i, (eid, rank) in enumerate(found, 1):
        h = _j(f"{API}/entry/{eid}/history/")
        if h is None:
            continue
        c = _classify(h.get("past") or [])
        panel += c["is_panel"]
        rows.append((eid, now, SOURCE_ID, c["seasons_played"], c["best_rank"],
                     c["median_rank"], c["elite_seasons"], c["is_panel"], rank))
        if progress and i % 50 == 0:
            progress(f"    graded {i}/{len(found)} ({panel} proven so far)")
    conn.executemany(
        "INSERT OR REPLACE INTO acq_manager (entry_id, observed_utc, source_id, "
        "seasons_played, best_rank, median_rank, elite_seasons, is_panel, "
        "discovered_rank) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return {"scanned": len(found), "graded": len(rows), "panel": panel}


def panel_entries(conn) -> list[int]:
    return [int(r[0]) for r in conn.execute(
        "SELECT entry_id FROM acq_manager WHERE is_panel=1 ORDER BY entry_id")]


def pull_picks(conn, *, season: str, gw: int, progress=None) -> dict:
    """Snapshot the panel's squads for a completed gameweek.

    Picks are locked at the deadline, so a row is point-in-time valid for any
    decision made from that deadline onward.
    """
    register(conn)
    init(conn)
    now = http.utcnow()
    entries = panel_entries(conn)
    rows, misses = [], 0
    for i, eid in enumerate(entries, 1):
        d = _j(f"{API}/entry/{eid}/event/{gw}/picks/")
        if not d or "picks" not in d:
            misses += 1
            continue
        chip = d.get("active_chip")
        for p in d["picks"]:
            rows.append((season, gw, eid, int(p["element"]), p.get("position"),
                         p.get("multiplier"), int(bool(p.get("is_captain"))),
                         int(bool(p.get("is_vice_captain"))), chip, now,
                         SOURCE_ID))
        if progress and i % 50 == 0:
            progress(f"    {i}/{len(entries)} squads")
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO acq_manager_pick (season, gw, entry_id, "
            "player_id, slot, multiplier, is_captain, is_vice, active_chip, "
            "observed_utc, source_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    storage.mark(conn, SOURCE_ID, bool(rows), now)
    return {"entries": len(entries), "squads": len(entries) - misses,
            "rows": len(rows), "missing": misses}


def panel_ownership(conn, season: str, gw: int):
    """Per player: panel ownership, start share and captaincy for a gameweek.

    Raw counts only. Turning these into a feature — a differential against
    overall ownership, say — belongs to the modelling engine, not here.
    """
    return conn.execute(
        "SELECT player_id, COUNT(*) owned, "
        "  SUM(CASE WHEN multiplier > 0 THEN 1 ELSE 0 END) started, "
        "  SUM(is_captain) captained, "
        "  (SELECT COUNT(DISTINCT entry_id) FROM acq_manager_pick "
        "     WHERE season=? AND gw=?) panel_size "
        "FROM acq_manager_pick WHERE season=? AND gw=? "
        "GROUP BY player_id ORDER BY owned DESC",
        (season, gw, season, gw)).fetchall()
