"""BBC Sport match data: lineups with positions and per-player stats, and the
live-text stream — keyless, and archived back to at least 2022-23.

WHAT WAS CHECKED AGAINST THE SITE (2026-09-14), not assumed:

* The match pages are assembled from named JSON "containers" under
  ``https://www.bbc.co.uk/wc-data/container/<name>?...``. Three matter here:
  ``sport-data-scores-fixtures`` (a day's fixtures with event URNs and the
  live-page id, works for any past date), ``match-lineups`` (both XIs with
  formation, pitch position, formation slot, captain, and per-player stats —
  minutes, shots, tackles, passes, fouls; xG/xA from 2024-25) and ``stream``
  (the live text, paginated, timestamped per post, from "Lineups are
  announced" through full time).
* ``football-match-preview`` carries head-to-head facts only — no team news
  — and a match's live page does not exist a week before kickoff, so
  nothing here is a PRE-DEADLINE signal. What this source gives is
  ground-truth per-match ROLE (position and slot in the formation) without
  Understat, per-match stats including tackles for seasons before the
  DefCon rule existed, and an in-match injury/substitution narrative with
  timestamps. Those are the three hypotheses it exists to test.
* robots.txt allows ``/sport`` for a generic agent (only ``/sport/alpha/`` is
  disallowed; the blanket disallows apply to named AI crawlers). BBC's terms
  restrict reuse to personal, non-commercial purposes. Requests are paced at
  one per second and every response is stored raw, so a parser can be
  re-run without re-downloading.

Tables (normalised layer): ``acq_bbc_match``, ``acq_bbc_lineup``,
``acq_bbc_stream``. Identity is BBC's own player URN; mapping to FPL ids is
the modelling engine's job (name + club, as for RotoWire), never done here.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from ..core import http
from .. import storage

SOURCE_ID = "bbc"
PARSER_VERSION = "1"
BASE = "https://www.bbc.co.uk/wc-data/container"
TOURNAMENT = "urn:bbc:sportsdata:football:tournament:premier-league"
DELAY = 1.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS acq_bbc_match (
    event_urn     TEXT PRIMARY KEY,
    season        TEXT NOT NULL,
    match_date    TEXT NOT NULL,        -- ISO date (UK)
    kickoff_utc   TEXT,
    live_id       TEXT,
    home          TEXT, away TEXT,
    home_score    INTEGER, away_score INTEGER,
    status        TEXT,
    lineups_done  INTEGER DEFAULT 0,
    stream_pages  INTEGER DEFAULT 0,
    stream_total  INTEGER,
    observed_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acq_bbc_match_season ON acq_bbc_match (season, match_date);
CREATE TABLE IF NOT EXISTS acq_bbc_lineup (
    event_urn     TEXT NOT NULL,
    player_urn    TEXT NOT NULL,
    team          TEXT NOT NULL,
    side          TEXT NOT NULL,        -- home | away
    name_last     TEXT, name_short TEXT, name_first TEXT, display TEXT,
    position      TEXT,                 -- Goalkeeper/Defender/Midfielder/Forward/Substitute
    formation     TEXT,                 -- e.g. 4-2-3-1
    formation_place INTEGER,            -- 1..11 slot in that formation (starters)
    pitch_row     INTEGER,              -- row in the formation graphic, 0 = GK
    is_starter    INTEGER NOT NULL,
    is_captain    INTEGER,
    shirt         INTEGER,
    mins          INTEGER,
    stats_json    TEXT,                 -- every per-player stat BBC shipped
    raw_id        INTEGER,
    PRIMARY KEY (event_urn, player_urn)
);
CREATE TABLE IF NOT EXISTS acq_bbc_stream (
    event_urn     TEXT NOT NULL,
    post_urn      TEXT NOT NULL,
    published_utc TEXT,
    minute        TEXT,
    text          TEXT,
    raw_id        INTEGER,
    PRIMARY KEY (event_urn, post_urn)
);
CREATE INDEX IF NOT EXISTS idx_acq_bbc_stream_time ON acq_bbc_stream (event_urn, published_utc);
-- every competition a Premier League club plays in: the club's REAL calendar
CREATE TABLE IF NOT EXISTS acq_bbc_calendar (
    event_urn     TEXT PRIMARY KEY,
    season        TEXT NOT NULL,
    match_date    TEXT NOT NULL,
    kickoff_utc   TEXT,
    competition   TEXT,
    home          TEXT, away TEXT,
    home_score    INTEGER, away_score INTEGER,
    status        TEXT,
    observed_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acq_bbc_calendar_season ON acq_bbc_calendar (season, match_date);
-- the crowd's eye test: BBC's player rater (average out of 10 per player per
-- match, closed 30 minutes after full time), server-rendered into the match
-- page's report tab from 2024-25 on
CREATE TABLE IF NOT EXISTS acq_bbc_rating (
    event_urn     TEXT NOT NULL,
    player_urn    TEXT NOT NULL,
    team          TEXT,
    short_name    TEXT,
    shirt         INTEGER,
    is_starter    INTEGER,
    rating        REAL,
    state         TEXT,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (event_urn, player_urn)
);
-- per-team match statistics (the `match-stats` container): possession,
-- shots, corners, box touches, crosses, tackles, clearances and Opta's
-- expected goals/assists split into open play and set play (2024-25 on)
CREATE TABLE IF NOT EXISTS acq_bbc_match_stats (
    event_urn     TEXT NOT NULL,
    team          TEXT NOT NULL,
    side          TEXT,
    possession    REAL,
    shots         INTEGER,
    shots_on      INTEGER,
    shots_blocked INTEGER,
    corners       INTEGER,
    touches_box   INTEGER,
    crosses       INTEGER,
    fouls         INTEGER,
    tackles       INTEGER,
    clearances    INTEGER,
    xg            REAL,
    xg_open       REAL,
    xg_set        REAL,
    xa            REAL,
    xa_open       REAL,
    xa_set        REAL,
    distance_km   REAL,
    sprint_pct    REAL,
    stats_json    TEXT,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (event_urn, team)
);
-- match officials, from the lineups payload (the appointment is public
-- before the deadline; only the referee's card rate is modelled)
CREATE TABLE IF NOT EXISTS acq_bbc_official (
    event_urn     TEXT NOT NULL,
    official_urn  TEXT NOT NULL,
    role          TEXT,
    name          TEXT,
    PRIMARY KEY (event_urn, official_urn)
);
"""

COLLATED = "urn:bbc:sportsdata:football:tournament-collection:collated"
PL_CLUBS = {  # BBC full names of every club that has been in the league since 2022-23
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton & Hove Albion",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich Town",
    "Leeds United", "Leicester City", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Newcastle United", "Nottingham Forest", "Sheffield United",
    "Southampton", "Sunderland", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers", "Brighton", "Ipswich", "Leeds", "Leicester", "Luton",
    "Sheff Utd", "Tottenham", "Wolves", "Man City", "Man Utd", "Newcastle",
    "Nottm Forest", "West Ham",
}


def register(conn) -> None:
    conn.executescript(SCHEMA)
    storage.register_source(
        conn, SOURCE_ID, name="BBC Sport match containers (lineups, live text)",
        base_url=BASE, source_type="api", enabled=1,
        robots_policy="/sport allowed for generic agents; /sport/alpha/ disallowed",
        terms_note="personal, non-commercial use per BBC terms; owner's decision",
        request_delay=DELAY, parser_version=PARSER_VERSION)


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------

def fixtures_url(day: date) -> str:
    d = day.isoformat()
    return (f"{BASE}/sport-data-scores-fixtures?selectedEndDate={d}"
            f"&selectedStartDate={d}&todayDate={date.today().isoformat()}"
            f"&urn={TOURNAMENT.replace(':', '%3A')}&useSdApi=false")


def calendar_url(day: date) -> str:
    d = day.isoformat()
    return (f"{BASE}/sport-data-scores-fixtures?selectedEndDate={d}"
            f"&selectedStartDate={d}&todayDate={date.today().isoformat()}"
            f"&urn={COLLATED.replace(':', '%3A')}&useSdApi=false")


def parse_calendar(payload: dict) -> list[dict]:
    """Every event on the day that involves a Premier League club, with the
    competition it belongs to (Champions League, FA Cup, ...)."""
    out = []
    for o in _walk(payload):
        urn = o.get("urn")
        if not (isinstance(urn, str) and ":event:" in urn and "home" in o and "away" in o):
            continue
        h, a = o["home"], o["away"]
        if h.get("fullName") not in PL_CLUBS and a.get("fullName") not in PL_CLUBS:
            continue
        t = o.get("tournament") or {}
        out.append({
            "event_urn": urn, "kickoff_utc": o.get("startDateTime"),
            "match_date": (o.get("date") or {}).get("isoDate") or (o.get("startDateTime") or "")[:10],
            "competition": (t.get("name") if isinstance(t, dict) else None)
                           or o.get("eventGroupingLabel"),
            "home": h.get("fullName"), "away": a.get("fullName"),
            "home_score": h.get("score"), "away_score": a.get("score"),
            "status": o.get("status"),
        })
    return out


def collect_calendar_day(conn, day: date) -> int:
    resp, payload = _fetch_json(calendar_url(day))
    if payload is None:
        return -1
    storage.store_raw(conn, SOURCE_ID, resp, parser_version=PARSER_VERSION, keep_payload=False)
    now = http.utcnow()
    rows = parse_calendar(payload)
    for ev in rows:
        conn.execute(
            "INSERT INTO acq_bbc_calendar (event_urn, season, match_date, kickoff_utc, "
            "competition, home, away, home_score, away_score, status, observed_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(event_urn) DO UPDATE SET "
            "home_score=excluded.home_score, away_score=excluded.away_score, "
            "status=excluded.status, kickoff_utc=excluded.kickoff_utc, "
            "observed_utc=excluded.observed_utc",
            (ev["event_urn"], season_of(day), ev["match_date"], ev["kickoff_utc"],
             ev["competition"], ev["home"], ev["away"], ev["home_score"],
             ev["away_score"], ev["status"], now))
    return len(rows)


def backfill_calendar(conn, season: str, *, ahead_days: int = 45, progress=print) -> dict:
    """Every day of the season, including the weeks AHEAD of today so the
    minutes model can see a club's next fixtures in every competition."""
    register(conn)
    y = int(season[:4])
    start = date(y, 8, 1)
    stop = min(date(y + 1, 6, 10), date.today() + timedelta(days=ahead_days))
    out = {"season": season, "days": 0, "events": 0, "errors": 0}
    d = start
    while d <= stop:
        n = collect_calendar_day(conn, d)
        out["days"] += 1
        if n < 0:
            out["errors"] += 1
        else:
            out["events"] += n
        if out["days"] % 30 == 0:
            conn.commit()
            progress(f"  {season} calendar to {d}: {out['events']} club-matches")
        d += timedelta(days=1)
    conn.commit()
    return out


def live_page_url(live_id: str) -> str:
    return f"https://www.bbc.co.uk/sport/football/live/{live_id}"


def parse_ratings(html: str) -> tuple[list[dict], str | None]:
    """The `playerRater` block embedded in the page's initial data.

    It is not served by any container: the averages are rendered into the
    report tab, so the page itself is the source. Returns (rows, state)."""
    import re as _re
    m = _re.search(r'window\.__INITIAL_DATA__="(.*?)";</script>', html, _re.S)
    if not m:
        return [], None
    raw = m.group(1).encode().decode("unicode_escape", errors="ignore")
    try:
        d = json.loads(raw)
    except ValueError:
        try:
            d = json.loads(raw.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore"))
        except ValueError:
            return [], None
    for o in _walk(d):
        if isinstance(o, dict) and o.get("type") == "playerRater":
            pr = (o.get("model") or {}).get("playerRaterData") or {}
            rows = []
            for side in ("homeTeam", "awayTeam"):
                t = pr.get(side) or {}
                for grp, starter in (("starters", 1), ("substitutes", 0)):
                    for pl in (t.get("players") or {}).get(grp) or []:
                        pid = pl.get("id")
                        if not pid:
                            continue
                        rows.append({
                            "player_urn": f"urn:bbc:sportsdata:football:player:{pid}",
                            "team": t.get("fullName"), "short_name": pl.get("shortName"),
                            "shirt": pl.get("shirtNumber"), "is_starter": starter,
                            "rating": pl.get("averageRating"),
                        })
            return rows, pr.get("state")
    return [], None


def collect_ratings(conn, event_urn: str, live_id: str) -> int:
    """One page fetch; -1 on failure, else rows stored (0 = no rater)."""
    resp = http.get(live_page_url(live_id), delay=DELAY)
    if not resp.ok:
        return -1
    rows, state = parse_ratings(resp.text)
    now = http.utcnow()
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO acq_bbc_rating (event_urn, player_urn, team, short_name, "
            "shirt, is_starter, rating, state, observed_utc) VALUES (?,?,?,?,?,?,?,?,?)",
            (event_urn, r["player_urn"], r["team"], r["short_name"], r["shirt"],
             r["is_starter"], r["rating"], state, now))
    conn.execute("UPDATE acq_bbc_match SET ratings_done=? WHERE event_urn=?",
                 (1 if rows or state else 2, event_urn))     # 2 = page has no rater
    return len(rows)


def backfill_ratings(conn, season: str, *, progress=print) -> dict:
    register(conn)
    try:
        conn.execute("ALTER TABLE acq_bbc_match ADD COLUMN ratings_done INTEGER DEFAULT 0")
    except Exception:  # noqa: BLE001 - already there
        pass
    todo = conn.execute(
        "SELECT event_urn, live_id, match_date FROM acq_bbc_match WHERE season=? AND "
        "status='PostEvent' AND live_id IS NOT NULL AND COALESCE(ratings_done,0)=0 "
        "ORDER BY match_date", (season,)).fetchall()
    out = {"season": season, "matches": len(todo), "rated": 0, "rows": 0, "no_rater": 0, "errors": 0}
    for i, (urn, live_id, d) in enumerate(todo, 1):
        n = collect_ratings(conn, urn, live_id)
        if n < 0:
            out["errors"] += 1
        elif n == 0:
            out["no_rater"] += 1
        else:
            out["rated"] += 1
            out["rows"] += n
        if i % 40 == 0:
            conn.commit()
            progress(f"  {season} ratings to {d}: {out['rated']} rated, {out['no_rater']} without")
    conn.commit()
    return out


def stats_url(event_urn: str) -> str:
    return f"{BASE}/match-stats?urn={event_urn.replace(':', '%3A')}"


def _stat(d: dict, *path):
    for k in path:
        d = (d or {}).get(k)
        if d is None:
            return None
    return d.get("total") if isinstance(d, dict) else d


def parse_stats(payload: dict) -> list[dict]:
    """One row per team from the `match-stats` container."""
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = payload.get(side) or {}
        st = t.get("stats") or {}
        name = ((t.get("name") or {}).get("fullName")) if isinstance(t.get("name"), dict) else t.get("name")
        if not name:
            continue
        dist = _stat(st, "fitness", "distanceMetres")
        rows.append({
            "team": name, "side": "home" if side == "homeTeam" else "away",
            "possession": _stat(st, "possessionPercentage"),
            "shots": _stat(st, "attack", "shotsTotal") or _stat(st, "shotsTotal"),
            "shots_on": _stat(st, "attack", "shotsOnTarget") or _stat(st, "shotsOnTarget"),
            "shots_blocked": _stat(st, "attack", "shotsBlocked"),
            "corners": _stat(st, "cornersWon"),
            "touches_box": _stat(st, "distribution", "touchesInBox") or _stat(st, "touchesInBox"),
            "crosses": _stat(st, "distribution", "totalCross"),
            "fouls": _stat(st, "defence", "foulsCommitted") or _stat(st, "foulsCommitted"),
            "tackles": _stat(st, "defence", "totalTackle"),
            "clearances": _stat(st, "defence", "totalClearance"),
            "xg": _stat(st, "expected", "goals"),
            "xg_open": _stat(st, "expected", "goalsOpenplay"),
            "xg_set": _stat(st, "expected", "goalsSetplay"),
            "xa": _stat(st, "expected", "assists"),
            "xa_open": _stat(st, "expected", "assistsOpenplay"),
            "xa_set": _stat(st, "expected", "assistsSetplay"),
            "distance_km": dist,
            "sprint_pct": _stat(st, "fitness", "sprintingPercentage"),
            "stats_json": json.dumps(st, separators=(",", ":")),
        })
    return rows


def parse_officials(payload: dict) -> list[dict]:
    out = []
    for o in payload.get("officials") or []:
        if not isinstance(o, dict) or not o.get("id"):
            continue
        name = " ".join(x for x in (o.get("firstName"), o.get("lastName")) if x)
        out.append({"official_urn": o["id"], "role": o.get("type"), "name": name})
    return out


def _store_officials(conn, event_urn: str, payload: dict) -> int:
    rows = parse_officials(payload)
    for o in rows:
        conn.execute("INSERT OR REPLACE INTO acq_bbc_official (event_urn, official_urn, role, name) "
                     "VALUES (?,?,?,?)", (event_urn, o["official_urn"], o["role"], o["name"]))
    return len(rows)


def collect_stats(conn, event_urn: str) -> int:
    """One request; -1 on failure, else rows stored (0 = no stats served)."""
    resp, payload = _fetch_json(stats_url(event_urn))
    if payload is None:
        return -1
    storage.store_raw(conn, SOURCE_ID, resp, parser_version=PARSER_VERSION)
    rows = parse_stats(payload)
    now = http.utcnow()
    for r in rows:
        conn.execute(
            "INSERT OR REPLACE INTO acq_bbc_match_stats (event_urn, team, side, possession, shots, "
            "shots_on, shots_blocked, corners, touches_box, crosses, fouls, tackles, clearances, "
            "xg, xg_open, xg_set, xa, xa_open, xa_set, distance_km, sprint_pct, stats_json, observed_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_urn, r["team"], r["side"], r["possession"], r["shots"], r["shots_on"],
             r["shots_blocked"], r["corners"], r["touches_box"], r["crosses"], r["fouls"],
             r["tackles"], r["clearances"], r["xg"], r["xg_open"], r["xg_set"], r["xa"],
             r["xa_open"], r["xa_set"], r["distance_km"], r["sprint_pct"], r["stats_json"], now))
    conn.execute("UPDATE acq_bbc_match SET stats_done=? WHERE event_urn=?",
                 (1 if rows else 2, event_urn))
    return len(rows)


def backfill_stats(conn, season: str, *, progress=print) -> dict:
    register(conn)
    try:
        conn.execute("ALTER TABLE acq_bbc_match ADD COLUMN stats_done INTEGER DEFAULT 0")
    except Exception:  # noqa: BLE001 - already there
        pass
    todo = conn.execute(
        "SELECT event_urn, match_date FROM acq_bbc_match WHERE season=? AND status='PostEvent' "
        "AND COALESCE(stats_done,0)=0 ORDER BY match_date", (season,)).fetchall()
    out = {"season": season, "matches": len(todo), "with_stats": 0, "rows": 0, "empty": 0, "errors": 0}
    for i, (urn, d) in enumerate(todo, 1):
        n = collect_stats(conn, urn)
        if n < 0:
            out["errors"] += 1
        elif n == 0:
            out["empty"] += 1
        else:
            out["with_stats"] += 1
            out["rows"] += n
        conn.commit()           # per match: other writers share this file
        if i % 40 == 0:
            progress(f"  {season} stats to {d}: {out['with_stats']} with stats, {out['empty']} empty")
    conn.commit()
    return out


def backfill_officials_from_raw(conn) -> dict:
    """Officials for every archived lineups payload — no network."""
    import urllib.parse as _up
    register(conn)
    rows = conn.execute(
        "SELECT url, payload FROM acq_raw_document WHERE url LIKE '%match-lineups%'").fetchall()
    out = {"payloads": len(rows), "officials": 0, "events": 0}
    for url, payload in rows:
        q = _up.parse_qs(_up.urlparse(url).query)
        urn = (q.get("urn") or [None])[0]
        if not urn:
            continue
        try:
            body = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
            d = json.loads(body)
        except (ValueError, AttributeError, UnicodeDecodeError):
            continue
        n = _store_officials(conn, urn, d)
        out["officials"] += n
        out["events"] += 1 if n else 0
    conn.commit()
    return out


def lineups_url(event_urn: str) -> str:
    return (f"{BASE}/match-lineups?formationGraphicVisible=true"
            f"&urn={event_urn.replace(':', '%3A')}")


def stream_url(event_id: str, page: int, live_id: str) -> str:
    # `type=football` is load-bearing: without it the endpoint answers an
    # empty 200 rather than an error
    return (f"{BASE}/stream?globalContainerPolling=true&liveTextStreamId={event_id}"
            f"&pageNumber={page}&pageSize=20"
            f"&pageUrl=%2Fsport%2Ffootball%2Flive%2F{live_id}&type=football")


# --------------------------------------------------------------------------
# parsers (pure; tested on stored payloads)
# --------------------------------------------------------------------------

def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def parse_fixtures(payload: dict) -> list[dict]:
    """Every Premier League event on the day: urn, teams, scores, status."""
    out = []
    for o in _walk(payload):
        urn = o.get("urn")
        if not (isinstance(urn, str) and ":event:" in urn and "home" in o and "away" in o):
            continue
        h, a = o["home"], o["away"]
        link = o.get("onwardJourneyLink") or ""
        out.append({
            "event_urn": urn, "event_id": o.get("id") or urn.rsplit(":", 1)[-1],
            "kickoff_utc": o.get("startDateTime"),
            "match_date": (o.get("date") or {}).get("isoDate")
                          or (o.get("startDateTime") or "")[:10],
            "live_id": link.rsplit("/", 1)[-1] if "/live/" in link else None,
            "home": h.get("fullName"), "away": a.get("fullName"),
            "home_score": h.get("score"), "away_score": a.get("score"),
            "status": o.get("status"),
        })
    return out


def _team_rows(side: str, team: dict) -> list[dict]:
    formation = ((team.get("formation") or {}).get("value") or "").replace(" ", "")
    rows = []
    # pitch row from the formation graphic: GK row 0, then defence …
    row_of = {}
    for r_i, row in enumerate(team.get("pitchLayout") or []):
        for p in row or []:
            if isinstance(p, dict) and p.get("urn"):
                row_of[p["urn"]] = r_i
    for grp, starter in (("starters", 1), ("substitutes", 0)):
        for p in (team.get("players") or {}).get(grp) or []:
            nm = p.get("name") or {}
            stats = {s.get("dataField"): s.get("statValue")
                     for s in p.get("stats") or [] if s.get("dataField")}
            mins = stats.get("minsPlayed")
            rows.append({
                "player_urn": p.get("urn"), "team": team.get("name", {}).get("fullName"),
                "side": side, "name_last": nm.get("last"), "name_short": nm.get("short"),
                "name_first": nm.get("first"), "display": p.get("displayName"),
                "position": p.get("position"), "formation": formation,
                "formation_place": p.get("formationPlace"),
                "pitch_row": row_of.get(p.get("urn")),
                "is_starter": starter, "is_captain": int(bool(p.get("isCaptain"))),
                "shirt": p.get("shirtNumber"),
                "mins": int(mins) if isinstance(mins, (int, float)) else None,
                "stats_json": json.dumps(stats, separators=(",", ":")),
            })
    return rows


def parse_lineups(payload: dict) -> list[dict]:
    d = payload.get("data", payload)
    out = []
    for side in ("homeTeam", "awayTeam"):
        t = d.get(side)
        if t:
            out += _team_rows(side.replace("Team", ""), t)
    return [r for r in out if r["player_urn"]]


def _post_text(o) -> str:
    parts = []
    for b in _walk(o):
        if b.get("type") == "paragraph" and isinstance(b.get("model"), dict):
            t = b["model"].get("text")
            if t:
                parts.append(t)
    return " ".join(parts).strip()


def parse_stream(payload: dict) -> tuple[list[dict], int]:
    """(posts, total_pages)."""
    d = payload.get("data", payload)
    total = int((d.get("page") or {}).get("total") or 1)
    posts = []
    for r in d.get("results") or []:
        dates = r.get("dates") or {}
        posts.append({
            "post_urn": r.get("urn") or f"{dates.get('firstPublished')}|{_post_text(r.get('content'))[:40]}",
            "published_utc": dates.get("firstPublished") or r.get("unformattedTime"),
            "minute": dates.get("time"),
            "text": _post_text(r.get("content")),
        })
    return posts, total


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def season_of(day: date) -> str:
    y = day.year if day.month >= 7 else day.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def season_days(season: str):
    y = int(season[:4])
    start, stop = date(y, 8, 1), min(date(y + 1, 6, 10), date.today())
    d = start
    while d <= stop:
        yield d
        d += timedelta(days=1)


def _fetch_json(url: str):
    resp = http.get(url, delay=DELAY)
    if not resp.ok:
        return resp, None
    try:
        return resp, json.loads(resp.text)
    except ValueError:
        return resp, None


def collect_day(conn, day: date, *, dry_run: bool = False) -> dict:
    """Fixtures for one day, then lineups + full stream for each finished match
    not already complete in the database. Returns counts."""
    now = http.utcnow()
    resp, payload = _fetch_json(fixtures_url(day))
    if payload is None:
        return {"day": day.isoformat(), "error": resp.error or resp.status}
    events = parse_fixtures(payload)
    counts = {"day": day.isoformat(), "events": len(events), "lineups": 0,
              "stream_posts": 0, "skipped": 0}
    if dry_run:
        return counts
    raw_id = storage.store_raw(conn, SOURCE_ID, resp, parser_version=PARSER_VERSION)
    for ev in events:
        conn.execute(
            "INSERT INTO acq_bbc_match (event_urn, season, match_date, kickoff_utc, "
            "live_id, home, away, home_score, away_score, status, observed_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(event_urn) DO UPDATE SET "
            "home_score=excluded.home_score, away_score=excluded.away_score, "
            "status=excluded.status, live_id=COALESCE(excluded.live_id, live_id), "
            "observed_utc=excluded.observed_utc",
            (ev["event_urn"], season_of(day), ev["match_date"], ev["kickoff_utc"],
             ev["live_id"], ev["home"], ev["away"], ev["home_score"],
             ev["away_score"], ev["status"], now))
        if ev["status"] != "PostEvent":
            continue                                # not finished: nothing to archive yet
        row = conn.execute("SELECT lineups_done, stream_pages, stream_total FROM "
                           "acq_bbc_match WHERE event_urn=?", (ev["event_urn"],)).fetchone()
        if row and row[0] and row[2] is not None and row[1] >= row[2]:
            counts["skipped"] += 1
            continue
        if not (row and row[0]):
            r2, lu = _fetch_json(lineups_url(ev["event_urn"]))
            if lu is not None:
                rid = storage.store_raw(conn, SOURCE_ID, r2, parser_version=PARSER_VERSION)
                rows = parse_lineups(lu)
                for p in rows:
                    conn.execute(
                        "INSERT OR REPLACE INTO acq_bbc_lineup (event_urn, player_urn, "
                        "team, side, name_last, name_short, name_first, display, position, "
                        "formation, formation_place, pitch_row, is_starter, is_captain, "
                        "shirt, mins, stats_json, raw_id) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ev["event_urn"], p["player_urn"], p["team"], p["side"],
                         p["name_last"], p["name_short"], p["name_first"], p["display"],
                         p["position"], p["formation"], p["formation_place"],
                         p["pitch_row"], p["is_starter"], p["is_captain"], p["shirt"],
                         p["mins"], p["stats_json"], rid))
                counts["lineups"] += len(rows)
                _store_officials(conn, ev["event_urn"], lu)
                conn.execute("UPDATE acq_bbc_match SET lineups_done=1 WHERE event_urn=?",
                             (ev["event_urn"],))
        if ev["live_id"]:
            page = (row[1] if row else 0) + 1
            total = row[2] if row and row[2] else None
            while total is None or page <= total:
                r3, st = _fetch_json(stream_url(ev["event_id"], page, ev["live_id"]))
                if st is None:
                    break
                rid = storage.store_raw(conn, SOURCE_ID, r3, parser_version=PARSER_VERSION)
                posts, total = parse_stream(st)
                for q in posts:
                    conn.execute(
                        "INSERT OR REPLACE INTO acq_bbc_stream (event_urn, post_urn, "
                        "published_utc, minute, text, raw_id) VALUES (?,?,?,?,?,?)",
                        (ev["event_urn"], q["post_urn"], q["published_utc"], q["minute"],
                         q["text"], rid))
                counts["stream_posts"] += len(posts)
                conn.execute("UPDATE acq_bbc_match SET stream_pages=?, stream_total=? "
                             "WHERE event_urn=?", (page, total, ev["event_urn"]))
                page += 1
        conn.commit()
    storage.mark(conn, SOURCE_ID, True, now)
    return counts


def pull(conn, *, season: str | None = None, dry_run: bool = False) -> dict:
    """Scheduled: the last eight days (results settle, streams complete),
    then the crowd ratings for any finished match not yet rated."""
    register(conn)
    today = date.today()
    out = {"days": 0, "events": 0, "lineups": 0, "stream_posts": 0, "ratings": 0}
    for k in range(8, -1, -1):
        c = collect_day(conn, today - timedelta(days=k), dry_run=dry_run)
        out["days"] += 1
        for key in ("events", "lineups", "stream_posts"):
            out[key] += c.get(key, 0)
    if not dry_run:
        r = backfill_ratings(conn, season or season_of(today), progress=lambda m: None)
        out["ratings"] = r.get("rows", 0)
        st = backfill_stats(conn, season or season_of(today), progress=lambda m: None)
        out["stats"] = st.get("rows", 0)
    return out


def backfill(conn, season: str, *, progress=print) -> dict:
    """Every day of a past season. Days with no fixtures cost one request;
    ~380 matches x (1 lineup + ~5 stream pages) is under an hour at 1 req/s."""
    register(conn)
    out = {"season": season, "days": 0, "events": 0, "lineups": 0,
           "stream_posts": 0, "skipped": 0}
    for day in season_days(season):
        c = collect_day(conn, day)
        out["days"] += 1
        for key in ("events", "lineups", "stream_posts", "skipped"):
            out[key] += c.get(key, 0)
        if c.get("events"):
            progress(f"  {day}: {c}")
    conn.commit()
    return out
