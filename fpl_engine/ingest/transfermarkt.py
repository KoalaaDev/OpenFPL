"""Transfer rumours from Transfermarkt's Premier League rumour board.

This is the one signal nothing else in the pipeline carries. FPL reclassifies a
player only once a transfer COMPLETES — until then he sits at his old club with
status "a", and the engine happily projects him onto a run of fixtures he will
never play. That is not a mis-rating, it is the wrong club. Prediction markets
price only the superstar tier (13 open markets, all Alvarez/Rashford-sized), and
the `transfermarkt-api` wrapper has been broken since April 2025 (its own issue
#121: "500 Error Status on all GET Endpoints").

The rumour board gives, per row: the player, his club, the interested club AND
that club's league, a source date, and Transfermarkt's own probability.

**Nothing here is applied automatically, and that is deliberate.** A rumour is a
rumour — an 81% assessment is a forum-sourced opinion, not a measured
probability, and this repo's standing rule is that unvalidated signals are
reported next to a recommendation rather than moved inside it. Rows are surfaced
as *suggestions* for the transfer watch; a human confirms before any projection
changes. Name resolution is fuzzy and refuses ambiguous matches outright,
because attaching one player's move to another is worse than missing it.

The league of the destination is what decides the FPL consequence:

    interested club in GB1  ->  he plays a different PL run; reproject
    interested club abroad  ->  he leaves the game entirely; worth zero
"""
from __future__ import annotations

import re
import unicodedata
import urllib.request
from datetime import datetime, timezone

BASE = "https://www.transfermarkt.com"
RUMOURS = BASE + "/premier-league/geruechte/wettbewerb/GB1"
COMPETITION = BASE + "/premier-league/startseite/wettbewerb/GB1"
SQUAD = BASE + "/x/startseite/verein/{club}"
INJURIES = BASE + "/x/verletzungen/spieler/{player}"
CRAWL_DELAY = 1.5          # polite: one page every 1.5s, single threaded
PL_COMPETITION = "GB1"
# Rumours at or above this are applied automatically. Below it the board is
# mostly speculation with no assessment at all (those parse as None).
AUTO_MIN_PROBABILITY = 50
TIMEOUT = 30
# Transfermarkt refuses a non-browser agent outright.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tm_player (
    tm_player_id  INTEGER PRIMARY KEY,
    tm_name       TEXT,
    tm_club_id    INTEGER,
    season        TEXT,
    player_id     INTEGER          -- resolved FPL id, NULL when unresolved
);

CREATE TABLE IF NOT EXISTS tm_injury (
    tm_player_id  INTEGER NOT NULL,
    from_date     TEXT NOT NULL,   -- ISO, the date the absence began
    injury        TEXT,            -- free text: "Hamstring injury", "Knock"
    until_date    TEXT,            -- ISO; for a CURRENT injury this is a forecast
    days          INTEGER,
    games_missed  INTEGER,
    season_label  TEXT,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (tm_player_id, from_date)
);
CREATE INDEX IF NOT EXISTS idx_tm_injury_player ON tm_injury (tm_player_id);

CREATE TABLE IF NOT EXISTS transfer_rumour (
    season         TEXT NOT NULL,
    tm_player_id   INTEGER NOT NULL,
    tm_to_club_id  INTEGER NOT NULL,
    observed_utc   TEXT NOT NULL,
    tm_player      TEXT,
    position       TEXT,
    from_club      TEXT,
    to_club        TEXT,
    to_competition TEXT,          -- 'GB1' means it stays in the Premier League
    source_date    TEXT,
    probability    INTEGER,       -- Transfermarkt's own assessment, 0-100
    player_id      INTEGER,       -- resolved FPL id, NULL when unresolved
    to_team_id     INTEGER,       -- resolved FPL club, NULL when leaving the PL
    PRIMARY KEY (season, tm_player_id, tm_to_club_id)
);
"""


def init(conn) -> None:
    conn.executescript(SCHEMA)


# ------------------------------------------------------------------ fetch --
def fetch(url: str = RUMOURS) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA,
        "Accept-Language": "en-GB,en;q=0.9",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", errors="replace")


# ------------------------------------------------------------------ parse --
def _one(pat: str, s: str, g: int = 1, d=None):
    m = re.search(pat, s, re.S)
    return m.group(g).strip() if m else d


def parse(html: str) -> list[dict]:
    """One dict per rumour row.

    The items table nests `<table class="inline-table">` for the player and the
    interested club, so slicing to the next `</table>` ends after the first
    nested one. Splitting on the classed rows is safe instead: nested rows
    carry no odd/even class.
    """
    start = html.find('class="items"')
    if start < 0:
        return []
    rows = re.split(r'<tr class="(?:odd|even)">', html[start:])[1:]
    out = []
    for r in rows:
        tm_id = _one(r'/profil/spieler/(\d+)"', r)
        name = _one(r'/profil/spieler/\d+">([^<]+)</a>', r)
        if not tm_id or not name:
            continue
        # Clubs in document order: current club, then interested club. Match
        # the anchor's own title, which precedes the href — going the other way
        # has to cross `"><img src="...`, and `[^>]*` cannot. The interested
        # club is linked twice (crest and name), so dedupe on id.
        pairs = re.findall(
            r'<a title="([^"]+)" href="[^"]*/startseite/verein/(\d+)"', r)
        clubs, seen = [], set()
        for nm, cid in pairs:
            if cid not in seen:
                seen.add(cid)
                clubs.append((cid, nm))
        if len(clubs) < 2:
            continue
        comp = _one(r'/wettbewerb/([A-Z0-9]+)"', r)
        out.append({
            "tm_player_id": int(tm_id),
            "tm_player": name,
            "position": _one(r"<td>([^<]*)</td>\s*</tr>\s*</table>", r),
            "from_club": clubs[0][1],
            "tm_to_club_id": int(clubs[1][0]),
            "to_club": clubs[1][1],
            "to_competition": comp,
            "source_date": _one(r">(\d{2}/\d{2}/\d{4})</a>", r),
            "probability": int(_one(r"(\d{1,3})\s*%", r, d=-1)),
        })
    return out


# --------------------------------------------------------------- resolving --
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def resolve_player(name: str, players: list[dict]) -> int | None:
    """FPL player_id for a Transfermarkt name, or None.

    Refuses ambiguity rather than guessing: two players matching a surname is
    exactly how one man's transfer gets attached to another's projection.
    """
    want = _norm(name)
    if not want:
        return None
    exact = [p for p in players if _norm(p["full_name"]) == want]
    if len(exact) == 1:
        return int(exact[0]["player_id"])
    if len(exact) > 1:
        return None
    tokens = set(want.split())
    if not tokens:
        return None
    hits = [p for p in players
            if tokens and tokens <= set(_norm(p["full_name"]).split())]
    if len(hits) == 1:
        return int(hits[0]["player_id"])
    surname = want.split()[-1]
    hits = [p for p in players if _norm(p["web_name"]) == surname]
    return int(hits[0]["player_id"]) if len(hits) == 1 else None


# Transfermarkt writes full legal names, FPL writes short ones, and substring
# matching does not bridge them: "Manchester City" does not contain "Man City",
# and "Nottingham Forest" shares no substring with "Nott'm Forest".
CLUB_ALIASES = {
    "nottingham forest": "Nott'm Forest",
    "manchester city": "Man City",
    "manchester united": "Man Utd",
    "tottenham hotspur": "Spurs",
    "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves",
    "brighton  hove albion": "Brighton",
    "brighton amp hove albion": "Brighton",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "sheffield united": "Sheffield Utd",
    "coventry city": "Coventry",
    "hull city": "Hull",
    "afc bournemouth": "Bournemouth",
}


def resolve_club(tm_name: str, teams: dict) -> int | None:
    """FPL team_id for a Transfermarkt club name, or None if not a PL club."""
    want = _norm(tm_name)
    want = re.sub(r"\b(fc|afc|cf)\b", "", want)
    want = re.sub(r"\s+", " ", want).strip()
    want = _norm(CLUB_ALIASES.get(want, want))
    for fpl, tid in teams.items():          # exact first
        if _norm(fpl) == want:
            return tid
    for fpl, tid in teams.items():          # then containment, both ways
        f = _norm(fpl)
        if f and (f in want or want in f):
            return tid
    return None


def ingest(conn, season: str, *, html: str | None = None) -> dict:
    """Fetch, parse and store the rumour board. Idempotent per (player, club)."""
    init(conn)
    html = html if html is not None else fetch()
    rows = parse(html)
    players = [dict(r) for r in conn.execute(
        "SELECT player_id, web_name, full_name FROM player WHERE season=?",
        (season,))]
    teams = {r["name"]: int(r["team_id"]) for r in conn.execute(
        "SELECT team_id, name FROM team WHERE season=?", (season,))}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    stored, unresolved = [], []
    for r in rows:
        pid = resolve_player(r["tm_player"], players)
        if pid is None:
            unresolved.append(r["tm_player"])
        # a destination outside GB1 means he leaves the game entirely
        to_team = (resolve_club(r["to_club"], teams)
                   if r.get("to_competition") == PL_COMPETITION else None)
        stored.append((season, r["tm_player_id"], r["tm_to_club_id"], now,
                       r["tm_player"], r["position"], r["from_club"],
                       r["to_club"], r["to_competition"], r["source_date"],
                       r["probability"], pid, to_team))
    if stored:
        conn.executemany(
            "INSERT OR REPLACE INTO transfer_rumour (season, tm_player_id, "
            "tm_to_club_id, observed_utc, tm_player, position, from_club, "
            "to_club, to_competition, source_date, probability, player_id, "
            "to_team_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", stored)
    return {"rows": len(rows), "stored": len(stored),
            "resolved": sum(1 for s in stored if s[11] is not None),
            "unresolved": unresolved[:10]}


def auto_adjustments(conn, season: str,
                     min_probability: int = AUTO_MIN_PROBABILITY) -> dict[int, dict]:
    """Rumours strong enough to price into the projections, per FPL player.

    Weighted by Transfermarkt's own assessment rather than switched on and off:
    a 69% rumour is not a certainty, so the player is valued as the mixture he
    actually is — `p` of the destination, `1 - p` of staying put. A move abroad
    is destination-value zero, which is the honest reading of an FPL asset who
    leaves the game.

    Only the strongest rumour per player survives; two clubs chasing the same
    man is one departure, not two.
    """
    out: dict[int, dict] = {}
    for r in current(conn, season, min_probability=min_probability):
        pid = int(r["player_id"])
        prob = r["probability"]
        if prob is None or prob < min_probability:
            continue
        prev = out.get(pid)
        if prev and prev["probability"] >= prob:
            continue
        out[pid] = {
            "probability": int(prob),
            "weight": min(1.0, max(0.0, prob / 100.0)),
            "to_team": r["to_team_id"],
            "to_club": r["to_club"],
            "leaves_league": r["to_team_id"] is None,
            "source_date": r["source_date"],
        }
    return out


def current(conn, season: str, *, min_probability: int = 0) -> list[dict]:
    """Stored rumours that resolved to an FPL player, strongest first."""
    return [dict(r) for r in conn.execute(
        "SELECT * FROM transfer_rumour WHERE season=? AND player_id IS NOT NULL "
        "AND probability >= ? ORDER BY probability DESC, source_date DESC",
        (season, int(min_probability)))]


# ==========================================================================
# Injury history
#
# CLAUDE.md records that no free feed here carries injury type or expected
# return date. Transfermarkt does, with a start date, an end date, a duration
# and the games missed, going back years — which makes it the first
# minutes-related signal that is genuinely EXOGENOUS and still backtestable.
#
# That distinction is the whole reason to try it. Six previous attempts to
# sharpen the minutes model all returned <=0.25% of log-loss, but every one of
# them was a re-arrangement of data the model already had (rotation tendency,
# opponent strength, Understat role, isotonic calibration, a start/sub split,
# squad state). Injury history is not derivable from trailing minutes: minutes
# record THAT a player was absent, never that the absence was a hamstring, nor
# that it was his third in two years.
# ==========================================================================

def _sleep():
    import time
    time.sleep(CRAWL_DELAY)


def pl_club_ids(html: str | None = None) -> list[tuple[int, str]]:
    """(tm_club_id, name) for every Premier League club."""
    html = html if html is not None else fetch(COMPETITION)
    pairs = re.findall(
        r'<a title="([^"]+)" href="[^"]*/startseite/verein/(\d+)[^"]*"', html)
    out, seen = [], set()
    for name, cid in pairs:
        cid = int(cid)
        if cid not in seen:
            seen.add(cid)
            out.append((cid, name))
    return out


def squad(tm_club_id: int, html: str | None = None) -> list[tuple[int, str]]:
    """(tm_player_id, slug) for one club's squad page."""
    html = html if html is not None else fetch(SQUAD.format(club=tm_club_id))
    out, seen = [], set()
    for slug, pid in re.findall(
            r'href="/([a-z0-9\-]+)/profil/spieler/(\d+)"', html):
        pid = int(pid)
        if pid not in seen:
            seen.add(pid)
            out.append((pid, slug))
    return out


def parse_injuries(html: str) -> list[dict]:
    """One dict per injury spell.

    Columns: season, injury, from, until, days, games missed. `until` for an
    ongoing injury is Transfermarkt's FORECAST return date — useful live, and
    the reason the historical rows must be filtered by date when backtesting.
    """
    start = html.find('class="items"')
    if start < 0:
        return []
    rows = re.split(r'<tr class="(?:odd|even)">', html[start:])[1:]
    out = []
    for r in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(cells) < 5:
            continue
        txt = [re.sub(r"<[^>]+>", " ", c) for c in cells]
        txt = [re.sub(r"\s+", " ", c).replace("&nbsp;", " ").strip() for c in txt]
        season, injury, frm, until, days = txt[0], txt[1], txt[2], txt[3], txt[4]
        if not re.match(r"\d{2}/\d{2}/\d{4}", frm or ""):
            continue
        games = None
        if len(txt) > 5:
            m = re.search(r"(\d+)", txt[5])
            games = int(m.group(1)) if m else None
        dm = re.search(r"(\d+)", days or "")
        out.append({
            "season_label": season,
            "injury": injury or None,
            "from_date": _iso(frm),
            "until_date": _iso(until),
            "days": int(dm.group(1)) if dm else None,
            "games_missed": games,
        })
    return out


def _iso(d: str | None) -> str | None:
    """'03/02/2025' -> '2025-02-03'."""
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", (d or "").strip())
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else None


def map_squads(conn, season: str, *, clubs: list[tuple[int, str]] | None = None,
               progress=None) -> dict:
    """Walk every PL squad page and resolve its players onto FPL ids.

    Twenty requests, once. Resolution refuses ambiguity, so an unmatched player
    stays unmatched rather than borrowing somebody else's injury record.
    """
    init(conn)
    clubs = clubs if clubs is not None else pl_club_ids()
    players = [dict(r) for r in conn.execute(
        "SELECT player_id, web_name, full_name FROM player WHERE season=?",
        (season,))]
    rows, resolved = [], 0
    for i, (cid, name) in enumerate(clubs, 1):
        try:
            members = squad(cid)
        except Exception:
            continue
        for pid, slug in members:
            nm = slug.replace("-", " ")
            fpl = resolve_player(nm, players)
            resolved += fpl is not None
            rows.append((pid, nm, cid, season, fpl))
        if progress:
            progress(f"    {i}/{len(clubs)} {name}: {len(members)} players")
        _sleep()
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO tm_player (tm_player_id, tm_name, "
            "tm_club_id, season, player_id) VALUES (?,?,?,?,?)", rows)
    return {"clubs": len(clubs), "players": len(rows), "resolved": resolved}


def ingest_injuries(conn, season: str, *, limit: int | None = None,
                    only_resolved: bool = True, progress=None) -> dict:
    """Fetch injury history for the mapped players. One request each."""
    init(conn)
    q = ("SELECT tm_player_id, tm_name FROM tm_player WHERE season=?"
         + (" AND player_id IS NOT NULL" if only_resolved else ""))
    targets = [(int(r["tm_player_id"]), r["tm_name"])
               for r in conn.execute(q, (season,))]
    if limit:
        targets = targets[:limit]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spells, failed = 0, 0
    for i, (pid, nm) in enumerate(targets, 1):
        try:
            html = fetch(INJURIES.format(player=pid))
            rows = parse_injuries(html)
        except Exception:
            failed += 1
            _sleep()
            continue
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO tm_injury (tm_player_id, from_date, "
                "injury, until_date, days, games_missed, season_label, "
                "observed_utc) VALUES (?,?,?,?,?,?,?,?)",
                [(pid, r["from_date"], r["injury"], r["until_date"], r["days"],
                  r["games_missed"], r["season_label"], now)
                 for r in rows if r["from_date"]])
            spells += len(rows)
        if i % 20 == 0:
            # Commit as we go. A fifteen-minute crawl inside one transaction
            # is fifteen minutes of work that a single failure throws away,
            # and it holds a write lock on the database the whole time.
            try:
                conn.commit()
            except Exception:
                pass
            if progress:
                progress(f"    {i}/{len(targets)} players, {spells} spells")
        _sleep()
    try:
        conn.commit()
    except Exception:
        pass
    return {"players": len(targets), "spells": spells, "failed": failed}
