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

def _sleep(seconds: float | None = None):
    import time
    time.sleep(CRAWL_DELAY if seconds is None else seconds)


# The two ceapi endpoints answer with a few kilobytes of JSON rather than a
# 180KB rendered page, so they are paced faster than the HTML crawl.
API_DELAY = 0.9


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


# ==========================================================================
# Squad detail, transfer history and market-value history
#
# Three datasets that are *dated* and therefore backtestable, which is what
# separates them from FPL's `status` (a snapshot with no history) and from the
# rumour board (an opinion about the future).
#
#   /kader/verein/{club}/saison_id/{y}/plus/1
#       one request per club-season: date of birth, height, foot, Transfermarkt's
#       detailed position (Left Winger, not "MID"), the date he joined, the club
#       he joined from and the fee, the contract expiry and today's market value.
#
#   /ceapi/transferHistory/list/{player}      JSON, dated, fee + market value
#   /ceapi/marketValueDevelopment/graph/{player}   JSON, every valuation dated
#
# WHAT IS AND IS NOT POINT-IN-TIME. The two ceapi feeds carry a date on every
# row, so a feature built from them can be filtered to "strictly before this
# kickoff" exactly like the injury spells. The squad page cannot: it serves
# TODAY's contract expiry and market value even when asked for saison_id 2023,
# and it even backdates a player's current joined-date onto the old squad (Raya
# reads "04/07/2024" on Arsenal's 2023-24 page, when he was on loan). Only the
# immutable attributes off that page — date of birth, height, foot — plus the
# name→id mapping are safe to use historically. Contract expiry is live-only,
# on the same footing as FPL's `status`.
# ==========================================================================

SQUAD_DETAIL = BASE + "/x/kader/verein/{club}/saison_id/{season}/plus/1"
COMPETITION_SEASON = (BASE + "/premier-league/startseite/wettbewerb/GB1"
                             "/plus/?saison_id={season}")
MV_GRAPH = BASE + "/ceapi/marketValueDevelopment/graph/{player}"
TRANSFER_LIST = BASE + "/ceapi/transferHistory/list/{player}"

SCHEMA_2 = """
CREATE TABLE IF NOT EXISTS tm_squad (
    season          TEXT NOT NULL,      -- FPL season label, e.g. '2025-26'
    tm_player_id    INTEGER NOT NULL,
    tm_club_id      INTEGER,
    tm_name         TEXT,
    shirt           INTEGER,
    detail_position TEXT,               -- 'Left Winger', 'Defensive Midfield'
    dob             TEXT,               -- ISO; immutable, safe historically
    height_cm       INTEGER,            -- immutable
    foot            TEXT,               -- immutable
    joined_date     TEXT,               -- LIVE-ONLY: backdated on old pages
    signed_from     TEXT,
    signed_from_id  INTEGER,
    signed_fee_eur  INTEGER,
    contract_until  TEXT,               -- LIVE-ONLY: today's contract
    market_value    INTEGER,            -- LIVE-ONLY: use tm_market_value instead
    observed_utc    TEXT NOT NULL,
    PRIMARY KEY (season, tm_player_id)
);

CREATE TABLE IF NOT EXISTS tm_transfer (
    tm_player_id  INTEGER NOT NULL,
    transfer_date TEXT NOT NULL,        -- ISO, dated: point-in-time safe
    to_club_id    INTEGER NOT NULL,
    to_club       TEXT,
    from_club_id  INTEGER,
    from_club     TEXT,
    fee_eur       INTEGER,              -- NULL for loans and undisclosed fees
    fee_text      TEXT,
    value_eur     INTEGER,              -- his market value at the time
    season_label  TEXT,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (tm_player_id, transfer_date, to_club_id)
);
CREATE INDEX IF NOT EXISTS idx_tm_transfer_player ON tm_transfer (tm_player_id);

CREATE TABLE IF NOT EXISTS tm_market_value (
    tm_player_id  INTEGER NOT NULL,
    value_date    TEXT NOT NULL,        -- ISO, dated: point-in-time safe
    value_eur     INTEGER,
    club          TEXT,
    age           INTEGER,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (tm_player_id, value_date)
);
CREATE INDEX IF NOT EXISTS idx_tm_mv_player ON tm_market_value (tm_player_id);
"""


def init2(conn) -> None:
    conn.executescript(SCHEMA_2)
    # `tm_player` predates the cross-season work and carries only a per-season
    # FPL element id. Those are REASSIGNED every season, so joining one to a
    # different season's `player` table silently hands one man another's
    # history — the Round 9 class of defect. `player_code` is stable.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tm_player)")}
    if cols and "player_code" not in cols:
        conn.execute("ALTER TABLE tm_player ADD COLUMN player_code INTEGER")


def tm_season(season: str) -> int:
    """'2024-25' -> 2024, Transfermarkt's saison_id for that season."""
    return int(str(season)[:4])


# --------------------------------------------------------------- parsing --
_MONEY = re.compile(r"€\s*([\d.,]+)\s*(bn|m|k)?", re.I)


def money_eur(text: str | None) -> int | None:
    """'€31.90m' -> 31900000, '€800k' -> 800000, 'free transfer' -> 0.

    Returns None for a loan, an undisclosed fee or a missing value — which is
    NOT the same as zero, and collapsing the two would tell the model a £60m
    signing arrived for nothing.
    """
    s = (text or "").strip().lower()
    if not s or s in ("-", "?", "n/a"):
        return None
    if "free" in s:
        return 0
    m = _MONEY.search(s)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return int(v * {"bn": 1e9, "m": 1e6, "k": 1e3}.get(
        (m.group(2) or "").lower(), 1))


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).replace(
        "&nbsp;", " ").strip()


def parse_squad(html: str) -> list[dict]:
    """One dict per squad member of a `kader/.../plus/1` page.

    Parsed by MEANING, not by column index. The layout is not stable across
    seasons — the current page carries a contract-expiry column that the
    2023-24 one does not, so a positional parser reads market value as a
    contract date on every historical page and never raises.
    """
    start = html.find('class="items"')
    if start < 0:
        return []
    rows = re.split(r'<tr class="(?:odd|even)">', html[start:])[1:]
    out = []
    for r in rows:
        pid = _one(r"/profil/spieler/(\d+)", r)
        # an injured player carries an icon span straight after his name, so
        # anchoring the name on `</a>` silently drops exactly the rows a
        # minutes model cares most about
        name = _one(r'/profil/spieler/\d+"[^>]*>\s*([^<]+?)\s*<', r)
        if not pid or not name:
            continue
        # the inline table's second row is Transfermarkt's detailed position
        pos = _one(r"<tr>\s*<td>\s*([^<]+?)\s*</td>\s*</tr>\s*</table>", r)
        fee_title = _one(r'<a title="([^"]*?):\s*Abl[^"]*?"', r)
        signed_id = int(_one(
            r'<a title="[^"]*?Abl[^"]*?" href="[^"]*/verein/(\d+)', r, d=0)) or None
        signed_fee = money_eur(_one(r'title="[^"]*?Abl\S*\s*([^"]*)"', r))
        # A new signing's crest carries `title="Joined from X; date: 09/07/2026;
        # fee: ..."`, which repeats the joined date inside an attribute. Left in,
        # it becomes the second bare date and is read as the contract expiry —
        # so every summer signing's contract reads as expiring the day he
        # arrived. Attributes are stripped before the dates are collected.
        body = re.sub(r'\s(?:title|alt)="[^"]*"', " ", r)
        dates = re.findall(r"(\d{2}/\d{2}/\d{4})", body)
        dob = _one(r"(\d{2}/\d{2}/\d{4})\s*\(\s*\d+\s*\)", body)
        rest = [d for d in dates if d != dob]
        hm = re.search(r"(\d),(\d{2})\s*m", body)
        out.append({
            "tm_player_id": int(pid),
            "tm_name": name,
            "shirt": int(_one(r"<div class=rn_nummer>\s*(\d+)", r, d=0)) or None,
            "detail_position": pos,
            "dob": _iso(dob),
            "height_cm": int(hm.group(1)) * 100 + int(hm.group(2)) if hm else None,
            "foot": _one(r">\s*(right|left|both)\s*<", body),
            # dates after the date of birth: joined, then contract expiry when
            # the page carries that column at all
            "joined_date": _iso(rest[0]) if rest else None,
            "contract_until": _iso(rest[1]) if len(rest) > 1 else None,
            "signed_from": fee_title,
            "signed_from_id": signed_id,
            "signed_fee_eur": signed_fee,
            "market_value": money_eur(_one(
                r'/marktwertverlauf/spieler/\d+">\s*([^<]+?)\s*</a>', r)),
        })
    return out


def parse_market_values(payload: str) -> list[dict]:
    """`ceapi/marketValueDevelopment/graph` -> dated valuations."""
    import json as _json
    try:
        data = _json.loads(payload)
    except Exception:
        return []
    out = []
    for row in data.get("list") or []:
        d = _iso(row.get("datum_mw"))
        if not d:
            continue
        try:
            v = int(row.get("y") or 0)
        except (TypeError, ValueError):
            v = None
        age = row.get("age")
        out.append({"value_date": d, "value_eur": v,
                    "club": row.get("verein"),
                    "age": int(age) if str(age).isdigit() else None})
    return out


def parse_transfer_list(payload: str) -> list[dict]:
    """`ceapi/transferHistory/list` -> dated moves with fee and market value.

    `dateUnformatted` is already ISO. Upcoming (agreed but not completed) moves
    are dropped: the point of the dataset is what has happened, and a future
    transfer is the rumour board's job.
    """
    import json as _json
    try:
        data = _json.loads(payload)
    except Exception:
        return []
    out = []
    for row in data.get("transfers") or []:
        if row.get("upcoming") or row.get("futureTransfer"):
            continue
        d = row.get("dateUnformatted") or _iso(row.get("date"))
        if not d:
            continue
        frm, to = row.get("from") or {}, row.get("to") or {}
        out.append({
            "transfer_date": d,
            "to_club": to.get("clubName"),
            "to_club_id": int(_one(r"/verein/(\d+)", to.get("href") or "", d=0)),
            "from_club": frm.get("clubName"),
            "from_club_id": int(_one(r"/verein/(\d+)", frm.get("href") or "", d=0)),
            "fee_text": row.get("fee"),
            "fee_eur": money_eur(row.get("fee")),
            "value_eur": money_eur(row.get("marketValue")),
            "season_label": row.get("season"),
        })
    return out


# --------------------------------------------------------------- ingest ---
def pl_club_ids_for(season: str) -> list[tuple[int, str]]:
    """(tm_club_id, name) for the twenty clubs of one Premier League season."""
    return pl_club_ids(fetch(COMPETITION_SEASON.format(season=tm_season(season))))


def _fpl_players(conn, season: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT player_id, code player_code, web_name, full_name FROM player "
        "WHERE season=? AND position IN ('GK','DEF','MID','FWD')", (season,))]


def ingest_squads(conn, seasons: list[str], *, progress=None) -> dict:
    """Crawl every club's detailed squad page for each season and resolve ids.

    Identity is carried on `player.code`, not on `player_id`. FPL reassigns its
    element ids every season, so a Transfermarkt player resolved against
    2026-27 and then joined to 2024-25's `player` table on `player_id` lands on
    a different footballer entirely — silently, with a full injury history
    attached. `code` is the stable key and is what every feature joins on.

    A code claimed by two Transfermarkt ids unsets BOTH, the same rule the
    Understat resolver uses: handing one player another's history is worse than
    having none.
    """
    init(conn); init2(conn)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    squad_rows: list[tuple] = []
    ident: dict[int, set[int]] = {}
    names: dict[int, str] = {}
    stats = {"clubs": 0, "rows": 0}
    for season in seasons:
        try:
            clubs = pl_club_ids_for(season)
        except Exception as e:                      # noqa: BLE001
            if progress:
                progress(f"    {season}: club list failed ({e})")
            continue
        _sleep()
        players = _fpl_players(conn, season)
        for i, (cid, cname) in enumerate(clubs, 1):
            try:
                html = fetch(SQUAD_DETAIL.format(club=cid,
                                                 season=tm_season(season)))
                members = parse_squad(html)
            except Exception:                       # noqa: BLE001
                _sleep()
                continue
            stats["clubs"] += 1
            for m in members:
                code = resolve_player(m["tm_name"], players)
                code = ({p["player_id"]: p["player_code"] for p in players}
                        .get(code) if code is not None else None)
                if code is not None:
                    ident.setdefault(int(code), set()).add(m["tm_player_id"])
                names[m["tm_player_id"]] = m["tm_name"]
                squad_rows.append((season, m["tm_player_id"], cid, m["tm_name"],
                                   m["shirt"], m["detail_position"], m["dob"],
                                   m["height_cm"], m["foot"], m["joined_date"],
                                   m["signed_from"], m["signed_from_id"],
                                   m["signed_fee_eur"], m["contract_until"],
                                   m["market_value"], now))
            if progress:
                progress(f"    {season} {i}/{len(clubs)} {cname}: "
                         f"{len(members)} players")
            _sleep()
    stats["rows"] = len(squad_rows)
    if squad_rows:
        conn.executemany(
            "INSERT OR REPLACE INTO tm_squad (season, tm_player_id, tm_club_id,"
            " tm_name, shirt, detail_position, dob, height_cm, foot,"
            " joined_date, signed_from, signed_from_id, signed_fee_eur,"
            " contract_until, market_value, observed_utc)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", squad_rows)

    ambiguous = {c: v for c, v in ident.items() if len(v) > 1}
    mapping = [(tm, names.get(tm), code) for code, v in ident.items()
               if len(v) == 1 for tm in v]
    if mapping:
        for tm, nm, code in mapping:
            cur = conn.execute(
                "SELECT tm_player_id FROM tm_player WHERE tm_player_id=?",
                (tm,)).fetchone()
            if cur:
                conn.execute("UPDATE tm_player SET player_code=?, tm_name=? "
                             "WHERE tm_player_id=?", (code, nm, tm))
            else:
                conn.execute(
                    "INSERT INTO tm_player (tm_player_id, tm_name, season, "
                    "player_code) VALUES (?,?,?,?)", (tm, nm, None, code))
    # a collision means neither claim is trustworthy
    for code, v in ambiguous.items():
        conn.executemany("UPDATE tm_player SET player_code=NULL "
                         "WHERE tm_player_id=?", [(tm,) for tm in v])
    conn.commit()
    stats["resolved"] = len(mapping)
    stats["ambiguous"] = len(ambiguous)
    stats["tm_players"] = len(names)
    return stats


def crawl_targets(conn, *, resolved_only: bool = False) -> list[int]:
    """Every Transfermarkt id we have ever seen in a Premier League squad."""
    init2(conn)
    if resolved_only:
        return [int(r[0]) for r in conn.execute(
            "SELECT DISTINCT tm_player_id FROM tm_player "
            "WHERE player_code IS NOT NULL")]
    return [int(r[0]) for r in conn.execute(
        "SELECT tm_player_id FROM tm_squad "
        "UNION SELECT tm_player_id FROM tm_player")]


_crawl_targets = crawl_targets


def ingest_injuries_for(conn, *, targets: list[int], progress=None) -> dict:
    """`ingest_injuries` over an explicit id list rather than one season's map."""
    init(conn); init2(conn)
    conn.execute("PRAGMA busy_timeout=60000")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spells, failed = 0, 0
    for i, pid in enumerate(targets, 1):
        try:
            rows = parse_injuries(fetch(INJURIES.format(player=pid)))
        except Exception:                            # noqa: BLE001
            failed += 1
            _sleep()
            continue
        rows = [r for r in rows if r["from_date"]]
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO tm_injury (tm_player_id, from_date, "
                "injury, until_date, days, games_missed, season_label, "
                "observed_utc) VALUES (?,?,?,?,?,?,?,?)",
                [(pid, r["from_date"], r["injury"], r["until_date"], r["days"],
                  r["games_missed"], r["season_label"], now) for r in rows])
            spells += len(rows)
        conn.commit()
        if i % 25 == 0:
            if progress:
                progress(f"    {i}/{len(targets)} players, {spells} spells")
        _sleep()
    conn.commit()
    return {"players": len(targets), "spells": spells, "failed": failed}


def ingest_market_values(conn, *, targets: list[int] | None = None,
                         progress=None) -> dict:
    """Dated market-value history, one lightweight JSON call per player."""
    init2(conn)
    conn.execute("PRAGMA busy_timeout=60000")
    targets = targets if targets is not None else crawl_targets(conn)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, failed = 0, 0
    for i, tm in enumerate(targets, 1):
        try:
            vals = parse_market_values(fetch(MV_GRAPH.format(player=tm)))
        except Exception:                            # noqa: BLE001
            failed += 1
            _sleep(API_DELAY)
            continue
        if vals:
            conn.executemany(
                "INSERT OR REPLACE INTO tm_market_value (tm_player_id, "
                "value_date, value_eur, club, age, observed_utc) "
                "VALUES (?,?,?,?,?,?)",
                [(tm, v["value_date"], v["value_eur"], v["club"], v["age"], now)
                 for v in vals])
            rows += len(vals)
        conn.commit()          # per player: a 25-request transaction holds the
        if i % 25 == 0:        # write lock longer than a sibling crawl waits
            if progress:
                progress(f"    {i}/{len(targets)} players, {rows} valuations")
        _sleep(API_DELAY)
    conn.commit()
    return {"players": len(targets), "valuations": rows, "failed": failed}


def ingest_transfers(conn, *, targets: list[int] | None = None,
                     progress=None) -> dict:
    """Dated transfer history, one lightweight JSON call per player."""
    init2(conn)
    conn.execute("PRAGMA busy_timeout=60000")
    targets = targets if targets is not None else crawl_targets(conn)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, failed = 0, 0
    for i, tm in enumerate(targets, 1):
        try:
            moves = parse_transfer_list(fetch(TRANSFER_LIST.format(player=tm)))
        except Exception:                            # noqa: BLE001
            failed += 1
            _sleep(API_DELAY)
            continue
        if moves:
            conn.executemany(
                "INSERT OR REPLACE INTO tm_transfer (tm_player_id, "
                "transfer_date, to_club_id, to_club, from_club_id, from_club, "
                "fee_eur, fee_text, value_eur, season_label, observed_utc) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [(tm, m["transfer_date"], m["to_club_id"], m["to_club"],
                  m["from_club_id"], m["from_club"], m["fee_eur"],
                  m["fee_text"], m["value_eur"], m["season_label"], now)
                 for m in moves])
            rows += len(moves)
        conn.commit()
        if i % 25 == 0:
            if progress:
                progress(f"    {i}/{len(targets)} players, {rows} transfers")
        _sleep(API_DELAY)
    conn.commit()
    return {"players": len(targets), "transfers": rows, "failed": failed}


# ==========================================================================
# Manager history
#
# CLAUDE.md listed manager identity as unreachable ("not in any feed the
# pipeline reads"). Transfermarkt's staff-history page carries it, DATED: every
# spell at a club with an appointment date, a departure date, matches and
# points per game. Twenty-seven requests cover every club that has played a
# Premier League season in this database.
#
# The dates are what make it usable. A manager's identity is a categorical with
# no history attached; his APPOINTMENT DATE turns it into a point-in-time fact —
# who was in charge on the day of this fixture, and how long he had been there.
# ==========================================================================

MANAGERS = BASE + "/x/mitarbeiterhistorie/verein/{club}/personalie_id/1"

SCHEMA_3 = """
CREATE TABLE IF NOT EXISTS tm_manager_spell (
    tm_club_id    INTEGER NOT NULL,
    tm_manager_id INTEGER NOT NULL,
    appointed     TEXT NOT NULL,     -- ISO, point-in-time safe
    left_date     TEXT,              -- ISO; NULL while he is still in post
    manager       TEXT,
    dob           TEXT,
    days          INTEGER,
    -- LEAK WARNING: `matches` and `ppg` are TODAY's career totals for the
    -- spell, not the figures as they stood at any past date. Stored because
    -- they are free; never usable as a point-in-time feature.
    matches       INTEGER,
    ppg           REAL,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (tm_club_id, tm_manager_id, appointed)
);
CREATE INDEX IF NOT EXISTS idx_tm_mgr_club ON tm_manager_spell (tm_club_id);
"""


def init3(conn) -> None:
    conn.executescript(SCHEMA_3)


def parse_managers(html: str) -> list[dict]:
    """One dict per managerial spell on a club's staff-history page.

    Parsed by meaning again. The date of birth sits inside the nested
    inline-table and the appointment and departure dates in plain cells, so
    collecting every date in document order and taking the birth date out by
    its own anchor is what keeps a caretaker's blank departure date from
    shifting every later column.
    """
    start = html.find('class="items"')
    if start < 0:
        return []
    rows = re.split(r'<tr class="(?:odd|even)">', html[start:])[1:]
    out = []
    for r in rows:
        mid = _one(r"/profil/trainer/(\d+)", r)
        name = _one(r'/profil/trainer/\d+"[^>]*>\s*([^<]+?)\s*<', r)
        if not mid or not name:
            continue
        body = re.sub(r'\s(?:title|alt)="[^"]*"', " ", r)
        dob = _one(r"<tr>\s*<td>\s*(\d{2}/\d{2}/\d{4})\s*</td>\s*</tr>\s*</table>",
                   body)
        dates = [d for d in re.findall(r"(\d{2}/\d{2}/\d{4})", body) if d != dob]
        if not dates:
            continue
        # matches and points-per-game come AFTER the tenure cell; reading the
        # numeric cells from the start picks up the day count instead
        days = _one(r">\s*([\d.,]+)\s*days", body)
        dm = re.search(r">\s*[\d.,]+\s*days", body)
        # the match count is wrapped in a link to the performance detail page,
        # so the cell has to be de-tagged before the number is readable
        nums = re.findall(r"<td[^>]*>(.*?)</td>",
                          body[dm.end():] if dm else "", re.S)
        nums = [t for t in (_text(n) for n in nums)
                if re.fullmatch(r"[\d.,]+", t or "")]
        matches = ppg = None
        if len(nums) >= 2:
            try:
                matches = int(float(nums[0].replace(",", "").replace(".", "")))
                ppg = float(nums[1].replace(",", "."))
            except ValueError:
                pass
        out.append({
            "tm_manager_id": int(mid),
            "manager": name,
            "dob": _iso(dob),
            "appointed": _iso(dates[0]),
            # a manager still in post has no departure date, and it must stay
            # NULL rather than borrow the next column
            "left_date": _iso(dates[1]) if len(dates) > 1 else None,
            "days": int(days.replace(".", "").replace(",", "")) if days else None,
            "matches": matches,
            "ppg": ppg,
        })
    return out


def ingest_managers(conn, *, clubs: list[int] | None = None,
                    progress=None) -> dict:
    """Staff history for every club that has played a season in this database."""
    init2(conn); init3(conn)
    conn.execute("PRAGMA busy_timeout=60000")
    if clubs is None:
        clubs = [int(r[0]) for r in conn.execute(
            "SELECT DISTINCT tm_club_id FROM tm_squad "
            "WHERE tm_club_id IS NOT NULL")]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spells, failed = 0, 0
    for i, cid in enumerate(clubs, 1):
        try:
            rows = parse_managers(fetch(MANAGERS.format(club=cid)))
        except Exception:                            # noqa: BLE001
            failed += 1
            _sleep()
            continue
        rows = [r for r in rows if r["appointed"]]
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO tm_manager_spell (tm_club_id, "
                "tm_manager_id, appointed, left_date, manager, dob, days, "
                "matches, ppg, observed_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(cid, r["tm_manager_id"], r["appointed"], r["left_date"],
                  r["manager"], r["dob"], r["days"], r["matches"], r["ppg"], now)
                 for r in rows])
            spells += len(rows)
        conn.commit()
        if progress:
            progress(f"    {i}/{len(clubs)} club {cid}: {len(rows)} spells")
        _sleep()
    return {"clubs": len(clubs), "spells": spells, "failed": failed}
