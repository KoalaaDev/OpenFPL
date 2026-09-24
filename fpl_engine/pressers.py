"""Press-conference and commentary text -> player-level availability observations.

Round 18, stage 0: a rules pass, deliberately simple, so that the two gates
that decide whether any of this is worth a model can be measured first.

Input (acquired by acquire/sources/bbc*.py, never interpreted there):
  * ``acq_bbc_presser``  — BBC's Friday "Premier League news conferences"
    live blog, one post per manager quote, labelled with the fixture it
    concerns and timestamped (pre-deadline).
  * ``acq_bbc_stream``   — in-match live text ("... because of an injury").

Output: ``presser_obs`` rows — (season, gw, player_id, class, published_utc,
source, phrase, snippet). ``gw`` is the first gameweek whose first kickoff is
after the post, i.e. the decision the observation could inform. Classes:

    out        ruled out / won't be involved / not available / suspended
    doubt      a doubt / will be assessed / 50-50 / late call / hopeful
    rested     rested / rotation / unlikely to start
    available  available / fit / back in training / in contention
    injury     (commentary only) forced off with an injury in the match

Name resolution: only the two clubs in the fixture label are searched, on
full name, web name and (when unique across both squads) surname, with word
boundaries. An ambiguous surname is skipped, never guessed. Everything here
is measured before it is trusted — see research/presser_gate.py.
"""
from __future__ import annotations

import re
from datetime import date

import pandas as pd

from . import db as _db

# BBC writes the club as it pleases ("Ipswich", "Nottingham Forest", "Spurs")
# and FPL renames clubs between seasons — it called Ipswich "Ipswich" in
# 2024-25 and "Ipswich Town" in 2026-27. A hardcoded map therefore goes stale
# silently: this one did, and 74 posts (47 Ipswich, 16 Coventry, 8 Hull,
# 3 Forest) were being dropped as "not a Premier League fixture". Clubs are now
# resolved against the SEASON'S OWN team names, with aliases only for the cases
# no rule can reach.
# The BBC *lineup* feed spells clubs in full ("Manchester City", "Brighton &
# Hove Albion"), and `xpts/bbc_role_features.py` and `xpts/bbc_context.py`
# join on this table. It is kept exactly as it was because those two feed the
# shipped minutes model: changing a model input is a backtest, not a tidy-up.
# The clubs it gets wrong for 2026-27 (Ipswich, Coventry, Hull) never appear
# in that spelling there — checked, not assumed — so the defect below was the
# presser fixture labels only.
BBC_TO_FPL = {
    "Tottenham": "Spurs", "Tottenham Hotspur": "Spurs",
    "Nottingham Forest": "Nott'm Forest", "Nottm Forest": "Nott'm Forest",
    "Manchester United": "Man Utd", "Manchester City": "Man City",
    "Newcastle United": "Newcastle", "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich", "Sheffield United": "Sheff Utd",
    "Luton Town": "Luton", "Leeds United": "Leeds", "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton", "Brighton and Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves", "AFC Bournemouth": "Bournemouth",
    "Crystal Palace": "Crystal Palace", "Aston Villa": "Aston Villa",
}

CLUB_ALIASES = {
    "spurs": "tottenham", "tottenham hotspur": "tottenham",
    "forest": "nottingham forest", "nottm forest": "nottingham forest",
    "notts forest": "nottingham forest",
    "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city",
    "wolves": "wolverhampton wanderers",
    "sheff utd": "sheffield united", "sheff wed": "sheffield wednesday",
    "west brom": "west bromwich albion",
}
CLUB_SUFFIXES = ("city", "town", "united", "albion", "wanderers", "hotspur",
                 "rovers", "county", "athletic", "fc", "afc")


def _norm_club(name: str) -> str:
    t = (name or "").lower().replace("&", "and").replace("'", "").replace(".", "")
    t = re.sub(r"[^a-z ]", " ", t)
    return " ".join(t.split())


def _club_keys(name: str) -> set[str]:
    """Every spelling of a club that should resolve to it."""
    base = _norm_club(name)
    keys = {base}
    words = base.split()
    while len(words) > 1 and words[-1] in CLUB_SUFFIXES:
        words = words[:-1]
        keys.add(" ".join(words))
    for short, full in CLUB_ALIASES.items():
        if _norm_club(full) in keys or short in keys:
            keys.add(short)
            keys.add(_norm_club(full))
            fw = _norm_club(full).split()
            while len(fw) > 1 and fw[-1] in CLUB_SUFFIXES:
                fw = fw[:-1]
                keys.add(" ".join(fw))
    return keys


def club_index(names) -> dict[str, str]:
    """{every spelling: the season's own club name}.

    A key two clubs could both answer to is dropped rather than given to
    whichever was seen first — "Manchester" is not an answer.
    """
    idx: dict[str, str] = {}
    clash: set[str] = set()
    for n in names:
        for k in _club_keys(n):
            if k in idx and idx[k] != n:
                clash.add(k)
            idx.setdefault(k, n)
    for k in clash:
        idx.pop(k, None)
    return idx


def resolve_club(name: str, idx: dict[str, str]) -> str | None:
    """A club as BBC spells it -> the club as this season's FPL data spells it."""
    t = _norm_club(name)
    words = [w for w in t.split() if w not in ("afc", "fc")]
    t = " ".join(words)
    for cand in (t, CLUB_ALIASES.get(t, t)):
        if cand in idx:
            return idx[cand]
    # "Brighton and Hove Albion" -> "Brighton": drop words from the end, which
    # covers both the club suffixes and the long ceremonial names
    for k in range(len(words) - 1, 0, -1):
        head = " ".join(words[:k])
        for cand in (head, CLUB_ALIASES.get(head, head)):
            if cand in idx:
                return idx[cand]
    return None


# ordered: the first family that matches the player's sentence wins
PATTERNS = [
    ("out", re.compile(
        r"ruled out|won'?t be involved|will not be involved|won'?t be available|"
        r"will not be available|not available|unavailable|is out\b|are out\b|"
        r"remains? out|still out|out for (the|a|an|several|weeks|months|\d)|"
        r"out until|out of (the|Saturday|Sunday|this|contention|squad)|sidelined|"
        r"miss(es|ing)? (the|Saturday|Sunday|Monday|this|out)|will miss|"
        r"no chance|surgery|operation|operated|long[- ]term|not fit\b|isn'?t fit|"
        r"won'?t make it|suspended|serves? (a|his) (ban|suspension)|banned|"
        r"won'?t play|will not play|not be (fit|ready)|"
        r"not going to be (fit|ready|involved)|"
        r"needs? (another|a few|more) (week|month)|"
        # what a manager says instead of "out"
        r"not in the squad|not be in the squad|out of the squad|not in contention|"
        r"won'?t feature|will not feature|unavailable for selection|"
        r"too soon for|comes? too soon|came too soon|not ready for (this|Saturday|Sunday)|"
        r"won'?t risk|not risk(ing)? him|not going to risk|"
        r"weeks? away|months? away|still (a few|some|several) weeks|"
        r"in the treatment room|did ?(n'?t|not) travel|not travell?ing|won'?t travel|"
        r"back (after|following) the international break|"
        r"(tear|torn|ruptur|fracture|broken)\w*", re.I)),
    ("doubt", re.compile(
        r"\bdoubt|assess(ed|ing|ment)?\b|will be assessed|we'?ll assess|"
        r"50-50|50/50|touch and go|wait and see|"
        r"late (call|decision|check|test)|not sure|unsure|hopeful|we'?ll see|"
        r"see how he|a chance|small chance|outside chance|might make it|"
        r"could (be|return|feature|make)|might (be|return|feature)|"
        r"may (be|return|feature)|not (yet|quite) (ready|there)|"
        r"question mark|fitness (test|check)|day[- ]to[- ]day|struggl|niggle|"
        r"pulled out|felt (something|his)|picked up a|"
        # the hedges managers actually use
        r"optimistic (about|that|he)|in the balance|borderline|"
        r"not 100|not fully fit|carrying (a|an) (knock|injury|problem)|"
        r"slight (problem|issue|knock|strain)|minor (problem|issue|knock)|"
        r"if he trains|trains? (today|tomorrow) (and|we)|part(-| )trained|"
        r"has ?n'?t trained (fully|much|with)|trained (partially|separately)|"
        r"we hope (he|to have)|hopefully (he|they)|fingers crossed", re.I)),
    ("rested", re.compile(
        r"\brest(ed|ing)?\b|rotat|unlikely to start|not start|won'?t start|"
        r"start on the bench|from the bench|manage his minutes|"
        r"protect(ed|ing)? him|"
        r"needs? a (rest|break)|give him a (rest|break)|"
        r"manage his (load|game time|minutes)|load management|"
        r"freshen (things|it|the team) up", re.I)),
    ("available", re.compile(
        r"\bavailable\b|\bfit\b|fully fit|back in training|trained (this|all|fully)|"
        r"in (full )?training|\breturns?\b|is back|are back|back in (contention|the squad)|"
        r"in contention|\bready\b|no problem|should be (fine|ok|okay)|all good|"
        r"good to go|in the squad|will be involved|can play|could start|"
        r"no (new|fresh|further) (injur|problem|concern)|recovered|clean bill|"
        # the all-clear, which is how most team news is actually delivered
        r"every(one|body) (is|'?s)? ?(fit|fine|available|good)|"
        r"(is|are|'?s) fine\b|all fine|no (injury )?(issues|problems|concerns)|"
        r"no absentees|nobody (is )?(out|missing)|"
        r"trained (today|yesterday|normally|with the (group|team|rest))|"
        r"came through (the (game|match|week))?|back available|available again|"
        r"fit again|fully available|100 ?(%|per cent)|in the mix", re.I)),
]
INJURY_STREAM = re.compile(r"because of an injury|injur(y|ed)|forced off|limp", re.I)
# a quoted answer ends with ." — without allowing the closing quote the
# splitter runs two managers' answers together, and the first verdict then
# reaches the second player's name
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"'\u201d\u2019)]*\s+|\n+")
# a sentence often names several players with different fates ("Saka is
# suspended but Odegaard returns"); the class is read from the CLAUSE that
# names the player, split on contrast words and punctuation
CLAUSE_SPLIT = re.compile(r",|;|:|\bbut\b|\bwhile\b|\bwhereas\b|\balthough\b|"
                          r"\bhowever\b|\bthough\b|\band\b(?=\s+[A-Z])", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS presser_obs (
    season        TEXT NOT NULL,
    gw            INTEGER,
    player_id     INTEGER NOT NULL,
    team_id       INTEGER,
    cls           TEXT NOT NULL,
    source        TEXT NOT NULL,        -- presser | commentary
    published_utc TEXT NOT NULL,
    phrase        TEXT,
    snippet       TEXT,
    page_id       TEXT,
    post_urn      TEXT NOT NULL,
    PRIMARY KEY (season, player_id, post_urn, cls)
);
CREATE INDEX IF NOT EXISTS idx_presser_obs_gw ON presser_obs (season, gw);
"""


def season_of(d: str) -> str:
    y, m = int(d[:4]), int(d[5:7])
    y0 = y if m >= 7 else y - 1
    return f"{y0}-{str(y0 + 1)[2:]}"


def fixture_clubs(label: str, idx: dict[str, str] | None = None) -> tuple[str, str] | None:
    """The two clubs a fixture label names, as the season calls them."""
    m = re.match(r"\s*(.+?)\s+v(?:s)?\.?\s+(.+?)(?:\s*\(|\s*$)", label or "")
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if idx is None:
        return a, b
    return resolve_club(a, idx) or a, resolve_club(b, idx) or b


def gw_kickoffs(conn, season: str) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT gw, MIN(kickoff_utc) k FROM ("
        " SELECT gw, kickoff_utc FROM fixture WHERE season=? AND gw IS NOT NULL"
        " UNION ALL SELECT gw, kickoff_utc FROM team_match WHERE season=? AND gw IS NOT NULL)"
        " WHERE kickoff_utc IS NOT NULL GROUP BY gw ORDER BY gw", (season, season)).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def gw_after(kickoffs: list[tuple[int, str]], when: str) -> int | None:
    for gw, k in kickoffs:
        if k > when:
            return gw
    return None


def _squads(conn, season: str) -> dict[str, list[dict]]:
    rows = conn.execute(
        "SELECT p.player_id, p.team_id, t.name team, p.web_name, p.full_name "
        "FROM player p JOIN team t ON t.season=p.season AND t.team_id=p.team_id "
        "WHERE p.season=? AND p.position IN ('GK','DEF','MID','FWD')", (season,)).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r[2], []).append({"player_id": r[0], "team_id": r[1],
                                          "web_name": r[3] or "", "full_name": r[4] or ""})
    return out


def _name_patterns(players: list[dict]) -> list[tuple[dict, re.Pattern]]:
    """(player, regex) pairs; surnames only when unique across the given squads."""
    # a short name (surname or FPL web name) is usable only when exactly one
    # player across the two squads answers to it
    count: dict[str, int] = {}
    for p in players:
        sn = (p["full_name"].split() or [""])[-1].lower()
        for n in {sn, p["web_name"].lower()}:
            if n:
                count[n] = count.get(n, 0) + 1
    out = []
    for p in players:
        names = {p["full_name"]}
        sn = (p["full_name"].split() or [""])[-1]
        if len(sn) >= 4 and count.get(sn.lower(), 0) == 1:
            names.add(sn)
        wn = p["web_name"]
        if len(wn) >= 3 and count.get(wn.lower(), 0) == 1:
            names.add(wn)
        alts = [re.escape(n) for n in names if len(n) >= 3]
        if alts:
            out.append((p, re.compile(r"(?<![A-Za-z])(" + "|".join(alts) + r")(?![A-Za-z])")))
    return out


def classify(sentence: str) -> tuple[str, str] | None:
    for cls, rx in PATTERNS:
        m = rx.search(sentence)
        if m:
            return cls, m.group(0)
    return None


# how far a verdict can reach past the sentence that named the players
CARRY_SENTENCES = 3


def extract_post(text: str, patterns) -> list[tuple[dict, str, str, str]]:
    """(player, cls, phrase, sentence) for every classified mention.

    Two passes. The first reads the clause that names the player, which is the
    conservative case. The second is the shape most of these posts actually
    take: a lead that names the players and says nothing about them, then the
    quote that answers — "Arteta was asked about the availability of Mosquera,
    White, Timber and Hincapie: 'Everyone is fine.'" The names are in one
    sentence and the verdict in the next, so the first pass sees nothing at
    all. A verdict carries only while the following sentences name nobody
    else, so a quote about a different player cannot be misattributed.
    """
    sents = [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]
    named = [[p for p, rx in patterns if rx.search(s)] for s in sents]
    rx_of = {id(p): rx for p, rx in patterns}
    out, classified = [], set()
    for i, sent in enumerate(sents):
        clauses = [c for c in CLAUSE_SPLIT.split(sent) if c and c.strip()]
        for p in named[i]:
            rx = rx_of[id(p)]
            # the clause naming him decides; a status stated once for a whole
            # list ("X, Y and Z are all out") falls back to the sentence
            hit = None
            for cl in clauses:
                if rx.search(cl):
                    hit = classify(cl)
                    if hit:
                        break
            if hit is None:
                # "asked about the availability of A, B and C: everyone is
                # fine" — the verdict is a clause of its own. It carries only
                # while it names nobody else, so "Saka is suspended but
                # Odegaard returns" still keeps its two halves apart.
                free = [cl for cl in clauses
                        if classify(cl) and not any(r.search(cl) for _q, r in patterns)]
                if free:
                    hit = classify(free[0])
                elif not any(classify(cl) for cl in clauses):
                    hit = classify(sent)
            if hit:
                out.append((p, hit[0], hit[1], sent[:240]))
                classified.add(i)
    for i, sent in enumerate(sents):
        if not named[i] or i in classified:
            continue
        for j in range(i + 1, min(i + 1 + CARRY_SENTENCES, len(sents))):
            if named[j]:
                break                       # the answer is about someone else
            hit = classify(sents[j])
            if hit:
                for p in named[i]:
                    out.append((p, hit[0], hit[1], f"{sent} {sents[j]}"[:240]))
                break
    return out


def extract_pressers(conn, seasons: list[str] | None = None) -> dict:
    conn.executescript(SCHEMA)
    pages = conn.execute("SELECT page_id, date_published, headline FROM acq_bbc_presser_page "
                         "WHERE date_published IS NOT NULL").fetchall()
    stats = {"pages": 0, "posts": 0, "labelled_pl": 0, "obs": 0, "unknown_clubs": {}}
    cache: dict[str, tuple] = {}
    for page_id, dpub, headline in pages:
        season = season_of(dpub)
        if seasons and season not in seasons:
            continue
        if season not in cache:
            sq = _squads(conn, season)
            cache[season] = (sq, gw_kickoffs(conn, season), club_index(sq))
        squads, kickoffs, cidx = cache[season]
        stats["pages"] += 1
        conn.execute("DELETE FROM presser_obs WHERE page_id=? AND source='presser'", (page_id,))
        for post_urn, pub, label, text in conn.execute(
                "SELECT post_urn, published_utc, fixture_label, text FROM acq_bbc_presser "
                "WHERE page_id=?", (page_id,)):
            stats["posts"] += 1
            clubs = fixture_clubs(label, cidx)
            if not clubs:
                continue
            players = []
            for cname in clubs:
                if cname in squads:
                    players += squads[cname]
                else:
                    stats["unknown_clubs"][cname] = stats["unknown_clubs"].get(cname, 0) + 1
            if len(players) < 15:            # not a Premier League fixture
                continue
            stats["labelled_pl"] += 1
            gw = gw_after(kickoffs, pub)
            for p, cls, phrase, sent in extract_post(text, _name_patterns(players)):
                conn.execute(
                    "INSERT OR REPLACE INTO presser_obs (season, gw, player_id, team_id, cls, "
                    "source, published_utc, phrase, snippet, page_id, post_urn) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (season, gw, p["player_id"], p["team_id"], cls, "presser", pub, phrase,
                     sent, page_id, post_urn))
                stats["obs"] += 1
    conn.commit()
    stats["unknown_clubs"] = dict(sorted(stats["unknown_clubs"].items(), key=lambda kv: -kv[1])[:8])
    return stats


def extract_commentary(conn, seasons: list[str] | None = None) -> dict:
    """In-match injuries -> an `injury` observation for the player's NEXT gameweek."""
    conn.executescript(SCHEMA)
    stats = {"matches": 0, "posts": 0, "obs": 0}
    cache: dict[str, tuple] = {}
    matches = conn.execute("SELECT event_urn, season, match_date, home, away FROM acq_bbc_match "
                           "WHERE lineups_done=1").fetchall()
    for urn, season, mdate, home, away in matches:
        if seasons and season not in seasons:
            continue
        if season not in cache:
            cache[season] = (_squads(conn, season), gw_kickoffs(conn, season))
        squads, kickoffs = cache[season]
        clubs = [BBC_TO_FPL.get(home, home), BBC_TO_FPL.get(away, away)]
        players = [p for c in clubs for p in squads.get(c, [])]
        if len(players) < 15:
            continue
        stats["matches"] += 1
        conn.execute("DELETE FROM presser_obs WHERE page_id=? AND source='commentary'", (urn,))
        pats = _name_patterns(players)
        for post_urn, pub, text in conn.execute(
                "SELECT post_urn, published_utc, text FROM acq_bbc_stream WHERE event_urn=?", (urn,)):
            if not text or not INJURY_STREAM.search(text):
                continue
            stats["posts"] += 1
            gw = gw_after(kickoffs, pub)
            for p, rx in pats:
                if rx.search(text):
                    conn.execute(
                        "INSERT OR REPLACE INTO presser_obs (season, gw, player_id, team_id, cls, "
                        "source, published_utc, phrase, snippet, page_id, post_urn) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (season, gw, p["player_id"], p["team_id"], "injury", "commentary", pub,
                         INJURY_STREAM.search(text).group(0), text[:240], urn, post_urn))
                    stats["obs"] += 1
    conn.commit()
    return stats


def observations(conn, season: str, gw: int, before: str | None = None,
                 sources=("presser",)) -> pd.DataFrame:
    """Point-in-time: only rows published strictly before `before`."""
    q = ("SELECT player_id, cls, source, published_utc, phrase FROM presser_obs "
         "WHERE season=? AND gw=? AND source IN (%s)" % ",".join("?" * len(sources)))
    args: list = [season, gw, *sources]
    if before:
        q += " AND published_utc < ?"
        args.append(before)
    return pd.read_sql_query(q, conn, params=args)


# how much of a player's expected exposure each class keeps (the research arm's
# prior; the gate measures the realised start rate per class to replace them)
CLASS_FACTORS = {"out": 0.05, "injury": 0.35, "doubt": 0.6, "rested": 0.4, "available": 1.0}

# LIVE overlay (Round 18): SHOWN, NOT MODELLED. Against the correct baseline
# the replay arm that scaled exposure by these statements made rank among
# played players significantly WORSE in 2024-25 (spearman_played -0.0022,
# p=0.018): the rules extractor is right about half the time when it calls a
# likely starter "out", and halving a real starter costs more than zeroing an
# absent one gains. So every factor is 1.0 — the statement is carried to the
# player card with its timestamp, exactly like Polymarket and the
# Transfermarkt dossier — until an extractor clears the precision bar.
LIVE_FACTORS = {"out": 1.0, "doubt": 1.0, "rested": 1.0, "available": 1.0}
CLASS_RANK = {"out": 0, "injury": 1, "doubt": 2, "rested": 3, "available": 4}


def exposure_factors(obs: pd.DataFrame, factors: dict | None = None) -> dict[int, float]:
    """One factor per player: the most severe class among his observations,
    the latest observation winning a tie within the same severity."""
    factors = factors or CLASS_FACTORS
    out: dict[int, float] = {}
    if obs is None or obs.empty:
        return out
    o = obs.copy()
    o["rank"] = o["cls"].map(CLASS_RANK).fillna(9)
    o = o.sort_values(["rank", "published_utc"], ascending=[True, False])
    for pid, d in o.groupby("player_id"):
        cls = d.iloc[0]["cls"]
        out[int(pid)] = float(factors.get(cls, 1.0))
    return out


def live_overlay(conn, season: str, gws: list[int], as_of_of) -> dict[int, dict[int, dict]]:
    """Per gameweek, per player: the factor to apply and what was said.

    ``as_of_of(gw)`` gives the point-in-time cutoff for that gameweek, so a
    Friday statement informs Saturday's deadline and nothing later leaks in.
    Never raises: a missing table means no overlay.
    """
    out: dict[int, dict[int, dict]] = {}
    for g in gws:
        try:
            obs = observations(conn, season, int(g), before=as_of_of(g))
        except Exception:      # noqa: BLE001 - no observations yet
            continue
        if obs is None or obs.empty:
            continue
        fac = exposure_factors(obs, LIVE_FACTORS)
        o = obs.copy()
        o["rank"] = o["cls"].map(CLASS_RANK).fillna(9)
        o = o.sort_values(["rank", "published_utc"], ascending=[True, False])
        per: dict[int, dict] = {}
        for pid, d in o.groupby("player_id"):
            r = d.iloc[0]
            per[int(pid)] = {"factor": fac.get(int(pid), 1.0), "cls": r["cls"],
                             "phrase": r["phrase"], "when": r["published_utc"]}
        out[int(g)] = per
    return out
