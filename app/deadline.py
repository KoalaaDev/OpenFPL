"""The admin "Deadline" view: what the model is thinking right now, and why.

One payload, assembled from the live sources the pipeline already archives,
so an admin can read the state of play before a deadline without opening a
terminal:

  * refresh   — scheduler state, last/next run, jobs in flight, cache age
  * model     — the minutes model actually serving (features, training
                seasons, holdout accuracy, when it was trained), the
                xPts/OpenFPL blend weight, the shipped rate units
  * news      — FPL availability changes in the last seven days, newest
                first (the change log, verbatim)
  * pressers  — what managers said for the coming gameweek (BBC Friday page)
  * lineups   — RotoWire's latest predicted XIs beside the model's P(start),
                with the disagreements the E8b pricing says are worth money
  * movers    — whose projection moved most since the previous build
  * feed_test — the running lineup-feed scorecard (band accuracy vs model)

Admin-only: it exposes nothing personal, but it is expensive to assemble and
is a view of the operator's machinery, not the product.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from fpl_engine import config, db
from fpl_engine.pipeline import next_gw

from . import jobs, scheduler, services

_cache: dict = {"t": 0.0, "v": None}
TTL = 120.0


def _names(conn, season: str) -> tuple[dict, dict]:
    pl = {int(r[0]): {"name": r[1], "team_id": int(r[2]), "pos": r[3]}
          for r in conn.execute("SELECT player_id, web_name, team_id, position FROM player "
                                "WHERE season=?", (season,))}
    tm = {int(r[0]): r[1] for r in conn.execute(
        "SELECT team_id, short_name FROM team WHERE season=?", (season,))}
    return pl, tm


def _market(conn, season: str, gw: int, tm: dict) -> dict:
    """Which upcoming fixtures carry a market price, and from where. The
    Odds API key was silently rejected (401) for the whole of 2026-27 to
    GW4, so the live model ran on the team model alone; this row makes
    that visible before it costs again."""
    from fpl_engine.xpts import odds_model as _om
    try:
        fx = conn.execute("SELECT fixture_id, team_h, team_a FROM fixture WHERE season=? AND gw=?",
                          (season, gw)).fetchall()
        src: dict = {}
        _om.fixture_odds_map(conn, season, [int(f["fixture_id"]) for f in fx], sources=src)
        rows = [{"fixture": f"{tm.get(f['team_h'])} v {tm.get(f['team_a'])}",
                 "source": src.get(int(f["fixture_id"]), "none")} for f in fx]
        counts = {k: sum(1 for r in rows if r["source"] == k) for k in ("bookmaker", "polymarket", "none")}
        return {"gw": gw, "counts": counts, "rows": rows,
                "warning": ("no bookmaker odds for any fixture — check ODDS_API_KEY (a rejected key "
                            "is skipped silently by the pull)") if counts["bookmaker"] == 0 else None}
    except Exception as exc:  # noqa: BLE001
        return {"gw": gw, "counts": {}, "rows": [], "warning": str(exc)}


def _model_meta() -> dict:
    out = {}
    path = os.path.join(config.MODELS_DIR, "xpts", "minutes_meta.json")
    try:
        m = json.load(open(path, encoding="utf-8"))
        feats = m.get("features") or []
        out["minutes"] = {
            "features": len(feats),
            "blocks": {"bbc_role": sum(f.startswith("bbc_") for f in feats),
                       "understat_line": sum(f.startswith("role_") for f in feats)},
            "train_seasons": m.get("train_seasons"),
            "holdout_season": m.get("valid_season"),
            "holdout_accuracy": m.get("holdout_accuracy"),
            "trained_at": datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc).isoformat(),
        }
    except (OSError, ValueError):
        out["minutes"] = None
    try:
        b = json.load(open(os.path.join(config.MODELS_DIR, "xpts", "blend.json"), encoding="utf-8"))
        out["blend"] = {"xpts_weight": b.get("weight"), "fitted_on": b.get("season")}
    except (OSError, ValueError):
        out["blend"] = None
    out["rates"] = "per-position xG conversion + xA in FPL-assist units (Round 17)"
    return out


def _news(conn, season: str, pl: dict, tm: dict, days: float = 7.0) -> list[dict]:
    since = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
    try:
        rows = conn.execute(
            "SELECT player_id, observed_utc, source_published_utc, status, chance_next, news "
            "FROM acq_player_availability WHERE season=? AND observed_utc>=? "
            "ORDER BY observed_utc DESC LIMIT 400", (season, since)).fetchall()
    except Exception:  # noqa: BLE001
        return []
    seen, out = set(), []
    for pid, obs, pub, status, chance, news in rows:
        if pid in seen:
            continue
        seen.add(int(pid))
        p = pl.get(int(pid))
        if not p:
            continue
        out.append({"player_id": int(pid), "name": p["name"], "team": tm.get(p["team_id"]),
                    "pos": p["pos"], "status": status, "chance": chance, "news": news,
                    "observed": obs, "published": pub})
    # FPL's own news_added is when the news broke; a row observed late (a
    # backfill, a poll that missed a window) must not jump the queue
    out.sort(key=lambda r: r["published"] or r["observed"] or "", reverse=True)
    return out[:80]


QUOTE_DAYS = 4          # a deadline's build-up, not the season's
QUOTE_MAX = 200
# Newest-first with one global cap, a long presser (Carrick's ran to 22
# posts) pushed the MORNING speakers off the end of a Friday. Capping each
# club keeps every manager who spoke on the panel.
QUOTE_PER_CLUB = 12
# A capitalised name, allowing the lowercase particles managers' names carry
# ("Roberto de Zerbi", "Arne ten Hag"): without them the name stopped at the
# particle and the desk credited quotes to "Roberto".
_NAME = (r"[A-Z][\w'\u2019.\-]+"
         r"(?:\s+(?:(?:de|da|di|del|der|den|van|von|ten|ter|le|la|du|dos|das)\s+)*"
         r"[A-Z][\w'\u2019.\-]+){0,3}")
# "<club> boss|manager|head coach <name>" ANYWHERE near the start of a post.
# BBC opens plenty of posts with a lead-in ("And here's Everton boss David
# Moyes", "A final word from Fulham boss ...", "In today's news conference,
# Arsenal manager Mikel Arteta ..."); anchoring at the first character made
# the lead-in the club. The club is whatever trailing words before the role
# RESOLVE to a club -- see _speaker.
_SPEAKER = re.compile(
    r"(?P<pre>[\w'\u2019&.\- ]{0,60}?)\s*"
    r"(?:interim\s+)?(?:boss|manager|head coach)\s+(?P<name>" + _NAME + ")")
_SPEAKER_WINDOW = 160   # only the opening of a post names its speaker
# A Champions League / Europa / Conference League news-conference page shares
# the BBC's title wording but is not a Premier League deadline's build-up.
_EUROPEAN = re.compile(r"champions league|europa|conference league|uefa", re.I)


def _club_resolver(conn, season: str):
    """BBC's spelling of a club -> (team_id, short_name), or None.

    Built on the extractor's club index (aliases, ceremonial names, suffixes),
    so "Manchester United", "Leeds United" and "Tottenham" resolve to FPL's
    "Man Utd", "Leeds" and "Spurs". The ad-hoc substring test it replaces left
    those three unresolved and split them from their own club in the picker.
    """
    from fpl_engine import pressers as _pr
    by_name = {name: (tid, short) for name, tid, short in
               ((str(r[1]), int(r[0]), str(r[2] or r[1])) for r in conn.execute(
                   "SELECT team_id, name, short_name FROM team WHERE season=?", (season,)))}
    idx = _pr.club_index(list(by_name))

    lower = {n.lower(): v for n, v in by_name.items()}

    def resolve(text: str):
        if not text or not text.strip():
            return None
        name = _pr.resolve_club(text, idx)
        if name:
            return by_name.get(name)
        # the season's name can be LONGER than BBC's ("Brighton & Hove Albion"
        # in some seasons' data, "Brighton" on the page): a unique prefix match
        t = text.strip().lower()
        hits = [v for n, v in lower.items() if len(t) >= 5 and n.startswith(t)]
        return hits[0] if len(hits) == 1 else None
    return resolve


def _speaker(body: str, resolve_club, known: dict):
    """(team_id, short, manager) for a post's opening, or None.

    The club is the shortest run of trailing words before the role that
    resolves -- "And here's Everton" -> Everton, "Let's move on to Spurs'
    opponents Chelsea. Here is" -> no club, but the name is kept so the
    manager memory (`known`) can place it.
    """
    head = body[:_SPEAKER_WINDOW]
    for m in _SPEAKER.finditer(head):
        name = re.sub(r"[\u2019']s?$", "", m.group("name").strip())   # "Carrick's"
        words = re.sub(r"[\u2019']s\b", "", m.group("pre")).split()
        club = None
        # SHORTEST run first: the club nearest the role word is the speaker's.
        # Longest-first let resolve_club's drop-from-the-end read "Liverpool.
        # Here is Arsenal manager ..." as Liverpool.
        for k in range(1, min(4, len(words)) + 1):
            club = resolve_club(" ".join(words[-k:]).strip(" ,.:;"))
            if club:
                break
        if club:
            return club[0], club[1], name
        if name in known:
            return known[name][0], known[name][1], name
        # a club this season's table does not hold: keep the speaker, and the
        # club as written -- only its capitalised tail, never the lead-in
        tail = []
        for w in reversed(words):
            if not w[:1].isupper():
                break
            tail.insert(0, w)
        if tail and len(tail) <= 3:
            return None, " ".join(tail[-3:]).strip(" ,.:;"), name
    # "Mathias Jaissle on 18-year-old midfielder ...": a post opening with a
    # manager already placed by another post on the page
    for name, (tid, short) in known.items():
        if head.startswith(name) or head.startswith(f"And {name}"):
            return tid, short, name
    return None


def _presser_quotes(conn, season: str, gw: int, pl: dict, tm: dict) -> list[dict]:
    """What the managers actually SAID, not only what the extractor could turn
    into a player statement.

    The rules extractor is deliberately conservative — it classifies the clause
    that names a player, so "was asked about the availability of Mosquera,
    White, Timber and Hincapie" followed by "everyone is fine" yields nothing —
    and the desk was showing only its output. On a Friday morning that reads as
    one line against two managers' worth of quotes. The quotes are archived
    anyway; this serves them, with whatever the extractor did resolve attached
    to the post it came from.
    """
    try:
        since = (datetime.now(timezone.utc) - timedelta(days=QUOTE_DAYS)).isoformat()
        posts = conn.execute(
            "SELECT page_id, post_urn, published_utc, fixture_label, text FROM acq_bbc_presser "
            "WHERE published_utc >= ? ORDER BY published_utc DESC LIMIT 400",
            (since,)).fetchall()
    except Exception:  # noqa: BLE001 - no presser archive yet
        return []
    # The extractor's tags are a decoration on a quote, never a precondition:
    # a missing or failing presser_obs used to return [] for the whole panel.
    try:
        obs_rows = conn.execute(
            "SELECT post_urn, player_id, cls FROM presser_obs "
            "WHERE season=? AND source='presser' AND post_urn IS NOT NULL", (season,)).fetchall()
    except Exception:  # noqa: BLE001
        obs_rows = []
    try:
        european = {str(r[0]) for r in conn.execute(
            "SELECT page_id, headline FROM acq_bbc_presser_page")
            if _EUROPEAN.search(r[1] or "") and "premier league" not in (r[1] or "").lower()}
    except Exception:  # noqa: BLE001 - no page table: nothing to exclude
        european = set()
    posts = [p for p in posts if str(p[0]) not in european]
    tagged: dict[str, list[dict]] = {}
    for urn, pid, cls in obs_rows:
        p = pl.get(int(pid))
        if not p:
            continue
        seen = tagged.setdefault(str(urn), [])
        if not any(t["player_id"] == int(pid) for t in seen):
            seen.append({"player_id": int(pid), "name": p["name"],
                         "team": tm.get(p["team_id"]), "cls": cls})
    resolve_club = _club_resolver(conn, season)
    bodies = [(urn, pub, fixture, " ".join((text or "").split()))
              for _page, urn, pub, fixture, text in posts]
    # Pass 1: managers placed by a post that names club AND manager. A page
    # introduces "Newcastle boss Mathias Jaissle" once, then quotes "Mathias
    # Jaissle on ..." -- the second form is only attributable through the first.
    known: dict = {}
    for _urn, _pub, _fx, body in bodies:
        s = _speaker(body, resolve_club, {})
        if s:
            known.setdefault(s[2], (s[0], s[1]))
    out, per_club = [], {}
    for urn, pub, fixture, body in bodies:
        s = _speaker(body, resolve_club, known)
        players = tagged.get(str(urn), [])
        # a post that names nobody and quotes nobody is live-blog chatter
        if not s and not players:
            continue
        club_id, club, manager = s if s else (None, None, None)
        if club_id is None and players:
            # narration about one club's players ("Arsenal fear Saka could be
            # out ...") belongs to that club, not to "Other"
            teams_of = {(pl.get(t["player_id"]) or {}).get("team_id") for t in players}
            if len(teams_of) == 1 and None not in teams_of:
                club_id = teams_of.pop()
                club = tm.get(club_id)
        key = club_id if club_id is not None else (club or "other")
        if per_club.get(key, 0) >= QUOTE_PER_CLUB:
            continue
        per_club[key] = per_club.get(key, 0) + 1
        # "<lead>: "the quote"" -- the lead is context, the quote is the news
        lead, quote = body, ""
        cut = min([i for i in (body.find('"'), body.find("“")) if i > 0] or [0])
        if cut:
            lead, quote = body[:cut].rstrip(" :"), body[cut:]
        out.append({"when": pub, "fixture": fixture, "club": club, "club_id": club_id,
                    "manager": manager,
                    "lead": lead[:400], "quote": quote[:900].strip(), "players": players})
        if len(out) >= QUOTE_MAX:
            break
    return out


def _pressers(conn, season: str, gw: int, pl: dict, tm: dict) -> list[dict]:
    try:
        rows = conn.execute(
            "SELECT player_id, cls, phrase, snippet, published_utc FROM presser_obs "
            "WHERE season=? AND gw=? AND source='presser' ORDER BY published_utc DESC",
            (season, gw)).fetchall()
    except Exception:  # noqa: BLE001
        return []
    out, seen = [], set()
    for pid, cls, phrase, snippet, pub in rows:
        key = (int(pid), cls)
        if key in seen:
            continue
        seen.add(key)
        p = pl.get(int(pid))
        if p:
            out.append({"player_id": int(pid), "name": p["name"], "team": tm.get(p["team_id"]),
                        "cls": cls, "phrase": phrase, "snippet": snippet, "when": pub})
    return out[:120]


# The predicted-lineup feeds that state a FORMATION, in order of trust.
#
# RotoWire is not one of them. Its per-player position is a template: across
# 239 predicted XIs this season it used five position sequences in total, 18 of
# 20 clubs never changed shape, and its implied shape matched what was played
# 85% of the time -- worse than simply repeating each club's last formation
# (92%). It had Chelsea and Leeds at 3-4-2-1 for GW5; they last played 4-2-3-1
# and 3-5-2.
#
# SportsGambler states a formation per fixture and lists the eleven by row, and
# every club's rows add up to its stated shape; for GW5 its formation matched
# each club's most recent played shape 19/20 times, the odd ones included.
# FFScout states one too but in its own vocabulary ("3-4-3" where the match
# data says 3-4-2-1), so it is the fallback. RotoWire stays the SCORED feed in
# lineup_feed.py -- its XI names are what that scorecard prices -- it just does
# not get to draw the shape.
FORMATION_FEEDS = ("sportsgambler", "ffscout")


def live_lineups_dir() -> str:
    """The server's own predicted-lineup archive (gitignored, server-local).

    The repository's `data/collected` archive is written ONLY by the scheduled
    GitHub Action and reaches a running server only when someone redeploys or
    pulls -- so a deployment left running served the line-ups it was deployed
    with, and from the next gameweek on, "no predicted XI archived for GW yet".
    The news loop now refreshes the formation feeds here (`refresh_feeds`).
    """
    return os.path.join(config.DATA_DIR, "web_cache", "live_feeds")


def refresh_feeds() -> dict:
    """Fetch SportsGambler and FFScout's predicted XIs into the live archive,
    with the Action's own collectors (same columns, same change detection), so
    the Live tab's line-ups are as fresh as the server's poll. Never raises."""
    try:
        from acquire import actions as _ac
        from acquire.core import http as _http
        boot = services.bootstrap()
        season = config.CURRENT_SEASON
        nxt = next((int(e["id"]) for e in boot.get("events", []) if e.get("is_next")), None)
        out_dir = live_lineups_dir()
        observed = _http.utcnow()
        return {"sportsgambler": _ac._collect_sportsgambler(out_dir, season, nxt, observed),
                "ffscout": (_ac._collect_ffscout(out_dir, season, nxt, observed) or {}).get("lineups"),
                "observed": observed}
    except Exception as exc:  # noqa: BLE001 - a lineup poll must never take the app down
        return {"error": str(exc)}


def _kickoff_date(text, observed):
    """SportsGambler's "Saturday 24 October 11:30" -> a date, the year taken
    from when it was observed (a December row naming January is next year).
    None when the text is not a date (FFScout states only the opponent)."""
    try:
        t = datetime.strptime(" ".join(str(text).split()[1:3]), "%d %B")
    except (ValueError, TypeError):
        return None
    y = observed.year + (1 if t.month < observed.month - 6 else 0)
    return t.replace(year=y).date()


def _feed_xis(season: str, gw: int, now, deadline=None) -> dict:
    """{club: {source, formation, status, observed, rows: [[name, ...], ...]}}
    -- the latest snapshot per club strictly before `now`, best feed first.

    The collectors stamp each row with FPL's NEXT gameweek at the moment they
    run, so a team sheet captured on Saturday afternoon -- the gameweek's own
    confirmed XI -- is filed under the FOLLOWING gameweek (528 of GW8's rows on
    23 Oct were GW7's Saturday sheets). A row whose stated kickoff falls before
    this gameweek's deadline belongs to the previous one and is dropped; it is
    not this week's line-up, confirmed or otherwise.

    Both archives are read: the repository's (written by the scheduled Action)
    and the server's own (`live_lineups_dir`, refreshed on the news loop), the
    newest observation winning -- see `refresh_feeds`.
    """
    from fpl_engine import lineup_feed as lf
    out: dict = {}
    for src in FORMATION_FEEDS:
        frames = []
        for base in (os.path.join(config.DATA_DIR, "collected"), live_lineups_dir()):
            try:
                frames.append(pd.read_csv(os.path.join(base, f"lineups_{src}", f"{season}.csv")))
            except (OSError, ValueError):
                continue
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True).drop_duplicates()
        d = d[(d["gw"] == gw) & d["formation"].notna() & d["player"].notna()]
        if d.empty:
            continue
        d = d.assign(observed=pd.to_datetime(d["observed_utc"], utc=True, format="ISO8601"))
        d = d[d["observed"] < now]
        if deadline is not None and "kickoff_text" in d:
            kd = [_kickoff_date(t, o) for t, o in zip(d["kickoff_text"], d["observed"])]
            stale = [k is not None and k < deadline.date() for k in kd]
            d = d[[not x for x in stale]]
        for abbr, g in d.groupby("team_abbr"):
            club = lf.ABBR.get(abbr, abbr)
            if club in out:
                continue                      # a better feed already has it
            g = g[g["observed"] == g["observed"].max()]
            rows = [[str(n) for n in r.sort_values("slot")["player"]]
                    for _, r in g.groupby("row", sort=True)]
            if sum(len(r) for r in rows) < 10:
                continue
            out[club] = {"source": src, "formation": str(g["formation"].iloc[0]),
                         "status": str(g["status"].iloc[0]),
                         "observed": g["observed"].max().isoformat(), "rows": rows}
    return out


def _last_played(conn, season: str, tm: dict, now) -> dict:
    """{club: formation} -- the shape each club actually lined up in last time,
    from BBC's archived team sheets. The reference a forecast is read against:
    a feed predicting a CHANGE of shape is the thing on this panel worth
    stopping for, because most clubs play the same way every week (71% of this
    season's club-matches were 4-2-3-1)."""
    from fpl_engine.ingest.odds import resolve_team
    try:
        rows = conn.execute(
            "SELECT m.kickoff_utc, l.team, l.formation FROM acq_bbc_lineup l "
            "JOIN acq_bbc_match m ON m.event_urn = l.event_urn "
            "WHERE m.season = ? AND l.is_starter = 1 AND l.formation IS NOT NULL "
            "AND m.kickoff_utc < ? GROUP BY m.event_urn, l.team "
            "ORDER BY m.kickoff_utc", (season, now.isoformat())).fetchall()
        names = {r[0]: r[1] for r in conn.execute(
            "SELECT name, short_name FROM team WHERE season = ?", (season,))}
    except Exception:  # noqa: BLE001 - no BBC archive: no reference, not an error
        return {}
    out: dict = {}
    for _kick, team, formation in rows:
        try:
            out[names[resolve_team(team, list(names))]] = formation
        except (KeyError, ValueError):
            continue                          # later matches overwrite earlier
    return out


def _lineups(conn, season: str, gw: int, pl: dict, tm: dict, start_p: dict) -> dict:
    """Each club's expected XI, in the formation a feed actually states, with
    the model's P(start) on every player.

    `lineup_feed.resolve` wants the same frame the backtest gives it --
    full_name, web_name and the club's short_name -- and returns a TRIPLE
    (resolved, unresolved, mismatched) keyed by (team, name). A resolution
    failure must never be able to present itself as a finding, so a miss is
    reported in `note` and marked on the player.
    """
    from fpl_engine import lineup_feed as lf
    now = pd.Timestamp.now(tz="UTC")
    try:
        k = conn.execute("SELECT MIN(kickoff_utc) FROM fixture WHERE season=? AND gw=?",
                         (season, gw)).fetchone()[0]
        deadline = (pd.Timestamp(k).tz_convert("UTC") - lf.DEADLINE_LEAD) if k else None
    except Exception:  # noqa: BLE001 - no fixture table: no staleness filter
        deadline = None
    feeds = _feed_xis(season, gw, now, deadline)
    if not feeds:
        return {"clubs": [], "gw": gw,
                "note": f"no predicted XI with a formation archived for GW{gw} yet"}
    players = pd.DataFrame([dict(r) for r in conn.execute(
        "SELECT p.player_id, p.full_name, p.web_name, p.team_id, p.position, t.short_name "
        "FROM player p JOIN team t ON t.team_id = p.team_id AND t.season = p.season "
        "WHERE p.season = ?", (season,))])
    if players.empty:
        return {"clubs": [], "gw": gw, "note": "no players loaded"}
    names_by_club = {c: {n for row in f["rows"] for n in row} for c, f in feeds.items()}
    try:
        resolved, unresolved, _ = lf.resolve(names_by_club, players)
    except Exception as exc:  # noqa: BLE001
        return {"clubs": [], "gw": gw, "note": f"could not resolve names: {exc}"}
    last = _last_played(conn, season, tm, now)

    abbr_to_team = {v: k for k, v in tm.items()}
    clubs = []
    for abbr, f in sorted(feeds.items()):
        names = names_by_club[abbr]
        ids = {resolved[(abbr, n)] for n in names if (abbr, n) in resolved}
        tid = abbr_to_team.get(abbr)
        squad = players[players["team_id"] == tid] if tid is not None else players.iloc[0:0]
        model: dict = {}
        for _, srow in squad.iterrows():
            pid = int(srow["player_id"])
            ps = start_p.get(pid)
            in_xi = pid in ids
            if in_xi or (ps is not None and ps >= 0.3):
                model[pid] = {
                    "player_id": pid, "name": srow["web_name"], "pos": srow["position"],
                    "predicted": in_xi,
                    "p_start": None if ps is None else round(float(ps), 2),
                    # only meaningful once the club's XI actually resolved:
                    # with nothing resolved every starter looks omitted
                    "disagree": len(ids) >= 8 and (
                        (in_xi and ps is not None and ps < 0.5)
                        or ((not in_xi) and ps is not None and ps >= 0.6))}
        # the feed's rows, keeper first, each row as the feed lists it --
        # right to left across the pitch, which with the keeper drawn at the
        # TOP is left to right on screen
        xi_rows = []
        for row in f["rows"]:
            cells = []
            for n in row:
                pid = resolved.get((abbr, n))
                m = model.get(pid) if pid is not None else None
                cells.append({"player_id": pid, "full_name": n,
                              "name": ((pl.get(pid) or {}).get("name") if pid is not None
                                       else n.split()[-1]),
                              "p_start": m["p_start"] if m else None,
                              "disagree": bool(m and m["disagree"]),
                              "resolved": pid is not None})
            xi_rows.append(cells)
        rows = sorted(model.values(), key=lambda r: (not r["predicted"], -(r["p_start"] or 0)))
        prev = last.get(abbr)
        clubs.append({"team": abbr, "team_id": tid, "observed": f["observed"],
                      "source": f["source"], "status": f["status"],
                      "formation": f["formation"], "last_formation": prev,
                      # the part worth an eye: a predicted change of shape
                      "shape_change": bool(prev and prev != f["formation"]),
                      "xi_rows": xi_rows,
                      "n_named": len(names), "n_resolved": len(ids), "rows": rows,
                      "disagreements": sum(1 for r in rows if r["disagree"])})
    note = None
    if unresolved:
        note = (f"{len(unresolved)} named players could not be matched to an "
                f"FPL id (e.g. {', '.join(n for _, n in unresolved[:3])})")
    return {"clubs": clubs, "gw": gw, "note": note,
            "sources": sorted({c["source"] for c in clubs})}


def _movers(gw: int, pl: dict, tm: dict) -> dict:
    hist = services._load_history()
    if len(hist) < 2:
        return {"rows": [], "from": None, "to": None}
    a, b = hist[-2], hist[-1]
    ga, gb = a.get("gws", {}).get(str(gw), {}), b.get("gws", {}).get(str(gw), {})
    rows = []
    for pid, ep in gb.items():
        prev = ga.get(pid)
        if prev is None:
            continue
        p = pl.get(int(pid))
        if not p:
            continue
        rows.append({"player_id": int(pid), "name": p["name"], "team": tm.get(p["team_id"]),
                     "pos": p["pos"], "before": round(float(prev), 2), "after": round(float(ep), 2),
                     "delta": round(float(ep) - float(prev), 2)})
    rows.sort(key=lambda r: -abs(r["delta"]))
    return {"rows": rows[:24], "from": a.get("built_at"), "to": b.get("built_at")}


def _feed_test(season: str) -> dict | None:
    """The running lineup-feed scorecard.

    The keys are `band_metrics.feed_accuracy` / `model_accuracy`; this used to
    read `band.feed_acc`, which does not exist, so the desk showed nothing
    even once the file had four gameweeks in it.
    """
    path = os.path.join(config.DATA_DIR, f"lineup_feed_{season}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    pooled = d.get("pooled") or {}
    return {"gws": sorted(int(g) for g in (d.get("gws") or {})),
            "band": pooled.get("band_metrics"),
            "all": pooled.get("all_metrics"),
            "priceable": pooled.get("priceable"),
            "rows_needed": pooled.get("rows_needed_for_estimate"),
            "updated": d.get("updated_utc")}


def payload(force: bool = False) -> dict:
    now = time.time()
    if not force and _cache["v"] is not None and now - _cache["t"] < TTL:
        return _cache["v"]
    season = config.CURRENT_SEASON
    status = services.status_payload()
    gw = status.get("editable_gw") or status.get("next_gw")
    conn = db.connect(config.DB_PATH)
    try:
        pl, tm = _names(conn, season)
        start_p = services._model_start_probs(season)
        out = {
            "season": season, "gw": gw, "deadline": (status.get("deadlines") or {}).get(str(gw)),
            "gw_in_progress": status.get("gw_in_progress"),
            "refresh": {**(status.get("auto_refresh") or {}),
                        "jobs_running": status.get("jobs_running"),
                        "proj_updated_at": status.get("proj_updated_at"),
                        "projected_gws": status.get("projected_gws")},
            "model": _model_meta(),
            "market": _market(conn, season, gw, tm),
            "news": _news(conn, season, pl, tm),
            "pressers": _pressers(conn, season, gw, pl, tm),
            "lineups": _lineups(conn, season, gw, pl, tm, start_p),
            "movers": _movers(gw, pl, tm),
            "feed_test": _feed_test(season),
            "built_at": now,
        }
    finally:
        conn.close()
    _cache["t"], _cache["v"] = now, out
    return out
