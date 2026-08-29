"""Data services for the web app.

Bridges fpl_engine (SQLite + models + optimiser) and the HTTP layer:
  * live FPL bootstrap / fixtures (in-memory TTL cache — ownership, FDR,
    shirt/photo codes, injury news)
  * point-in-time projections for a horizon, cached on disk per (season, gw)
  * chip-aware solve orchestration
  * draft persistence (one JSON document)
"""
from __future__ import annotations

import json
import os
import threading
import time

import pandas as pd

from fpl_engine import (config, db, manager, predict as predict_mod,
                        price_model)
from fpl_engine.http import get_text
from fpl_engine.optimise import chips, project
from fpl_engine.pipeline import next_gw, resolve_blend

from . import jobs

WEB_CACHE = os.path.join(config.DATA_DIR, "web_cache")
PROJ_PATH = os.path.join(WEB_CACHE, "projections.json")
DRAFTS_PATH = os.path.join(WEB_CACHE, "drafts.json")
HISTORY_PATH = os.path.join(WEB_CACHE, "projection_history.json")
HISTORY_KEEP = 40            # snapshots kept (one per build, >=1h apart)
MYTEAM_PATH = os.path.join(WEB_CACHE, "my_team.json")

FPL_BASE = "https://fantasy.premierleague.com/api"
_TTL = 600.0
# bumped whenever the API contract changes; the frontend compares it with
# its own build so a stale `python -m app` process is flagged, not puzzling
API_VERSION = "2026-08-27.1"

_mem: dict[str, tuple[float, object]] = {}
_bundle = None
_bundle_lock = threading.Lock()
_proj_lock = threading.Lock()


# --------------------------------------------------------------------------
# live FPL endpoints (memory-cached)
# --------------------------------------------------------------------------

def _live_json(path: str):
    now = time.monotonic()
    hit = _mem.get(path)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    data = json.loads(get_text(f"{FPL_BASE}/{path}", use_cache=False))
    _mem[path] = (now, data)
    return data


def _live_json_soft(path: str):
    """Like :func:`_live_json` but failure-tolerant: returns None instead of
    raising, and caches the miss so it is not re-fetched for the TTL. Used for
    per-entry endpoints that legitimately 404 (e.g. rivals without picks)."""
    now = time.monotonic()
    hit = _mem.get(path)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        data = json.loads(get_text(f"{FPL_BASE}/{path}", use_cache=False,
                                   retries=2))
    except Exception:
        data = None
    _mem[path] = (now, data)
    return data


def bootstrap() -> dict:
    return _live_json("bootstrap-static/")


def live_fixtures() -> list:
    return _live_json("fixtures/")


def _history_rates() -> dict[int, dict]:
    """Per player_code aggregates from the local match log (player_gw).

    Rows are deduped per (season, gw, fixture) so the fpl and vaastav sources
    never double count a match. Used for expected minutes and the fallback
    per-90 rates before the current season has enough data.
    """
    now = time.monotonic()
    hit = _mem.get("_history_rates")
    if hit and now - hit[0] < _TTL:
        return hit[1]
    conn = db.connect(config.DB_PATH)
    try:
        rows = conn.execute("""
            WITH m AS (
                SELECT player_code, MAX(minutes) AS minutes,
                       MAX(starts) AS starts, MAX(goals_scored) AS goals,
                       MAX(assists) AS assists, MAX(clean_sheets) AS cs,
                       MAX(xg) AS xg, MAX(xa) AS xa,
                       MAX(total_points) AS pts, MAX(kickoff_utc) AS kickoff_utc
                FROM player_gw
                WHERE player_code IS NOT NULL AND kickoff_utc IS NOT NULL
                GROUP BY player_code, season, gw, fixture_id
            ), r AS (
                SELECT m.*, ROW_NUMBER() OVER (
                    PARTITION BY player_code ORDER BY kickoff_utc DESC) AS rn
                FROM m
            )
            SELECT * FROM r WHERE rn <= 38 ORDER BY player_code, rn
        """).fetchall()
    except Exception:       # empty/uninitialised DB -> no history
        rows = []
    finally:
        conn.close()

    out: dict[int, dict] = {}
    by_code: dict[int, list] = {}
    for r in rows:
        by_code.setdefault(int(r["player_code"]), []).append(r)
    from datetime import datetime, timezone
    now_utc = datetime.now(timezone.utc)
    for code, ms in by_code.items():
        last5 = ms[:5]
        last10 = ms[:10]
        xmins = sum((r["minutes"] or 0) for r in last5) / max(1, len(last5))
        # across a season break (newest match > 3 weeks old) stale absences
        # are injury/rest, not selection: take the better of recent/long-run
        try:
            newest = datetime.fromisoformat(
                str(ms[0]["kickoff_utc"]).replace("Z", "+00:00"))
            if (now_utc - newest).days > 21:
                long_run = sum((r["minutes"] or 0) for r in ms) / len(ms)
                xmins = max(xmins, long_run)
        except (ValueError, TypeError):
            pass
        started = [
            (r["starts"] if r["starts"] is not None else
             (1.0 if (r["minutes"] or 0) >= 60 else 0.0))
            for r in last10]
        tot_min = sum((r["minutes"] or 0) for r in ms)
        played60 = sum(1 for r in ms if (r["minutes"] or 0) >= 60)
        per90 = (lambda k: (sum((r[k] or 0) for r in ms) / tot_min * 90.0)
                 if tot_min else 0.0)
        out[code] = {
            "xmins": round(xmins, 1),
            "start_rate": round(sum(started) / max(1, len(started)), 2),
            "recent_mins": [round(r["minutes"] or 0) for r in last5],
            # xG-based rates when the log carries FPL xG, else realised goals
            "g90": round(per90("xg" if any(r["xg"] is not None for r in ms) else "goals"), 2),
            "a90": round(per90("xa" if any(r["xa"] is not None for r in ms) else "assists"), 2),
            "cs90": round(sum((r["cs"] or 0) for r in ms if
                              (r["minutes"] or 0) >= 60) / max(1, played60), 2),
        }
    _mem["_history_rates"] = (now, out)
    return out


def _pk_share(order) -> float:
    """Crude penalty share from FPL's declared penalty order."""
    if order == 1:
        return 0.85
    if order == 2:
        return 0.10
    if order is not None:
        return 0.05
    return 0.0


def _model_start_probs(season: str | None = None) -> dict[int, float]:
    """Calibrated P(starts) per player, from the engine's minutes model.

    The UI has been showing ``start_rate`` — the share of his last ten matches
    a player started. That heuristic carries more than twice the log-loss of
    the model the engine already runs (0.57/0.54 against 0.28/0.25 on the two
    replayed seasons; AUC 0.90 against 0.95), so the number on screen was
    materially worse than what the pipeline already knew.

    Cached for the lifetime of the process; a failure here must not take the
    page down, so it degrades to the heuristic.
    """
    season = season or config.CURRENT_SEASON
    hit = _mem.get("_start_probs")
    if hit and hit[0] == season and time.time() - hit[1] < 900:
        return hit[2]
    out: dict[int, float] = {}
    try:
        from fpl_engine.xpts import engine as _eng, minutes_model as _mm
        from fpl_engine.pipeline import next_gw
        conn = db.connect()
        try:
            gw = next_gw(conn, season)
            as_of = _eng.first_kickoff(conn, season, gw)
            clf, meta = _mm.ensure(conn)
            mins = _mm.predict_gw(conn, season, as_of, clf, meta, gw=gw)
            if "p_start" in mins:
                out = {int(r.player_id): float(r.p_start)
                       for r in mins.itertuples()}
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - the page must render regardless
        out = {}
    _mem["_start_probs"] = (season, time.time(), out)
    return out


def players_payload() -> dict:
    """Static player + team info merged from live bootstrap (ownership, codes,
    injury status, per-90 rates) and the local DB (recent-minutes form used
    for expected minutes)."""
    bs = bootstrap()
    hist = _history_rates()
    start_p = _model_start_probs()
    teams = {t["id"]: {"id": t["id"], "name": t["name"], "short": t["short_name"],
                       "code": t["code"]} for t in bs["teams"]}
    pos_map = config.ELEMENT_TYPE_TO_POSITION
    fnum = lambda v: float(v or 0.0)
    players = []
    for e in bs["elements"]:
        pos = pos_map.get(e["element_type"])
        if pos is None:
            continue
        h = hist.get(e.get("code")) or {}
        mins = e.get("minutes") or 0
        # per-90 rates: xG-based once the season has data, else DB history
        # (which includes last season's backfill — the pre-season fallback).
        use_live = mins >= 270
        cbi = fnum(e.get("clearances_blocks_interceptions"))
        tackles = fnum(e.get("tackles"))
        recov = fnum(e.get("recoveries"))
        dc = cbi + tackles + (recov if pos in ("MID", "FWD") else 0.0)
        players.append({
            "id": e["id"], "web_name": e["web_name"],
            "name": f"{e['first_name']} {e['second_name']}",
            "team_id": e["team"], "position": pos,
            "price": e["now_cost"] / 10.0,
            "own": fnum(e.get("selected_by_percent")),
            "status": e.get("status"), "news": e.get("news") or "",
            "code": e.get("code"),
            "chance": e.get("chance_of_playing_next_round"),
            "mins": mins, "starts": e.get("starts") or 0,
            "form": fnum(e.get("form")), "ppg": fnum(e.get("points_per_game")),
            "total_points": e.get("total_points") or 0,
            # last completed gameweek's actual score - what a
            # "best XI in hindsight" has to be built from
            "event_points": e.get("event_points") or 0,
            "g90": round(fnum(e.get("expected_goals_per_90")), 2)
                   if use_live else h.get("g90", 0.0),
            "a90": round(fnum(e.get("expected_assists_per_90")), 2)
                   if use_live else h.get("a90", 0.0),
            "cs90": round(fnum(e.get("clean_sheets_per_90")), 2)
                    if use_live else h.get("cs90", 0.0),
            "dc90": round(dc / mins * 90.0, 2) if mins else 0.0,
            "pk_share": _pk_share(e.get("penalties_order")),
            # expected minutes: recent match log if pulled, else season
            # minutes-per-start from bootstrap as a crude fallback
            "xmins": h.get("xmins") if h.get("xmins") is not None else (
                round(min(90.0, mins / (e.get("starts") or 1)), 1)
                if mins else 0.0),
            "start_rate": h.get("start_rate"),
            # calibrated P(he is in the starting XI) from the minutes model;
            # start_rate above is the trailing heuristic, kept for comparison
            "p_start": start_p.get(int(e["id"])),
            "recent_mins": h.get("recent_mins") or [],
        })
    events = [{"id": ev["id"], "name": ev["name"],
               "deadline": ev["deadline_time"],
               "finished": ev["finished"], "is_next": ev["is_next"]}
              for ev in bs["events"]]
    return {"players": players, "teams": teams, "events": events}


def _strength_scale(vals: list[float]):
    """Min-max map a strength distribution onto the familiar 1–5 FDR scale."""
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return lambda x: 3.0
    return lambda x: round(1.0 + 4.0 * (x - lo) / (hi - lo), 1)


def _team_form_strengths() -> dict[tuple[str, int], dict]:
    """Per (club name, was_home) scoring/conceding rates from the local match
    log (last season's worth per venue, cross-season via the club name).

    Gives genuinely different attacking/defensive difficulty once data is
    pulled — FPL's bootstrap ships attack/defence strengths as zeroed
    placeholders early in the season. Empty dict when nothing is pulled yet.
    """
    now = time.monotonic()
    hit = _mem.get("_team_form")
    if hit and now - hit[0] < _TTL:
        return hit[1]
    conn = db.connect(config.DB_PATH)
    try:
        rows = conn.execute("""
            WITH m AS (
                SELECT COALESCE(t.code, t.name) AS key, tm.was_home AS was_home,
                       tm.goals_for AS gf, tm.goals_against AS ga,
                       ROW_NUMBER() OVER (PARTITION BY COALESCE(t.code, t.name),
                                          tm.was_home
                                          ORDER BY tm.kickoff_utc DESC) AS rn
                FROM team_match tm
                JOIN team t ON t.season = tm.season AND t.team_id = tm.team_id
                WHERE tm.goals_for IS NOT NULL AND tm.kickoff_utc IS NOT NULL
            )
            SELECT key, was_home, AVG(gf) AS gf, AVG(ga) AS ga, COUNT(*) AS n
            FROM m WHERE rn <= 19 GROUP BY key, was_home
        """).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    # keyed by FPL club code where the log carries it (rename-proof), else name
    out = {(r["key"], int(r["was_home"])): {"gf": r["gf"], "ga": r["ga"]}
           for r in rows if r["n"] >= 5}
    _mem["_team_form"] = (now, out)
    return out


def prices_payload(limit: int = 30) -> dict:
    """Who is about to rise or fall, and what that is actually worth.

    The model discriminates strongly - held out forward in time, the top-10
    ranked risers actually rose 67% / 75% of the time against a 2% base rate.
    But a signal is only worth what it converts to, and the conversion is
    deliberately deflating: a rise is realised only on sale, FPL returns the
    purchase price plus HALF the profit, and that half then buys a better
    squad for whatever season is left. The strongest riser in a week is worth
    about 0.2 points with 37 gameweeks to go.

    So `points` is reported next to every row. It belongs beside the solver's
    recommendation as a tie-breaker, not inside its objective.
    """
    try:
        with db.connect() as conn:
            d = price_model.predict(conn, config.CURRENT_SEASON)
            nxt = next_gw(conn, config.CURRENT_SEASON) or 1
    except Exception as exc:                      # untrained model, empty DB
        return {"ok": False, "error": str(exc), "risers": [], "fallers": []}
    if d is None or d.empty:
        return {"ok": False, "error": "no price data for the current season",
                "risers": [], "fallers": []}
    left = max(0, 38 - int(nxt) + 1)

    def rows(frame):
        out = []
        for r in frame.itertuples():
            out.append({
                "player_id": int(r.player_id),
                "price": round(float(r.price_m), 1),
                "p_rise": round(float(r.p_rise), 3),
                "p_fall": round(float(r.p_fall), 3),
                "score": round(float(r.score), 3),
                "e_delta": round(float(r.e_delta), 3),
                "points": round(price_model.points_value(
                    float(r.e_delta), left), 3),
            })
        return out

    return {"ok": True, "gw": int(d["gw"].iloc[0]), "gws_remaining": left,
            "pts_per_million_per_gw": price_model.POINTS_PER_MILLION_PER_GW,
            "risers": rows(d.head(limit)),
            "fallers": rows(d.tail(limit).iloc[::-1])}


def _fixture_odds() -> dict[int, dict]:
    """Bookmaker prices for this season's fixtures, keyed by FPL fixture id.

    Only fixtures a source has actually priced appear. The free Odds API tier
    covers upcoming matches only, so early in a season most of the grid has no
    entry - the UI must show that as "no price yet" rather than inventing one.
    """
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT fixture_id, source, p_home, p_draw, p_away, p_over25, "
                "lam_home, lam_away FROM match_odds WHERE season=?",
                (config.CURRENT_SEASON,)).fetchall()
    except Exception:
        return {}
    out = {}
    for r in rows:
        if r["fixture_id"] is None or r["p_home"] is None:
            continue
        out[int(r["fixture_id"])] = dict(r)
    return out


def _market_quotes() -> dict[int, dict]:
    """Prediction-market prices per fixture, for DISPLAY only.

    Deliberately separate from `_fixture_odds`: these never reach the model.
    Every Polymarket EPL market is team level, and the team-level odds channel
    is measured to be second order for player points. What they are good for is
    showing a human where a prediction market and a sportsbook disagree.
    """
    try:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT fixture_id, source, p_home, p_draw, p_away, volume, "
                "liquidity, observed_utc FROM market_quote WHERE season=?",
                (config.CURRENT_SEASON,)).fetchall()
    except Exception:
        return {}
    return {int(r["fixture_id"]): dict(r) for r in rows if r["p_home"] is not None}


def _odds_status(odds: dict[int, dict]) -> dict:
    """Whether the market view can actually say anything.

    Coverage here is normally poor and the reasons are structural, not a bug:
    football-data publishes odds for matches that have already been played, and
    The Odds API's free tier serves upcoming fixtures only if a key is set. The
    UI has to be told the difference, because a fixture with no price silently
    falling back to FDR looks exactly like a working market view.
    """
    out = {"key_set": bool(os.environ.get("ODDS_API_KEY")),
           "priced": len(odds), "sources": sorted(
               {r.get("source") for r in odds.values() if r.get("source")}),
           "upcoming": 0, "priced_upcoming": 0}
    try:
        with db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) n, SUM(CASE WHEN o.fixture_id IS NOT NULL "
                "  THEN 1 ELSE 0 END) p "
                "FROM fixture f LEFT JOIN match_odds o "
                "  ON o.season=f.season AND o.fixture_id=f.fixture_id "
                "WHERE f.season=? AND f.finished=0", (config.CURRENT_SEASON,)
            ).fetchone()
            out["upcoming"] = int(row["n"] or 0)
            out["priced_upcoming"] = int(row["p"] or 0)
    except Exception:
        pass
    return out


def fixtures_payload() -> dict:
    """Team-centric fixture grid for the heatmap.

    Each fixture carries FPL's coarse integer FDR plus continuous 1–5
    difficulty scores: `diff` (opponent overall strength), `diff_att` (how
    hard the opponent is to score against) and `diff_def` (how hard it is to
    keep a clean sheet against them). Attacking/defensive scores come from
    the local match log (venue-split goals conceded/scored) once data is
    pulled; before that they fall back to bootstrap strengths. Every score
    is averaged with FPL's curated FDR so the two never contradict.
    """
    bs = bootstrap()
    names = {t["id"]: t["name"] for t in bs["teams"]}
    codes = {t["id"]: t.get("code") for t in bs["teams"]}
    ovr, att, dfn = {}, {}, {}
    for t in bs["teams"]:
        for venue in ("home", "away"):
            ovr[(t["id"], venue)] = t.get(f"strength_overall_{venue}") or 0
            att[(t["id"], venue)] = t.get(f"strength_attack_{venue}") or 0
            dfn[(t["id"], venue)] = t.get(f"strength_defence_{venue}") or 0
    # early-season bootstraps ship attack/defence strengths as all-zero
    # placeholders — fall back to the overall ratings so the att/def views
    # degrade to "overall" instead of a flat 3.0
    if len(set(att.values())) <= 1:
        att = dict(ovr)
    if len(set(dfn.values())) <= 1:
        dfn = dict(ovr)
    s_ovr = _strength_scale(list(ovr.values()))
    s_att = _strength_scale(list(att.values()))
    s_dfn = _strength_scale(list(dfn.values()))

    form = _team_form_strengths()
    s_gf = s_ga = None
    if form:
        s_gf = _strength_scale([v["gf"] for v in form.values()])
        s_ga = _strength_scale([v["ga"] for v in form.values()])

    def cell(opp: int, home: bool, f: dict) -> dict:
        venue = "away" if home else "home"   # the opponent plays at the other end
        fdr = f.get("team_h_difficulty" if home else "team_a_difficulty")
        # anchor the continuous score to FPL's curated FDR: average the
        # strength-derived scale with the (integer) FDR when both exist
        blend = (lambda s: round((s + fdr) / 2.0, 1) if fdr else s)
        d_att = s_dfn(dfn.get((opp, venue), 0))
        d_def = s_att(att.get((opp, venue), 0))
        opp_form = (form.get((codes.get(opp), 0 if home else 1))
                    or form.get((names.get(opp), 0 if home else 1)))
        if opp_form:
            # opponent conceding little at their venue -> hard to attack
            d_att = round(6.0 - s_ga(opp_form["ga"]), 1)
            # opponent scoring a lot -> hard to keep a clean sheet
            d_def = s_gf(opp_form["gf"])
        o = odds.get(f.get("id"))
        q = quotes.get(f.get("id"))
        return {"opp": opp, "home": home, "fdr": fdr,
                "diff": blend(s_ovr(ovr.get((opp, venue), 0))),
                "diff_att": blend(d_att),
                "diff_def": blend(d_def),
                # the market's own view, de-margined, when a bookmaker price
                # exists for this fixture. `p_win` is from THIS team's side.
                "odds": None if not o else {
                    "p_win": round(o["p_home"] if home else o["p_away"], 3),
                    "p_draw": round(o["p_draw"], 3),
                    "p_lose": round(o["p_away"] if home else o["p_home"], 3),
                    "p_over25": (round(o["p_over25"], 3)
                                 if o["p_over25"] is not None else None),
                    "xg": round(o["lam_home"] if home else o["lam_away"], 2),
                    "xg_against": round(o["lam_away"] if home else o["lam_home"], 2),
                    "source": o["source"]},
                # a prediction market's view of the same fixture, shown
                # beside the bookmaker's and never mixed into the model
                "market": None if not q else {
                    "p_win": round(q["p_home"] if home else q["p_away"], 3),
                    "p_draw": round(q["p_draw"], 3),
                    "p_lose": round(q["p_away"] if home else q["p_home"], 3),
                    "liquidity": q["liquidity"], "volume": q["volume"],
                    "source": q["source"]},
                "kickoff": f.get("kickoff_time"), "finished": f.get("finished")}

    odds = _fixture_odds()
    quotes = _market_quotes()
    fx = live_fixtures()
    grid: dict[int, dict[int, list]] = {}
    for f in fx:
        gw = f.get("event")
        if gw is None:
            continue
        h, a = f["team_h"], f["team_a"]
        grid.setdefault(h, {}).setdefault(gw, []).append(cell(a, True, f))
        grid.setdefault(a, {}).setdefault(gw, []).append(cell(h, False, f))
    return {"grid": {str(t): {str(g): v for g, v in gws.items()}
                     for t, gws in grid.items()},
            "odds_status": {**_odds_status(odds), "market_quotes": len(quotes)}}


def entry_payload(entry_id: int) -> dict:
    try:
        info = manager.fetch_entry(entry_id)
    except Exception:
        return {"entry_id": entry_id, "exists": False}
    state = squad_state(entry_id)
    out = {"entry_id": entry_id, "exists": True,
           "team_name": info.get("name"),
           "player_name": f"{info.get('player_first_name', '')} "
                          f"{info.get('player_last_name', '')}".strip(),
           "overall_points": info.get("summary_overall_points"),
           "overall_rank": info.get("summary_overall_rank"),
           "squad": None, "squad_source": None}
    if state is not None:
        out.update({"squad": state["squad"], "bank": state["bank"],
                    "free_transfers": state["free_transfers"],
                    "unlimited_transfers": bool(state.get("unlimited_transfers")),
                    "picks_gw": state.get("gw"),
                    "squad_source": state.get("source", "public")})
    return out


# --------------------------------------------------------------------------
# "my team": the squad the planner/solver should start from.
#
# Before a gameweek deadline passes, FPL's public API does not expose an
# entry's picks (they are private), so pre-season / pre-deadline squads can
# only come from (a) the authenticated my-team endpoint using the user's own
# browser session cookie, or (b) a manually entered squad. Both are stored in
# one local file and take precedence over the public picks endpoints.
# --------------------------------------------------------------------------

def load_my_team() -> dict | None:
    if os.path.exists(MYTEAM_PATH):
        with open(MYTEAM_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_my_team(doc: dict | None) -> dict | None:
    os.makedirs(WEB_CACHE, exist_ok=True)
    if doc is None:
        if os.path.exists(MYTEAM_PATH):
            os.remove(MYTEAM_PATH)
        return None
    doc = {"entry_id": doc.get("entry_id"),
           "squad": doc["squad"],                 # [{element, selling_price, ...}]
           "bank": float(doc.get("bank") or 0.0),
           "free_transfers": int(doc.get("free_transfers") or 1),
           "unlimited_transfers": bool(doc.get("unlimited_transfers")),
           "team_value": doc.get("team_value"),
           "source": doc.get("source", "manual"),
           "saved_at": time.time()}
    tmp = MYTEAM_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, MYTEAM_PATH)
    return doc


def import_my_team_with_cookie(entry_id: int, cookie: str) -> dict:
    """Fetch the authenticated my-team endpoint with the user's own FPL
    browser session cookie (never stored — only the resulting squad is)."""
    import urllib.request
    req = urllib.request.Request(
        f"{FPL_BASE}/my-team/{entry_id}/",
        headers={"Cookie": cookie.strip(),
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    if "picks" not in data:
        raise ValueError("response has no picks — cookie rejected?")
    return _save_my_team_json(entry_id, data, "fpl-login")


def import_my_team_from_payload(payload, entry_id: int | None = None) -> dict:
    """Squad from what the OpenFPL bookmarklet copies to the clipboard —
    ``{"entry": id, "my_team": <my-team response>}`` — or a raw my-team
    response pasted by hand. No cookie ever touches this app."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    data = payload.get("my_team") if "my_team" in payload else payload
    eid = payload.get("entry") or entry_id
    if not isinstance(data, dict) or "picks" not in data:
        raise ValueError("that isn't FPL my-team data — click the bookmark on "
                         "fantasy.premierleague.com while logged in, then paste "
                         "exactly what it copied")
    return _save_my_team_json(int(eid) if eid else None, data, "bookmarklet")


def _save_my_team_json(entry_id: int | None, data: dict, source: str) -> dict:
    """Normalise an FPL my-team response into the saved-squad document."""
    squad = [{
        "element": p["element"],
        "selling_price": p.get("selling_price", 0) / 10.0,
        "purchase_price": p.get("purchase_price", 0) / 10.0,
        "is_captain": bool(p.get("is_captain")),
        "is_vice": bool(p.get("is_vice_captain")),
        "multiplier": p.get("multiplier", 1),
    } for p in data["picks"]]
    tr = data.get("transfers", {}) or {}
    limit = tr.get("limit")
    made = tr.get("made", 0) or 0
    # before the GW1 deadline FPL grants unlimited free transfers
    unlimited = tr.get("status") == "unlimited"
    return save_my_team({
        "entry_id": entry_id, "squad": squad,
        "bank": (tr.get("bank", 0) or 0) / 10.0,
        "free_transfers": max(0, (limit or 1) - made) if limit is not None else 1,
        "unlimited_transfers": unlimited,
        "team_value": (tr.get("value") or 0) / 10.0 or None,
        "source": source})


def squad_state(entry_id: int) -> dict | None:
    """Current squad for an entry: public picks if available (post-deadline,
    authoritative), else the locally saved my-team (cookie import or manual)."""
    state = None
    try:
        state = manager.current_squad(entry_id)
    except Exception:
        state = None
    if state is not None:
        state["source"] = "public"
        return state
    mine = load_my_team()
    if mine and (mine.get("entry_id") in (None, entry_id)):
        return {"entry_id": entry_id, "name": None, "gw": None,
                "bank": mine["bank"], "squad": mine["squad"],
                "free_transfers": mine["free_transfers"],
                "unlimited_transfers": bool(mine.get("unlimited_transfers")),
                "source": mine.get("source", "manual")}
    return None


# --------------------------------------------------------------------------
# projections (disk-cached per season+gw)
# --------------------------------------------------------------------------

def _load_proj_cache() -> dict:
    if os.path.exists(PROJ_PATH):
        with open(PROJ_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"season": config.CURRENT_SEASON, "gws": {}, "players": {}}


def _save_proj_cache(cache: dict) -> None:
    os.makedirs(WEB_CACHE, exist_ok=True)
    tmp = PROJ_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, PROJ_PATH)


def _get_bundle():
    global _bundle
    with _bundle_lock:
        if _bundle is None:
            _bundle = predict_mod.load_models()
        return _bundle


def projections_payload() -> dict:
    cache = _load_proj_cache()
    return cache


def build_projections(job_id: str | None, gws: list[int], *,
                      force: bool = False, blend=None) -> dict:
    """Compute and cache per-player projections for each gw in ``gws``.

    Serialised under a lock so concurrent solves don't duplicate model runs.
    """
    season = config.CURRENT_SEASON
    with _proj_lock:
        cache = _load_proj_cache()
        if cache.get("season") != season:
            cache = {"season": season, "gws": {}, "players": {}}
        todo = [g for g in gws if force or str(g) not in cache["gws"]]
        if not todo:
            return cache
        conn = db.connect(config.DB_PATH)
        try:
            n_fx = conn.execute(
                "SELECT COUNT(*) FROM fixture WHERE season=?",
                (season,)).fetchone()[0]
            if not n_fx:
                raise RuntimeError(
                    "No fixture data in the local database — run a data pull "
                    "(⟳ Data, top right) first.")
            bundle = _get_bundle()
            retrained, weight = resolve_blend(conn, season, blend)
            from fpl_engine.pipeline import xpts_weight
            xw = xpts_weight()
            pens = None
            if xw:
                try:   # first-choice penalty takers from the live bootstrap
                    pens = {e["id"]: e.get("penalties_order")
                            for e in bootstrap()["elements"]
                            if e.get("penalties_order")}
                except Exception:
                    pens = None
            for i, g in enumerate(todo):
                if job_id:
                    jobs.progress(job_id, f"Projecting GW{g} "
                                  f"({i + 1}/{len(todo)})…",
                                  pct=0.05 + 0.6 * (i / max(1, len(todo))))
                df = project.horizon_projections(
                    conn, season, [g], bundle=bundle, retrained=retrained,
                    blend=weight, xpts_w=xw, penalty_takers=pens)
                if df.empty:
                    continue
                for r in df.itertuples():
                    pid = str(int(r.player_id))
                    rec = cache["players"].setdefault(pid, {
                        "player_id": int(r.player_id), "player": r.player,
                        "position": r.position, "team_id": int(r.team_id),
                        "team": r.team, "price": float(r.price),
                        "available": float(r.available), "ep": {}})
                    rec["price"] = float(r.price)
                    rec["available"] = float(r.available)
                    xm = getattr(r, "xmins", None)
                    xm = None if xm is None or pd.isna(xm) else float(xm)
                    # per gameweek, not one scalar: this loop runs once per gw,
                    # so a single field was simply the last gameweek's value
                    # overwriting the rest — which is part of why the player
                    # card looked identical down every row.
                    rec.setdefault("xm", {})[str(g)] = (
                        None if xm is None else round(xm, 1))
                    rec["xmins"] = xm          # kept: older callers read this
                    rec["ep"][str(g)] = round(float(getattr(r, f"ep_gw{g}")), 3)
                cache["gws"][str(g)] = {"built_at": time.time()}
            cache["updated_at"] = time.time()
            _save_proj_cache(cache)
            _append_history(cache)
        finally:
            conn.close()
        return cache


def _load_history() -> list:
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def _append_history(cache: dict) -> None:
    """Keep a trail of projection builds so the UI can show whether the model
    is moving up or down on a player (one snapshot per build; builds within
    an hour replace the previous snapshot)."""
    snap = {"built_at": cache.get("updated_at") or time.time(),
            "gws": {g: {pid: rec["ep"][g] for pid, rec in cache["players"].items()
                        if g in rec["ep"]} for g in cache["gws"]}}
    hist = _load_history()
    if hist and snap["built_at"] - hist[-1]["built_at"] < 3600:
        hist[-1] = snap
    else:
        hist.append(snap)
    hist = hist[-HISTORY_KEEP:]
    tmp = HISTORY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(hist, f)
    os.replace(tmp, HISTORY_PATH)


def projection_history_payload() -> dict:
    return {"snapshots": _load_history()}


def _proj_frame(cache: dict, gws: list[int], decay: float) -> pd.DataFrame:
    rows = []
    for rec in cache["players"].values():
        eps = {g: rec["ep"].get(str(g)) for g in gws}
        if all(v is None for v in eps.values()):
            continue
        row = {"player_id": rec["player_id"], "player": rec["player"],
               "position": rec["position"], "team_id": rec["team_id"],
               "team": rec["team"], "price": rec["price"],
               "available": rec["available"]}
        total = 0.0
        for i, g in enumerate(gws):
            v = eps[g] or 0.0
            row[f"ep_gw{g}"] = v
            total += (decay ** i) * v
        row["ep_total"] = total
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# solve orchestration
# --------------------------------------------------------------------------

def unlimited_applies(flag: bool, played_gws: int, first_gw: int, next_gw_: int) -> bool:
    """FPL's pre-deadline unlimited transfers make moves free only in the
    season's opening gameweek: the flag (a snapshot from the squad import)
    applies iff nothing has been played yet and the plan starts at that
    next gameweek. A horizon starting later, or a stale flag after the
    first deadline, gets normal FT/hit accounting."""
    return bool(flag) and played_gws == 0 and first_gw == next_gw_


def run_solve(job_id: str, params: dict) -> dict:
    season = config.CURRENT_SEASON
    conn = db.connect(config.DB_PATH)
    try:
        start = next_gw(conn, season)
        solve_from = int(params.get("solve_from") or start)
        horizon = int(params.get("horizon") or 5)
        scheduled = [r["gw"] for r in conn.execute(
            "SELECT DISTINCT gw FROM fixture WHERE season=? AND gw>=? AND gw "
            "IS NOT NULL ORDER BY gw", (season, max(start, solve_from)))]
        gws = scheduled[:horizon] or [start]
        played = conn.execute(
            "SELECT COUNT(DISTINCT gw) FROM fixture WHERE season=? AND finished=1",
            (season,)).fetchone()[0]
    finally:
        conn.close()

    jobs.progress(job_id, f"Planning horizon GW{gws[0]}–GW{gws[-1]}", pct=0.02)
    cache = build_projections(job_id, gws, blend=params.get("blend"))
    decay = float(params.get("decay") or 0.85)
    proj = _proj_frame(cache, gws, decay)
    if proj.empty:
        raise RuntimeError("No projections available — run a data pull first.")

    jobs.progress(job_id, "Fetching entry state…", pct=0.68)
    entry_id = params.get("entry")
    initial, bank, fts = None, float(params.get("budget") or 100.0), 1
    entry_state = None
    if entry_id:
        entry_state = squad_state(int(entry_id))
        if entry_state is not None:
            initial = {p["element"]: p["selling_price"]
                       for p in entry_state["squad"]}
            bank = entry_state["bank"]
            fts = entry_state["free_transfers"]
    # An explicit squad overrides the live entry. The Planner needs this: to
    # ask "what is the best Free Hit in GW9" the solver has to start from the
    # squad the DRAFT reaches by GW9, not from the fifteen currently owned.
    seed = params.get("initial_squad")
    if seed:
        initial = {int(k): float(v) for k, v in seed.items()}
        if params.get("bank") is not None:
            bank = float(params["bank"])
    if params.get("free_transfers") is not None:
        fts = int(params["free_transfers"])

    locked = set(params.get("locked") or [])
    avoid = set(params.get("avoid") or [])
    must_keep = set(initial or {}) | locked
    proj_p = project.prune(
        proj, keep_per_position=int(params.get("keep_per_position") or 30),
        must_keep=must_keep)

    chip_gws: dict[str, list[int]] = {}
    chip_force: dict[str, int] = {}
    for c, cfg in (params.get("chips") or {}).items():
        if c not in chips.CHIPS or not cfg or not cfg.get("enabled"):
            continue
        allowed = cfg.get("gws") or gws
        chip_gws[c] = [g for g in allowed if g in gws]
        if cfg.get("force") and int(cfg["force"]) in gws:
            chip_force[c] = int(cfg["force"])

    unlimited = unlimited_applies(
        bool((entry_state or {}).get("unlimited_transfers")), played, gws[0], start)
    jobs.progress(job_id, f"Solving MILP over {len(proj_p)} players × "
                  f"{len(gws)} GWs…", pct=0.72)
    # Three plans means three *playstyles* to choose between, not three
    # near-identical optima; n_plans>1 with no explicit styles uses the presets.
    styles = params.get("playstyles")
    if styles is None and int(params.get("n_plans") or 1) > 1:
        styles = chips.DEFAULT_PLAYSTYLES[:int(params["n_plans"])]
    solver = (chips.optimise_playstyles if styles else chips.optimise_with_chips)
    extra = {"styles": styles} if styles else {"n_plans": int(params.get("n_plans") or 1)}
    plans = solver(
        proj_p, gws,
        initial=initial, bank=bank, free_transfers=fts,
        budget=float(params.get("budget") or 100.0),
        decay=decay,
        hit_cost=float(params.get("hit_cost") or 4.0),
        bench_weight=float(params.get("bench_weight") or 0.1),
        ft_value=float(params.get("ft_value") or 1.5),
        max_transfers_per_gw=int(params.get("max_transfers") or 3),
        time_limit=int(params.get("time_limit") or 60),
        chip_gws=chip_gws, chip_force=chip_force,
        chip_reserve={k: float(v) for k, v in
                      (params.get("chip_reserve") or {}).items()},
        locked=locked, avoid=avoid,
        banned_clubs=set(params.get("banned_teams") or []),
        sell_clubs=set(params.get("sell_teams") or []),
        forced_in={int(g): v for g, v in (params.get("forced_in") or {}).items()},
        forced_out={int(g): v for g, v in (params.get("forced_out") or {}).items()},
        min_ft={int(g): int(v) for g, v in (params.get("min_ft") or {}).items()},
        **extra,
        # pre-GW1 deadline: FPL grants unlimited free transfers, so first-gw
        # moves cost nothing and don't bank into the next gw — but only for
        # the season's first gameweek itself, never a later-starting horizon
        unlimited_first=unlimited,
        on_progress=lambda m: jobs.progress(job_id, m),
    )
    if not plans:
        raise RuntimeError("Solver found no feasible plan — check constraints "
                           "(locked/avoid/banned may conflict).")
    return {
        "mode": "optimise-transfers" if initial else "build-from-scratch",
        "entry_id": entry_id, "gws": gws,
        "state": {"bank": bank, "free_transfers": fts,
                  "unlimited_transfers": unlimited,
                  "team_name": (entry_state or {}).get("name") if entry_state
                  else None},
        "plans": [{"objective": p.objective, "status": p.status,
                   "per_gw": p.per_gw, "style": p.style,
                   "style_label": p.style_label, "style_note": p.style_note,
                   "total_ep": p.total_ep} for p in plans],
    }


def run_pull(job_id: str, understat: bool = False) -> dict:
    from fpl_engine import progress
    from fpl_engine.pipeline import pull
    jobs.progress(job_id, "Pulling FPL live data + backfill…", pct=0.1)
    db.init_db(config.DB_PATH)
    # relay the engine's step/progress lines into the job so the UI shows
    # what the (possibly several-minute) pull is doing
    if job_id:
        progress.set_listener(
            lambda m: jobs.progress(job_id, m.replace("[fpl] ", "").strip()))
    try:
        # db.session commits on success — a bare connect() would silently
        # roll back the entire pull when the connection closes
        with db.session(config.DB_PATH) as conn:
            summary = pull(conn, use_cache=True, with_understat=understat)
    finally:
        progress.set_listener(None)
    # prices/status changed -> projections stale
    if os.path.exists(PROJ_PATH):
        os.remove(PROJ_PATH)
    _mem.clear()
    jobs.progress(job_id, "Pull complete; projection cache invalidated.", pct=1.0)
    return {"summary": {k: str(v) for k, v in summary.items()}}


# --------------------------------------------------------------------------
# mini league analysis
# --------------------------------------------------------------------------

def league_payload(league_id: int, gw: int | None = None,
                   limit: int = 20) -> dict:
    """Classic mini-league standings plus each rival's public squad.

    Picks are public only for gameweeks whose deadline has passed; before
    that (and pre-season) entries come back without picks and the frontend
    degrades to standings-only. Everything is TTL-cached in ``_mem`` via
    ``_live_json`` so refreshes are cheap.
    """
    st = _live_json(f"leagues-classic/{league_id}/standings/")
    league = st.get("league") or {}
    results = (st.get("standings") or {}).get("results") or []
    pre_season = not results
    if pre_season:  # before GW1 entries live in new_entries instead
        results = [{
            "entry": r.get("entry"), "entry_name": r.get("entry_name"),
            "player_name": f"{r.get('player_first_name', '')} "
                           f"{r.get('player_last_name', '')}".strip(),
            "rank": None, "last_rank": None, "total": 0, "event_total": 0,
        } for r in ((st.get("new_entries") or {}).get("results") or [])]

    evs = bootstrap()["events"]
    finished = [e["id"] for e in evs if e.get("finished")]
    current = next((e["id"] for e in evs if e.get("is_current")), None)
    pick_gw = gw or current or (finished[-1] if finished else None)

    # fetch every rival's picks + history concurrently: the FPL API can be
    # slow on matchdays, so overlapping the waits matters far more than the
    # (still throttled) request pacing
    from concurrent.futures import ThreadPoolExecutor
    rows = [r for r in results[:limit] if r.get("entry")]

    def _fetch(r):
        eid = r["entry"]
        picks = (_live_json_soft(f"entry/{eid}/event/{pick_gw}/picks/")
                 if pick_gw else None)
        return r, picks, _live_json_soft(f"entry/{eid}/history/")

    with ThreadPoolExecutor(max_workers=8) as pool:
        fetched = list(pool.map(_fetch, rows))

    entries = []
    for r, picks, hist in fetched:
        eid = r["entry"]
        eh = (picks or {}).get("entry_history") or {}
        entries.append({
            "entry": eid, "team": r.get("entry_name"),
            "manager": r.get("player_name"),
            "rank": r.get("rank"), "last_rank": r.get("last_rank"),
            "total": r.get("total"), "event_total": r.get("event_total"),
            "picks": [{"element": p["element"],
                       "multiplier": p.get("multiplier", 0),
                       "is_captain": bool(p.get("is_captain")),
                       "is_vice": bool(p.get("is_vice_captain"))}
                      for p in (picks or {}).get("picks", [])],
            "active_chip": (picks or {}).get("active_chip"),
            "chips_used": [c.get("name") for c in (hist or {}).get("chips", [])],
            "history": [{
                "gw": h.get("event"), "points": h.get("points"),
                "total": h.get("total_points"),
                "overall_rank": h.get("overall_rank"),
                "transfers": h.get("event_transfers") or 0,
                "hit_points": h.get("event_transfers_cost") or 0,
                "bench_points": h.get("points_on_bench") or 0,
                "value": (h.get("value") or 0) / 10.0,
            } for h in (hist or {}).get("current", [])],
            "bank": (eh.get("bank") or 0) / 10.0,
            "value": (eh.get("value") or 0) / 10.0,
            "gw_points": eh.get("points"),
        })
    return {"league_id": league_id, "name": league.get("name"),
            "gw": pick_gw, "pre_season": pre_season,
            "total_entries": len(results), "analysed": len(entries),
            "entries": entries}


# --------------------------------------------------------------------------
# drafts
# --------------------------------------------------------------------------

def load_drafts() -> dict:
    if os.path.exists(DRAFTS_PATH):
        with open(DRAFTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"drafts": [], "updated_at": None}


def save_drafts(doc: dict) -> dict:
    os.makedirs(WEB_CACHE, exist_ok=True)
    doc = {"drafts": doc.get("drafts") or [], "updated_at": time.time()}
    tmp = DRAFTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    os.replace(tmp, DRAFTS_PATH)
    return doc


def status_payload() -> dict:
    season = config.CURRENT_SEASON
    conn = db.connect(config.DB_PATH)
    try:
        n_players = conn.execute(
            "SELECT COUNT(*) FROM player WHERE season=?", (season,)).fetchone()[0]
        gw = next_gw(conn, season)
        scheduled = [r["gw"] for r in conn.execute(
            "SELECT DISTINCT gw FROM fixture WHERE season=? AND gw IS NOT NULL "
            "ORDER BY gw", (season,))]
    finally:
        conn.close()
    if not scheduled:
        # no data pulled yet — fall back to the live FPL API so the planner
        # and fixture heatmap still know the calendar
        try:
            scheduled = sorted({f["event"] for f in live_fixtures()
                                if f.get("event") is not None})
            nxt = next((e["id"] for e in bootstrap()["events"]
                        if e.get("is_next")), None)
            gw = nxt or gw
        except Exception:
            pass
    cache = _load_proj_cache()
    return {"season": season, "next_gw": gw, "scheduled_gws": scheduled,
            "api_version": API_VERSION,
            "db_ready": bool(n_players),
            "projected_gws": sorted(int(g) for g in cache.get("gws", {})),
            "proj_updated_at": cache.get("updated_at"),
            "default_entry": manager.DEFAULT_ENTRY,
            "jobs_running": [j["kind"] for j in jobs.running()]}
