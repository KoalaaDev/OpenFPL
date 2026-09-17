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
import time
from datetime import datetime, timezone

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


def _feed_xis(season: str, gw: int, now) -> dict:
    """{club: {source, formation, status, observed, rows: [[name, ...], ...]}}
    -- the latest snapshot per club strictly before `now`, best feed first."""
    from fpl_engine import lineup_feed as lf
    out: dict = {}
    for src in FORMATION_FEEDS:
        path = os.path.join(config.DATA_DIR, "collected", f"lineups_{src}", f"{season}.csv")
        try:
            d = pd.read_csv(path)
        except (OSError, ValueError):
            continue
        d = d[(d["gw"] == gw) & d["formation"].notna() & d["player"].notna()]
        if d.empty:
            continue
        d = d.assign(observed=pd.to_datetime(d["observed_utc"], utc=True, format="ISO8601"))
        d = d[d["observed"] < now]
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
    feeds = _feed_xis(season, gw, now)
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
