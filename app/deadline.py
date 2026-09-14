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


def _lineups(conn, season: str, gw: int, pl: dict, tm: dict, start_p: dict) -> dict:
    """RotoWire's latest predicted XI per club against the model's P(start)."""
    from fpl_engine import lineup_feed as lf
    try:
        archive = lf.load_archive(season)
    except Exception:  # noqa: BLE001
        return {"clubs": [], "note": "no lineup archive"}
    if archive.empty:
        return {"clubs": [], "note": "no lineup archive yet"}
    forecasts = lf.pre_deadline_forecasts(archive, gw, pd.Timestamp.now(tz="UTC"))
    players = pd.DataFrame([{"player_id": pid, "web_name": v["name"], "team_id": v["team_id"]}
                            for pid, v in pl.items()])
    abbr_to_team = {v: k for k, v in tm.items()}
    clubs = []
    for abbr, (when, names) in sorted(forecasts.items()):
        tid = abbr_to_team.get(abbr)
        squad = players[players["team_id"] == tid] if tid is not None else players.iloc[0:0]
        try:
            resolved = lf.resolve({abbr: names}, players if tid is None else squad)
        except Exception:  # noqa: BLE001
            resolved = {}
        ids = set()
        for v in resolved.values():
            ids |= {int(x) for x in (v if isinstance(v, (set, list)) else [v]) if x is not None}
        rows = []
        for _, s in squad.iterrows():
            pid = int(s["player_id"])
            ps = start_p.get(pid)
            in_xi = pid in ids
            if in_xi or (ps is not None and ps >= 0.3):
                rows.append({"player_id": pid, "name": s["web_name"], "predicted": in_xi,
                             "p_start": None if ps is None else round(float(ps), 2),
                             "disagree": (in_xi and ps is not None and ps < 0.5)
                                         or ((not in_xi) and ps is not None and ps >= 0.6)})
        rows.sort(key=lambda r: (not r["predicted"], -(r["p_start"] or 0)))
        clubs.append({"team": abbr, "observed": when, "n_named": len(names),
                      "n_resolved": len(ids), "rows": rows,
                      "disagreements": sum(1 for r in rows if r["disagree"])})
    return {"clubs": clubs, "gw": gw, "note": None}


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
    path = os.path.join(config.DATA_DIR, f"lineup_feed_{season}.json")
    try:
        d = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    pooled = d.get("pooled") or {}
    return {"gws": sorted(int(g) for g in (d.get("gws") or d.get("reports") or {}).keys()) if isinstance(d.get("gws") or d.get("reports"), dict) else None,
            "pooled": pooled}


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
