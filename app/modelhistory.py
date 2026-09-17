"""Is the model getting better? — the admin's answer, assembled from what the
pipeline already writes after every gameweek.

Three sources, none of them new work:

  * ``data/postmortem_<season>_gw<n>.json`` — what the model believed against
    what happened: rank correlation, how many of its top 20 actually finished
    in the real top 20, its captain against the best possible one, and the
    per-component predicted-vs-actual sums that the Round 17 calibration audit
    reads. Written by ``fpl_engine.postmortem``.
  * ``data/lineup_feed_<season>.json`` — the running scorecard for the one
    paid lever this repo has priced (a predicted-XI feed), in the ambiguous
    band where minutes value actually lives. Written by
    ``fpl_engine.lineup_feed``.
  * ``models/xpts/*`` — the model actually serving: which features it has,
    what they are worth to it, when it was trained and on what.

Both JSON series are produced automatically by the scheduled refresh
(``scheduler.score_finished_gameweeks``), so this page fills itself in.

Read the trend, not a gameweek. A single gameweek's Spearman moves by more
than any real improvement does — that is the standing lesson of this repo,
and the page says so where an admin will read it.
"""
from __future__ import annotations

import json
import os
import time

from fpl_engine import config

_cache: dict = {"t": 0.0, "v": None}
TTL = 120.0


def _load(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def postmortems(season: str) -> list[dict]:
    """One row per scored gameweek, oldest first."""
    rows = []
    try:
        names = os.listdir(config.DATA_DIR)
    except OSError:
        return rows
    prefix = f"postmortem_{season}_gw"
    for name in names:
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        d = _load(os.path.join(config.DATA_DIR, name))
        if not d:
            continue
        comp = d.get("components_60plus") or {}

        def ratio(key, actual_key="actual"):
            c = comp.get(key) or {}
            a, p = c.get(actual_key), c.get("predicted")
            return round(p / a, 3) if a and p is not None else None

        rows.append({
            "gw": int(d.get("gw") or 0),
            "spearman": d.get("spearman"),
            "top20_hits": d.get("top20_hits"),
            "captain_pick": d.get("captain_pick"),
            "captain_actual": d.get("captain_actual"),
            "captain_best": d.get("captain_best"),
            "predicted_total": d.get("predicted_total"),
            "actual_total": d.get("actual_total"),
            # >1 means the model expected more than the pitch delivered
            "goals_ratio": ratio("goals"),
            "assists_ratio": ratio("assists"),
            "cs_ratio": ratio("clean_sheets_gk_def"),
            "n_60plus": (comp.get("n") if isinstance(comp.get("n"), int) else None),
        })
    rows.sort(key=lambda r: r["gw"])
    return rows


def feed_scorecard(season: str) -> dict | None:
    """The predicted-lineup feed against the model, per gameweek and pooled.

    E8b prices the feed on its accuracy in the ambiguous band only, so that
    is what is charted; `all` is carried because a reader will ask.
    """
    d = _load(os.path.join(config.DATA_DIR, f"lineup_feed_{season}.json"))
    if not d:
        return None
    per = []
    for gw, rep in sorted((d.get("gws") or {}).items(), key=lambda kv: int(kv[0])):
        b = rep.get("band_metrics") or {}
        per.append({"gw": int(gw), "n": b.get("n"),
                    "feed": b.get("feed_accuracy"),
                    "model": b.get("model_accuracy"),
                    "implied_points": b.get("implied_points_per_season")})
    pooled = d.get("pooled") or {}
    return {"per_gw": per, "band": pooled.get("band_metrics"),
            "all": pooled.get("all_metrics"),
            "updated": d.get("updated_utc"),
            "priceable": pooled.get("priceable"),
            "rows_needed": pooled.get("rows_needed_for_estimate")}


def minutes_model() -> dict:
    """The serving minutes model, and what its features are actually worth.

    Importances are read from the booster on disk rather than stored at train
    time, so this reports the model that is *serving*, not the one someone
    meant to ship. Gain, not weight: a feature used in many cheap splits is
    not thereby important.
    """
    meta = _load(os.path.join(config.MODELS_DIR, "xpts", "minutes_meta.json")) or {}
    out = {
        "features": len(meta.get("features") or []),
        "feature_names": meta.get("features") or [],
        "train_seasons": meta.get("train_seasons"),
        "holdout_season": meta.get("valid_season"),
        "holdout_accuracy": meta.get("holdout_accuracy"),
        "trained_at": meta.get("trained_at"),
        "xgboost": meta.get("xgboost"),
        "importances": [],
        "blocks": {},
    }
    names = out["feature_names"]
    # which family each feature belongs to, so the chart can say where the
    # model's information comes from rather than listing 48 column names
    fams = {"bbc_": "BBC role", "role_": "Understat line", "sel_": "Selection",
            "abs_": "Absence", "own_": "Crowd", "tm_": "Transfermarkt"}
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(os.path.join(config.MODELS_DIR, "xpts", "minutes_xgb.json"))
        score = booster.get_score(importance_type="total_gain")
        total = sum(score.values()) or 1.0
        rows = []
        for k, v in score.items():
            # the booster may name features f0..fn when it was fitted on a
            # bare matrix; map those back through the meta's feature list
            idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
            name = names[idx] if (idx is not None and idx < len(names)) else k
            rows.append({"feature": name, "gain": round(v / total, 4)})
        rows.sort(key=lambda r: -r["gain"])
        out["importances"] = rows[:25]
        byfam: dict[str, float] = {}
        for r in rows:
            fam = next((v for k, v in fams.items() if r["feature"].startswith(k)),
                       "History & fixture")
            byfam[fam] = byfam.get(fam, 0.0) + r["gain"]
        out["blocks"] = {k: round(v, 4) for k, v in
                         sorted(byfam.items(), key=lambda kv: -kv[1])}
    except Exception as exc:  # noqa: BLE001 - a chart must not break the page
        out["importance_error"] = str(exc)
    return out


def backtests() -> list[dict]:
    """The committed replay of each past season — the bar any change has to
    clear. Season means, not gameweeks: the per-gameweek series in these files
    exist for paired tests, not for a trend line."""
    out = []
    try:
        names = sorted(n for n in os.listdir(config.DATA_DIR)
                       if n.startswith("backtest_") and n.endswith(".json"))
    except OSError:
        return out
    keep = ("spearman", "spearman_played", "p_at_20", "top11", "top30",
            "captain", "captain_best", "rmse")
    for name in names:
        d = _load(os.path.join(config.DATA_DIR, name))
        if not isinstance(d, dict):
            continue
        summary = d.get("summary")
        if not isinstance(summary, dict):
            continue
        # one row per MODEL in the replay: the engine next to the baselines it
        # has to beat, which is the only way a season mean means anything
        arms = []
        for arm, vals in summary.items():
            if not isinstance(vals, dict):
                continue
            row = {"arm": arm}
            row.update({k: round(float(vals[k]), 4) for k in keep
                        if isinstance(vals.get(k), (int, float))})
            arms.append(row)
        out.append({"season": d.get("season") or name[9:-5],
                    "gws": len(d.get("gws") or []),
                    "minutes_holdout_accuracy": d.get("minutes_holdout_accuracy"),
                    "arms": arms})
    return out


def season_record(season: str) -> dict:
    """The per-gameweek record, oldest first, and what it adds up to."""
    from . import modelrecord
    doc = modelrecord.load(season)
    gws = [doc["gws"][k] for k in sorted(doc.get("gws", {}), key=int)]
    if not gws:
        return {"gws": [], "season": None}

    def tot(path):
        vals = []
        for r in gws:
            v = r
            for k in path:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return round(sum(vals), 1) if vals else None

    beat = sum(1 for r in gws if r.get("model_squad") and r["fpl"].get("average") is not None
               and r["model_squad"]["actual"] > r["fpl"]["average"])
    calib: dict = {}
    for r in gws:
        for b in r.get("calibration") or []:
            c = calib.setdefault((b["lo"], b["hi"]), {"lo": b["lo"], "hi": b["hi"], "n": 0, "pred": 0.0, "act": 0.0})
            c["n"] += b["n"]
            c["pred"] += b["pred"]
            c["act"] += b["act"]
    by_pos: dict = {}
    for r in gws:
        for pos, v in ((r.get("accuracy") or {}).get("by_pos") or {}).items():
            d = by_pos.setdefault(pos, {"n": 0, "mae": 0.0, "bias": 0.0})
            d["n"] += v["n"]
            d["mae"] += v["mae"] * v["n"]
            d["bias"] += v["bias"] * v["n"]
    for d in by_pos.values():
        d["mae"] = round(d["mae"] / d["n"], 2) if d["n"] else None
        d["bias"] = round(d["bias"] / d["n"], 2) if d["n"] else None
    comps: dict = {}
    for r in gws:
        for k, v in (r.get("components") or {}).items():
            if v.get("pred") is None:
                continue
            c = comps.setdefault(k, {"pred": 0.0, "act": 0.0, "gws": 0})
            c["pred"] += v["pred"]
            c["act"] += v["act"]
            c["gws"] += 1
    squad, hind = tot(["model_squad", "actual"]), tot(["hindsight_squad", "actual"])
    return {
        "gws": gws,
        "season": {
            "n": len(gws),
            "live_weeks": sum(1 for r in gws if r.get("source") == "live"),
            "model_squad": squad, "model_xi": tot(["model_xi", "actual"]),
            "average": tot(["fpl", "average"]), "highest": tot(["fpl", "highest"]),
            "hindsight_squad": hind, "hindsight_xi": tot(["hindsight_xi", "actual"]),
            "weeks_beat_average": beat,
            "captured": round(squad / hind, 3) if squad and hind else None,
            "captain": {"model": tot(["captain", "model", "pts"]),
                        "crowd": tot(["captain", "crowd", "pts"]),
                        "best": tot(["captain", "best", "pts"])},
            "calibration": [{**c, "pred": round(c["pred"], 1)} for c in calib.values()],
            "by_pos": by_pos,
            "components": {k: {**v, "pred": round(v["pred"], 1)} for k, v in comps.items()},
        },
    }


def _team(season: str) -> dict:
    from . import modelteam
    try:
        return modelteam.summary(season)
    except Exception as exc:  # noqa: BLE001 - one broken file must not blank the tab
        return {"weeks": [], "error": str(exc)}


def payload(force: bool = False) -> dict:
    now = time.time()
    if not force and _cache["v"] is not None and now - _cache["t"] < TTL:
        return _cache["v"]
    season = config.CURRENT_SEASON
    out = {
        "season": season,
        "postmortems": postmortems(season),
        "feed": feed_scorecard(season),
        "minutes": minutes_model(),
        "blend": _load(os.path.join(config.MODELS_DIR, "xpts", "blend.json")),
        "market_stretch": _load(os.path.join(config.MODELS_DIR, "xpts",
                                             "market_stretch.json")),
        "backtests": backtests(),
        "record": season_record(season),
        "team": _team(season),
        "built_at": now,
    }
    _cache["t"], _cache["v"] = now, out
    return out
