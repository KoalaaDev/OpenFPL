"""Forward-in-time backtest: replay past gameweeks and score every model
against what actually happened, with decision-relevant metrics.

Models compared per gameweek (all strictly point-in-time):
  xpts     — the component engine (fpl_engine.xpts)
  openfpl  — the frozen OpenFPL ensemble (on a subsample of gws; its feature
             build is expensive)
  ppg      — points-per-appearance so far (naive baseline)
  trail4   — mean of the last 4 gameweek scores (form baseline)

Metrics per gameweek:
  spearman        rank correlation between prediction and actual points
  spearman_played the same, restricted to players who actually got on the
                  pitch. Plain ``spearman`` is dominated by ranking who plays
                  at all, which every model gets roughly right; this is the
                  part that decides who to captain and start.
  p_at_20         |top-20 predicted ∩ top-20 actual| / 20
  top11 / top30   mean ACTUAL points of the 11 / 30 highest-predicted players
                  — points per pick, the currency the squad is built in
  captain         actual points of the #1 predicted player
  rmse            plain error magnitude (least decision-relevant, still
                  reported, and the lowest-variance of these)

A run also prints a paired comparison against ``ppg``/``trail4`` — the per-
gameweek series are kept in the JSON so an experiment can be re-tested with a
paired t-test rather than compared on season means, which are noisy enough to
make a real 0.1 pts/pick gain and pure chance look identical.

The minutes classifier is (re)trained only on seasons *before* the backtest
season, and cached under a per-season tag so it never becomes the model that
serves live predictions. The OpenFPL/xpts blend weight is fitted on the first
half of the season and evaluated on the second, then saved to
models/xpts/blend.json for live use.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config, db, features, predict as predict_mod, progress, scoring
from .xpts import engine as xpts_engine, minutes_model

BLEND_PATH = os.path.join(config.MODELS_DIR, "xpts", "blend.json")


METRIC_KEYS = ("spearman", "spearman_played", "p_at_20", "top11", "top30",
               "captain", "captain_best", "rmse")


def compare(before: str, after: str, *, model: str = "xpts",
            seasons: list[str] | None = None) -> dict:
    """Paired per-gameweek comparison of two saved backtest reports.

    Season means move around by more than a real improvement does, so a change
    is only believable when the *same* gameweeks are better. Usage: keep a copy
    of ``data/backtest_<season>.json`` before a change, re-run the backtest
    after it, then::

        python -m fpl_engine compare-backtests before.json after.json

    ``before``/``after`` are paths to those JSON files (or directories holding
    ``backtest_<season>.json`` for several seasons, which are pooled).
    """
    from scipy.stats import ttest_rel

    def _load(path):
        if os.path.isdir(path):
            out = {}
            for s in (seasons or []) or sorted(
                    f[len("backtest_"):-len(".json")] for f in os.listdir(path)
                    if f.startswith("backtest_") and f.endswith(".json")):
                with open(os.path.join(path, f"backtest_{s}.json"),
                          encoding="utf-8") as fh:
                    out[s] = json.load(fh)
            return out
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        return {r.get("season", "?"): r}

    a, b = _load(before), _load(after)
    rows_a, rows_b = [], []
    for s in sorted(set(a) & set(b)):
        pa = a[s].get("per_gw", {}).get(model, {})
        pb = b[s].get("per_gw", {}).get(model, {})
        for g in sorted(set(pa) & set(pb), key=int):
            rows_a.append(pa[g])
            rows_b.append(pb[g])
    if not rows_a:
        raise ValueError(f"no overlapping gameweeks for model {model!r}")
    out = {"model": model, "gws": len(rows_a), "metrics": {}}
    for k in METRIC_KEYS:
        # captain_best is the gameweek's own ceiling, identical for every
        # model — comparing it is a no-op that only produces a NaN t-statistic
        if k not in rows_a[0] or k == "captain_best":
            continue
        x = np.array([r[k] for r in rows_a], float)
        y = np.array([r[k] for r in rows_b], float)
        ok = np.isfinite(x) & np.isfinite(y)
        t, p = ttest_rel(y[ok], x[ok])
        out["metrics"][k] = {"before": float(x[ok].mean()),
                             "after": float(y[ok].mean()),
                             "delta": float((y - x)[ok].mean()),
                             "t": float(t), "p": float(p)}
    return out


def _actuals(conn, season: str) -> pd.DataFrame:
    """Realised points per player-gameweek, players only.

    FPL's Assistant Managers are selectable entries that score points and never
    play a minute. Left in, they occupy slots in the "actual top 20" that no
    player model can reach — in 2024-25 that was up to 8 of 20 in a gameweek,
    silently capping precision@20 and depressing top-11/top-30 for every model
    in the comparison.
    """
    return pd.read_sql_query(
        "SELECT pg.gw, pg.player_id, SUM(pg.total_points) pts, "
        "SUM(pg.minutes) mins FROM player_gw pg JOIN player p "
        "ON p.season=pg.season AND p.player_id=pg.player_id "
        "WHERE pg.season=? AND p.position IN ('GK','DEF','MID','FWD') "
        "GROUP BY pg.gw, pg.player_id",
        conn, params=(season,))


def _openfpl_gw(conn, season: str, gw: int, bundle) -> pd.DataFrame | None:
    """OpenFPL predictions with player ids, no availability multiplier."""
    try:
        df = features.build_samples(conn, season, gw, include_ids=True)
    except ValueError:
        return None
    preds = predict_mod.predict(df, bundle=bundle)
    merged = preds.merge(
        df[["player", "team", "position", "player_id"]].drop_duplicates(
            ["player", "team", "position"]),
        on=["player", "team", "position"], how="left")
    out = merged[["player_id", "prediction"]].dropna()
    out["player_id"] = out["player_id"].astype(int)
    return out


def _metrics(pred: pd.DataFrame, actual_gw: pd.DataFrame) -> dict | None:
    j = actual_gw.merge(pred, on="player_id", how="left")
    j["prediction"] = j["prediction"].fillna(0.0)
    if len(j) < 30 or j["prediction"].std() < 1e-9:
        return None
    rho = float(spearmanr(j["prediction"], j["pts"]).statistic)
    top_pred = set(j.nlargest(20, "prediction")["player_id"])
    top_act = set(j.nlargest(20, "pts")["player_id"])
    cap_row = j.loc[j["prediction"].idxmax()]
    played = j[j["mins"] > 0] if "mins" in j else j.iloc[0:0]
    return {
        "spearman": rho,
        "spearman_played": (float(spearmanr(played["prediction"],
                                            played["pts"]).statistic)
                            if len(played) > 30 else float("nan")),
        "p_at_20": len(top_pred & top_act) / 20.0,
        "top11": float(j.nlargest(11, "prediction")["pts"].mean()),
        "top30": float(j.nlargest(30, "prediction")["pts"].mean()),
        "captain": float(cap_row["pts"]),
        "captain_best": float(j["pts"].max()),
        "rmse": float(np.sqrt(((j["prediction"] - j["pts"]) ** 2).mean())),
    }


def run(conn, season: str = "2025-26", *, gws: list[int] | None = None,
        openfpl_every: int = 4, retrain_minutes: bool = False,
        with_openfpl: bool = True, out_dir: str | None = None) -> dict:
    train_seasons = [s for s in config.BACKFILL_SEASONS if s < season]
    # cached under the replayed season: a model that has never seen it must
    # never become the one that serves live predictions. An optional feature
    # block gets its own tag too, so an A/B does not make each arm retrain the
    # other's cache away.
    extra = "".join(sorted(minutes_model.EXTRA_FEATURES))
    tag = f"bt{season}" + (f".x{abs(hash(extra)) % 10**8}" if extra else "")
    if retrain_minutes or minutes_model.load(tag)[0] is None:
        progress.step(f"Training minutes model on {train_seasons}…")
        meta = minutes_model.train(conn, seasons=train_seasons, tag=tag)
        if meta.get("holdout_accuracy") is not None:
            progress.step(f"  holdout accuracy ({meta['valid_season']}): "
                          f"{meta['holdout_accuracy']:.3f}")
    clf, meta = minutes_model.load(tag)

    actual = _actuals(conn, season)
    all_gws = sorted(int(g) for g in actual["gw"].dropna().unique())
    gws = [int(g) for g in gws] if gws else [g for g in all_gws if g >= 2]
    rules = scoring.load_rules()
    bundle = predict_mod.load_models() if with_openfpl else None

    per_model: dict[str, dict[int, dict]] = {}
    preds_store: dict[tuple[str, int], pd.DataFrame] = {}
    cum: dict[int, list] = {}
    trail: dict[int, list] = {}

    for g in gws:
        as_of = xpts_engine.first_kickoff(conn, season, g)
        act_g = actual[actual["gw"] == g][["player_id", "pts", "mins"]]
        progress.step(f"GW{g}…")

        x = xpts_engine.xpts_predict_gw(conn, season, g, as_of=as_of,
                                        use_availability=False,
                                        minutes_bundle=(clf, meta), rules=rules)
        if not x.empty:
            p = x[["player_id", "prediction"]]
            preds_store[("xpts", g)] = p
            m = _metrics(p, act_g)
            if m:
                per_model.setdefault("xpts", {})[g] = m

        # naive baselines from accumulated actuals
        hist = actual[actual["gw"] < g]
        played = hist[hist["mins"] > 0]
        ppg = (played.groupby("player_id")["pts"].mean().rename("prediction")
               .reset_index())
        m = _metrics(ppg, act_g)
        if m:
            per_model.setdefault("ppg", {})[g] = m
        t4 = (hist[hist["gw"] >= g - 4].groupby("player_id")["pts"].mean()
              .rename("prediction").reset_index())
        m = _metrics(t4, act_g)
        if m:
            per_model.setdefault("trail4", {})[g] = m

        if with_openfpl and (g - gws[0]) % openfpl_every == 0:
            o = _openfpl_gw(conn, season, g, bundle)
            if o is not None and not o.empty:
                preds_store[("openfpl", g)] = o
                m = _metrics(o, act_g)
                if m:
                    per_model.setdefault("openfpl", {})[g] = m

    # ---- blend fit: first half picks w, second half judges it ----
    blend_info = None
    ogws = sorted(g for (name, g) in preds_store if name == "openfpl")
    both = [g for g in ogws if ("xpts", g) in preds_store]
    if len(both) >= 4:
        half = both[:len(both) // 2]
        rest = both[len(both) // 2:]

        def blended_rho(w, sel, key="spearman_played"):
            vals = []
            for g in sel:
                o = preds_store[("openfpl", g)].rename(columns={"prediction": "o"})
                x = preds_store[("xpts", g)].rename(columns={"prediction": "x"})
                jj = o.merge(x, on="player_id", how="outer").fillna(0)
                jj["prediction"] = (1 - w) * jj["o"] + w * jj["x"]
                m = _metrics(jj[["player_id", "prediction"]],
                             actual[actual["gw"] == g][["player_id", "pts", "mins"]])
                if m:
                    vals.append(m[key])
            return float(np.nanmean(vals)) if vals else -1.0

        grid = [round(w, 2) for w in np.arange(0, 1.01, 0.1)]
        best_w = max(grid, key=lambda w: blended_rho(w, half))
        blend_info = {
            "weight": best_w, "season": season,
            "fit_gws": half, "eval_gws": rest,
            "eval_spearman_played": {"openfpl": blended_rho(0.0, rest),
                                     "xpts": blended_rho(1.0, rest),
                                     "blend": blended_rho(best_w, rest)},
            "eval_top30": {"openfpl": blended_rho(0.0, rest, "top30"),
                           "xpts": blended_rho(1.0, rest, "top30"),
                           "blend": blended_rho(best_w, rest, "top30")},
        }
        os.makedirs(os.path.dirname(BLEND_PATH), exist_ok=True)
        with open(BLEND_PATH, "w", encoding="utf-8") as fh:
            json.dump(blend_info, fh, indent=2)

    summary = {}
    for name, res in per_model.items():
        arr = list(res.values())
        summary[name] = {k: round(float(np.nanmean([m[k] for m in arr])), 4)
                         for k in arr[0]}
        summary[name]["gws"] = len(arr)
    report = {"season": season, "gws": gws, "summary": summary,
              "blend": blend_info,
              "minutes_holdout_accuracy": meta.get("holdout_accuracy")}
    out_dir = out_dir or config.DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"backtest_{season}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({**report, "per_gw": {m: {str(g): v for g, v in res.items()}
                                        for m, res in per_model.items()}},
                  fh, indent=2, default=float)
    return report
