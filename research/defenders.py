"""E17: "the model struggles to pick defenders" -- where, by how much, and
ten pre-registered hypotheses replayed as paired arms.

    python research/defenders.py diagnose --frame <frame.csv>
    python research/defenders.py arms --out <dir> [--arms h1 h2 ...]

The frame is one row per player-gameweek with the engine's per-component
projection (``c_*`` columns from ``xpts_predict_gw``) and the realised event
counts, for both replay seasons. ``diagnose`` reports rank quality and points
per pick by position against naive baselines, then decomposes the defender
error by component so the hypotheses are aimed at the part that is wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpl_engine import backtest, config, db, scoring          # noqa: E402
from fpl_engine.xpts.engine import _realised                  # noqa: E402

SEASONS = ["2024-25", "2025-26"]
COMPONENTS = ["goals", "assists", "cs", "conceded", "saves", "bonus", "cards",
              "defcon", "appearance", "residual"]


# ------------------------------------------------------------------ diagnose

def realised_components(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Realised points per component, from the same scoring rules the engine
    uses (``engine._realised``), so modelled and realised are like for like."""
    out = {k: [] for k in COMPONENTS}
    for r in df.to_dict("records"):
        for k in COMPONENTS:
            if k == "residual":
                continue
            v = _realised(k, r, r["position"], rules)
            out[k].append(0.0 if v is None else float(v))
    res = pd.DataFrame(out, index=df.index)
    res["residual"] = df["pts"].to_numpy() - res.drop(columns=["residual"]).sum(axis=1)
    return res.add_prefix("r_")


def by_position(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = []
    for (s, g, pos), d in df.groupby(["season", "gw", "position"]):
        dp = d[d["mins"] > 0]
        rows.append({
            "season": s, "gw": g, "position": pos,
            "sp_played": spearmanr(dp["prediction"], dp["pts"]).statistic if len(dp) > 10 else np.nan,
            f"top{k}": d.nlargest(k, "prediction")["pts"].mean(),
            "top5": d.nlargest(5, "prediction")["pts"].mean(),
            f"top{k}_pred": d.nlargest(k, "prediction")["prediction"].mean(),
            f"best{k}": d.nlargest(k, "pts")["pts"].mean(),
        })
    return pd.DataFrame(rows)


def diagnose(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    rules = scoring.load_rules()
    rc = realised_components(df, rules)
    df = pd.concat([df, rc], axis=1)

    print("\n== rank quality and points per pick by position (engine) ==")
    bp = by_position(df)
    print(bp.groupby("position")[["sp_played", "top5", "top10", "top10_pred", "best10"]]
          .mean().round(3))
    print("(compare: DEF ppg baseline sp_played 0.21 / top10 3.0; trail4 0.20 / 3.3)")

    d = df[df["position"] == "DEF"].copy()
    print("\n== DEF: modelled vs realised points by component, all rows / played 60+ / top-10 picks ==")
    top = d.sort_values("prediction", ascending=False).groupby(["season", "gw"]).head(10)
    on = d[d["max_mins"] >= 60]
    tab = []
    for k in COMPONENTS:
        tab.append({"component": k,
                    "all_model": d[f"c_{k}"].mean(), "all_real": d[f"r_{k}"].mean(),
                    "on_model": on[f"c_{k}"].mean(), "on_real": on[f"r_{k}"].mean(),
                    "top10_model": top[f"c_{k}"].mean(), "top10_real": top[f"r_{k}"].mean()})
    tab.append({"component": "TOTAL",
                "all_model": d["prediction"].mean(), "all_real": d["pts"].mean(),
                "on_model": on["prediction"].mean(), "on_real": on["pts"].mean(),
                "top10_model": top["prediction"].mean(), "top10_real": top["pts"].mean()})
    t = pd.DataFrame(tab).set_index("component")
    t["top10_gap"] = t["top10_real"] - t["top10_model"]
    print(t.round(3))

    print("\n== DEF top-10 picks: which component's error explains the miss? ==")
    # variance of (realised - modelled) per component among top-10 picks
    err = {k: (top[f"r_{k}"] - top[f"c_{k}"]) for k in COMPONENTS}
    E = pd.DataFrame(err)
    tot = top["pts"] - top["prediction"]
    print(pd.DataFrame({"sd_err": E.std(), "cov_with_total_err": E.apply(lambda c: np.cov(c, tot)[0, 1]),
                        "share_of_total_var": E.apply(lambda c: np.cov(c, tot)[0, 1]) / tot.var()}).round(3))

    print("\n== DEF: P(60+) calibration ==")
    d["p60_bin"] = pd.cut(d["p_60"], [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)
    print(d.groupby("p60_bin", observed=True).agg(n=("pts", "size"), p_60=("p_60", "mean"),
                                                   played60=("max_mins", lambda s: (s >= 60).mean())).round(3))

    print("\n== DEF who played 60+: clean-sheet calibration by venue and by price ==")
    on = on.assign(cs_real=(on["clean_sheets"] > 0).astype(float),
                   p_cs_cond=on["p_cs"] / on["p_60"].clip(lower=1e-6))
    on = on[on["n_fixtures"] == 1]
    print(on.groupby("was_home").agg(n=("pts", "size"), p_cs=("p_cs_cond", "mean"), cs=("cs_real", "mean")).round(3))
    on["price_bin"] = pd.qcut(on["price"], 4, labels=["cheap", "q2", "q3", "premium"], duplicates="drop")
    print(on.groupby("price_bin", observed=True).agg(n=("pts", "size"), pred=("prediction", "mean"), real=("pts", "mean"),
                                                      p_cs=("p_cs_cond", "mean"), cs=("cs_real", "mean"),
                                                      bonus_m=("c_bonus", "mean"), bonus_r=("r_bonus", "mean"),
                                                      dc_m=("c_defcon", "mean"), dc_r=("r_defcon", "mean")).round(3))

    print("\n== DEF who played 60+: points given a clean sheet vs not (modelled vs realised) ==")
    print(on.groupby("cs_real").agg(n=("pts", "size"), pred=("prediction", "mean"), real=("pts", "mean"),
                                     bonus_m=("c_bonus", "mean"), bonus_r=("r_bonus", "mean")).round(3))

    print("\n== DEF: modelled bonus vs realised, by realised events ==")
    on["ev"] = np.where(on["goals_scored"] + on["assists"] > 0, "attacking return",
                        np.where(on["cs_real"] > 0, "cs only", "nothing"))
    print(on.groupby("ev").agg(n=("pts", "size"), bonus_m=("c_bonus", "mean"), bonus_r=("r_bonus", "mean"),
                               pred=("prediction", "mean"), real=("pts", "mean")).round(3))

    print("\n== DEF top-10 picks per gw by season: predicted vs realised, and the realised best-10 ==")
    print(bp[bp["position"] == "DEF"].groupby("season")[["sp_played", "top10", "top10_pred", "best10"]].mean().round(3))

    print("\n== DEF: per-club concentration of top-10 picks (share of picks from the most-picked club per gw) ==")
    conc = top.groupby(["season", "gw"])["team_id"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    print(f"mean share from the single most-picked club: {conc.mean():.3f}")


# ---------------------------------------------------------------------- arms

def _arm_kwargs(name: str, cfg: dict) -> dict:
    return cfg


def run_arms(conn, out: str, seasons: list[str], arms: dict, base_dir: str) -> dict:
    results = {}
    for name, cfg in arms.items():
        arm_dir = os.path.join(out, name)
        for s in seasons:
            if not os.path.exists(os.path.join(arm_dir, f"backtest_{s}.json")):
                print(f"[{name}] {s}", flush=True)
                backtest.run(conn, s, with_openfpl=False, out_dir=arm_dir, xpts_kwargs=cfg)
        results[name] = backtest.compare(base_dir, arm_dir)
    return results


def print_arms(results: dict) -> None:
    keys = ["spearman_played", "top11", "top30", "captain", "rmse",
            "def_top5", "def_top10", "def_spearman_played"]
    print(f"\n{'arm':<12}" + "".join(f"{k:>22}" for k in keys))
    for name, r in results.items():
        cells = []
        for k in keys:
            m = r["metrics"].get(k)
            cells.append(f"{m['delta']:+.4f} (p={m['p']:.3f})" if m else "-")
        print(f"{name:<12}" + "".join(f"{c:>22}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "arms"])
    ap.add_argument("--frame")
    ap.add_argument("--out", default=os.path.join(config.DATA_DIR, "e17"))
    ap.add_argument("--base", help="directory holding the baseline backtest_<season>.json")
    ap.add_argument("--seasons", nargs="*", default=SEASONS)
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    if a.cmd == "diagnose":
        diagnose(pd.read_csv(a.frame))
        return
    from defender_arms import ARMS       # noqa: E402  (pre-registered list)
    arms = {k: ARMS[k] for k in (a.arms or list(ARMS))}
    with db.session(config.DB_PATH) as conn:
        res = run_arms(conn, a.out, a.seasons, arms, a.base)
    with open(os.path.join(a.out, "arms.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print_arms(res)


if __name__ == "__main__":
    main()
