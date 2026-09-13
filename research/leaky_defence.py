"""E16: are defenders from leaky teams over-projected on easy fixtures?

Two halves, both forward-in-time:

  diagnose   score the SHIPPED engine's clean-sheet probability and defender
             points against what happened, split by the club's own trailing
             scoreline record (goals against, heavy-defeat share, GA - xGA)
             and by fixture ease. A logistic regression with the engine's own
             logit(P(CS)) as an offset asks whether leakiness carries anything
             lambda does not.
  arms       replay both seasons with ``defence_leak`` arms through the
             standard backtest and paired-compare them against the shipped
             engine on the same gameweeks.

    python research/leaky_defence.py diagnose --out /tmp/e16
    python research/leaky_defence.py arms --out /tmp/e16

Needs the backtest-tagged minutes models (``bt<season>``), which the baseline
backtest trains; ``arms`` runs that baseline first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpl_engine import backtest, config, db, scoring          # noqa: E402
from fpl_engine.xpts import engine, leaky, minutes_model       # noqa: E402

SEASONS = ["2024-25", "2025-26"]
ARMS = {
    "goals_def": {"mode": "goals_def"},
    "ga_a05": {"mode": "ga", "alpha": 0.5},
    "ga_a10": {"mode": "ga", "alpha": 1.0},
    "tail_a10": {"mode": "tail", "alpha": 1.0},
}


# ------------------------------------------------------------------ diagnose

def _actual_rows(conn, season: str) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT pg.gw, pg.player_id, p.position, COUNT(*) n_rows, "
        "SUM(pg.minutes) mins, MAX(pg.minutes) max_mins, "
        "SUM(pg.total_points) pts, SUM(pg.clean_sheets) cs, "
        "SUM(pg.goals_conceded) ga "
        "FROM player_gw pg JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        "WHERE pg.season=? AND p.position IN ('GK','DEF','MID','FWD') "
        "GROUP BY pg.gw, pg.player_id", conn, params=(season,))


def build_frame(conn, seasons: list[str]) -> pd.DataFrame:
    rules = scoring.load_rules()
    out = []
    for season in seasons:
        clf, meta = minutes_model.load(f"bt{season}")
        if clf is None:
            raise SystemExit(f"no bt{season} minutes model: run the baseline backtest first")
        act = _actual_rows(conn, season)
        for g in sorted(act["gw"].unique()):
            if g < 2:
                continue
            as_of = engine.first_kickoff(conn, season, int(g))
            pred = engine.xpts_predict_gw(conn, season, int(g), as_of=as_of,
                                          use_availability=False,
                                          minutes_bundle=(clf, meta), rules=rules)
            if pred.empty:
                continue
            feats = leaky.team_leak_features(conn, season, as_of)
            lg = feats.attrs.get("league", {})
            pred = pred.merge(feats[["ga", "cs", "two_plus", "ga_xga", "n_matches"]]
                              .rename(columns={"ga": "t_ga", "cs": "t_cs",
                                               "two_plus": "t_two", "ga_xga": "t_gaxga"}),
                              left_on="team_id", right_index=True, how="left")
            pred["league_ga"] = lg.get("ga")
            pred["league_two"] = lg.get("two_plus")
            j = act[act["gw"] == g].merge(pred, on="player_id", how="inner",
                                          suffixes=("_act", ""))
            j["season"] = season
            out.append(j)
            print(f"  {season} GW{g}: {len(j)} rows", flush=True)
    return pd.concat(out, ignore_index=True)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logistic_offset(y, X, offset, names, n_iter=50):
    """Logistic regression with a fixed offset (IRLS); returns coef, se, z."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        eta = offset + X @ b
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None])
        grad = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), grad)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    eta = offset + X @ b
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return pd.DataFrame({"coef": b, "se": se, "z": b / se}, index=names)


def _z(s):
    return (s - s.mean()) / s.std()


def diagnose(df: pd.DataFrame) -> dict:
    res = {}
    d = df[df["position"].isin(["GK", "DEF"])].copy()
    single = d[(d["n_fixtures"] == 1) & (d["n_rows"] == 1)].copy()
    single["p_cs_cond"] = np.exp(-single["lam_against"])
    on = single[single["max_mins"] >= 60].copy()       # CS is at stake
    on["cs"] = (on["cs"] > 0).astype(float)
    on["leaky"] = on["t_ga"] > on["league_ga"]
    on["easy"] = on["lam_against"] < on.groupby(["season", "gw"])["lam_against"].transform("median")

    # D1: calibration of P(CS | 60+) in deciles, then inside leakiness terciles
    on["dec"] = pd.qcut(on["p_cs_cond"], 10, labels=False, duplicates="drop")
    d1 = on.groupby("dec").agg(n=("cs", "size"), predicted=("p_cs_cond", "mean"),
                               realised=("cs", "mean"))
    d1["gap"] = d1["realised"] - d1["predicted"]
    res["d1_deciles"] = d1.round(4)
    on["leak_tercile"] = pd.qcut(on["t_ga"], 3, labels=["solid", "mid", "leaky"])
    on["pcs_bin"] = pd.qcut(on["p_cs_cond"], 4, labels=["low", "q2", "q3", "high"])
    d1b = on.groupby(["pcs_bin", "leak_tercile"], observed=True).agg(
        n=("cs", "size"), predicted=("p_cs_cond", "mean"), realised=("cs", "mean"))
    d1b["gap"] = d1b["realised"] - d1b["predicted"]
    res["d1_by_leak"] = d1b.round(4)
    d1c = on.groupby("leak_tercile", observed=True).agg(
        n=("cs", "size"), predicted=("p_cs_cond", "mean"), realised=("cs", "mean"),
        t_ga=("t_ga", "mean"), lam=("lam_against", "mean"))
    d1c["gap"] = d1c["realised"] - d1c["predicted"]
    res["d1_leak_marginal"] = d1c.round(4)

    # D2: logistic regression with the engine's own logit as an offset
    reg = on.dropna(subset=["t_ga", "t_two", "t_cs"]).copy()
    reg["z_ga"] = _z(reg["t_ga"])
    reg["z_two"] = _z(reg["t_two"])
    reg["z_cs"] = _z(reg["t_cs"])
    reg["z_gaxga"] = _z(reg["t_gaxga"].fillna(0))
    reg["z_lam"] = _z(reg["lam_against"])
    reg["easy_f"] = reg["easy"].astype(float)
    off = _logit(reg["p_cs_cond"])
    rows = {}
    for name, cols in {
        "ga": ["z_ga"], "two_plus": ["z_two"], "cs_rate": ["z_cs"],
        "ga_minus_xga": ["z_gaxga"],
        "ga x easy": ["z_ga", "easy_f", "z_ga:easy"],
        "all": ["z_ga", "z_two", "z_gaxga"],
    }.items():
        X = []
        for c in cols:
            if c == "z_ga:easy":
                X.append(reg["z_ga"] * reg["easy_f"])
            else:
                X.append(reg[c])
        X = np.column_stack([np.ones(len(reg))] + X)
        fit = logistic_offset(reg["cs"], X, off, ["intercept"] + cols)
        rows[name] = fit
    res["d2_logit"] = rows
    res["d2_n"] = len(reg)
    # the offset's own slope: 1.0 = the engine's ordering is right
    X = np.column_stack([np.ones(len(reg)), off])
    res["d2_slope"] = logistic_offset(reg["cs"], X, np.zeros(len(reg)),
                                      ["intercept", "logit_pcs"])

    # D3: the decision view -- the 10 highest-predicted DEF/GK each gameweek
    top = (d.sort_values("prediction", ascending=False)
            .groupby(["season", "gw"]).head(10).copy())
    top["leaky"] = top["t_ga"] > top["league_ga"]
    top["easy"] = top["lam_against"] < top.groupby(["season", "gw"])["lam_against"].transform("median")
    d3 = top.groupby(["leaky", "easy"]).agg(
        n=("pts", "size"), predicted=("prediction", "mean"), realised=("pts", "mean"),
        p_cs=("p_cs", "mean"), cs_rate=("cs", lambda s: float((s > 0).mean())))
    d3["gap"] = d3["realised"] - d3["predicted"]
    res["d3_top10_def"] = d3.round(3)
    d3b = top.groupby("leaky").agg(
        n=("pts", "size"), predicted=("prediction", "mean"), realised=("pts", "mean"))
    d3b["gap"] = d3b["realised"] - d3b["predicted"]
    res["d3_top10_by_leaky"] = d3b.round(3)
    # per-gameweek paired: realised - predicted for leaky vs solid picks
    pg = top.groupby(["season", "gw", "leaky"]).apply(
        lambda s: (s["pts"] - s["prediction"]).mean(), include_groups=False).unstack()
    if True in pg.columns and False in pg.columns:
        diff = (pg[True] - pg[False]).dropna()
        from scipy.stats import ttest_1samp
        t = ttest_1samp(diff, 0)
        res["d3_paired"] = {"gws": int(len(diff)), "mean_diff": float(diff.mean()),
                            "t": float(t.statistic), "p": float(t.pvalue)}
    # share of the top-10 that comes from leaky clubs
    res["d3_share_leaky"] = float(top["leaky"].mean())
    return res


def print_diagnosis(res: dict) -> None:
    pd.set_option("display.width", 160)
    print("\n== D1. P(CS | 60+) calibration by decile ==")
    print(res["d1_deciles"])
    print("\n== D1. by own-team leakiness tercile (marginal) ==")
    print(res["d1_leak_marginal"])
    print("\n== D1. inside P(CS) quartile, by leakiness tercile ==")
    print(res["d1_by_leak"])
    print(f"\n== D2. logistic regression, offset = engine logit P(CS), n={res['d2_n']} ==")
    print("slope on the engine's own logit (1.0 = ordering right):")
    print(res["d2_slope"].round(4))
    for name, fit in res["d2_logit"].items():
        print(f"-- {name}")
        print(fit.round(4))
    print("\n== D3. top-10 predicted DEF/GK per gameweek: predicted vs realised ==")
    print(res["d3_top10_by_leaky"])
    print(res["d3_top10_def"])
    print("paired per-gw (leaky picks' error minus solid picks' error):", res.get("d3_paired"))
    print(f"share of top-10 DEF picks from leaky clubs: {res['d3_share_leaky']:.3f}")


# ---------------------------------------------------------------------- arms

def run_arms(conn, out: str, seasons: list[str], arms: dict) -> dict:
    base_dir = os.path.join(out, "base")
    for s in seasons:
        if not os.path.exists(os.path.join(base_dir, f"backtest_{s}.json")):
            print(f"[base] {s}", flush=True)
            backtest.run(conn, s, with_openfpl=False, out_dir=base_dir)
    results = {}
    for name, cfg in arms.items():
        arm_dir = os.path.join(out, name)
        for s in seasons:
            if not os.path.exists(os.path.join(arm_dir, f"backtest_{s}.json")):
                print(f"[{name}] {s}", flush=True)
                backtest.run(conn, s, with_openfpl=False, out_dir=arm_dir,
                             xpts_kwargs={"defence_leak": cfg})
        results[name] = backtest.compare(base_dir, arm_dir)
    return results


def print_arms(results: dict) -> None:
    keys = ["spearman_played", "p_at_20", "top11", "top30", "captain", "rmse",
            "def_top5", "def_top10", "def_spearman_played"]
    print(f"\n{'arm':<10}" + "".join(f"{k:>22}" for k in keys))
    for name, r in results.items():
        cells = []
        for k in keys:
            m = r["metrics"].get(k)
            cells.append(f"{m['delta']:+.4f} (p={m['p']:.3f})" if m else "-")
        print(f"{name:<10}" + "".join(f"{c:>22}" for c in cells))
    print(f"gameweeks paired: {next(iter(results.values()))['gws']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diagnose", "arms"])
    ap.add_argument("--out", default=os.path.join(config.DATA_DIR, "e16"))
    ap.add_argument("--seasons", nargs="*", default=SEASONS)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with db.session(config.DB_PATH) as conn:
        if a.cmd == "diagnose":
            path = os.path.join(a.out, "frame.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = build_frame(conn, a.seasons)
                df.to_csv(path, index=False)
            res = diagnose(df)
            print_diagnosis(res)
        else:
            res = run_arms(conn, a.out, a.seasons, {k: ARMS[k] for k in a.arms})
            with open(os.path.join(a.out, "arms.json"), "w") as fh:
                json.dump(res, fh, indent=2)
            print_arms(res)


if __name__ == "__main__":
    main()
