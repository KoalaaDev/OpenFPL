"""E18 Stage 1: is a learned P(clean sheet) better than the Poisson zero?

Held-out, forward in time: for each replay season the model is trained on
the seasons before it and scored on every team-match of that season against
the engine's own exp(-lambda_blend). Log-loss and Brier over ~1,500
team-matches per season is a far better-powered test than 74 gameweeks of
defender picks, and it is the information question asked directly.

    python research/cs_engine.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fpl_engine import config, db                             # noqa: E402
from fpl_engine.xpts import cs_model                           # noqa: E402

LAM_ONLY = ["log_lam_market", "log_lam_model", "market_missing", "home"]
RECORD = ["own_xga", "own_ga", "own_cs", "own_ga_xga", "opp_xg", "opp_gf", "opp_blank", "opp_gf_xg"]


def logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def main():
    pd.set_option("display.width", 200)
    with db.session(config.DB_PATH) as conn:
        frames = {s: cs_model.season_rows(conn, [s]) for s in config.BACKFILL_SEASONS}
    for season in ("2024-25", "2025-26"):
        train = [s for s in config.BACKFILL_SEASONS if s < season]
        tr = pd.concat([frames[s] for s in train], ignore_index=True)
        te = frames[season].dropna(subset=["cs"]).copy()
        y = te["cs"].to_numpy(float)
        arms = {
            "poisson zero, engine blend (shipped)": np.exp(-te["lam_blend"].to_numpy()),
            "poisson zero, market only": np.exp(-np.exp(te["log_lam_market"].to_numpy())),
            "poisson zero, team model only": np.exp(-np.exp(te["log_lam_model"].to_numpy())),
            "logit: lambdas + venue": cs_model.CSModel("logit", LAM_ONLY).fit(tr).predict(te),
            "logit: full features": cs_model.CSModel("logit", cs_model.FEATURES).fit(tr).predict(te),
            "gbm: full features": cs_model.CSModel("gbm", cs_model.FEATURES).fit(tr).predict(te),
            "offset logit, no intercept: record only": cs_model.CSModel(
                "offset", RECORD).fit(tr).predict(te),
            "offset logit, no intercept: record + lambdas": cs_model.CSModel(
                "offset", RECORD + LAM_ONLY).fit(tr).predict(te),
        }
        print(f"\n== {season}: trained on {train}, {len(tr)} rows; scored on {len(te)} team-matches, "
              f"base rate {y.mean():.3f} ==")
        print(f"{'model':<40}{'log-loss':>10}{'brier':>9}{'mean p':>9}")
        for k, p in arms.items():
            print(f"{k:<40}{logloss(y, p):>10.4f}{brier(y, p):>9.4f}{p.mean():>9.3f}")
        # calibration of the shipped zero vs the full logit, by decile of the shipped p
        te["p_ship"] = arms["poisson zero, engine blend (shipped)"]
        te["p_logit"] = arms["logit: full features"]
        te["dec"] = pd.qcut(te["p_ship"], 10, labels=False, duplicates="drop")
        print(te.groupby("dec").agg(n=("cs", "size"), shipped=("p_ship", "mean"),
                                    logit=("p_logit", "mean"), realised=("cs", "mean")).round(3).to_string())
        m = cs_model.CSModel("logit", cs_model.FEATURES).fit(tr)
        coef = pd.Series(m.est.coef_[0], index=cs_model.FEATURES)
        print("logit coefficients (standardised features):")
        print(coef.round(3).to_string())


if __name__ == "__main__":
    main()
