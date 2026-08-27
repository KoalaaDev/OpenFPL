"""Price-change model: who rises and who falls before the next gameweek.

FPL prices move on transfer momentum. Buying a player the week he rises and
selling before he falls is worth real squad value over a season, and both
signals — the transfer flow and the ownership it is measured against — are
free and already in ``player_gw`` (``transfers_in``/``transfers_out``/
``selected``, plus the per-gameweek ``price``).

Target: ``price[gw+1] - price[gw]`` in tenths of £m, as three classes
(fall / hold / rise). Only 7% of player-gameweeks move, so accuracy is a
useless metric — what matters is whether the *ranking* concentrates the
movers. Held out forward in time (train on earlier seasons, score a later
one), the top-10 ranked risers rise **67-75%** of the time against a 2% base
rate, and the top-30-minus-bottom-30 spread is **0.87-0.91 tenths per pick**
against 0.40 for a naive net-transfers rule.

Timing. A row for gameweek ``t`` carries the transfer flow of ``t``'s window,
the ownership and price at ``t``, and what the player scored in ``t``. Live,
the last completed gameweek supplies all of that, so predicting the move into
``t+1`` uses nothing from the future. To prove that the gameweek boundary is
not being crossed, ``FEATURES_DEADLINE`` drops everything about ``t``'s
matches; it still ranks the top-10 risers at 57-64%.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from . import config

# What £1m of team value is worth, in points, per remaining gameweek.
#
# Not a guess. Measured by rebuilding the best legal 15 (2/5/5/3, max 3 per
# club, legal XI, captain) under a budget and then again with more, across 18
# replayed gameweeks:
#
#     extra budget    +0.5m   +1.0m   +2.0m   +4.0m
#     points per £1m  0.163   0.117   0.107   0.093
#
# Diminishing, as it must be. Price movements are small (±0.1-0.3m), so the
# marginal 0.163 is the rate that applies to them.
#
# The realised check on the same squads is -1.2 +/- 1.0 points and cannot
# settle anything: with a per-gameweek sd of 4.1, confirming an 0.08-point
# effect would need ~19,000 gameweeks. The modelled figure is the usable one,
# and it is an upper bound in the sense that it assumes the extra budget is
# spent as well as the optimiser would spend it.
POINTS_PER_MILLION_PER_GW = 0.163
SELL_ON_SHARE = 0.5      # FPL returns the purchase price plus half the profit

MODEL_DIR = os.path.join(config.MODELS_DIR, "price")
MODEL_PATH = os.path.join(MODEL_DIR, "price_xgb.json")
META_PATH = os.path.join(MODEL_DIR, "price_meta.json")

FLOW_FEATURES = ["net_frac", "tin_frac", "tout_frac", "log_selected"]
PRICE_FEATURES = ["price", "dp_lag1", "dp_lag2", "season_price_change"]
CONTEXT_FEATURES = ["is_gk", "is_def", "is_mid", "is_fwd", "gw"]
FORM_FEATURES = ["pts_l1", "pts_l3", "mins_l1"]

FEATURES = FLOW_FEATURES + PRICE_FEATURES + CONTEXT_FEATURES + FORM_FEATURES
# the leak-proof variant: nothing about the most recent gameweek's matches
FEATURES_DEADLINE = FLOW_FEATURES + PRICE_FEATURES + CONTEXT_FEATURES

CLASSES = {0: "fall", 1: "hold", 2: "rise"}


def _frame(conn, seasons: list[str]) -> pd.DataFrame:
    """One row per player-gameweek with the move into the NEXT gameweek."""
    df = pd.read_sql_query(
        "SELECT pg.season, pg.gw, pg.player_id, MAX(pg.price) price, "
        "MAX(pg.selected) selected, SUM(pg.transfers_in) tin, "
        "SUM(pg.transfers_out) tout, SUM(pg.total_points) pts, "
        "SUM(pg.minutes) mins, "
        "(SELECT position FROM player p WHERE p.season=pg.season "
        " AND p.player_id=pg.player_id) position "
        "FROM player_gw pg JOIN player p ON p.season=pg.season "
        "AND p.player_id=pg.player_id "
        "WHERE p.position IN ('GK','DEF','MID','FWD') AND pg.season IN (%s) "
        "GROUP BY pg.season, pg.gw, pg.player_id" % ",".join("?" * len(seasons)),
        conn, params=list(seasons))
    if df.empty:
        return df
    df = df.sort_values(["season", "player_id", "gw"]).reset_index(drop=True)
    g = df.groupby(["season", "player_id"], sort=False)
    df["dp"] = g["price"].shift(-1) - df["price"]
    df["dp_lag1"] = df["price"] - g["price"].shift(1)
    df["dp_lag2"] = g["dp_lag1"].shift(1)
    df["season_price_change"] = df["price"] - g["price"].transform("first")
    sel = df["selected"].clip(lower=1)
    df["net_frac"] = (df["tin"] - df["tout"]) / sel
    df["tin_frac"] = df["tin"] / sel
    df["tout_frac"] = df["tout"] / sel
    df["log_selected"] = np.log1p(df["selected"].fillna(0))
    df["pts_l1"] = df["pts"]
    df["pts_l3"] = g["pts"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["mins_l1"] = df["mins"]
    for pos, col in (("GK", "is_gk"), ("DEF", "is_def"),
                     ("MID", "is_mid"), ("FWD", "is_fwd")):
        df[col] = (df["position"] == pos).astype(float)
    df["label"] = np.select([df["dp"] > 0, df["dp"] < 0], [2, 0], 1)
    return df


def train(conn, *, seasons: list[str] | None = None,
          features: list[str] | None = None, device: str | None = None) -> dict:
    """Train and cache the classifier; returns metadata with holdout numbers."""
    import xgboost as xgb
    seasons = seasons or config.BACKFILL_SEASONS
    features = features or FEATURES
    frame = _frame(conn, seasons).dropna(subset=["dp"])
    if frame.empty:
        raise ValueError("no price history — run `python -m fpl_engine pull`")

    def _fit(d):
        clf = xgb.XGBClassifier(
            objective="multi:softprob", num_class=3, n_estimators=300,
            max_depth=5, learning_rate=0.07, subsample=0.9,
            colsample_bytree=0.8, eval_metric="mlogloss", device=device or "cpu")
        clf.fit(d[features], d["label"].astype(int))
        return clf

    holdout = None
    valid_season = seasons[-1] if len(seasons) > 1 else None
    if valid_season:
        va = frame[frame["season"] == valid_season]
        probe = _fit(frame[frame["season"] < valid_season])
        if len(va):
            p = probe.predict_proba(va[features])
            score = p[:, 2] - p[:, 0]
            va = va.assign(_score=score)
            top = va.groupby("gw", group_keys=False).apply(
                lambda d: (d.nlargest(10, "_score")["dp"] > 0).mean(),
                include_groups=False).mean()
            holdout = {"season": valid_season,
                       "p_rise_given_top10": float(top),
                       "base_rate": float((va["dp"] > 0).mean())}
    clf = _fit(frame)      # refit on every season, as the minutes model does
    os.makedirs(MODEL_DIR, exist_ok=True)
    clf.save_model(MODEL_PATH)
    meta = {"features": features, "train_seasons": seasons, "holdout": holdout}
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load():
    """Return (clf, meta), or (None, None) when absent or built on old features."""
    import xgboost as xgb
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        return None, None
    with open(META_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("features") != FEATURES:
        return None, None
    clf = xgb.XGBClassifier()
    clf.load_model(MODEL_PATH)
    return clf, meta


def ensure(conn, *, seasons: list[str] | None = None, device: str | None = None):
    clf, meta = load()
    if clf is None:
        train(conn, seasons=seasons, device=device)
        clf, meta = load()
    return clf, meta


def predict(conn, season: str, *, gw: int | None = None, bundle=None) -> pd.DataFrame:
    """P(fall/hold/rise) and E[price change] for the move out of ``gw``.

    ``gw`` defaults to the last gameweek with price data — the state a manager
    is actually looking at. ``e_delta`` is in £m (FPL stores tenths).
    """
    clf, meta = bundle or ensure(conn)
    seasons = list(dict.fromkeys(config.BACKFILL_SEASONS + [season]))
    frame = _frame(conn, seasons)
    cur = frame[frame["season"] == season]
    if cur.empty:
        return pd.DataFrame()
    gw = gw if gw is not None else int(cur.loc[cur["price"].notna(), "gw"].max())
    d = cur[cur["gw"] == gw].copy()
    if d.empty:
        return pd.DataFrame()
    p = clf.predict_proba(d[meta["features"]].astype(float))
    d["p_fall"], d["p_hold"], d["p_rise"] = p[:, 0], p[:, 1], p[:, 2]
    d["score"] = d["p_rise"] - d["p_fall"]
    # tenths -> £m; a move is always exactly one tenth in practice
    d["e_delta"] = (d["p_rise"] - d["p_fall"]) * 0.1
    d["price_m"] = d["price"] / 10.0
    return d[["player_id", "gw", "price_m", "p_fall", "p_hold", "p_rise",
              "score", "e_delta"]].sort_values("score", ascending=False)


def points_value(e_delta_m: float, gws_remaining: int) -> float:
    """Convert an expected price move (£m) into points.

    A rise is only realised when the player is sold, and FPL hands back the
    purchase price plus **half** the profit — so a 0.1 rise is 0.05 of usable
    budget, which then buys a better squad for however much season is left.

    The numbers this produces are small on purpose: catching a 0.1 rise with
    30 gameweeks to go is worth about 0.2 points. That is a tie-breaker
    between transfers you already rate equally, not a term that should be
    allowed to overturn an expected-points ranking.
    """
    if not gws_remaining or gws_remaining < 1:
        return 0.0
    return (float(e_delta_m) * SELL_ON_SHARE
            * POINTS_PER_MILLION_PER_GW * int(gws_remaining))
