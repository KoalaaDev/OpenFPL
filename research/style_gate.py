"""Round 22b gate — does the OPPONENT's style reach goalkeepers and forwards
beyond the strength the engine already scales by?

Per club-fixture, the opponent's decayed PRIOR (strictly before kickoff)
match stats from `acq_bbc_match_stats`: possession, shots, shots on target,
box touches, crosses, xG for, xG against (the xG split exists from Dec
2024; the rest from 2022-23). The strength control is the opponent's prior
xG for (keepers) / xG against (forwards) — what the team model's lambda
carries. Within player-season:

  GK:  saves/90        ~ opp_xg_prior + opp_shots_prior (+ shots on target)
  FWD: xG/90 (FPL Opta) ~ opp_xga_prior + opp_poss_prior + opp_shots_allowed
  MID (attacking): the same as FWD

If the style term is significant with the strength term present, the engine
is missing it (it scales saves by lambda_against^0.6 and attack by
lambda_for/league only).

    python research/style_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402
from fpl_engine.xpts import bbc_context as bc  # noqa: E402

HALF_LIFE = 240.0
K0 = 5.0


def prior_stats(conn) -> pd.DataFrame:
    """Per (season, fixture_id, team_id): the club's decayed prior stats and
    what it CONCEDED (the other side's stats), strictly before kickoff."""
    st = pd.read_sql_query(
        "SELECT event_urn, side, possession, shots, shots_on, touches_box, crosses, xg FROM acq_bbc_match_stats", conn)
    fm = bc.fixture_map(conn)
    d = st.merge(fm, on="event_urn")
    d["team_id"] = np.where(d["side"] == "home", d["home_id"], d["away_id"])
    # the other side's row = what this club conceded
    other = d[["event_urn", "side", "shots", "shots_on", "touches_box", "crosses", "xg", "possession"]].copy()
    other["side"] = np.where(other["side"] == "home", "away", "home")
    other = other.rename(columns={"shots": "shots_allowed", "shots_on": "sot_allowed", "touches_box": "box_allowed",
                                  "crosses": "crosses_allowed", "xg": "xga", "possession": "poss_against"})
    d = d.merge(other, on=["event_urn", "side"])
    d = d.sort_values("kick").reset_index(drop=True)
    cols = ["possession", "shots", "shots_on", "touches_box", "crosses", "xg",
            "shots_allowed", "sot_allowed", "box_allowed", "crosses_allowed", "xga"]
    lam = np.log(2) / HALF_LIFE
    days = ((d["kick"] - pd.Timestamp("2000-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy(float)
    out = {c: np.full(len(d), np.nan) for c in cols}
    for _, idx in d.groupby("team_id").indices.items():
        idx = np.asarray(idx)
        for c in cols:
            vals = d[c].to_numpy(float)[idx]
            for j in range(1, len(idx)):
                w = np.exp(-lam * (days[idx[j]] - days[idx[:j]]))
                ok = ~np.isnan(vals[:j])
                if ok.sum() == 0:
                    continue
                lg = np.nanmean(vals[:j])          # crude prior: the club's own mean
                out[c][idx[j]] = ((w[ok] * vals[:j][ok]).sum() + K0 * lg) / (w[ok].sum() + K0)
    for c in cols:
        d[f"{c}_prior"] = out[c]
    return d[["season", "fixture_id", "team_id"] + [f"{c}_prior" for c in cols]]


def within(df, cols, by):
    g = df.groupby(by)
    for c in cols:
        df[c + "_w"] = df[c] - g[c].transform("mean")
    return df


def ols(df, y, xs):
    """OLS with standard errors clustered by fixture (no statsmodels here)."""
    from scipy import stats
    d = df.dropna(subset=[y] + xs)
    X = np.column_stack([np.ones(len(d))] + [d[x].to_numpy(float) for x in xs])
    yy = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ beta
    bread = np.linalg.inv(X.T @ X)
    groups = d["fixture_key"].to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for _, idx in pd.Series(range(len(d))).groupby(groups).indices.items():
        s_g = X[idx].T @ e[idx]
        meat += np.outer(s_g, s_g)
    G = len(np.unique(groups)); n, k = X.shape
    V = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    print(f"  {y} ~ {' + '.join(xs)}   n={len(d)}  clusters={G}")
    for i, x in enumerate(xs, start=1):
        t = beta[i] / se[i]
        print(f"      {x:<26} coef {beta[i]:+.4f}   t {t:+.2f}   p {2 * stats.t.sf(abs(t), G - 1):.4f}")


def main():
    conn = db.connect(config.DB_PATH)
    ps = prior_stats(conn)
    opp = ps.rename(columns={"team_id": "opponent_id"})
    pg = pd.read_sql_query(
        "SELECT pg.season, pg.fixture_id, pg.player_id, pg.opponent_id, pg.minutes, pg.saves, pg.xg, pg.goals_scored, "
        "pg.total_points, p.position FROM player_gw pg JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        "WHERE pg.minutes >= 60 AND p.position IN ('GK','FWD','MID')", conn)
    d = pg.merge(opp, on=["season", "fixture_id", "opponent_id"], how="inner")
    d["fixture_key"] = d["season"] + ":" + d["fixture_id"].astype(str)
    d["saves90"] = d["saves"] / d["minutes"] * 90
    d["xg90"] = pd.to_numeric(d["xg"], errors="coerce") / d["minutes"] * 90
    d["pts90"] = d["total_points"] / d["minutes"] * 90
    print(f"rows {len(d)} | seasons {sorted(d.season.unique())}")

    gk = d[d.position == "GK"].copy()
    gk = within(gk, ["saves90", "xg_prior", "shots_prior", "shots_on_prior", "possession_prior"], ["season", "player_id"])
    print("\n=== GOALKEEPERS: saves per 90 vs the opponent's prior style, within keeper-season")
    ols(gk, "saves90_w", ["xg_prior_w"])
    ols(gk, "saves90_w", ["xg_prior_w", "shots_prior_w"])
    ols(gk, "saves90_w", ["xg_prior_w", "shots_on_prior_w"])
    ols(gk, "saves90_w", ["xg_prior_w", "possession_prior_w"])
    gk["sb"] = pd.qcut(gk["shots_prior"], 4, labels=["few shots", "q2", "q3", "many shots"])
    print(gk.groupby("sb", observed=True).agg(n=("saves90", "size"), saves90=("saves90", "mean"), within=("saves90_w", "mean"), opp_xg=("xg_prior", "mean")).round(3).to_string())

    for pos in ("FWD", "MID"):
        f = d[d.position == pos].copy()
        f = within(f, ["xg90", "pts90", "xga_prior", "possession_prior", "shots_allowed_prior", "box_allowed_prior", "crosses_allowed_prior"], ["season", "player_id"])
        print(f"\n=== {pos}: xG per 90 vs the opponent's prior style, within player-season (strength control = opponent's prior xGA)")
        ols(f, "xg90_w", ["xga_prior_w"])
        ols(f, "xg90_w", ["xga_prior_w", "possession_prior_w"])
        ols(f, "xg90_w", ["xga_prior_w", "shots_allowed_prior_w"])
        ols(f, "xg90_w", ["xga_prior_w", "box_allowed_prior_w"])
        ols(f, "pts90_w", ["xga_prior_w", "possession_prior_w", "box_allowed_prior_w"])
        f["pb"] = pd.qcut(f["possession_prior"], 4, labels=["low poss opp", "q2", "q3", "high poss opp"])
        print(f.groupby("pb", observed=True).agg(n=("xg90", "size"), xg90=("xg90", "mean"), within=("xg90_w", "mean"), opp_xga=("xga_prior", "mean")).round(3).to_string())
    conn.close()


if __name__ == "__main__":
    main()
