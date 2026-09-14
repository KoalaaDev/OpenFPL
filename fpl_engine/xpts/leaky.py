"""Own-team leakiness: does a club's recent scoreline record say anything the
engine's clean-sheet lambda does not?

The engine prices a defender's clean sheet and conceded points from
``lambda_against`` -- the market's implied goals for the opponent (85%)
blended with the team model's rate, whose defence rating is fitted on realised
goals against blended with xGA. A club that keeps conceding therefore already
carries a higher lambda. The hypothesis tested here (RESEARCH_LOG E16) is that
this is NOT enough: that a team which has shown it concedes -- by scoreline --
concedes more than lambda says, so its defenders are over-projected on easy
fixtures however good their own numbers are.

Two things live here, both research affordances that are None on every
shipped path:

* ``team_leak_features`` -- point-in-time trailing scoreline features per club
  (goals against, clean-sheet share, share conceding two or more, goals
  against minus xGA), cross-season by ``team.code``.
* ``defence_leak_factors`` -- a multiplier on ``lambda_against`` for the club's
  DEFENSIVE components only (clean sheet, conceded, saves); its opponent's
  attack is untouched. ``engine.xpts_predict_gw(defence_leak=...)`` applies it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRAIL_MATCHES = 10
SHRINK_MATCHES = 3.0     # matches of league-mean prior behind every trailing rate


def team_leak_features(conn, season: str, as_of: str, *,
                       n: int = TRAIL_MATCHES) -> pd.DataFrame:
    """One row per club of ``season``: trailing scoreline features before as_of.

    ``ga``/``cs``/``two_plus`` are shrunk toward the league mean with
    ``SHRINK_MATCHES`` pseudo-matches, so a promoted club with two matches on
    record is not called leaky on one bad afternoon. ``n_matches`` is the raw
    count. ``ga_xga`` is only over matches that carry xGA (NaN otherwise).
    """
    tm = pd.read_sql_query(
        "SELECT t.code, tm.kickoff_utc, tm.goals_against, tm.xga "
        "FROM team_match tm JOIN team t ON t.season=tm.season AND t.team_id=tm.team_id "
        "WHERE tm.kickoff_utc < ? AND tm.goals_against IS NOT NULL",
        conn, params=(as_of,))
    clubs = pd.read_sql_query(
        "SELECT team_id, code FROM team WHERE season=?", conn, params=(season,))
    cols = ["ga", "cs", "two_plus", "ga_xga", "n_matches"]
    if tm.empty or clubs.empty:
        out = clubs.assign(**{c: np.nan for c in cols})
        out["n_matches"] = 0
        return out.set_index("team_id")
    tm = tm.sort_values("kickoff_utc")
    last = tm.groupby("code").tail(n)
    last = last.assign(cs=(last["goals_against"] == 0).astype(float),
                       two_plus=(last["goals_against"] >= 2).astype(float),
                       ga_xga=last["goals_against"] - last["xga"])
    # league priors: from the same trailing window, pooled over every club
    league = {"ga": float(last["goals_against"].mean()),
              "cs": float(last["cs"].mean()),
              "two_plus": float(last["two_plus"].mean())}
    g = last.groupby("code")
    agg = pd.DataFrame({
        "sum_ga": g["goals_against"].sum(), "sum_cs": g["cs"].sum(),
        "sum_two": g["two_plus"].sum(), "n_matches": g.size(),
        "ga_xga": g["ga_xga"].mean(),
    })
    k = SHRINK_MATCHES
    agg["ga"] = (agg["sum_ga"] + k * league["ga"]) / (agg["n_matches"] + k)
    agg["cs"] = (agg["sum_cs"] + k * league["cs"]) / (agg["n_matches"] + k)
    agg["two_plus"] = (agg["sum_two"] + k * league["two_plus"]) / (agg["n_matches"] + k)
    out = clubs.merge(agg[cols], left_on="code", right_index=True, how="left")
    out["n_matches"] = out["n_matches"].fillna(0).astype(int)
    for c in ("ga", "cs", "two_plus"):      # a club with no record sits at the league mean
        out[c] = out[c].fillna(league[c])
    out.attrs["league"] = league
    return out.set_index("team_id")


def defence_leak_factors(conn, season: str, as_of: str,
                         cfg: dict) -> dict[int, float]:
    """Multiplier on lambda_against per club, from its trailing scorelines.

    ``cfg["mode"]``:
      ``"ga"``    (ga / league_ga) ** alpha -- goals conceded relative to the league
      ``"tail"``  1 + alpha * (two_plus - league_two_plus) -- the heavy-defeat share
      ``"cs"``    ((1 - cs) / (1 - league_cs)) ** alpha -- how rarely it keeps them out
    ``cfg["alpha"]`` is the strength (0 = shipped engine); ``cfg["n"]`` the window.
    """
    mode = cfg.get("mode", "ga")
    alpha = float(cfg.get("alpha", 1.0))
    if alpha == 0.0:
        return {}
    feats = team_leak_features(conn, season, as_of, n=int(cfg.get("n", TRAIL_MATCHES)))
    league = feats.attrs.get("league") or {}
    if feats.empty or not league:
        return {}
    out: dict[int, float] = {}
    for team_id, r in feats.iterrows():
        if mode == "ga":
            f = (max(1e-6, r["ga"]) / max(1e-6, league["ga"])) ** alpha
        elif mode == "tail":
            f = 1.0 + alpha * (r["two_plus"] - league["two_plus"])
        elif mode == "cs":
            f = (max(1e-6, 1.0 - r["cs"]) / max(1e-6, 1.0 - league["cs"])) ** alpha
        else:
            raise ValueError(f"unknown defence_leak mode {mode!r}")
        out[int(team_id)] = float(np.clip(f, 0.4, 2.5))
    return out
