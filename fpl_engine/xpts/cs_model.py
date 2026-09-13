"""A learned clean-sheet model (RESEARCH_LOG E18).

The shipped engine prices a clean sheet as P(60+) x exp(-lambda_against),
where lambda_against is the bookmaker's implied goals for the opponent
blended with the team model's rate. That is a Poisson zero with a single
parameter. This module asks whether a supervised model of "did the club keep
a clean sheet" -- fitted forward in time on team-matches, from features the
engine already has at the deadline -- is a better P(no goals) than the
Poisson zero:

  * the two lambdas themselves (market and model, on the log scale, and the
    engine's own blend), so the model can re-weight them
  * venue
  * the club's trailing defensive record: xGA, goals against, clean-sheet
    share, goals against minus xGA (keeper / luck)
  * the opponent's trailing attacking record: xG, goals for, blank share
    (matches without a goal), goals for minus xG

Everything is point-in-time: features for a gameweek use matches strictly
before that gameweek's first kickoff, cross-season by ``team.code``. The
model is trained only on seasons BEFORE the one being predicted (the
backtest passes the same list it trains the minutes model on) and cached per
training set for the life of the process. It is a research affordance:
``xpts_predict_gw(tweaks={"cs_model": {...}})`` uses it; nothing shipped
does.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import leaky, odds_model, team_model

FEATURES = ["log_lam_blend", "log_lam_model", "log_lam_market", "market_missing",
            "home", "own_xga", "own_ga", "own_cs", "own_ga_xga",
            "opp_xg", "opp_gf", "opp_blank", "opp_gf_xg"]
TRAIL = 10
SHRINK = 3.0
_CACHE: dict = {}


def _first_kickoff(conn, season: str, gw: int):
    r = conn.execute("SELECT MIN(kickoff_utc) k FROM team_match WHERE season=? AND gw=?",
                     (season, gw)).fetchone()
    return r["k"] if r and r["k"] else None


def _trailing(conn, season: str, as_of: str) -> pd.DataFrame:
    """Per club of ``season``: shrunk trailing attack and defence record."""
    tm = pd.read_sql_query(
        "SELECT t.code, tm.kickoff_utc, tm.goals_for, tm.goals_against, tm.xg, tm.xga "
        "FROM team_match tm JOIN team t ON t.season=tm.season AND t.team_id=tm.team_id "
        "WHERE tm.kickoff_utc < ? AND tm.goals_against IS NOT NULL", conn, params=(as_of,))
    clubs = pd.read_sql_query("SELECT team_id, code FROM team WHERE season=?", conn, params=(season,))
    cols = {"ga": "goals_against", "gf": "goals_for", "xga": "xga", "xg": "xg"}
    if tm.empty:
        out = clubs.copy()
        for c in ("ga", "gf", "xga", "xg", "cs", "blank", "ga_xga", "gf_xg"):
            out[c] = np.nan
        return out.set_index("team_id")
    last = tm.sort_values("kickoff_utc").groupby("code").tail(TRAIL).copy()
    last["cs"] = (last["goals_against"] == 0).astype(float)
    last["blank"] = (last["goals_for"] == 0).astype(float)
    last["xga"] = last["xga"].fillna(last["goals_against"])
    last["xg"] = last["xg"].fillna(last["goals_for"])
    last["ga_xga"] = last["goals_against"] - last["xga"]
    last["gf_xg"] = last["goals_for"] - last["xg"]
    league = {c: float(last[src].mean()) for c, src in
              {**cols, "cs": "cs", "blank": "blank", "ga_xga": "ga_xga", "gf_xg": "gf_xg"}.items()}
    g = last.groupby("code")
    n = g.size()
    agg = pd.DataFrame({"n": n})
    for c, src in {**cols, "cs": "cs", "blank": "blank", "ga_xga": "ga_xga", "gf_xg": "gf_xg"}.items():
        agg[c] = (g[src].sum() + SHRINK * league[c]) / (n + SHRINK)
    out = clubs.merge(agg, left_on="code", right_index=True, how="left")
    for c in league:
        out[c] = out[c].fillna(league[c])
    return out.set_index("team_id")


def gw_rows(conn, season: str, gw: int, *, as_of: str | None = None,
            with_target: bool = True) -> pd.DataFrame:
    """One row per (club, fixture) of the gameweek with the model's features.

    ``with_target`` attaches ``cs`` (kept a clean sheet) from ``team_match``
    for training; prediction rows leave it NaN.
    """
    as_of = as_of or _first_kickoff(conn, season, gw)
    if as_of is None:
        return pd.DataFrame(columns=FEATURES + ["season", "gw", "team_id", "fixture_id", "cs"])
    fx = conn.execute(
        "SELECT tm.fixture_id, tm.team_id team_h, tm.opponent_id team_a, "
        "th.code hcode, ta.code acode, tm.goals_against ga_home, tm.goals_for ga_away "
        "FROM team_match tm JOIN team th ON th.season=tm.season AND th.team_id=tm.team_id "
        "JOIN team ta ON ta.season=tm.season AND ta.team_id=tm.opponent_id "
        "WHERE tm.season=? AND tm.gw=? AND tm.was_home=1", (season, gw)).fetchall()
    if not fx:
        return pd.DataFrame(columns=FEATURES + ["season", "gw", "team_id", "fixture_id", "cs"])
    tm = team_model.fit(conn, as_of)
    omap = odds_model.fixture_odds_map(conn, season, [f["fixture_id"] for f in fx])
    tr = _trailing(conn, season, as_of)
    ow = odds_model.ODDS_WEIGHT
    rows = []
    for f in fx:
        lh, la = tm.fixture(f["hcode"], f["acode"])
        od = omap.get(f["fixture_id"])
        for team, opp, home, lam_model, lam_market, ga in (
                (f["team_h"], f["team_a"], 1.0, la, od[1] if od else None, f["ga_home"]),
                (f["team_a"], f["team_h"], 0.0, lh, od[0] if od else None, f["ga_away"])):
            mk = lam_market if lam_market is not None else lam_model
            blend = (1 - ow) * lam_model + ow * mk if lam_market is not None else lam_model
            t, o = tr.loc[team], tr.loc[opp]
            rows.append({
                "season": season, "gw": gw, "team_id": int(team), "fixture_id": int(f["fixture_id"]),
                "log_lam_blend": math.log(max(1e-3, blend)),
                "log_lam_model": math.log(max(1e-3, lam_model)),
                "log_lam_market": math.log(max(1e-3, mk)),
                "market_missing": 0.0 if lam_market is not None else 1.0,
                "home": home,
                "own_xga": t["xga"], "own_ga": t["ga"], "own_cs": t["cs"], "own_ga_xga": t["ga_xga"],
                "opp_xg": o["xg"], "opp_gf": o["gf"], "opp_blank": o["blank"], "opp_gf_xg": o["gf_xg"],
                "lam_blend": blend,
                "cs": (float(ga == 0) if (with_target and ga is not None) else np.nan),
            })
    return pd.DataFrame(rows)


def season_rows(conn, seasons: list[str], *, min_gw: int = 2) -> pd.DataFrame:
    out = []
    for s in seasons:
        gws = [r["gw"] for r in conn.execute(
            "SELECT DISTINCT gw FROM team_match WHERE season=? AND gw>=? ORDER BY gw",
            (s, min_gw))]
        for g in gws:
            out.append(gw_rows(conn, s, int(g)))
    return pd.concat([o for o in out if len(o)], ignore_index=True) if out else pd.DataFrame()


class CSModel:
    """A fitted P(clean sheet | features)."""

    def __init__(self, kind: str, features: list[str]):
        self.kind, self.features = kind, features
        self.est = None
        self.mean = None
        self.std = None

    def fit(self, df: pd.DataFrame) -> "CSModel":
        d = df.dropna(subset=["cs"])
        X = d[self.features].to_numpy(float)
        y = d["cs"].to_numpy(float)
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-9
        Xs = (X - self.mean) / self.std
        if self.kind == "logit":
            from sklearn.linear_model import LogisticRegression
            self.est = LogisticRegression(C=1.0, max_iter=1000).fit(Xs, y)
        elif self.kind == "offset":
            # the shipped Poisson zero as a fixed offset, NO free intercept:
            # the features can only bend the engine's own probability, so a
            # season whose base rate differs from the training seasons' is
            # not inherited through an intercept (the plain logit's failure)
            off = -d["lam_blend"].to_numpy(float)
            self.est = _fit_offset_logit(Xs, y, _logit(np.exp(off)))
        elif self.kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.est = HistGradientBoostingClassifier(
                max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                l2_regularization=1.0, random_state=0).fit(Xs, y)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        Xs = (df[self.features].to_numpy(float) - self.mean) / self.std
        if self.kind == "offset":
            eta = _logit(np.exp(-df["lam_blend"].to_numpy(float))) + Xs @ self.est
            return 1.0 / (1.0 + np.exp(-eta))
        return self.est.predict_proba(Xs)[:, 1]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit_offset_logit(X, y, offset, n_iter=50, ridge=1.0):
    """Ridge-regularised logistic coefficients with a fixed offset (IRLS)."""
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(offset + X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(len(b))
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def ensure(conn, train_seasons: list[str], *, kind: str = "logit",
           features: list[str] | None = None) -> CSModel:
    """Fit (once per process) a model on ``train_seasons``."""
    feats = list(features or FEATURES)
    key = (kind, tuple(train_seasons), tuple(feats))
    if key not in _CACHE:
        df = season_rows(conn, train_seasons)
        _CACHE[key] = CSModel(kind, feats).fit(df)
    return _CACHE[key]


def p_clean_sheet_map(conn, season: str, gw: int, as_of: str, train_seasons: list[str],
                      *, kind: str = "logit",
                      features: list[str] | None = None) -> dict[tuple[int, int], float]:
    """{(team_id, fixture_id): P(no goals conceded)} for the gameweek."""
    model = ensure(conn, train_seasons, kind=kind, features=features)
    rows = gw_rows(conn, season, gw, as_of=as_of, with_target=False)
    if rows.empty:
        return {}
    p = model.predict(rows)
    return {(int(t), int(f)): float(v) for t, f, v in zip(rows["team_id"], rows["fixture_id"], p)}
