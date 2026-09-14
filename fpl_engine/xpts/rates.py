"""Per-90 event rates with empirical-Bayes shrinkage.

For each player (identified by stable ``player_code`` so history crosses
seasons) we estimate time-decayed per-90 rates for the events the scoring
rules pay for: xG, xA, saves, yellow cards — plus two structured extras:

* **Bonus** is not a flat rate: it is driven by the same events the fixture
  scaler multiplies (goals, assists, clean sheets). We fit a league-wide
  per-position weighted least squares ``bonus ~ goals + assists + cs`` and
  keep only each player's *deviation* from that fit as a flat per-90 rate
  (``bonus_resid90``); the engine reconstructs E[bonus] from its own expected
  events, so a striker in a great fixture is credited the bonus that comes
  with the goals he is expected to score there. The coefficients ride along
  in ``DataFrame.attrs["bonus_coef"]`` (read them before merging — pandas
  drops attrs on merge).
* **Residual rate** = actual points minus reconstructed points, which absorbs
  DefCon (no raw tackles/CBI stats exist in the DB). DefCon has only existed
  since the season named in the scoring rules
  (``defensive_contribution.since``); matches before that era are excluded
  from the residual estimate — otherwise a DefCon regular's rate is diluted
  toward zero by seasons where the rule did not exist. Era data is scarce,
  so the residual uses a smaller shrinkage constant than the base stats.

Units (Round 17): xG is converted into realised-goal units with a per-position
conversion ratio and xA into FPL-assist units with a per-position ratio, both
measured from the same point-in-time history; ``$FPL_XPTS_VARIANT=legacy_rates``
restores the pre-Round-17 estimator for a paired comparison.

Shrinkage: rate = (Σ w·stat + k·prior_pos) / (Σ w·mins/90 + k) — a player with
little recent playing time regresses to his position's league rate instead of
producing wild small-sample estimates. Priors are computed from the data at
fit time, never hardcoded.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from .. import scoring

HALF_LIFE_DAYS = 240.0
SEASON_BREAK_DECAY = 0.7    # extra weight factor per season boundary — an
                            # outlier season (Salah 2024-25) must not be
                            # carried whole across the summer
K_EFFECTIVE_90S = 6.0
K_RESIDUAL_90S = 3.0    # DefCon era is short — trust the player's own rate sooner
BASE_STATS = ["xg", "xa", "saves", "yellow_cards"]
STATS = BASE_STATS + ["bonus_resid", "defcon_cross", "residual"]


def _parse(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, format="ISO8601")


def fit(conn, season: str, as_of: str, *, rules: dict | None = None,
        bonus_defcon: bool = False, xa_blend: dict | None = None,
        k_by_pos: dict | None = None, calibrate_by_pos: bool = False) -> pd.DataFrame:
    """Return one row per current-season player with shrunk per-90 rates.

    Uses all player_gw history strictly before ``as_of`` across seasons.
    League-level bonus coefficients ride along in ``.attrs["bonus_coef"]``.

    The keyword options are research variants (RESEARCH_LOG E17), all off on
    every shipped path: ``bonus_defcon`` adds DefCon threshold crossings to
    the bonus regression (fitted on rule-era rows only, coefficient list
    becomes [g, a, cs, dc, c0]); ``xa_blend`` maps position -> weight on Opta
    xA against realised FPL assists (default 0.5 everywhere); ``k_by_pos``
    maps position -> shrinkage pseudo-90s for the base stats;
    ``calibrate_by_pos`` multiplies every player's xG90 / xA90 by his
    position's decay-weighted realised-goals / xG (assists / blended xA)
    ratio over the same history, i.e. a point-in-time finishing calibration.
    """
    rules = rules or scoring.load_rules()
    hist = pd.read_sql_query(
        "SELECT pg.season, pg.player_code, pg.kickoff_utc, pg.minutes, pg.total_points, "
        "pg.goals_scored, pg.assists, pg.clean_sheets, pg.goals_conceded, "
        "pg.own_goals, pg.penalties_saved, pg.penalties_missed, "
        "pg.yellow_cards, pg.red_cards, pg.saves, pg.bonus, pg.xg, pg.xa, pg.defcon, "
        "(SELECT position FROM player p WHERE p.season=pg.season "
        " AND p.player_id=pg.player_id) position, "
        "(SELECT tm.goals_for - tm.goals_against FROM team_match tm "
        " WHERE tm.season=pg.season AND tm.team_id=pg.team_id "
        " AND tm.fixture_id=pg.fixture_id) margin "
        "FROM player_gw pg WHERE pg.kickoff_utc < ? AND pg.minutes > 0",
        conn, params=(as_of,))
    import os as _os
    use_gd = "bonus_gd" in {v.strip() for v in
                            _os.environ.get("FPL_XPTS_VARIANT", "").split(",")}
    players = pd.read_sql_query(
        "SELECT player_id, code player_code, position FROM player "
        "WHERE season=?", conn, params=(season,))
    if hist.empty:
        out = players.copy()
        for s in STATS:
            out[f"{s}90"] = 0.0
        out["exposure"] = 0.0
        out = out[["player_id", "player_code", "position", "exposure"]
                  + [f"{s}90" for s in STATS]]
        out.attrs["bonus_coef"] = {}
        return out

    # decay weights first — every estimate below is decay-weighted
    ref = pd.Timestamp(datetime.fromisoformat(as_of.replace("Z", "+00:00")))
    days = (ref - _parse(hist["kickoff_utc"])).dt.days.clip(lower=0)
    hist["w"] = 0.5 ** (days / HALF_LIFE_DAYS)
    if SEASON_BREAK_DECAY < 1.0:   # an outlier season shouldn't be carried whole
        cur = int(season[:4])
        n_breaks = (cur - hist["season"].str[:4].astype(int)).clip(lower=0)
        hist["w"] *= SEASON_BREAK_DECAY ** n_breaks
    hist["w90"] = hist["w"] * hist["minutes"].fillna(0) / 90.0

    # DefCon: raw counts exist from the rule era on (vaastav/FPL both publish
    # them); the modelled event is *crossing the position threshold*
    thr = (rules.get("defensive_contribution") or {}).get("threshold", {})
    thr_of = hist["position"].map(thr).astype(float)   # NaN -> never crosses
    hist["defcon_cross"] = np.where(
        hist["defcon"].notna(),
        (hist["defcon"] >= thr_of.fillna(np.inf)).astype(float), np.nan)
    hist["has_dc"] = hist["defcon"].notna().astype(float)

    # residual = actual points - full reconstruction (including actual DefCon
    # where counts exist) -> genuinely unmodelled scraps only
    def _base_points(r):
        t = thr.get(r.position or "MID")
        crossed = (1 if (r.defcon is not None and t is not None
                         and r.defcon >= t) else 0)
        return scoring.points_from_events({
            "minutes": r.minutes, "goals_scored": r.goals_scored,
            "assists": r.assists, "clean_sheets": r.clean_sheets,
            "goals_conceded": r.goals_conceded, "own_goals": r.own_goals,
            "penalties_saved": r.penalties_saved,
            "penalties_missed": r.penalties_missed,
            "yellow_cards": r.yellow_cards, "red_cards": r.red_cards,
            "saves": r.saves, "bonus": r.bonus,
            "defensive_contribution": crossed,
        }, r.position or "MID", rules=rules)

    hist["residual"] = (hist["total_points"].fillna(0)
                        - np.array([_base_points(r) for r in hist.itertuples()]))
    era = (rules.get("defensive_contribution") or {}).get("since")
    hist["in_era"] = 1.0 if era is None else (hist["season"] >= era).astype(float)

    # xg missing (old seasons without Understat/Opta) -> fall back to goals
    hist["xg"] = hist["xg"].fillna(hist["goals_scored"]).fillna(0)
    # FPL assists are much broader than Opta xA (rebounds, won penalties,
    # deflected passes all count), so pure xA systematically lowballs the
    # assist rate (GW1 2026-27: league xA 15.4 vs 24 FPL assists). Blend the
    # stable estimator with the realised FPL-definition rate 50/50.
    _var = {v.strip() for v in _os.environ.get("FPL_XPTS_VARIANT", "").split(",")}
    legacy = "legacy_rates" in _var       # the pre-Round-17 estimator, for A/Bs
    if not legacy:
        # Round 17 (shipped): goals per xG differ by position and the gap is
        # persistent (defenders convert 0.76-0.93 of their xG across seasons,
        # midfielders and forwards ~1.0), so the goal rate is xG in
        # REALISED-goal units, per position, from the same decay-weighted
        # point-in-time history, shrunk toward 1. Paired over 74 gameweeks
        # with the xA fix below: spearman_played +0.0009 (p=0.002), top-30
        # +0.04 pts/pick, rmse unchanged.
        hx = hist[hist["xg"].notna() & hist["position"].notna()]
        num = (hx["w"] * hx["goals_scored"].fillna(0)).groupby(hx["position"]).sum()
        den = (hx["w"] * hx["xg"].fillna(0)).groupby(hx["position"]).sum()
        k0 = 60.0                                 # ~60 weighted xG of prior at 1.0
        conv = ((num + k0) / (den + k0)).to_dict()
        hist["xg"] = np.where(hist["xg"].notna(),
                              hist["xg"].fillna(0) * hist["position"].map(conv).fillna(1.0),
                              hist["xg"])
    if not legacy:
        # Round 17 (shipped): FPL assists per Opta xA by position (DEF ~1.2,
        # MID ~1.35, FWD ~2.1 — stable across four seasons), applied before
        # the 50/50 blend so both halves are in FPL-assist units. The audit
        # had assists under-predicted 9-18% in every replayed season.
        hx = hist[hist["xa"].notna() & hist["position"].notna()]
        num = (hx["w"] * hx["assists"].fillna(0)).groupby(hx["position"]).sum()
        den = (hx["w"] * hx["xa"].fillna(0)).groupby(hx["position"]).sum()
        k0 = 30.0
        ratio = ((num + k0) / (den + k0)).to_dict()
        has = hist["xa"].notna()
        bxa = hist["position"].map(xa_blend or {}).fillna(0.5).to_numpy(float)
        hist["xa"] = np.where(has,
                              bxa * hist["xa"].fillna(0) * hist["position"].map(ratio).fillna(1.0)
                              + (1 - bxa) * hist["assists"].fillna(0),
                              hist["assists"].fillna(0))
    elif "xa_scaled" in _var:          # research arm: one league-wide ratio
        # research arm: put xA in FPL-assist units first. The league-wide,
        # decay-weighted ratio of FPL assists to Opta xA is measured from the
        # same history (point-in-time), so the blend no longer mixes two
        # different units 50/50.
        has = hist["xa"].notna()
        num = (hist.loc[has, "w"] * hist.loc[has, "assists"].fillna(0)).sum()
        den = (hist.loc[has, "w"] * hist.loc[has, "xa"].fillna(0)).sum()
        ratio = float(num / den) if den > 0 else 1.0
        hist["xa"] = np.where(has,
                              0.5 * hist["xa"].fillna(0) * ratio
                              + 0.5 * hist["assists"].fillna(0),
                              hist["assists"].fillna(0))
    else:
        bxa = hist["position"].map(xa_blend or {}).fillna(0.5).to_numpy(float)
        hist["xa"] = np.where(hist["xa"].notna(),
                              bxa * hist["xa"].fillna(0) + (1 - bxa) * hist["assists"].fillna(0),
                              hist["assists"].fillna(0))

    # league bonus structure: per-position weighted least squares on the
    # events the engine models; each player keeps only his deviation
    hb = hist.dropna(subset=["position"])
    if bonus_defcon:      # only rows that carry DefCon counts can inform it
        hb = hb[hb["has_dc"] > 0]
    bonus_coef: dict[str, list[float]] = {}
    for pos, d in hb.groupby("position"):
        cols = [d["goals_scored"].fillna(0).to_numpy(float),
                d["assists"].fillna(0).to_numpy(float),
                d["clean_sheets"].fillna(0).to_numpy(float)]
        if use_gd:   # research arm: the winning side collects more BPS
            cols.append(d["margin"].fillna(0).to_numpy(float))
        if bonus_defcon:   # E17 tweak: DefCon crossings in the bonus fit
            cols.append(d["defcon_cross"].fillna(0).to_numpy(float))
        X = np.column_stack(cols + [np.ones(len(d))])
        y = d["bonus"].fillna(0).to_numpy(float)
        sw = np.sqrt(d["w"].to_numpy(float))
        coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        bonus_coef[pos] = [float(v) for v in coef]
    # the coefficient order, published with the coefficients so the engine
    # applies research terms by NAME rather than by the vector's length
    bonus_terms = (["g", "a", "cs"] + (["gd"] if use_gd else [])
                   + (["dc"] if bonus_defcon else []) + ["c0"])
    for i, name in enumerate(bonus_terms):
        hist[f"_bc_{name}"] = hist["position"].map(
            {p: c[i] for p, c in bonus_coef.items()}).fillna(0.0)
    gd_term = (hist["_bc_gd"] * hist["margin"].fillna(0)) if use_gd else 0.0
    dc_term = (hist["_bc_dc"] * hist["defcon_cross"].fillna(0)) if bonus_defcon else 0.0
    hist["bonus_resid"] = (hist["bonus"].fillna(0)
                           - hist["_bc_g"] * hist["goals_scored"].fillna(0)
                           - hist["_bc_a"] * hist["assists"].fillna(0)
                           - hist["_bc_cs"] * hist["clean_sheets"].fillna(0)
                           - gd_term - dc_term
                           - hist["_bc_c0"])

    gates = {"residual": hist["in_era"], "defcon_cross": hist["has_dc"]}
    for s in STATS:
        hist[f"_w_{s}"] = hist["w"] * gates.get(s, 1.0) * hist[s].fillna(0)
    hist["w90_era"] = hist["w90"] * hist["in_era"]
    hist["w90_dc"] = hist["w90"] * hist["has_dc"]

    agg = hist.groupby("player_code").agg(
        exposure=("w90", "sum"), exposure_era=("w90_era", "sum"),
        exposure_dc=("w90_dc", "sum"),
        **{f"sum_{s}": (f"_w_{s}", "sum") for s in STATS})

    # position priors: league per-90 rate per position (exposure-weighted);
    # the residual prior comes from the DefCon era only
    hist_pos = hist.dropna(subset=["position"])
    era_pos = hist_pos[hist_pos["in_era"] > 0]
    dc_pos = hist_pos[hist_pos["has_dc"] > 0]
    prior = {}
    for s in STATS:
        src = hist_pos
        if s == "residual" and len(era_pos):
            src = era_pos
        elif s == "defcon_cross":
            src = dc_pos if len(dc_pos) else hist_pos.iloc[0:0]
        if not len(src):
            prior[s] = {}
            continue
        by_pos = src.groupby("position").apply(
            lambda d, s=s: (d["w"] * d[s].fillna(0)).sum()
            / max(1e-9, d["w90"].sum()), include_groups=False)
        prior[s] = by_pos.to_dict()

    out = players.merge(agg, left_on="player_code", right_index=True, how="left")
    out["exposure"] = out["exposure"].fillna(0.0)
    out["exposure_era"] = out["exposure_era"].fillna(0.0)
    out["exposure_dc"] = out["exposure_dc"].fillna(0.0)
    expo_of = {"residual": out["exposure_era"], "defcon_cross": out["exposure_dc"]}
    for s in STATS:
        pri = out["position"].map(prior[s]).fillna(0.0)
        k = (K_RESIDUAL_90S if s in ("residual", "defcon_cross")
             else K_EFFECTIVE_90S)
        if k_by_pos and s in BASE_STATS:
            k = out["position"].map(k_by_pos).fillna(k).to_numpy(float)
        expo = expo_of.get(s, out["exposure"])
        out[f"{s}90"] = ((out[f"sum_{s}"].fillna(0) + k * pri) / (expo + k))
    if calibrate_by_pos:
        hp = hist.dropna(subset=["position"])
        for stat, real in (("xg", "goals_scored"), ("xa", "assists")):
            num = (hp["w"] * hp[real].fillna(0)).groupby(hp["position"]).sum()
            den = (hp["w"] * hp[stat].fillna(0)).groupby(hp["position"]).sum()
            ratio = (num / den.replace(0, np.nan)).clip(0.5, 1.5)
            out[f"{stat}90"] = out[f"{stat}90"] * out["position"].map(ratio).fillna(1.0)
    out = out[["player_id", "player_code", "position", "exposure"]
              + [f"{s}90" for s in STATS]]
    out.attrs["bonus_coef"] = bonus_coef
    out.attrs["bonus_terms"] = bonus_terms
    return out
