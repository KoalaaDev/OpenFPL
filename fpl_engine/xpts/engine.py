"""Assemble expected points from the component models via the scoring YAML.

For each player and each of his team's fixtures in the target gameweek:

  exposure   = E[minutes]/90 from the minutes model
  E[goals]   = xG90 · exposure · fixture attack scaler (team model)
  E[assists] = xA90 · exposure · fixture attack scaler
  P(CS)      = P(60+) · exp(-λ_opponent)
  conceded   = E[floor(GA/2)] under GA ~ Poisson(λ_opponent), on-pitch share
  saves      = E[floor(S/3)] under S ~ Poisson(saves90·exposure·opp scaler)
  E[bonus]   = league per-position event coefficients · expected events
               + the player's own bonus deviation rate · exposure
  E[DefCon]  = threshold-crossing rate per 90 (raw counts, rule era) · exposure
  cards/residual = shrunk per-90 rates · exposure (residual = leftover scraps)

Every point value comes from config/scoring_rules_*.yaml — the scoring engine
stays the single source of truth. Double gameweeks sum naturally over the
player's fixtures; blank gameweeks yield 0.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import scoring
from . import (minutes_model, odds_model, rates as rates_mod,
               set_pieces, team_model)

ATTACK_SCALER_CAP = (0.55, 1.75)
GOALS_DISPERSION = 1.078        # var/mean of team goals, 3,060 team-matches (Round 6)
BUDGET_CLIP = (0.80, 1.25)      # how far the squad-budget renormalisation may pull


def variants() -> set[str]:
    """Research arms, switched on per process with $FPL_XPTS_VARIANT
    (comma-separated). Empty on every shipped path, so the engine stays
    bit-identical to the one every number in CLAUDE.md was measured on.

      budget      a club fields exactly 11 starters per fixture: rescale each
                  club's P(start)/P(60+)/P(sub)/E[min] so that sum P(start) = 11
      budget_min  the same law on minutes: sum E[min] = 990 per fixture
      nb_cs       P(clean sheet) from a negative-binomial P(GA=0) with the
                  measured dispersion instead of the Poisson exp(-lambda)
      bonus_gd    the league bonus regression gains a goal-margin term, so
                  E[bonus] rises with the fixture's expected margin
      referee     the appointed referee's prior card rate scales the cards
                  term (BBC officials, Round 20)
      setplay     the opponent's set-play / open-play xGA split reshapes xG
                  by the position's set-play share (BBC match stats, Round 20)
      absent_zero players known absent (bans, dated injuries, Friday "out";
                  $FPL_ABSENCE_KINDS) get zero exposure
      spill       absent_zero, plus their attacking share handed to their
                  club-mates in proportion ($FPL_SPILL_F, default 1.0)
    """
    import os
    return {v.strip() for v in os.environ.get("FPL_XPTS_VARIANT", "").split(",")
            if v.strip()}


def p_zero_goals(lam: float, negbin: bool = False) -> float:
    """P(opponent scores 0). Poisson by default; with ``negbin`` a gamma-
    Poisson mixture with the measured var/mean, which puts a little more
    mass on zero than the Poisson does."""
    if lam <= 0:
        return 1.0
    if not negbin or GOALS_DISPERSION <= 1.0:
        return math.exp(-lam)
    r = lam / (GOALS_DISPERSION - 1.0)
    return (r / (r + lam)) ** r


def apply_budget(df: pd.DataFrame, law: str = "start") -> pd.DataFrame:
    """Renormalise a club's minutes-model output to the conservation law.

    Each player is predicted independently, so nothing forces a club's
    expected starters to add to 11 or its expected minutes to 990. When the
    sum is short (or long) every player in the club is scaled by the same
    factor, clipped so one bad club cannot be pulled to absurd values, and
    probabilities are capped at 1.
    """
    df = df.copy()
    if "team_id" not in df.columns or df.empty:
        return df
    key = "p_start" if law == "start" else "e_min"
    target = 11.0 if law == "start" else 990.0
    sums = df.groupby("team_id")[key].transform("sum")
    f = (target / sums.replace(0, np.nan)).fillna(1.0).clip(*BUDGET_CLIP)
    for col in ("p_start", "p_full", "p_sub", "e_min"):
        if col in df.columns:
            df[col] = df[col].fillna(0.0) * f
    for col in ("p_start", "p_full"):
        df[col] = df[col].clip(upper=1.0)
    over = (df["p_full"] + df["p_sub"]) - 1.0
    df.loc[over > 0, "p_sub"] = (df.loc[over > 0, "p_sub"] - over[over > 0]).clip(lower=0.0)
    if "e_min" in df.columns:
        df["e_min"] = df["e_min"].clip(upper=90.0)
    return df
SAVES_OPP_EXP = 0.35            # saves scale sublinearly with opponent threat
SAVES_OPP_CAP = (0.75, 1.35)


def _e_floor_div(lam: float, per: int, kmax: int = 12) -> float:
    """E[floor(X/per)] for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0
    e, p = 0.0, math.exp(-lam)
    total = p
    for k in range(1, kmax + 1):
        p *= lam / k
        total += p
        e += (k // per) * p
    # tail correction: assume tail mass sits at kmax
    e += (1 - total) * (kmax // per)
    return e


def first_kickoff(conn, season: str, gw: int) -> str | None:
    r = conn.execute(
        "SELECT MIN(kickoff_utc) k FROM fixture WHERE season=? AND gw=?",
        (season, gw)).fetchone()
    if r and r["k"]:
        return r["k"]
    r = conn.execute(   # historical seasons: fixtures live in team_match only
        "SELECT MIN(kickoff_utc) k FROM team_match WHERE season=? AND gw=?",
        (season, gw)).fetchone()
    return r["k"] if r and r["k"] else None


def _gw_fixtures(conn, season: str, gw: int) -> list:
    rows = conn.execute(
        "SELECT f.fixture_id, f.team_h, f.team_a, th.code hcode, ta.code acode "
        "FROM fixture f JOIN team th ON th.season=f.season AND th.team_id=f.team_h "
        "JOIN team ta ON ta.season=f.season AND ta.team_id=f.team_a "
        "WHERE f.season=? AND f.gw=?", (season, gw)).fetchall()
    if rows:
        return rows
    return conn.execute(   # derive from the home-perspective team_match rows
        "SELECT tm.fixture_id, tm.team_id team_h, tm.opponent_id team_a, "
        "th.code hcode, ta.code acode FROM team_match tm "
        "JOIN team th ON th.season=tm.season AND th.team_id=tm.team_id "
        "JOIN team ta ON ta.season=tm.season AND ta.team_id=tm.opponent_id "
        "WHERE tm.season=? AND tm.gw=? AND tm.was_home=1", (season, gw)).fetchall()



def _realised(name: str, row: dict | None, pos: str, rules: dict):
    """A component's ACTUAL points for the gameweek, or None when unknown.

    Used only by the oracle decomposition. A player with no row did not feature
    in the database at all, which is not the same as scoring zero, so he is
    left on the modelled value.
    """
    if not row:
        return None
    import math as _m

    def _n(k):
        v = row.get(k)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    if name == "goals":
        return _n("goals_scored") * rules["goal"].get(pos, 4)
    if name == "assists":
        return _n("assists") * rules["assist"]
    if name == "cs":
        return _n("clean_sheets") * rules["clean_sheet"].get(pos, 0)
    if name == "conceded":
        if pos not in ("GK", "DEF"):
            return 0.0
        gc = rules["goals_conceded"]
        return _m.floor(_n("goals_conceded") / gc["per"]) * gc["points"]
    if name == "saves":
        if pos != "GK":
            return 0.0
        return _m.floor(_n("saves") / rules["saves_per_point"])
    if name == "bonus":
        return _n("bonus")
    if name == "cards":
        return (_n("yellow_cards") * rules["card"]["yellow"]
                + _n("red_cards") * rules["card"]["red"])
    if name == "defcon":
        dc = rules.get("defensive_contribution") or {}
        thr = (dc.get("threshold") or {}).get(pos)
        if not thr:
            return 0.0
        return (dc.get("points", 0) if _n("defcon") >= thr else 0.0)
    if name == "appearance":
        mins, app = _n("minutes"), rules["appearance"]
        return (app["played_60"] if mins >= 60
                else (app["played_any"] if mins > 0 else 0.0))
    return None


def xpts_predict_gw(conn, season: str, gw: int, *, as_of: str | None = None,
                    use_availability: bool = True,
                    minutes_bundle=None, rules: dict | None = None,
                    penalty_takers: dict[int, int] | None = None,
                    odds_weight: float | None = None,
                    team_override: dict[int, int] | None = None,
                    minutes_override: "pd.DataFrame | None" = None,
                    oracle: dict | None = None,
                    rate_scale: dict | None = None,
                    rate_override: "pd.DataFrame | None" = None,
                    lambda_override: dict | None = None,
                    calib: dict | None = None) -> pd.DataFrame:
    """Expected points per player for one gameweek (point-in-time at as_of).

    Returns player_id-indexed frame with the prediction and its components.
    ``penalty_takers`` maps player_id -> penalties_order (1 = first choice),
    available live from bootstrap; first-choice takers get a small xG90 boost.

    ``rate_scale`` multiplies a named rate column before it is used;
    ``rate_override`` replaces named rate columns per player outright, and
    ``lambda_override`` replaces a fixture's (lambda_for, lambda_against).
    All three are research affordances for asking "what would a better
    ESTIMATOR be worth", as distinct from the outcome oracle's "what would
    clairvoyance be worth", and none is ever set on a shipped path.

    ``minutes_override`` and ``oracle`` exist for the ORACLE DECOMPOSITION —
    "what would a perfect estimate of X be worth?" — and are None on every
    shipped path, which stays bit-identical. ``minutes_override`` replaces the
    minutes model's output frame; ``oracle`` is
    ``{"substitute": {"goals", "bonus", ...}, "actual": frame}``, and each
    named component's MODELLED contribution is swapped for the realised one
    after the fixture loop. The swap is a delta on the finished total rather
    than a different way of computing it, so an empty ``substitute`` set
    reproduces the baseline exactly.
    """
    rules = rules or scoring.load_rules()
    as_of = as_of or first_kickoff(conn, season, gw)
    if as_of is None:
        return pd.DataFrame()

    fixtures = _gw_fixtures(conn, season, gw)
    if not fixtures:
        return pd.DataFrame()

    tm = team_model.fit(conn, as_of)
    clf, meta = minutes_bundle or minutes_model.ensure(conn)
    if clf is None:
        raise RuntimeError("minutes model could not be trained — is player_gw "
                           "populated? run `python -m fpl_engine pull` first")
    mins = (minutes_override if minutes_override is not None
            else minutes_model.predict_gw(conn, season, as_of, clf, meta, gw=gw,
                                          use_availability=use_availability))
    rates = rates_mod.fit(conn, season, as_of, rules=rules)
    bonus_coef = rates.attrs.get("bonus_coef", {})   # merge drops attrs
    df = mins.merge(rates.drop(columns=["position"]), on="player_id", how="left")
    team_of = {r["player_id"]: r["team_id"] for r in conn.execute(
        "SELECT player_id, team_id FROM player WHERE season=?", (season,))}
    # `team_override` answers "what is he worth if he moves?" — his own rates
    # against a different club's fixtures. FPL only reclassifies a player once
    # a transfer completes, so between the deal being agreed and that update
    # the engine projects him onto the wrong club's run entirely. Everything
    # downstream keys off team_id, so overriding it here is enough: fixtures,
    # opponent strength and the clean-sheet lambda all follow.
    if team_override:
        team_of = {**team_of, **{int(k): int(v) for k, v in team_override.items()}}
    df["team_id"] = df["player_id"].map(team_of)
    import os as _os
    var = variants()
    if "pressers" in var:
        # Round 18 research arm: manager press-conference statements (BBC,
        # Friday, pre-deadline) scale a player's expected exposure. Factors
        # per class come from $FPL_PRESSER_FACTORS ("out:0.6,doubt:0.85"),
        # fitted on a DIFFERENT season than the one replayed.
        from .. import pressers as _pr
        _f = {}
        for part in _os.environ.get("FPL_PRESSER_FACTORS", "").split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                try:
                    _f[k.strip()] = float(v)
                except ValueError:
                    pass
        try:
            _obs = _pr.observations(conn, season, gw, before=as_of)
        except Exception:      # noqa: BLE001 - no table: arm is a no-op
            _obs = None
        fac = _pr.exposure_factors(_obs, _f or None) if _obs is not None else {}
        if fac:
            m = df["player_id"].map(fac).fillna(1.0)
            for col in ("p_start", "p_full", "p_sub", "e_min"):
                if col in df.columns:
                    df[col] = df[col] * m
            for col in ("p_start", "p_full"):
                df[col] = df[col].clip(upper=1.0)
            over = (df["p_full"] + df["p_sub"]) - 1.0
            df.loc[over > 0, "p_sub"] = (df.loc[over > 0, "p_sub"] - over[over > 0]).clip(lower=0.0)
            df["e_min"] = df["e_min"].clip(upper=90.0)
    if "budget" in var:
        df = apply_budget(df, "start")
    elif "budget_min" in var:
        df = apply_budget(df, "min")
    negbin_cs = "nb_cs" in var
    # per-component level calibration, e.g. FPL_XPTS_CALIB="cs:1.12,bonus:0.95"
    # (research arm; fitted on EARLIER seasons than the one replayed)
    import os as _os
    _calib_env: dict[str, float] = {}
    for part in _os.environ.get("FPL_XPTS_CALIB", "").split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                _calib_env[k.strip()] = float(v)
            except ValueError:
                pass
    # ``calib`` keys are "<component>" or "<POS>:<component>"; a position-
    # specific key wins over the component-wide one
    calib = {**_calib_env, **(calib or {})}

    # per-team fixture list: (λ_for, λ_against). Market odds, where stored,
    # are blended into the team-model rates — the market prices team news and
    # motivation that trailing form cannot see.
    ow = odds_model.ODDS_WEIGHT if odds_weight is None else float(odds_weight)
    omap = (odds_model.fixture_odds_map(
                conn, season, [f["fixture_id"] for f in fixtures])
            if ow > 0 else {})
    team_fixtures: dict[int, list[tuple[float, float]]] = {}
    team_meta: dict[int, list[tuple[int, int]]] = {}     # (fixture_id, opponent), same order
    for f in fixtures:
        lh, la = tm.fixture(f["hcode"], f["acode"])
        lo = (lambda_override or {}).get(f["fixture_id"])
        if lo:
            lh, la = float(lo[0]), float(lo[1])
        od = omap.get(f["fixture_id"])
        if od:
            lh = (1 - ow) * lh + ow * od[0]
            la = (1 - ow) * la + ow * od[1]
        team_fixtures.setdefault(f["team_h"], []).append((lh, la))
        team_fixtures.setdefault(f["team_a"], []).append((la, lh))
        team_meta.setdefault(f["team_h"], []).append((f["fixture_id"], f["team_a"]))
        team_meta.setdefault(f["team_a"], []).append((f["fixture_id"], f["team_h"]))
    league = max(1e-6, tm.league_rate)

    # Penalty duty enters as a CORRECTION toward today's published order, not
    # as a bonus: the player's trailing xG already contains the penalties he
    # took while he had the duty, so a flat boost double-counts the incumbent
    # and does nothing for the player who has just lost it. Zero when the
    # published duty already matches the history. See xpts/set_pieces.py.
    pen = penalty_takers or {}
    try:
        pen_delta = dict(zip(*set_pieces.duty(conn, season, as_of)
                             [["player_id", "pen_xg90_delta"]].to_numpy().T))
    except Exception:      # noqa: BLE001 - duty is an enhancement, never a gate
        pen_delta = {}
    p_goal = rules["goal"]
    p_cs = rules["clean_sheet"]
    p_app_any, p_app_60 = rules["appearance"]["played_any"], rules["appearance"]["played_60"]
    gc_per, gc_pts = rules["goals_conceded"]["per"], rules["goals_conceded"]["points"]
    dc_pts = (rules.get("defensive_contribution") or {}).get("points", 0)

    if rate_override is not None and len(rate_override):
        ro = rate_override.set_index("player_id")
        for _col in ro.columns:
            if _col in df.columns:
                _new = df["player_id"].map(ro[_col])
                df[_col] = pd.to_numeric(_new, errors="coerce").fillna(
                    pd.to_numeric(df[_col], errors="coerce"))
    for _col, _mul in (rate_scale or {}).items():
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce") * float(_mul)

    # ---- Round 20 research arms (env-gated; every shipped path is untouched)
    # referee: the appointed referee's prior card rate scales the cards term
    # setplay: the opponent's set-play / open-play xGA split reshapes xG by
    #          the player's positional set-play share (mean-preserving)
    # absent_zero: players KNOWN absent (bans, dated injuries, Friday "out")
    #          get zero exposure; spill: additionally their attacking mass is
    #          handed to their club-mates in proportion ($FPL_SPILL_F)
    ref_map: dict = {}
    sp_map: dict = {}
    pos_set_share: dict = {}
    if "referee" in var or "setplay" in var:
        from . import bbc_context as _bc
        ref_map, sp_map = _bc.factor_maps(conn, season, [f["fixture_id"] for f in fixtures])
        if "referee" not in var:
            ref_map = {}
        if "setplay" not in var:
            sp_map = {}
        pos_set_share = _bc.POS_SET_SHARE if sp_map else {}
    if "spill" in var or "absent_zero" in var:
        from . import absence as _abs
        _out_ids = _abs.known_out_ids(conn, season, gw, as_of)
        if _out_ids:
            _is_out = df["player_id"].isin(list(_out_ids))
            if "spill" in var:
                _att = ((pd.to_numeric(df["xg90"], errors="coerce").fillna(0.0)
                         + pd.to_numeric(df["xa90"], errors="coerce").fillna(0.0))
                        * pd.to_numeric(df["e_min"], errors="coerce").fillna(0.0) / 90.0)
                _a_out = _att.where(_is_out, 0.0).groupby(df["team_id"]).transform("sum")
                _t_all = _att.groupby(df["team_id"]).transform("sum")
                _f = float(_os.environ.get("FPL_SPILL_F", "1.0"))
                _mult = (1.0 + _f * _a_out / (_t_all - _a_out).clip(lower=1e-6)).where(~_is_out, 1.0)
                df["xg90"] = pd.to_numeric(df["xg90"], errors="coerce") * _mult
                df["xa90"] = pd.to_numeric(df["xa90"], errors="coerce") * _mult
            for _col in ("p_start", "p_full", "p_sub", "e_min"):
                if _col in df.columns:
                    df.loc[_is_out, _col] = 0.0

    sub = set((oracle or {}).get("substitute") or ())
    actual = (oracle or {}).get("actual")
    act = ({int(k): v for k, v in actual.set_index("player_id").to_dict("index").items()}
           if actual is not None and len(actual) else {})

    rows = []
    for r in df.itertuples():
        fx = team_fixtures.get(r.team_id, [])
        pos = r.position or "MID"
        exposure = (r.e_min or 0.0) / 90.0
        p_play = (r.p_sub or 0) + (r.p_full or 0)
        xg90 = max(0.0, (r.xg90 or 0.0) + pen_delta.get(r.player_id, 0.0))
        if pen.get(r.player_id) == 1 and not pen_delta:
            # no shot history to correct against: fall back to the old flat
            # first-choice boost rather than ignoring duty entirely
            xg90 += 0.10
        total = 0.0
        e_goals = e_assists = e_cs = 0.0
        comp = {k: 0.0 for k in ("goals", "assists", "cs", "conceded",
                                 "saves", "bonus", "cards", "defcon",
                                 "appearance", "residual")}
        meta = team_meta.get(r.team_id, [])
        for k_fx, (lam_for, lam_against) in enumerate(fx):
            fid, _opp = meta[k_fx] if k_fx < len(meta) else (None, None)
            scaler = float(np.clip(lam_for / league, *ATTACK_SCALER_CAP))
            g = xg90 * exposure * scaler
            if sp_map:
                _sp = sp_map.get((fid, r.team_id))
                if _sp:
                    _si = pos_set_share.get(pos, 0.2)
                    g *= _si * _sp[0] + (1.0 - _si) * _sp[1]
            _rf = ref_map.get(fid, 1.0) if ref_map else 1.0
            a = (r.xa90 or 0.0) * exposure * scaler
            cs = (r.p_full or 0.0) * p_zero_goals(lam_against, negbin_cs)
            e_goals += g
            e_assists += a
            e_cs += cs
            total += g * p_goal.get(pos, 4) + a * rules["assist"]
            comp["goals"] += g * p_goal.get(pos, 4)
            comp["assists"] += a * rules["assist"]
            total += cs * p_cs.get(pos, 0)
            comp["cs"] += cs * p_cs.get(pos, 0)
            if pos in ("GK", "DEF"):
                _gc = gc_pts * _e_floor_div(lam_against * max(p_play, 0.0),
                                            gc_per)
                total += _gc
                comp["conceded"] += _gc
            if pos == "GK":
                sv = float(np.clip((lam_against / league) ** SAVES_OPP_EXP,
                                   *SAVES_OPP_CAP))
                _sv = _e_floor_div((r.saves90 or 0.0) * exposure * sv,
                                   rules["saves_per_point"])
                total += _sv
                comp["saves"] += _sv
            bc = bonus_coef.get(pos)
            if bc and len(bc) == 5:   # bonus_gd arm: + expected goal margin
                _b = (bc[0] * g + bc[1] * a + bc[2] * cs
                      + bc[3] * (lam_for - lam_against) * p_play + bc[4] * p_play)
            elif bc:   # E[bonus] from expected events + player deviation
                _b = bc[0] * g + bc[1] * a + bc[2] * cs + bc[3] * p_play
                total += _b
                comp["bonus"] += _b
            total += (r.bonus_resid90 or 0.0) * exposure
            comp["bonus"] += (r.bonus_resid90 or 0.0) * exposure
            total += (r.yellow_cards90 or 0.0) * exposure * rules["card"]["yellow"] * _rf
            comp["cards"] += (r.yellow_cards90 or 0.0) * exposure * rules["card"]["yellow"] * _rf
            total += (r.defcon_cross90 or 0.0) * exposure * dc_pts
            comp["defcon"] += (r.defcon_cross90 or 0.0) * exposure * dc_pts
            total += (r.residual90 or 0.0) * exposure
            comp["residual"] += (r.residual90 or 0.0) * exposure
            total += (r.p_sub or 0.0) * p_app_any + (r.p_full or 0.0) * p_app_60
            comp["appearance"] += ((r.p_sub or 0.0) * p_app_any
                                   + (r.p_full or 0.0) * p_app_60)
        if calib:
            for k in list(comp):
                mul = calib.get(f"{pos}:{k}", calib.get(k))
                if mul is not None and mul != 1.0:
                    total += comp[k] * (mul - 1.0)
                    comp[k] *= mul
        if sub:
            a_row = act.get(int(r.player_id))
            for name in sub:
                real = _realised(name, a_row, pos, rules)
                if real is not None:
                    total += real - comp.get(name, 0.0)
        rows.append({
            "player_id": r.player_id, "position": pos, "team_id": r.team_id,
            "n_fixtures": len(fx),
            "p_play": round(p_play, 4), "p_60": round(r.p_full or 0.0, 4),
            "p_start": round(getattr(r, "p_start", None) or 0.0, 4),
            "e_min": round(r.e_min or 0.0, 1),
            "e_goals": round(e_goals, 3), "e_assists": round(e_assists, 3),
            "p_cs": round(e_cs, 3),
            "prediction": round(total, 3),
            # the modelled points per component, for calibration audits
            **{f"c_{k}": round(v, 4) for k, v in comp.items()},
        })
    return pd.DataFrame(rows)
