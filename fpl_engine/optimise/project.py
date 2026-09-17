"""Build a per-player projection table across a planning horizon.

Produces one row per player with position, club, price, availability and the
OpenFPL-projected points for each gameweek in the horizon (plus a discounted
total). This is the input the MILP optimises over.

Forward projections use current form: no matches occur between now and a future
gameweek, so each horizon gameweek's point-in-time features equal today's form
applied against that gameweek's fixture (opponent). Fixture difficulty therefore
still varies across the horizon via the opponent features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, features, minutes, predict as predict_mod

# Pre-season prior blend (stop-gap for the form model's stale trailing
# windows): at GW1 the model EP is blended with last season's shrunken
# points-per-90 x expected minutes, fading to zero once PRESEASON_BLEND_GWS
# gameweeks of the new season have been played.
PRESEASON_BLEND_MAX = 0.5
PRESEASON_BLEND_GWS = 3
PRIOR_SHRINK_MINS = 450.0      # minutes of position-mean rate mixed into each player

# Fallback share of a projection that arrives as goals/assists/bonus, used
# only when the component engine is not in the blend and so cannot say.
# Rough league averages; the real number comes from `c_*` per player per gw.
EXPLOSIVE_SHARE = {"GK": 0.08, "DEF": 0.22, "MID": 0.45, "FWD": 0.62}

# the engine's per-component points (xpts/engine.py `c_*`), carried for display
COMPONENTS = ("goals", "assists", "bonus", "cs", "defcon", "saves",
              "appearance", "conceded", "cards")


def preseason_weight(n_played: int) -> float:
    """Weight on the last-season prior after ``n_played`` finished gameweeks."""
    return PRESEASON_BLEND_MAX * max(0.0, 1.0 - n_played / PRESEASON_BLEND_GWS)


def _played_gws(conn, season: str) -> int:
    return conn.execute(
        "SELECT COUNT(DISTINCT gw) FROM fixture WHERE season=? AND finished=1",
        (season,)).fetchone()[0]


def preseason_priors(conn, season: str, profiles: dict) -> dict[int, float]:
    """player_id -> prior EP per gw: last season's total points per 90
    (shrunk toward the position mean by PRIOR_SHRINK_MINS minutes) times the
    expected minutes from the minutes profile. Players without last-season
    minutes or without a profile get no prior."""
    y = int(season.split("-")[0])
    prev = f"{y - 1}-{str(y)[-2:]}"
    agg = {r["player_code"]: (r["pts"] or 0.0, r["mins"] or 0.0) for r in conn.execute(
        """
        WITH m AS (
            SELECT player_code, MAX(total_points) AS pts, MAX(minutes) AS mins
            FROM player_gw WHERE season=? AND player_code IS NOT NULL
            GROUP BY player_code, gw, fixture_id
        )
        SELECT player_code, SUM(pts) AS pts, SUM(mins) AS mins FROM m
        GROUP BY player_code
        """, (prev,))}
    players = conn.execute("SELECT player_id, code, position FROM player WHERE season=?",
                           (season,)).fetchall()
    # position means (minutes-weighted) for shrinkage
    tot: dict[str, list[float]] = {}
    for p in players:
        pts, mins = agg.get(p["code"], (0.0, 0.0))
        t = tot.setdefault(p["position"], [0.0, 0.0])
        t[0] += pts
        t[1] += mins
    pos90 = {pos: (v[0] / v[1] * 90.0 if v[1] else 0.0) for pos, v in tot.items()}
    out: dict[int, float] = {}
    for p in players:
        pts, mins = agg.get(p["code"], (0.0, 0.0))
        prof = profiles.get(int(p["player_id"]))
        if mins <= 0 or not prof:
            continue
        rate = (pts + pos90.get(p["position"], 0.0) * PRIOR_SHRINK_MINS / 90.0) \
            / (mins + PRIOR_SHRINK_MINS) * 90.0
        out[int(p["player_id"])] = rate * prof["xmins"] / 90.0
    return out


def horizon_projections(conn, season: str, gws: list[int], *, bundle=None,
                        decay: float = 0.85, retrained=None,
                        blend: float = 0.0, xpts_w: float | None = None,
                        penalty_takers: dict[int, int] | None = None) -> pd.DataFrame:
    """Return a projection dataframe indexed by player_id.

    Columns: player_id, player, position, team, team_id, price, available,
    xmins, ep_gw{g} for each g, and ep_total (decayed sum).

    EP is scaled per player by the expected-minutes factor from
    ``fpl_engine.minutes`` (injury flags + start-pattern shifts relative to
    the trailing baseline the model features already assume). Players with
    no match history fall back to the plain availability multiplier.

    ``xpts_w`` blends the component xPts engine into each gameweek's model EP:
    final = (1-w)*openfpl + w*xpts (weight fitted by the backtest and stored
    in models/xpts/blend.json — see ``fpl_engine.pipeline.xpts_weight``).
    """
    bundle = bundle or predict_mod.load_models()

    minutes_bundle = None
    if xpts_w:
        from ..xpts import minutes_model
        minutes_bundle = minutes_model.ensure(conn)
        if minutes_bundle[0] is None:
            xpts_w = None      # xpts untrained -> degrade to pure OpenFPL

    # Static player attributes (price, club, availability) for the season.
    attrs = {r["player_id"]: dict(r) for r in conn.execute(
        "SELECT player_id, full_name, position, team_id, now_cost, status, "
        "chance_next FROM player WHERE season=?", (season,))}
    team_name = {r["team_id"]: r["name"] for r in conn.execute(
        "SELECT team_id, name FROM team WHERE season=?", (season,))}

    # Expected-minutes profiles at the horizon's point-in-time boundary (no
    # matches occur inside the horizon, so one profile serves every gw).
    profiles = minutes.minutes_profiles(
        conn, season, features.gw_as_of(conn, season, gws[0]) if gws else None)
    prior_w = preseason_weight(_played_gws(conn, season))
    priors = preseason_priors(conn, season, profiles) if prior_w > 0 else {}

    # Expected minutes FOR DISPLAY come from the xpts minutes model, not from
    # `minutes.minutes_profiles`. The latter is a trailing-history estimate
    # kept for the OpenFPL scaling factor and the pre-season prior, and it
    # reports a nailed starter's realised average — 90.0 for a player who has
    # started once and played the full match. The model, which is what the
    # engine actually prices him with, says 81.2 for the same player because it
    # carries the chance he is rested or hooked. Showing the trailing number
    # next to model points repeats the Round 8b mistake, where the UI displayed
    # a trailing start rate against a calibrated P(start) and was materially
    # worse. Nothing about the projection changes here — only what is reported.
    model_xmins: dict[int, float] = {}
    if minutes_bundle and minutes_bundle[0] is not None and gws:
        try:
            from ..xpts import minutes_model as _mm
            _as_of = features.gw_as_of(conn, season, gws[0])
            _mf = _mm.predict_gw(conn, season, _as_of, minutes_bundle[0],
                                 minutes_bundle[1], gw=gws[0])
            model_xmins = {int(r.player_id): float(r.e_min)
                           for r in _mf.itertuples()}
        except Exception:
            model_xmins = {}

    from .. import progress
    ep_by_gw: dict[int, dict[int, float]] = {}
    # The lumpy half of a projection: points that arrive as goals, assists and
    # the bonus they attract, as opposed to appearance points, clean sheets,
    # saves and DefCon, which turn up almost every week a player starts. Two
    # players on the same expected points are not the same bet, and the solver
    # playstyles are the place that distinction is allowed to matter (see
    # optimise/style.py). Only the component engine can supply it, so a
    # pure-OpenFPL run falls back to a per-position share.
    ex_by_gw: dict[int, dict[int, float]] = {}
    # The whole breakdown, for display: what KIND of player a projection is
    # (an attacking return, a clean sheet, a DefCon crossing) and the raw
    # expectations behind it. Nothing downstream of the solver reads these;
    # the Live desk does, to say why a player is a pick.
    comp_by_gw: dict[int, dict[int, dict]] = {}
    for g in gws:
        progress.log(f"    projecting GW{g}…")
        try:
            df = features.build_samples(conn, season, g, include_ids=True)
        except ValueError:
            continue  # gw not scheduled
        preds = predict_mod.predict(df, bundle=bundle, retrained=retrained,
                                    blend=blend)
        preds = preds.reset_index(drop=True)
        # align predictions to player_id via the id column carried on df
        # (predict preserves row order within each position block, so join by
        # the metadata key instead to be safe)
        merged = preds.merge(
            df[["player", "team", "position", "player_id"]].drop_duplicates(
                ["player", "team", "position"]),
            on=["player", "team", "position"], how="left")
        ep_by_gw[g] = {int(pid): float(ep) for pid, ep in
                       zip(merged["player_id"], merged["prediction"])
                       if pd.notna(pid)}
        if xpts_w:
            # blend the component engine in (availability handled below for
            # both, so the engine runs without its own availability overlay)
            from ..xpts import engine as xpts_engine
            xdf = xpts_engine.xpts_predict_gw(
                conn, season, g, use_availability=False,
                minutes_bundle=minutes_bundle, penalty_takers=penalty_takers)
            if not xdf.empty:
                xmap = dict(zip(xdf["player_id"].astype(int),
                                xdf["prediction"].astype(float)))
                ex_cols = [c for c in ("c_goals", "c_assists", "c_bonus")
                           if c in xdf.columns]
                if ex_cols:
                    ex_by_gw[g] = dict(zip(
                        xdf["player_id"].astype(int),
                        xdf[ex_cols].sum(axis=1).astype(float)))
                comp_by_gw[g] = {
                    int(r["player_id"]): {
                        "pred": float(r.get("prediction") or 0.0),
                        **{k: float(r.get(f"c_{k}") or 0.0) for k in COMPONENTS},
                        "eg": float(r.get("e_goals") or 0.0),
                        "ea": float(r.get("e_assists") or 0.0),
                        "pcs": float(r.get("p_cs") or 0.0),
                    }
                    for r in xdf.to_dict("records")}
                ep_by_gw[g] = {
                    pid: (1 - xpts_w) * v + xpts_w * xmap.get(pid, v)
                    for pid, v in ep_by_gw[g].items()}
                # players OpenFPL has no sample for (e.g. new signings with no
                # trailing windows) still get an xPts value
                for pid, xv in xmap.items():
                    ep_by_gw[g].setdefault(pid, xpts_w * xv)

    # Round 18: what the manager said on Friday (BBC press conferences),
    # applied as an exposure factor on top of FPL's own status. Live path
    # only — the backtest never reaches this function.
    try:
        from .. import pressers as _pressers
        presser = _pressers.live_overlay(
            conn, season, gws, lambda g: features.gw_as_of(conn, season, g))
    except Exception:      # noqa: BLE001
        presser = {}

    rows = []
    for pid, a in attrs.items():
        if a["position"] not in ("GK", "DEF", "MID", "FWD"):
            continue
        eps = {g: ep_by_gw.get(g, {}).get(pid, np.nan) for g in gws}
        if all(np.isnan(v) for v in eps.values()):
            continue  # never plays in the horizon (e.g. no fixture)
        avail = a["chance_next"] if a["chance_next"] is not None else (
            1.0 if a["status"] in (None, "a") else 0.0)
        prof = profiles.get(pid)
        factor = prof["factor"] if prof else avail
        prior = priors.get(pid)
        vals = {}
        ex_vals = {}
        comp_vals = {}
        for g in gws:
            v = eps[g]
            if np.isnan(v):
                vals[g] = 0.0
                ex_vals[g] = 0.0
                continue
            # the explosive SHARE survives every scaling below (availability,
            # press-conference factors and the pre-season prior all scale the
            # whole projection), so it is captured before them and re-applied
            raw_ex = ex_by_gw.get(g, {}).get(pid)
            share = (max(0.0, min(1.0, raw_ex / v)) if raw_ex is not None and v > 0
                     else EXPLOSIVE_SHARE.get(a["position"], 0.4))
            v = v * factor
            pr = presser.get(g, {}).get(pid)
            if pr:
                v = v * pr["factor"]
            if prior is not None and prior_w > 0:
                v = (1.0 - prior_w) * v + prior_w * prior
            vals[g] = v
            ex_vals[g] = v * share
            # every component takes the same exposure scaling the projection
            # took (availability, press-conference factor, pre-season prior),
            # so the parts still add up to the whole
            raw = comp_by_gw.get(g, {}).get(pid)
            if raw is not None and raw["pred"] > 0:
                k = v / raw["pred"]
                comp_vals[g] = {
                    **{c: round(raw[c] * k, 3) for c in COMPONENTS},
                    "eg": round(raw["eg"] * k, 3), "ea": round(raw["ea"] * k, 3),
                    "pcs": round(min(1.0, raw["pcs"] * k), 3)}
        total = sum((decay ** i) * vals[g] for i, g in enumerate(gws)
                    if not np.isnan(eps[g]))
        row = {
            "player_id": pid, "player": a["full_name"], "position": a["position"],
            "team_id": a["team_id"], "team": team_name.get(a["team_id"]),
            "price": a["now_cost"] or 0.0, "available": avail,
            "xmins": model_xmins.get(pid,
                                      prof["xmins"] if prof else None),
            "prior_w": prior_w if prior is not None else 0.0,
            "ep_total": total,
            # the coming gameweek's manager statement, for the player card
            "presser": (presser.get(gws[0], {}).get(pid) if gws else None),
        }
        for g in gws:
            row[f"ep_gw{g}"] = vals[g]
            row[f"ex_gw{g}"] = ex_vals[g]
            row[f"comp_gw{g}"] = comp_vals.get(g)
        rows.append(row)

    proj = pd.DataFrame(rows)
    return proj.sort_values("ep_total", ascending=False).reset_index(drop=True)


def prune(proj: pd.DataFrame, *, keep_per_position: int = 30,
          cheap_per_position: int = 8, must_keep: set[int] | None = None) -> pd.DataFrame:
    """Keep the strongest and the cheapest players per position (plus owned).

    Shrinks the MILP without changing the optimum: only strong players, the
    incumbent squad, and cheap "enabler" fodder (needed to afford premiums under
    the £100m budget) can appear in an optimal 15. Keeping cheap options per
    position guarantees the budget constraint stays feasible.
    """
    must_keep = must_keep or set()
    keep = proj[proj["player_id"].isin(must_keep)]
    top = (proj.sort_values("ep_total", ascending=False)
               .groupby("position", group_keys=False).head(keep_per_position))
    cheap = (proj.sort_values(["price", "ep_total"], ascending=[True, False])
                 .groupby("position", group_keys=False).head(cheap_per_position))
    out = (pd.concat([top, cheap, keep])
             .drop_duplicates("player_id").reset_index(drop=True))
    return out
