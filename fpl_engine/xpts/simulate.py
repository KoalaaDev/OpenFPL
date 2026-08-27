"""Correlated match simulator: FPL points as a joint distribution.

The analytic engine (``xpts.engine``) returns E[points] per player, computed
independently. This simulates the match itself, so every player's outcome
hangs off the same drawn scoreline:

  1. draw the scoreline           G_h ~ Poisson(lam_h), G_a ~ Poisson(lam_a)
  2. draw who is on the pitch     from the minutes model's three classes
  3. give each goal a scorer      multinomial over xG90 x minutes played
  4. give it an assister          multinomial over xA90 x minutes played
  5. clean sheets, conceded, saves, cards and DefCon follow from that draw
  6. bonus is the top-3 BPS rank *within the simulated match*
  7. convert with the scoring YAML's constants

What it is for
--------------
**Joint risk.** Independence understates the spread of a starting XI's total by
~8% (sd 15.0 vs 13.9 over 74 replayed gameweeks) and of a one-club triple-up by
~6%; across three different clubs the ratio is 1.00. Any chip, rank or
differential calculation done player-by-player is wrong by about that much.
It also gives the floor/ceiling/P(haul) view a single expectation cannot.

What it is NOT for
------------------
**Ranking.** Measured, not assumed: its mean tracks the analytic engine at
r = 0.992 but ranks *worse* (spearman_played -0.0043, p = 0.012), and inside a
realistic captain choice set — 740 candidate player-gameweeks — every
criterion built from the distribution (mean, median, P(haul>=10), 90th
percentile, mean±sd) ranks the same as plain expected points, all pairwise
p > 0.2. Use ``engine.xpts_predict_gw`` to rank. Use this for risk.

Point values come from the rules dict, so the scoring YAML stays the single
source of truth (CLAUDE.md principle #2); they are unpacked into arrays only
because calling ``points_from_events`` per simulated player-match would be
hopelessly slow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import scoring
from . import engine as engine_mod, minutes_model, odds_model, rates as rates_mod, team_model

P_ASSISTED = 0.75          # share of goals credited an assist
BPS_SD_FLOOR = 3.0
BPS_HALF_LIFE_DAYS = 240.0
BPS_SEASON_BREAK_DECAY = 0.7
K_BPS_90S = 6.0
DEFAULT_SIMS = 4000
SUB_MINUTES = 22.0

# Plug-in simulators are underdispersed: they treat every fitted rate as if it
# were known exactly. Measured against 55k player-gameweeks the point-estimate
# version understated P(haul >= 10) by ~27% and a starting XI's spread by ~11%
# (predicted sd 15.2 against a realised 17.1), which is precisely the error
# that makes a risk engine undervalue high-variance plays. These add the
# parameter uncertainty back.
RATE_DISPERSION = True     # who converts the chances is drawn, not fixed
SHARE_CONCENTRATION = 12.0  # Dirichlet concentration on the goal/assist shares
LAMBDA_K = 25.0            # team goals are mildly overdispersed vs Poisson
                            # (measured var/mean 1.078 over 3,060 team-matches)
MINUTES_SD = 9.0           # spread of minutes given a 60+ appearance

# Game state. A static Poisson draw says a team attacks the same way at 0-0 and
# at 3-0, which is false. Measured WITHIN team-match (each team against its own
# output while level, so team strength, opponent and venue all difference out;
# 1,160 matches with an exactly reconstructable running score, 29,776 shots):
#
#     score difference   -2 or worse   -1   level   +1   +2 or better
#     xG per 90 vs level        1.251  1.330   1.00  0.729       0.762
#     of which shots/90         1.333  1.287   1.00  0.760       0.755
#     of which xG per shot      0.939  1.033   1.00  0.960       1.009
#
# So it is almost entirely VOLUME: a chasing side takes ~29% more shots of the
# same quality and a leading side ~24% fewer. Note the naive split — not
# differenced within match — reports the opposite sign (+26% when two up),
# because a team two goals up is disproportionately a good team.
# ...and yet this is OFF, because the effect is already priced in. The team
# model's lambda is fitted on *realised* goals, which by construction contain
# every game-state dynamic that actually happened, so re-applying it
# double-counts. Scored against 1,480 replayed team-matches, the predicted
# distribution of a team's goals is equally calibrated either way (PIT
# uniformity chi2 9.9 static vs 8.9 path-dependent, against a 16.9 critical
# value — the difference is noise). Turning it on narrows the scoreline
# distribution (P(0 goals) 0.134 -> 0.101, mean |score difference| 1.53 ->
# 1.36) for no measurable gain.
#
# Kept, tested and switchable because the effect itself is real and the
# conclusion is specific to a lambda estimated from goals: if the team model
# ever moves to a basis that does not already absorb it, re-run that test.
GAME_STATE = False
STATE_MULT = {-2: 1.251, -1: 1.330, 0: 1.0, 1: 0.729, 2: 0.762}
STATE_INTERVALS = 6        # 15-minute blocks; the score updates between them


# ------------------------------------------------------------------ BPS -----
def fit_bps(conn, season: str, as_of: str):
    """Per-position BPS coefficients, personal deviations and residual sd.

    Bonus is a *rank* within a match, so the simulator needs a BPS scale, not
    the bonus regression ``rates.fit`` already provides. Same shape: a league
    per-position weighted least squares on the events the engine predicts, and
    each player keeps only his deviation from it.
    """
    hist = pd.read_sql_query(
        "SELECT pg.season, pg.player_code, pg.kickoff_utc, pg.minutes, pg.bps, "
        "pg.goals_scored, pg.assists, pg.clean_sheets, "
        "(SELECT position FROM player p WHERE p.season=pg.season "
        " AND p.player_id=pg.player_id) position "
        "FROM player_gw pg WHERE pg.kickoff_utc < ? AND pg.minutes > 0",
        conn, params=(as_of,))
    players = pd.read_sql_query(
        "SELECT player_id, code player_code FROM player WHERE season=?",
        conn, params=(season,))
    if hist.empty:
        return {}, players.assign(bps_resid90=0.0)[["player_id", "bps_resid90"]], {}

    ref = pd.Timestamp(as_of.replace("Z", "+00:00"))
    days = (ref - pd.to_datetime(hist["kickoff_utc"], utc=True,
                                 format="ISO8601")).dt.days.clip(lower=0)
    hist["w"] = 0.5 ** (days / BPS_HALF_LIFE_DAYS)
    breaks = (int(season[:4]) - hist["season"].str[:4].astype(int)).clip(lower=0)
    hist["w"] *= BPS_SEASON_BREAK_DECAY ** breaks
    hist["w90"] = hist["w"] * hist["minutes"] / 90.0
    hist = hist.dropna(subset=["position"])

    coef, sd = {}, {}
    hist["_fit"] = np.nan
    for pos, d in hist.groupby("position"):
        X = np.c_[d["goals_scored"].fillna(0), d["assists"].fillna(0),
                  d["clean_sheets"].fillna(0), d["minutes"].fillna(0),
                  np.ones(len(d))]
        y = d["bps"].fillna(0).to_numpy(float)
        sw = np.sqrt(d["w"].to_numpy(float))
        c, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        coef[pos] = [float(v) for v in c]
        pred = X @ c
        hist.loc[d.index, "_fit"] = pred
        sd[pos] = float(np.sqrt(np.average((y - pred) ** 2, weights=d["w"])))

    hist["resid"] = hist["bps"].fillna(0) - hist["_fit"]
    agg = hist.groupby("player_code").agg(
        e=("w90", "sum"), r=("resid", "sum"))
    agg["r"] = (hist["w"] * hist["resid"]).groupby(hist["player_code"]).sum()
    out = players.merge(agg, left_on="player_code", right_index=True, how="left")
    # residuals are zero-mean per position by construction, so the prior is 0
    out["bps_resid90"] = out["r"].fillna(0) / (out["e"].fillna(0) + K_BPS_90S)
    return coef, out[["player_id", "bps_resid90"]], sd


# ------------------------------------------------------------- assemble -----
def _inputs(conn, season, gw, as_of, rules, minutes_bundle, use_availability,
            odds_weight=None):
    tm = team_model.fit(conn, as_of)
    fixtures = engine_mod._gw_fixtures(conn, season, gw)
    if not fixtures:
        return None
    clf, meta = minutes_bundle or minutes_model.ensure(conn)
    mins = minutes_model.predict_gw(conn, season, as_of, clf, meta, gw=gw,
                                    use_availability=use_availability)
    rates = rates_mod.fit(conn, season, as_of, rules=rules)
    df = mins.merge(rates.drop(columns=["position"]), on="player_id", how="left")
    team_of = {r["player_id"]: r["team_id"] for r in conn.execute(
        "SELECT player_id, team_id FROM player WHERE season=?", (season,))}
    df["team_id"] = df["player_id"].map(team_of)

    ow = odds_model.ODDS_WEIGHT if odds_weight is None else float(odds_weight)
    omap = (odds_model.fixture_odds_map(
                conn, season, [f["fixture_id"] for f in fixtures])
            if ow > 0 else {})
    lams = {}
    for f in fixtures:
        lh, la = tm.fixture(f["hcode"], f["acode"])
        od = omap.get(f["fixture_id"])
        if od:
            lh = (1 - ow) * lh + ow * od[0]
            la = (1 - ow) * la + ow * od[1]
        lams[(f["team_h"], f["fixture_id"])] = (lh, la)
        lams[(f["team_a"], f["fixture_id"])] = (la, lh)

    # E[minutes | he appears], published by the minutes model. (Do not try to
    # invert e_min: it is P(plays) x this, not a blend of two class means.)
    df["m_started"] = df["m_played"].clip(60, 90) if "m_played" in df else 85.0
    for c in ("xg90", "xa90", "saves90", "yellow_cards90", "defcon_cross90",
              "residual90", "exposure"):
        df[c] = df[c].fillna(0.0) if c in df else 0.0
    return df, fixtures, lams


def simulate_gw(conn, season: str, gw: int, *, as_of: str | None = None,
                n_sims: int = DEFAULT_SIMS, seed: int | None = None,
                use_availability: bool = True, minutes_bundle=None,
                rules: dict | None = None, odds_weight: float | None = None,
                with_bonus: bool = True, dispersion: bool | None = None,
                game_state: bool | None = None) -> dict:
    """Simulate one gameweek. Returns {"players": ids, "points": (n_sims, n)}."""
    rules = rules or scoring.load_rules()
    as_of = as_of or engine_mod.first_kickoff(conn, season, gw)
    if as_of is None:
        return {"players": np.array([], dtype=int),
                "points": np.zeros((n_sims, 0), dtype=np.float32)}
    got = _inputs(conn, season, gw, as_of, rules, minutes_bundle,
                  use_availability, odds_weight)
    if got is None:
        return {"players": np.array([], dtype=int),
                "points": np.zeros((n_sims, 0), dtype=np.float32)}
    x, fixtures, team_lams = got
    coef, resid, sd = fit_bps(conn, season, as_of)
    return _run(x, fixtures, team_lams, rules, coef, resid, sd,
                n_sims=n_sims, seed=gw if seed is None else seed,
                with_bonus=with_bonus,
                dispersion=RATE_DISPERSION if dispersion is None else dispersion,
                game_state=GAME_STATE if game_state is None else game_state)


def _state_factor(diff: np.ndarray) -> np.ndarray:
    """Attacking multiplier for the current score difference."""
    d = np.clip(diff, -2, 2)
    out = np.empty(d.shape, dtype=float)
    for k, v in STATE_MULT.items():
        out[d == k] = v
    return out


def _draw_with_state(lam_of, home, away, n_sims, rng):
    """Goals drawn block by block, with the rate reacting to the scoreline.

    The multipliers are normalised so the unconditional expectation is still
    the team's lambda — the point is to reshape the distribution, not to move
    its mean, which is the same discipline the Dirichlet share draw follows.
    """
    per = {s_: lam_of[s_] / STATE_INTERVALS for s_ in (home, away)}
    tot = {s_: np.zeros(n_sims, dtype=np.int64) for s_ in (home, away)}
    diff = np.zeros(n_sims, dtype=np.int64)      # home perspective
    for _ in range(STATE_INTERVALS):
        fh = _state_factor(diff)
        fa = _state_factor(-diff)
        gh = rng.poisson(per[home] * fh)
        ga = rng.poisson(per[away] * fa)
        tot[home] += gh
        tot[away] += ga
        diff += gh - ga
    # restore the intended mean: the state path is mean-reverting, so leaving
    # it uncorrected would quietly shift every team's expected goals
    for s_ in (home, away):
        want = lam_of[s_].mean()
        got = tot[s_].mean()
        if got > 1e-9 and want > 1e-9:
            keep = rng.random(tot[s_].shape) < min(1.0, want / got)
            extra = rng.poisson(np.maximum(0.0, want - got), n_sims)
            tot[s_] = np.where(keep, tot[s_], 0) + extra
    return tot


def _dirichlet_like(w: np.ndarray, conc: float, rng) -> np.ndarray:
    """Redraw each row's weights from Dirichlet(conc * share), same total.

    E[share] is preserved exactly, so the mean is untouched and only the
    spread widens.
    """
    tot = w.sum(1, keepdims=True)
    share = np.divide(w, np.where(tot > 0, tot, 1.0))
    alpha = np.clip(conc * share, 1e-3, None)
    g = rng.gamma(alpha, 1.0)
    gs = g.sum(1, keepdims=True)
    return np.divide(g, np.where(gs > 0, gs, 1.0)) * tot


def _run(x, fixtures, team_lams, rules, bps_coef, bps_resid, bps_sd, *,
         n_sims, seed, with_bonus=True, dispersion=True, game_state=True):
    rng = np.random.default_rng(seed)
    ids = x["player_id"].to_numpy()
    idx = {p: i for i, p in enumerate(ids)}
    pts = np.zeros((n_sims, len(ids)), dtype=np.float32)

    p_goal, p_cs = rules["goal"], rules["clean_sheet"]
    p_app1 = rules["appearance"]["played_any"]
    p_app60 = rules["appearance"]["played_60"]
    gc_per, gc_pts = rules["goals_conceded"]["per"], rules["goals_conceded"]["points"]
    dc_pts = (rules.get("defensive_contribution") or {}).get("points", 0)
    sv_per, yellow = rules["saves_per_point"], rules["card"]["yellow"]
    assist_pts = rules["assist"]

    resid_of = dict(zip(bps_resid["player_id"], bps_resid["bps_resid90"]))
    by_team = {t: g for t, g in x.groupby("team_id")}

    for f in fixtures:
        fid = f["fixture_id"]
        if by_team.get(f["team_h"]) is None or by_team.get(f["team_a"]) is None:
            continue
        lam_of = {}
        for side in (f["team_h"], f["team_a"]):
            lam = team_lams.get((side, fid), (1.4, 1.4))[0]
            if dispersion:      # the team rate is itself an estimate
                lam = rng.gamma(LAMBDA_K, lam / LAMBDA_K, n_sims)
            lam_of[side] = np.broadcast_to(np.asarray(lam, dtype=float),
                                           (n_sims,)).copy()
        if game_state:
            goals = _draw_with_state(lam_of, f["team_h"], f["team_a"],
                                     n_sims, rng)
        else:
            goals = {s_: rng.poisson(lam_of[s_], n_sims)
                     for s_ in (f["team_h"], f["team_a"])}
        side = {}
        for team in (f["team_h"], f["team_a"]):
            d = by_team[team]
            n = len(d)
            p3 = np.clip(np.c_[d["p_none"], d["p_sub"], d["p_full"]], 0, None)
            p3 = p3 / p3.sum(1, keepdims=True)
            u = rng.random((n_sims, n))
            cls = ((u > p3[:, 0]).astype(np.int8)
                   + (u > p3[:, 0] + p3[:, 1]).astype(np.int8))
            m_full = np.clip(d["m_started"].fillna(85.0).to_numpy(), 60, 90)
            full_draw = np.broadcast_to(m_full[None, :], (n_sims, n))
            sub_draw = np.full((n_sims, n), SUB_MINUTES)
            if dispersion:      # a starter does not play the same minutes twice
                full_draw = np.clip(
                    rng.normal(m_full[None, :], MINUTES_SD, (n_sims, n)), 60, 90)
                sub_draw = np.clip(
                    rng.exponential(SUB_MINUTES, (n_sims, n)), 1, 59)
            mins = np.where(cls == 2, full_draw,
                            np.where(cls == 1, sub_draw, 0.0)).astype(np.float32)
            side[team] = dict(d=d, n=n, mins=mins, expo=mins / 90.0)

        for team, opp in ((f["team_h"], f["team_a"]), (f["team_a"], f["team_h"])):
            s = side[team]
            d, expo, mins, n = s["d"], s["expo"], s["mins"], s["n"]
            G, GA = goals[team], goals[opp]

            # the team's simulated goals are shared out over who was on the
            # pitch: the player rates and the team/market rate cannot disagree
            wg = d["xg90"].to_numpy()[None, :] * expo
            wa = d["xa90"].to_numpy()[None, :] * expo
            if dispersion:
                # Who converts the team's chances is itself uncertain: the
                # shrunk per-90 rates are point estimates, and a plug-in
                # simulator that treats them as known comes out underdispersed
                # exactly where a chip decision reads it. Draw the shares from
                # a Dirichlet centred on them.
                #
                # It has to be a real Dirichlet — gamma(k, 1/k) multipliers
                # normalised are NOT one unless every scale matches, and the
                # Jensen gap shrinks the dominant player's share, which is the
                # opposite of the intended effect.
                wg = _dirichlet_like(wg, SHARE_CONCENTRATION, rng)
                wa = _dirichlet_like(wa, SHARE_CONCENTRATION, rng)
            sg, sa = wg.sum(1, keepdims=True), wa.sum(1, keepdims=True)
            # a goal always has a scorer, so with no xG on the pitch it is
            # shared uniformly; an assist is optional, so with no xA on the
            # pitch none is credited rather than one being invented
            cg = np.cumsum(np.where(sg > 0, wg / np.where(sg > 0, sg, 1), 1.0 / n), 1)
            ca = np.cumsum(np.where(sa > 0, wa / np.where(sa > 0, sa, 1), 0.0), 1)
            has_assister = (sa[:, 0] > 0)
            scored = np.zeros((n_sims, n), dtype=np.float32)
            assisted = np.zeros((n_sims, n), dtype=np.float32)
            rows_all = np.arange(n_sims)
            for k in range(int(G.max()) if len(G) else 0):
                live = G > k
                if not live.any():
                    continue
                who = (cg < rng.random((n_sims, 1))).sum(1).clip(0, n - 1)
                np.add.at(scored, (rows_all[live], who[live]), 1.0)
                whoa = (ca < rng.random((n_sims, 1))).sum(1).clip(0, n - 1)
                ok = (live & has_assister & (rng.random(n_sims) < P_ASSISTED)
                      & (whoa != who))
                np.add.at(assisted, (rows_all[ok], whoa[ok]), 1.0)

            played = (mins > 0).astype(np.float32)
            full = (mins >= 60).astype(np.float32)
            cs = full * (GA == 0)[:, None]
            pos = d["position"].fillna("MID").to_numpy()
            gp = np.array([p_goal.get(p, 4) for p in pos], dtype=np.float32)
            cp = np.array([p_cs.get(p, 0) for p in pos], dtype=np.float32)
            gkdef = np.isin(pos, ("GK", "DEF")).astype(np.float32)
            gk = (pos == "GK").astype(np.float32)

            p = (scored * gp[None, :] + assisted * assist_pts + cs * cp[None, :]
                 + played * p_app1 + full * (p_app60 - p_app1))
            p += np.floor((GA[:, None] * expo) / gc_per) * gc_pts * gkdef[None, :]
            # saves rise with the opponent's realised attacking output
            gm = max(1e-6, float(GA.mean()))
            sv_lam = d["saves90"].to_numpy()[None, :] * expo * (
                1.0 + 0.35 * (GA[:, None] - gm) / (gm + 1))
            p += np.floor(rng.poisson(np.clip(sv_lam, 0, 20)) / sv_per) * gk[None, :]
            p += (rng.random((n_sims, n)) < np.clip(
                d["yellow_cards90"].to_numpy()[None, :] * expo, 0, 1)) * yellow
            p += (rng.random((n_sims, n)) < np.clip(
                d["defcon_cross90"].to_numpy()[None, :] * expo, 0, 1)) * dc_pts
            res = d["residual90"].to_numpy()[None, :] * expo
            if dispersion:
                # unmodelled scraps (own goals, penalty saves, reds, oddments)
                # are lumpy events, not a constant trickle
                res = rng.poisson(np.clip(np.abs(res), 0, 5)) * np.sign(res)
            p += res
            s.update(scored=scored, assisted=assisted, cs=cs, played=played, pts=p)

        if not with_bonus:      # isolates the rest of the scoring
            for team in (f["team_h"], f["team_a"]):
                s = side[team]
                cols = np.array([idx[i] for i in s["d"]["player_id"]])
                pts[:, cols] += s["pts"]
            continue

        # bonus: the top three BPS in this simulated match, across both teams
        mats = []
        for team in (f["team_h"], f["team_a"]):
            s = side[team]
            d = s["d"]
            pos = d["position"].fillna("MID").to_numpy()
            c = np.array([bps_coef.get(p, [0, 0, 0, 0, 0]) for p in pos],
                         dtype=np.float32)
            dev = np.array([resid_of.get(i, 0.0) for i in d["player_id"]],
                           dtype=np.float32)
            mu = (c[:, 0] * s["scored"] + c[:, 1] * s["assisted"]
                  + c[:, 2] * s["cs"] + c[:, 3] * s["mins"] + c[:, 4]
                  + dev[None, :] * s["mins"] / 90.0)
            sdv = np.array([max(bps_sd.get(p, 10.0), BPS_SD_FLOOR) for p in pos],
                           dtype=np.float32)
            b = mu + rng.normal(0, 1, mu.shape).astype(np.float32) * sdv[None, :]
            mats.append(np.where(s["played"] > 0, b, -1e9))
        allb = np.concatenate(mats, axis=1)
        order = np.argsort(-allb, axis=1)
        bonus = np.zeros_like(allb)
        rows = np.arange(n_sims)
        for rank, bp in ((0, 3.0), (1, 2.0), (2, 1.0)):
            j = order[:, rank]
            valid = allb[rows, j] > -1e8
            bonus[rows[valid], j[valid]] = bp
        off = 0
        for team in (f["team_h"], f["team_a"]):
            s = side[team]
            cols = np.array([idx[i] for i in s["d"]["player_id"]])
            pts[:, cols] += s["pts"] + bonus[:, off:off + s["n"]]
            off += s["n"]

    return {"players": ids, "points": pts}


def summarise(out: dict) -> pd.DataFrame:
    """Per-player distribution summary: floor, median, ceiling, P(haul)."""
    p = out["points"]
    if not len(out["players"]):
        return pd.DataFrame()
    return pd.DataFrame({
        "player_id": out["players"],
        "mean": p.mean(0),
        "sd": p.std(0),
        "floor": np.percentile(p, 10, axis=0),
        "median": np.median(p, 0),
        "ceiling": np.percentile(p, 90, axis=0),
        "p_blank": (p <= 2).mean(0),
        "p_haul": (p >= 10).mean(0),
    })


def portfolio(out: dict, player_ids) -> dict:
    """Distribution of a squad's TOTAL, with the correlation kept.

    ``independent_sd`` is what you would get by adding variances player by
    player; the gap is the part a per-player model cannot see.
    """
    idx = {p: i for i, p in enumerate(out["players"])}
    cols = [idx[p] for p in player_ids if p in idx]
    if not cols:
        return {}
    tot = out["points"][:, cols].sum(1)
    return {"mean": float(tot.mean()), "sd": float(tot.std()),
            "independent_sd": float(np.sqrt(out["points"][:, cols].var(0).sum())),
            "floor": float(np.percentile(tot, 10)),
            "ceiling": float(np.percentile(tot, 90)),
            "n_players": len(cols)}
