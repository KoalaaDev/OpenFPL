"""Forward-in-time A/B of the decision layer: MFRU vs the current max-xP rule.

The projection backtest (``backtest.py``) scores *predictions*. This scores
*decisions*: hold the squad fixed to the ownership template — the same
controlled design as the captaincy-vs-crowd study — and let each rule choose
XI, captain, vice and bench order from that identical 15. Every arm therefore
sees the same players, the same fixtures and the same information; the only
difference is the objective. Realised scoring applies FPL's real automatic
substitutions and armband transfer to what actually happened.

Arms per gameweek (all strictly point-in-time; ownership lagged one gw):
  crowd     the template's own XI and captain, by effective ownership
  xp        the CURRENT rule: max analytic-mean XI, best-mean captain,
            bench ordered by mean — what ranking by expected points does
  mfru_g0   MFRU at gamma=0: risk-neutral, but autosub- and armband-aware
            inside the joint draws (the only channels where a risk-neutral
            decision can legally differ from max-xP)
  mfru_g±x  MFRU with the EO risk price gamma on sd(Delta): positive chases
            rank variance (differentials), negative shadows the template

Per gw, per arm: realised points (autosubs + armband applied to actual
minutes) and realised Delta against the ownership mean-field. ``compare``
runs paired t-tests over the pooled gameweeks — the standing rule applies:
p > 0.05 is unproven, whatever the means look like.
"""
from __future__ import annotations

import json
import os
from itertools import permutations

import numpy as np
import pandas as pd

from . import config, db, progress, scoring
from .backtest import _actuals
from .xpts import engine as xpts_engine, minutes_model, simulate
from .xpts.rank_utility import (MeanFieldRankUtility, autosub_points,
                                effective_ownership, legal_xis, POS_MIN,
                                SQUAD_SHAPE)

GAMMAS = (-1.0, -0.5, -0.3, -0.15, 0.15, 0.3, 0.5, 1.0)
OBJECTIVES = ("p_beat", "cvar20", "q80")   # rank functionals beyond mean±sd
N_SIMS = 3000
MAX_PER_CLUB = 3
N_RANDOM = 20               # random-valid decisions averaged per gameweek
PREMIUM_TENTHS = 90         # the human captain bias: £9.0m+ is "a premium"


# ---------------------------------------------------------------- template
def template_squad(players: pd.DataFrame) -> list[int]:
    """The crowd's 15: most-owned legal 2/5/5/3 with <=3 per club.

    ``players`` needs player_id, position, team_id, selected and must already
    be restricted to players simulated this gameweek (a blanked player cannot
    be scored, so no arm may hold him — applied identically to every arm).
    """
    quota = dict(SQUAD_SHAPE)
    club: dict[int, int] = {}
    out: list[int] = []
    for r in players.sort_values("selected", ascending=False).itertuples():
        if quota.get(r.position, 0) <= 0:
            continue
        if club.get(r.team_id, 0) >= MAX_PER_CLUB:
            continue
        quota[r.position] -= 1
        club[r.team_id] = club.get(r.team_id, 0) + 1
        out.append(int(r.player_id))
        if len(out) == 15:
            break
    if len(out) < 15:
        raise ValueError("could not build a legal template squad")
    return out


def _crowd_decision(squad: list[int], meta: pd.DataFrame) -> dict:
    """The template's own XI/captain: most-owned legal XI, most-owned captain."""
    m = meta.set_index("player_id").loc[squad]
    by_pos = {p: [pid for pid in squad if m.at[pid, "position"] == p]
              for p in POS_MIN}
    for p in by_pos:      # already ownership-ordered via squad order
        by_pos[p].sort(key=lambda pid: -m.at[pid, "selected"])
    xi = [by_pos["GK"][0]]
    for p in ("DEF", "MID", "FWD"):
        xi += by_pos[p][:POS_MIN[p]]
    rest = [pid for pid in squad
            if pid not in xi and m.at[pid, "position"] != "GK"]
    rest.sort(key=lambda pid: -m.at[pid, "selected"])
    for pid in rest:
        if len(xi) == 11:
            break
        xi.append(pid)
    cap = max(xi, key=lambda pid: m.at[pid, "selected"])
    vice = max((pid for pid in xi if pid != cap),
               key=lambda pid: m.at[pid, "selected"])
    bench = [by_pos["GK"][1]] + [pid for pid in rest if pid not in xi]
    return {"xi": xi, "captain": cap, "vice": vice, "bench": bench}


def _mean_decision(squad: list[int], mean_of: dict[int, float],
                   positions_of: dict[int, str]) -> dict:
    """What ranking by expected points does: everything by the mean.

    Serves two arms: ``xp`` (the analytic engine's prediction, i.e. the
    current model verbatim) and ``xp_sim`` (the simulator's own mean, which
    controls for the estimator so that mfru-vs-xp_sim isolates the objective).
    """
    mean = np.array([float(mean_of.get(p, 0.0)) for p in squad])
    positions = [positions_of.get(p, "MID") for p in squad]
    by_pos = {p: [i for i, q in enumerate(positions) if q == p]
              for p in POS_MIN}
    best, best_xi = -np.inf, None
    for xi in legal_xis(by_pos):
        tot = float(mean[list(xi)].sum())
        if tot > best:
            best, best_xi = tot, list(xi)
    order = sorted(best_xi, key=lambda i: -mean[i])
    cap, vice = order[0], order[1]
    bench = [i for i in range(15) if i not in best_xi]
    bench_gk = [i for i in bench if positions[i] == "GK"]
    outfield = sorted((i for i in bench if positions[i] != "GK"),
                      key=lambda i: -mean[i])
    return {"xi": [squad[i] for i in best_xi],
            "captain": squad[cap], "vice": squad[vice],
            "bench": [squad[i] for i in bench_gk + outfield]}


_SITE_EP_CACHE: dict[str, dict[tuple[int, int], float]] = {}


def _site_ep(season: str) -> dict[tuple[int, int], float]:
    """FPL's own published expected points (ep_this), as archived per
    gameweek by the vaastav dataset's ``xP`` column.

    **Measured as leaky — excluded from the default arms.** The hope was a
    season-long public projection benchmark; the measurement killed it. In
    2023-24 / 2024-25 the archived column reaches spearman_played 0.58 /
    0.51 — the level of this repo's *perfect-minutes oracle* (~0.59, see
    CLAUDE.md) — and assigns xP <= 0.05 to players who then played 0 minutes
    with 89-94% precision. No deadline-honest projection can know who will
    not play that well; the archive embeds realised minutes (its capture
    time is the archiver's, after the fact). In 2025-26 the same column is
    near-random among players who played (spearman_played 0.07). Either way
    it cannot benchmark a deadline decision. Kept only so the leak test in
    the research log stays reproducible; do not re-add it to the arms
    without new provenance. Returns {} when unreachable.
    """
    if season not in _SITE_EP_CACHE:
        from .ingest.vaastav import RAW, _csv_rows
        try:
            rows = _csv_rows(f"{RAW}/{season}/gws/merged_gw.csv", True)
        except Exception:      # noqa: BLE001 - benchmark arm, never a gate
            rows = []
        out: dict[tuple[int, int], float] = {}
        for m in rows:
            try:
                key = (int(m["GW"]), int(m["element"]))
                out[key] = out.get(key, 0.0) + float(m.get("xP") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
        _SITE_EP_CACHE[season] = out
    return _SITE_EP_CACHE[season]


def _human_scores(conn, season: str, gw: int, squad: list[int],
                  fixtures: list) -> tuple[dict[int, float], dict[int, float]]:
    """The competent-human signal: form x nailedness + fixture ease + home.

    Deliberately built only from what a person reads off public sites at the
    deadline — trailing points, who has been starting, the opponent's
    leakiness (their FDR proxy), venue — and NOT from the simulator, the
    minutes model or the rates. Returns (score, price) per player.
    """
    hist = pd.read_sql_query(
        "SELECT player_id, gw, minutes, total_points, starts, price "
        "FROM player_gw WHERE season=? AND gw<? AND gw>=?",
        conn, params=(season, gw, gw - 5))
    conceded = pd.read_sql_query(
        "SELECT team_id, AVG(goals_against) c FROM (SELECT team_id, "
        "goals_against FROM team_match WHERE season=? AND gw<? "
        "ORDER BY gw DESC) GROUP BY team_id", conn, params=(season, gw))
    conc_of = dict(zip(conceded["team_id"], conceded["c"]))
    league_c = float(np.mean(list(conc_of.values()))) if conc_of else 1.3

    opp_ease: dict[int, float] = {}
    home_of: dict[int, float] = {}
    for f in fixtures:
        for team, opp, home in ((f["team_h"], f["team_a"], 1.0),
                                (f["team_a"], f["team_h"], 0.0)):
            opp_ease[team] = opp_ease.get(team, 0.0) + (
                conc_of.get(opp, league_c) - league_c)
            home_of[team] = max(home_of.get(team, 0.0), home)
    team_of = {r["player_id"]: r["team_id"] for r in conn.execute(
        "SELECT player_id, team_id FROM player WHERE season=?", (season,))}

    scores, prices = {}, {}
    by_p = hist.groupby("player_id")
    for pid in squad:
        try:
            h = by_p.get_group(pid)
        except KeyError:
            scores[pid], prices[pid] = 0.0, 0.0
            continue
        played = h[h["minutes"] > 0]
        form = float(played["total_points"].mean()) if len(played) else 0.0
        n_gws = max(1, h["gw"].nunique())
        nailed = min(1.0, float(h["starts"].fillna(
            (h["minutes"] > 60)).sum()) / n_gws)
        t = team_of.get(pid)
        scores[pid] = (form * (0.4 + 0.6 * nailed)
                       + 0.8 * opp_ease.get(t, 0.0)
                       + 0.2 * home_of.get(t, 0.0))
        prices[pid] = float(h.sort_values("gw")["price"].iloc[-1] or 0.0)
    return scores, prices


def _human_decision(conn, season: str, gw: int, squad: list[int],
                    fixtures: list,
                    positions_of: dict[int, str]) -> dict:
    """The competent-human arm: heuristic XI, obvious-premium captain."""
    scores, prices = _human_scores(conn, season, gw, squad, fixtures)
    d = _mean_decision(squad, scores, positions_of)
    premiums = [p for p in d["xi"] if prices.get(p, 0.0) >= PREMIUM_TENTHS]
    pool = premiums or d["xi"]
    cap = max(pool, key=lambda p: scores.get(p, 0.0))
    vice = max((p for p in d["xi"] if p != cap),
               key=lambda p: scores.get(p, 0.0))
    return {**d, "captain": cap, "vice": vice}


def _random_decisions(squad: list[int], positions_of: dict[int, str],
                      seed: int, n: int = N_RANDOM) -> list[dict]:
    """Uniform random legal decisions — the floor every model must clear."""
    rng = np.random.default_rng(seed)
    positions = [positions_of.get(p, "MID") for p in squad]
    by_pos = {p: [i for i, q in enumerate(positions) if q == p]
              for p in POS_MIN}
    xis = legal_xis(by_pos)
    out = []
    for _ in range(n):
        xi = list(xis[rng.integers(len(xis))])
        cap, vice = rng.choice(xi, size=2, replace=False)
        bench = [i for i in range(15) if i not in xi]
        bench_gk = [i for i in bench if positions[i] == "GK"]
        outfield = [i for i in bench if positions[i] != "GK"]
        rng.shuffle(outfield)
        out.append({"xi": [squad[i] for i in xi],
                    "captain": squad[int(cap)], "vice": squad[int(vice)],
                    "bench": [squad[i] for i in bench_gk + outfield]})
    return out


def _rebench(decision: dict, model: MeanFieldRankUtility) -> dict:
    """Keep the XI and captain, re-order the bench by rank utility.

    Isolates the bench-order/autosub channel from the XI and captain
    channels: if mfru_g0's edge is real, this shows which lever carries it.
    """
    squad = decision["xi"] + decision["bench"]
    cols = [model.col[p] for p in squad]
    sub = model.draws[:, cols]
    played = model.played[:, cols]
    positions = [model.position[c] for c in cols]
    xi_idx = list(range(11))
    cap = squad.index(decision["captain"])
    vice = squad.index(decision["vice"])
    bench_gk = [i for i in range(11, 15) if positions[i] == "GK"]
    outfield = [i for i in range(11, 15) if positions[i] != "GK"]
    best, best_border = -np.inf, None
    for perm in permutations(outfield):
        border = bench_gk + list(perm)
        s = autosub_points(sub, played, positions, xi_idx, cap, vice, border)
        u = model.utility(s)
        if u > best:
            best, best_border = u, border
    return {**decision, "bench": [squad[i] for i in best_border]}


# ---------------------------------------------------------------- realised
def realised_score(decision: dict, actual: pd.DataFrame,
                   positions_of: dict[int, str]) -> float:
    """Actual FPL points of a decision: real minutes, real autosubs."""
    squad = decision["xi"] + decision["bench"]
    a = actual.set_index("player_id")
    pts = np.array([[float(a["pts"].get(p, 0.0)) for p in squad]])
    played = np.array([[float(a["mins"].get(p, 0.0)) > 0 for p in squad]])
    positions = [positions_of.get(p, "MID") for p in squad]
    xi_idx = list(range(11))
    bench_idx = list(range(11, 15))
    cap = squad.index(decision["captain"])
    vice = squad.index(decision["vice"])
    s = autosub_points(pts, played, positions, xi_idx, cap, vice, bench_idx)
    return float(s[0])


# --------------------------------------------------------------------- run
def run(conn, season: str = "2025-26", *, gws: list[int] | None = None,
        n_sims: int = N_SIMS, gammas: tuple = GAMMAS) -> dict:
    tag = f"bt{season}"
    if minutes_model.load(tag)[0] is None:
        train_seasons = [s for s in config.BACKFILL_SEASONS if s < season]
        progress.step(f"Training minutes model on {train_seasons}…")
        minutes_model.train(conn, seasons=train_seasons, tag=tag)
    bundle = minutes_model.load(tag)
    rules = scoring.load_rules()

    actual = _actuals(conn, season)
    all_gws = sorted(int(g) for g in actual["gw"].dropna().unique())
    gws = [int(g) for g in gws] if gws else [g for g in all_gws if g >= 2]

    players_all = pd.read_sql_query(
        "SELECT player_id, position, team_id FROM player WHERE season=? "
        "AND position IN ('GK','DEF','MID','FWD')", conn, params=(season,))
    positions_of = dict(zip(players_all["player_id"], players_all["position"]))

    per_gw: dict[str, dict[int, dict]] = {}
    site_ep: dict = {}      # measured leaky — see _site_ep; not an arm
    arms = (["random", "human", "crowd", "xp", "xp_sim",
             "xp_bench", "mfru_g0"]
            + [f"mfru_g{g:+g}" for g in gammas]
            + [f"mfru_{o}" for o in OBJECTIVES])
    squads: dict[int, list[int]] = {}
    for g in gws:
        progress.step(f"GW{g}…")
        as_of = xpts_engine.first_kickoff(conn, season, g)
        if as_of is None:
            continue
        # ownership lagged one gameweek: the newest `selected` strictly
        # before this gw (a manager sees last week's ownership at the deadline)
        own = pd.read_sql_query(
            "SELECT player_id, selected FROM player_gw WHERE season=? AND "
            "gw=(SELECT MAX(gw) FROM player_gw WHERE season=? AND gw<? "
            "    AND selected IS NOT NULL) GROUP BY player_id",
            conn, params=(season, season, g))
        if own.empty or own["selected"].fillna(0).sum() <= 0:
            continue
        sim = simulate.simulate_gw(conn, season, g, as_of=as_of,
                                   n_sims=n_sims, use_availability=False,
                                   minutes_bundle=bundle, rules=rules)
        if not len(sim["players"]):
            continue
        meta = (players_all.merge(own, on="player_id", how="left")
                .fillna({"selected": 0.0}))
        meta = meta[meta["player_id"].isin(set(sim["players"]))]
        try:
            squad = template_squad(meta)
        except ValueError:
            continue
        eo = effective_ownership(
            meta.set_index("player_id")["selected"].astype(float))

        sim_players = pd.DataFrame({"player_id": sim["players"]}).merge(
            players_all, on="player_id", how="left")
        act_g = actual[actual["gw"] == g][["player_id", "pts", "mins"]]
        field_actual = float(sum(
            eo.get(p, 0.0) * float(pt) for p, pt in
            zip(act_g["player_id"], act_g["pts"])))

        decisions = {}
        m0 = MeanFieldRankUtility(sim["points"], sim["mins"], sim_players,
                                  eo, gamma=0.0)
        decisions["crowd"] = _crowd_decision(squad, meta)
        # the current model verbatim: the analytic engine's expected points
        xp = xpts_engine.xpts_predict_gw(conn, season, g, as_of=as_of,
                                         use_availability=False,
                                         minutes_bundle=bundle, rules=rules)
        decisions["xp"] = _mean_decision(
            squad, dict(zip(xp["player_id"], xp["prediction"])), positions_of)
        # same rule on the simulator's own mean: controls for the estimator,
        # so mfru-vs-xp_sim is purely the objective
        sim_mean = dict(zip(sim["players"],
                            np.asarray(sim["points"]).mean(axis=0)))
        decisions["xp_sim"] = _mean_decision(squad, sim_mean, positions_of)
        decisions["xp_bench"] = _rebench(decisions["xp"], m0)
        decisions["mfru_g0"] = m0.decide(squad)
        for gam in gammas:
            mg = MeanFieldRankUtility(sim["points"], sim["mins"], sim_players,
                                      eo, gamma=gam)
            decisions[f"mfru_g{gam:+g}"] = mg.decide(squad)
        for obj in OBJECTIVES:
            mo = MeanFieldRankUtility(sim["points"], sim["mins"], sim_players,
                                      eo, objective=obj)
            decisions[f"mfru_{obj}"] = mo.decide(squad)
        fixtures = xpts_engine._gw_fixtures(conn, season, g)
        decisions["human"] = _human_decision(conn, season, g, squad, fixtures,
                                             positions_of)
        if site_ep:
            decisions["site_ep"] = _mean_decision(
                squad, {p: site_ep.get((g, p), 0.0) for p in squad},
                positions_of)
        squads[g] = squad

        for arm in arms:
            if arm == "random":
                rand = _random_decisions(squad, positions_of, seed=1000 + g)
                pts = float(np.mean([realised_score(r, act_g, positions_of)
                                     for r in rand]))
                per_gw.setdefault(arm, {})[g] = {
                    "pts": pts, "delta": pts - field_actual, "captain": -1}
                continue
            d = decisions[arm]
            pts = realised_score(d, act_g, positions_of)
            rec = {"pts": pts, "delta": pts - field_actual,
                   "captain": int(d["captain"])}
            if "e_points" in d:
                rec.update(e_points=d["e_points"], sd_delta=d["sd_delta"],
                           p_beat_field=d["p_beat_field"])
            per_gw.setdefault(arm, {})[g] = rec

    summary = {arm: {k: round(float(np.mean([r[k] for r in res.values()])), 3)
                     for k in ("pts", "delta")} | {"gws": len(res)}
               for arm, res in per_gw.items()}
    report = {"season": season, "n_sims": n_sims, "summary": summary,
              "squads": {str(g): s for g, s in squads.items()},
              "per_gw": {a: {str(g): v for g, v in res.items()}
                         for a, res in per_gw.items()}}
    out_path = os.path.join(config.DATA_DIR, f"rank_backtest_{season}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


# ----------------------------------------------------------------- compare
def compare(paths: list[str], *, baseline: str = "xp") -> dict:
    """Pool seasons and paired-t every arm against the baseline rule."""
    from scipy.stats import ttest_rel

    per: dict[str, dict[str, dict]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        for arm, res in r["per_gw"].items():
            for g, v in res.items():
                per.setdefault(arm, {})[f"{r['season']}:{g}"] = v
    if baseline not in per:
        raise ValueError(f"baseline arm {baseline!r} not in reports")
    n_arms = len(per) - 1
    out = {"baseline": baseline, "n_comparisons": n_arms,
           "bonferroni_alpha": round(0.05 / max(1, n_arms), 4), "arms": {}}
    for arm, res in per.items():
        if arm == baseline:
            continue
        keys = sorted(set(res) & set(per[baseline]))
        stats = {}
        for k in ("pts", "delta"):
            x = np.array([per[baseline][q][k] for q in keys], float)
            y = np.array([res[q][k] for q in keys], float)
            d = y - x
            t, p = ttest_rel(y, x)
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
            by_season: dict[str, float] = {}
            for q, dv in zip(keys, d):
                s = q.split(":")[0]
                by_season.setdefault(s, []).append(dv)
            stats[k] = {
                "baseline": round(float(x.mean()), 3),
                "arm": round(float(y.mean()), 3),
                "delta": round(float(d.mean()), 3),
                "median": round(float(np.median(d)), 3),
                "sd": round(float(d.std(ddof=1)), 3),
                "ci95": [round(float(d.mean() - 1.96 * se), 3),
                         round(float(d.mean() + 1.96 * se), 3)],
                "t": round(float(t), 3), "p": round(float(p), 4),
                "per_season": {s: round(float(np.mean(v)), 3)
                               for s, v in sorted(by_season.items())},
            }
        # rank proxy: how often each arm's realised score beat the field
        beat = float(np.mean([res[q]["delta"] > 0 for q in keys]))
        beat_base = float(np.mean(
            [per[baseline][q]["delta"] > 0 for q in keys]))
        same_cap = float(np.mean(
            [res[q]["captain"] == per[baseline][q]["captain"] for q in keys]))
        out["arms"][arm] = {"gws": len(keys), "same_captain": round(same_cap, 3),
                            "beat_field": round(beat, 3),
                            "beat_field_baseline": round(beat_base, 3),
                            **stats}
    return out
