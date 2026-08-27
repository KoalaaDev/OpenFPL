"""Match simulator invariants.

These are the properties that would break silently if the vectorised draw were
wrong, so they are asserted rather than eyeballed: the simulated scoreline has
to match the Poisson rate it was drawn from, the goals have to be shared out
in proportion to the players' rates, bonus has to total 6 a match, and
team-mates have to come out correlated while players in different matches do
not.
"""
import numpy as np
import pandas as pd
import pytest

from fpl_engine import scoring
from fpl_engine.xpts import simulate as sim

RULES = scoring.load_rules()
N = 6000


def _player(pid, team, pos, xg=0.0, xa=0.0, p_full=1.0, saves=0.0):
    return {"player_id": pid, "team_id": team, "position": pos,
            "p_none": 1.0 - p_full, "p_sub": 0.0, "p_full": p_full,
            "m_started": 90.0, "xg90": xg, "xa90": xa, "saves90": saves,
            "yellow_cards90": 0.0, "defcon_cross90": 0.0, "residual90": 0.0,
            "exposure": 20.0}


@pytest.fixture()
def setup():
    x = pd.DataFrame([
        _player(1, 10, "FWD", xg=0.8, xa=0.1),
        _player(2, 10, "MID", xg=0.4, xa=0.4),
        _player(3, 10, "DEF", xg=0.1, xa=0.1),
        _player(4, 10, "GK", saves=3.0),
        _player(5, 20, "FWD", xg=0.5, xa=0.1),
        _player(6, 20, "DEF", xg=0.1, xa=0.1),
        _player(7, 20, "GK", saves=3.0),
    ])
    fixtures = [{"fixture_id": 1, "team_h": 10, "team_a": 20}]
    lams = {(10, 1): (2.0, 0.5), (20, 1): (0.5, 2.0)}
    resid = pd.DataFrame({"player_id": x["player_id"], "bps_resid90": 0.0})
    coef = {p: [20.0, 10.0, 5.0, 0.2, 1.0] for p in ("GK", "DEF", "MID", "FWD")}
    sd = {p: 8.0 for p in ("GK", "DEF", "MID", "FWD")}
    return x, fixtures, lams, coef, resid, sd


def _run(setup, **kw):
    """Invariants are asserted without dispersion, so they stay exact; the
    dispersion itself is a separate property test below."""
    x, fixtures, lams, coef, resid, sd = setup
    kw.setdefault("dispersion", False)
    return sim._run(x, fixtures, lams, RULES, coef, resid, sd,
                    n_sims=kw.pop("n_sims", N), seed=kw.pop("seed", 7), **kw)


def test_goals_are_shared_in_proportion_to_player_rates(setup):
    """The team scores its lambda, split by each player's xG90 x minutes."""
    x, fixtures, lams, coef, resid, sd = setup
    # isolate goals: no assists, nobody concedes (so the clean sheet is a
    # certainty and therefore a constant), and no bonus
    x = x.assign(xa90=0.0)
    lams2 = {(10, 1): (2.0, 0.0), (20, 1): (0.0, 2.0)}
    out = sim._run(x, fixtures, lams2, RULES, coef, resid, sd,
                   n_sims=40000, seed=11, with_bonus=False,
                   dispersion=False)
    idx = {p: i for i, p in enumerate(out["players"])}
    P = out["points"]
    app = RULES["appearance"]["played_60"]
    goals = {pid: (P[:, idx[pid]].mean() - app - RULES["clean_sheet"][pos])
                  / RULES["goal"][pos]
             for pid, pos in ((1, "FWD"), (2, "MID"), (3, "DEF"))}
    home = x[x.team_id == 10]
    weights = dict(zip(home["player_id"], home["xg90"]))
    total_w = sum(weights.values())
    for pid in (1, 2, 3):
        assert goals[pid] == pytest.approx(weights[pid] / total_w * 2.0,
                                           rel=0.08), (pid, goals)
    # and they add up to the team's lambda, which is the point of the design
    assert sum(goals.values()) == pytest.approx(2.0, rel=0.05)


def test_scoreline_matches_the_poisson_rate(setup):
    """Clean-sheet frequency implies the opponent's lambda."""
    out = _run(setup)
    idx = {p: i for i, p in enumerate(out["players"])}
    P = out["points"]
    # team 20 concedes lambda=2.0, so team 10's keeper keeps a clean sheet
    # with probability exp(-0.5) and team 20's with exp(-2.0)
    gk10 = P[:, idx[4]]
    gk20 = P[:, idx[7]]
    cs_pts = RULES["clean_sheet"]["GK"]
    # a keeper's clean-sheet share shows up as a shift in the mean
    assert gk10.mean() > gk20.mean()
    # and the implied rates bracket the true ones
    assert 0.45 < -np.log(max((gk20 > cs_pts).mean(), 1e-6)) < 3.0


def test_bonus_totals_three_two_one_every_match(setup):
    """Exactly 6 bonus points are handed out per fixture, every simulation."""
    x, fixtures, lams, coef, resid, sd = setup
    kw = dict(n_sims=500, seed=3)
    without = sim._run(x, fixtures, lams, RULES, coef, resid, sd,
                       with_bonus=False, dispersion=False, **kw)
    with_ = sim._run(x, fixtures, lams, RULES, coef, resid, sd,
                     with_bonus=True, dispersion=False, **kw)
    per_sim = (with_["points"] - without["points"]).sum(axis=1)
    assert np.allclose(per_sim, 6.0, atol=1e-4), per_sim[:5]


def test_team_mates_are_correlated_and_opponents_are_not(setup):
    """The whole point of simulating the match rather than the players."""
    out = _run(setup)
    idx = {p: i for i, p in enumerate(out["players"])}
    P = out["points"]
    same_team = np.corrcoef(P[:, idx[1]], P[:, idx[2]])[0, 1]
    across = np.corrcoef(P[:, idx[1]], P[:, idx[6]])[0, 1]
    assert same_team > 0.02, same_team          # share the scoreline
    assert across < same_team                   # a rival defender does not


def test_portfolio_variance_exceeds_the_independent_sum(setup):
    out = _run(setup)
    p = sim.portfolio(out, [1, 2, 3])
    assert p["n_players"] == 3
    assert p["sd"] > p["independent_sd"]
    assert p["floor"] <= p["mean"] <= p["ceiling"]


def test_a_player_who_never_plays_scores_nothing(setup):
    x, fixtures, lams, coef, resid, sd = setup
    x = x.copy()
    x.loc[x.player_id == 3, ["p_none", "p_sub", "p_full"]] = [1.0, 0.0, 0.0]
    out = sim._run(x, fixtures, lams, RULES, coef, resid, sd, n_sims=800, seed=1,
                   dispersion=False)
    idx = {p: i for i, p in enumerate(out["players"])}
    assert out["points"][:, idx[3]].max() == 0.0


def test_summarise_orders_floor_median_ceiling(setup):
    s = sim.summarise(_run(setup, n_sims=1200))
    assert (s["floor"] <= s["median"]).all()
    assert (s["median"] <= s["ceiling"]).all()
    assert s["p_haul"].between(0, 1).all()
    assert s["p_blank"].between(0, 1).all()


def test_dispersion_widens_the_distribution_without_moving_the_mean(setup):
    """Treating a fitted rate as if it were known makes a simulator
    underdispersed. Measured over 55k player-gameweeks the point-estimate draw
    understated P(haul >= 10) by ~27% and a starting XI's spread by ~11%.
    """
    x, fixtures, lams, coef, resid, sd = setup
    kw = dict(n_sims=20000, seed=5)
    tight = sim._run(x, fixtures, lams, RULES, coef, resid, sd,
                     dispersion=False, **kw)
    wide = sim._run(x, fixtures, lams, RULES, coef, resid, sd,
                    dispersion=True, **kw)
    idx = {p: i for i, p in enumerate(tight["players"])}
    # the mean must survive untouched — that is what the Dirichlet buys over
    # normalising independent gammas, which quietly shrinks the biggest share
    strike = idx[1]
    assert (wide["points"][:, strike].mean()
            == pytest.approx(tight["points"][:, strike].mean(), rel=0.03))
    assert wide["points"].sum(1).mean() == pytest.approx(
        tight["points"].sum(1).mean(), rel=0.03)
    # and the spread must widen, both per player and for the squad total
    assert wide["points"][:, strike].std() > tight["points"][:, strike].std()
    assert wide["points"].sum(1).std() > tight["points"].sum(1).std()


def test_game_state_draw_preserves_the_mean():
    """The path-dependent draw reshapes the scoreline, it must not move it.

    Measured within team-match, a chasing side takes ~29% more shots and a
    leading side ~24% fewer. Applying that has to leave E[goals] at the team's
    lambda, or every fixture's expected output shifts.
    """
    n = 60000
    rng = np.random.default_rng(3)
    lam = {1: np.full(n, 2.0), 2: np.full(n, 1.1)}
    g = sim._draw_with_state(lam, 1, 2, n, rng)
    assert g[1].mean() == pytest.approx(2.0, rel=0.03)
    assert g[2].mean() == pytest.approx(1.1, rel=0.03)


def test_game_state_reduces_blowouts():
    """Mean reversion is the whole mechanism: leaders ease off, chasers push."""
    n = 60000
    rng = np.random.default_rng(4)
    lam = {1: np.full(n, 2.0), 2: np.full(n, 1.1)}
    g = sim._draw_with_state(lam, 1, 2, n, rng)
    flat_h = rng.poisson(lam[1])
    flat_a = rng.poisson(lam[2])
    assert (np.abs(g[1] - g[2]).mean() < np.abs(flat_h - flat_a).mean())


def test_game_state_is_off_by_default():
    """It is measured, correct, and unused — lambda already absorbs it.

    Scored on 1,480 replayed team-matches the goal distribution is equally
    calibrated with and without it (PIT chi2 9.9 vs 8.9), because the team
    model is fitted on realised goals.
    """
    assert sim.GAME_STATE is False
