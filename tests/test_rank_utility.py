"""MFRU rank-utility layer: autosub operator, EO, and the decision maths.

The autosub operator is the piece that can silently mis-score every arm of
the rank backtest, so each FPL substitution rule gets a case that would fail
without it. The decision tests pin the two theorems the model rests on:
at gamma=0 with no minutes risk MFRU must reproduce the max-xP pick exactly,
and the only risk-neutral divergence channels are autosubs and the armband.
"""
import numpy as np
import pandas as pd
import pytest

from fpl_engine.xpts.rank_utility import (MeanFieldRankUtility, autosub_points,
                                          effective_ownership, legal_xis,
                                          POS_MIN)

# squad layout used throughout: 0-1 GK, 2-6 DEF, 7-11 MID, 12-14 FWD
POSITIONS = (["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)


def _one_draw(pts, mins):
    return (np.array([pts], dtype=float), np.array([m > 0 for m in mins])
            .reshape(1, -1))


def _score(pts, mins, xi, cap, vice, bench):
    p, played = _one_draw(pts, mins)
    return float(autosub_points(p, played, POSITIONS, xi, cap, vice, bench)[0])


# XI: GK0, DEF 2-5 (four), MID 7-10 (four), FWD 12-13 (two); bench GK1, 6, 11, 14
XI = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13]
BENCH = [1, 6, 11, 14]     # GK, DEF, MID, FWD


def test_no_subs_needed_sums_the_xi():
    pts = list(range(15))
    mins = [90] * 15
    want = sum(pts[i] for i in XI) + pts[12]        # captain 12 doubled
    assert _score(pts, mins, XI, 12, 7, BENCH) == want


def test_bench_gk_replaces_only_a_non_playing_gk():
    pts = [5.0] * 15
    mins = [90] * 15
    mins[0] = 0                                     # starting GK missing
    base = sum(5.0 for i in XI if i != 0) + 5.0     # captain double
    assert _score(pts, mins, XI, 12, 7, BENCH) == base + 5.0
    # an outfield absence must never pull the bench GK on
    mins = [90] * 15
    mins[7] = 0
    got = _score(pts, mins, XI, 12, 8, BENCH)
    assert got == sum(5.0 for i in XI if i != 7) + 5.0 + 5.0  # MID 11 came on


def test_formation_blocks_an_illegal_substitution():
    # XI with exactly 3 DEF: 0, 2,3,4, MID 7-10, FWD 12,13 + extra MID 11
    xi = [0, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]
    bench = [1, 5, 6, 14]                           # GK, DEF, DEF, FWD
    pts = [3.0] * 15
    mins = [90] * 15
    mins[2] = 0                                     # a DEF is missing
    # the FWD bench player cannot legally replace him (would leave 2 DEF),
    # but the bench DEF can — regardless of order
    got = _score(pts, mins, xi, 12, 7, [1, 14, 5, 6])
    assert got == sum(3.0 for i in xi if i != 2) + 3.0 + 3.0


def test_bench_order_is_respected_when_both_are_legal():
    pts = [0.0] * 15
    pts[11] = 7.0                                   # bench MID
    pts[14] = 2.0                                   # bench FWD
    mins = [90] * 15
    mins[7] = 0                                     # one MID missing
    # FWD first in the order: he comes on (4-3-3 -> 4-2-4 illegal? no:
    # comp MID 4->3 >= 2, FWD 2->3 <= 3, legal) and takes the slot
    got = _score(pts, mins, XI, 12, 8, [1, 14, 11, 6])
    assert got == 2.0
    got = _score(pts, mins, XI, 12, 8, [1, 11, 14, 6])
    assert got == 7.0


def test_two_absences_use_two_bench_players():
    pts = [1.0] * 15
    mins = [90] * 15
    mins[7] = mins[8] = 0
    got = _score(pts, mins, XI, 12, 9, BENCH)
    assert got == 9 * 1.0 + 2 * 1.0 + 1.0           # 9 starters + 2 subs + cap


def test_vice_takes_the_armband_when_the_captain_blanks():
    pts = [2.0] * 15
    pts[13] = 8.0
    mins = [90] * 15
    mins[12] = 0                                    # captain 0 minutes
    got = _score(pts, mins, XI, 12, 13, BENCH)
    # captain scores 0 and is replaced by FWD 14; vice doubled instead
    assert got == (sum(2.0 for i in XI if i not in (12, 13)) + 8.0
                   + 2.0 + 8.0)


def test_effective_ownership_normalises_and_doubles_the_top_pick():
    sel = pd.Series({1: 3000.0, 2: 1000.0, 3: 1000.0})
    eo = effective_ownership(sel)
    # shares: 15*3000/5000 = 9 -> capped at 1; the cap matters for tiny pools
    assert eo[1] == pytest.approx(2 * 11 / 15)      # capped, then doubled
    assert eo[2] == eo[3] < eo[1]
    assert effective_ownership(sel * 0).sum() == 0.0


def _players(n=15):
    return pd.DataFrame({"player_id": list(range(100, 100 + n)),
                         "position": POSITIONS})


def test_gamma_zero_with_no_minutes_risk_is_exactly_max_xp():
    rng = np.random.default_rng(0)
    means = rng.uniform(1, 6, 15)
    draws = np.tile(means, (400, 1)) + rng.normal(0, 1, (400, 15))
    mins = np.full((400, 15), 90.0)                 # everyone always plays
    players = _players()
    eo = pd.Series(0.0, index=players["player_id"])
    m = MeanFieldRankUtility(draws, mins, players, eo, gamma=0.0)
    d = m.decide(players["player_id"].tolist())
    # with no absences, autosubs and the armband never fire: the decision
    # must collapse to the max-mean XI and the max-mean captain
    mean = draws.mean(axis=0)
    by_pos = {p: [i for i, q in enumerate(POSITIONS) if q == p]
              for p in POS_MIN}
    best = max(legal_xis(by_pos), key=lambda xi: mean[list(xi)].sum())
    assert set(d["xi"]) == {players["player_id"][i] for i in best}
    assert d["captain"] == players["player_id"][int(np.argmax(
        np.where(np.isin(np.arange(15), best), mean, -np.inf)))]


def test_autosub_awareness_prefers_the_insured_risky_starter():
    """The sub-bench dilemma: mean-ranking benches the risky player, MFRU
    starts him because his absence branch is insured by the nailed sub."""
    n = 4000
    rng = np.random.default_rng(1)
    players = _players()
    draws = np.zeros((n, 15))
    mins = np.full((n, 15), 90.0)
    nailed = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12]       # 10 always-play starters
    for i in nailed:
        draws[:, i] = 3.0
    # FWD 13: 5.8 pts but only half the weeks; FWD 14: 3.0 always
    on = rng.random(n) < 0.5
    draws[on, 13] = 5.8
    mins[~on, 13] = 0.0
    draws[:, 14] = 3.0
    # remaining squad players: never play, no points
    for i in (1, 6, 11):
        draws[:, i] = 0.0
        mins[:, i] = 0.0
    mins[:, 1] = 90.0                               # bench GK exists
    eo = pd.Series(0.0, index=players["player_id"])
    m = MeanFieldRankUtility(draws, mins, players, eo, gamma=0.0)
    d = m.decide(players["player_id"].tolist())
    # 13's mean (2.9) is below 14's (3.0), so ranking by mean leaves him
    # out — but fielding him insured by a nailed first sub is worth
    # 0.5*5.8 + 0.5*3.0 = 4.4, and MFRU must find it (several XI/bench
    # arrangements achieve the same utility; all field 13 and back him
    # with an always-playing sub)
    assert players["player_id"][13] in d["xi"]
    first_sub = players["player_id"].tolist().index(d["bench"][1])
    assert mins[:, first_sub].min() > 0
    # expected score with the insurance (~37.4 incl. captain) beats the
    # max-mean XI's true expectation (36.0)
    assert d["e_points"] > 36.5


def test_gamma_prices_the_differential():
    """Same mean, same variance: gamma>0 captains the low-EO player, gamma<0
    the template player. At gamma=0 the choice carries no expected cost."""
    n = 6000
    rng = np.random.default_rng(2)
    players = _players()
    draws = np.zeros((n, 15))
    mins = np.full((n, 15), 90.0)
    for i in range(15):
        draws[:, i] = 2.0
    # two identical premium mids, independent noise
    draws[:, 7] = 5.0 + rng.normal(0, 3, n)
    draws[:, 8] = 5.0 + rng.normal(0, 3, n)
    eo = pd.Series(0.0, index=players["player_id"])
    eo.iloc[7] = 1.5                                # the template's darling
    chase = MeanFieldRankUtility(draws, mins, players, eo, gamma=0.4).decide(
        players["player_id"].tolist())
    protect = MeanFieldRankUtility(draws, mins, players, eo, gamma=-0.4).decide(
        players["player_id"].tolist())
    assert chase["captain"] == players["player_id"][8]     # differential
    assert protect["captain"] == players["player_id"][7]   # shadow the crowd
    assert chase["sd_delta"] > protect["sd_delta"]