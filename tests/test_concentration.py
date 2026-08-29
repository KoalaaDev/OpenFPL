"""Concentration: the one nonlinearity a rank-aware objective can actually use.

Maximising expected margin over the crowd is algebraically identical to
maximising expected points - the ownership term does not depend on what you
pick, so it drops out of the argmax. Every differential-aware objective that is
linear in ownership therefore changes nothing at all. What is left is that
players from the same club share a fixture and so are correlated, which the
simulator measured: independence understates a one-club triple-up's spread by
6% and a whole XI's by 8%.

The trap these tests exist for: a one-sided `z >= n - 1` prices a penalty
correctly but leaves the objective UNBOUNDED the moment the weight goes
negative, because nothing stops z running away. The solver returns garbage
rather than an error, so the failure is silent - which is exactly the class of
bug this repo keeps finding.
"""
import pandas as pd
import pytest

from fpl_engine.optimise import chips


def _proj(n_clubs=8, per_club=8):
    """A pool rich enough to field a legal squad, with one club's players
    deliberately the best so an unconstrained solver wants to stack them."""
    rows, pid = [], 1
    for c in range(n_clubs):
        for i in range(per_club):
            pos = ["GK", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i]
            rows.append({
                "player_id": pid, "player": f"p{pid}", "position": pos,
                "team_id": c, "price": 5.0,
                # club 0 is uniformly the strongest, so concentration is the
                # unconstrained optimum and the weight has something to push on
                "ep_gw1": (6.0 if c == 0 else 4.0) + 0.01 * i,
            })
            pid += 1
    return pd.DataFrame(rows)


def _max_club(plan):
    xi = [r for r in plan.per_gw[0]["squad"] if r["in_xi"]]
    counts = {}
    for r in xi:
        counts[r["team_id"]] = counts.get(r["team_id"], 0) + 1
    return max(counts.values())


def _solve(conc):
    plans = chips.optimise_with_chips(_proj(), [1], initial=None, budget=100.0,
                                      chip_gws={}, time_limit=20,
                                      concentration=conc)
    assert plans, f"no plan at concentration={conc}"
    return plans[0]


def test_zero_is_an_exact_no_op():
    """The default must not perturb a single decision."""
    a, b = _solve(0.0), _solve(0.0)
    assert a.objective == pytest.approx(b.objective)
    off = chips.optimise_with_chips(_proj(), [1], initial=None, budget=100.0,
                                    chip_gws={}, time_limit=20)[0]
    assert a.objective == pytest.approx(off.objective)
    assert [r["player_id"] for r in a.per_gw[0]["squad"]] == \
           [r["player_id"] for r in off.per_gw[0]["squad"]]


def test_a_negative_weight_does_not_run_away():
    """Rewarding concentration must stay bounded.

    With a one-sided formulation this returns an unbounded objective and the
    plan is meaningless - and nothing raises.
    """
    plan = _solve(-1.0)
    assert plan.objective < 1e6
    assert _max_club(plan) <= chips.MAX_PER_CLUB


def test_the_excess_is_counted_exactly():
    """Reward and penalty must move the XI in opposite directions."""
    spread = _max_club(_solve(2.0))
    neutral = _max_club(_solve(0.0))
    stack = _max_club(_solve(-2.0))
    assert spread <= neutral <= stack
    assert stack >= 2, "rewarding correlation should stack at least a pair"


def test_a_penalty_never_breaks_the_squad_rules():
    for conc in (-2.0, -0.5, 0.5, 2.0):
        plan = _solve(conc)
        squad = plan.per_gw[0]["squad"]
        assert len(squad) == 15
        assert sum(1 for r in squad if r["in_xi"]) == 11
        assert _max_club(plan) <= chips.MAX_PER_CLUB
