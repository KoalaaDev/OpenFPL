"""Turning £m into points, at a measured rate rather than an invented one.

The exchange rate is the marginal value of the budget constraint in the
squad-selection problem: rebuild the best legal 15 with a little more money and
see what it buys. Measured across 18 replayed gameweeks it is 0.163 points per
£1m per gameweek at the margin, falling to 0.093 by +£4m.

The point of these tests is the *size*. A price rise is realised only on sale,
and FPL hands back half the profit, so catching a 0.1 rise with a full season
left is worth about 0.2 points — a tie-breaker, not a driver.
"""
import pytest

from fpl_engine import price_model as pm


def test_no_season_left_means_no_value():
    assert pm.points_value(0.1, 0) == 0.0
    assert pm.points_value(0.1, None) == 0.0
    assert pm.points_value(-0.1, 0) == 0.0


def test_the_sell_on_haircut_is_applied():
    """FPL returns the purchase price plus HALF the profit."""
    gross = pm.POINTS_PER_MILLION_PER_GW * 10
    assert pm.points_value(1.0, 10) == pytest.approx(gross * pm.SELL_ON_SHARE)


def test_value_scales_with_the_move_and_the_season_left():
    assert pm.points_value(0.2, 10) == pytest.approx(2 * pm.points_value(0.1, 10))
    assert pm.points_value(0.1, 20) == pytest.approx(2 * pm.points_value(0.1, 10))


def test_a_faller_is_worth_negative_points():
    assert pm.points_value(-0.08, 30) < 0
    assert pm.points_value(-0.08, 30) == pytest.approx(-pm.points_value(0.08, 30))


def test_a_price_rise_is_a_tie_breaker_not_a_driver():
    """The whole reason it is kept out of the solver's objective.

    The strongest signal the model produces is about a 0.076 expected move.
    Even with a full season to run that has to stay small enough that it can
    only separate transfers already rated equally.
    """
    best_case = pm.points_value(0.076, 37)
    assert 0.1 < best_case < 0.35
    # and by the run-in it is negligible
    assert pm.points_value(0.076, 5) < 0.05


def test_the_rate_is_the_measured_one():
    assert pm.POINTS_PER_MILLION_PER_GW == pytest.approx(0.163)
    assert pm.SELL_ON_SHARE == 0.5
