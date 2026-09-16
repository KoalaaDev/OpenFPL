"""What a playstyle actually changes.

A style is a *preference*, never a rule: it may bend which players it prefers
and how far ahead it looks, and it may not re-price a -4 or relax a squad
constraint. These pin both halves — that the tilt does what it says on the
projections, and that the shipped Balanced style leaves them alone, so the
default objective is still the one every backtest in CLAUDE.md measured.
"""
import pandas as pd
import pytest

from fpl_engine.optimise import chips, style


def frame():
    # two players on identical expected points: one explosive (a striker whose
    # points are goals), one steady (a defender's appearance + clean sheet)
    return pd.DataFrame({
        "player_id": [1, 2],
        "ep_gw5": [5.0, 5.0], "ep_gw6": [5.0, 5.0],
        "ex_gw5": [4.0, 0.5], "ex_gw6": [4.0, 0.5],
        "price_points": [0.20, 0.0],
    })


def test_balanced_does_not_touch_the_explosive_split():
    """upside=0 is the shipped default; a tilt of zero must be a no-op on the
    axis it governs, or the default objective has silently moved."""
    tilted = style.tilt_projection(frame(), [5, 6], upside=0.0, price=1.0)
    assert tilted.loc[0, "ep_gw5"] - tilted.loc[1, "ep_gw5"] == pytest.approx(
        0.20 / 2)      # only the price tie-breaker separates them


def test_no_tilt_at_all_returns_the_same_frame():
    f = frame()
    assert style.tilt_projection(f, [5, 6]) is f


def test_aggressive_prefers_the_explosive_player():
    t = style.tilt_projection(frame(), [5, 6], upside=0.15, price=0.0)
    assert t.loc[0, "ep_gw5"] > t.loc[1, "ep_gw5"]


def test_conservative_prefers_the_steady_one():
    t = style.tilt_projection(frame(), [5, 6], upside=-0.12, price=0.0)
    assert t.loc[1, "ep_gw5"] > t.loc[0, "ep_gw5"]


def test_price_value_is_spread_over_the_horizon_not_counted_per_gameweek():
    """`price_points` is already the points value of the whole hold, so adding
    it once per gameweek would multiply a 0.2-point tie-breaker by the horizon
    until it started overturning the ranking it is meant to break ties in."""
    f = frame()
    t = style.tilt_projection(f, [5, 6], price=1.0)
    added = ((t["ep_gw5"] - f["ep_gw5"]) + (t["ep_gw6"] - f["ep_gw6"]))
    assert added.iloc[0] == pytest.approx(0.20)


def test_a_tilt_never_produces_negative_expected_points():
    f = frame()
    f.loc[0, "ep_gw5"] = 0.05
    t = style.tilt_projection(f, [5, 6], upside=-5.0)
    assert (t["ep_gw5"] >= 0).all()


def test_missing_columns_degrade_instead_of_raising():
    """A projection cache built before the split existed has no ex_gw columns;
    the styles must still solve, just without the preference."""
    f = frame().drop(columns=["ex_gw5", "ex_gw6", "price_points"])
    t = style.tilt_projection(f, [5, 6], upside=0.15, price=1.0)
    assert t["ep_gw5"].tolist() == [5.0, 5.0]


def test_every_style_declares_both_tilts_and_the_rules_stay_fixed():
    for key, spec in chips.PLAYSTYLES.items():
        p = spec["params"]
        assert "upside" in p and "price" in p, key
        # a style may refuse hits outright but may never make one cheap
        assert "hit_cost" not in p, key
    assert chips.PLAYSTYLES["balanced"]["params"]["upside"] == 0.0


def test_the_old_style_key_still_resolves():
    """A saved preference from before the rename must not silently drop a plan."""
    assert chips.PLAYSTYLE_ALIASES["patient"] == "conservative"


def test_solving_does_not_mutate_the_shared_style_table():
    """optimise_playstyles pops the tilt keys out of its parameters; if that
    reached PLAYSTYLES the second solve of a session would lose the tilt."""
    before = dict(chips.PLAYSTYLES["aggressive"]["params"])
    params = {**{"decay": 0.9}, **chips.PLAYSTYLES["aggressive"]["params"]}
    params.pop("upside", None)
    params.pop("price", None)
    assert chips.PLAYSTYLES["aggressive"]["params"] == before
