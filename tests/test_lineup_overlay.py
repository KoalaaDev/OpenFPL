"""The predicted-lineup overlay: the feed resolves the band, nothing else.

E16 located the minutes error in the band where the model itself says a start
is a coin flip — 11% of rows carrying 40% of the whole minutes ceiling — and
priced a feed by its accuracy THERE. This season's archive says a player the
feed names in that band starts 76% of the time against the model's 56%, and
one it leaves out 33% against 46%, so the overlay replaces the start
probability in the band and leaves the rest of the board alone.
"""
import numpy as np
import pandas as pd
import pytest

from fpl_engine import lineup_feed as lf
from fpl_engine.xpts import engine


def test_the_feed_only_speaks_where_the_model_is_undecided():
    # in the band, either way
    assert lf.target_p_start(0.50, True) == 0.74
    assert lf.target_p_start(0.50, False) == 0.35
    assert lf.target_p_start(0.30, True) == 0.74 and lf.target_p_start(0.70, False) == 0.35
    # a surprise starter the model had written off
    assert lf.target_p_start(0.10, True) == 0.38
    # and the three cells where it is not better than the model: left alone
    assert lf.target_p_start(0.10, False) is None      # agree to within a point
    assert lf.target_p_start(0.95, True) is None       # nailed anyway
    assert lf.target_p_start(0.95, False) is None      # an omission here was wrong 78%


def test_every_shipped_number_is_a_shrunk_measurement():
    """No hand-tuning: each is the realised rate pulled toward what the model
    said, so a ten-row cell cannot swing the live board."""
    for (cell, in_xi), v in lf.START_POSTERIOR.items():
        assert 0.0 < v < 1.0
    # the thin cell is shrunk hard: 6/10 started, the model said 0.16
    assert lf.START_POSTERIOR[("unlikely", True)] == pytest.approx(
        (6 + lf.SHRINK_N * 0.16) / (10 + lf.SHRINK_N), abs=0.02)


def frame():
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "position": ["MID"] * 4,
        # 1: band, 2: band, 3: nailed, 4: written off
        "p_start": np.array([0.50, 0.50, 0.95, 0.05], dtype="float32"),
        "p_full": np.array([0.45, 0.45, 0.90, 0.04], dtype="float32"),
        "p_sub": np.array([0.20, 0.20, 0.05, 0.20], dtype="float32"),
        "p_none": np.array([0.35, 0.35, 0.05, 0.76], dtype="float32"),
        "e_min": np.array([40.0, 40.0, 80.0, 12.0], dtype="float32"),
        "m_played": np.array([62.0, 62.0, 85.0, 50.0], dtype="float32"),
    })


def test_a_named_player_gains_exposure_and_an_omitted_one_loses_it():
    out = engine._apply_lineup_xi(frame(), {1: True, 2: False})
    a = out.set_index("player_id")
    assert a.loc[1, "p_start"] == 0.74 and a.loc[2, "p_start"] == 0.35
    assert a.loc[1, "p_full"] > 0.45 and a.loc[2, "p_full"] < 0.45
    assert a.loc[1, "e_min"] > 40 and a.loc[2, "e_min"] < 40
    # he is less likely to be a substitute precisely because he is starting
    assert a.loc[1, "p_sub"] < 0.20 and a.loc[2, "p_sub"] > 0.20


def test_it_keeps_the_players_own_shape():
    """Only P(start) is replaced. How long he lasts GIVEN he starts, and how
    often he appears GIVEN he does not, are his own — a keeper and a rotated
    winger must not be flattened into the same player."""
    out = engine._apply_lineup_xi(frame(), {1: True})
    a = out.set_index("player_id")
    assert a.loc[1, "p_full"] / a.loc[1, "p_start"] == pytest.approx(0.45 / 0.50, rel=1e-3)
    assert a.loc[1, "p_sub"] / (1 - a.loc[1, "p_start"]) == pytest.approx(0.20 / 0.50, rel=1e-3)
    assert a.loc[1, "e_min"] == pytest.approx(
        (a.loc[1, "p_sub"] + a.loc[1, "p_full"]) * 62.0, rel=1e-6)


def test_the_confident_rows_are_untouched():
    out = engine._apply_lineup_xi(frame(), {3: False, 4: False})
    before, after = frame().set_index("player_id"), out.set_index("player_id")
    for pid in (3, 4):
        for col in ("p_start", "p_full", "p_sub", "e_min"):
            assert after.loc[pid, col] == pytest.approx(float(before.loc[pid, col]), rel=1e-6)


def test_probabilities_stay_a_distribution():
    out = engine._apply_lineup_xi(frame(), {1: True, 2: False, 4: True})
    total = out["p_none"] + out["p_sub"] + out["p_full"]
    assert np.allclose(total, 1.0, atol=1e-6)
    assert (out[["p_none", "p_sub", "p_full", "p_start"]] >= 0).all().all()
    assert (out[["p_none", "p_sub", "p_full", "p_start"]] <= 1).all().all()


def test_a_player_the_feed_says_nothing_about_is_left_alone():
    out = engine._apply_lineup_xi(frame(), {})
    assert out.equals(frame()) or np.allclose(out["p_start"], frame()["p_start"])
    assert "lineup_rows" not in out.attrs


def test_the_engine_ignores_the_feed_unless_it_is_given_one():
    """Backtests and replays pass nothing, so `data/bt_base` cannot move: the
    archive begins in 2026-27 and no replayed season can see it."""
    import inspect
    src = inspect.getsource(engine.xpts_predict_gw)
    assert "lineup_xi" in str(inspect.signature(engine.xpts_predict_gw))
    assert "if lineup_xi:" in src
    import fpl_engine.backtest as bt
    assert "lineup_xi" not in inspect.getsource(bt.run)


def test_only_the_forecast_gameweek_is_overlaid():
    """A feed forecasts the imminent XI and says nothing about GW+2; carrying
    one week's rotation across a horizon would be inventing information."""
    import inspect

    from fpl_engine.optimise import project
    src = inspect.getsource(project.horizon_projections)
    assert "g == lineup_gw" in src
