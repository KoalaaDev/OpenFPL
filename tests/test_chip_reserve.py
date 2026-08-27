"""A saved chip's worth is an option value, and it decays with the season.

`DEFAULT_CHIP_RESERVE` priced holding any chip at a flat 15-20 points. What it
is really trying to express is E[max over the gameweeks left] minus the value
of playing it in the best-looking week — a premium that shrinks as the season
runs out and that a constant cannot represent. Simulating all 74 replayed
gameweeks put it at 5.4 / 7.0 / 8.1 (Triple Captain) and 8.0 / 10.5 / 12.1
(Bench Boost) with 5 / 10 / 20 gameweeks left.
"""
import pytest

from fpl_engine.optimise import chips


def test_reserve_grows_with_the_season_left():
    tc = [chips.chip_reserve_for("triple_captain", n) for n in (3, 5, 10, 20, 30)]
    assert tc == sorted(tc)
    bb = [chips.chip_reserve_for("bench_boost", n) for n in (3, 5, 10, 20, 30)]
    assert bb == sorted(bb)


def test_reserve_matches_what_was_simulated():
    for chip, expected in (("triple_captain", {5: 5.40, 10: 7.00, 20: 8.06}),
                           ("bench_boost", {5: 8.04, 10: 10.54, 20: 12.12})):
        for n, want in expected.items():
            got = chips.chip_reserve_for(chip, n)
            assert got == pytest.approx(want, abs=0.5), (chip, n, got, want)


def test_bench_boost_is_worth_more_held_than_triple_captain():
    """Its payoff is four players, not one, so its best week is further out."""
    for n in (5, 10, 20):
        assert (chips.chip_reserve_for("bench_boost", n)
                > chips.chip_reserve_for("triple_captain", n))


def test_the_flat_heuristic_was_too_expensive_at_a_real_horizon():
    """Over-reserving makes the solver hoard a chip it should have played."""
    for n in (3, 5, 10, 20):
        assert chips.chip_reserve_for("triple_captain", n) < 15.0
        assert chips.chip_reserve_for("bench_boost", n) < 15.0


def test_unmeasured_chips_keep_the_documented_heuristic():
    """Wildcard and Free Hit rebuild a squad, so the one-week option value
    measured here does not apply to them."""
    for chip in ("wildcard", "freehit"):
        assert (chips.chip_reserve_for(chip, 10)
                == chips.DEFAULT_CHIP_RESERVE[chip])


def test_unknown_horizon_falls_back_to_the_flat_map():
    assert chips.default_reserve(None) == chips.DEFAULT_CHIP_RESERVE
    assert chips.chip_reserve_for("triple_captain", 0) == 15.0


def test_reserve_is_never_negative():
    for chip in chips.DEFAULT_CHIP_RESERVE:
        for n in (1, 2, 38):
            assert chips.chip_reserve_for(chip, n) >= 0.0
