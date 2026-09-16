"""Free transfers after a chip.

FPL's own wording: "if you had 2 saved free transfers before playing your
Wildcard, you will still have 2 saved free transfers in the following
Gameweek." Two in, two out — so a Wildcard or Free Hit week neither spends
the stock nor adds that week's +1 to it.

The estimator used to debit the chip week's transfers and then add one, which
is wrong in both directions at once: after a Wildcard with a dozen moves it
reported 1, and on a quiet week after one it reported 3 against FPL's 2. The
MILP always had this right; only the public-history estimator did not.
"""
import pytest

from fpl_engine.manager import MAX_FREE_TRANSFERS, estimate_free_transfers


def hist(events, chips=()):
    return {"current": [{"event": gw, "event_transfers": n}
                        for gw, n in events],
            "chips": [{"name": name, "event": gw} for name, gw in chips]}


def test_the_stock_starts_at_zero_not_one():
    """Pre-GW1 transfers are unlimited and bank nothing; the first free
    transfer is the one granted after GW1 finishes."""
    assert estimate_free_transfers(hist([(1, 15)])) == 1
    assert estimate_free_transfers(hist([(1, 0)])) == 1


def test_a_quiet_run_banks_one_a_week_to_the_cap():
    assert estimate_free_transfers(hist([(g, 0) for g in range(1, 4)])) == 3
    assert estimate_free_transfers(
        hist([(g, 0) for g in range(1, 12)])) == MAX_FREE_TRANSFERS


def test_transfers_made_are_debited():
    # gw1 -> 1, gw2 spends it -> 0, +1 -> 1, gw3 quiet -> 2
    assert estimate_free_transfers(hist([(1, 0), (2, 1), (3, 0)])) == 2


def test_a_wildcard_week_neither_spends_nor_accrues():
    """The regression: a Wildcard in GW3 followed by a quiet GW4 read 3 when
    FPL says 2, because the chip week was still granting its +1."""
    events = [(1, 0), (2, 1), (3, 0), (4, 0)]
    assert estimate_free_transfers(hist(events, [("wildcard", 3)])) == 2
    # and without the chip the same weeks are worth one more
    assert estimate_free_transfers(hist(events)) == 3


def test_a_wildcard_with_many_transfers_does_not_wipe_the_stock():
    """The other half of the same bug: debiting a dozen chip-week moves drove
    the stock to zero, so the optimiser planned -4 hits it did not need."""
    events = [(g, 0) for g in range(1, 4)] + [(4, 11), (5, 0)]
    assert estimate_free_transfers(hist(events, [("wildcard", 4)])) == 4


def test_free_hit_is_frozen_too():
    events = [(1, 0), (2, 0), (3, 9), (4, 0)]
    assert estimate_free_transfers(hist(events, [("freehit", 3)])) == 3


def test_bench_boost_and_triple_captain_are_normal_weeks():
    """Neither touches transfers, so a transfer made under one still costs."""
    events = [(1, 0), (2, 1), (3, 0)]
    for chip in ("bboost", "3xc"):
        assert estimate_free_transfers(hist(events, [(chip, 2)])) == 2


def test_the_stock_never_exceeds_the_cap_or_falls_below_one():
    assert estimate_free_transfers(
        hist([(g, 0) for g in range(1, 30)])) == MAX_FREE_TRANSFERS
    assert estimate_free_transfers(hist([(1, 0), (2, 5)])) == 1


def test_no_history_is_one_not_a_crash():
    assert estimate_free_transfers({}) == 1
    assert estimate_free_transfers({"current": [], "chips": []}) == 1


def test_an_unknown_chip_name_is_ignored_rather_than_freezing_the_week():
    """FPL adds chips (2024-25's assistant manager); an unrecognised one must
    not silently start preserving the stock."""
    events = [(1, 0), (2, 1), (3, 0), (4, 0)]
    assert estimate_free_transfers(hist(events, [("manager", 3)])) == 3
