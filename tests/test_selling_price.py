"""Selling-price reconstruction + the guard that stops £0.0 reaching the MILP.

The public picks endpoint carries no selling prices. Defaulting them to £0.0
makes every sale raise nothing, so no transfer is affordable and the optimiser
returns a plausible-looking "do nothing" plan. These tests pin both halves of
the fix: the reconstruction is correct, and a zero price can never pass
silently.
"""
import pandas as pd
import pytest

from fpl_engine import manager
from fpl_engine.optimise import chips, milp


# --- FPL's sell rule: purchase + half the profit, rounded down -------------

@pytest.mark.parametrize("purchase, now, expected", [
    (50, 50, 50),     # unchanged
    (50, 54, 52),     # +0.4 profit -> keep half (+0.2)
    (50, 55, 52),     # +0.5 profit -> rounds down to +0.2
    (50, 56, 53),     # +0.6 profit -> +0.3
    (50, 45, 45),     # price fell: full loss, sell at current value
    (120, 130, 125),  # premium riser
])
def test_selling_price_rule(purchase, now, expected):
    assert manager.selling_price(purchase, now) == expected


def test_selling_price_never_exceeds_current_value():
    for purchase in range(35, 150):
        for now in range(35, 150):
            assert manager.selling_price(purchase, now) <= max(now, purchase)


def test_reconstruct_prices_uses_transfer_cost_then_start_price(monkeypatch):
    # element 1 never transferred in -> season-start price 5.0 (55-5);
    # risen to 5.5, so it sells for 5.0 + (0.5 profit // 2) = 5.2
    # element 2 bought for 60 and has since risen to 70 -> 60 + (70-60)//2
    monkeypatch.setattr(manager, "fetch_bootstrap", lambda use_cache=True: {
        "elements": [
            {"id": 1, "now_cost": 55, "cost_change_start": 5},
            {"id": 2, "now_cost": 70, "cost_change_start": 10},
        ]})
    monkeypatch.setattr(manager, "fetch_transfers", lambda e, use_cache=False: [
        {"event": 2, "time": "t1", "element_in": 2, "element_in_cost": 60},
    ])
    out = manager.reconstruct_prices(1, [1, 2])
    assert out[1] == {"purchase_price": 5.0, "selling_price": 5.2}
    assert out[2] == {"purchase_price": 6.0, "selling_price": 6.5}


def test_reconstruct_prices_last_transfer_in_wins(monkeypatch):
    monkeypatch.setattr(manager, "fetch_bootstrap", lambda use_cache=True: {
        "elements": [{"id": 7, "now_cost": 80, "cost_change_start": 0}]})
    monkeypatch.setattr(manager, "fetch_transfers", lambda e, use_cache=False: [
        {"event": 9, "time": "t2", "element_in": 7, "element_in_cost": 78},
        {"event": 3, "time": "t1", "element_in": 7, "element_in_cost": 70},
    ])
    # sorted by (event, time): the gw9 buy at 7.8 is the one that counts
    assert manager.reconstruct_prices(1, [7])[7]["purchase_price"] == 7.8


# --- the guard -------------------------------------------------------------

def _proj():
    return pd.DataFrame({"player_id": [1, 2], "player": ["a", "b"],
                         "position": ["MID", "MID"], "team_id": [1, 2],
                         "price": [5.0, 5.0], "ep_gw1": [3.0, 4.0]})


@pytest.mark.parametrize("solve", [
    lambda **kw: chips.optimise_with_chips(_proj(), [1], **kw),
    lambda **kw: milp.optimise(_proj(), [1], **kw),
])
def test_zero_selling_price_fails_loud(solve):
    with pytest.raises(ValueError, match=r"selling price"):
        solve(initial={1: 0.0, 2: 0.0})


@pytest.mark.parametrize("solve", [
    lambda **kw: chips.optimise_with_chips(_proj(), [1], **kw),
    lambda **kw: milp.optimise(_proj(), [1], **kw),
])
def test_partial_zero_selling_price_also_fails(solve):
    with pytest.raises(ValueError, match=r"selling price"):
        solve(initial={1: 5.0, 2: 0.0})


def test_scratch_build_is_unaffected():
    # initial=None means "build from budget" — the guard must not fire
    chips.optimise_with_chips(_proj(), [1], initial=None, budget=100.0)


# --- free-transfer accrual -------------------------------------------------

def _hist(*made):
    return {"current": [{"event": i + 1, "event_transfers": m}
                        for i, m in enumerate(made)]}


@pytest.mark.parametrize("made, expected", [
    ((0,), 1),              # after a quiet GW1 you have ONE FT for GW2 —
                            # pre-deadline transfers are unlimited and bank nothing
    ((0, 0), 2),            # quiet GW2 banks the second
    ((0, 1), 1),            # used it in GW2 -> back to 1
    ((0, 2), 1),            # took a hit in GW2 -> still 1
    ((0, 0, 0, 0, 0, 0), 5),   # capped at 5
    ((), 1),                # no completed gameweeks -> default 1
])
def test_free_transfer_accrual(made, expected):
    assert manager.estimate_free_transfers(_hist(*made)) == expected


def test_free_transfers_never_exceed_cap():
    assert manager.estimate_free_transfers(_hist(*([0] * 20))) == \
        manager.MAX_FREE_TRANSFERS
