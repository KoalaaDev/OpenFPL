"""Three defects found from the planner UI, all silent.

None of them raised. The squad the solver produced was legal every time; what
was wrong was what the plan *said* about it, and one rule it got wrong in the
manager's favour.

1. `transfers_in` / `transfers_out` were two independent id-ordered sets. Any
   consumer pairing them by index - the Planner does, and so does the CLI's
   IN/OUT block - rendered moves like "DEF -> FWD" that the solver never made
   and the rules forbid.
2. A Free Hit pins `tin`/`tout` to zero, because it is not a transfer and it
   reverts. Correct, but it meant the plan reported no changes at all for the
   one gameweek that changes most.
3. The free-transfer stock accrued +1 in a chip gameweek. FPL preserves the
   stock instead: "if you had 2 saved free transfers, you will still have 2
   saved free transfers the Gameweek after playing the chip."
"""
import pandas as pd
import pytest

from fpl_engine.optimise import chips


def _proj(gws=(1, 2), boost_gw=None, boost_clubs=3):
    rows, pid = [], 1
    for c in range(12):
        for i in range(8):
            pos = ["GK", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD"][i]
            r = {"player_id": pid, "player": f"p{pid}", "position": pos,
                 "team_id": c, "price": 5.0}
            for g in gws:
                if g == boost_gw:
                    r[f"ep_gw{g}"] = 9.0 if c < boost_clubs else 1.0
                else:
                    r[f"ep_gw{g}"] = 2.0 + ((pid * 7) % 11) * 0.1
            rows.append(r); pid += 1
    return pd.DataFrame(rows)


def _seed(proj, gws):
    p0 = chips.optimise_with_chips(proj, [gws[0]], initial=None, budget=100.0,
                                   chip_gws={}, time_limit=20)[0]
    return {r["player_id"]: 5.0 for r in p0.per_gw[0]["squad"]}


# ------------------------------------------------------ 1. transfer pairing ---
def test_transfers_pair_up_by_position():
    """The squad quota is enforced every gameweek, so the number leaving a
    position always equals the number arriving in it. Ordering both lists by
    position makes index-pairing legal by construction."""
    proj = _proj((1, 2), boost_gw=2)
    plan = chips.optimise_with_chips(proj, [1, 2], initial=_seed(proj, (1, 2)),
                                     bank=0.0, free_transfers=3, chip_gws={},
                                     max_transfers_per_gw=3, time_limit=40)[0]
    moved = False
    for g in plan.per_gw:
        assert len(g["transfers_in"]) == len(g["transfers_out"])
        for out, inn in zip(g["transfers_out"], g["transfers_in"]):
            assert out["position"] == inn["position"], (
                f"GW{g['gw']}: {out['position']} -> {inn['position']} is not a "
                "legal single transfer")
            moved = True
    assert moved, "the fixture never produced a transfer to check"


def test_the_squad_itself_was_always_legal():
    """The bug was presentational: composition never actually broke."""
    proj = _proj((1, 2), boost_gw=2)
    plan = chips.optimise_with_chips(proj, [1, 2], initial=_seed(proj, (1, 2)),
                                     bank=0.0, free_transfers=3, chip_gws={},
                                     time_limit=40)[0]
    for g in plan.per_gw:
        counts = {}
        for r in g["squad"]:
            counts[r["position"]] = counts.get(r["position"], 0) + 1
        assert counts == {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


# ------------------------------------------------------------ 2. free hit ----
def test_a_free_hit_reports_what_it_changes():
    proj = _proj((1, 2), boost_gw=2)
    plan = chips.optimise_with_chips(
        proj, [1, 2], initial=_seed(proj, (1, 2)), bank=0.0, free_transfers=1,
        chip_gws={"freehit": [2]}, chip_force={"freehit": 2},
        chip_reserve={"freehit": 0.0}, time_limit=60)[0]
    fh = next(g for g in plan.per_gw if g["chip"] == "freehit")
    assert fh["n_transfers"] == 0, "a Free Hit is not a transfer"
    assert fh["fh_in"], "the chip must report who it brings in"
    assert len(fh["fh_in"]) == len(fh["fh_out"])
    for out, inn in zip(fh["fh_out"], fh["fh_in"]):
        assert out["position"] == inn["position"]
    assert sum(1 for r in fh["squad"] if r["in_xi"]) == 11
    assert fh["captain_id"] is not None


def test_a_normal_gameweek_reports_no_free_hit_changes():
    proj = _proj((1, 2), boost_gw=2)
    plan = chips.optimise_with_chips(proj, [1, 2], initial=_seed(proj, (1, 2)),
                                     bank=0.0, free_transfers=2, chip_gws={},
                                     time_limit=40)[0]
    for g in plan.per_gw:
        assert not g["fh_in"] and not g["fh_out"]


# ------------------------------------------- 3. free transfers on chip weeks ---
def _stock(chip):
    proj = _proj((1, 2, 3), boost_gw=2)
    cg = {chip: [2]} if chip else {}
    plan = chips.optimise_with_chips(
        proj, [1, 2, 3], initial=_seed(proj, (1, 2, 3)), bank=0.0,
        free_transfers=2, chip_gws=cg,
        chip_force=({chip: 2} if chip else {}),
        chip_reserve={"wildcard": 0.0, "freehit": 0.0},
        max_transfers_per_gw=15, time_limit=60)[0]
    return {g["gw"]: g["free_used"] + g["free_after"] for g in plan.per_gw}, plan


@pytest.mark.parametrize("chip", ["wildcard", "freehit"])
def test_a_chip_week_preserves_the_stock_but_does_not_add_to_it(chip):
    stock, plan = _stock(chip)
    played = next(g for g in plan.per_gw if g["chip"] == chip)
    assert played["gw"] == 2
    assert played["free_used"] == 0, "a chip week consumes no free transfers"
    # two in, two out: the gameweek AFTER the chip has the same stock
    assert stock[3] == pytest.approx(stock[2]), (
        f"stock went {stock[2]} -> {stock[3]} across a {chip}; FPL preserves "
        "it rather than accruing on top")


def test_a_normal_week_still_accrues_one():
    proj = _proj((1, 2, 3))
    plan = chips.optimise_with_chips(
        proj, [1, 2, 3], initial=_seed(proj, (1, 2, 3)), bank=0.0,
        free_transfers=1, chip_gws={}, min_ft={1: 1, 2: 2}, time_limit=40)[0]
    stock = {g["gw"]: g["free_used"] + g["free_after"] for g in plan.per_gw}
    assert stock[2] == pytest.approx(min(chips.MAX_FREE_TRANSFERS, stock[1] + 1))
