"""Playstyles, club-level sell rules, and chip option value.

These pin the three behaviours a rolling-horizon optimiser gets wrong by
default: alternative plans that are near-duplicates of one optimum, no way to
say "get me out of this club", and chips treated as free upside so every one
gets burned inside the horizon.
"""
import pandas as pd
import pytest

from fpl_engine.optimise import chips


def _proj(gws, n_per_pos=6):
    """A squad-sized pool: enough players to form a legal 15 + alternatives."""
    rows, pid = [], 1
    quota = {"GK": 3, "DEF": 8, "MID": 8, "FWD": 5}
    for pos, n in quota.items():
        for i in range(n):
            rows.append({"player_id": pid, "player": f"{pos}{i}", "position": pos,
                         "team_id": (pid % 6) + 1, "price": 5.0,
                         **{f"ep_gw{g}": 2.0 + (i * 0.5) for g in gws}})
            pid += 1
    return pd.DataFrame(rows)


def _legal_squad(proj):
    """15 players honouring 2/5/5/3 and <=3 per club."""
    want = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    picked, per_club = [], {}
    for pos, n in want.items():
        for r in proj[proj.position == pos].itertuples():
            if per_club.get(r.team_id, 0) >= 3:
                continue
            picked.append(r.player_id)
            per_club[r.team_id] = per_club.get(r.team_id, 0) + 1
            n -= 1
            if n == 0:
                break
    return {p: 5.0 for p in picked}


def test_playstyles_return_one_plan_per_style():
    gws = [1, 2]
    proj = _proj(gws)
    plans = chips.optimise_playstyles(proj, gws, initial=_legal_squad(proj),
                                      bank=2.0, free_transfers=1, time_limit=20)
    assert [p.style for p in plans] == chips.DEFAULT_PLAYSTYLES
    assert all(p.style_label and p.style_note for p in plans)


def test_patient_style_never_takes_a_hit():
    gws = [1, 2]
    proj = _proj(gws)
    plans = chips.optimise_playstyles(proj, gws, initial=_legal_squad(proj),
                                      bank=2.0, free_transfers=1,
                                      styles=["patient"], time_limit=20)
    assert sum(g.get("hits", 0) or 0 for g in plans[0].per_gw) == 0


def test_allow_hits_false_forbids_paid_transfers():
    gws = [1]
    proj = _proj(gws)
    plan = chips.optimise_with_chips(proj, gws, initial=_legal_squad(proj),
                                     bank=50.0, free_transfers=1,
                                     allow_hits=False, max_transfers_per_gw=3,
                                     time_limit=20)[0]
    assert sum(g.get("hits", 0) or 0 for g in plan.per_gw) == 0


def test_total_ep_is_undecayed_and_net_of_hits():
    plan = chips.ChipPlan(gws=[1, 2], objective=0.0, per_gw=[
        {"xi_points": 50.0, "hits": 0}, {"xi_points": 40.0, "hits": 1}])
    assert plan.total_ep == 86.0        # 50 + 40 - 4, no decay applied


def test_sell_clubs_empties_that_club_by_horizon_end():
    gws = [1, 2, 3]
    proj = _proj(gws)
    initial = _legal_squad(proj)
    club_of = dict(zip(proj.player_id, proj.team_id))
    target = club_of[next(iter(initial))]
    owned = {p for p in initial if club_of[p] == target}
    assert owned, "fixture should own at least one player from the target club"
    plan = chips.optimise_with_chips(proj, gws, initial=initial, bank=50.0,
                                     free_transfers=3, sell_clubs={target},
                                     time_limit=20)[0]
    final = {p["player_id"] if isinstance(p, dict) else p
             for p in (plan.per_gw[-1].get("squad") or [])}
    assert not (owned & final), "players from a sell_club must be gone by the end"


def test_chip_reserve_stops_a_marginal_chip_being_burned():
    gws = [1, 2]
    proj = _proj(gws)
    initial = _legal_squad(proj)
    common = dict(initial=initial, bank=2.0, free_transfers=1,
                  chip_gws={"triple_captain": gws}, time_limit=20)
    # with no option value the chip is free upside and always played
    greedy = chips.optimise_with_chips(proj, gws,
                                       chip_reserve={"triple_captain": 0.0},
                                       **common)[0]
    assert any(g.get("chip") == "triple_captain" for g in greedy.per_gw)
    # priced as worth more later, a small gain no longer justifies it
    held = chips.optimise_with_chips(proj, gws,
                                     chip_reserve={"triple_captain": 500.0},
                                     **common)[0]
    assert not any(g.get("chip") == "triple_captain" for g in held.per_gw)


def test_default_chip_reserve_covers_every_chip():
    assert set(chips.DEFAULT_CHIP_RESERVE) == set(chips.CHIPS)
    assert all(v > 0 for v in chips.DEFAULT_CHIP_RESERVE.values())
