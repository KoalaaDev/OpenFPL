"""The model's season record.

The dashboard is only worth reading if three things hold: a gameweek is
scored on what the model believed BEFORE the deadline (never a projection
rebuilt afterwards), the teams it builds obey the rules they claim, and the
scoring follows FPL (autosubs, the armband passing to the vice).
"""
import pytest

from app import modelrecord as mr


def row(pid, pos, team, ep, pts, mins=90, price=5.0):
    return {"player_id": pid, "name": f"p{pid}", "pos": pos, "team_id": team,
            "price": price, "ep": ep, "pts": pts, "mins": mins}


def league(clubs=8):
    out, pid = [], 0
    for club in range(1, clubs + 1):
        for pos, k in (("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
            for i in range(k):
                pid += 1
                out.append(row(pid, pos, club, ep=6.0 - club * 0.3 - i * 0.1,
                               pts=(pid * 7) % 13, price=4.0 + (club % 3)))
    return out


# ------------------------------------------------------------ selection --
def test_unlimited_xi_is_legal_and_capped_at_three_per_club():
    sel = mr._select(league(), "ep")
    xi = sel["xi"]
    assert len(xi) == 11
    n = {p: sum(x["pos"] == p for x in xi) for p in mr.POS}
    for pos, (lo, hi) in mr.XI_LIMITS.items():
        assert lo <= n[pos] <= hi
    from collections import Counter
    assert max(Counter(x["team_id"] for x in xi).values()) <= 3
    assert sel["captain"] in xi and sel["vice"] in xi and sel["captain"] is not sel["vice"]


def test_budget_squad_is_fifteen_within_budget_and_two_five_five_three():
    sel = mr._select(league(), "ep", budget=mr.BUDGET)
    squad = sel["xi"] + sel["bench"]
    assert len(squad) == 15 and len(sel["bench"]) == 4
    assert sum(p["price"] for p in squad) <= mr.BUDGET + 1e-9
    assert {p: sum(x["pos"] == p for x in squad) for p in mr.POS} == mr.SQUAD_SIZE
    assert sel["bench"][0]["pos"] == "GK"          # FPL bench: keeper first


def test_a_tight_budget_changes_the_team():
    """If the budget never bit, the '£100m squad' would just be the free XI."""
    rich = [row(900 + i, pos, 20 + i, ep=12.0, pts=0, price=14.0)
            for i, pos in enumerate(["FWD", "FWD", "MID", "MID"])]
    ps = league() + rich
    free = {p["player_id"] for p in mr._select(ps, "ep")["xi"]}
    capped = {p["player_id"] for p in mr._select(ps, "ep", budget=mr.BUDGET)["xi"]}
    assert len(free & {900, 901, 902, 903}) == 4
    assert len(capped & {900, 901, 902, 903}) < 4


# --------------------------------------------------------------- scoring --
def _sel(xi, cap, vice, bench=None):
    s = {"xi": xi, "captain": cap, "vice": vice}
    if bench is not None:
        s["bench"] = bench
    return s


def test_armband_passes_to_the_vice_when_the_captain_does_not_play():
    xi = [row(i, "MID", i, 5, 5) for i in range(1, 12)]
    cap = row(99, "FWD", 50, 9, 0, mins=0)
    vice = xi[0]
    xi[-1] = cap
    # captain blanked (0 minutes): vice's 5 is doubled instead
    assert mr._score(_sel(xi, cap, vice), False) == 10 * 5 + 0 + 5


def test_autosubs_bring_on_the_bench_for_a_starter_who_did_not_play():
    gk = row(1, "GK", 1, 4, 3)
    defs = [row(i, "DEF", i, 4, 2) for i in range(2, 6)]
    mids = [row(i, "MID", i, 5, 4) for i in range(6, 10)]
    fwds = [row(10, "FWD", 10, 6, 6), row(11, "FWD", 11, 6, 0, mins=0)]
    xi = [gk] + defs + mids + fwds
    bench = [row(12, "GK", 12, 3, 9), row(13, "FWD", 13, 3, 7),
             row(14, "DEF", 14, 3, 1), row(15, "MID", 15, 3, 1)]
    got = mr._score(_sel(xi, fwds[0], mids[0], bench), True)
    base = 3 + 2 * 4 + 4 * 4 + 6 + 0
    # the non-playing forward is replaced by the first eligible bench player
    # (the forward, 7); the bench keeper stays off; captain 6 doubled
    assert got == base + 7 + 6


# ----------------------------------------------------------- provenance --
def test_a_live_snapshot_is_the_last_one_built_before_the_deadline(monkeypatch):
    from app import services
    monkeypatch.setattr(services, "_load_history", lambda: [
        {"built_at": 100.0, "gws": {"5": {"1": 2.0}}},
        {"built_at": 200.0, "gws": {"5": {"1": 3.0}}},
        {"built_at": 300.0, "gws": {"5": {"1": 9.9}}},     # after the deadline
        {"built_at": 250.0, "gws": {"6": {"1": 4.0}}},     # a different gameweek
    ])
    ep, built = mr._live_snapshot(5, deadline=260.0)
    assert built == 200.0 and ep == {1: 3.0}


def test_no_snapshot_before_the_deadline_means_none():
    from app import services
    import unittest.mock as um
    with um.patch.object(services, "_load_history", return_value=[
            {"built_at": 500.0, "gws": {"5": {"1": 2.0}}}]):
        assert mr._live_snapshot(5, deadline=400.0) is None


def test_refresh_builds_each_finished_gameweek_once(tmp_path, monkeypatch):
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(mr, "_events", lambda: {
        1: {"id": 1, "finished": True}, 2: {"id": 2, "finished": True},
        3: {"id": 3, "finished": False}})
    calls = []
    monkeypatch.setattr(mr, "build_gw", lambda conn, s, gw, ev: calls.append(gw) or {"gw": gw})
    assert mr.refresh()["built"] == [1, 2]
    assert mr.refresh()["built"] == []             # nothing rebuilt on the next tick
    assert calls == [1, 2]
    assert set(mr.load()["gws"]) == {"1", "2"}
