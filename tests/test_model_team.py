"""The model's own season-long team follows a manager's rules.

The solver is stubbed: these pin the bookkeeping around it — free transfers
bank one a week up to five, the opening squad is bought at price and owned
players are sold at FPL's selling price, hits are deducted, autosubs apply.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from app import modelteam
from fpl_engine.optimise import chips, project

META = {i: {"name": f"P{i}", "pos": pos, "team_id": i % 10 + 1}
        for i, pos in enumerate(["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["MID"], start=1)}


def _row(pid, in_xi=True, cap=False, vice=False, ep=2.0):
    m = META[pid]
    return {"player_id": pid, "name": m["name"], "position": m["pos"], "team_id": m["team_id"],
            "price": 5.0, "ep": ep, "in_xi": in_xi, "is_captain": cap, "is_vice": vice}


def _plan(squad_ids, ins=(), outs=(), used=0, hits=0, bank=1.0):
    xi = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
    rows = [_row(p, in_xi=p in xi, cap=p == 13, vice=p == 14) for p in squad_ids]
    return [SimpleNamespace(per_gw=[{
        "squad": rows, "transfers_in": [{"player_id": p} for p in ins],
        "transfers_out": [{"player_id": p} for p in outs],
        "free_used": used, "hits": hits, "bank": bank}])]


@pytest.fixture
def solver(monkeypatch):
    """Returns whatever plan the test sets on it, recording the solver kwargs."""
    holder = SimpleNamespace(plan=None, calls=[])
    monkeypatch.setattr(project, "prune", lambda f, **k: f)

    def fake(frame, gws, **kw):
        holder.calls.append(kw)
        return holder.plan
    monkeypatch.setattr(chips, "optimise_with_chips", fake)
    return holder


EPS = {5: {p: 2.0 for p in META}}
PRICES = {p: 5.0 for p in META}
FIFTEEN = list(range(1, 16))


def test_the_opening_squad_is_bought_at_price_and_leaves_one_free_transfer(solver):
    solver.plan = _plan(FIFTEEN, bank=0.5)
    out = modelteam._decide(None, "s", 5, EPS, None, PRICES, META)
    assert solver.calls[0]["initial"] is None and solver.calls[0]["budget"] == 100.0
    assert out["decision"]["build"] and out["decision"]["hits"] == 0
    assert out["state"]["ft"] == 1
    assert out["state"]["squad"]["1"] == 50                 # paid in tenths


def test_owned_players_sell_at_purchase_plus_half_the_profit(solver):
    state = {"squad": {str(p): 50 for p in FIFTEEN}, "bank": 0.0, "ft": 2}
    solver.plan = _plan(FIFTEEN)
    modelteam._decide(None, "s", 5, EPS, state, {**PRICES, 1: 5.3}, META)
    assert solver.calls[0]["initial"][1] == 5.1              # 5.0 + 0.3/2 rounded down
    assert solver.calls[0]["free_transfers"] == 2


def test_free_transfers_bank_to_five_and_a_transfer_spends_them(solver):
    state = {"squad": {str(p): 50 for p in FIFTEEN}, "bank": 0.0, "ft": 5}
    solver.plan = _plan(FIFTEEN)
    assert modelteam._decide(None, "s", 5, EPS, state, PRICES, META)["state"]["ft"] == 5

    state["ft"] = 1
    squad = [p for p in FIFTEEN if p != 2] + [16]
    solver.plan = _plan(squad, ins=[16], outs=[2], used=1)
    out = modelteam._decide(None, "s", 5, EPS, state, PRICES, META)
    assert out["state"]["ft"] == 1                          # 1 - 1 + 1
    assert "2" not in out["state"]["squad"] and out["state"]["squad"]["16"] == 50
    assert out["decision"]["transfers"][0]["out"]["player_id"] == 2


def test_a_hit_is_charged_four_points():
    decision = {"xi": [{"player_id": p, "pos": META[p]["pos"]} for p in [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]],
                "bench": [{"player_id": p, "pos": META[p]["pos"]} for p in [2, 6, 7, 12]],
                "captain": 13, "vice": 14, "hits": 1}
    actual = pd.DataFrame({"player_id": list(range(1, 16)), "pts": [2] * 15, "mins": [90] * 15})
    out = modelteam._score(decision, actual)
    assert out["points"] == 24.0                            # 11 x 2, captain doubled
    assert out["net"] == 20.0


def test_a_starter_who_did_not_play_is_autosubbed():
    decision = {"xi": [{"player_id": p, "pos": META[p]["pos"]} for p in [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]],
                "bench": [{"player_id": p, "pos": META[p]["pos"]} for p in [2, 6, 7, 12]],
                "captain": 13, "vice": 14, "hits": 0}
    actual = pd.DataFrame({"player_id": list(range(1, 16)), "pts": [2] * 15,
                           "mins": [0 if p == 8 else 90 for p in range(1, 16)]})
    actual.loc[actual.player_id == 8, "pts"] = 0
    assert modelteam._score(decision, actual)["points"] == 24.0   # bench DEF 6 comes on


def test_the_season_is_a_chain_and_summary_adds_it_up(tmp_path, monkeypatch):
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    doc = modelteam.load("2099-00")
    doc["weeks"] = {"1": {"gw": 1, "net": 60.0, "average": 50, "transfers": [], "hits": 0},
                    "2": {"gw": 2, "net": 40.0, "average": 45, "transfers": [{}], "hits": 1}}
    modelteam._save(doc)
    s = modelteam.summary("2099-00")
    assert [w["gw"] for w in s["weeks"]] == [1, 2]
    assert s["totals"] == {"net": 100.0, "average": 95, "transfers": 1, "hits": 1, "weeks_above_average": 1}
    assert "replay_cache" not in s
