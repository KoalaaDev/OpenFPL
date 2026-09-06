"""Chip awareness: what the entry has spent, holds, and has active right now.

The distinction these tests pin is the one that made the planner guess:
``entry/{id}/history/`` only reports chips whose gameweek has been PLAYED, so a
Wildcard activated for the coming deadline is invisible to the public API and
arrives only with an authenticated my-team import.
"""
import pytest

from fpl_engine import manager
from app.services import _apply_chip_state

BOOT = {
    "events": [{"id": 1, "finished": True}, {"id": 2, "finished": True},
               {"id": 3, "is_next": True, "finished": False}],
    "chips": [
        {"name": "wildcard", "start_event": 2, "stop_event": 19},
        {"name": "wildcard", "start_event": 20, "stop_event": 38},
        {"name": "freehit", "start_event": 2, "stop_event": 19},
        {"name": "bboost", "start_event": 1, "stop_event": 19},
        {"name": "3xc", "start_event": 1, "stop_event": 19},
        {"name": "freehit", "start_event": 20, "stop_event": 38},
        {"name": "bboost", "start_event": 20, "stop_event": 38},
        {"name": "3xc", "start_event": 20, "stop_event": 38},
        # a chip the optimiser does not model (2024-25's assistant manager)
        {"name": "manager", "start_event": 24, "stop_event": 28},
    ],
}


def state(**kw):
    return manager.chip_state(boot=BOOT, **kw)


def test_played_chips_come_from_public_history_under_engine_names():
    st = state(history={"chips": [{"name": "bboost", "event": 2}]})
    assert st["played"] == [{"chip": "bench_boost", "gw": 2}]
    assert st["active"] is None
    assert st["source"] == "public"


def test_a_chip_spent_in_one_half_leaves_the_other_half_available():
    st = state(history={"chips": [{"name": "wildcard", "event": 3}]})
    used = [w for w in st["windows"] if w["chip"] == "wildcard" and w["used_gw"]]
    assert [(w["start"], w["stop"]) for w in used] == [(2, 19)]
    # still offered, because the GW20-38 wildcard is untouched
    assert "wildcard" in st["available"]
    free = [w for w in st["windows"]
            if w["chip"] == "wildcard" and w["used_gw"] is None]
    assert [(w["start"], w["stop"]) for w in free] == [(20, 38)]


def test_both_halves_spent_removes_the_chip_entirely():
    st = state(history={"chips": [{"name": "3xc", "event": 5},
                                  {"name": "3xc", "event": 25}]})
    assert "triple_captain" not in st["available"]
    assert all(w["used_gw"] for w in st["windows"] if w["chip"] == "triple_captain")


def test_only_my_team_can_report_a_chip_active_for_the_coming_deadline():
    mine = manager.my_team_chips({"chips": [
        {"name": "wildcard", "status_for_entry": "active",
         "start_event": 2, "stop_event": 19, "played_by_entry": []},
        {"name": "bboost", "status_for_entry": "played",
         "start_event": 1, "stop_event": 19, "played_by_entry": [2]},
    ]})
    st = state(history={"chips": []}, my_team_chips_=mine)
    assert st["active"] == "wildcard"
    assert st["source"] == "my-team"
    # the my-team block also back-fills what history has not recorded yet
    assert {"chip": "bench_boost", "gw": 2} in st["played"]
    # an active chip is not also offered as a free choice
    assert "wildcard" not in st["available"]


def test_unmodelled_chips_are_dropped_not_guessed_at():
    assert all(w["chip"] in manager.CHIP_LABELS for w in state()["windows"])


def test_missing_bootstrap_table_falls_back_to_the_two_halves():
    windows = manager.chip_windows({"chips": []})
    assert len(windows) == 8
    assert {(w["start"], w["stop"]) for w in windows} == {(1, 19), (20, 38)}


# --------------------------------------------------------------------------
# the solver-side enforcement
# --------------------------------------------------------------------------

def test_a_spent_chip_is_dropped_from_the_solve():
    st = state(history={"chips": [{"name": "bboost", "event": 2},
                                  {"name": "bboost", "event": 25}]})
    gws, reserve = [3, 4, 5], {}
    chip_gws, force = {"bench_boost": [3, 4, 5]}, {}
    dropped = _apply_chip_state(st, chip_gws, force, reserve, gws)
    assert dropped == ["bench_boost"] and chip_gws == {}


def test_a_chip_is_held_to_the_half_it_is_still_owned_in():
    st = state(history={"chips": [{"name": "wildcard", "event": 3}]})
    chip_gws, force, reserve = {"wildcard": [10, 11, 20, 21]}, {}, {}
    assert _apply_chip_state(st, chip_gws, force, reserve, [10, 11, 20, 21]) == []
    assert chip_gws["wildcard"] == [20, 21]      # first-half one is spent


def test_forcing_a_chip_into_a_spent_window_drops_it_rather_than_moving_it():
    st = state(history={"chips": [{"name": "wildcard", "event": 3}]})
    chip_gws, force, reserve = {"wildcard": [10, 20]}, {"wildcard": 10}, {}
    assert _apply_chip_state(st, chip_gws, force, reserve, [10, 20]) == ["wildcard"]
    assert chip_gws == {} and force == {}


def test_an_active_chip_is_pinned_to_the_first_gameweek_with_no_option_value():
    mine = manager.my_team_chips({"chips": [
        {"name": "wildcard", "status_for_entry": "active",
         "start_event": 2, "stop_event": 19, "played_by_entry": []}]})
    st = state(history={"chips": []}, my_team_chips_=mine)
    chip_gws, force, reserve = {}, {}, {}
    _apply_chip_state(st, chip_gws, force, reserve, [3, 4, 5])
    assert chip_gws["wildcard"] == [3] and force["wildcard"] == 3
    assert reserve["wildcard"] == 0.0     # already spent: saving it is not on offer


def test_an_active_chip_is_not_pinned_to_a_horizon_that_starts_later():
    mine = manager.my_team_chips({"chips": [
        {"name": "wildcard", "status_for_entry": "active",
         "start_event": 2, "stop_event": 19, "played_by_entry": []}]})
    st = state(history={"chips": []}, my_team_chips_=mine)
    chip_gws, force, reserve = {}, {}, {}
    _apply_chip_state(st, chip_gws, force, reserve, [7, 8, 9])
    assert chip_gws == {} and force == {} and reserve == {}


# --------------------------------------------------------------------------
# which squad the planner starts from, once a chip is live mid-week
# --------------------------------------------------------------------------

DEADLINE = "2026-08-28T17:30:00Z"          # GW2's real 2026-27 deadline
GW2_DEADLINE_EPOCH = 1787938200.0
PUBLIC = {"gw": 2, "name": "Pen Palmer Royale", "bank": 0.5,
          "free_transfers": 1, "squad": [{"element": 1}]}


def _services(monkeypatch, mine):
    from app import services
    monkeypatch.setattr(services, "bootstrap",
                        lambda: {"events": [{"id": 2, "deadline_time": DEADLINE}]})
    monkeypatch.setattr(services.manager, "current_squad",
                        lambda eid, **kw: dict(PUBLIC))
    monkeypatch.setattr(services, "load_my_team", lambda: mine)
    return services


def _import(saved_at, **kw):
    doc = {"entry_id": 883566, "squad": [{"element": 2}], "bank": 1.2,
           "free_transfers": 0, "unlimited_transfers": True,
           "active_chip": "wildcard", "chips": [], "source": "bookmarklet",
           "saved_at": saved_at}
    doc.update(kw)
    return doc


def test_an_import_taken_after_the_deadline_beats_the_frozen_public_picks(monkeypatch):
    s = _services(monkeypatch, _import(GW2_DEADLINE_EPOCH + 3600))
    st = s.squad_state(883566)
    assert st["source"] == "bookmarklet"
    assert st["squad"] == [{"element": 2}]
    assert st["active_chip"] == "wildcard"
    assert st["name"] == "Pen Palmer Royale"     # public still names the team


def test_an_import_taken_before_the_deadline_loses_to_the_public_picks(monkeypatch):
    s = _services(monkeypatch, _import(GW2_DEADLINE_EPOCH - 3600))
    st = s.squad_state(883566)
    assert st["source"] == "public" and st["squad"] == [{"element": 1}]
    assert st["active_chip"] is None


def test_a_hand_entered_squad_never_overrides_the_real_one(monkeypatch):
    s = _services(monkeypatch, _import(GW2_DEADLINE_EPOCH + 3600, source="manual"))
    assert s.squad_state(883566)["source"] == "public"
