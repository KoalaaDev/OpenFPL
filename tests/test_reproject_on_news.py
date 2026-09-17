"""Projections rebuild when team news matters — and only then.

A rebuild costs minutes of server time, so a status change triggers one only
when it is material AND the player is widely owned or relied on by the model;
news inside the cooldown waits rather than being dropped.
"""
import sqlite3

import pytest

from app import jobs, scheduler, services


def test_availability_classes():
    assert scheduler._availability_class("a", None) == ("a", 100)
    assert scheduler._availability_class("d", 75) == ("d", 75)
    assert scheduler._availability_class("d", None) == ("d", 50)
    assert scheduler._availability_class("i", None) == ("o", 0)
    assert scheduler._availability_class("s", 0) == ("o", 0)


def test_what_counts_as_material():
    assert scheduler.material(("a", 100), ("o", 0))
    assert scheduler.material(("d", 75), ("d", 25))          # 50-point move
    assert scheduler.material(("d", 75), ("d", 50))          # 25 points is the bar
    assert not scheduler.material(("d", 75), ("d", 60))
    assert not scheduler.material(("d", 50), ("d", 50))
    assert not scheduler.material(None, ("a", 100))          # a fit newcomer is not news
    assert scheduler.material(None, ("o", 0))


@pytest.fixture
def news_db(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE acq_player_availability (season, player_id, observed_utc, status, chance_next, raw_id)")
    conn.execute("CREATE TABLE player (season, player_id, web_name)")
    conn.executemany("INSERT INTO player VALUES ('s', ?, ?)", [(1, "Star"), (2, "Fringe"), (3, "Nailed")])
    rows = [(p, "2026-09-01T00:00:00Z", "a", None, 1) for p in (1, 2, 3)]
    rows += [(p, "2026-09-02T00:00:00Z", "i", 0, 2) for p in (1, 2, 3)]
    conn.executemany("INSERT INTO acq_player_availability VALUES ('s', ?, ?, ?, ?, ?)", rows)
    monkeypatch.setattr(services, "players_payload", lambda: {"players": [
        {"id": 1, "own": 35.0}, {"id": 2, "own": 0.4}, {"id": 3, "own": 1.0}]})
    monkeypatch.setattr(services, "_load_proj_cache", lambda: {"players": {
        "1": {"ep": {"5": 7.1}}, "2": {"ep": {"5": 0.6}}, "3": {"ep": {"5": 2.1}}}})
    monkeypatch.setattr(services, "editable_gw", lambda *a, **k: 5)
    monkeypatch.setattr(services, "_model_start_probs", lambda season=None: {1: 0.95, 2: 0.1, 3: 0.8})
    return conn


def test_only_players_who_matter_trigger(news_db):
    out = scheduler.significant_changes(news_db, "s", 2)
    assert [t["name"] for t in out] == ["Star", "Nailed"]
    star = out[0]
    assert star["from"] == "a" and star["to"] == "o"
    assert any("owned" in w for w in star["why"]) and any("projected" in w for w in star["why"])
    assert out[1]["why"] == ["P(start) 0.80"]


@pytest.fixture
def fresh_state(monkeypatch):
    monkeypatch.setattr(scheduler, "_state", {})
    monkeypatch.setattr(scheduler, "_save_state", lambda: None)
    started = []
    monkeypatch.setattr(jobs, "start", lambda kind, fn, owner=None: started.append(kind) or "job-1")
    monkeypatch.setattr(jobs, "running", lambda: [])
    return started


T = {"player_id": 1, "name": "Star", "why": ["35% owned"]}


def test_a_trigger_starts_one_rebuild_then_waits_out_the_cooldown(fresh_state):
    assert scheduler.maybe_reproject([T]) == "job-1"
    assert fresh_state == ["reproject"]
    assert scheduler._state["reproject_last_reason"] == ["Star (35% owned)"]
    # more news straight after: queued, not dropped, not rebuilt
    assert scheduler.maybe_reproject([{**T, "player_id": 2, "name": "Other"}]) is None
    assert [t["name"] for t in scheduler._state["reproject_pending"]] == ["Other"]
    # once the cooldown has passed, the waiting news gets its rebuild
    scheduler._state["reproject_last_run"] -= scheduler.reproject_minutes() * 60 + 1
    assert scheduler.maybe_reproject([]) == "job-1"
    assert scheduler._state["reproject_pending"] == []


def test_no_rebuild_while_another_job_runs(fresh_state, monkeypatch):
    monkeypatch.setattr(jobs, "running", lambda: ["refresh"])
    assert scheduler.maybe_reproject([T]) is None
    assert fresh_state == [] and scheduler._state["reproject_pending"]


def test_nothing_pending_means_nothing_starts(fresh_state):
    assert scheduler.maybe_reproject([]) is None and fresh_state == []


def test_cooldown_has_a_floor(monkeypatch):
    monkeypatch.setenv("FPLABS_REPROJECT_MINUTES", "1")
    assert scheduler.reproject_minutes() == 20.0
    monkeypatch.delenv("FPLABS_REPROJECT_MINUTES")
    assert scheduler.reproject_minutes() == 60.0
