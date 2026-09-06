"""The live FPL ingest only writes results, never matches in progress.

FPL's element-summary lists a fixture in ``history`` the moment it goes live,
with zero minutes. A pull during kickoff therefore wrote a phantom 0-minute
appearance for every player of both clubs, which the minutes model read as
"did not play" in his trailing window. The boundary is FPL's own
``finished_provisional`` (full time); ``finished`` waits for bonus and has been
observed lagging by days.
"""
import os
import tempfile

import pytest

from fpl_engine import db
from fpl_engine.ingest import fpl_api

SEASON = "2026-27"


def test_done_means_finished_or_provisionally_finished():
    assert fpl_api._done({"finished": True, "finished_provisional": True})
    assert fpl_api._done({"finished": False, "finished_provisional": True})
    assert not fpl_api._done({"finished": False, "finished_provisional": False,
                              "started": True, "minutes": 3})
    assert not fpl_api._done({})


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db.init_db(path)
    c = db.connect(path)
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def test_history_rows_are_written_only_for_finished_fixtures(conn, monkeypatch):
    boot = {"elements": [{"id": 7, "code": 1007, "first_name": "Bukayo",
                          "second_name": "Saka"}],
            "events": [{"id": 1, "finished": True}]}
    fixtures = [
        {"id": 1, "event": 1, "kickoff_time": "2026-08-21T19:00:00Z",
         "team_h": 1, "team_a": 2, "team_h_score": 2, "team_a_score": 0,
         "finished": True, "finished_provisional": True},
        {"id": 2, "event": 2, "kickoff_time": "2026-08-28T19:00:00Z",
         "team_h": 1, "team_a": 3, "team_h_score": 1, "team_a_score": 1,
         "finished": False, "finished_provisional": True},     # full time
        {"id": 3, "event": 3, "kickoff_time": "2026-09-04T19:00:00Z",
         "team_h": 1, "team_a": 4, "team_h_score": 0, "team_a_score": 0,
         "finished": False, "finished_provisional": False,
         "started": True, "minutes": 4},                       # in play
    ]

    def hist(fixture, minutes):
        return {"element": 7, "fixture": fixture, "opponent_team": 2,
                "round": fixture, "kickoff_time": "2026-08-21T19:00:00Z",
                "was_home": True, "minutes": minutes, "total_points": 2,
                "starts": 1 if minutes else 0, "value": 100}

    summary = {"history": [hist(1, 90), hist(2, 88), hist(3, 0)]}
    monkeypatch.setattr(fpl_api, "fetch_fixtures", lambda use_cache=False: fixtures)
    monkeypatch.setattr(fpl_api, "fetch_element_summary",
                        lambda pid, use_cache=True: summary)
    monkeypatch.setattr(fpl_api, "_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(fpl_api, "_update_team_xg", lambda *a, **k: None)

    n = fpl_api.ingest_current_season_history(conn, SEASON, boot=boot,
                                              use_cache=False)
    rows = conn.execute("SELECT fixture_id, minutes FROM player_gw WHERE season=? "
                        "ORDER BY fixture_id", (SEASON,)).fetchall()
    assert [(r[0], r[1]) for r in rows] == [(1, 90), (2, 88)]
    assert n == 2
