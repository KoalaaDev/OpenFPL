"""The invariants have to actually fire.

Every identity defect this project has shipped was silent — the pipeline kept
running and produced NaNs, or plausible numbers. So each check is tested by
breaking the data on purpose and asserting that it is caught.
"""
import os
import tempfile

import pytest

from fpl_engine import db, verify

SEASON = "2025-26"


def _player(pid, code, uid=None, pos="MID", cost=7.0, name=None):
    return {"season": SEASON, "player_id": pid, "code": code,
            "web_name": name or f"P{pid}", "full_name": f"Player {pid}",
            "team_id": 1, "position": pos, "understat_id": uid,
            "now_cost": cost}


def _gw(pid, gw=1, price=70.0, minutes=90.0, kick="2025-09-01T14:00:00Z"):
    return {"season": SEASON, "gw": gw, "source": "fpl", "player_id": pid,
            "fixture_id": gw, "player_code": 100 + pid, "team_id": 1,
            "opponent_id": 2, "was_home": 1, "kickoff_utc": kick,
            "minutes": minutes, "total_points": 5, "price": price}


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db.init_db(path)
    c = db.connect(path)
    db.upsert(c, "player", [_player(1, 101, "11"), _player(2, 102, "22")])
    db.upsert(c, "player_gw", [_gw(1), _gw(2)])
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def _checks(conn, level=None):
    r = verify.run(conn)
    src = r.errors if level == "error" else r.findings
    return {f.check for f in src}


def test_clean_data_passes(conn):
    r = verify.run(conn)
    assert r.ok(), [f.check for f in r.errors]


def test_two_players_sharing_an_understat_id_is_an_error(conn):
    """Gabriel and Gabriel Jesus, in every season, undetected."""
    conn.execute("UPDATE player SET understat_id='11' WHERE player_id=2")
    conn.commit()
    assert "identity.one_understat_per_player" in _checks(conn, "error")


def test_a_player_whose_understat_id_changes_between_seasons_is_an_error(conn):
    db.upsert(conn, "player", [{**_player(1, 101, "99"), "season": "2024-25"}])
    conn.commit()
    assert "identity.stable_across_seasons" in _checks(conn, "error")


def test_an_understat_id_claimed_by_two_fpl_codes_is_an_error(conn):
    db.upsert(conn, "player", [{**_player(3, 103, "11"), "season": "2024-25"}])
    conn.commit()
    assert "identity.one_player_per_understat" in _checks(conn, "error")


def test_orphan_player_gw_is_an_error(conn):
    db.upsert(conn, "player_gw", [_gw(99)])
    conn.commit()
    assert "refs.player_gw_has_player" in _checks(conn, "error")


def test_price_in_the_wrong_unit_is_an_error(conn):
    """player_gw.price is tenths; writing £m into it reads as a 0.7m player."""
    conn.execute("UPDATE player_gw SET price=7.0 WHERE player_id=1")
    conn.commit()
    assert "units.player_gw_price_is_tenths" in _checks(conn, "error")


def test_now_cost_in_the_wrong_unit_is_an_error(conn):
    conn.execute("UPDATE player SET now_cost=70.0 WHERE player_id=1")
    conn.commit()
    assert "units.now_cost_is_millions" in _checks(conn, "error")


def test_a_future_match_that_already_has_minutes_is_an_error(conn):
    db.upsert(conn, "player_gw", [_gw(1, gw=38, kick="2099-01-01T12:00:00Z")])
    conn.commit()
    assert "pit.no_future_outcomes" in _checks(conn, "error")


def test_a_played_match_with_no_kickoff_is_an_error(conn):
    conn.execute("UPDATE player_gw SET kickoff_utc=NULL WHERE player_id=1")
    conn.commit()
    assert "pit.kickoff_present" in _checks(conn, "error")


def test_impossible_minutes_are_an_error(conn):
    conn.execute("UPDATE player_gw SET minutes=200 WHERE player_id=1")
    conn.commit()
    assert "values.minutes" in _checks(conn, "error")


def test_assistant_managers_are_reported_but_do_not_fail(conn):
    """They are a real, documented part of the feed — the requirement is that
    they are visible, and that models exclude them."""
    db.upsert(conn, "player", [_player(9, 109, None, pos=None, cost=None,
                                       name="Arteta")])
    conn.commit()
    r = verify.run(conn)
    assert r.ok()
    assert "nonplayers.present" in {f.check for f in r.warnings}


def test_a_non_player_price_does_not_trip_the_unit_check(conn):
    """An Assistant Manager legitimately costs £1.5m, below any player's floor."""
    db.upsert(conn, "player", [_player(9, 109, None, pos=None, cost=None)])
    db.upsert(conn, "player_gw", [_gw(9, price=15.0, minutes=0.0)])
    conn.commit()
    r = verify.run(conn)
    assert "units.player_gw_price_is_tenths" not in {f.check for f in r.errors}
