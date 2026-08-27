"""Penalty duty is a correction toward today's order, never a flat bonus.

The property that matters most is the one that keeps it safe: it is zero
whenever the published duty already agrees with the history, and zero outright
in a replayed season, where FPL's order field is NULL for everyone and would
otherwise read as "nobody takes penalties".
"""
import os
import tempfile

import pytest

from fpl_engine import db
from fpl_engine.xpts import set_pieces as sp

SEASON = "2026-27"


def _shot(sid, uid, date, situation="Penalty"):
    return {"shot_id": str(sid), "season": SEASON, "understat_id": str(uid),
            "understat_match_id": "1", "match_date": date, "minute": 50.0,
            "situation": situation, "shot_type": "RightFoot", "result": "Goal",
            "xg": 0.76, "assisted_name": None, "last_action": "Standard",
            "x": 0.88, "y": 0.5}


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db.init_db(path)
    c = db.connect(path)
    db.upsert(c, "player", [
        # took every recent penalty AND is still first choice -> no change
        {"season": SEASON, "player_id": 1, "code": 1, "full_name": "Incumbent",
         "team_id": 1, "position": "MID", "understat_id": "11",
         "penalties_order": 1},
        # newly handed the duty, no penalty history -> boosted
        {"season": SEASON, "player_id": 2, "code": 2, "full_name": "Newcomer",
         "team_id": 1, "position": "MID", "understat_id": "22",
         "penalties_order": None},
        # took them last season, no longer on the list -> docked
        {"season": SEASON, "player_id": 3, "code": 3, "full_name": "Deposed",
         "team_id": 2, "position": "FWD", "understat_id": "33",
         "penalties_order": None},
        {"season": SEASON, "player_id": 4, "code": 4, "full_name": "Successor",
         "team_id": 2, "position": "MID", "understat_id": "44",
         "penalties_order": 1},
    ])
    rows = []
    for i, d in enumerate(("2026-08-01", "2026-08-08", "2026-08-15",
                           "2026-08-20", "2026-08-22")):
        rows.append(_shot(100 + i, 11, d))       # team 1: player 1 takes them
        rows.append(_shot(200 + i, 33, d))       # team 2: player 3 took them
    db.upsert(c, "understat_shot", rows)
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


AS_OF = "2026-08-25T12:00:00Z"


def test_incumbent_is_barely_moved(conn):
    """History and published duty agree, so the correction is small."""
    d = sp.duty(conn, SEASON, AS_OF).set_index("player_id")
    assert d.loc[1, "order_share"] == pytest.approx(sp.ORDER_SHARE[1])
    assert d.loc[1, "hist_share"] > 0.5
    assert abs(d.loc[1, "pen_xg90_delta"]) < 0.02


def test_a_deposed_taker_is_docked(conn):
    """His trailing xG still carries penalties he will not take again."""
    d = sp.duty(conn, SEASON, AS_OF).set_index("player_id")
    assert d.loc[3, "order_share"] == 0.0
    assert d.loc[3, "hist_share"] > 0.3
    assert d.loc[3, "pen_xg90_delta"] < 0


def test_a_new_taker_is_credited(conn):
    d = sp.duty(conn, SEASON, AS_OF).set_index("player_id")
    assert d.loc[4, "hist_share"] == pytest.approx(0.0)
    assert d.loc[4, "pen_xg90_delta"] > 0
    # and the size is the measured one, not a chosen one
    assert d.loc[4, "pen_xg90_delta"] == pytest.approx(
        sp.TEAM_PEN_RATE * sp.ORDER_SHARE[1] * sp.PEN_XG, rel=1e-6)


def test_a_replayed_season_gets_no_correction(conn):
    """FPL publishes the order for the current season only. In a backtest it
    is NULL for everyone, which must not read as "nobody is on duty"."""
    conn.execute("UPDATE player SET penalties_order=NULL WHERE season=?", (SEASON,))
    conn.commit()
    d = sp.duty(conn, SEASON, AS_OF)
    assert (d["pen_xg90_delta"] == 0).all()


def test_no_shot_history_means_no_correction(conn):
    conn.execute("DELETE FROM understat_shot")
    conn.commit()
    d = sp.duty(conn, SEASON, AS_OF)
    assert (d["pen_xg90_delta"] == 0).all()


def test_correction_is_bounded_by_the_duty_it_can_move(conn):
    """It can never move a player by more than a full club's penalty load."""
    d = sp.duty(conn, SEASON, AS_OF)
    cap = sp.TEAM_PEN_RATE * sp.PEN_XG
    assert d["pen_xg90_delta"].abs().max() <= cap + 1e-9


def test_shares_are_a_plausible_split_of_one_clubs_penalties():
    total = sum(sp.ORDER_SHARE.values())
    assert 0.9 < total < 1.0          # the tail is taken by others
    assert sp.ORDER_SHARE[1] > sp.ORDER_SHARE[2] > sp.ORDER_SHARE[3]
