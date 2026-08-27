"""Elite-manager panel: selection must not see the season in progress.

The whole value of this dataset rests on one property. If membership depends
on how a manager is doing *now*, then every pick they made this season looks
prescient by construction, and nothing measured against it means anything.
Selection therefore uses past seasons only, and these tests hold that line.
"""
import os
import tempfile

import pytest

from acquire import storage
from acquire.sources import fpl_managers as fm


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    from fpl_engine import db
    db.init_db(path)
    c = storage.connect(path)
    storage.init(c)
    fm.register(c)
    fm.init(c)
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def _past(*ranks):
    return [{"season_name": f"20{20+i}/{21+i}", "total_points": 2000,
             "rank": r} for i, r in enumerate(ranks)]


# ------------------------------------------------------------- selection ---
def test_repeatedly_good_managers_make_the_panel():
    c = fm._classify(_past(40_000, 80_000, 500_000))
    assert c["elite_seasons"] == 2
    assert c["is_panel"] == 1
    assert c["best_rank"] == 40_000


def test_one_good_season_is_not_enough():
    """Once lucky is not proven - the whole point of the threshold."""
    c = fm._classify(_past(50_000, 3_000_000, 4_000_000))
    assert c["elite_seasons"] == 1
    assert c["is_panel"] == 0


def test_a_manager_with_no_history_is_excluded():
    """15% of the sampled top 250 were brand-new accounts."""
    c = fm._classify([])
    assert c["is_panel"] == 0
    assert c["best_rank"] is None


def test_a_typical_top_250_manager_is_excluded():
    """Sampled after one gameweek, the median member of the overall top 250
    had a career median rank of 2.6 million."""
    c = fm._classify(_past(2_400_000, 2_600_000, 2_900_000))
    assert c["is_panel"] == 0


def test_current_rank_is_recorded_but_never_selects(conn):
    """discovered_rank is provenance, not a criterion."""
    now = "2026-08-27T12:00:00Z"
    rows = [
        # rank 1 today, but a poor history -> out
        (111, now, fm.SOURCE_ID, 3, 900_000, 2_500_000, 0, 0, 1),
        # rank 4000 today, proven over years -> in
        (222, now, fm.SOURCE_ID, 4, 12_000, 45_000, 3, 1, 4000),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO acq_manager (entry_id, observed_utc, source_id,"
        " seasons_played, best_rank, median_rank, elite_seasons, is_panel,"
        " discovered_rank) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    assert fm.panel_entries(conn) == [222]


# ------------------------------------------------------------ pick storage ---
def _pick(entry, player, mult=1, cap=0, gw=1):
    return ("2026-27", gw, entry, player, 1, mult, cap, 0, None,
            "2026-08-27T12:00:00Z", fm.SOURCE_ID)


def test_ownership_counts_separate_owning_starting_and_captaining(conn):
    rows = [_pick(1, 100, mult=2, cap=1), _pick(1, 200, mult=0),
            _pick(2, 100, mult=1), _pick(2, 200, mult=1),
            _pick(3, 100, mult=1), _pick(3, 300, mult=0)]
    conn.executemany(
        "INSERT OR REPLACE INTO acq_manager_pick (season, gw, entry_id, "
        "player_id, slot, multiplier, is_captain, is_vice, active_chip, "
        "observed_utc, source_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    own = {r["player_id"]: r for r in fm.panel_ownership(conn, "2026-27", 1)}
    assert own[100]["owned"] == 3 and own[100]["started"] == 3
    assert own[100]["captained"] == 1
    # owned by two, benched by one: owning and starting are not the same thing
    assert own[200]["owned"] == 2 and own[200]["started"] == 1
    assert own[100]["panel_size"] == 3


def test_picks_are_idempotent(conn):
    for _ in range(3):
        conn.executemany(
            "INSERT OR REPLACE INTO acq_manager_pick (season, gw, entry_id, "
            "player_id, slot, multiplier, is_captain, is_vice, active_chip, "
            "observed_utc, source_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [_pick(1, 100)])
        conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM acq_manager_pick").fetchone()[0]
    assert n == 1


def test_ownership_is_raw_counts_not_a_feature(conn):
    """Layer discipline: shares against overall ownership are the model's job."""
    conn.executemany(
        "INSERT OR REPLACE INTO acq_manager_pick (season, gw, entry_id, "
        "player_id, slot, multiplier, is_captain, is_vice, active_chip, "
        "observed_utc, source_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [_pick(1, 100)])
    conn.commit()
    cols = fm.panel_ownership(conn, "2026-27", 1)[0].keys()
    assert set(cols) == {"player_id", "owned", "started", "captained",
                         "panel_size"}


def test_thresholds_are_what_the_measurement_justified():
    assert fm.ELITE_RANK == 100_000
    assert fm.ELITE_SEASONS >= 2
