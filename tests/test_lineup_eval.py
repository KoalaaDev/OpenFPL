"""The lineup-feed evaluator (E17): every guard here protects a way the
evaluation could silently lie.

* an empty or missing archive must RAISE, never score as zero signal
  (the E16 process rule — an empty source produced a plausible zero twice)
* the snapshot must use only forecasts observed strictly before the
  deadline, and the LAST one per club
* the stored gw column must not be trusted — rows are reassigned to the
  club's next kickoff after the observation
* the hard/soft arms must move exposure in the promised direction and
  leave uncovered clubs bit-identical
"""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from fpl_engine import lineup_eval as le


def test_missing_archive_raises():
    with pytest.raises(FileNotFoundError):
        le.load_feed("2099-00", os.path.join(tempfile.gettempdir(),
                                             "no-such-lineups.csv"))


def test_empty_archive_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("observed_utc,gw,team_abbr,side,status,position,slot,"
                 "player,rotowire_id\n")
    with pytest.raises(ValueError):
        le.load_feed("2024-25", str(p))


def test_unknown_abbreviation_fails_loud():
    with pytest.raises(ValueError):
        le.resolve_abbr("XXX", {"ALP": 1})


def _feed(rows):
    df = pd.DataFrame(rows, columns=["observed_utc", "gw", "team_abbr",
                                     "side", "status", "position", "slot",
                                     "player", "rotowire_id"])
    df["observed_utc"] = pd.to_datetime(df["observed_utc"], utc=True)
    return df


def test_snapshot_deadline_and_gw_reassignment(conn):
    # conftest: teams ALP(1)/BET(2), season 2024-25, player_gw kickoffs
    # 2024-08-01..03 for gws 1..3. Deadline for gw2 = kick - 90min.
    rows = [
        # stale forecast (gw2, before deadline) - must lose to the later one
        ("2024-08-01T20:00:00Z", 2, "ALP", "home", "predicted", "MID", 1,
         "Mid Player", "1"),
        # the last pre-deadline forecast: labelled gw 9 by the collector,
        # but ALP's next kickoff after it is the gw2 match -> reassigned
        ("2024-08-02T10:00:00Z", 9, "ALP", "home", "predicted", "MID", 1,
         "Mid Player", "1"),
        # post-deadline forecast must be excluded (gw2 kick 14:00, dl 12:30)
        ("2024-08-02T13:00:00Z", 2, "ALP", "home", "predicted", "MID", 1,
         "Someone Else", "2"),
        # confirmed rows are never an input
        ("2024-08-02T11:00:00Z", 2, "BET", "away", "confirmed", "GK", 1,
         "Keep Er", "3"),
    ]
    snap = le.snapshot(_feed(rows), conn, "2024-25", 2,
                       require_full_xi=False)
    assert set(snap.team_abbr) == {"ALP"}
    assert list(snap.player) == ["Mid Player"]
    assert snap.player_id.iloc[0] == 10          # matched by full name
    assert snap.observed_utc.iloc[0].hour == 10  # the LAST pre-deadline one


def test_partial_xi_club_cannot_assert_absences(conn):
    # a club with an unmatched XI name (or fewer than 11 rows) is dropped:
    # the unreadable name would otherwise read as "benched" — the GW3 dry
    # run flagged Alisson as a removed starter for exactly that reason
    rows = [("2024-08-02T10:00:00Z", 2, "ALP", "home", "predicted", "MID", 1,
             "Mid Player", "1")]
    with pytest.raises(ValueError):
        le.snapshot(_feed(rows), conn, "2024-25", 2)


def test_snapshot_empty_coverage_raises(conn):
    rows = [("2024-08-02T13:59:00Z", 2, "ALP", "home", "predicted", "MID", 1,
             "Mid Player", "1")]  # after the deadline
    with pytest.raises(ValueError):
        le.snapshot(_feed(rows), conn, "2024-25", 2)


def _base_frame():
    return pd.DataFrame({
        "player_id": [10, 11, 20],
        "team_id": [1, 1, 2],
        "p_none": [0.1, 0.5, 0.1],
        "p_sub": [0.2, 0.3, 0.2],
        "p_full": [0.7, 0.2, 0.7],
        "p_start": [0.75, 0.25, 0.8],
        "e_min": [70.0, 25.0, 75.0],
        "m_played": [80.0, 40.0, 85.0],
    })


def _snap():
    return pd.DataFrame({"fpl_team_id": [1], "player_id": [11.0]})


def test_hard_frame_overrides_covered_club_only():
    prof = {"start": {"p_none": 0.0, "p_sub": 0.2, "p_full": 0.8,
                      "p_start": 1.0, "e_min": 80.0},
            "bench": {"p_none": 0.6, "p_sub": 0.35, "p_full": 0.05,
                      "p_start": 0.0, "e_min": 12.0}}
    out = le.hard_frame(_base_frame(), _snap(), prof)
    r11 = out[out.player_id == 11].iloc[0]    # in the fed XI
    r10 = out[out.player_id == 10].iloc[0]    # covered club, not in XI
    r20 = out[out.player_id == 20].iloc[0]    # uncovered club
    assert r11.p_start == 1.0 and r11.p_full == 0.8
    assert r10.p_start == 0.0 and r10.p_none == 0.6
    assert r20.p_start == 0.8 and r20.p_full == 0.7   # untouched


def test_soft_frame_moves_odds_the_right_way_and_bounds():
    out = le.soft_frame(_base_frame(), _snap(), 4.0, 4.0)
    r11 = out[out.player_id == 11].iloc[0]
    r10 = out[out.player_id == 10].iloc[0]
    r20 = out[out.player_id == 20].iloc[0]
    assert r11.p_start > 0.25 and r10.p_start < 0.75
    assert r20.p_start == 0.8                          # untouched
    for r in (r11, r10):
        assert 0 <= r.p_full <= 0.97 and 0 <= r.p_sub <= 1
        assert abs(r.p_none + r.p_sub + r.p_full - 1) < 1e-6
    # LR=1 is a no-op
    same = le.soft_frame(_base_frame(), _snap(), 1.0, 1.0)
    assert np.allclose(same.p_start, _base_frame().p_start)


def test_default_lr_when_no_history():
    assert le.fitted_lrs(None) == (le.DEFAULT_LR, le.DEFAULT_LR)
    assert le.fitted_lrs(pd.DataFrame({"started": [True], "feed_start":
                                       [True]})) == (le.DEFAULT_LR,
                                                     le.DEFAULT_LR)
