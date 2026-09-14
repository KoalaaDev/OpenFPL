"""Round 21: the selection block builds strictly from prior rows."""
import numpy as np
import pandas as pd

from fpl_engine.xpts import selection_features as sel


def _data():
    kicks = {1: "2025-08-16T14:00:00Z", 2: "2025-08-23T14:00:00Z", 3: "2025-08-30T14:00:00Z",
             4: "2025-09-13T14:00:00Z"}
    pg = pd.DataFrame({
        "season": ["2025-26"] * 8,
        "player_code": [100, 100, 100, 100, 200, 200, 200, 200],
        "fixture_id": [1, 2, 3, 4, 1, 2, 3, 4],
        "total_points": [2, 9, 1, 0, 6, 1, 2, 0], "goals_scored": [0, 1, 0, 0, 1, 0, 0, 0],
        "assists": [0, 1, 0, 0, 0, 0, 0, 0], "yellow_cards": [1, 1, 0, 0, 0, 0, 1, 0],
        "starts": [1, 1, 0, 0, 1, 1, 1, 0], "minutes": [90, 90, 0, 0, 90, 70, 90, 0],
    })
    tm = pd.DataFrame({"season": ["2025-26"] * 4, "team_id": [1] * 4, "fixture_id": [1, 2, 3, 4],
                       "kickoff_utc": [kicks[i] for i in (1, 2, 3, 4)],
                       "goals_for": [2, 0, 3, 1], "goals_against": [0, 2, 3, 1]})
    fx = pd.DataFrame(columns=["season", "fixture_id", "kickoff_utc", "team_h", "team_a"])
    # player 100 on the bench unused in fixture 3, not in the squad for 4
    squad = pd.DataFrame({"season": ["2025-26"] * 3, "player_code": [100, 200, 200],
                          "fixture_id": [3, 3, 4], "is_starter": [0, 1, 0], "mins": [0, 90, 0]})
    # player 200 known absent for fixture 4 (returns next)
    absent = pd.DataFrame({"season": ["2025-26"], "team_id": [1], "player_code": [200],
                           "fixture_id": [4], "kind": ["inj"]})
    return {"pg": pg, "tm": tm, "fx": fx, "squad": squad, "absent": absent}


def _frame():
    kicks = ["2025-08-16T14:00:00Z", "2025-08-23T14:00:00Z", "2025-08-30T14:00:00Z", "2025-09-13T14:00:00Z"]
    rows = []
    for code, starts, mins in ((100, [1, 1, 0, 0], [90, 90, 0, 0]), (200, [1, 1, 1, 0], [90, 70, 90, 0])):
        for i in range(4):
            rows.append({"season": "2025-26", "gw": i + 1, "player_code": code, "player_id": code,
                         "team_id": 1, "fixture_id": i + 1, "kick": pd.Timestamp(kicks[i]),
                         "started": starts[i], "minutes": mins[i], "was_home": 1 - (i % 2),
                         "position": "MID", "consec_starts": 0, "team_matches_14d": 1})
    return pd.DataFrame(rows)


def test_previous_match_status_and_returns_are_prior_only():
    out = sel.add_features(_frame(), _data())
    p100 = out[out.player_code == 100].set_index("fixture_id")
    # fixture 4 for player 100: last match (3) he was an unused sub, 1 point, no involvement
    assert p100.loc[4, "prev_unused"] == 1.0 and p100.loc[4, "prev_absent"] == 0.0
    assert p100.loc[4, "pts_l1"] == 1 and p100.loc[4, "gi_l1"] == 0
    # fixture 3: last match (2) he scored and assisted -> gi_l1 == 2, pts_l1 == 9
    assert p100.loc[3, "gi_l1"] == 2 and p100.loc[3, "pts_l1"] == 9
    # first row has no previous match
    assert np.isnan(p100.loc[1, "pts_l1"])
    # yellows to date count bookings strictly before the match
    assert p100.loc[3, "yellows_todate"] == 2 and p100.loc[1, "yellows_todate"] == 0


def test_team_last_result_and_returning_regular():
    out = sel.add_features(_frame(), _data())
    p100 = out[out.player_code == 100].set_index("fixture_id")
    # the club's previous match before fixture 3 was fixture 2: lost 0-2
    assert p100.loc[3, "team_ga_l1"] == 2 and p100.loc[3, "team_lost_l1"] == 1.0 and p100.loc[3, "team_cs_l1"] == 0.0
    assert p100.loc[2, "team_cs_l1"] == 1.0
    # player 200 was absent for fixture 4 only; his return would be the club's
    # 5th match, which is not in the frame, so nothing flags inside it
    assert out["self_returning"].sum() == 0
    assert set(sel.CORE) <= set(sel.FEATURES)


def test_empty_archives_give_all_nan_or_zero_not_errors():
    d = _data()
    d["squad"] = d["squad"].iloc[0:0]
    d["absent"] = d["absent"].iloc[0:0]
    out = sel.add_features(_frame(), d)
    assert out["prev_unused"].isna().all()          # no bench archive for the season
    assert (out["self_returning"] == 0).all()
