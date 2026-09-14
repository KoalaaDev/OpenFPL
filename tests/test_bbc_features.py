"""Round 19 minutes-model blocks built from the BBC archive."""
import numpy as np
import pandas as pd

from fpl_engine.xpts import bbc_role_features as br, calendar_features as cal


def _frame():
    return pd.DataFrame({
        "season": ["2024-25"] * 3, "team_id": [1, 1, 2], "player_code": [100, 100, 200],
        "position": ["MID", "MID", "DEF"],
        "kick": pd.to_datetime(["2024-09-14T14:00:00Z", "2024-09-21T14:00:00Z",
                                "2024-09-21T14:00:00Z"], utc=True),
    })


def test_calendar_features_see_europe_and_are_point_in_time():
    calendar = pd.DataFrame({
        "season": ["2024-25"] * 4, "team_id": [1, 1, 1, 2],
        "kick": pd.to_datetime(["2024-09-14T14:00:00Z", "2024-09-17T19:00:00Z",   # PL, then CL Tue
                                "2024-09-21T14:00:00Z", "2024-09-21T14:00:00Z"], utc=True),
        "europe": [False, True, False, False],
    })
    out = cal.add_features(_frame(), calendar)
    r0, r1, r2 = out.iloc[0], out.iloc[1], out.iloc[2]
    assert r0["cal_days_rest"] == 30.0 and r0["cal_prev7"] == 0          # season opener
    assert r0["cal_next4_europe"] == 1.0 and r0["cal_next7"] == 2.0      # Tue in Europe + next Sat
    assert abs(r1["cal_days_rest"] - 3.79) < 0.05                        # Tue 19:00 -> Sat 14:00
    assert r1["cal_prev4_europe"] == 1.0 and r1["cal_prev7"] == 2.0
    assert r2["cal_days_rest"] == 30.0                                   # club 2's first match
    empty = cal.add_features(_frame(), pd.DataFrame())
    assert empty["cal_days_rest"].isna().all()


def test_bbc_role_features_use_prior_starts_only():
    roles = pd.DataFrame({
        "player_code": [100, 100, 100],
        "kick": pd.to_datetime(["2024-08-31T14:00:00Z", "2024-09-14T14:00:00Z",
                                "2024-09-21T14:00:00Z"], utc=True),
        "pos": [3.0, 2.0, 2.5], "row": [1.0, 0.5, 0.75],
        "is_am": [1.0, 0.0, 0.0], "is_dm": [0.0, 1.0, 0.0],
    })
    out = br.add_features(_frame(), roles)
    r0, r1, r2 = out.iloc[0], out.iloc[1], out.iloc[2]
    # 14 Sep: only the 31 Aug start is strictly prior
    assert r0["bbc_row_l1"] == 1.0 and r0["bbc_is_am"] == 1.0 and r0["bbc_pos_vs_fpl"] == 0.5
    # 21 Sep: the 14 Sep start is the latest prior; the same-day one is excluded
    assert r1["bbc_row_l1"] == 0.5 and r1["bbc_is_dm"] == 1.0 and r1["bbc_row_l5"] == 0.75
    assert np.isnan(r2["bbc_row_l1"])                                    # unmapped player
    assert br.POS_ORD["Attacking Midfielder"] > br.POS_ORD["Defensive Midfielder"]


def test_name_normalisation_strips_accents_and_case():
    assert br._norm("Gabriel Magalhães") == "gabriel magalhaes"
    assert br._norm("  O'Reilly ") == "oreilly"
