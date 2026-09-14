"""The Round-17 research hooks: off by default, and each does what it says."""
import math

import numpy as np
import pandas as pd

from fpl_engine.xpts import engine


def test_variants_are_off_unless_asked(monkeypatch):
    monkeypatch.delenv("FPL_XPTS_VARIANT", raising=False)
    assert engine.variants() == set()
    monkeypatch.setenv("FPL_XPTS_VARIANT", "budget, nb_cs")
    assert engine.variants() == {"budget", "nb_cs"}


def test_negbin_zero_probability_exceeds_poisson_and_keeps_the_limit():
    for lam in (0.4, 1.0, 1.4, 2.5):
        p = engine.p_zero_goals(lam, negbin=False)
        q = engine.p_zero_goals(lam, negbin=True)
        assert math.isclose(p, math.exp(-lam))
        assert q > p                                  # heavier tail at zero
        assert q - p < 0.03                            # but only slightly
    assert engine.p_zero_goals(0.0, True) == 1.0


def test_budget_renormalisation_conserves_eleven_starters():
    df = pd.DataFrame({
        "player_id": range(6), "team_id": [1] * 3 + [2] * 3,
        "p_start": [0.9, 0.5, 0.4, 0.95, 0.95, 0.95],      # 1.8 and 2.85
        "p_full": [0.8, 0.4, 0.3, 0.9, 0.9, 0.9],
        "p_sub": [0.1, 0.4, 0.3, 0.05, 0.05, 0.05],
        "e_min": [75, 40, 30, 85, 85, 85],
    })
    out = engine.apply_budget(df, "start")
    # both clubs are far short of 11, so both hit the clip, the same way
    f = engine.BUDGET_CLIP[1]
    assert np.allclose(out["e_min"], np.minimum(90.0, df["e_min"] * f))
    assert (out["p_start"] <= 1.0).all() and (out["p_full"] + out["p_sub"] <= 1.0 + 1e-9).all()
    # a club already summing to 11 is untouched
    full = pd.DataFrame({"player_id": range(11), "team_id": 7,
                         "p_start": 1.0, "p_full": 0.9, "p_sub": 0.05, "e_min": 85.0})
    same = engine.apply_budget(full, "start")
    assert np.allclose(same["e_min"], 85.0) and np.allclose(same["p_full"], 0.9)


def test_rates_put_xg_and_xa_in_position_units(conn, monkeypatch):
    """A defender who never scores from steady xG is rated below his raw xG;
    the legacy estimator takes the xG at face value."""
    from fpl_engine import db
    from fpl_engine.xpts import rates
    rows = []
    for gw in range(1, 9):
        rows.append({"season": "2024-25", "gw": gw, "source": "vaastav",
                     "player_id": 30, "fixture_id": 100 + gw, "player_code": 3000,
                     "full_name": "Def Ender", "team_id": 1, "opponent_id": 2,
                     "was_home": 1, "kickoff_utc": f"2024-08-{10 + gw:02d}T14:00:00Z",
                     "minutes": 90, "total_points": 2, "goals_scored": 0, "assists": 1,
                     "clean_sheets": 0, "goals_conceded": 1, "own_goals": 0,
                     "penalties_saved": 0, "penalties_missed": 0, "yellow_cards": 0,
                     "red_cards": 0, "saves": 0, "bonus": 0, "bps": 10, "starts": 1,
                     "xg": 0.5, "xa": 0.2})
    db.upsert(conn, "player", [{"season": "2024-25", "player_id": 30, "code": 3000,
                                "web_name": "Def", "full_name": "Def Ender",
                                "team_id": 1, "position": "DEF", "understat_id": None}])
    db.upsert(conn, "player_gw", rows)
    conn.commit()
    monkeypatch.delenv("FPL_XPTS_VARIANT", raising=False)
    new = rates.fit(conn, "2024-25", "2024-09-01T00:00:00Z").set_index("player_id")
    monkeypatch.setenv("FPL_XPTS_VARIANT", "legacy_rates")
    old = rates.fit(conn, "2024-25", "2024-09-01T00:00:00Z").set_index("player_id")
    assert new.loc[30, "xg90"] < old.loc[30, "xg90"]        # 0 goals from 4.0 xG
    assert new.loc[30, "xa90"] > old.loc[30, "xa90"]        # 8 assists from 1.6 xA
