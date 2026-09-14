"""E18: the learned clean-sheet model is point-in-time and off by default."""
import inspect

import numpy as np

from fpl_engine.xpts import cs_model, engine


def _with_codes(conn):
    conn.execute("UPDATE team SET code = team_id * 100")
    conn.commit()


def test_hook_is_off_by_default():
    assert inspect.signature(engine.xpts_predict_gw).parameters["tweaks"].default is None


def test_rows_are_point_in_time_and_carry_the_target(conn):
    _with_codes(conn)
    # GW3 rows see GW1-2 only; the trailing record must not include GW3's own result
    r3 = cs_model.gw_rows(conn, "2024-25", 3)
    assert len(r3) == 2 and set(r3["team_id"]) == {1, 2}
    home = r3[r3["team_id"] == 1].iloc[0]
    assert home["home"] == 1.0
    # club 1 conceded 1 in each of GW1-2 (shrunk toward the league mean of 1.5)
    assert 1.0 < home["own_ga"] < 1.5
    # targets: club 1 conceded 1 in GW3 -> no clean sheet; club 2 conceded 2
    assert (r3["cs"] == 0).all()
    # prediction rows carry no target
    assert cs_model.gw_rows(conn, "2024-25", 3, with_target=False)["cs"].isna().all()
    # the future GW4 row must be excluded from GW4's own features: identical
    # trailing goals against to what GW4 can legally see (GW1-3)
    r4 = cs_model.gw_rows(conn, "2024-25", 4)
    assert abs(r4[r4["team_id"] == 1].iloc[0]["own_ga"] - (3 * 1 + 3 * 1.5) / 6) < 1e-9


def test_offset_model_reduces_to_the_poisson_zero_with_zero_coefficients(conn):
    _with_codes(conn)
    rows = cs_model.season_rows(conn, ["2024-25"])
    m = cs_model.CSModel("offset", ["own_ga", "opp_gf"])
    m.mean = np.zeros(2)
    m.std = np.ones(2)
    m.est = np.zeros(2)
    p = m.predict(rows)
    assert np.allclose(p, np.exp(-rows["lam_blend"].to_numpy()), atol=1e-6)


def test_predict_map_is_keyed_by_club_and_fixture(conn):
    _with_codes(conn)
    m = cs_model.p_clean_sheet_map(conn, "2024-25", 3, "2024-08-03T00:00:00Z", ["2024-25"],
                                   kind="offset", features=["own_ga", "opp_gf"])
    assert set(m) == {(1, 3), (2, 3)}
    assert all(0.0 < v < 1.0 for v in m.values())
