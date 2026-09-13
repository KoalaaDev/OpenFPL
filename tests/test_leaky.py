"""E16: own-team leakiness features and the defence_leak research hook."""
import inspect

from fpl_engine.xpts import engine, leaky, team_model


def _with_codes(conn):
    """conftest clubs carry no stable code; leakiness joins on it."""
    conn.execute("UPDATE team SET code = team_id * 100")
    conn.commit()
    return conn


def test_leak_features_are_point_in_time_and_order_clubs(conn):
    # conftest: club 1 concedes 1 a match, club 2 concedes 2; GW4 is in the future
    _with_codes(conn)
    f = leaky.team_leak_features(conn, "2024-25", "2024-08-04T00:00:00Z")
    assert set(f.index) == {1, 2}
    assert f.loc[1, "n_matches"] == 3 and f.loc[2, "n_matches"] == 3
    assert f.loc[2, "ga"] > f.loc[1, "ga"]
    assert f.loc[2, "two_plus"] > f.loc[1, "two_plus"]
    # shrinkage pulls both toward the league mean rather than to the raw 1 and 2
    assert 1.0 < f.loc[1, "ga"] < f.loc[2, "ga"] < 2.0
    # before any match there is nothing to say: every club sits at the prior
    f0 = leaky.team_leak_features(conn, "2024-25", "2024-08-01T00:00:00Z")
    assert f0["n_matches"].sum() == 0


def test_leak_factors_scale_around_one_and_zero_alpha_is_off(conn):
    _with_codes(conn)
    as_of = "2024-08-04T00:00:00Z"
    assert leaky.defence_leak_factors(conn, "2024-25", as_of, {"mode": "ga", "alpha": 0}) == {}
    for mode in ("ga", "tail"):
        f = leaky.defence_leak_factors(conn, "2024-25", as_of, {"mode": mode, "alpha": 1.0})
        assert f[2] > 1.0 > f[1], mode          # the leakier club concedes more
    # neither club has kept a clean sheet, so the clean-sheet mode sees no difference
    f = leaky.defence_leak_factors(conn, "2024-25", as_of, {"mode": "cs", "alpha": 1.0})
    assert f[1] == f[2] == 1.0
    weak = leaky.defence_leak_factors(conn, "2024-25", as_of, {"mode": "ga", "alpha": 0.5})
    strong = leaky.defence_leak_factors(conn, "2024-25", as_of, {"mode": "ga", "alpha": 1.0})
    assert 1.0 < weak[2] < strong[2]


def test_defence_leak_hook_is_off_by_default():
    sig = inspect.signature(engine.xpts_predict_gw)
    assert sig.parameters["defence_leak"].default is None
    assert inspect.signature(team_model.fit).parameters["xg_blend_def"].default is None


def test_defence_target_blend_defaults_to_the_shipped_fit(conn):
    """xg_blend_def=None must reproduce the shipped ratings exactly; 0 must
    change only the defence side (the attack target is untouched)."""
    _with_codes(conn)
    as_of = "2024-08-04T00:00:00Z"
    conn.execute("UPDATE team_match SET xg = goals_for * 0.5")   # make xG matter
    conn.commit()
    base = team_model.fit(conn, as_of)
    same = team_model.fit(conn, as_of, xg_blend_def=None)
    assert same.defence == base.defence and same.attack == base.attack
    goals_only = team_model.fit(conn, as_of, xg_blend_def=0.0)
    assert goals_only.defence != base.defence
    # realised goals exceed the blended target, so the leakier club is rated worse
    assert goals_only.defence[200] < base.defence[200]
