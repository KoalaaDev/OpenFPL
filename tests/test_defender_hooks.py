"""E17: the defender research hooks are off by default and do what they say."""
import inspect

from fpl_engine import scoring
from fpl_engine.xpts import engine, rates


def _with_codes(conn):
    conn.execute("UPDATE team SET code = team_id * 100")
    conn.commit()


def test_hooks_are_off_on_every_shipped_path():
    sig = inspect.signature(engine.xpts_predict_gw)
    assert sig.parameters["tweaks"].default is None
    rsig = inspect.signature(rates.fit)
    assert rsig.parameters["bonus_defcon"].default is False
    assert rsig.parameters["calibrate_by_pos"].default is False
    assert rsig.parameters["xa_blend"].default is None
    assert rsig.parameters["k_by_pos"].default is None


def test_rate_options_change_only_what_they_name(conn):
    """conftest: one MID with three played matches, one GK with none."""
    _with_codes(conn)
    conn.execute("UPDATE player_gw SET xg = 0.5, xa = 0.2, assists = 1")
    # a second MID with a much lower xG, so the position prior differs from
    # either player's own rate and shrinkage has somewhere to pull
    conn.execute("INSERT INTO player (season, player_id, code, web_name, full_name, "
                 "team_id, position) VALUES ('2024-25', 11, 1100, 'Low', 'Low Mid', 1, 'MID')")
    for gw in (1, 2, 3):
        conn.execute("INSERT INTO player_gw (season, gw, source, player_id, fixture_id, "
                     "player_code, team_id, opponent_id, was_home, kickoff_utc, minutes, "
                     "total_points, goals_scored, assists, xg, xa) VALUES "
                     "('2024-25', ?, 'vaastav', 11, ?, 1100, 1, 2, 1, ?, 90, 2, 0, 0, 0.05, 0.05)",
                     (gw, gw, f"2024-08-0{gw}T14:00:00Z"))
    conn.commit()
    as_of, rules = "2024-08-04T00:00:00Z", scoring.load_rules()
    base = rates.fit(conn, "2024-25", as_of, rules=rules)
    same = rates.fit(conn, "2024-25", as_of, rules=rules, bonus_defcon=False,
                     xa_blend=None, k_by_pos=None, calibrate_by_pos=False)
    assert same.equals(base)
    mid = lambda df: df.loc[df["player_id"] == 10].iloc[0]     # noqa: E731

    # calibration: the MID scores 1 goal on 0.5 xG each match, so xG90 rises
    cal = rates.fit(conn, "2024-25", as_of, rules=rules, calibrate_by_pos=True)
    assert mid(cal)["xg90"] > mid(base)["xg90"]
    assert mid(cal)["saves90"] == mid(base)["saves90"]        # untouched

    # xA blend: pure realised assists (1 a match) beat 0.2 xA, so xA90 rises
    xa = rates.fit(conn, "2024-25", as_of, rules=rules, xa_blend={"MID": 0.0})
    assert mid(xa)["xa90"] > mid(base)["xa90"]
    assert mid(xa)["xg90"] == mid(base)["xg90"]

    # heavier shrinkage for MID pulls his xG90 toward the (MID-only) prior
    # and leaves the GK exactly where he was
    k = rates.fit(conn, "2024-25", as_of, rules=rules, k_by_pos={"MID": 60.0})
    gk = lambda df: df.loc[df["position"] == "GK"].iloc[0]    # noqa: E731
    assert gk(k)["xg90"] == gk(base)["xg90"]
    assert mid(k)["xg90"] != mid(base)["xg90"]

    # DefCon in the bonus fit: five coefficients where counts exist
    conn.execute("UPDATE player_gw SET defcon = 12")
    conn.commit()
    dc = rates.fit(conn, "2024-25", as_of, rules=rules, bonus_defcon=True)
    assert all(len(c) == 5 for c in dc.attrs["bonus_coef"].values())
    assert all(len(c) == 4 for c in base.attrs["bonus_coef"].values())
