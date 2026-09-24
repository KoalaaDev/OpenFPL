"""Round 18: press-conference text -> availability observations (rules pass)."""
import pandas as pd

from fpl_engine import pressers


def test_fixture_label_parsing_maps_bbc_club_names_to_fpl():
    # clubs resolve against the SEASON's own names — a hardcoded table went
    # stale the summer FPL renamed Ipswich "Ipswich Town" (test_presser_rules)
    idx = pressers.club_index(["Spurs", "Everton", "Nott'm Forest", "Man Utd"])
    assert pressers.fixture_clubs("Tottenham v Everton (Sat, 17:30 BST)", idx) == ("Spurs", "Everton")
    assert pressers.fixture_clubs("Nottingham Forest vs Man Utd", idx) == ("Nott'm Forest", "Man Utd")
    assert pressers.fixture_clubs("Wolves 1-0 Burnley", idx) is None     # a result, not a fixture
    assert pressers.fixture_clubs("Today's papers", idx) is None


def test_classification_precedence_and_phrases():
    assert pressers.classify("He [Nathan Collins] won't be involved this side of the break.")[0] == "out"
    assert pressers.classify("James Maddison is available to face Everton.")[0] == "available"
    assert pressers.classify("Saka will be assessed tomorrow.")[0] == "doubt"
    assert pressers.classify("Caicedo is not available for selection.")[0] == "out"   # out beats available
    assert pressers.classify("He will be rested for this one.")[0] == "rested"
    assert pressers.classify("Great atmosphere expected at the stadium.") is None


def test_mentions_resolve_only_unique_surnames():
    players = [
        {"player_id": 1, "team_id": 1, "web_name": "Collins", "full_name": "Nathan Collins"},
        {"player_id": 2, "team_id": 2, "web_name": "N.Collins", "full_name": "Neil Collins"},
        {"player_id": 3, "team_id": 1, "web_name": "Thiago", "full_name": "Igor Thiago"},
    ]
    pats = pressers._name_patterns(players)
    out = pressers.extract_post("Collins won't be involved. Igor Thiago is available.", pats)
    got = {(p["player_id"], cls) for p, cls, _, _ in out}
    assert (3, "available") in got
    assert not any(pid in (1, 2) for pid, _ in got)     # 'Collins' is ambiguous: skipped, never guessed


def test_exposure_factors_take_the_most_severe_latest_class():
    obs = pd.DataFrame({
        "player_id": [7, 7, 9, 11],
        "cls": ["available", "out", "doubt", "unknown"],
        "published_utc": ["2026-09-11T08:00:00Z", "2026-09-11T09:00:00Z",
                          "2026-09-11T10:00:00Z", "2026-09-11T10:00:00Z"],
    })
    f = pressers.exposure_factors(obs, {"out": 0.5, "doubt": 0.8, "available": 1.0})
    assert f == {7: 0.5, 9: 0.8, 11: 1.0}
    assert pressers.exposure_factors(pd.DataFrame()) == {}


def test_gw_after_picks_the_next_kickoff():
    ks = [(3, "2026-09-05T11:30:00Z"), (4, "2026-09-12T14:00:00Z"), (5, "2026-09-19T14:00:00Z")]
    assert pressers.gw_after(ks, "2026-09-11T10:00:00Z") == 4
    assert pressers.gw_after(ks, "2026-09-12T15:00:00Z") == 5
    assert pressers.gw_after(ks, "2026-09-30T00:00:00Z") is None
    assert pressers.season_of("2026-09-11") == "2026-27" and pressers.season_of("2026-05-01") == "2025-26"


def test_live_overlay_is_point_in_time_and_conservative(conn):
    """A Friday statement is attached to Saturday's gameweek; one made after
    the cutoff does not count; a missing table -> empty."""
    import sqlite3
    conn.executescript(pressers.SCHEMA)
    rows = [
        ("2024-25", 4, 10, 1, "out", "presser", "2024-08-03T12:00:00Z", "ruled out", "…", "pg", "u1"),
        ("2024-25", 4, 20, 2, "doubt", "presser", "2024-08-03T13:00:00Z", "assessed", "…", "pg", "u2"),
        ("2024-25", 4, 20, 2, "out", "presser", "2024-08-04T15:00:00Z", "ruled out", "…", "pg", "u3"),  # too late
    ]
    conn.executemany("INSERT INTO presser_obs VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    ov = pressers.live_overlay(conn, "2024-25", [4, 5], lambda g: "2024-08-04T14:00:00Z")
    # shown, not modelled: the class is carried, the factor is 1.0 for every class
    assert ov[4][10]["factor"] == 1.0 and ov[4][10]["cls"] == "out"
    assert ov[4][20]["factor"] == 1.0 and ov[4][20]["cls"] == "doubt"    # the late 'out' is excluded
    assert all(v == 1.0 for v in pressers.LIVE_FACTORS.values())
    assert 5 not in ov
    fresh = sqlite3.connect(":memory:")
    assert pressers.live_overlay(fresh, "2024-25", [4], lambda g: "x") == {}
