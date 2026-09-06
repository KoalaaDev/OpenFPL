"""Pricing the lineup feed: the point-in-time boundaries and the arithmetic.

E8b says a lineup source is judged on its accuracy in the band where the
model is unsure, and that the value is linear in that accuracy. These tests
pin the two things that would silently corrupt that number: which forecast
counts as pre-deadline, and whether a name reaches the right player.
"""
import pandas as pd
import pytest

from fpl_engine import lineup_feed as lf


def _players():
    return pd.DataFrame([
        {"player_id": 1, "web_name": "Ødegaard", "full_name": "Martin Ødegaard",
         "team_id": 1, "short_name": "ARS"},
        {"player_id": 2, "web_name": "Saka", "full_name": "Bukayo Saka",
         "team_id": 1, "short_name": "ARS"},
        {"player_id": 3, "web_name": "Groß", "full_name": "Pascal Groß",
         "team_id": 2, "short_name": "BHA"},
        {"player_id": 4, "web_name": "Gibbs-White",
         "full_name": "Morgan Gibbs-White", "team_id": 3, "short_name": "NFO"},
        {"player_id": 5, "web_name": "Danso", "full_name": "Kevin Danso",
         "team_id": 4, "short_name": "TOT"},
        {"player_id": 6, "web_name": "J.Timber", "full_name": "Jurriën Timber",
         "team_id": 1, "short_name": "ARS"},
    ])


def test_names_resolve_across_accents_hyphens_and_rotowire_abbreviations():
    resolved, unresolved, mismatched = lf.resolve(
        {"ARS": {"Martin Odegaard", "Bukayo Saka", "Jurrien Timber"},
         "BHA": {"Pascal Gross"},
         "NOT": {"Morgan Gibbs-White"}},          # RotoWire says NOT, FPL NFO
        _players())
    assert resolved[("ARS", "Martin Odegaard")] == 1
    assert resolved[("ARS", "Jurrien Timber")] == 6
    assert resolved[("BHA", "Pascal Gross")] == 3
    assert resolved[("NOT", "Morgan Gibbs-White")] == 4
    assert unresolved == [] and mismatched == []


def test_a_player_the_local_table_still_has_at_his_old_club_is_flagged_not_dropped():
    # the DB lags a completed transfer: found league-wide, reported as a mismatch
    resolved, unresolved, mismatched = lf.resolve({"SUN": {"Kevin Danso"}},
                                                  _players())
    assert resolved[("SUN", "Kevin Danso")] == 5
    assert mismatched == [("SUN", "Kevin Danso", "TOT")]
    # and an unknown name is reported, never silently skipped
    _, unresolved, _ = lf.resolve({"ARS": {"Nobody Atall"}}, _players())
    assert unresolved == [("ARS", "Nobody Atall")]


def _archive():
    rows = [
        # predicted well before the deadline, then revised, then confirmed
        ("2026-09-01T09:00:00+00:00", 3, "ARS", "predicted", "A"),
        ("2026-09-01T09:00:00+00:00", 3, "ARS", "predicted", "B"),
        ("2026-09-04T14:00:00+00:00", 3, "ARS", "predicted", "A"),
        ("2026-09-04T14:00:00+00:00", 3, "ARS", "predicted", "C"),
        # a forecast published AFTER the deadline must not count
        ("2026-09-04T18:00:00+00:00", 3, "ARS", "predicted", "D"),
        ("2026-09-05T13:00:00+00:00", 3, "ARS", "confirmed", "A"),
        ("2026-09-05T13:00:00+00:00", 3, "ARS", "confirmed", "D"),
        # a different gameweek
        ("2026-09-10T09:00:00+00:00", 4, "ARS", "predicted", "Z"),
    ]
    df = pd.DataFrame(rows, columns=["observed_utc", "gw", "team_abbr",
                                     "status", "player"])
    df["observed"] = pd.to_datetime(df["observed_utc"], utc=True)
    return df


def test_the_forecast_is_the_last_predicted_xi_strictly_before_the_deadline():
    cutoff = pd.Timestamp("2026-09-04T17:30:00Z")
    fc = lf.pre_deadline_forecasts(_archive(), 3, cutoff)
    assert fc["ARS"][1] == {"A", "C"}          # the revision, not the first
    assert lf.confirmed_xis(_archive(), 3)["ARS"] == {"A", "D"}
    # nothing pre-deadline for a gameweek whose forecasts all came later
    assert lf.pre_deadline_forecasts(_archive(), 3,
                                     pd.Timestamp("2026-09-01T00:00:00Z")) == {}


def test_band_accuracy_converts_to_points_on_the_e8b_line():
    assert lf.implied_points(0.55) == pytest.approx(0.0)
    assert lf.implied_points(1.00) == pytest.approx(89.0)
    assert lf.implied_points(0.775) == pytest.approx(44.5)
    d = pd.DataFrame({"p_start": [0.4, 0.6, 0.5, 0.35],
                      "feed_xi": [True, False, True, False],
                      "started": [True, False, False, False],
                      "team": ["A"] * 4})
    m = lf.block(d)
    assert m["n"] == 4
    assert m["feed_accuracy"] == pytest.approx(0.75)
    # the model calls 0.6 and 0.5 starters (they did not) and 0.4 a non-starter
    # (he did): one of four right
    assert m["model_accuracy"] == pytest.approx(0.25)
    assert m["disagreements"] == 2
    assert m["implied_points_per_season"] == pytest.approx(
        round((0.75 - 0.55) / 0.45 * 89, 1))
    assert lf.block(d.iloc[0:0]) == {"n": 0}


def test_availability_factor_mirrors_the_live_overlay():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE acq_player_availability (season, player_id, "
                 "observed_utc, source_id, source_published_utc, status, "
                 "chance_next, news, raw_id)")
    rows = [("2026-27", 1, "2026-09-01T00:00:00Z", "fpl", None, "a", None, None, None),
            ("2026-27", 1, "2026-09-03T00:00:00Z", "fpl", None, "d", 0.75, "x", None),
            ("2026-27", 1, "2026-09-05T00:00:00Z", "fpl", None, "i", 0.0, "y", None),
            ("2026-27", 2, "2026-09-01T00:00:00Z", "fpl", None, "s", None, "ban", None),
            ("2026-27", 3, "2026-09-01T00:00:00Z", "fpl", None, "d", 50, "pct", None)]
    conn.executemany("INSERT INTO acq_player_availability VALUES (?,?,?,?,?,?,?,?,?)",
                     rows)
    at = lf.availability_at(conn, "2026-27", "2026-09-04T17:30:00Z")
    assert at[1] == pytest.approx(0.75)      # the doubt, not the later injury
    assert at[2] == 0.0                      # suspended
    assert at[3] == pytest.approx(0.5)       # a percentage is normalised
    assert lf.availability_at(conn, "2026-27", "2026-08-01T00:00:00Z") == {}
