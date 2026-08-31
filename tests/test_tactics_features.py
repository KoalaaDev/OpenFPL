"""The tactics expert's point-in-time contract and its two sign traps.

Both defects here produced plausible output rather than an error: the
manager x opponent family measured the wrong side's strength, and the formation
carried Understat's resolution rate instead of the shape.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fpl_engine.ingest import transfermarkt as tm
from fpl_engine.xpts import tactics_features as tf


MANAGER_ROW = '''
<table class="items"><tbody>
<tr class="odd">
  <td><table class=inline-table><tr>
    <td rowspan=2><a href="/mikel-arteta/profil/trainer/47620"><img
        title="Mikel Arteta" /></a></td>
    <td class=hauptlink><a title="Mikel Arteta" id="47620"
        href="/mikel-arteta/profil/trainer/47620">Mikel Arteta</a></td></tr>
    <tr><td>26/03/1982</td></tr></table></td>
  <td class="zentriert"><img title="Spain" /></td>
  <td class="zentriert">22/12/2019</td>
  <td class="zentriert"></td>
  <td class="rechts">2444 days&nbsp;</td>
  <td class="zentriert"><a href="/trainer/leistungsdatenDetail?id=47620">354</a></td>
  <td class="zentriert">2.03</td>
</tr>
<tr class="even">
  <td><table class=inline-table><tr>
    <td rowspan=2><a href="/unai-emery/profil/trainer/5075"><img /></a></td>
    <td class=hauptlink><a title="Unai Emery" id="5075"
        href="/unai-emery/profil/trainer/5075">Unai Emery</a></td></tr>
    <tr><td>03/11/1971</td></tr></table></td>
  <td class="zentriert"><img title="Spain" /></td>
  <td class="zentriert">01/07/2018</td>
  <td class="zentriert">29/11/2019</td>
  <td class="rechts">516 days&nbsp;</td>
  <td class="zentriert"><a href="/x">78</a></td>
  <td class="zentriert">1.85</td>
</tr></tbody></table>'''


def test_a_sitting_manager_has_no_departure_date():
    rows = tm.parse_managers(MANAGER_ROW)
    assert len(rows) == 2
    arteta, emery = rows
    assert arteta["appointed"] == "2019-12-22" and arteta["left_date"] is None
    assert emery["appointed"] == "2018-07-01"
    assert emery["left_date"] == "2019-11-29"


def test_the_match_count_is_not_the_day_count():
    # the tenure cell reads "2444 days" and the match count sits after it,
    # wrapped in a link — reading numeric cells from the start returns 2444
    rows = tm.parse_managers(MANAGER_ROW)
    assert rows[0]["days"] == 2444
    assert rows[0]["matches"] == 354
    assert rows[0]["ppg"] == pytest.approx(2.03)


# --------------------------------------------------------------- features --
def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
      CREATE TABLE player (season TEXT, player_id INTEGER, code INTEGER,
        team_id INTEGER, position TEXT, understat_id TEXT);
      CREATE TABLE team (season TEXT, team_id INTEGER, understat_name TEXT);
      CREATE TABLE team_match (season TEXT, fixture_id INTEGER, team_id INTEGER,
        opponent_id INTEGER);
      CREATE TABLE fixture (season TEXT, fixture_id INTEGER, team_h INTEGER,
        team_a INTEGER);
      CREATE TABLE understat_team_match (season TEXT, understat_team TEXT,
        match_date TEXT, understat_match_id TEXT, is_home INTEGER, xg REAL,
        xga REAL, deep REAL, deep_allowed REAL, ppda_att REAL, ppda_def REAL,
        ppda_allowed_att REAL, ppda_allowed_def REAL);
      CREATE TABLE understat_player_match (season TEXT, understat_id TEXT,
        match_date TEXT, understat_match_id TEXT, minutes REAL, position TEXT);
    """)
    tm.init(conn)
    tm.init2(conn)
    tm.init3(conn)
    return conn


def _seed(conn):
    conn.execute("INSERT INTO team VALUES ('2025-26',1,'Arsenal')")
    conn.execute("INSERT INTO team VALUES ('2025-26',2,'Chelsea')")
    conn.executemany(
        "INSERT INTO player (season, player_id, code, team_id, position, "
        "understat_id) VALUES (?,?,?,?,?,?)",
        [("2025-26", 10, 111, 1, "MID", "u1"),
         ("2025-26", 11, 112, 1, "MID", "u2")])
    conn.execute("INSERT INTO tm_player (tm_player_id, tm_name, player_code) "
                 "VALUES (1,'A',111)")
    conn.execute("INSERT INTO tm_squad (season, tm_player_id, tm_club_id, "
                 "observed_utc) VALUES ('2025-26',1,11,'x')")
    conn.execute("INSERT INTO tm_manager_spell (tm_club_id, tm_manager_id, "
                 "appointed, left_date, manager, observed_utc) "
                 "VALUES (11, 900, '2025-08-01', NULL, 'M', 'x')")


def test_the_manager_in_charge_is_the_one_appointed_before_the_kickoff():
    conn = _db()
    _seed(conn)
    frame = pd.DataFrame({
        "season": ["2025-26", "2025-26"], "player_code": [111, 111],
        "team_id": [1, 1], "position": ["MID", "MID"], "fixture_id": [1, 2],
        "kick": ["2025-07-01T14:00:00Z", "2025-09-01T14:00:00Z"]})
    out = tf.add_features(frame, tf.load(conn))
    assert pd.isna(out["mgr_days_in_post"].iloc[0])      # before he arrived
    assert out["mgr_days_in_post"].iloc[1] == pytest.approx(31.58, abs=0.1)
    assert out["mgr_new_60d"].iloc[1] == 1.0


def test_the_formation_is_a_share_not_a_count_of_resolved_players():
    # Understat resolves about two thirds of a squad, at a rate that varies by
    # club and season. Counts would carry the resolution rate; shares do not.
    conn = _db()
    _seed(conn)
    rows = []
    for d in ("2025-08-10", "2025-08-17", "2025-08-24"):
        # only two of the eleven starters are resolved: one DC, one AMC
        rows += [("2025-26", "u1", d, "m", 90.0, "DC"),
                 ("2025-26", "u2", d, "m", 90.0, "AMC")]
    conn.executemany("INSERT INTO understat_player_match (season, understat_id,"
                     " match_date, understat_match_id, minutes, position) "
                     "VALUES (?,?,?,?,?,?)", rows)
    frame = pd.DataFrame({
        "season": ["2025-26"], "player_code": [111], "team_id": [1],
        "position": ["MID"], "fixture_id": [9],
        "kick": ["2025-09-01T14:00:00Z"]})
    out = tf.add_features(frame, tf.load(conn))
    # one DEF and one AM out of two observed starters: 0.5 each, not 1 each
    assert out["form_def_l5"].iloc[0] == pytest.approx(0.5)
    assert out["form_am_l5"].iloc[0] == pytest.approx(0.5)
    assert out["form_seen_l5"].iloc[0] == pytest.approx(2.0)


def test_the_players_line_is_read_from_understat_not_from_fpls_label():
    conn = _db()
    _seed(conn)
    conn.executemany("INSERT INTO understat_player_match (season, understat_id,"
                     " match_date, understat_match_id, minutes, position) "
                     "VALUES (?,?,?,?,?,?)",
                     [("2025-26", "u1", "2025-08-10", "m", 90.0, "DMC"),
                      ("2025-26", "u2", "2025-08-10", "m", 90.0, "AMC")])
    frame = pd.DataFrame({
        "season": ["2025-26"], "player_code": [111], "team_id": [1],
        "position": ["MID"], "fixture_id": [9],
        "kick": ["2025-09-01T14:00:00Z"]})
    out = tf.add_features(frame, tf.load(conn))
    assert out["role_is_dm"].iloc[0] == 1.0
    assert out["role_is_am"].iloc[0] == 0.0
    # FPL calls him a MID (line 3); Understat says he sat in front of the back
    # four (line 2), which is the distinction the family exists to carry
    assert out["role_vs_fpl_line"].iloc[0] == pytest.approx(-1.0)


def test_a_role_dated_after_the_kickoff_is_not_visible():
    conn = _db()
    _seed(conn)
    conn.execute("INSERT INTO understat_player_match (season, understat_id, "
                 "match_date, understat_match_id, minutes, position) "
                 "VALUES ('2025-26','u1','2025-10-01','m',90.0,'DMC')")
    frame = pd.DataFrame({
        "season": ["2025-26"], "player_code": [111], "team_id": [1],
        "position": ["MID"], "fixture_id": [9],
        "kick": ["2025-09-01T14:00:00Z"]})
    out = tf.add_features(frame, tf.load(conn))
    assert pd.isna(out["role_slots_l5"].iloc[0])
    assert pd.isna(out["role_vs_fpl_line"].iloc[0])


def test_the_shipped_line_features_degrade_to_nan_without_understat():
    """The three shipped features must be safe when Understat is unreachable.

    They are in `minutes_model.FEATURES`, so a database with no Understat data
    has to produce NaN rather than raise — the classifier then behaves exactly
    as it did before the block existed.
    """
    from fpl_engine.xpts import minutes_model as mm
    conn = _db()
    _seed(conn)
    frame = pd.DataFrame({
        "season": ["2025-26"], "player_code": [111], "team_id": [1],
        "position": ["MID"], "kick": ["2025-09-01T14:00:00Z"]})
    out = tf.add_line_features(frame, tf.load_roles(conn))
    assert set(tf.TAC_LINE) <= set(out.columns)
    assert out[tf.TAC_LINE].isna().all().all()
    assert all(f in mm.FEATURES for f in tf.TAC_LINE)


def test_an_unknown_line_is_nan_and_not_a_denial():
    conn = _db()
    _seed(conn)
    conn.executemany("INSERT INTO understat_player_match (season, understat_id,"
                     " match_date, understat_match_id, minutes, position) "
                     "VALUES (?,?,?,?,?,?)",
                     [("2025-26", "u1", "2025-08-10", "m", 90.0, "AMC")])
    frame = pd.DataFrame({
        "season": ["2025-26", "2025-26"], "player_code": [111, 112],
        "team_id": [1, 1], "position": ["MID", "MID"],
        "kick": ["2025-09-01T14:00:00Z", "2025-09-01T14:00:00Z"]})
    out = tf.add_line_features(frame, tf.load_roles(conn))
    assert out["role_is_am"].iloc[0] == 1.0
    assert out["role_vs_fpl_line"].iloc[0] == pytest.approx(1.0)
    # 112 has no Understat role: he is UNSEEN, not "not an attacking midfielder"
    assert pd.isna(out["role_is_am"].iloc[1])
    assert pd.isna(out["role_is_dm"].iloc[1])


def test_a_substitute_appearance_does_not_name_a_line():
    conn = _db()
    _seed(conn)
    conn.execute("INSERT INTO understat_player_match (season, understat_id, "
                 "match_date, understat_match_id, minutes, position) "
                 "VALUES ('2025-26','u1','2025-08-10','m',20.0,'Sub')")
    frame = pd.DataFrame({
        "season": ["2025-26"], "player_code": [111], "team_id": [1],
        "position": ["MID"], "kick": ["2025-09-01T14:00:00Z"]})
    out = tf.add_line_features(frame, tf.load_roles(conn))
    assert pd.isna(out["role_is_am"].iloc[0])
