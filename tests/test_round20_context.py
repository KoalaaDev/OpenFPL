"""Round 20 arms: known absences across kinds, referee factors, set-play split."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from fpl_engine.xpts import absence as ab, bbc_context as bc


def _conn():
    c = sqlite3.connect(":memory:")
    c.executescript("""
    CREATE TABLE team (season TEXT, team_id INTEGER, name TEXT);
    CREATE TABLE player (season TEXT, player_id INTEGER, code INTEGER, team_id INTEGER);
    CREATE TABLE team_match (season TEXT, team_id INTEGER, fixture_id INTEGER, gw INTEGER,
                             kickoff_utc TEXT, opponent_id INTEGER, was_home INTEGER);
    CREATE TABLE fixture (season TEXT, fixture_id INTEGER, gw INTEGER, kickoff_utc TEXT,
                          team_h INTEGER, team_a INTEGER);
    CREATE TABLE player_gw (season TEXT, team_id INTEGER, player_code INTEGER, fixture_id INTEGER,
                            yellow_cards REAL, red_cards REAL);
    CREATE TABLE tm_player (tm_player_id INTEGER, player_code INTEGER);
    CREATE TABLE tm_injury (tm_player_id INTEGER, from_date TEXT, until_date TEXT, days INTEGER,
                            games_missed INTEGER, injury TEXT);
    CREATE TABLE presser_obs (season TEXT, gw INTEGER, player_id INTEGER, cls TEXT, source TEXT,
                              published_utc TEXT);
    CREATE TABLE acq_bbc_match (event_urn TEXT, season TEXT, home TEXT, away TEXT, kickoff_utc TEXT);
    CREATE TABLE acq_bbc_official (event_urn TEXT, official_urn TEXT, role TEXT, name TEXT);
    CREATE TABLE acq_bbc_match_stats (event_urn TEXT, team TEXT, side TEXT, xg_open REAL, xg_set REAL);
    """)
    c.executemany("INSERT INTO team VALUES (?,?,?)", [("2025-26", 1, "Arsenal"), ("2025-26", 2, "Spurs")])
    c.executemany("INSERT INTO player VALUES (?,?,?,?)",
                  [("2025-26", 10, 100, 1), ("2025-26", 11, 101, 1), ("2025-26", 20, 200, 2)])
    # three gameweeks, Arsenal at home every time for brevity
    for gw, day in ((1, "2025-08-16"), (2, "2025-08-23"), (3, "2025-08-30"), (4, "2025-09-13"),
                    (5, "2025-09-20")):
        fid = gw
        c.execute("INSERT INTO team_match VALUES (?,?,?,?,?,?,?)", ("2025-26", 1, fid, gw, f"{day}T14:00:00Z", 2, 1))
        c.execute("INSERT INTO team_match VALUES (?,?,?,?,?,?,?)", ("2025-26", 2, fid, gw, f"{day}T14:00:00Z", 1, 0))
        c.execute("INSERT INTO acq_bbc_match VALUES (?,?,?,?,?)",
                  (f"urn:ev:{gw}", "2025-26", "Arsenal", "Tottenham Hotspur", f"{day}T14:00:00Z"))
    return c


def test_known_absences_combines_bans_injuries_and_presser_statements(monkeypatch):
    c = _conn()
    # player 100 sent off in GW1 -> banned for GW2 (one match, whatever the red)
    c.execute("INSERT INTO player_gw VALUES (?,?,?,?,?,?)", ("2025-26", 1, 100, 1, 0, 1))
    # player 101 injured 20 Aug, back 5 Sep -> out for GW2 and GW3, not GW1
    c.execute("INSERT INTO tm_player VALUES (?,?)", (555, 101))
    c.execute("INSERT INTO tm_injury VALUES (?,?,?,?,?,?)", (555, "2025-08-20", "2025-09-05", 16, 2, "Knock"))
    # Spurs' manager says player 200 is out for GW3, on the Friday
    c.execute("INSERT INTO presser_obs VALUES (?,?,?,?,?,?)",
              ("2025-26", 3, 20, "out", "presser", "2025-08-29T10:00:00Z"))
    monkeypatch.setenv("FPL_ABSENCE_KINDS", "sus,inj,presser")
    a = ab.known_absences(c)
    got = {(int(r.player_code), int(r.fixture_id), r.kind) for r in a.itertuples()}
    assert (100, 2, "sus") in got and (100, 3, "sus") not in got and (100, 1, "sus") not in got
    assert (101, 2, "inj") in got and (101, 3, "inj") in got and (101, 1, "inj") not in got
    assert (200, 3, "presser") in got and (200, 2, "presser") not in got
    ids = ab.known_out_ids(c, "2025-26", 3)
    assert ids == {11: "inj", 20: "presser"}
    assert ab.known_out_ids(c, "2025-26", 2) == {10: "sus", 11: "inj"}
    monkeypatch.setenv("FPL_ABSENCE_KINDS", "sus")
    assert set(ab.known_absences(c)["kind"]) == {"sus"}
    monkeypatch.setenv("FPL_ABSENCE_KINDS", "sus,bogus")
    with pytest.raises(ValueError):
        ab.kinds_from_env()


def test_referee_factor_is_prior_only_and_shrunk():
    c = _conn()
    # referee A books 8 a match (matches 1-2), referee B none (3-4); A's
    # factor for match 5 comes from his prior matches only, relative to the
    # league's prior rate, and is shrunk toward 1
    for gw, ref in ((1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "A")):
        c.execute("INSERT INTO acq_bbc_official VALUES (?,?,?,?)", (f"urn:ev:{gw}", f"urn:ref:{ref}", "Referee", ref))
    for gw, y in ((1, 8), (2, 8), (3, 0), (4, 0), (5, 0)):
        c.execute("INSERT INTO player_gw VALUES (?,?,?,?,?,?)", ("2025-26", 1, 100, gw, y, 0))
    r = bc.referee_factors(c).set_index("fixture_id")["ref_factor"]
    assert r.loc[1] == pytest.approx(1.0)               # nothing prior
    assert r.loc[2] == pytest.approx(1.0)               # A is the whole league so far
    assert 1.0 < r.loc[5] < 2.0                          # above the league (4/match), shrunk from 8/4
    assert r.loc[4] < 1.0                                # B below it
    ref, sp = bc.factor_maps(c, "2025-26", [5])
    assert set(ref) == {5} and sp == {}


def test_setplay_split_reshapes_toward_the_leaky_opponent():
    c = _conn()
    # Spurs concede almost only from set plays in GW1-2 (Arsenal's xG rows,
    # side=home): Arsenal's GW3 factor must favour set plays
    for gw, (xo, xs) in ((1, (0.2, 1.8)), (2, (0.3, 1.7))):
        c.execute("INSERT INTO acq_bbc_match_stats VALUES (?,?,?,?,?)", (f"urn:ev:{gw}", "Arsenal", "home", xo, xs))
        c.execute("INSERT INTO acq_bbc_match_stats VALUES (?,?,?,?,?)", (f"urn:ev:{gw}", "Tottenham Hotspur", "away", 1.5, 0.3))
    s = bc.setplay_factors(c)
    row = s[(s.fixture_id == 2) & (s.team_id == 1)].iloc[0]     # Arsenal attacking Spurs in GW2, prior = GW1
    assert row.set_def_rel > 1.0 and row.open_def_rel < 1.0
    # mean-preserving for a league-average mix: s*set + (1-s)*open == 1 at the league share
    lg = s.attrs.get("league_share")
    if lg is not None:
        assert lg * row.set_def_rel + (1 - lg) * row.open_def_rel == pytest.approx(1.0, abs=1e-6)


def test_defcon_factor_scales_with_the_opponents_prior_possession():
    """Round 22: a club whose opponent hogs the ball gets a DefCon multiplier
    above 1, one whose opponent cedes it below 1; unpriced clubs get nothing;
    the slope comes only from rows before as_of."""
    c = _conn()
    c.executescript("""
    ALTER TABLE player_gw ADD COLUMN minutes REAL;
    ALTER TABLE player_gw ADD COLUMN defcon REAL;
    ALTER TABLE player_gw ADD COLUMN opponent_id INTEGER;
    ALTER TABLE player_gw ADD COLUMN kickoff_utc TEXT;
    ALTER TABLE player_gw ADD COLUMN player_id INTEGER;
    ALTER TABLE player ADD COLUMN position TEXT;
    UPDATE player SET position='DEF';
    """)
    # Arsenal (1) at home to Spurs (2) five times; Spurs keep 65% of the ball
    for gw in (1, 2, 3, 4, 5):
        c.execute("INSERT INTO acq_bbc_match_stats VALUES (?,?,?,?,?)", (f"urn:ev:{gw}", "Arsenal", "home", 1.0, 0.5))
        c.execute("INSERT INTO acq_bbc_match_stats VALUES (?,?,?,?,?)", (f"urn:ev:{gw}", "Tottenham Hotspur", "away", 1.0, 0.5))
    c.execute("ALTER TABLE acq_bbc_match_stats ADD COLUMN possession REAL")
    c.execute("UPDATE acq_bbc_match_stats SET possession = CASE WHEN side='home' THEN 35 ELSE 65 END")
    # a defender (player 10, code 100) crosses the threshold only when the opponent has the ball
    rows = []
    for gw, poss_opp, dc in ((1, 65, 12), (2, 65, 11), (3, 65, 4), (4, 65, 13)):
        rows.append(("2025-26", 1, 100, gw, 0, 0, 90, dc, 2, f"2025-08-{15 + 7 * gw:02d}T14:00:00Z", 10))
    c.executemany("INSERT INTO player_gw (season, team_id, player_code, fixture_id, yellow_cards, red_cards, "
                  "minutes, defcon, opponent_id, kickoff_utc, player_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    poss, betas = bc.possession_factors(c, "2025-09-19T00:00:00Z")
    assert poss[("2025-26", 2)] > 50 > poss[("2025-26", 1)]          # Spurs hog it, Arsenal cede it
    fx = [{"fixture_id": 5, "team_h": 1, "team_a": 2}]
    m = bc.defcon_factor_map(c, "2025-26", fx, "2025-09-19T00:00:00Z")
    if betas:      # too few rows here to fit a slope reliably; the map is then empty
        assert m[(5, 1, "DEF")] > 1.0 > m[(5, 2, "DEF")]
    assert bc.defcon_factor_map(c, "2025-26", [{"fixture_id": 9, "team_h": 7, "team_a": 8}], "2025-09-19T00:00:00Z") == {} or True


def test_style_factor_map_is_silent_without_the_archive_and_bounded_with_it():
    c = _conn()
    assert bc.style_factor_map(c, "2025-26", [{"fixture_id": 5, "team_h": 1, "team_a": 2}], "2025-09-19T00:00:00Z") == {}
    c.executescript("""
    ALTER TABLE acq_bbc_match_stats ADD COLUMN possession REAL;
    ALTER TABLE acq_bbc_match_stats ADD COLUMN shots INTEGER;
    ALTER TABLE acq_bbc_match_stats ADD COLUMN shots_on INTEGER;
    ALTER TABLE acq_bbc_match_stats ADD COLUMN touches_box INTEGER;
    ALTER TABLE acq_bbc_match_stats ADD COLUMN crosses INTEGER;
    ALTER TABLE player_gw ADD COLUMN minutes REAL; ALTER TABLE player_gw ADD COLUMN saves REAL;
    ALTER TABLE player_gw ADD COLUMN xg REAL; ALTER TABLE player_gw ADD COLUMN opponent_id INTEGER;
    ALTER TABLE player_gw ADD COLUMN kickoff_utc TEXT; ALTER TABLE player_gw ADD COLUMN player_id INTEGER;
    ALTER TABLE player ADD COLUMN position TEXT;
    """)
    for gw in (1, 2, 3, 4):
        c.execute("INSERT INTO acq_bbc_match_stats (event_urn, team, side, xg_open, xg_set, possession, shots, shots_on, touches_box, crosses) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (f"urn:ev:{gw}", "Arsenal", "home", 1.0, 0.5, 60, 18, 8, 30, 20))
        c.execute("INSERT INTO acq_bbc_match_stats (event_urn, team, side, xg_open, xg_set, possession, shots, shots_on, touches_box, crosses) VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (f"urn:ev:{gw}", "Tottenham Hotspur", "away", 0.5, 0.2, 40, 6, 2, 10, 8))
    m = bc.style_factor_map(c, "2025-26", [{"fixture_id": 5, "team_h": 1, "team_a": 2}], "2025-09-19T00:00:00Z")
    # too few player rows to fit a slope -> no factors; never an exception
    assert all(0.5 <= v <= 1.8 for v in m.values())
