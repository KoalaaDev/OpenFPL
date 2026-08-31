"""Entity resolution: FPL <-> Understat player matching."""
from fpl_engine import db, resolve


def test_within_club_pass_resolves_display_names(conn):
    # FPL full names are long/official; Understat uses display names
    db.upsert(conn, "team", [
        {"season": "2024-25", "team_id": 1, "name": "Man Utd", "short_name": "MUN",
         "understat_name": "Manchester United"},
        {"season": "2024-25", "team_id": 2, "name": "Arsenal", "short_name": "ARS",
         "understat_name": "Arsenal"},
    ])
    db.upsert(conn, "player", [
        {"season": "2024-25", "player_id": 10, "code": 1000, "web_name": "B.Fernandes",
         "full_name": "Bruno Miguel Borges Fernandes", "team_id": 1, "position": "MID"},
        {"season": "2024-25", "player_id": 30, "code": 3000, "web_name": "Fernandes",
         "full_name": "Matheus Fernandes", "team_id": 1, "position": "MID"},
        {"season": "2024-25", "player_id": 40, "code": 4000, "web_name": "White",
         "full_name": "Benjamin White", "team_id": 2, "position": "DEF"},
        {"season": "2024-25", "player_id": 50, "code": 5000, "web_name": "Gabriel",
         "full_name": "Gabriel dos Santos Magalhaes", "team_id": 2, "position": "DEF"},
        {"season": "2024-25", "player_id": 60, "code": 6000, "web_name": "Martinelli",
         "full_name": "Gabriel Martinelli Silva", "team_id": 2, "position": "MID"},
    ])
    conn.commit()
    names = {"u1": "Bruno Fernandes", "u3": "Matheus Fernandes", "u4": "Ben White",
             "u5": "Gabriel", "u6": "Gabriel Martinelli", "u7": "Gabriel Jesus"}
    clubs = {"u1": "Manchester United", "u3": "Manchester United", "u4": "Arsenal",
             "u5": "Arsenal", "u6": "Arsenal", "u7": "Arsenal"}
    res = resolve.resolve_players(conn, "2024-25", names, understat_teams=clubs)
    got = {r["player_id"]: r["understat_id"] for r in conn.execute(
        "SELECT player_id, understat_id FROM player WHERE season='2024-25'")}
    assert got[10] == "u1"          # surname + first initial within the club
    assert got[30] == "u3"          # exact-name pass
    assert got[40] == "u4"          # "Benjamin White" -> "Ben White" via web_name
    assert got[50] == "u5"          # exact display name "Gabriel" wins outright
    assert got[60] == "u6"          # "Martinelli" token, initial G agrees
    assert not res["ambiguous"]


def test_club_pass_reports_ambiguity_instead_of_guessing(conn):
    db.upsert(conn, "team", [{"season": "2024-25", "team_id": 3, "name": "Chelsea",
                              "short_name": "CHE", "understat_name": "Chelsea"}])
    db.upsert(conn, "player", [{"season": "2024-25", "player_id": 70, "code": 7000,
                                "web_name": "James", "full_name": "Reece James",
                                "team_id": 3, "position": "DEF"}])
    conn.commit()
    names = {"u8": "Reece James", "u9": "Reece James"}      # duplicate display name
    clubs = {"u8": "Chelsea", "u9": "Chelsea"}
    res = resolve.resolve_players(conn, "2024-25", names, understat_teams=clubs)
    assert "Reece James" in res["ambiguous"]
    assert conn.execute("SELECT understat_id FROM player WHERE player_id=70").fetchone()[0] is None


def test_one_player_claiming_two_understat_ids_is_unset_in_every_season():
    """FPL's Amad Diallo resolved to two different footballers.

    8127 is "Amad Diallo Traore" (Manchester United) and 12200 is "Amadou
    Diallo" (Newcastle United) — a different man, whose shots and match roles
    were attached to him in three of five seasons. The cross-season fill only
    touches NULLs, so nothing reconciled the disagreement.

    There is no safe automatic tiebreak: the wrong id won on BOTH the number of
    seasons (3 vs 2) and the volume of stored match data (1 row vs 0). So the
    rule is refusal, matching the resolver's existing collision rule.
    """
    import sqlite3
    from fpl_engine import pipeline

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE player (season TEXT, player_id INTEGER, "
                 "code INTEGER, understat_id TEXT)")
    conn.executemany(
        "INSERT INTO player VALUES (?,?,?,?)",
        [("2022-23", 1, 493250, "12200"), ("2023-24", 2, 493250, "12200"),
         ("2024-25", 3, 493250, "8127"), ("2025-26", 4, 493250, "8127"),
         # an untouched player with a single consistent id
         ("2024-25", 9, 111, "500"), ("2025-26", 9, 111, "500")])
    out = pipeline.reconcile_understat_ids(conn)
    assert out["ambiguous_codes"] == 1
    rows = dict(conn.execute(
        "SELECT code, COUNT(understat_id) FROM player GROUP BY code").fetchall())
    assert rows[493250] == 0        # unset in every season, not just one
    assert rows[111] == 2           # the consistent player is untouched
