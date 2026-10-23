"""The Live desk serves what managers said, not only what a rule could parse.

The panel used to render `presser_obs` — the extractor's player-level output.
That extractor classifies the clause that NAMES a player, so a post like "was
asked about the availability of Mosquera, White, Timber and Hincapie" followed
by "everyone is fine" yields nothing, and a Friday morning with two managers'
worth of quotes showed one line. These pin the feed that fixed it.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app import deadline as dd

PL = {10: {"name": "Hadjam", "pos": "DEF", "team_id": 3},
      11: {"name": "Timber", "pos": "DEF", "team_id": 1}}
TM = {1: "ARS", 3: "BHA"}


def iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE acq_bbc_presser (page_id, post_urn, published_utc, fixture_label, text, raw_id)")
    c.execute("CREATE TABLE presser_obs (season, gw, player_id, team_id, cls, source, "
              "published_utc, phrase, snippet, page_id, post_urn)")
    c.execute("CREATE TABLE team (season, team_id, name, short_name)")
    c.executemany("INSERT INTO team VALUES ('2026-27', ?, ?, ?)", [
        (1, "Arsenal", "ARS"), (3, "Brighton & Hove Albion", "BHA")])
    return c


def post(c, urn, hours, text, fixture="Brighton v Arsenal (Sat, 15:00 BST)"):
    c.execute("INSERT INTO acq_bbc_presser VALUES ('p1', ?, ?, ?, ?, NULL)",
              (urn, iso(hours), fixture, text))


def test_a_manager_quote_is_served_with_its_club_speaker_and_split(conn):
    post(conn, "u1", 2, 'Arsenal boss Mikel Arteta on Jurrien Timber\'s fitness:  "He is '
                        'fine. We will use the final session to make sure he is ready."')
    q, = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert q["club_id"] == 1 and q["club"] == "ARS"
    assert q["manager"] == "Mikel Arteta"
    assert q["lead"] == "Arsenal boss Mikel Arteta on Jurrien Timber's fitness"
    assert q["quote"].startswith('"He is fine.')
    assert q["fixture"].startswith("Brighton v Arsenal")
    assert q["players"] == []


def test_a_club_whose_name_is_longer_than_the_bbcs_still_resolves(conn):
    post(conn, "u1", 1, 'Brighton boss Fabian Hurzeler on Lewis Dunk: "That number is impressive."')
    q, = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert q["club"] == "BHA" and q["club_id"] == 3 and q["manager"] == "Fabian Hurzeler"


def test_live_blog_chatter_is_dropped(conn):
    post(conn, "u1", 1, "Good morning football fans. We've got a packed weekend ahead.", fixture=None)
    post(conn, "u2", 1, "And here he is. Mikel is in the building.", fixture=None)
    post(conn, "u3", 1, 'Everton boss David Moyes on his squad: "We are all fit."')
    assert [q["manager"] for q in dd._presser_quotes(conn, "2026-27", 5, PL, TM)] == ["David Moyes"]


def test_a_post_the_extractor_read_is_kept_even_without_a_speaker(conn):
    post(conn, "u1", 1, "An update on Jaouen Hadjam, who picked up a knock in midweek.", fixture=None)
    conn.execute("INSERT INTO presser_obs VALUES ('2026-27', 5, 10, 3, 'doubt', 'presser', ?, "
                 "'hopeful', 'snippet', 'p1', 'u1')", (iso(1),))
    q, = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert q["manager"] is None
    assert q["players"] == [{"player_id": 10, "name": "Hadjam", "team": "BHA", "cls": "doubt"}]


def test_statements_attach_to_their_own_post_and_do_not_repeat(conn):
    post(conn, "u1", 2, 'Brighton boss Fabian Hurzeler on Hadjam: "He is hopeful."')
    post(conn, "u2", 1, 'Arsenal boss Mikel Arteta on Timber: "He trained."')
    conn.executemany(
        "INSERT INTO presser_obs VALUES ('2026-27', 5, ?, ?, ?, 'presser', ?, 'p', 's', 'p1', ?)", [
            (10, 3, "doubt", iso(2), "u1"), (10, 3, "doubt", iso(2), "u1"), (11, 1, "available", iso(1), "u2")])
    by_urn = {q["manager"]: q["players"] for q in dd._presser_quotes(conn, "2026-27", 5, PL, TM)}
    assert [p["name"] for p in by_urn["Fabian Hurzeler"]] == ["Hadjam"]     # deduped
    assert [p["name"] for p in by_urn["Mikel Arteta"]] == ["Timber"]


def test_only_this_deadlines_build_up_is_served(conn):
    post(conn, "u1", 2, 'Arsenal boss Mikel Arteta on form: "We are good."')
    post(conn, "u2", 24 * dd.QUOTE_DAYS + 3, 'Arsenal boss Mikel Arteta on last month: "Old news."')
    q = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert len(q) == 1 and "We are good" in q[0]["quote"]


def test_newest_first_and_capped_per_club(conn):
    for i in range(dd.QUOTE_PER_CLUB + 10):
        post(conn, f"u{i}", i / 10 + 1, f'Arsenal boss Mikel Arteta on point {i}: "Quote {i}."')
    q = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert len(q) == dd.QUOTE_PER_CLUB
    assert "point 0" in q[0]["lead"] and q[0]["when"] > q[-1]["when"]


def test_a_long_presser_cannot_push_the_morning_speakers_off_the_panel(conn):
    # GW8: Carrick's 22 posts at lunchtime crowded Fulham's 10:28 speaker out
    # of a newest-first global cap
    post(conn, "early", 9, 'Brighton boss Fabian Hurzeler on Dunk: "Fine."')
    for i in range(40):
        post(conn, f"a{i}", 1 + i / 100, f'Arsenal boss Mikel Arteta on point {i}: "Quote."')
    q = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert "Fabian Hurzeler" in {x["manager"] for x in q}


def test_a_missing_table_is_not_an_error(conn):
    conn.execute("DROP TABLE presser_obs")
    assert dd._presser_quotes(conn, "2026-27", 5, PL, TM) == []


# ---- GW8 (23 Oct 2026): the phrasings the anchored regex got wrong --------

@pytest.fixture
def gw8(conn):
    conn.executemany("INSERT INTO team VALUES ('2026-27', ?, ?, ?)", [
        (9, "Everton", "EVE"), (10, "Fulham", "FUL"), (13, "Man Utd", "MUN"),
        (12, "Leeds", "LEE"), (19, "Spurs", "TOT"), (17, "Newcastle", "NEW"),
        (6, "Chelsea", "CHE")])
    return conn


@pytest.mark.parametrize("text,club,manager", [
    ("And here's Everton boss David Moyes on his contract:", "EVE", "David Moyes"),
    ("A final word from Fulham boss Alvaro Arbeloa on team news. He said", "FUL", "Alvaro Arbeloa"),
    ("In today's news conference, Arsenal manager Mikel Arteta did not rule out", "ARS", "Mikel Arteta"),
    ("Manchester United manager Michael Carrick on Bruno Fernandes:", "MUN", "Michael Carrick"),
    ("Leeds United head coach Daniel Farke on team news", "LEE", "Daniel Farke"),
    ("Tottenham boss Roberto de Zerbi has confirmed", "TOT", "Roberto de Zerbi"),
])
def test_lead_ins_aliases_and_particles_resolve(gw8, text, club, manager):
    post(gw8, "u1", 1, text + ' "A quote."')
    q, = dd._presser_quotes(gw8, "2026-27", 8, PL, TM)
    assert (q["club"], q["manager"]) == (club, manager)
    assert q["club_id"] is not None


def test_a_possessive_is_not_part_of_the_name(gw8):
    post(gw8, "u1", 1, "We have reached the end of Manchester United manager Michael Carrick's news conference.")
    q, = dd._presser_quotes(gw8, "2026-27", 8, PL, TM)
    assert q["manager"] == "Michael Carrick"


def test_a_post_naming_only_the_manager_is_placed_by_his_earlier_post(gw8):
    post(gw8, "u1", 2, 'Newcastle boss Mathias Jaissle has provided a fitness update: "Fine."')
    post(gw8, "u2", 1, 'Mathias Jaissle on 18-year-old midfielder Sean Steur: "He is ready."')
    got = {x["lead"][:15]: x["club"] for x in dd._presser_quotes(gw8, "2026-27", 8, PL, TM)}
    assert got["Mathias Jaissle"] == "NEW"


def test_one_lead_in_does_not_become_a_club(gw8):
    post(gw8, "u1", 1, 'And here\'s Everton boss David Moyes on injuries: "Bits and pieces."')
    post(gw8, "u2", 2, 'Everton boss David Moyes on Dibling: "Good."')
    assert {x["club"] for x in dd._presser_quotes(gw8, "2026-27", 8, PL, TM)} == {"EVE"}


def test_a_european_news_conference_page_is_not_the_deadline_build_up(gw8):
    gw8.execute("CREATE TABLE acq_bbc_presser_page (page_id, headline)")
    gw8.executemany("INSERT INTO acq_bbc_presser_page VALUES (?, ?)", [
        ("p1", "Premier League team news: Calvert-Lewin, Manzambi & Saka updates"),
        ("ucl", "Champions League news conferences: Iraola on Frimpong & Szoboszlai")])
    post(gw8, "u1", 1, 'Arsenal boss Mikel Arteta on Saka: "Weeks."')
    gw8.execute("INSERT INTO acq_bbc_presser VALUES ('ucl', 'u9', ?, 'Liverpool v Villarreal', ?, NULL)",
                (iso(1), 'Chelsea boss Xabi Alonso on Europe: "Big night."'))
    assert [x["manager"] for x in dd._presser_quotes(gw8, "2026-27", 8, PL, TM)] == ["Mikel Arteta"]


def test_a_missing_extractor_table_does_not_blank_the_quotes(conn):
    conn.execute("DROP TABLE presser_obs")
    post(conn, "u1", 1, 'Arsenal boss Mikel Arteta on Timber: "He trained."')
    assert [x["manager"] for x in dd._presser_quotes(conn, "2026-27", 5, PL, TM)] == ["Mikel Arteta"]


def test_narration_about_one_clubs_player_takes_that_club(conn):
    post(conn, "u1", 1, "Arsenal fear winger Timber could be out for weeks.", fixture=None)
    conn.execute("INSERT INTO presser_obs VALUES ('2026-27', 5, 11, 1, 'out', 'presser', ?, "
                 "'out', 'snippet', 'p1', 'u1')", (iso(1),))
    q, = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert (q["club_id"], q["club"], q["manager"]) == (1, "ARS", None)


def test_the_club_nearest_the_role_word_is_the_speakers(gw8):
    post(gw8, "u1", 1, 'Enough on Liverpool. Here is Arsenal manager Mikel Arteta: "Fine."')
    q, = dd._presser_quotes(gw8, "2026-27", 8, PL, TM)
    assert (q["club"], q["manager"]) == ("ARS", "Mikel Arteta")


def test_a_short_lead_word_does_not_prefix_match_a_club(gw8):
    gw8.execute("INSERT INTO team VALUES ('2026-27', 14, 'Liverpool', 'LIV')")
    post(gw8, "u1", 1, 'Live boss Somebody Else on nothing: "Hm."')
    q, = dd._presser_quotes(gw8, "2026-27", 8, PL, TM)
    assert q["club_id"] is None
