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


def test_newest_first_and_capped(conn):
    for i in range(dd.QUOTE_MAX + 10):
        post(conn, f"u{i}", i / 10 + 1, f'Arsenal boss Mikel Arteta on point {i}: "Quote {i}."')
    q = dd._presser_quotes(conn, "2026-27", 5, PL, TM)
    assert len(q) == dd.QUOTE_MAX
    assert "point 0" in q[0]["lead"] and q[0]["when"] > q[-1]["when"]


def test_a_missing_table_is_not_an_error(conn):
    conn.execute("DROP TABLE presser_obs")
    assert dd._presser_quotes(conn, "2026-27", 5, PL, TM) == []
