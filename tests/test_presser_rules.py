"""What the extractor reads out of a press conference.

Three defects it had, all found on the 2026-27 live season:

* clubs were mapped by a hardcoded table that went stale when FPL renamed
  Ipswich "Ipswich Town", so 74 posts a gameweek were dropped as "not a
  Premier League fixture";
* a verdict one sentence away from the names it applies to was invisible,
  which is the shape most of these posts take ("asked about the availability
  of A, B and C: 'Everyone is fine'");
* the phrase list only knew the formal wordings, not what managers say.
"""
import pytest

from fpl_engine import pressers as pr

SEASON_CLUBS = ["Arsenal", "Bournemouth", "Brighton", "Coventry City", "Hull City",
                "Ipswich Town", "Leeds", "Man City", "Man Utd", "Newcastle",
                "Nott'm Forest", "Spurs"]
IDX = pr.club_index(SEASON_CLUBS)


@pytest.mark.parametrize("bbc,fpl", [
    ("Ipswich", "Ipswich Town"),             # the rename that broke it
    ("Ipswich Town", "Ipswich Town"),
    ("Coventry", "Coventry City"),
    ("Hull", "Hull City"),
    ("Forest", "Nott'm Forest"),
    ("Nottingham Forest", "Nott'm Forest"),
    ("Tottenham", "Spurs"),
    ("Tottenham Hotspur", "Spurs"),
    ("Manchester United", "Man Utd"),
    ("Manchester City", "Man City"),
    ("Brighton & Hove Albion", "Brighton"),
    ("Brighton and Hove Albion", "Brighton"),
    ("AFC Bournemouth", "Bournemouth"),
    ("Leeds United", "Leeds"),
])
def test_clubs_resolve_to_the_seasons_own_names(bbc, fpl):
    assert pr.resolve_club(bbc, IDX) == fpl


def test_a_club_outside_the_league_stays_unresolved():
    for other in ("Sabah", "Napoli", "Atletico Madrid"):
        assert pr.resolve_club(other, IDX) is None


def test_a_name_two_clubs_answer_to_is_refused():
    """Both Manchester clubs shorten to "Manchester"; that is not an answer."""
    assert pr.resolve_club("Manchester", IDX) is None


def test_fixture_labels_resolve_both_sides():
    assert pr.fixture_clubs("Ipswich v Arsenal (Sat, 15:00 BST)", IDX) == ("Ipswich Town", "Arsenal")
    assert pr.fixture_clubs("not a fixture", IDX) is None


# --------------------------------------------------------------- phrases --
def cls_of(sentence, name="Timber"):
    pats = [({"player_id": 1, "web_name": name}, __import__("re").compile(name))]
    out = pr.extract_post(sentence, pats)
    return out[0][1] if out else None


@pytest.mark.parametrize("sentence,cls", [
    ("Timber is not in the squad for this one.", "out"),
    ("It comes too soon for Timber.", "out"),
    ("We won't risk Timber on Saturday.", "out"),
    ("Timber did not travel with the group.", "out"),
    ("Timber is back after the international break.", "out"),
    ("Timber has a torn hamstring.", "out"),
    ("Timber is carrying a knock.", "doubt"),
    ("I am optimistic about Timber.", "doubt"),
    ("Timber is not 100 per cent.", "doubt"),
    ("Timber trained partially with the group.", "doubt"),
    ("Timber needs a rest after three games.", "rested"),
    ("Timber trained normally today.", "available"),
    ("Timber came through the game.", "available"),
    ("Timber is fit again.", "available"),
    ("Timber is available again for selection.", "available"),
])
def test_what_managers_actually_say(sentence, cls):
    assert cls_of(sentence) == cls


def test_out_still_beats_the_softer_readings():
    """A sentence carrying both readings is the severe one: "no chance" is not
    a chance, and a suspension is not a doubt."""
    assert cls_of("There is no chance for Timber, we will assess him next week.") == "out"
    assert cls_of("Timber is suspended, hopefully he returns soon.") == "out"


# ------------------------------------------------------------ carry-over --
import re  # noqa: E402


def pats(*names):
    return [({"player_id": i, "web_name": n}, re.compile(n)) for i, n in enumerate(names, 1)]


def test_a_verdict_in_the_next_sentence_reaches_the_names_in_this_one():
    post = ("Arsenal boss Mikel Arteta was asked about the availability of Mosquera, "
            "White and Timber: \"Everyone is fine. We still have another session.\"")
    got = {p["web_name"]: cls for p, cls, _ph, _s in pr.extract_post(post, pats("Mosquera", "White", "Timber"))}
    assert got == {"Mosquera": "available", "White": "available", "Timber": "available"}


def test_a_verdict_about_somebody_else_does_not_carry():
    post = ("Arteta on Timber: \"He trained today.\" On Saka: \"He is ruled out.\"")
    got = {p["web_name"]: cls for p, cls, _ph, _s in pr.extract_post(post, pats("Timber", "Saka"))}
    assert got == {"Timber": "available", "Saka": "out"}


def test_the_carry_does_not_reach_across_a_whole_answer():
    post = ("Arteta on Timber: \"Thanks for the question.\" " + "Filler. " * 4
            + "Somebody is ruled out.")
    assert pr.extract_post(post, pats("Timber")) == []


def test_a_sentence_that_classifies_itself_is_not_carried_into():
    post = "Arteta says Timber is ruled out. The rest of them are fine."
    got = [(p["web_name"], cls) for p, cls, _ph, _s in pr.extract_post(post, pats("Timber"))]
    assert got == [("Timber", "out")]
