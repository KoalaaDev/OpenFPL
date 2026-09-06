"""The predicted-lineup collector: the one channel the decomposition leaves open.

A perfect rate estimate is worth nothing and a perfect LOCAL rate is worse, so
the only reachable headroom is whether a player starts and lasts an hour. This
source is the forecast of that, and every test here guards a distinction that
decides whether the archive is worth anything later.
"""
import csv
import os
import tempfile

from acquire import actions
from acquire.sources import predicted_lineups as pl


def _side(abbr, side, status, players, injuries=()):
    body = f'<ul class="lineup__list is-{side}">'
    body += f'<li class="lineup__status {status}">x</li>'
    for pos, name in players:
        body += (f'<li class="lineup__player"><div class="lineup__pos ">{pos}'
                 f'</div><a title="{name}" href="/soccer/player/{name.lower()}'
                 f'-{abs(hash(name)) % 99999}">{name}</a></li>')
    if injuries:
        body += '<li class="lineup__title is-middle">Injuries</li>'
        for pos, name in injuries:
            body += (f'<li class="lineup__player"><div class="lineup__pos ">'
                     f'{pos}</div><a title="{name}" href="/soccer/player/x-1">'
                     f'{name}</a></li>')
    return body + "</ul>"


XI = [("GK", "Keeper"), ("DL", "LeftBack"), ("DC", "CentreA"),
      ("DC", "CentreB"), ("DR", "RightBack"), ("DMC", "Holder"),
      ("MC", "Middle"), ("AML", "LeftWing"), ("AMC", "Ten"),
      ("AMR", "RightWing"), ("FW", "Striker")]


def _page(home_status="is-expected", away_status="is-confirmed"):
    return ('<div class="lineup is-soccer">'
            '<div class="lineup__abbr">AVL</div>'
            '<div class="lineup__abbr">ARS</div>'
            + _side("AVL", "home", home_status, XI,
                    injuries=[("F", "Crocked"), ("M", "Doubtful")])
            + _side("ARS", "visit", away_status, XI)
            + "</div>")


def test_the_injury_list_is_not_part_of_the_starting_eleven():
    # the doubt list reuses `lineup__player` with a one-letter position, so
    # counting to eleven works today and breaks the first time a side is
    # listed short — the "Injuries" heading is the real boundary
    rows = pl.parse(_page())
    home = [r for r in rows if r["team_abbr"] == "AVL"]
    assert len(home) == 11
    assert "Crocked" not in {r["player"] for r in rows}
    assert all(r["in_xi"] for r in rows)


def test_each_side_carries_its_own_status():
    # one club can confirm its XI while the opponent has not; labelling both
    # alike would turn a post-deadline fact into a pre-deadline forecast
    rows = pl.parse(_page())
    by = {r["team_abbr"]: r["status"] for r in rows}
    assert by["AVL"] == "predicted"
    assert by["ARS"] == "confirmed"


def test_positions_and_order_survive():
    rows = [r for r in pl.parse(_page()) if r["team_abbr"] == "AVL"]
    assert [r["position"] for r in rows][:3] == ["GK", "DL", "DC"]
    assert [r["slot"] for r in rows] == list(range(1, 12))
    assert rows[0]["player"] == "Keeper"


def test_a_page_with_no_published_xi_yields_nothing_rather_than_guessing():
    empty = ('<div class="lineup is-soccer">'
             '<div class="lineup__abbr">EVE</div>'
             '<div class="lineup__abbr">FUL</div>'
             '<ul class="lineup__list is-home">'
             '<li class="lineup__status is-expected">x</li>'
             '<li class="lineup__title is-middle">Injuries</li>'
             '<li class="lineup__player"><div class="lineup__pos ">F</div>'
             '<a title="Crocked" href="/soccer/player/x-1">Crocked</a></li>'
             '</ul></div>')
    assert pl.parse(empty) == []


# ------------------------------------------------------------------ archive
class _Resp:
    ok, status = True, 200

    def __init__(self, text):
        self.text = text


def test_the_archive_appends_only_when_the_forecast_changes(monkeypatch):
    """A four-times-daily run must not write the same XI four times.

    The file records CHANGES of forecast with the run's own clock, which is
    what makes a deadline-decay study possible later — and what keeps the git
    history honest and small.
    """
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(actions.http, "get", lambda *a, **k: _Resp(_page()))
        first = actions._collect_lineups(d, "2026-27", 3, "2026-08-31T09:00:00Z")
        assert first == 22
        again = actions._collect_lineups(d, "2026-27", 3, "2026-08-31T13:00:00Z")
        assert again == 0, "an unchanged XI must not be re-appended"

        # a genuine change to one side appends that side only
        changed = XI[:-1] + [("FW", "Someone Else")]
        page2 = ('<div class="lineup is-soccer">'
                 '<div class="lineup__abbr">AVL</div>'
                 '<div class="lineup__abbr">ARS</div>'
                 + _side("AVL", "home", "is-expected", changed)
                 + _side("ARS", "visit", "is-confirmed", XI) + "</div>")
        monkeypatch.setattr(actions.http, "get", lambda *a, **k: _Resp(page2))
        moved = actions._collect_lineups(d, "2026-27", 3, "2026-08-31T17:00:00Z")
        assert moved == 11

        path = os.path.join(d, "lineups", "2026-27.csv")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 33
        assert len({r["observed_utc"] for r in rows}) == 2


def test_a_failing_third_party_page_cannot_break_the_scheduled_run(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("network")
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(actions.http, "get", _boom)
        assert actions._collect_lineups(d, "2026-27", 3, "now") == 0


# ------------------------------------------------------- gameweek labelling
def test_kickoff_text_is_parsed_to_utc_with_us_daylight_saving():
    # "September 6 9:00 AM ET" seen on 5 September: EDT, so 13:00Z
    assert pl.kickoff_utc("September 6 9:00 AM ET",
                          "2026-09-05T13:28:57Z") == "2026-09-06T13:00:00Z"
    # winter: EST, so +5h; and the year rolls over January
    assert pl.kickoff_utc("January 3 3:00 PM ET",
                          "2026-12-30T00:00:00Z") == "2027-01-03T20:00:00Z"
    assert pl.kickoff_utc("December 26 12:30 PM ET",
                          "2027-01-02T00:00:00Z") == "2026-12-26T17:30:00Z"
    assert pl.kickoff_utc(None, "2026-09-05T00:00:00Z") is None
    assert pl.kickoff_utc("TBD", "2026-09-05T00:00:00Z") is None


def test_the_parser_carries_each_match_date():
    page = ('<div class="lineup is-soccer">'
            '<div class="lineup__meta"><div class="lineup__time"><b>September 6'
            '</b>&nbsp; 9:00 AM ET</div></div>'
            '<div class="lineup__abbr">AVL</div>'
            '<div class="lineup__abbr">ARS</div>'
            + _side("AVL", "home", "is-expected", XI)
            + _side("ARS", "visit", "is-expected", XI) + "</div>")
    rows = pl.parse(page)
    assert {r["kickoff_text"] for r in rows} == {"September 6 9:00 AM ET"}


def test_a_lineup_is_filed_under_the_gameweek_whose_deadline_precedes_it():
    """The bug this guards: FPL's is_next flips at Friday's deadline, so a
    confirmed Saturday XI of gameweek 3 was archived as gameweek 4."""
    deadlines = [("2026-08-28T17:30:00Z", 2), ("2026-09-04T17:30:00Z", 3),
                 ("2026-09-12T12:30:00Z", 4)]
    saturday_of_gw3 = "2026-09-05T14:00:00Z"
    assert actions.lineup_gw(saturday_of_gw3, deadlines) == 3
    assert actions.lineup_gw("2026-09-12T14:00:00Z", deadlines) == 4
    assert actions.lineup_gw("2026-08-01T14:00:00Z", deadlines) is None
    assert actions.lineup_gw(None, deadlines) is None
    assert actions.lineup_gw(saturday_of_gw3, []) is None


def test_the_same_eleven_named_for_the_next_match_is_a_new_forecast(monkeypatch):
    """An unchanged XI is not re-appended within a match, but the identical
    eleven predicted for the FOLLOWING fixture must be — otherwise a settled
    side would have no pre-deadline forecast on file for that gameweek."""
    def page(date):
        return ('<div class="lineup is-soccer">'
                f'<div class="lineup__time"><b>{date}</b>&nbsp; 10:00 AM ET</div>'
                '<div class="lineup__abbr">AVL</div>'
                '<div class="lineup__abbr">ARS</div>'
                + _side("AVL", "home", "is-expected", XI)
                + _side("ARS", "visit", "is-expected", XI) + "</div>")
    deadlines = [("2026-09-04T17:30:00Z", 3), ("2026-09-12T12:30:00Z", 4)]
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setattr(actions.http, "get",
                            lambda *a, **k: _Resp(page("September 5")))
        assert actions._collect_lineups(d, "2026-27", 3, "2026-09-03T09:00:00Z",
                                        deadlines) == 22
        assert actions._collect_lineups(d, "2026-27", 3, "2026-09-04T09:00:00Z",
                                        deadlines) == 0
        # the page moves on to next week's match with the same eleven; FPL's
        # is_next still says 4 in both cases, the kickoff decides the label
        monkeypatch.setattr(actions.http, "get",
                            lambda *a, **k: _Resp(page("September 12")))
        assert actions._collect_lineups(d, "2026-27", 4, "2026-09-08T09:00:00Z",
                                        deadlines) == 22
        path = os.path.join(d, "lineups", "2026-27.csv")
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert {(r["gw"], r["kickoff_utc"]) for r in rows} == {
            ("3", "2026-09-05T14:00:00Z"), ("4", "2026-09-12T14:00:00Z")}
