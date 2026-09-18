"""BBC Sport collector: parsers on the container shapes seen on the site."""
import os
import tempfile
from datetime import date

from acquire import storage
from acquire.sources import bbc


FIXTURES = {"eventGroups": [{"events": [{
    "id": "s-abc", "urn": "urn:bbc:sportsdata:football:event:s-abc",
    "startDateTime": "2025-05-25T15:00:00Z", "date": {"isoDate": "2025-05-25"},
    "status": "PostEvent", "onwardJourneyLink": "/sport/football/live/czr88kgr0ryt",
    "home": {"fullName": "Bournemouth", "score": 2},
    "away": {"fullName": "Leicester City", "score": 0},
}, {
    "id": "s-future", "urn": "urn:bbc:sportsdata:football:event:s-future",
    "startDateTime": "2026-09-20T14:00:00Z", "date": {"isoDate": "2026-09-20"},
    "status": "PreEvent", "home": {"fullName": "A"}, "away": {"fullName": "B"},
}]}]}

LINEUPS = {"homeTeam": {
    "name": {"fullName": "Ipswich Town"}, "formation": {"value": "4 - 2 - 3 - 1"},
    "pitchLayout": [[{"urn": "p:gk"}], [{"urn": "p:cb"}]],
    "players": {
        "starters": [
            {"urn": "p:gk", "name": {"short": "C. Walton", "first": "Christian", "last": "Walton"},
             "position": "Goalkeeper", "shirtNumber": 28, "formationPlace": 1, "isCaptain": False,
             "displayName": "C. Walton",
             "stats": [{"dataField": "minsPlayed", "statValue": 90}, {"dataField": "totalTackle", "statValue": 0}]},
            {"urn": "p:cb", "name": {"last": "Greaves"}, "position": "Defender", "shirtNumber": 4,
             "formationPlace": 5, "isCaptain": True,
             "stats": [{"dataField": "minsPlayed", "statValue": 90}, {"dataField": "totalTackle", "statValue": 3}]},
        ],
        "substitutes": [
            {"urn": "p:sub", "name": {"last": "Boniface"}, "position": "Substitute", "shirtNumber": 48,
             "stats": [{"dataField": "minsPlayed", "statValue": 12}]},
        ]}},
    "awayTeam": {"name": {"fullName": "West Ham United"}, "formation": {"value": "3-4-3"},
                 "players": {"starters": [{"urn": "p:wh1", "name": {"last": "Bowen"},
                                           "position": "Forward", "formationPlace": 10}],
                             "substitutes": []}}}

STREAM = {"page": {"index": 5, "total": 5}, "results": [
    {"urn": "urn:...?=commentary=1", "dates": {"firstPublished": "2025-05-25T14:00:12Z", "time": "1'"},
     "content": {"model": {"blocks": [{"type": "text", "model": {"blocks": [
         {"type": "paragraph", "model": {"text": "Lineups are announced and players are warming up."}}]}}]}}},
    {"urn": "urn:...?=commentary=2", "dates": {"firstPublished": "2025-05-25T15:20:00Z", "time": "19'"},
     "content": {"model": {"blocks": [{"type": "text", "model": {"blocks": [
         {"type": "paragraph", "model": {"text": "Substitution, West Ham United."}},
         {"type": "paragraph", "model": {"text": "Andy Irving replaces Tomas Soucek because of an injury."}}]}}]}}},
]}


def test_parse_fixtures_keeps_finished_and_upcoming_with_live_ids():
    ev = bbc.parse_fixtures(FIXTURES)
    assert [e["home"] for e in ev] == ["Bournemouth", "A"]
    assert ev[0]["live_id"] == "czr88kgr0ryt" and ev[0]["home_score"] == 2
    assert ev[0]["status"] == "PostEvent" and ev[1]["status"] == "PreEvent"
    assert ev[1]["live_id"] is None and ev[0]["match_date"] == "2025-05-25"


def test_parse_lineups_positions_captain_and_stats():
    rows = bbc.parse_lineups(LINEUPS)
    by = {r["player_urn"]: r for r in rows}
    assert by["p:gk"]["is_starter"] == 1 and by["p:gk"]["pitch_row"] == 0
    assert by["p:cb"]["is_captain"] == 1 and by["p:cb"]["pitch_row"] == 1
    assert by["p:cb"]["formation"] == "4-2-3-1" and by["p:cb"]["formation_place"] == 5
    assert by["p:sub"]["is_starter"] == 0 and by["p:sub"]["mins"] == 12
    assert '"totalTackle":3' in by["p:cb"]["stats_json"]
    assert by["p:wh1"]["side"] == "away" and by["p:wh1"]["team"] == "West Ham United"


def test_parse_stream_joins_paragraphs_and_reports_total_pages():
    posts, total = bbc.parse_stream(STREAM)
    assert total == 5 and len(posts) == 2
    assert posts[1]["text"].startswith("Substitution, West Ham United. Andy Irving")
    assert posts[1]["minute"] == "19'" and posts[0]["published_utc"] == "2025-05-25T14:00:12Z"


def test_season_helpers():
    from datetime import date
    assert bbc.season_of(date(2025, 5, 25)) == "2024-25"
    assert bbc.season_of(date(2025, 8, 16)) == "2025-26"
    days = list(bbc.season_days("2023-24"))
    assert days[0] == date(2023, 8, 1) and days[-1] == date(2024, 6, 10)


def test_collect_day_writes_normalised_rows(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = storage.connect(path)
    storage.init(conn)
    bbc.register(conn)
    import json
    from acquire.core import http

    def fake_get(url, **kw):
        if "scores-fixtures" in url:
            body = FIXTURES
        elif "match-lineups" in url:
            body = LINEUPS
        else:
            body = {**STREAM, "page": {"index": 1, "total": 1}}   # one page
        return http.Response(url, 200, json.dumps(body), http.utcnow(), "application/json")

    monkeypatch.setattr(http, "get", fake_get)
    from datetime import date
    c = bbc.collect_day(conn, date(2025, 5, 25))
    assert c["events"] == 2 and c["lineups"] == 4 and c["stream_posts"] == 2
    assert conn.execute("SELECT COUNT(*) FROM acq_bbc_match").fetchone()[0] == 2
    assert conn.execute("SELECT lineups_done, stream_pages, stream_total FROM acq_bbc_match "
                        "WHERE event_urn LIKE '%s-abc'").fetchone()[:3] == (1, 1, 1)
    # a second pass over the same day fetches nothing new
    c2 = bbc.collect_day(conn, date(2025, 5, 25))
    assert c2["skipped"] == 1 and c2["lineups"] == 0
    conn.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


SEARCH = {"initialResults": {"count": 3, "items": [
    {"uri": "urn:bbc:tipo:topic:c64gdwl1yrqwt", "headline": "Recap: Friday's Premier League news conferences",
     "categories": ["urn:bbc:tipo:type:live"], "datePublished": "2026-09-11T08:14:43.541Z"},
    {"uri": "urn:bbc:tipo:topic:cxxx", "headline": "Man Utd 2-0 Arsenal: report",
     "categories": ["urn:bbc:tipo:type:live"], "datePublished": "2026-09-10T00:00:00Z"},
    {"uri": "urn:bbc:tipo:topic:cyyy", "headline": "Premier League news conferences: Emery on Watkins",
     "categories": ["urn:bbc:tipo:type:article"], "datePublished": "2026-09-04T00:00:00Z"},
]}}

PRESSER_STREAM = {"page": {"index": 1, "total": 1}, "results": [
    {"urn": "urn:x:1", "dates": {"firstPublished": "2026-09-11T14:21:28.000Z"},
     "header": {"model": {"blocks": [{"type": "text", "model": {"blocks": [
         {"type": "paragraph", "model": {"text": "Bournemouth v Brentford (Sat, 15:00 BST)"}}]}}]}},
     "content": {"model": {"blocks": [{"type": "text", "model": {"blocks": [
         {"type": "paragraph", "model": {"text": "Brentford boss Keith Andrews on team news: \"He [Nathan Collins] won't be involved.\""}}]}}]}}},
]}


def test_pressers_search_filter_keeps_only_live_news_conference_pages():
    from acquire.sources import bbc_pressers as bp
    rows = bp.parse_search(SEARCH)
    assert [r["page_id"] for r in rows] == ["c64gdwl1yrqwt"]
    assert rows[0]["date_published"].startswith("2026-09-11")


def test_pressers_stream_key_and_posts():
    from acquire.sources import bbc_pressers as bp
    html = ('... "stream?assetId=c64gdwl1yrqwt&enableDotcomAds=true&globalContainerPolling=true'
            '&home=sport&language=en-GB&liveTextStreamId=215E26F2051B45B18304E6E06A470FB9'
            '&pageNumber=1&pageSize=20&pageUrl=%2Fsport%2Ffootball%2Flive%2Fc64gdwl1yrqwt" ...')
    assert bp.stream_key_from_html(html) == "215E26F2051B45B18304E6E06A470FB9"
    assert bp.stream_key_from_html("<html></html>") is None
    posts, total = bp.parse_stream(PRESSER_STREAM)
    assert total == 1 and posts[0]["fixture_label"] == "Bournemouth v Brentford (Sat, 15:00 BST)"
    assert "Collins" in posts[0]["text"] and posts[0]["published_utc"].startswith("2026-09-11T14:21")


def test_pressers_collect_page_is_incremental(monkeypatch):
    import json
    from acquire.core import http
    from acquire.sources import bbc_pressers as bp
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = storage.connect(path)
    storage.init(conn)
    bp.register(conn)
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if "/sport/football/live/" in url:
            return http.Response(url, 200, 'x "stream?assetId=c64g&liveTextStreamId=ABC123&pageNumber=1" y',
                                 http.utcnow(), "text/html")
        return http.Response(url, 200, json.dumps(PRESSER_STREAM), http.utcnow(), "application/json")

    monkeypatch.setattr(http, "get", fake_get)
    page = {"page_id": "c64g", "headline": "Recap", "date_published": "2026-09-11T08:14:43Z"}
    r = bp.collect_page(conn, page, progress=lambda m: None)
    assert r["posts"] == 1 and r["pages"] == 1
    assert conn.execute("SELECT stream_id, pages_done, pages_total, posts FROM acq_bbc_presser_page").fetchone()[:4] == ("ABC123", 1, 1, 1)
    n = len(calls)
    assert bp.collect_page(conn, page, progress=lambda m: None)["skipped"] is True
    assert len(calls) == n                          # nothing refetched
    conn.close()
    try:
        os.remove(path)
    except PermissionError:
        pass



def test_pressers_refresh_rereads_the_page_a_live_blog_is_still_writing(monkeypatch):
    """A Friday page is published in the morning and written all day, in blocks,
    as each manager takes his turn. `collect_page` stops at "every stream page
    read", so without a refresh the archive freezes at whatever had been posted
    when it first ran — which is what left the desk showing the morning only."""
    import copy
    import json
    from acquire.core import http
    from acquire.sources import bbc_pressers as bp
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = storage.connect(path)
    storage.init(conn)
    bp.register(conn)
    stream = copy.deepcopy(PRESSER_STREAM)

    def fake_get(url, **kw):
        if "/sport/football/live/" in url:
            return http.Response(url, 200, 'x "stream?assetId=c64g&liveTextStreamId=ABC123&pageNumber=1" y',
                                 http.utcnow(), "text/html")
        return http.Response(url, 200, json.dumps(stream), http.utcnow(), "application/json")

    monkeypatch.setattr(http, "get", fake_get)
    page = {"page_id": "c64g", "headline": "Recap", "date_published": date.today().isoformat()}
    assert bp.collect_page(conn, page, progress=lambda m: None)["posts"] == 1
    # the afternoon's managers are appended to the same stream page
    post = copy.deepcopy(stream["results"][0])
    post["urn"] = "urn:x:2"
    post["dates"] = {"firstPublished": "2026-09-11T16:05:00.000Z"}
    stream["results"].append(post)

    assert bp.collect_page(conn, page, progress=lambda m: None)["skipped"] is True
    assert conn.execute("SELECT COUNT(*) FROM acq_bbc_presser").fetchone()[0] == 1

    out = bp.refresh_live(conn, progress=lambda m: None)
    assert out["new"] == 1 and out["pages"] == 1
    assert conn.execute("SELECT COUNT(*) FROM acq_bbc_presser").fetchone()[0] == 2
    # and the page's own count is the truth, not a running total
    assert conn.execute("SELECT posts FROM acq_bbc_presser_page").fetchone()[0] == 2
    # nothing new: a refresh writes nothing and reports nothing
    assert bp.refresh_live(conn, progress=lambda m: None)["new"] == 0
    conn.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def test_pressers_refresh_needs_a_recent_page(monkeypatch):
    from acquire.sources import bbc_pressers as bp
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    conn = storage.connect(path)
    storage.init(conn)
    bp.register(conn)
    conn.execute("INSERT INTO acq_bbc_presser_page (page_id, headline, date_published, "
                 "observed_utc) VALUES ('old', 'Recap', '2026-01-01T08:00:00Z', '2026-01-01T09:00:00Z')")
    assert bp.refresh_live(conn, progress=lambda m: None) == {"pages": 0, "posts": 0, "new": 0}
    conn.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def test_parse_ratings_reads_the_embedded_player_rater():
    import json
    blob = {"data": {"article?x": {"data": {"content": {"model": {"blocks": [
        {"type": "playerRater", "model": {"playerRaterData": {"state": "CLOSED", "eventId": "s-abc",
         "homeTeam": {"fullName": "Coventry City", "players": {
             "starters": [{"id": "s-1", "shortName": "C. Rushworth", "shirtNumber": 19, "averageRating": 3.976}],
             "substitutes": [{"id": "s-2", "shortName": "S. Cherif", "shirtNumber": 49, "averageRating": 3.46}]}},
         "awayTeam": {"fullName": "Brighton & Hove Albion", "players": {
             "starters": [{"id": "s-3", "shortName": "B. Verbruggen", "shirtNumber": 1, "averageRating": 8.5}],
             "substitutes": []}}}}}]}}}}}}
    html = 'x<script>window.__INITIAL_DATA__="' + json.dumps(blob).replace('"', '\\"') + '";</script>y'
    rows, state = bbc.parse_ratings(html)
    assert state == "CLOSED" and len(rows) == 3
    by = {r["short_name"]: r for r in rows}
    assert by["B. Verbruggen"]["rating"] == 8.5 and by["B. Verbruggen"]["is_starter"] == 1
    assert by["S. Cherif"]["is_starter"] == 0 and by["S. Cherif"]["player_urn"].endswith(":s-2")
    assert bbc.parse_ratings("<html>no data</html>") == ([], None)



def test_parse_stats_reads_both_teams_and_the_set_play_split():
    payload = {"homeTeam": {"name": {"fullName": "West Ham United"}, "alignment": "home", "stats": {
        "possessionPercentage": {"total": 42.2}, "cornersWon": {"total": 6},
        "attack": {"shotsTotal": {"total": 16}, "shotsOnTarget": {"total": 9}, "shotsBlocked": {"total": 3}},
        "distribution": {"touchesInBox": {"total": 27}, "totalCross": {"total": 25}},
        "defence": {"foulsCommitted": {"total": 11}, "totalTackle": {"total": 12}, "totalClearance": {"total": 22}},
        "expected": {"goals": {"total": 2.574}, "goalsOpenplay": {"total": 1.9964}, "goalsSetplay": {"total": 0.5777},
                     "assists": {"total": 0.89}, "assistsSetplay": {"total": 0.25}, "assistsOpenplay": {"total": 0.64}},
        "fitness": {"distanceMetres": {"total": 100.46}, "sprintingPercentage": {"total": 6.68}}}},
        "awayTeam": {"name": {"fullName": "Leeds United"}, "alignment": "away", "stats": {"shotsTotal": {"total": 5}}}}
    rows = bbc.parse_stats(payload)
    assert [r["team"] for r in rows] == ["West Ham United", "Leeds United"]
    h, a = rows
    assert h["xg_set"] == 0.5777 and h["xg_open"] == 1.9964 and h["shots"] == 16 and h["corners"] == 6
    assert h["distance_km"] == 100.46 and h["sprint_pct"] == 6.68
    assert a["shots"] == 5 and a["xg"] is None
    assert bbc.parse_stats({}) == []


def test_parse_officials_names_the_referee():
    payload = {"officials": [
        {"id": "urn:bbc:sportsdata:football:official:abc", "type": "Referee", "lastName": "England", "firstName": "Darren"},
        {"id": "urn:bbc:sportsdata:football:official:def", "type": "Assistant referee 1", "lastName": "X"},
        {"type": "no id"}]}
    rows = bbc.parse_officials(payload)
    assert len(rows) == 2
    assert rows[0] == {"official_urn": "urn:bbc:sportsdata:football:official:abc", "role": "Referee", "name": "Darren England"}
