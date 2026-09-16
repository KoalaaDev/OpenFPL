"""The two keyless predicted-lineup feeds added on 2026-09-16, parsed on
markup shaped like the live pages."""
from acquire.sources import ffscout_lineups as ffs, sportsgambler_lineups as sg

FFS_PAGE = """
<h2 class="!m-0">   Watchlists  </h2>
<h2 class="!m-0">
    Aston Villa
</h2></header></div><div class="next-match"><strong>Next Match:</strong> Nottingham Forest (H)</div>
<div class="scout-picks scout-picks-pitch formation formation-4-2-3-1">
<ul class="row-1"><li            title="Suzuki (Zion)"><img src="x"><span class="player-name truncate">Suzuki</span></li></ul>
<ul class="row-2"><li title="Cash (Matthew)"><span class="player-name">Cash</span></li><li title="Lindelof"><span class="player-name">Lindelof</span></li></ul>
</div>
<ul class="story-parts">
<li class="headers"><strong>Out:</strong><ul class="players"><li>Goretzka</li><li>Mvom Onana</li></uL></li>
<li class="headers"><strong>Doubts:</strong><ul class="players"><li> Cash <span class="doubt-percent">75%</span></li></ul></li>
<li class="headers"><strong>Banned:</strong></li>
<li><p><strong>Latest News: </strong>Cash will be assessed after a &#39;minor issue&#39;.</p></li>
</ul>
<h2 class="!m-0"> Bournemouth </h2>
<div class="scout-picks scout-picks-pitch formation formation-4-4-2"><ul class="row-1"><li title="Kepa"><span class="player-name">Kepa</span></li></ul></div>
"""


def test_ffscout_parses_xis_and_team_news_per_club():
    lineups, news = ffs.parse(FFS_PAGE)
    avl = [r for r in lineups if r["team_abbr"] == "AVL"]
    assert [r["player"] for r in avl] == ["Suzuki (Zion)", "Cash (Matthew)", "Lindelof"]
    assert avl[0]["row"] == 1 and avl[2]["row"] == 2 and avl[0]["formation"] == "4-2-3-1"
    assert avl[0]["next_match"] == "Nottingham Forest (H)" and avl[0]["short"] == "Suzuki"
    assert [r["player"] for r in lineups if r["team_abbr"] == "BOU"] == ["Kepa"]
    kinds = {(r["kind"], r["player"]): r for r in news if r["team_abbr"] == "AVL"}
    assert ("out", "Goretzka") in kinds and ("out", "Mvom Onana") in kinds
    assert kinds[("doubt", "Cash")]["pct"] == 75
    assert kinds[("news", None)]["news"].startswith("Cash will be assessed after a 'minor issue'")
    assert not [r for r in lineups if r["team_abbr"] not in ("AVL", "BOU")]


SG_INDEX = """
<h3 class="date-headline">Friday 18 September</h3>
<div class="lineup-row"><div class="fxs-info"><span class="fxs-time">23:00</span><span class="fxs-league h-sm">Premier League</span></div>
<div class="fxs-game"><span class="fxs-team home">Brentford</span><span class="vs-teams"> vs </span><span class="fxs-team">Chelsea</span></div>
<div class="fxs-btn"><a href="#" rel="#lineup5795456" id="5795456"><span class="h-sm">Predicted Lineups</span></a></div></div>
<div class="lineup-row"><div class="fxs-info"><span class="fxs-time">20:00</span><span class="fxs-league h-sm">Championship</span></div>
<div class="fxs-game"><span class="fxs-team home">Luton</span><span class="vs-teams"> vs </span><span class="fxs-team">Derby</span></div>
<div class="fxs-btn"><a href="#" rel="#lineup1" id="1"><span class="h-sm">Confirmed Lineups</span></a></div></div>
<div class="lineup-row"><div class="fxs-info"><span class="fxs-time">15:00</span><span class="fxs-league h-sm">Premier League</span></div>
<div class="fxs-game"><span class="fxs-team home">Nott’m Forest</span><span class="vs-teams"> vs </span><span class="fxs-team">Man United</span></div>
<div class="fxs-btn"><a href="#" rel="#lineup2" id="2"><span class="h-sm">Confirmed Lineups</span></a></div></div>
"""

SG_LINEUP = """
<div class="lineups-formation"><h3><span>Brentford Predicted Lineup</span> <span class="lineups-toggle-formation">4-2-3-1</span></h3>
<h3><span>Chelsea Predicted Lineup</span> <span class="lineups-toggle-formation">3-4-3</span></h3></div>
<div class="lineups-container"><div class="lineups-home reverse">
<div class="players-line goalie"><span class="lineups-player"><span class="player-profile">1</span><span class="player-name">Caoimhin Kelleher</span></span></div>
<div class="players-line"><span class="lineups-player"><span class="player-profile">2</span><span class="player-name">Aaron Hickey</span></span>
<span class="lineups-player"><span class="player-profile">20</span><span class="player-name">K. Vassbakk Ajer</span></span></div>
</div><!--lineups home--><div class="lineups-away">
<div class="players-line goalie"><span class="lineups-player"><span class="player-profile">1</span><span class="player-name">Robert Sanchez</span></span></div>
</div><!--lineups away--></div>
"""


def test_sportsgambler_index_keeps_premier_league_fixtures_and_maps_clubs():
    fx = sg.parse_index(SG_INDEX)
    assert [f["id"] for f in fx] == ["5795456", "2"]           # the Championship row is dropped
    assert fx[0]["home_abbr"] == "BRE" and fx[0]["away_abbr"] == "CHE" and fx[0]["status"] == "predicted"
    assert fx[1]["home_abbr"] == "NFO" and fx[1]["away_abbr"] == "MUN" and fx[1]["status"] == "confirmed"
    assert fx[0]["date"] == "Friday 18 September" and fx[0]["time"] == "23:00"


def test_sportsgambler_lineup_gives_rows_slots_shirts_and_formations():
    lu = sg.parse_lineup(SG_LINEUP)
    assert lu["formation"] == {"home": "4-2-3-1", "away": "3-4-3"}
    assert [(r["row"], r["slot"], r["shirt"], r["player"]) for r in lu["home"]] == [
        (1, 1, 1, "Caoimhin Kelleher"), (2, 2, 2, "Aaron Hickey"), (2, 3, 20, "K. Vassbakk Ajer")]
    assert lu["away"] == [{"row": 1, "slot": 1, "shirt": 1, "player": "Robert Sanchez"}]
    assert sg.parse_lineup("<div>nothing</div>") == {"home": [], "away": [], "formation": {}}
