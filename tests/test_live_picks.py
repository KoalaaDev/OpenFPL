"""The Live desk's model sections: the best legal XI and the picks by fixture.

Both are displays of projections that already exist, so what has to hold is
that they obey the rules they claim to (a legal FPL XI, the club cap, the
captain counted twice) and that a player's label comes from the engine's own
components rather than from his position.
"""
import pytest

from app import live


def P(pid, pos, team, ep, **comp):
    c = {"goals": 0, "assists": 0, "bonus": 0, "cs": 0, "defcon": 0, "saves": 0,
         "eg": 0, "ea": 0, "pcs": 0, **comp}
    return {"player_id": pid, "name": f"p{pid}", "team_id": team, "pos": pos,
            "price": 5.0, "ep": ep, "xmins": 85, "own": 10.0, "pk": False,
            "status": "a", "chance": None, "tag": live._tag(pos, c),
            "att": c["goals"] + c["assists"], "xgi": c["eg"] + c["ea"],
            "eg": c["eg"], "ea": c["ea"], "defcon": c["defcon"],
            "p_defcon": min(1.0, c["defcon"] / 2), "cs_pts": c["cs"],
            "p_cs": c["pcs"], "bonus": c["bonus"], "saves": c["saves"]}


def pool(n_per_club=6, clubs=6):
    out, pid = [], 0
    for club in range(1, clubs + 1):
        for pos, k in (("GK", 1), ("DEF", 2), ("MID", 2), ("FWD", 1)):
            for i in range(k):
                pid += 1
                out.append(P(pid, pos, club, 10.0 - club - i * 0.1))
    return out


# ---------------------------------------------------------------- the XI --
def test_best_xi_is_a_legal_fpl_eleven():
    xi = live.best_xi(pool())
    players = [p for row in xi["rows"] for p in row]
    assert len(players) == 11
    n = {pos: sum(p["pos"] == pos for p in players) for pos in ("GK", "DEF", "MID", "FWD")}
    assert n["GK"] == 1 and 3 <= n["DEF"] <= 5 and 2 <= n["MID"] <= 5 and 1 <= n["FWD"] <= 3
    assert xi["formation"] == f"{n['DEF']}-{n['MID']}-{n['FWD']}"


def test_no_more_than_three_from_a_club():
    """A greedy pick of the top eleven would take six from the best club."""
    ps = [P(i, pos, 1, 20.0) for i, pos in enumerate(
        ["GK", "DEF", "DEF", "DEF", "MID", "MID", "MID", "FWD", "FWD"], start=1)]
    ps += pool()
    xi = live.best_xi(ps)
    from collections import Counter
    assert max(Counter(p["team_id"] for row in xi["rows"] for p in row).values()) <= 3


def test_the_captain_is_the_top_projection_and_counted_twice():
    ps = pool()
    ps.append(P(999, "FWD", 7, 15.0))
    xi = live.best_xi(ps)
    players = [p for row in xi["rows"] for p in row]
    cap = next(p for p in players if p["captain"])
    assert cap["player_id"] == 999
    assert xi["points"] == pytest.approx(sum(p["ep"] for p in players) + 15.0, abs=0.06)
    assert sum(p["vice"] for p in players) == 1 and not cap["vice"]


def test_too_few_players_is_none_not_a_crash():
    assert live.best_xi(pool()[:5]) is None


# ---------------------------------------------------------------- labels --
def test_the_label_comes_from_the_components_not_the_position():
    """A defender whose projection is mostly goals and assists is an attacking
    pick; a midfielder whose projection is mostly DefCon is a DefCon pick."""
    assert live._tag("DEF", {"goals": 1.2, "assists": 0.8, "cs": 1.0, "defcon": 0.3}) == "ATT"
    assert live._tag("MID", {"goals": 0.2, "assists": 0.1, "cs": 0.0, "defcon": 1.1}) == "DEFCON"
    assert live._tag("DEF", {"goals": 0.1, "assists": 0.1, "cs": 1.6, "defcon": 0.4}) == "CS"
    assert live._tag("GK", {"cs": 1.2, "saves": 0.4}) == "CS"
    assert live._tag("GK", {"cs": 0.3, "saves": 0.9}) == "SAVES"


# ------------------------------------------------------ picks by fixture --
def _grid(monkeypatch, cells):
    from app import services
    monkeypatch.setattr(services, "fixtures_payload", lambda: {"grid": cells})


def test_clubs_are_ranked_by_expected_goal_difference(monkeypatch):
    ps = [P(1, "FWD", 1, 6, goals=3, eg=0.7), P(2, "FWD", 2, 6, goals=3, eg=0.7)]
    _grid(monkeypatch, {
        "1": {"5": [{"opp": 2, "home": True, "odds": {"xg": 1.1, "xg_against": 1.4}}]},
        "2": {"5": [{"opp": 1, "home": False, "odds": {"xg": 2.4, "xg_against": 0.6}}]},
    })
    out = live.fixture_picks(5, ps)
    assert [c["team_id"] for c in out] == [2, 1]
    assert out[0]["p_cs"] == pytest.approx(0.55, abs=0.01)     # exp(-0.6)


def test_attacking_picks_come_before_defcon_and_clean_sheet_picks(monkeypatch):
    ps = [P(1, "DEF", 1, 7.0, cs=2.0, defcon=0.6, pcs=0.5),          # highest ep, but CS
          P(2, "MID", 1, 5.5, goals=2.0, assists=1.0, eg=0.5, ea=0.3),
          P(3, "MID", 1, 4.0, defcon=1.4, goals=0.2),
          P(4, "FWD", 1, 6.0, goals=3.0, eg=0.8)]
    _grid(monkeypatch, {"1": {"5": [{"opp": 2, "home": True,
                                     "odds": {"xg": 2.0, "xg_against": 0.8}}]}})
    club = live.fixture_picks(5, ps)[0]
    assert [p["player_id"] for p in club["attack"]] == [4, 2]     # by ep, attackers only
    assert [p["player_id"] for p in club["defence"]] == [1, 3]


def test_a_benchwarmer_is_not_a_pick(monkeypatch):
    bench = P(1, "FWD", 1, 6.0, goals=3.0)
    bench["xmins"] = 20
    _grid(monkeypatch, {"1": {"5": [{"opp": 2, "home": True,
                                     "odds": {"xg": 2.0, "xg_against": 0.8}}]}})
    assert live.fixture_picks(5, [bench])[0]["attack"] == []


def test_a_blank_gameweek_club_is_left_out(monkeypatch):
    _grid(monkeypatch, {"1": {"5": []}})
    assert live.fixture_picks(5, [P(1, "FWD", 1, 6.0, goals=3)]) == []


def test_an_unpriced_fixture_still_ranks_and_says_so(monkeypatch):
    _grid(monkeypatch, {"1": {"5": [{"opp": 2, "home": True}]}})
    club = live.fixture_picks(5, [P(1, "FWD", 1, 6.0, goals=3, eg=0.9)])[0]
    assert club["priced"] is False and club["xga"] == live.LEAGUE_GOALS_PER_TEAM
