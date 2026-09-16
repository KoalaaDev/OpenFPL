"""Oddschecker match pages: the embedded market JSON -> de-margined 1X2, the
scoreline distribution, per-player anytime prices."""
import json

import pytest

from fpl_engine.ingest import oddschecker as oc


def _page(home="Brentford", away="Chelsea"):
    bets = {
        "1": {"ocBetId": 1, "betName": away, "marketId": 10, "line": None, "probability": 42},
        "2": {"ocBetId": 2, "betName": home, "marketId": 10, "line": None, "probability": 34},
        "3": {"ocBetId": 3, "betName": "Draw", "marketId": 10, "line": None, "probability": 24},
        # correct score: winner's goals first; a draw is symmetric
        "4": {"ocBetId": 4, "betName": "Draw", "marketId": 20, "line": "0-0"},
        "5": {"ocBetId": 5, "betName": home, "marketId": 20, "line": "1-0"},
        "6": {"ocBetId": 6, "betName": away, "marketId": 20, "line": "2-1"},
        "7": {"ocBetId": 7, "betName": "Draw", "marketId": 20, "line": "2-2"},
        "8": {"ocBetId": 8, "betName": away, "marketId": 20, "line": "1-0"},
        "9": {"ocBetId": 9, "betName": home, "marketId": 20, "line": "3-1"},
        "10": {"ocBetId": 10, "betName": "Draw", "marketId": 20, "line": "1-1"},
        "11": {"ocBetId": 11, "betName": away, "marketId": 20, "line": "3-0"},
        "12": {"ocBetId": 12, "betName": home, "marketId": 20, "line": "2-0"},
        "13": {"ocBetId": 13, "betName": away, "marketId": 20, "line": "2-0"},
        "14": {"ocBetId": 14, "betName": "Igor Thiago", "marketId": 30, "line": None, "probability": 46.4},
        "15": {"ocBetId": 15, "betName": "Cole Palmer", "marketId": 30, "line": None, "probability": None},
    }
    def bk(*decimals):
        return {f"B{i}": {"oddsDecimal": d, "status": "ACTIVE"} for i, d in enumerate(decimals)}
    odds = {"1": bk(2.2, 2.3, 2.25), "2": bk(2.8, 2.9), "3": bk(3.7, 3.8, 3.75),
            "4": bk(12.0), "5": bk(9.0), "6": bk(9.0), "7": bk(11.0), "8": bk(8.0), "9": bk(20.0),
            "10": bk(7.0), "11": bk(25.0), "12": bk(15.0), "13": bk(13.0),
            "14": bk(2.05, 2.31, 2.0), "15": bk(2.5, 2.4, 3.0)}
    markets = {"10": {"ocMarketId": 10, "marketTypeName": "Win Market"},
               "20": {"ocMarketId": 20, "marketTypeName": "Correct Score"},
               "30": {"ocMarketId": 30, "marketTypeName": "Anytime Goalscorer"},
               "40": {"ocMarketId": 40, "marketTypeName": "Asian Handicap"}}
    mk = {"bestOdds": {"bets": {"entities": bets}, "odds": odds, "markets": {"entities": markets}}}
    hdr = {"subeventName": f"{home} vs {away}", "subeventStartTime": "2026-09-18T19:00:00Z",
           "siblingSubevents": [{"subeventUrl": "football/english/premier-league/tottenham-v-aston-villa/winner"}]}
    return ('<html><script type="application/json" data-hypernova-key="subeventheader" data-hypernova-id="x"><!--'
            + json.dumps(hdr) + '--></script><script type="application/json" data-hypernova-key="subeventmarkets" '
            'data-hypernova-id="y"><!--' + json.dumps(mk).replace("<", "&lt;") + '--></script></html>')


def test_parse_index_finds_match_pages_once_each():
    html = ('<a href="football/english/premier-league/brentford-v-chelsea/winner">20:00</a>'
            '<a href="football/english/premier-league/brentford-v-chelsea/winner">Brentford v Chelsea</a>'
            '<a href="/football/english/premier-league/tottenham-v-aston-villa/winner">x</a>'
            '<a href="/football/english/premier-league/winner">outright</a>')
    assert oc.parse_index(html) == [
        "https://www.oddschecker.com/football/english/premier-league/brentford-v-chelsea/winner",
        "https://www.oddschecker.com/football/english/premier-league/tottenham-v-aston-villa/winner"]


def test_parse_and_derive_a_match_page():
    m = oc.parse_match(_page())
    assert m["home"] == "Brentford" and m["away"] == "Chelsea" and m["kickoff_utc"].startswith("2026-09-18")
    assert {r["name"] for r in m["win"]} == {"Brentford", "Chelsea", "Draw"}
    assert len(m["correct_score"]) == 10 and len(m["anytime"]) == 2
    thiago = next(r for r in m["anytime"] if r["name"] == "Igor Thiago")
    assert thiago["median"] == 2.05 and thiago["best"] == 2.31 and thiago["n"] == 3
    d = oc.derive(m)
    assert abs(d["p_home"] + d["p_draw"] + d["p_away"] - 1.0) < 1e-9
    assert d["p_away"] > d["p_home"] > d["p_draw"]                # Chelsea favourites at 2.25 vs 2.85
    # scoreline orientation: "Chelsea 2-1" is home 1, away 2; "Brentford 1-0" is home 1, away 0
    assert 0 < d["p_cs_home"] < 1 and 0 < d["p_cs_away"] < 1 and 0 < d["p_over25"] < 1
    # both-score, home clean sheet and away clean sheet partition the scorelines,
    # except that 0-0 is a clean sheet for both: the three sum to 1 + P(0-0)
    assert 1.0 < d["p_btts"] + d["p_cs_home"] + d["p_cs_away"] < 1.2
    assert d["lam_home"] > 0 and d["lam_away"] > d["lam_home"]
    assert next(a for a in d["anytime"] if a["player"] == "Igor Thiago")["prob"] == pytest.approx(1 / 2.05)


def test_ingest_writes_match_odds_and_props(tmp_path, monkeypatch):
    import sqlite3
    from fpl_engine import config, db
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    path = tmp_path / "t.sqlite"
    db.init_db(str(path))
    c = db.connect(str(path))
    db.upsert(c, "team", [{"season": "2026-27", "team_id": 4, "name": "Brentford", "code": 94},
                          {"season": "2026-27", "team_id": 6, "name": "Chelsea", "code": 8}])
    db.upsert(c, "fixture", [{"season": "2026-27", "fixture_id": 41, "gw": 5,
                              "kickoff_utc": "2026-09-18T19:00:00Z", "team_h": 4, "team_a": 6}])
    url = "https://www.oddschecker.com/football/english/premier-league/brentford-v-chelsea/winner"
    res = oc.ingest(c, "2026-27", pages={url: _page()})
    assert res["fixtures"] == 1 and res["unresolved"] == [] and res["props"] == 5
    row = c.execute("SELECT source, p_home, p_over25, lam_home FROM match_odds WHERE fixture_id=41").fetchone()
    assert row["source"] == "oddschecker" and 0 < row["p_home"] < 1 and row["lam_home"] > 0
    kinds = {r["kind"] for r in c.execute("SELECT kind FROM market_prop WHERE fixture_id=41")}
    assert kinds == {"cs_home", "cs_away", "btts", "anytime"}
    assert (tmp_path / "collected" / "oddschecker" / "2026-27.csv").exists()
