"""The market stretch for unpriced fixtures, and bookmaker-name resolution."""
import json
import math

from fpl_engine.ingest import oddschecker as oc
from fpl_engine.xpts import odds_model as om


def test_apply_stretch_widens_the_spread_and_clips():
    rec = {"a": -0.07, "b": 1.25}
    lo, hi = om.apply_stretch(0.9, rec), om.apply_stretch(2.0, rec)
    assert lo < 0.9 and hi > 2.0                          # gaps widen on both sides
    assert math.isclose(om.apply_stretch(1.0, rec), math.exp(-0.07))
    assert om.apply_stretch(0.01, rec) == 0.2 and om.apply_stretch(50.0, rec) == 4.0
    assert om.apply_stretch(0.0, rec) == 0.0


def test_load_market_stretch_honours_the_env_switch(tmp_path, monkeypatch):
    path = tmp_path / "market_stretch.json"
    path.write_text(json.dumps({"a": -0.07, "b": 1.25, "n": 2162}), encoding="utf-8")
    monkeypatch.delenv("FPL_MARKET_STRETCH", raising=False)
    assert om.load_market_stretch(str(path))["b"] == 1.25
    monkeypatch.setenv("FPL_MARKET_STRETCH", "0")
    assert om.load_market_stretch(str(path)) is None
    monkeypatch.delenv("FPL_MARKET_STRETCH", raising=False)
    assert om.load_market_stretch(str(tmp_path / "missing.json")) is None


def test_match_player_resolves_accents_and_refuses_ambiguity():
    squad = [
        {"player_id": 1, "web_name": "Ødegaard", "full_name": "Martin Ødegaard", "team_id": 1},
        {"player_id": 2, "web_name": "Groß", "full_name": "Pascal Groß", "team_id": 1},
        {"player_id": 3, "web_name": "João Pedro", "full_name": "João Pedro Junqueira de Jesus", "team_id": 2},
        {"player_id": 4, "web_name": "Silva", "full_name": "Bernardo Silva", "team_id": 1},
        {"player_id": 5, "web_name": "Silva", "full_name": "Fábio Silva", "team_id": 2},
    ]
    assert oc.match_player("Martin Odegaard", squad) == 1
    assert oc.match_player("Pascal Gross", squad) == 2
    assert oc.match_player("de Jesus Joao Pedro", squad) == 3
    assert oc.match_player("Bernardo Silva", squad) == 4
    assert oc.match_player("Silva", squad) is None            # two Silvas: never a guess
    assert oc.match_player("Somebody Unknown", squad) is None
