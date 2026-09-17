"""The player card's "where the points come from" adds up to the projection."""
from fastapi.testclient import TestClient

from app import services

CACHE = {"players": {"7": {
    "position": "MID",
    "ep": {"5": 6.3, "6": 4.0},
    "comp": {"5": {"goals": 2.1, "assists": 0.9, "bonus": 0.8, "appearance": 1.9, "cs": 0.4,
                   "defcon": 0.1, "saves": 0.0, "conceded": 0.0, "cards": -0.1,
                   "eg": 0.42, "ea": 0.21, "pcs": 0.38},
             "6": {"goals": 1.0, "assists": 0.5, "bonus": 0.4, "appearance": 1.8, "cs": 0.1,
                   "defcon": 0.1, "saves": 0.0, "conceded": 0.0, "cards": -0.1}},
}}}


def test_parts_and_other_sum_to_the_projection(monkeypatch):
    monkeypatch.setattr(services, "_load_proj_cache", lambda: CACHE)
    out = services.player_breakdown(7)
    assert list(out["gws"]) == ["5", "6"]
    for g, w in out["gws"].items():
        assert round(sum(w["parts"].values()) + w["other"], 2) == CACHE["players"]["7"]["ep"][g]
    assert out["gws"]["5"]["other"] == 0.2
    assert out["gws"]["5"]["xg"] == 0.42 and out["gws"]["5"]["p_cs"] == 0.38


def test_unknown_player_is_none(monkeypatch):
    monkeypatch.setattr(services, "_load_proj_cache", lambda: CACHE)
    assert services.player_breakdown(99) is None


def test_endpoint(monkeypatch):
    from app.main import app
    monkeypatch.setattr(services, "_load_proj_cache", lambda: CACHE)
    c = TestClient(app)
    assert c.get("/api/player/7/breakdown").json()["position"] == "MID"
    assert c.get("/api/player/99/breakdown").status_code == 404
    assert c.get("/api/player/0/breakdown").status_code == 400
