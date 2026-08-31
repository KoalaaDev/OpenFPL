"""xpts minutes model: point-in-time discipline and train/serve parity.

The features that serve a prediction are built by the same code that builds the
training rows — these tests pin that down, because a silent divergence between
the two paths degrades every prediction without failing anything.
"""
import os
import tempfile

import numpy as np
import pytest

from fpl_engine import db
from fpl_engine.xpts import minutes_model as mm

SEASON = "2025-26"
PREV = "2024-25"


class _StubClf:
    """Returns a fixed distribution so predict_gw can be tested without XGBoost."""
    def __init__(self, proba=(0.1, 0.2, 0.7)):
        self.proba = proba
        self.seen = None

    def predict_proba(self, X):
        self.seen = X
        return np.tile(np.array(self.proba, float), (len(X), 1))


class _StubStart:
    """Fixed P(start)."""
    def __init__(self, p=0.8):
        self.p = p

    def predict_proba(self, X):
        return np.tile(np.array([1 - self.p, self.p], float), (len(X), 1))


class _StubReg:
    """Fixed E[minutes | plays]."""
    def __init__(self, value=70.0):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _pgw(pid, code, gw, mins, starts, day, team=1, fixture=None, month=9):
    return {"season": SEASON, "gw": gw, "source": "fpl", "player_id": pid,
            "fixture_id": fixture if fixture is not None else gw,
            "player_code": code, "full_name": f"P{pid}", "team_id": team,
            "opponent_id": 2 if team == 1 else 1, "was_home": 1,
            "kickoff_utc": f"2025-{month:02d}-{day:02d}T14:00:00Z",
            "minutes": mins, "total_points": 2, "starts": starts,
            "price": 10 * (50 + pid), "selected": 1000 * pid,
            "transfers_in": 10, "transfers_out": 5}


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db.init_db(path)
    c = db.connect(path)
    db.upsert(c, "team", [
        {"season": SEASON, "team_id": 1, "name": "Alpha", "code": 11},
        {"season": SEASON, "team_id": 2, "name": "Beta", "code": 22},
        {"season": SEASON, "team_id": 3, "name": "Gamma", "code": 33},
    ])
    db.upsert(c, "player", [
        # a 90-minute man and one habitually hooked on the hour
        {"season": SEASON, "player_id": 1, "code": 100, "full_name": "Ninety",
         "team_id": 1, "position": "MID", "now_cost": 51.0, "status": "a",
         "chance_next": None},
        {"season": SEASON, "player_id": 2, "code": 200, "full_name": "Hooked",
         "team_id": 1, "position": "MID", "now_cost": 52.0, "status": "a",
         "chance_next": None},
        # plays for a club with no fixture in the target gameweek
        {"season": SEASON, "player_id": 3, "code": 300, "full_name": "Blank",
         "team_id": 3, "position": "MID", "now_cost": 53.0, "status": "a",
         "chance_next": None},
        # flagged 50%
        {"season": SEASON, "player_id": 4, "code": 400, "full_name": "Doubt",
         "team_id": 1, "position": "MID", "now_cost": 54.0, "status": "d",
         "chance_next": 0.5},
    ])
    rows = []
    for i, day in enumerate((1, 8, 15, 22, 29)):
        rows.append(_pgw(1, 100, i + 1, 90, 1, day))
        rows.append(_pgw(2, 200, i + 1, 62, 1, day))
        rows.append(_pgw(3, 300, i + 1, 90, 1, day, team=3, fixture=100 + i))
        rows.append(_pgw(4, 400, i + 1, 90, 1, day))
    db.upsert(c, "player_gw", rows)
    # target gameweek 6: only teams 1 and 2 play
    db.upsert(c, "fixture", [{"season": SEASON, "fixture_id": 6, "gw": 6,
                              "kickoff_utc": "2025-10-06T14:00:00Z",
                              "team_h": 1, "team_a": 2}])
    # a match AFTER the target that must never influence its features
    rows_future = [_pgw(1, 100, 7, 0, 0, 13, month=10)]
    db.upsert(c, "player_gw", rows_future)
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


AS_OF = "2025-10-06T00:00:00Z"


def test_target_features_ignore_matches_after_as_of(conn):
    """The 0-minute match a week later must not touch the target row."""
    df = mm._frame(conn, [SEASON], before=AS_OF, target=(SEASON, 6, AS_OF))
    t = df[(df["_target"] == 1) & (df["player_id"] == 1)].iloc[0]
    assert t["mins_l1"] == 90.0
    assert t["mins_l5"] == 90.0
    assert t["starts_l5"] == 5.0
    assert t["since_last_app"] == 0.0


def test_serving_features_match_the_training_row(conn):
    """A target row and the real row for the same match must agree.

    Built two ways: as a synthetic target (the live path) and as an ordinary
    history row (the training path). Every rolling feature must line up.
    """
    # every squad member of the playing club gets a row, as the real feeds do
    db.upsert(conn, "player_gw",
              [_pgw(1, 100, 6, 77, 1, 6, month=10, fixture=6),
               _pgw(2, 200, 6, 55, 1, 6, month=10, fixture=6),
               _pgw(4, 400, 6, 90, 1, 6, month=10, fixture=6)])
    conn.commit()
    live = mm._frame(conn, [SEASON], before=AS_OF, target=(SEASON, 6, AS_OF))
    live = live[live["_target"] == 1].set_index("player_id")
    real = mm._frame(conn, [SEASON])
    real = real[real["gw"] == 6].set_index("player_id")
    for pid in (1, 2, 4):
        for f in mm.FEATURES:
            a, b = live.loc[pid, f], real.loc[pid, f]
            if np.isnan(a) and np.isnan(b):
                continue
            assert a == pytest.approx(b), f"{f} differs for player {pid}"


def test_expected_minutes_use_the_player_not_the_league(conn):
    """A 62-minute starter must not be handed the league's 84-minute average."""
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0}}
    out = mm.predict_gw(conn, SEASON, AS_OF, _StubClf((0.0, 0.0, 1.0)), meta,
                        gw=6, use_availability=False).set_index("player_id")
    assert out.loc[1, "e_min"] == pytest.approx(90.0)
    assert out.loc[2, "e_min"] == pytest.approx(62.0)


def test_blank_gameweek_player_still_gets_a_row(conn):
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0}}
    out = mm.predict_gw(conn, SEASON, AS_OF, _StubClf(), meta, gw=6,
                        use_availability=False).set_index("player_id")
    assert 3 in out.index                      # club has no fixture in gw6
    assert out.loc[3, "p_none"] == 1.0
    assert out.loc[3, "e_min"] == 0.0


def test_availability_overlay_scales_the_played_mass(conn):
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0}}
    stub = _StubClf((0.0, 0.0, 1.0))
    off = mm.predict_gw(conn, SEASON, AS_OF, stub, meta, gw=6,
                        use_availability=False).set_index("player_id")
    on = mm.predict_gw(conn, SEASON, AS_OF, stub, meta, gw=6,
                       use_availability=True).set_index("player_id")
    assert off.loc[4, "p_full"] == pytest.approx(1.0)
    assert on.loc[4, "p_full"] == pytest.approx(0.5)     # chance_next = 0.5
    assert on.loc[1, "p_full"] == pytest.approx(1.0)     # fit player untouched


def test_stale_feature_cache_is_rejected(tmp_path, monkeypatch):
    """A model cached before a feature change must not be loaded silently."""
    import json
    monkeypatch.setattr(mm, "META_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(mm, "MODEL_PATH", str(tmp_path / "model.json"))
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({"features": ["starts_l5"], "mean_minutes": {}}),
        encoding="utf-8")
    assert mm.load() == (None, None)


def test_hybrid_expected_minutes_is_p_play_times_conditional(conn):
    """E[min] = P(plays) x E[min | plays], not a blend of two class means.

    The reconstruction it replaced was worth 0.43 / 0.47 minutes of MAE over
    two replayed seasons, and exposure multiplies every rate in the engine.
    """
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0},
            "_reg": _StubReg(70.0)}
    out = mm.predict_gw(conn, SEASON, AS_OF, _StubClf((0.4, 0.1, 0.5)), meta,
                        gw=6, use_availability=False).set_index("player_id")
    assert out.loc[1, "e_min"] == pytest.approx(0.6 * 70.0)
    # and the class-mean path is still there for a cache without a regressor
    plain = mm.predict_gw(conn, SEASON, AS_OF, _StubClf((0.4, 0.1, 0.5)),
                          {k: v for k, v in meta.items() if k != "_reg"},
                          gw=6, use_availability=False).set_index("player_id")
    assert plain.loc[1, "e_min"] == pytest.approx(0.1 * 30.0 + 0.5 * 90.0)


def test_load_rejects_a_cache_without_the_regressor(tmp_path, monkeypatch):
    """An old two-file cache must retrain, not silently lose the regressor."""
    import json
    monkeypatch.setattr(mm, "META_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(mm, "MODEL_PATH", str(tmp_path / "model.json"))
    monkeypatch.setattr(mm, "REG_PATH", str(tmp_path / "reg.json"))
    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({"features": mm.FEATURES, "mean_minutes": {}}),
        encoding="utf-8")
    assert mm.load() == (None, None)      # reg.json missing


def test_p_start_is_published_for_every_player(conn):
    """The question people actually ask is "will he start", not "will he reach
    60 minutes". The engine had the signal and only exposed the latter."""
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0},
            "_start": _StubStart(0.77)}
    out = mm.predict_gw(conn, SEASON, AS_OF, _StubClf(), meta, gw=6,
                        use_availability=False).set_index("player_id")
    assert "p_start" in out.columns
    assert out.loc[1, "p_start"] == pytest.approx(0.77)
    # a club with no fixture this gameweek cannot field anyone
    assert out.loc[3, "p_start"] == 0.0


def test_p_start_falls_back_when_no_start_model_is_cached(conn):
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0}}
    out = mm.predict_gw(conn, SEASON, AS_OF, _StubClf((0.1, 0.2, 0.7)), meta,
                        gw=6, use_availability=False).set_index("player_id")
    assert out.loc[1, "p_start"] == pytest.approx(0.7)


def test_availability_scales_p_start_too(conn):
    """A player ruled out cannot start, and the overlay must reach that column
    as well as the minutes ones."""
    meta = {"features": mm.FEATURES, "mean_minutes": {"sub": 30.0, "full": 84.0},
            "_start": _StubStart(0.9)}
    on = mm.predict_gw(conn, SEASON, AS_OF, _StubClf(), meta, gw=6,
                       use_availability=True).set_index("player_id")
    assert on.loc[4, "p_start"] == pytest.approx(0.45)   # chance_next = 0.5
    assert on.loc[1, "p_start"] == pytest.approx(0.9)


def test_a_stale_two_model_cache_is_rejected(tmp_path, monkeypatch):
    """Adding the start model must invalidate caches that predate it."""
    import json
    monkeypatch.setattr(mm, "META_PATH", str(tmp_path / "meta.json"))
    monkeypatch.setattr(mm, "MODEL_PATH", str(tmp_path / "model.json"))
    monkeypatch.setattr(mm, "REG_PATH", str(tmp_path / "reg.json"))
    monkeypatch.setattr(mm, "START_PATH", str(tmp_path / "start.json"))
    for f in ("model.json", "reg.json"):
        (tmp_path / f).write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        json.dumps({"features": mm.FEATURES, "mean_minutes": {}}),
        encoding="utf-8")
    assert mm.load() == (None, None)      # start.json missing


def test_fixture_congestion_is_counted_in_days_not_in_the_frames_time_unit():
    """pandas keeps whatever resolution a timestamp was parsed at.

    Since pandas 2.0 an ISO8601 string parses to MICROseconds, so the old
    ``astype("int64") / 86_400e9`` returned days/1000. ``days_rest`` survived
    it (a tree reads only the order), but a 14-day window that really means
    14,000 days counts every previous match of the season — the congestion
    feature was a gameweek counter.
    """
    import pandas as pd
    from fpl_engine.xpts import minutes_model as mm

    kicks = pd.to_datetime(pd.Series([
        "2025-08-16T14:00:00Z", "2025-08-19T19:00:00Z",   # 3 days later
        "2025-08-23T14:00:00Z", "2025-10-01T14:00:00Z",   # then a long gap
    ]), utc=True, format="ISO8601")
    tm = pd.DataFrame({"season": "2025-26", "team_id": 1,
                       "fixture_id": [1, 2, 3, 4], "kick": kicks})
    out = mm._team_congestion(tm).sort_values("kick")
    assert list(out["days_rest"].round(2)[1:]) == [3.21, 3.79, 39.0]
    # three matches inside a fortnight, then a lone one after the gap
    assert list(out["team_matches_14d"]) == [0.0, 1.0, 2.0, 0.0]
