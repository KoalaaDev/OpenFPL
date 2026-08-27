"""Price-change model: panel construction and point-in-time discipline."""
import os
import tempfile

import numpy as np
import pytest

from fpl_engine import db, price_model as pm

SEASON = "2025-26"


class _StubClf:
    def __init__(self, proba=(0.1, 0.6, 0.3)):
        self.proba = proba

    def predict_proba(self, X):
        return np.tile(np.array(self.proba, float), (len(X), 1))


def _row(pid, gw, price, selected, tin, tout, pts=2, mins=90):
    return {"season": SEASON, "gw": gw, "source": "fpl", "player_id": pid,
            "fixture_id": gw, "player_code": 100 + pid, "full_name": f"P{pid}",
            "team_id": 1, "opponent_id": 2, "was_home": 1,
            "kickoff_utc": f"2025-09-{gw:02d}T14:00:00Z",
            "minutes": mins, "total_points": pts, "price": price,
            "selected": selected, "transfers_in": tin, "transfers_out": tout}


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    db.init_db(path)
    c = db.connect(path)
    db.upsert(c, "player", [
        {"season": SEASON, "player_id": 1, "code": 101, "full_name": "Riser",
         "team_id": 1, "position": "MID", "now_cost": 7.0},
        {"season": SEASON, "player_id": 2, "code": 102, "full_name": "Faller",
         "team_id": 1, "position": "FWD", "now_cost": 8.0},
    ])
    rows = []
    for gw in range(1, 6):
        rows.append(_row(1, gw, 70 + (gw - 1), 100_000, 50_000, 1_000))
        rows.append(_row(2, gw, 80 - (gw - 1), 100_000, 1_000, 50_000))
    db.upsert(c, "player_gw", rows)
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


def test_target_is_the_move_into_the_next_gameweek(conn):
    f = pm._frame(conn, [SEASON]).set_index(["player_id", "gw"])
    assert f.loc[(1, 1), "dp"] == pytest.approx(1.0)     # 7.0 -> 7.1
    assert f.loc[(2, 1), "dp"] == pytest.approx(-1.0)
    assert np.isnan(f.loc[(1, 5), "dp"])                 # no next gameweek yet
    assert f.loc[(1, 1), "label"] == 2 and f.loc[(2, 1), "label"] == 0


def test_features_never_use_the_future(conn):
    """Every feature of gameweek g must be computable at gameweek g."""
    f = pm._frame(conn, [SEASON]).set_index(["player_id", "gw"])
    # price history looks strictly backwards
    assert np.isnan(f.loc[(1, 1), "dp_lag1"])
    assert f.loc[(1, 2), "dp_lag1"] == pytest.approx(1.0)
    assert f.loc[(1, 3), "dp_lag2"] == pytest.approx(1.0)
    # a database that stops at gameweek 3 must produce the same gameweek-3
    # features as one that also holds gameweeks 4 and 5
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        db.init_db(path)
        c2 = db.connect(path)
        db.upsert(c2, "player", [dict(r) for r in conn.execute(
            "SELECT * FROM player WHERE season=?", (SEASON,))])
        db.upsert(c2, "player_gw", [dict(r) for r in conn.execute(
            "SELECT * FROM player_gw WHERE season=? AND gw<=3", (SEASON,))])
        c2.commit()
        short = pm._frame(c2, [SEASON]).set_index(["player_id", "gw"])
        c2.close()
    finally:
        try:
            os.remove(path)
        except PermissionError:
            pass
    for col in pm.FEATURES:
        if col == "gw":          # in the index, and trivially point-in-time
            continue
        a, b = f.loc[(1, 3), col], short.loc[(1, 3), col]
        assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b), col


def test_transfer_flow_is_scaled_by_ownership(conn):
    f = pm._frame(conn, [SEASON]).set_index(["player_id", "gw"])
    assert f.loc[(1, 1), "net_frac"] == pytest.approx(0.49)
    assert f.loc[(2, 1), "net_frac"] == pytest.approx(-0.49)


def test_deadline_feature_set_excludes_this_gameweeks_matches(conn):
    """The leak-proof variant must not carry anything about gw t's results."""
    assert not (set(pm.FEATURES_DEADLINE) & set(pm.FORM_FEATURES))
    assert set(pm.FEATURES_DEADLINE) < set(pm.FEATURES)


def test_predict_returns_probabilities_and_pounds(conn, monkeypatch):
    monkeypatch.setattr(pm, "load", lambda: (
        _StubClf((0.2, 0.5, 0.3)), {"features": pm.FEATURES}))
    out = pm.predict(conn, SEASON, gw=4)
    assert len(out) == 2
    assert np.allclose(out[["p_fall", "p_hold", "p_rise"]].sum(axis=1), 1.0)
    # e_delta is in £m and one tenth is the only move size FPL makes
    assert out["e_delta"].abs().max() <= 0.1 + 1e-9
    assert out["price_m"].between(4.0, 15.0).all()


def test_stale_feature_cache_is_rejected(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(pm, "MODEL_PATH", str(tmp_path / "m.json"))
    monkeypatch.setattr(pm, "META_PATH", str(tmp_path / "meta.json"))
    (tmp_path / "m.json").write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({"features": ["net_frac"]}),
                                        encoding="utf-8")
    assert pm.load() == (None, None)
