"""Phase 2.5 exploitation infrastructure: errors, sensitivity, decay.

These modules are how systematic champion errors get discovered, so their
own failure modes (misclassified misses, a stability number that ignores
the draws, decay buckets that leak post-deadline snapshots) are tested
directly on synthetic data.
"""
import gzip
import os

import numpy as np
import pandas as pd
import pytest

from fpl_engine import db as fdb, decay, errors
from fpl_engine.xpts import sensitivity

POSITIONS = (["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)


def test_error_classification_covers_the_causal_cases():
    mk = lambda e_min, minutes, err: {"e_min": e_min, "minutes": minutes,  # noqa: E731
                                      "err": err}
    assert errors.classify(mk(70, 0, -4)) == "did_not_play"
    assert errors.classify(mk(5, 90, 8)) == "unexpected_appearance"
    assert errors.classify(mk(85, 30, -2)) == "under_minutes"
    assert errors.classify(mk(80, 90, 9)) == "haul_missed"
    assert errors.classify(mk(80, 90, -5)) == "blank_despite_minutes"
    assert errors.classify(mk(80, 88, 0.5)) == "ok"
    assert errors.classify(mk(25, 30, 4)) == "other"


@pytest.fixture()
def conn(tmp_path):
    path = os.path.join(str(tmp_path), "t.sqlite")
    fdb.init_db(path)
    c = fdb.connect(path)
    yield c
    c.close()


def _seed_gw(conn, season="2025-26", gw=7):
    for pid, code, nm, pos, tid in ((1, 101, "A", "MID", 1),
                                    (2, 102, "B", "FWD", 2)):
        conn.execute(
            "INSERT INTO player (season, player_id, code, web_name, "
            "full_name, position, team_id) VALUES (?,?,?,?,?,?,?)",
            (season, pid, code, nm, nm, pos, tid))
    for pid, mins, pts in ((1, 90, 12), (2, 0, 0)):
        conn.execute(
            "INSERT INTO player_gw (season, gw, source, player_id, "
            "fixture_id, minutes, total_points, price, kickoff_utc) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (season, gw, "t", pid, pid, mins, pts, 75,
             "2025-10-04T14:00:00Z"))
    conn.commit()


def test_error_recording_is_idempotent_and_classified(conn):
    _seed_gw(conn)
    preds = pd.DataFrame({
        "player_id": [1, 2], "prediction": [4.0, 5.0],
        "e_min": [80.0, 70.0], "p_play": [0.95, 0.9], "p_60": [0.9, 0.8],
        "position": ["MID", "FWD"], "team_id": [1, 2]})
    assert errors.record_gw(conn, "2025-26", 7, preds) == 2
    assert errors.record_gw(conn, "2025-26", 7, preds) == 2   # replace, not dup
    rows = conn.execute("SELECT player_id, err, class FROM model_error "
                        "ORDER BY player_id").fetchall()
    assert tuple(rows[0]) == (1, 8.0, "haul_missed")
    assert tuple(rows[1]) == (2, -5.0, "did_not_play")
    res = errors.analyse(conn, "2025-26")
    assert res["rows"] == 2
    assert res["classes"] == {"haul_missed": 1, "did_not_play": 1}


def _players():
    return pd.DataFrame({"player_id": list(range(100, 115)),
                         "position": POSITIONS})


def test_sensitivity_flags_the_coin_flip_and_trusts_the_landslide():
    rng = np.random.default_rng(0)
    n = 3000
    players = _players()
    squad = players["player_id"].tolist()
    base = np.full((n, 15), 2.0) + rng.normal(0, 0.5, (n, 15))
    # landslide captain: one player far above the rest
    clear = base.copy()
    clear[:, 7] += 4.0
    r1 = sensitivity.analyse_squad(clear, players, squad, n_boot=100)
    assert r1["captain"] == 107
    assert r1["captain_margin"] > 3
    assert r1["captain_stability"] == 1.0
    # coin flip: two players tied, noisy
    tied = base.copy()
    tied[:, 7] += 3.0 + rng.normal(0, 3, n)
    tied[:, 8] += 3.0 + rng.normal(0, 3, n)
    r2 = sensitivity.analyse_squad(tied, players, squad, n_boot=100)
    assert abs(r2["captain_margin"]) < 0.5
    assert r2["captain_stability"] < 0.95
    assert r2["fragile"]


def test_sensitivity_swap_margins_are_position_legal():
    players = _players()
    squad = players["player_id"].tolist()
    draws = np.tile(np.arange(15, dtype=float), (200, 1))
    res = sensitivity.analyse_squad(draws, players, squad, n_boot=10)
    # every reported swap keeps a legal XI: a GK never swaps with an
    # outfielder — swap partners always share legality, and margins are
    # non-negative for a max-xP XI
    pos = dict(zip(squad, POSITIONS))
    for s in res["tightest_swaps"]:
        assert (pos[s["out"]] == "GK") == (pos[s["in"]] == "GK")
        assert s["margin"] >= 0


def _snap(dirpath, stamp, gw, rows):
    os.makedirs(dirpath, exist_ok=True)
    path = os.path.join(dirpath, f"{stamp}_gw{gw:02d}.csv.gz")
    df = pd.DataFrame(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        df.to_csv(fh, index=False)


def test_decay_buckets_by_hours_and_excludes_post_deadline(conn, tmp_path):
    _seed_gw(conn, season="2026-27", gw=7)     # kickoff 14:00 -> deadline 12:30
    snapdir = str(tmp_path / "snaps")
    rows = [{"id": 1, "code": 101, "status": "a", "chance_next": "",
             "news_added": "", "now_cost": 75, "selected_by_percent": "10",
             "transfers_in_event": 0, "transfers_out_event": 0,
             "ep_next": "4.0", "form": "3.0"},
            {"id": 2, "code": 102, "status": "d", "chance_next": 0.25,
             "news_added": "", "now_cost": 75, "selected_by_percent": "10",
             "transfers_in_event": 0, "transfers_out_event": 0,
             "ep_next": "4.0", "form": "3.0"}]
    _snap(snapdir, "20251002T1000", 7, rows)   # ~50h before deadline
    _snap(snapdir, "20251004T0900", 7, rows)   # ~3.5h before deadline
    _snap(snapdir, "20251004T1400", 7, rows)   # after deadline: excluded
    # pad so the >=50 rows-per-bucket floor is met
    many = rows * 30
    _snap(snapdir, "20251002T1001", 7, many)
    _snap(snapdir, "20251004T0901", 7, many)
    res = decay.analyse(conn, "2026-27", snap_dir=snapdir)
    buckets = {b["bucket"]: b for b in res["buckets"]}
    assert "T-48h..72h" in buckets and "T-0h..6h" in buckets
    assert "T-12h..24h" not in buckets
    b = buckets["T-0h..6h"]
    # player 1 (available, played): brier 0; player 2 (25%, did not play):
    # 0.0625 -> mean 0.03125
    assert b["brier_play"] == pytest.approx(0.0313, abs=1e-3)
    assert b["flagged_share"] == pytest.approx(0.5, abs=0.01)


def test_decay_reports_gracefully_with_no_data(conn, tmp_path):
    res = decay.analyse(conn, "2026-27", snap_dir=str(tmp_path / "none"))
    assert res["buckets"] == [] and "no snapshots" in res["note"]
