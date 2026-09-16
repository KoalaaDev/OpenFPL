"""The admin's "is the model getting better?" page.

It reads files other code wrote, so the risk is entirely in the key names —
exactly the defect it replaced, where the deadline desk read
`pooled.band.feed_acc` against a file that says `band_metrics.feed_accuracy`
and therefore showed nothing at all while four gameweeks sat on disk.
"""
import json

import pytest

from app import modelhistory as mh
from fpl_engine import config


def test_postmortems_are_parsed_and_ordered(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    for gw, sp in ((3, 0.71), (1, 0.42)):
        (tmp_path / f"postmortem_2026-27_gw{gw}.json").write_text(json.dumps({
            "season": "2026-27", "gw": gw, "spearman": sp, "top20_hits": 4,
            "captain_pick": "X", "captain_actual": 8, "captain_best": 17,
            "predicted_total": 900, "actual_total": 950,
            "components_60plus": {"n": 200,
                                  "goals": {"predicted": 12.0, "actual": 10},
                                  "assists": {"predicted": 8.0, "actual": 10},
                                  "clean_sheets_gk_def": {"predicted": 5.0, "actual": 5}},
        }), encoding="utf-8")
    rows = mh.postmortems("2026-27")
    assert [r["gw"] for r in rows] == [1, 3]
    # expected / actual, so >1 means the model over-called it
    assert rows[0]["goals_ratio"] == 1.2
    assert rows[0]["assists_ratio"] == 0.8
    assert rows[0]["cs_ratio"] == 1.0


def test_a_zero_actual_does_not_divide_by_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    (tmp_path / "postmortem_2026-27_gw1.json").write_text(json.dumps({
        "gw": 1, "components_60plus": {"goals": {"predicted": 3.0, "actual": 0}},
    }), encoding="utf-8")
    assert mh.postmortems("2026-27")[0]["goals_ratio"] is None


def test_feed_scorecard_uses_the_keys_the_file_actually_has(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    (tmp_path / "lineup_feed_2026-27.json").write_text(json.dumps({
        "gws": {"3": {"band_metrics": {"n": 60, "feed_accuracy": 0.75,
                                       "model_accuracy": 0.63,
                                       "implied_points_per_season": 39.6}}},
        "pooled": {"band_metrics": {"n": 60, "feed_accuracy": 0.75},
                   "all_metrics": {"n": 600}, "priceable": False,
                   "rows_needed_for_estimate": 450},
        "updated_utc": "2026-09-16T12:00:00Z",
    }), encoding="utf-8")
    d = mh.feed_scorecard("2026-27")
    assert d["per_gw"] == [{"gw": 3, "n": 60, "feed": 0.75, "model": 0.63,
                            "implied_points": 39.6}]
    assert d["band"]["feed_accuracy"] == 0.75
    assert d["priceable"] is False


def test_missing_files_are_none_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    assert mh.feed_scorecard("2026-27") is None
    assert mh.postmortems("2026-27") == []
    assert mh.backtests() == []


def test_backtests_keep_one_row_per_arm(tmp_path, monkeypatch):
    """A season mean means nothing without the baselines beside it, which is
    why the page lists every arm of the replay rather than just the engine."""
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    (tmp_path / "backtest_2025-26.json").write_text(json.dumps({
        "season": "2025-26", "gws": [1, 2, 3],
        "summary": {"xpts": {"spearman_played": 0.375, "top30": 4.55, "rmse": 1.9},
                    "ppg": {"spearman_played": 0.276, "top30": 3.45, "rmse": 2.2}},
        "minutes_holdout_accuracy": 0.80,
    }), encoding="utf-8")
    rows = mh.backtests()
    assert rows[0]["season"] == "2025-26" and rows[0]["gws"] == 3
    assert {a["arm"] for a in rows[0]["arms"]} == {"xpts", "ppg"}
    assert rows[0]["arms"][0]["spearman_played"] == 0.375


def test_importances_resolve_f_indices_to_feature_names():
    """xgboost names features f0..fn when fitted on a bare matrix; a chart of
    'f13' tells an admin nothing."""
    d = mh.minutes_model()
    assert isinstance(d["importances"], list)
    if d["importances"]:
        assert not d["importances"][0]["feature"].startswith("f")
        assert sum(r["gain"] for r in d["importances"]) <= 1.0001


def test_payload_has_every_section_the_page_renders():
    d = mh.payload(force=True)
    assert set(d) >= {"season", "postmortems", "feed", "minutes", "blend",
                      "backtests", "built_at"}
