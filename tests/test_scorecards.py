"""The scorecards score themselves.

The post-mortem and the lineup-feed report used to be things the admin desk
told you to run in a terminal, which meant they were usually empty and the
product read as though nothing were automatic. They are library calls, and
the scheduled refresh makes them now — so what has to hold is that the runner
is idempotent (a refresh must not re-score a gameweek, or rewrite a file, on
every tick) and that it cannot take a data pull down with it.
"""
import json
import os

import pytest

from app import scheduler
from fpl_engine import config


@pytest.fixture(autouse=True)
def _no_engine(monkeypatch):
    """These tests are about the runner's bookkeeping — what it skips, what it
    reports, that it cannot raise — not about the engine. Left real, every
    call re-ran a post-mortem, the lineup-feed scoring and the model record's
    replays for each finished gameweek against the live database: minutes per
    test, and a full suite that went from five minutes to half an hour. Each
    of those has its own tests; a test here overrides a stub when it needs to."""
    from app import modelrecord
    from fpl_engine import lineup_feed, postmortem
    calls = {"postmortem": []}
    monkeypatch.setattr(modelrecord, "refresh", lambda *a, **k: {"built": [], "total": 0})
    from app import modelteam
    monkeypatch.setattr(modelteam, "build", lambda *a, **k: {"built": [], "weeks": 0, "next": None})
    monkeypatch.setattr(postmortem, "run",
                        lambda conn, season=None, gw=None, **k: calls["postmortem"].append(gw) or {})
    monkeypatch.setattr(lineup_feed, "score_gw", lambda *a, **k: {"gw": 0})
    monkeypatch.setattr(lineup_feed, "save", lambda *a, **k: "")
    return calls


def test_already_scored_gameweeks_are_skipped(tmp_path, monkeypatch, _no_engine):
    """The file on disk is the "done" marker. Without this a daily refresh
    would re-run a season's post-mortems every single day."""
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    season = config.CURRENT_SEASON
    marker = tmp_path / f"postmortem_{season}_gw1.json"
    marker.write_text("{}", encoding="utf-8")

    out = scheduler.score_finished_gameweeks()
    assert 1 not in out["postmortem"]
    assert 1 not in _no_engine["postmortem"]      # never even called for it


def test_a_failing_scorecard_is_reported_not_raised(tmp_path, monkeypatch):
    """A scorecard is a nice-to-have; a data pull is not. If the post-mortem
    throws, the refresh must still finish and the reason must survive."""
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    import fpl_engine.postmortem as pmod
    monkeypatch.setattr(pmod, "run", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("no results yet")))
    out = scheduler.score_finished_gameweeks()
    assert isinstance(out, dict)
    assert all(isinstance(e, str) for e in out["errors"])


def test_scored_feed_gameweeks_reads_the_saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    season = config.CURRENT_SEASON
    (tmp_path / f"lineup_feed_{season}.json").write_text(
        json.dumps({"gws": {"2": {}, "3": {}}}), encoding="utf-8")
    assert scheduler._scored_feed_gws(season) == {2, 3}


def test_a_missing_or_broken_feed_file_is_an_empty_set(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    season = config.CURRENT_SEASON
    assert scheduler._scored_feed_gws(season) == set()
    (tmp_path / f"lineup_feed_{season}.json").write_text("not json", encoding="utf-8")
    assert scheduler._scored_feed_gws(season) == set()


def test_the_refresh_actually_calls_the_runner():
    """Pins the wiring: it is the whole reason the desk stops saying "run
    python ...". A rename that lost this call would be silent otherwise."""
    import inspect
    assert "score_finished_gameweeks(job_id)" in inspect.getsource(scheduler._refresh)
