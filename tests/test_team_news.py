"""Team news keeps itself current.

The Live desk's news feed reads the availability change log, and for a week
nothing wrote it: the full refresh overwrote each player's CURRENT status but
never appended to the log, so the desk said "6 d ago" while FPL had news from
that morning. These pin the three things that fixed it.
"""
import inspect

import pytest

from app import deadline, scheduler


def test_the_app_polls_team_news_on_its_own_thread():
    src = inspect.getsource(scheduler.start)
    assert "_news_loop" in src and "fplabs-news" in src


def test_the_full_refresh_records_team_news_too():
    assert "pull_team_news()" in inspect.getsource(scheduler._refresh)


def test_the_poll_interval_is_frequent_and_bounded(monkeypatch):
    monkeypatch.delenv("FPLABS_NEWS_MINUTES", raising=False)
    assert scheduler.news_minutes() == 15.0
    monkeypatch.setenv("FPLABS_NEWS_MINUTES", "1")
    assert scheduler.news_minutes() == 5.0          # never hammer FPL
    monkeypatch.setenv("FPLABS_NEWS_MINUTES", "nonsense")
    assert scheduler.news_minutes() == 15.0


def test_a_failed_poll_is_recorded_not_raised(monkeypatch):
    from acquire.sources import fpl_availability
    monkeypatch.setattr(fpl_availability, "pull",
                        lambda conn, season: (_ for _ in ()).throw(RuntimeError("FPL is down")))
    out = scheduler.pull_team_news()
    assert "FPL is down" in out["error"]
    assert "FPL is down" in scheduler.state().get("news_last_error", "")


def test_news_is_ordered_by_when_fpl_published_it(tmp_path):
    """A row recorded late (a backfill, a missed poll) must not jump ahead of
    news that broke after it."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE acq_player_availability (season, player_id, observed_utc, "
                 "source_id, source_published_utc, status, chance_next, news, raw_id)")
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    iso = lambda h: (now - timedelta(hours=h)).isoformat()
    conn.executemany("INSERT INTO acq_player_availability VALUES (?,?,?,?,?,?,?,?,?)", [
        # observed most recently, but the news itself is two days old
        ("2026-27", 1, iso(0), "fpl_api", iso(48), "d", 50, "old knock", None),
        # observed earlier, but broke this morning
        ("2026-27", 2, iso(1), "fpl_api", iso(2), "i", 0, "hamstring", None),
    ])
    pl = {1: {"name": "Old", "team_id": 1, "pos": "MID"}, 2: {"name": "New", "team_id": 1, "pos": "DEF"}}
    rows = deadline._news(conn, "2026-27", pl, {1: "ARS"})
    assert [r["name"] for r in rows] == ["New", "Old"]
