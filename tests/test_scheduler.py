"""The auto-refresh cadence: daily, and shortly before every deadline."""
from datetime import datetime, timezone

from app import scheduler


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


def test_daily_slot_rolls_to_tomorrow_once_passed():
    now = _ts("2026-09-11T10:00:00")
    when, why = scheduler.next_run(now, [], (4, 30), 2.0)
    assert why == "daily"
    assert when == _ts("2026-09-12T04:30:00")
    when2, _ = scheduler.next_run(_ts("2026-09-11T04:00:00"), [], (4, 30), 2.0)
    assert when2 == _ts("2026-09-11T04:30:00")


def test_pre_deadline_run_wins_when_it_comes_first():
    now = _ts("2026-09-11T10:00:00")
    deadline = _ts("2026-09-11T17:30:00")          # tonight's deadline
    when, why = scheduler.next_run(now, [deadline], (4, 30), 2.0)
    assert why == "pre-deadline"
    assert when == _ts("2026-09-11T15:30:00")


def test_past_deadlines_are_ignored():
    now = _ts("2026-09-11T10:00:00")
    past = _ts("2026-09-05T17:30:00")
    when, why = scheduler.next_run(now, [past, _ts("2026-09-19T10:00:00")], (4, 30), 2.0)
    assert why == "daily" and when == _ts("2026-09-12T04:30:00")


def test_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("FPLABS_AUTO_REFRESH", "0")
    assert not scheduler.enabled()
    assert scheduler.start() is False
    monkeypatch.setenv("FPLABS_HORIZON", "99")
    assert scheduler.horizon() == 8
