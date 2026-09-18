"""What managers said has to keep up with a deadline morning.

Press conferences reached the site only through the full refresh — daily, and
once ~2 h before the deadline — while the BBC page is published through the
Friday morning, so the panel could be hours behind the news it exists to
carry. It now polls beside team news while a deadline is close.
"""
import inspect
import time

from app import scheduler


def test_the_news_thread_pulls_press_conferences_near_a_deadline():
    src = inspect.getsource(scheduler._news_loop)
    assert "pull_pressers" in src and "PRESSER_WINDOW_H" in src


def test_the_window_is_wide_enough_for_fridays_page():
    # a Friday 18:30 deadline: the page starts at ~08:00 on Friday, and the
    # Thursday quotes that precede it are worth having too
    assert scheduler.PRESSER_WINDOW_H >= 24


def test_the_poll_interval_is_bounded(monkeypatch):
    monkeypatch.delenv("FPLABS_PRESSER_MINUTES", raising=False)
    assert scheduler.presser_minutes() == 20.0
    monkeypatch.setenv("FPLABS_PRESSER_MINUTES", "1")
    assert scheduler.presser_minutes() == 10.0          # never hammer the BBC
    monkeypatch.setenv("FPLABS_PRESSER_MINUTES", "nonsense")
    assert scheduler.presser_minutes() == 20.0


def test_hours_to_deadline_ignores_deadlines_that_have_passed(monkeypatch):
    now = time.time()
    monkeypatch.setattr(scheduler, "_deadlines",
                        lambda: [now - 86400, now + 3600 * 5, now + 86400 * 8])
    assert 4.9 < scheduler.hours_to_deadline() < 5.1
    monkeypatch.setattr(scheduler, "_deadlines", lambda: [now - 10])
    assert scheduler.hours_to_deadline() is None
    monkeypatch.setattr(scheduler, "_deadlines", lambda: [])
    assert scheduler.hours_to_deadline() is None


def test_a_failed_pull_is_recorded_not_raised(monkeypatch):
    from acquire.sources import bbc_pressers
    monkeypatch.setattr(bbc_pressers, "pull",
                        lambda conn, **k: (_ for _ in ()).throw(RuntimeError("BBC is down")))
    out = scheduler.pull_pressers()
    assert "BBC is down" in out["error"]
    assert "BBC is down" in scheduler.state().get("pressers_last_error", "")


def test_a_successful_pull_records_what_it_found(monkeypatch):
    from acquire.sources import bbc_pressers
    from fpl_engine import pressers
    monkeypatch.setattr(bbc_pressers, "pull", lambda conn, **k: {"pages": 2, "posts": 7})
    monkeypatch.setattr(pressers, "extract_pressers", lambda conn, seasons=None: {"obs": 3})
    out = scheduler.pull_pressers()
    assert out["posts"] == 7 and out["obs"] == 3
    st = scheduler.state()
    assert st["pressers_last_posts"] == 7 and st["pressers_last_error"] is None
    assert time.time() - st["pressers_last_run"] < 30
