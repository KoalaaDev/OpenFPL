"""The Live desk's schedule.

The tab exists only inside a window, so the window IS the feature: get it
wrong and either nobody sees the desk before a deadline or it sits there all
week as furniture. Four boundaries, pinned.
"""
import pandas as pd
import pytest

from app import live

DL = {"5": "2026-09-19T10:00:00Z", "6": "2026-09-26T10:00:00Z"}
T5 = live._ts(DL["5"])
T6 = live._ts(DL["6"])


@pytest.mark.parametrize("offset,phase,gw", [
    (-48 * 3600, "idle", 5),        # a week out: no tab
    (-24 * 3600 - 60, "idle", 5),   # a minute before the window opens
    (-24 * 3600 + 60, "open", 5),   # a minute after it opens
    (-60, "open", 5),               # a minute before the deadline
    (60, "closed", 5),              # a minute after: over, still on screen
    (6 * 3600 - 60, "closed", 5),   # last minute of the six-hour tail
    (6 * 3600 + 60, "idle", 6),     # gone, and now pointed at the next gw
])
def test_window_phases(offset, phase, gw):
    w = live.window(DL, T5 + offset)
    assert w["phase"] == phase
    assert w["gw"] == gw


def test_window_reports_the_deadline_it_is_counting_to():
    w = live.window(DL, T5 - 3600)
    assert w["deadline"] == T5
    assert w["seconds"] == pytest.approx(3600)


def test_no_deadlines_is_idle_not_a_crash():
    w = live.window({}, T5)
    assert w["phase"] == "idle" and w["gw"] is None


def test_unparseable_deadline_is_skipped():
    w = live.window({"5": "not a date", "6": DL["6"]}, T6 - 3600)
    assert w["phase"] == "open" and w["gw"] == 6


def test_status_payload_carries_the_window():
    """The shell decides whether to render the tab from `status.live`, so the
    field has to survive the payload, not just the helper."""
    from app import services
    st = services.status_payload()
    assert set(st["live"]) >= {"phase", "gw"}
    assert st["live"]["phase"] in ("idle", "open", "closed")
