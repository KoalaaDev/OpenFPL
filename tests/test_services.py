"""Web-app service helpers that are pure enough to unit test."""
from app.services import unlimited_applies


def test_unlimited_transfers_only_for_the_opening_gameweek():
    assert unlimited_applies(True, 0, 1, 1)           # pre-deadline, plan starts GW1
    assert not unlimited_applies(True, 0, 3, 1)       # planning ahead from GW3
    assert not unlimited_applies(True, 2, 3, 3)       # stale flag after deadlines
    assert not unlimited_applies(False, 0, 1, 1)      # no flag



def test_first_open_gw_skips_a_gameweek_whose_deadline_has_passed():
    from app.services import first_open_gw
    events = [{"id": 3, "deadline_time": "2026-09-05T10:00:00Z"},
              {"id": 4, "deadline_time": "2026-09-12T13:00:00Z"},
              {"id": 5, "deadline_time": "2026-09-19T13:00:00Z"}]
    import calendar, time
    at = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))
    assert first_open_gw(events, at("2026-09-04T00:00:00Z")) == 3
    assert first_open_gw(events, at("2026-09-13T15:00:00Z")) == 5     # GW4 in progress -> GW5
    assert first_open_gw(events, at("2026-09-12T12:59:59Z")) == 4
    assert first_open_gw(events, at("2026-12-01T00:00:00Z")) is None
    assert first_open_gw([{"id": 1}], 0) is None
