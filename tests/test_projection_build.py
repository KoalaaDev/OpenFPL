"""Building projections for a plan that reaches past the built horizon.

The cap on how many gameweeks one request models used to be applied to the
REQUESTED weeks, before the cached ones were skipped — so a draft extended to
GW16 asked for GW5-16, the first eight were exactly the ones already built,
the job modelled nothing, and every added week stayed at 0.00 through any
number of rebuilds.
"""
import pytest

from app import jobs, main, services

CACHE = {"gws": {str(g): {} for g in range(5, 13)}}


def test_the_cap_applies_to_what_is_missing_not_to_what_was_asked_for():
    want = list(range(5, 17))
    assert services.gws_to_build(CACHE, want, max_new=8) == [13, 14, 15, 16]
    assert services.gws_to_build(CACHE, want) == [13, 14, 15, 16]
    assert services.gws_to_build(CACHE, [5, 6, 7]) == []


def test_force_rebuilds_cached_weeks_under_the_same_cap():
    assert services.gws_to_build(CACHE, list(range(5, 17)), force=True, max_new=3) == [5, 6, 7]


def test_an_empty_cache_builds_everything_it_is_allowed_to():
    assert services.gws_to_build({}, [4, 5, 6], max_new=2) == [4, 5]


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(jobs, "running", lambda *a, **k: [])
    monkeypatch.setattr(jobs, "start",
                        lambda kind, fn, *a, **kw: calls.append((a, kw)) or "job-1")
    return calls


def test_the_endpoint_forwards_every_requested_gameweek(started):
    main.projections_build({"gws": [7, 5, 16, 5]}, None, admin=True)
    (gws,), kw = started[0]
    assert gws == [5, 7, 16]                       # sorted, de-duplicated, whole
    assert kw["max_new"] == main.BUILD_MAX_NEW and kw["force"] is False


def test_nonsense_gameweeks_are_refused(started):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        main.projections_build({"gws": [0, 99]}, None, admin=True)
    with pytest.raises(HTTPException):
        main.projections_build({"gws": []}, None, admin=True)
