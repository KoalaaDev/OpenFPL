"""The team model must stay identified when a club has almost no match log.

A promoted club whose only match finished goalless (and carries no xG) used to
send its multiplicative Poisson update to log(0) = -inf. The divergence was not
visible in the fixture rates — attack/defence absorbed it — but it destroyed
the intercept, and with it ``league_rate``. Since the engine's fixture attack
scaler is ``lam_for / league_rate``, a collapsed ``league_rate`` pins every
player in the league at the scaler cap and silently deletes the fixture signal.
"""
import math

import pytest

from fpl_engine.xpts import team_model


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, q, args=()):
        class _C:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows
        return _C(self.rows)


def _match(code, opp, home, gf, kick, xg=None):
    return {"season": "s", "team_id": 0, "opponent_id": 0, "was_home": home,
            "kickoff_utc": kick, "goals_for": gf, "xg": xg,
            "code": code, "opp_code": opp}


def _established(n_days=20):
    """Eighteen ordinary clubs with a full, goal-scoring match log."""
    rows = []
    for d in range(1, n_days + 1):
        kick = f"2025-01-{d:02d}T15:00:00Z"
        for c in range(1, 19, 2):
            rows.append(_match(c, c + 1, 1, 2, kick))
            rows.append(_match(c + 1, c, 0, 1, kick))
    return rows


def test_goalless_newcomer_does_not_break_the_league_rate():
    rows = _established()
    # a promoted club (code 99) with a single goalless match, no xG
    rows.append(_match(99, 1, 0, 0, "2025-01-21T15:00:00Z"))
    rows.append(_match(1, 99, 1, 3, "2025-01-21T15:00:00Z"))
    tm = team_model.fit(_FakeConn(rows), "2025-01-22T00:00:00Z")

    assert 0.8 < tm.league_rate < 3.0, "league scoring rate must stay sane"
    # every rating stays in a plausible band -> no runaway update
    assert all(abs(v) < 2.0 for v in tm.attack.values())
    assert all(abs(v) < 2.0 for v in tm.defence.values())
    # and the fixture scaler the engine derives is not pinned at its cap
    lam_h, lam_a = tm.fixture(1, 3)
    for lam in (lam_h, lam_a):
        assert 0.6 < lam / tm.league_rate < 1.75


def test_league_rate_matches_the_observed_scoring_rate():
    """It is the average goals per team-match, not a function of the intercept."""
    rows = _established()
    tm = team_model.fit(_FakeConn(rows), "2025-01-22T00:00:00Z")
    observed = sum(r["goals_for"] for r in rows) / len(rows)   # 1.5
    assert tm.league_rate == pytest.approx(observed, abs=0.05)


def test_goalless_newcomer_is_still_rated_weakest():
    rows = _established()
    rows.append(_match(99, 1, 0, 0, "2025-01-21T15:00:00Z"))
    rows.append(_match(1, 99, 1, 3, "2025-01-21T15:00:00Z"))
    tm = team_model.fit(_FakeConn(rows), "2025-01-22T00:00:00Z")
    # regularisation must not flatten the signal away entirely
    assert tm.attack[99] < max(tm.attack.values())
    assert tm.attack[99] == pytest.approx(team_model.PROMOTED_PRIOR, abs=0.35)


def test_stronger_attack_still_outranks_weaker_after_regularisation():
    rows = []
    for d in range(1, 13):
        kick = f"2025-01-{d:02d}T15:00:00Z"
        rows.append(_match(1, 3, 1, 4, kick))     # prolific
        rows.append(_match(3, 1, 0, 0, kick))
        rows.append(_match(2, 3, 1, 1, kick))     # modest
        rows.append(_match(3, 2, 0, 0, kick))
    tm = team_model.fit(_FakeConn(rows), "2025-02-01T00:00:00Z")
    assert tm.attack[1] > tm.attack[2]
    lam1, _ = tm.fixture(1, 3)
    lam2, _ = tm.fixture(2, 3)
    assert lam1 > lam2 > 0
