"""True fixture congestion from the all-competition club calendar (E15/P7).

The shipped ``team_matches_14d`` counts Premier League matches only, because
``team_match`` holds nothing else — a club playing Wednesday in the Champions
League reads as rested. ``tm_club_match`` (every competitive fixture per
club-season, from Transfermarkt's calendar) closes that gap: 13% of eval-season
rows had their 14-day match count understated by 2+.

What the screen found (predicted vs realised start rate, shipped feature set):
every cup-adjacency bucket is calibrated within noise EXCEPT likely starters
(P(start) > 0.7) with cup matches on BOTH sides of the PL fixture, who are
overpredicted by −0.052 / −0.062 (n=327/496) in 2024-25 / 2025-26 — rotation
during double-cup congestion. Adding these four features repairs about half of
that gap (−0.041 → −0.006 and −0.056 → −0.031) and moves overall log-loss
−0.33% / −0.28% — replicated but small, and bounded above by the return-
dynamics block (5× the overall gain, zero decision-metric movement over 74
paired gameweeks), so it cannot move a decision either. Kept as an extras
block for the record; the shipped model is bit-identical without it.

Point-in-time note: fixture calendars are public well before the deadline, so
"the club plays in Europe three days after this kickoff" is deadline-honest.
The one approximation is that the table stores final (rescheduled) dates; cup
tie dates are known 1–3 weeks ahead in practice, which covers the ±4-day
window used here. Rows whose club-season has no mapped calendar stay NaN —
an absent observation is not a negative one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["cal_matches_14d", "cal_days_since_any",
            "cal_cup_prev4", "cal_cup_next4"]

_DAY = np.timedelta64(1, "D")
_ADJ_DAYS = 4


def calendar(conn) -> pd.DataFrame:
    """Per-(season, team_id) all-competition fixture dates.

    Transfermarkt club ids map to FPL teams through the squads (majority
    vote), never through club names — same rule as tactics_features.
    """
    # tm_club_id -> FPL team.code by majority vote over EVERY season's squad,
    # then per-season team_id through the code. A club's identity does not
    # change between seasons, so a single missing squad page (City and United's
    # 2024-25 pages failed silently in the first crawl) must not blank that
    # club-season's calendar — the season-specific version of this mapping did
    # exactly that.
    club = pd.read_sql_query(
        "SELECT s.tm_club_id, t.code, COUNT(*) n "
        "FROM tm_squad s "
        "JOIN tm_player m ON m.tm_player_id = s.tm_player_id "
        "JOIN player p ON p.code = m.player_code AND p.season = s.season "
        "JOIN team t ON t.team_id = p.team_id AND t.season = p.season "
        "WHERE m.player_code IS NOT NULL "
        "GROUP BY s.tm_club_id, t.code", conn)
    teams = pd.read_sql_query("SELECT season, team_id, code FROM team", conn)
    cal = pd.read_sql_query(
        "SELECT tm_club_id, season_id, match_date, competition "
        "FROM tm_club_match", conn)
    if club.empty or cal.empty:
        return pd.DataFrame(columns=["season", "team_id", "match_date", "is_pl"])
    club = (club.sort_values("n", ascending=False)
                .drop_duplicates(["tm_club_id"]))
    cal["season"] = cal["season_id"].map(lambda y: f"{y}-{str(y + 1)[2:]}")
    cal = (cal.merge(club[["tm_club_id", "code"]], on="tm_club_id", how="inner")
              .merge(teams, on=["season", "code"], how="inner"))
    cal["match_date"] = pd.to_datetime(cal["match_date"])
    cal["is_pl"] = cal["competition"].str.contains(
        "Premier League", case=False, na=False)
    return cal[["season", "team_id", "match_date", "is_pl"]]


def add_features(frame: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = {f: np.full(len(out), np.nan) for f in FEATURES}
    if not cal.empty and len(out):
        kick = pd.to_datetime(out["kickoff_utc"], utc=True).dt.tz_localize(None)
        kd = kick.dt.normalize().to_numpy().astype("datetime64[ns]")
        keyarr = pd.Series(list(zip(out["season"], out["team_id"])),
                           index=out.index)
        by_all = {k: np.sort(g["match_date"].values)
                  for k, g in cal.groupby(["season", "team_id"])}
        by_cup = {k: np.sort(g["match_date"].values)
                  for k, g in cal[~cal["is_pl"]].groupby(["season", "team_id"])}
        for k, dates in by_all.items():
            m = (keyarr == k).to_numpy()
            if not m.any():
                continue
            lo = np.searchsorted(dates, kd[m] - 14 * _DAY)
            hi = np.searchsorted(dates, kd[m])
            cols["cal_matches_14d"][m] = hi - lo
            prev = np.where(
                hi > 0, (kd[m] - dates[np.maximum(hi - 1, 0)]) / _DAY, np.nan)
            cols["cal_days_since_any"][m] = np.clip(prev, 0, 30)
            cdates = by_cup.get(k)
            if cdates is not None and len(cdates):
                ci = np.searchsorted(cdates, kd[m])
                dprev = np.where(
                    ci > 0,
                    (kd[m] - cdates[np.maximum(ci - 1, 0)]) / _DAY, 999)
                cj = np.searchsorted(cdates, kd[m], side="right")
                dnext = np.where(
                    cj < len(cdates),
                    (cdates[np.minimum(cj, len(cdates) - 1)] - kd[m]) / _DAY,
                    999)
                cols["cal_cup_prev4"][m] = (
                    (dprev >= 1) & (dprev <= _ADJ_DAYS)).astype(float)
                cols["cal_cup_next4"][m] = (
                    (dnext >= 1) & (dnext <= _ADJ_DAYS)).astype(float)
            else:
                cols["cal_cup_prev4"][m] = 0.0
                cols["cal_cup_next4"][m] = 0.0
    for f in FEATURES:
        out[f] = cols[f]
    return out
