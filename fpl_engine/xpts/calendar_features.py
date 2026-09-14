"""Round 19 block `cal`: a club's REAL calendar for the minutes model.

The shipped congestion features (`days_rest`, `team_matches_14d`) are built
from Premier League matches only, because that is all the database held. A
club that played in Europe on Thursday therefore looks rested for a full
week, and a club with a cup replay next Tuesday looks like it has nothing on.
BBC's collated fixtures feed (acquire/sources/bbc.py, `acq_bbc_calendar`)
carries every competition, so the same two ideas can be computed honestly,
plus the one everybody actually reasons about: is there a European match
right after this one?

All features are point-in-time by construction: past matches are strictly
before the row's kickoff, and future ones are scheduled fixtures, which are
public weeks ahead (the deadline knows them). Missing calendar → NaN, which
the classifier tolerates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pressers import BBC_TO_FPL

FEATURES = ["cal_days_rest", "cal_prev7", "cal_next7", "cal_next4_europe",
            "cal_prev4_europe"]
EUROPE = ("Champions League", "Europa League", "Conference League", "UEFA")


def _is_europe(name: str | None) -> bool:
    n = name or ""
    return any(k.lower() in n.lower() for k in EUROPE)


def load(conn) -> pd.DataFrame:
    """One row per (season, team_id, kick) for every competition."""
    try:
        cal = pd.read_sql_query(
            "SELECT season, kickoff_utc, competition, home, away, status "
            "FROM acq_bbc_calendar WHERE kickoff_utc IS NOT NULL", conn)
    except Exception:      # noqa: BLE001 - table absent
        return pd.DataFrame(columns=["season", "team_id", "kick", "europe"])
    if cal.empty:
        return pd.DataFrame(columns=["season", "team_id", "kick", "europe"])
    # BBC names the women's and youth sides exactly like the men's club, so
    # the competition is the only thing that separates a WSL Sunday from a
    # Premier League one
    junk = cal["competition"].fillna("").str.contains(
        r"women|wsl|u2[13]|u18|youth|premier league 2|reserve", case=False, regex=True)
    cal = cal[~junk]
    teams = pd.read_sql_query("SELECT season, team_id, name FROM team", conn)
    key = {(r.season, r.name): int(r.team_id) for r in teams.itertuples()}
    rows = []
    for r in cal.itertuples():
        for club in (r.home, r.away):
            tid = key.get((r.season, BBC_TO_FPL.get(club, club)))
            if tid is None:
                continue
            rows.append((r.season, tid, r.kickoff_utc, _is_europe(r.competition)))
    out = pd.DataFrame(rows, columns=["season", "team_id", "kick", "europe"])
    out["kick"] = pd.to_datetime(out["kick"], utc=True, errors="coerce")
    out = out.dropna(subset=["kick"]).drop_duplicates(["season", "team_id", "kick"])
    return out.sort_values(["season", "team_id", "kick"]).reset_index(drop=True)


def add_features(frame: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """Attach the five calendar features to a minutes-model frame that has
    `season`, `team_id` and `kick` (UTC timestamp) columns. Vectorised per
    club with searchsorted over its sorted match times."""
    frame = frame.copy()
    for f in FEATURES:
        frame[f] = np.nan
    if cal is None or cal.empty or frame.empty:
        return frame
    kick = pd.to_datetime(frame["kick"], utc=True, errors="coerce")
    # SECONDS since the epoch, by dividing by a Timedelta — never an integer
    # view: pandas keeps the parsed resolution (an ISO string parses to
    # microseconds), which is the bug that once turned days into days/1000
    EPOCH = pd.Timestamp(0, tz="UTC")
    t_ns = ((kick - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    DAY_NS = 86_400.0
    six_h = 6 * 3600.0
    seasons = frame["season"].to_numpy()
    teams = pd.to_numeric(frame["team_id"], errors="coerce").to_numpy()
    rest = np.full(len(frame), np.nan); prev7 = np.full(len(frame), np.nan)
    next7 = np.full(len(frame), np.nan); next4e = np.full(len(frame), np.nan)
    prev4e = np.full(len(frame), np.nan)
    for (season, tid), g in cal.groupby(["season", "team_id"]):
        idx = np.where((seasons == season) & (teams == tid))[0]
        if not len(idx):
            continue
        ks = ((g["kick"] - EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)   # sorted
        eu = g["europe"].to_numpy(bool)
        eu_cum = np.concatenate([[0], np.cumsum(eu)])
        t = t_ns[idx]
        ok = ~np.isnan(t)
        tt = t[ok]
        # matches strictly before (t - 6h) and strictly after (t + 6h)
        n_before = np.searchsorted(ks, tt - six_h, side="left")
        n_upto_after = np.searchsorted(ks, tt + six_h, side="right")
        has_prev = n_before > 0
        last_prev = np.where(has_prev, ks[np.maximum(n_before - 1, 0)], np.nan)
        r = np.where(has_prev, np.minimum(30.0, (tt - last_prev) / DAY_NS), 30.0)
        lo7 = np.searchsorted(ks, tt - 7 * DAY_NS, side="left")
        p7 = n_before - lo7
        lo4 = np.searchsorted(ks, tt - 4 * DAY_NS, side="left")
        p4e = eu_cum[n_before] - eu_cum[lo4]
        hi7 = np.searchsorted(ks, tt + 7 * DAY_NS, side="right")
        n7 = hi7 - n_upto_after
        hi4 = np.searchsorted(ks, tt + 4 * DAY_NS, side="right")
        n4e = eu_cum[hi4] - eu_cum[n_upto_after]
        sel = idx[ok]
        rest[sel] = r; prev7[sel] = p7; prev4e[sel] = p4e; next7[sel] = n7; next4e[sel] = n4e
    frame["cal_days_rest"] = rest
    frame["cal_prev7"] = prev7
    frame["cal_next7"] = next7
    frame["cal_next4_europe"] = next4e
    frame["cal_prev4_europe"] = prev4e
    return frame
