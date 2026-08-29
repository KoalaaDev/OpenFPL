"""Deadline-decay analysis: what is an hour of information worth?

The scheduled collector (acquire/actions.py) writes one market snapshot per
run under data/collected/snapshots/, each carrying FPL's availability state
(status, chance_next) at that moment. Once gameweeks finish, every snapshot
becomes a graded prediction: "given what was public T hours before the
deadline, who actually played?" This module grades them, bucketed by hours
to deadline, so the curve fills in by itself as the archive grows.

It answers, with data instead of intuition:

* how much start/minutes accuracy improves as the deadline approaches —
  i.e. when a user should (re)run the model;
* whether the 4x-daily collection cadence is enough, or the last hours
  carry most of the information (then the cron should densify near
  deadlines);
* a floor for the info-value framework: any fancier source must beat the
  free availability field at the same lead time.

The deadline is approximated as 90 minutes before the gameweek's first
kickoff (FPL's standard offset); the approximation is shared by every
bucket so it cannot bias the comparison between them.

Run: ``python -m fpl_engine decay``. Prints "insufficient data" until the
archive contains snapshots for finished gameweeks — by design, the code
ships before the data exists.
"""
from __future__ import annotations

import glob
import gzip
import os
import re

import numpy as np
import pandas as pd

from . import config

SNAP_DIR = os.path.join(config.DATA_DIR, "collected", "snapshots")
BUCKETS = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 24 * 14)]
DEADLINE_OFFSET_H = 1.5
_NAME = re.compile(r"(\d{8}T\d{4})_gw(\d+)\.csv\.gz$")


def load_snapshots(snap_dir: str = SNAP_DIR) -> pd.DataFrame:
    """Every archived snapshot row, tagged with its run time and target gw."""
    frames = []
    for path in sorted(glob.glob(os.path.join(snap_dir, "*.csv.gz"))):
        m = _NAME.search(os.path.basename(path))
        if not m:
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            df = pd.read_csv(fh)
        df["observed_utc"] = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M",
                                            utc=True)
        df["gw"] = int(m.group(2))
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _p_play(df: pd.DataFrame) -> pd.Series:
    """The availability field as a probability: chance_next where FPL gives
    one, else 1 for an 'a' (available) flag and 0 for i/s/u/n/d-without-pct."""
    p = pd.to_numeric(df["chance_next"], errors="coerce")
    return p.fillna((df["status"] == "a").astype(float)).clip(0, 1)


def analyse(conn, season: str | None = None,
            snap_dir: str = SNAP_DIR) -> dict:
    season = season or config.CURRENT_SEASON
    snaps = load_snapshots(snap_dir)
    if snaps.empty:
        return {"buckets": [], "note": "no snapshots archived yet"}

    real = pd.read_sql_query(
        "SELECT gw, player_id, SUM(minutes) minutes, "
        "MAX(COALESCE(starts,0)) started, MIN(kickoff_utc) kick "
        "FROM player_gw WHERE season=? AND kickoff_utc IS NOT NULL "
        "AND minutes IS NOT NULL GROUP BY gw, player_id",
        conn, params=(season,))
    finished = pd.read_sql_query(
        "SELECT gw, MIN(kickoff_utc) first_kick, MAX(kickoff_utc) last_kick "
        "FROM player_gw WHERE season=? AND minutes IS NOT NULL "
        "GROUP BY gw", conn, params=(season,))
    if real.empty or finished.empty:
        return {"buckets": [], "note": "no finished gameweeks in the DB yet"}
    deadline = {int(r.gw): pd.Timestamp(r.first_kick, tz="UTC")
                - pd.Timedelta(hours=DEADLINE_OFFSET_H)
                for r in finished.itertuples()}

    j = snaps.merge(real, left_on=["gw", "id"], right_on=["gw", "player_id"],
                    how="inner")
    if j.empty:
        return {"buckets": [],
                "note": "snapshots exist but none overlap a finished gameweek"}
    j["hours"] = [
        (deadline[g] - t).total_seconds() / 3600.0
        for g, t in zip(j["gw"], j["observed_utc"])]
    j = j[j["hours"] >= 0]                     # post-deadline runs grade gw+1
    j["p_play"] = _p_play(j)
    j["played"] = (j["minutes"] > 0).astype(float)
    j["p60"] = (j["minutes"] >= 60).astype(float)

    out = []
    for lo, hi in BUCKETS:
        b = j[(j["hours"] >= lo) & (j["hours"] < hi)]
        if len(b) < 50:
            continue
        flagged = b[b["p_play"] < 1]           # where team news actually is
        out.append({
            "bucket": f"T-{lo}h..{hi}h", "n": int(len(b)),
            "gws": int(b["gw"].nunique()),
            "brier_play": round(float(((b["p_play"] - b["played"]) ** 2)
                                      .mean()), 4),
            "brier_play_flagged": (round(float(
                ((flagged["p_play"] - flagged["played"]) ** 2).mean()), 4)
                if len(flagged) >= 20 else None),
            "flagged_share": round(float((b["p_play"] < 1).mean()), 3),
        })
    return {"season": season, "buckets": out,
            "note": None if out else
            "insufficient overlap yet — the curve fills in as the scheduled "
            "runs accumulate over finished gameweeks"}
