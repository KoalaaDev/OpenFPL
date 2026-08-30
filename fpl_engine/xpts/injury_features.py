"""Point-in-time injury features from the Transfermarkt spell history.

**Not wired into the model.** The measurement below is promising but not
finished, and this repo does not ship an unproven feature. It lives here so the
experiment is reproducible rather than trapped in a scratchpad.

WHY THIS ONE IS DIFFERENT. Six attempts to sharpen the minutes model all
returned <=0.25% of log-loss and none shipped — but every one was a
re-arrangement of data the model already had (rotation tendency, opponent
strength, Understat role, isotonic calibration, a start/sub split, squad
state). Injury history is exogenous: trailing minutes record THAT a player was
absent, never that it was a hamstring, nor that it was his third in two years.
It is also the first minutes signal that is backtestable, because Transfermarkt
dates every spell — availability, expected lineups and manager picks all died
on the "no archived history" wall.

WHAT WAS MEASURED (2022-25 train / held-out season, ~57k player-matches):

    arm                          2024-25    2025-26
    baseline (no injury data)     0.4962     0.4425
    + currently-out only          -5.11%     -7.33%
    + injury history only         -1.39%     -1.96%
    + both                        -7.08%     -9.25%
    history ON TOP of currently-out  -2.08%    -2.08%

Read that carefully, because the headline is misleading. Backtests disable the
availability overlay (stored FPL status is today's, not that gameweek's), so
the baseline has NO availability information at all. Most of the -7%/-9% is
therefore re-deriving what the LIVE model already gets free from FPL's
`status`/`chance_next`. Quoting it as a model improvement would repeat the
site_ep mistake: a benchmark that looks extraordinary because the baseline was
handicapped.

The honest number is **-2.08%**, the part that survives once the model already
knows he is out — and it replicated to two decimals across two independent
seasons, which is not the shape of noise. It is still 8x the best of the six
previous attempts.

STILL OPEN before this can ship:
  1. Score it on decision metrics, not log-loss. Understat's rates were 3.9%
     better and moved nothing (all p > 0.19); log-loss is not the bar.
  2. Check TM's record against FPL's `status` where both exist. If TM is only a
     noisier copy of a flag we already read live, the live gain is nil even
     though the backtest gain is real.

LEAKAGE BOUNDARY, which is the whole risk. Each spell has `from_date` and
`until_date`, but `until_date` is knowable in advance only for a spell that has
already ended:

  * ended before the deadline          -> everything usable
  * started before, not yet ended      -> only the FACT that he is out; its
                                          duration and end date are the future

Audited for the obvious failure: an injury sustained DURING a match being
credited to that match. Only 133 spells begin on a fixture day and 85% of those
players did play, so same-day contamination is not driving the result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Soft-tissue injuries are the recurrence-prone ones; a hamstring predicts
# another hamstring in a way an ankle knock does not.
SOFT_TISSUE = ("hamstring", "muscle", "muscular", "calf", "groin", "thigh",
               "adductor", "strain")

FEATURES = ["inj_days_365", "inj_spells_365", "inj_spells_730",
            "inj_soft_730", "inj_days_since_return", "inj_currently_out"]
# the subset that is NOT an availability proxy — the genuinely novel part
HISTORY_ONLY = [f for f in FEATURES if f != "inj_currently_out"]


def spells(conn, season: str) -> pd.DataFrame:
    """Injury spells joined onto FPL `player_code` (stable across seasons)."""
    df = pd.read_sql_query(
        "SELECT i.tm_player_id, i.from_date, i.until_date, i.days, "
        "       i.games_missed, i.injury, m.player_id "
        "FROM tm_injury i JOIN tm_player m ON m.tm_player_id = i.tm_player_id "
        "WHERE m.player_id IS NOT NULL", conn)
    if df.empty:
        return df.assign(player_code=[], from_dt=[], until_dt=[], soft=[])
    codes = pd.read_sql_query(
        "SELECT player_id, code player_code FROM player WHERE season=?",
        conn, params=(season,))
    df = df.merge(codes, on="player_id", how="inner")
    df["from_dt"] = pd.to_datetime(df["from_date"], errors="coerce", utc=True)
    df["until_dt"] = pd.to_datetime(df["until_date"], errors="coerce", utc=True)
    df["soft"] = df["injury"].fillna("").str.lower().apply(
        lambda s: int(any(k in s for k in SOFT_TISSUE)))
    return df.dropna(subset=["from_dt", "player_code"])


def add_features(frame: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time injury features to a minutes-model frame.

    `frame` needs `player_code` and `kick` (the fixture's kickoff). Every value
    is computed only from spells strictly before that kickoff.
    """
    out = frame.copy()
    for f in FEATURES:
        out[f] = 0.0
    if sp is None or sp.empty:
        out["inj_days_since_return"] = 999.0
        return out

    kick = pd.to_datetime(out["kick"], errors="coerce", utc=True)
    by_code = {c: g for c, g in sp.groupby("player_code")}
    n = len(out)
    days365 = np.zeros(n); spells365 = np.zeros(n); spells730 = np.zeros(n)
    soft730 = np.zeros(n); since = np.full(n, 999.0); out_now = np.zeros(n)
    codes = out["player_code"].to_numpy()
    ks = kick.to_numpy()

    for i in range(n):
        g = by_code.get(codes[i])
        if g is None or pd.isna(ks[i]):
            continue
        t = pd.Timestamp(ks[i])
        started = g[g["from_dt"] < t]
        if started.empty:
            continue
        ended = started[started["until_dt"].notna() & (started["until_dt"] < t)]
        if not ended.empty:
            d365 = ended[ended["until_dt"] >= t - pd.Timedelta(days=365)]
            days365[i] = float(d365["days"].fillna(0).sum())
            spells365[i] = float(len(d365))
            d730 = ended[ended["until_dt"] >= t - pd.Timedelta(days=730)]
            spells730[i] = float(len(d730))
            soft730[i] = float(d730["soft"].sum())
            since[i] = float((t - ended["until_dt"].max()).days)
        ongoing = started[started["until_dt"].isna() | (started["until_dt"] >= t)]
        out_now[i] = float(len(ongoing) > 0)

    out["inj_days_365"] = days365
    out["inj_spells_365"] = spells365
    out["inj_spells_730"] = spells730
    out["inj_soft_730"] = soft730
    out["inj_days_since_return"] = np.clip(since, 0, 999)
    out["inj_currently_out"] = out_now
    return out
