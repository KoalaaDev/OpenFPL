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
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DAY = pd.Timedelta(days=1)

SOFT_TISSUE = ("hamstring", "muscle", "muscular", "calf", "groin", "thigh",
               "adductor", "strain")

FEATURES = ["inj_days_365", "inj_spells_365", "inj_spells_730",
            "inj_soft_730", "inj_days_since_return", "inj_currently_out"]
# the subset that is NOT an availability proxy — the genuinely novel part
HISTORY_ONLY = [f for f in FEATURES if f != "inj_currently_out"]


def spells(conn, season: str | None = None) -> pd.DataFrame:
    """Injury spells joined onto FPL `player_code` (stable across seasons).

    `player_code` is read straight off `tm_player`, never reconstructed from
    `tm_player.player_id` against a season's `player` table. FPL reassigns its
    element ids every summer — measured on this database, **99.7% of ids point
    to a different footballer one season later** — so the id route resolves a
    player against the current squad and then hands his injury history to
    whoever inherited his number two seasons ago. `season` is accepted and
    ignored, because there is nothing season-specific left to do.
    """
    df = pd.read_sql_query(
        "SELECT i.tm_player_id, i.from_date, i.until_date, i.days, "
        "       i.games_missed, i.injury, m.player_code "
        "FROM tm_injury i JOIN tm_player m ON m.tm_player_id = i.tm_player_id "
        "WHERE m.player_code IS NOT NULL", conn)
    if df.empty:
        return df.assign(player_code=[], from_dt=[], until_dt=[], soft=[])
    df["from_dt"] = pd.to_datetime(df["from_date"], errors="coerce", utc=True)
    df["until_dt"] = pd.to_datetime(df["until_date"], errors="coerce", utc=True)
    df["soft"] = df["injury"].fillna("").str.lower().apply(
        lambda s: int(any(k in s for k in SOFT_TISSUE)))
    return df.dropna(subset=["from_dt", "player_code"])


def _positions(code_rank: np.ndarray, when: np.ndarray,
               sorted_keys: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """How many of each player's spells lie strictly before `when`.

    A per-row scan is the obvious way to write this and costs 68 seconds on a
    115k-row frame — which a 38-gameweek backtest would pay once per gameweek.
    Encoding (player, day) as one sortable key turns the whole thing into a
    single vectorised searchsorted: days are ~2e4 apart, so a 1e9 stride keeps
    the players from ever colliding.
    """
    key = code_rank.astype(np.float64) * 1e9 + when
    return np.searchsorted(sorted_keys, key, side="left") - starts[code_rank]


def add_features(frame: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time injury features to a minutes-model frame.

    `frame` needs `player_code` and `kick` (the fixture's kickoff). Every value
    is computed only from spells strictly before that kickoff: a spell that had
    ENDED contributes its duration, and one still running contributes only the
    fact that he is out — its end date is the future.
    """
    out = frame.copy()
    for f in FEATURES:
        out[f] = 0.0
    out["inj_days_since_return"] = 999.0
    if sp is None or sp.empty:
        return out

    # Days since the epoch, as floats. Dividing by a Timedelta rather than
    # rescaling `astype("int64")` is deliberate: pandas 2.x keeps whatever
    # resolution a timestamp was built at, so the integer view is nanoseconds
    # for one column and microseconds for the next — a silent 1000x.
    kick = pd.to_datetime(out["kick"], errors="coerce", utc=True)
    ok = kick.notna().to_numpy()
    t = np.where(ok, (kick - EPOCH) / DAY, np.nan)

    g = sp.dropna(subset=["from_dt"]).copy()
    g["_from"] = ((g["from_dt"] - EPOCH) / DAY).to_numpy()
    until = np.where(g["until_dt"].notna().to_numpy(),
                     ((g["until_dt"] - EPOCH) / DAY).to_numpy(), np.nan)
    # an unfinished spell has no end: it must sort after every real date so it
    # never counts as "ended", and never becomes the last return date
    g["_until"] = np.where(np.isnan(until), 1e8, until)
    g["_days"] = pd.to_numeric(g["days"], errors="coerce").fillna(0.0)

    codes = pd.Index(sorted(set(g["player_code"].astype("int64"))))
    rank_of = pd.Series(np.arange(len(codes)), index=codes)
    row_rank = out["player_code"].map(rank_of)
    known = ok & row_rank.notna().to_numpy()
    if not known.any():
        return out
    rr = row_rank.fillna(0).to_numpy().astype("int64")
    g["_rank"] = g["player_code"].astype("int64").map(rank_of).to_numpy()

    def _sorted(by):
        d = g.sort_values(["_rank", by])
        keys = d["_rank"].to_numpy(np.float64) * 1e9 + d[by].to_numpy()
        starts = np.searchsorted(keys, np.arange(len(codes),
                                                 dtype=np.float64) * 1e9)
        return d, keys, starts

    d_end, k_end, s_end = _sorted("_until")
    d_beg, k_beg, s_beg = _sorted("_from")

    n_end = _positions(rr, t, k_end, s_end)
    n_beg = _positions(rr, t, k_beg, s_beg)
    n_365 = _positions(rr, t - 365.0, k_end, s_end)
    n_730 = _positions(rr, t - 730.0, k_end, s_end)

    # prefix sums inside each player's block, so a window is one subtraction
    def _prefix(col):
        v = np.concatenate([[0.0], np.cumsum(d_end[col].to_numpy(np.float64))])
        return v
    pre_days, pre_soft = _prefix("_days"), _prefix("soft")
    pre_n = np.arange(len(d_end) + 1, dtype=np.float64)
    base = s_end[rr]
    hi, l365, l730 = base + n_end, base + n_365, base + n_730

    days365 = pre_days[hi] - pre_days[l365]
    spells365 = pre_n[hi] - pre_n[l365]
    spells730 = pre_n[hi] - pre_n[l730]
    soft730 = pre_soft[hi] - pre_soft[l730]

    last_end = d_end["_until"].to_numpy(np.float64)
    since = np.full(len(out), 999.0)
    has_end = known & (n_end > 0)
    since[has_end] = t[has_end] - last_end[(hi - 1)[has_end]]

    zero = ~known
    for name, arr in (("inj_days_365", days365), ("inj_spells_365", spells365),
                      ("inj_spells_730", spells730), ("inj_soft_730", soft730)):
        arr = np.where(zero, 0.0, arr)
        out[name] = arr
    out["inj_days_since_return"] = np.clip(np.where(zero, 999.0, since), 0, 999)
    out["inj_currently_out"] = np.where(zero, 0.0,
                                        (n_beg - n_end > 0).astype(float))
    return out
