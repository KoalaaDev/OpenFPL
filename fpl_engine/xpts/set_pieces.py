"""Penalty duty as a correction, not a bonus.

A player's historical xG already contains the penalties he took while he had
the duty. Adding a flat boost for today's first-choice taker therefore
double-counts him — and does nothing at all for the player who has just *lost*
the duty, whose trailing xG is still inflated by penalties he will not take
again. The correction here is the difference between the two:

    delta = team_pen_rate x (published_share - historical_share) x PEN_XG

which is **zero whenever today's duty matches the history**, so it cannot
quietly reshape the board; it moves only the players whose duty has actually
changed.

Constants are measured, not chosen (2022-23 to 2025-26, 358 penalties across
64 club-seasons with at least three):

    a club wins             0.126 penalties per match
    its first-choice taker  takes 71% of them, the second 21%, the third 6%
    a penalty is worth      0.79 xG

Duty itself comes from FPL's published ``penalties_order`` where available.
That field is a current snapshot with no history so it cannot be backtested —
but it is the better source: predicting the next taker from the shot log alone
is only 55% accurate (against 53% for "whoever has taken the most" and 49% for
"whoever took the last one"), because duty changes between one penalty and the
next 35% of the time. The shot-derived share is what the model falls back on,
and what the historical half of the correction is built from.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 300.0     # duty is stickier than form
K_PEN_SHARE = 3.0          # pseudo-penalties, shrinking toward "not the taker"
PEN_XG = 0.79              # conversion of a penalty
TEAM_PEN_RATE = 0.1257     # league default, penalties per team-match
# share of a club's penalties taken by its 1st / 2nd / 3rd choice
ORDER_SHARE = {1: 0.710, 2: 0.208, 3: 0.063}


def duty(conn, season: str, as_of: str) -> pd.DataFrame:
    """Per-player penalty-duty correction in xG90 terms.

    Returns player_id, hist_share, order_share, team_pen_rate and
    ``pen_xg90_delta`` — the amount to add to (or subtract from) the player's
    expected non-penalty goal rate. Empty when no shot history is loaded.
    """
    as_of_date = str(as_of)[:10]
    players = pd.read_sql_query(
        "SELECT player_id, understat_id, team_id, penalties_order "
        "FROM player WHERE season=?", conn, params=(season,))
    out = players[["player_id"]].copy()
    out["hist_share"] = np.nan
    out["order_share"] = players["penalties_order"].map(ORDER_SHARE).fillna(0.0)
    out["team_pen_rate"] = TEAM_PEN_RATE
    out["pen_xg90_delta"] = 0.0
    if players.empty:
        return out
    # The published order exists for the CURRENT season only. In a replayed
    # season every value is NULL, which would read as "nobody is on duty" and
    # strip the penalty component from every taker who ever had it. No
    # published order at all means no correction, full stop.
    if players["penalties_order"].notna().sum() == 0:
        return out

    pens = pd.read_sql_query(
        "SELECT understat_id, match_date FROM understat_shot "
        "WHERE situation='Penalty' AND match_date < ?", conn, params=(as_of_date,))
    if pens.empty or players["understat_id"].notna().sum() == 0:
        # no shot history: fall back to the published order alone, with the
        # historical share unknown and therefore assumed to already match
        return out

    ref = pd.Timestamp(as_of_date)
    days = (ref - pd.to_datetime(pens["match_date"])).dt.days.clip(lower=0)
    pens["w"] = 0.5 ** (days / HALF_LIFE_DAYS)
    team_of = dict(zip(players["understat_id"].dropna(),
                       players.loc[players["understat_id"].notna(), "team_id"]))
    pens["team_id"] = pens["understat_id"].map(team_of)
    by_player = pens.groupby("understat_id")["w"].sum()
    by_team = pens.dropna(subset=["team_id"]).groupby("team_id")["w"].sum()

    took = players["understat_id"].map(by_player).fillna(0.0)
    team_w = players["team_id"].map(by_team).fillna(0.0)
    hist = (took / (team_w + K_PEN_SHARE)).where(players["understat_id"].notna())
    out["hist_share"] = hist.to_numpy()
    # only correct where the history is actually known
    known = hist.notna()
    delta = TEAM_PEN_RATE * (out["order_share"] - hist.fillna(0.0)) * PEN_XG
    out["pen_xg90_delta"] = np.where(known, delta, 0.0)
    return out
