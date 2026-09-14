"""Round 20 fixture context from the BBC archive: the referee's card rate and
each club's set-play / open-play split, both point-in-time.

* `referee_factors` — the appointed referee's decayed yellow cards per match
  over his PRIOR matches, shrunk to the league rate and expressed relative to
  it. Officials come from the lineups payload (`acq_bbc_official`), the cards
  from `player_gw`. The appointment is public days before the deadline, so
  for a replay the match's own referee is a legal input.
* `setplay_factors` — for every club-fixture, how the OPPONENT's expected
  goals conceded split between set play and open play over its prior
  matches (`acq_bbc_match_stats`, Opta xG split, 2024-25 on), relative to
  the league split and shrunk toward it. A club that leaks from corners is a
  better fixture for a defender than its total xGA says. The player's own
  set-play share is his position's league share (`POS_SET_SHARE`) until a
  per-player shot log is loaded, so the factor is mean-preserving for a
  league-average shot mix and only reshapes it.

Every factor is 1.0 wherever the archive is silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pressers import BBC_TO_FPL

REF_HALF_LIFE_DAYS = 365.0
REF_K0 = 15.0            # matches of shrinkage toward the league rate
SET_HALF_LIFE_DAYS = 180.0
SET_K0 = 8.0
# share of a position's goals that come from set plays (corners, free kicks,
# throw-ins; penalties excluded), league-wide — the shape prior until a
# per-player shot log is available
POS_SET_SHARE = {"GK": 0.0, "DEF": 0.55, "MID": 0.20, "FWD": 0.28}


def fixture_map(conn) -> pd.DataFrame:
    """event_urn -> (season, fixture_id, home_id, away_id, kick)."""
    bm = pd.read_sql_query(
        "SELECT event_urn, season, home, away, kickoff_utc FROM acq_bbc_match "
        "WHERE kickoff_utc IS NOT NULL", conn)
    if bm.empty:
        return pd.DataFrame(columns=["event_urn", "season", "fixture_id", "home_id", "away_id", "kick"])
    teams = pd.read_sql_query("SELECT season, team_id, name FROM team", conn)
    key = {(r.season, r.name): int(r.team_id) for r in teams.itertuples()}
    bm["home_id"] = [key.get((s, BBC_TO_FPL.get(t, t))) for s, t in zip(bm["season"], bm["home"])]
    bm["away_id"] = [key.get((s, BBC_TO_FPL.get(t, t))) for s, t in zip(bm["season"], bm["away"])]
    bm["kick"] = pd.to_datetime(bm["kickoff_utc"], utc=True, errors="coerce")
    bm = bm.dropna(subset=["home_id", "away_id", "kick"])
    bm["day"] = bm["kick"].dt.floor("D")
    tm = pd.read_sql_query(
        "SELECT season, team_id AS home_id, fixture_id, kickoff_utc FROM team_match "
        "WHERE was_home=1 AND kickoff_utc IS NOT NULL", conn)
    fx = pd.read_sql_query(
        "SELECT season, team_h AS home_id, fixture_id, kickoff_utc FROM fixture "
        "WHERE kickoff_utc IS NOT NULL", conn)
    f = pd.concat([tm, fx], ignore_index=True).drop_duplicates(["season", "fixture_id"])
    f["day"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce").dt.floor("D")
    f["home_id"] = pd.to_numeric(f["home_id"], errors="coerce")
    bm["home_id"] = bm["home_id"].astype(float)
    out = bm.merge(f[["season", "home_id", "day", "fixture_id"]], on=["season", "home_id", "day"], how="inner")
    return out[["event_urn", "season", "fixture_id", "home_id", "away_id", "kick"]].astype(
        {"fixture_id": int, "home_id": int, "away_id": int})


def _decayed_prior(df: pd.DataFrame, group: str, value: str, half_life: float,
                   k0: float, league: pd.Series | None = None) -> pd.Series:
    """Per row: decayed mean of ``value`` over the group's STRICTLY prior rows,
    shrunk with k0 pseudo-matches toward the league prior; NaN -> 1.0 ratio
    is left to the caller. Rows must be sorted by kick."""
    out = np.full(len(df), np.nan)
    w_sum = np.zeros(len(df)); v_sum = np.zeros(len(df))
    lam = np.log(2) / half_life
    # days as floats: divide by a Timedelta, never an integer view (E11c)
    days_all = ((df["kick"] - pd.Timestamp("2000-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy(float)
    for _, idx in df.groupby(group, sort=False).indices.items():
        idx = np.asarray(idx)
        ks = days_all[idx]; vs = df[value].to_numpy(float)[idx]
        for j in range(1, len(idx)):
            dt = ks[j] - ks[:j]
            w = np.exp(-lam * dt)
            ok = ~np.isnan(vs[:j])
            w_sum[idx[j]] = w[ok].sum(); v_sum[idx[j]] = (w[ok] * vs[:j][ok]).sum()
    return pd.Series(w_sum, index=df.index), pd.Series(v_sum, index=df.index)


def referee_factors(conn) -> pd.DataFrame:
    """(season, fixture_id, ref_factor): the referee's prior yellow rate
    relative to the league's, shrunk; only fixtures with a known referee."""
    off = pd.read_sql_query(
        "SELECT event_urn, official_urn FROM acq_bbc_official WHERE role='Referee'", conn)
    fm = fixture_map(conn)
    if off.empty or fm.empty:
        return pd.DataFrame(columns=["season", "fixture_id", "ref_factor"])
    yc = pd.read_sql_query(
        "SELECT season, fixture_id, SUM(COALESCE(yellow_cards,0)) AS yellows FROM player_gw "
        "GROUP BY season, fixture_id", conn)
    d = fm.merge(off, on="event_urn").merge(yc, on=["season", "fixture_id"], how="left")
    d = d.sort_values("kick").reset_index(drop=True)
    # league prior: expanding mean of yellows per match over all prior matches
    lg = d["yellows"].shift(1).expanding().mean().fillna(d["yellows"].mean())
    w, v = _decayed_prior(d, "official_urn", "yellows", REF_HALF_LIFE_DAYS, REF_K0)
    rate = (v + REF_K0 * lg) / (w + REF_K0)
    d["ref_factor"] = (rate / lg.replace(0, np.nan)).fillna(1.0).clip(0.6, 1.6)
    return d[["season", "fixture_id", "ref_factor"]]


def setplay_factors(conn) -> pd.DataFrame:
    """(season, fixture_id, team_id, set_def_rel, open_def_rel) — for the club
    ATTACKING in that fixture, the opponent's prior set-play / open-play
    share of xG conceded relative to the league's, shrunk."""
    st = pd.read_sql_query(
        "SELECT event_urn, team, side, xg_open, xg_set FROM acq_bbc_match_stats "
        "WHERE xg_open IS NOT NULL AND xg_set IS NOT NULL", conn)
    fm = fixture_map(conn)
    if st.empty or fm.empty:
        return pd.DataFrame(columns=["season", "fixture_id", "team_id", "set_def_rel", "open_def_rel"])
    d = st.merge(fm, on="event_urn")
    # the xG a club CONCEDES in the match is the other side's xG
    d["conceder"] = np.where(d["side"] == "home", d["away_id"], d["home_id"])
    d = d.sort_values("kick").reset_index(drop=True)
    d["set_share"] = d["xg_set"] / (d["xg_set"] + d["xg_open"]).replace(0, np.nan)
    lg = d["set_share"].shift(1).expanding().mean().fillna(d["set_share"].mean())
    w, v = _decayed_prior(d, "conceder", "set_share", SET_HALF_LIFE_DAYS, SET_K0)
    share = (v + SET_K0 * lg) / (w + SET_K0)
    d["set_def_rel"] = (share / lg).fillna(1.0)
    d["open_def_rel"] = ((1 - share) / (1 - lg)).fillna(1.0)
    # the row belongs to the ATTACKING club (the one whose xG this was)
    d["team_id"] = np.where(d["side"] == "home", d["home_id"], d["away_id"])
    return d[["season", "fixture_id", "team_id", "set_def_rel", "open_def_rel"]].astype(
        {"fixture_id": int, "team_id": int})


def factor_maps(conn, season: str, fixture_ids: list[int]) -> tuple[dict, dict]:
    """(fixture_id -> ref_factor, (fixture_id, team_id) -> (set_rel, open_rel))
    for one gameweek; empty dicts when the archive has nothing."""
    ref, sp = {}, {}
    try:
        r = referee_factors(conn)
        r = r[(r["season"] == season) & r["fixture_id"].isin(fixture_ids)]
        ref = dict(zip(r["fixture_id"].astype(int), r["ref_factor"].astype(float)))
    except Exception:      # noqa: BLE001
        ref = {}
    try:
        s = setplay_factors(conn)
        s = s[(s["season"] == season) & s["fixture_id"].isin(fixture_ids)]
        sp = {(int(a), int(b)): (float(c), float(d))
              for a, b, c, d in zip(s["fixture_id"], s["team_id"], s["set_def_rel"], s["open_def_rel"])}
    except Exception:      # noqa: BLE001
        sp = {}
    return ref, sp
