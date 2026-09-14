"""Round 20 block `absent`: known absences and who they spill onto.

The live availability overlay zeroes a player FPL says is out and stops
there — nothing hands his minutes to the man behind him. The gate
(`research/absence_gate.py`) put a number on that: when a regular starter
misses a match, the highest-P(start) depth player at his position starts 73%
of the time against the model's 61%, plays 11 minutes more than expected and
scores +0.31 points over his projection; for a goalkeeper it is +1.4.

A replay cannot see injuries (no dated availability history in this
database), but it CAN see suspensions, which follow from the card log before
the deadline: a red card is one match (FPL does not separate a second yellow
from a straight red, and measured on 697 rows the longer bans are wrong more
often than right past the first match), and 5/10/15 yellows by matchday
19/32/38 are 1/2/3 (98.5% / 92% precise). So the block is built on suspensions
only, in training and at serve time alike — the same mechanism the live
overlay handles for injuries, learned where it is knowable. Features:

  sus_self           he is banned for this fixture
  pos_regulars_out   regular starters (>=4 of his last 5 starts) at his own
                     team+position banned for this fixture, excluding himself
  team_regulars_out  the same across the whole club

Point-in-time by construction: a ban derives from cards in matches strictly
before the fixture it covers, and the fixture calendar is public.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["sus_self", "pos_regulars_out", "team_regulars_out"]
REGULAR_STARTS_L5 = 4
YELLOW_THRESHOLDS = {5: (1, 19), 10: (2, 32), 15: (3, 38)}   # yellows: (ban, by matchday)


def load(conn) -> pd.DataFrame:
    """One row per (season, team_id, player_code, fixture_id) the player is
    known absent for. Suspensions come from the card log and the fixture
    calendar; with `$FPL_ABSENCE_KINDS` naming more kinds (inj, presser) the
    dated injury spells and Friday "out" statements are added (see
    `xpts/absence.py`)."""
    from . import absence as _ab
    kinds = _ab.kinds_from_env()
    if set(kinds) != {"sus"}:
        return _ab.known_absences(conn, kinds)[["season", "team_id", "player_code", "fixture_id"]]
    return load_suspensions(conn)


def load_suspensions(conn) -> pd.DataFrame:
    try:
        cards = pd.read_sql_query(
            "SELECT season, team_id, player_code, fixture_id, yellow_cards, red_cards "
            "FROM player_gw WHERE (yellow_cards > 0 OR red_cards > 0)", conn)
        # `fixture` holds the live season only; every replayed season's
        # calendar is in `team_match` (one row per club per match)
        fx = pd.read_sql_query(
            "SELECT season, fixture_id, kickoff_utc, team_h, team_a FROM fixture "
            "WHERE kickoff_utc IS NOT NULL", conn)
        tm = pd.read_sql_query(
            "SELECT season, fixture_id, kickoff_utc, team_id AS team_h, opponent_id AS team_a "
            "FROM team_match WHERE kickoff_utc IS NOT NULL AND was_home=1", conn)
    except Exception:      # noqa: BLE001 - tables absent
        return pd.DataFrame(columns=["season", "team_id", "player_code", "fixture_id"])
    fx = pd.concat([fx, tm], ignore_index=True).drop_duplicates(["season", "fixture_id"])
    return suspensions(cards, fx)


def suspensions(cards: pd.DataFrame, fx: pd.DataFrame) -> pd.DataFrame:
    if cards.empty or fx.empty:
        return pd.DataFrame(columns=["season", "team_id", "player_code", "fixture_id"])
    fx = fx.copy()
    fx["kick"] = pd.to_datetime(fx["kickoff_utc"], utc=True, errors="coerce")
    fx = fx.dropna(subset=["kick"])
    long = pd.concat([fx.rename(columns={"team_h": "team_id"})[["season", "team_id", "fixture_id", "kick"]],
                      fx.rename(columns={"team_a": "team_id"})[["season", "team_id", "fixture_id", "kick"]]])
    long = long.sort_values("kick")
    seq = {k: list(d["fixture_id"]) for k, d in long.groupby(["season", "team_id"], sort=False)}
    order = {(s, t, f): i for (s, t), fids in seq.items() for i, f in enumerate(fids)}
    cards = cards.copy()
    cards["seq_i"] = [order.get((s, t, f)) for s, t, f in zip(cards.season, cards.team_id, cards.fixture_id)]
    cards = cards.dropna(subset=["seq_i", "player_code"]).sort_values("seq_i")
    rows = []
    for (season, team, code), d in cards.groupby(["season", "team_id", "player_code"], sort=False):
        fids = seq[(season, team)]
        yellows = 0
        for r in d.itertuples():
            i = int(r.seq_i)
            y = int(r.yellow_cards or 0)
            ban, rule = 0, None
            if r.red_cards and r.red_cards > 0:
                # FPL's card log does not separate a second yellow from a
                # straight red, and most reds are one-match bans: measured on
                # 697 derived rows, the player was out for 90% of first
                # matches but PLAYED 54% of second and 62% of third ones. One
                # match is the honest derivation; a violent-conduct ban's
                # later matches are left to the live availability overlay
                ban, rule = 1, "red"
            if y:
                before, yellows = yellows, yellows + y
                for thr, (n, by) in YELLOW_THRESHOLDS.items():
                    if before < thr <= yellows and i + 1 <= by and n > ban:
                        ban, rule = n, f"yellows_{thr}"
            for k in range(1, ban + 1):
                if i + k < len(fids):
                    rows.append((season, team, int(code), fids[i + k], rule, k))
    out = pd.DataFrame(rows, columns=["season", "team_id", "player_code", "fixture_id", "rule", "match_no"])
    return out.drop_duplicates(["season", "team_id", "player_code", "fixture_id"])


def add_features(frame: pd.DataFrame, sus: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for f in FEATURES:
        frame[f] = 0.0
    if frame.empty or sus is None or sus.empty:
        return frame
    key = ["season", "team_id", "player_code", "fixture_id"]
    s = sus[key].drop_duplicates().assign(_sus=1.0)
    tmp = frame[key].copy()
    for c in ("team_id", "player_code", "fixture_id"):
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        s[c] = pd.to_numeric(s[c], errors="coerce")
    m = tmp.merge(s, on=key, how="left")["_sus"].fillna(0.0).to_numpy()
    frame["sus_self"] = m
    regular = (pd.to_numeric(frame.get("starts_l5"), errors="coerce").fillna(0)
               >= REGULAR_STARTS_L5).astype(float).to_numpy()
    out = frame["sus_self"].to_numpy() * regular
    frame["_ro"] = out
    pos = frame.groupby(["season", "team_id", "fixture_id", "position"])["_ro"].transform("sum")
    team = frame.groupby(["season", "team_id", "fixture_id"])["_ro"].transform("sum")
    frame["pos_regulars_out"] = (pos - out).clip(lower=0)
    frame["team_regulars_out"] = (team - out).clip(lower=0)
    return frame.drop(columns=["_ro"])
