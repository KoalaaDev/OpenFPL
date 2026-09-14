"""Round 18 gates for press-conference / commentary observations.

Gate 2 — information value (2024-25, 2025-26): for every (player, gameweek)
the extractor classified, what did the replayed minutes model (no
availability overlay, exactly as the backtest runs) believe, and what
happened? If "out" players still carry a high model P(start) and then do not
start, the text knows something the replay engine does not.

Gate 1 — timing against FPL's own status log (2026-27 only, the only season
with a stored change log): for each "out"/"doubt" observation, had FPL
already flagged the player when the manager spoke, and had it by the
deadline? If FPL is there first almost every time, the text's value is the
nuance, not the news.

    python research/presser_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402
from fpl_engine.xpts import engine  # noqa: E402


def gate2(conn) -> None:
    frames = []
    for season in ("2024-25", "2025-26"):
        path = os.path.join(config.DATA_DIR, "bt_base", f"audit_{season}.csv")
        if not os.path.exists(path):
            print(f"  (no audit for {season})")
            continue
        a = pd.read_csv(path, usecols=["gw", "player_id", "position", "p_start", "p_60",
                                       "e_min", "starts", "minutes", "a_total", "prediction"])
        a["season"] = season
        frames.append(a)
    audit = pd.concat(frames, ignore_index=True)
    audit["started"] = (audit["starts"] > 0).astype(float)
    audit["got60"] = (audit["minutes"] >= 60).astype(float)

    obs = pd.read_sql_query(
        "SELECT season, gw, player_id, cls, source, published_utc FROM presser_obs "
        "WHERE season IN ('2024-25','2025-26') AND gw IS NOT NULL", conn)
    # one class per player-gw: the most severe
    rank = {"out": 0, "injury": 1, "doubt": 2, "rested": 3, "available": 4}
    obs["rank"] = obs["cls"].map(rank)
    obs = obs.sort_values("rank").drop_duplicates(["season", "gw", "player_id", "source"])
    j = audit.merge(obs, on=["season", "gw", "player_id"], how="left")
    j["cls"] = j["cls"].fillna("(none)")
    print("\n=== Gate 2: what the text says vs what the replay model believed vs what happened")
    print(f"{'class':<12}{'n':>7}{'model P(start)':>16}{'started':>10}{'60+':>8}"
          f"{'model P(60+)':>14}{'pts':>7}{'model xP':>10}")
    for cls, d in j.groupby("cls"):
        print(f"{cls:<12}{len(d):>7}{d.p_start.mean():>16.3f}{d.started.mean():>10.3f}"
              f"{d.got60.mean():>8.3f}{d.p_60.mean():>14.3f}{d.a_total.mean():>7.2f}"
              f"{d.prediction.mean():>10.2f}")
    print("\n--- the same, restricted to players the model rated a likely starter (P(start) >= 0.6)")
    hi = j[j.p_start >= 0.6]
    for cls, d in hi.groupby("cls"):
        print(f"{cls:<12}{len(d):>7}{d.p_start.mean():>16.3f}{d.started.mean():>10.3f}"
              f"{d.got60.mean():>8.3f}{d.p_60.mean():>14.3f}{d.a_total.mean():>7.2f}"
              f"{d.prediction.mean():>10.2f}")
    print("\n--- the ambiguous band (0.3 <= P(start) < 0.7): does the text resolve it?")
    band = j[(j.p_start >= 0.3) & (j.p_start < 0.7)]
    for cls, d in band.groupby("cls"):
        print(f"{cls:<12}{len(d):>7}{d.p_start.mean():>16.3f}{d.started.mean():>10.3f}")
    # value of a perfect use: if we set P(start)=started for text-classified rows only
    print("\n--- per-source split")
    for src, d in j[j.cls != "(none)"].groupby("source"):
        print(f"  {src}: n={len(d)}, classes={d.cls.value_counts().to_dict()}")


def gate1(conn) -> None:
    obs = pd.read_sql_query(
        "SELECT season, gw, player_id, cls, published_utc FROM presser_obs "
        "WHERE season='2026-27' AND source='presser' AND cls IN ('out','doubt')", conn)
    if obs.empty:
        print("\n=== Gate 1: no 2026-27 observations")
        return
    log = pd.read_sql_query(
        "SELECT player_id, observed_utc, status, chance_next FROM acq_player_availability "
        "WHERE season='2026-27' ORDER BY observed_utc", conn)
    if log.empty:
        print("\n=== Gate 1: availability change log is empty")
        return

    def state_at(pid, when):
        d = log[(log.player_id == pid) & (log.observed_utc <= when)]
        if d.empty:
            return None
        r = d.iloc[-1]
        return (r["status"], r["chance_next"])

    rows = []
    for r in obs.itertuples():
        ko = engine.first_kickoff(conn, "2026-27", int(r.gw)) if r.gw == r.gw else None
        deadline = None
        if ko:
            deadline = (pd.Timestamp(ko) - pd.Timedelta(minutes=90)).isoformat()
        before = state_at(r.player_id, r.published_utc)
        at_dl = state_at(r.player_id, deadline) if deadline else None
        flagged = lambda s: bool(s) and (s[0] not in (None, "a") or (s[1] is not None and s[1] < 100))
        rows.append({"cls": r.cls, "fpl_flagged_before_presser": flagged(before),
                     "fpl_flagged_by_deadline": flagged(at_dl), "known": before is not None})
    g = pd.DataFrame(rows)
    print(f"\n=== Gate 1 (2026-27, n={len(g)} out/doubt statements; FPL log covers "
          f"{log.observed_utc.min()[:10]} on)")
    for cls, d in g.groupby("cls"):
        d = d[d.known]
        print(f"  {cls:<6} n={len(d):>3}  FPL already flagged when the manager spoke: "
              f"{d.fpl_flagged_before_presser.mean():.2f}   flagged by the deadline: "
              f"{d.fpl_flagged_by_deadline.mean():.2f}")


if __name__ == "__main__":
    conn = db.connect(config.DB_PATH)
    try:
        gate2(conn)
        gate1(conn)
    finally:
        conn.close()
