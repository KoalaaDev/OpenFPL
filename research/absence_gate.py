"""Round 20 gate — does a starter's absence spill over onto his team-mates?

The live availability overlay zeroes a player FPL says is out; nothing
redistributes his minutes, shots or set pieces to whoever replaces him. In a
replay the overlay is off, so the question can be asked cleanly against the
audit rows (what the engine expected per player-gameweek, point-in-time):

  when a REGULAR STARTER (started >= 4 of his club's last 5 league matches)
  plays no minutes this gameweek, what happens to the residual
  (actual - expected) of his team-mates at the same position?

Two absence definitions:
  * `absent`    — he played 0 minutes (hindsight; sizes the whole channel)
  * `suspended` — a ban derivable from the card log before the deadline
                  (second yellow 1 match, straight red 3, 5/10/15 yellows
                  by matchday 19/32/38 -> 1/2/3). Fully knowable, so this is
                  the reachable floor; injuries known pre-deadline would add
                  to it but this DB carries no dated injury history.

    python research/absence_gate.py [--audit data/bt_base_pre_r17]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402

SEASONS = ("2023-24", "2024-25", "2025-26")


def load_pgw(conn, season):
    g = pd.read_sql_query(
        "SELECT gw, player_id, team_id, fixture_id, kickoff_utc, starts, minutes, "
        "yellow_cards, red_cards FROM player_gw WHERE season=?", conn, params=(season,))
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    return g.dropna(subset=["kick"])


def regular_starters(g: pd.DataFrame, gw_first_kick: pd.Series) -> pd.DataFrame:
    """(gw, team_id, player_id) rows for players who started >=4 of the club's
    last 5 league matches strictly before the gameweek's first kickoff."""
    rows = []
    team_fx = {t: d.drop_duplicates("fixture_id").sort_values("kick")[["fixture_id", "kick"]]
               for t, d in g.groupby("team_id")}
    starts = g.groupby(["team_id", "player_id", "fixture_id"])["starts"].max()
    for gw, k0 in gw_first_kick.items():
        for t, fx in team_fx.items():
            prev = fx[fx["kick"] < k0].tail(5)
            if len(prev) < 5:
                continue
            fids = set(prev["fixture_id"])
            s = starts.loc[t]
            s = s[s.index.get_level_values("fixture_id").isin(fids)]
            n = s.groupby(level="player_id").sum()
            for pid, c in n[n >= 4].items():
                rows.append((gw, t, pid))
    return pd.DataFrame(rows, columns=["gw", "team_id", "player_id"])


def suspensions(g: pd.DataFrame) -> set:
    """(team_id, player_id, fixture_id) the player is banned for, from cards."""
    out = set()
    for (t, pid), d in g.groupby(["team_id", "player_id"]):
        team_fx = g[g.team_id == t].drop_duplicates("fixture_id").sort_values("kick")
        order = {f: i for i, f in enumerate(team_fx["fixture_id"])}
        d = d.sort_values("kick")
        yellows = 0
        thresholds = {5: (1, 19), 10: (2, 32), 15: (3, 38)}
        for r in d.itertuples():
            idx = order.get(r.fixture_id)
            if idx is None:
                continue
            ban = 0
            if r.red_cards and r.red_cards > 0:
                ban = 1 if (r.yellow_cards or 0) > 0 else 3
            if r.yellow_cards:
                yellows += int(r.yellow_cards)
                for thr, (n, by) in thresholds.items():
                    if yellows >= thr and yellows - int(r.yellow_cards) < thr and idx + 1 <= by:
                        ban = max(ban, n)
            for k in range(1, ban + 1):
                if idx + k < len(team_fx):
                    out.add((t, pid, team_fx["fixture_id"].iloc[idx + k]))
    return out


def run(audit_dir: str):
    conn = db.connect(config.DB_PATH)
    frames = []
    for season in SEASONS:
        path = os.path.join(audit_dir, f"audit_{season}.csv")
        if not os.path.exists(path):
            continue
        a = pd.read_csv(path, usecols=["gw", "player_id", "team_id", "position", "p_start", "e_min",
                                       "prediction", "starts", "minutes", "a_total", "n_fixtures"])
        a["season"] = season
        g = load_pgw(conn, season)
        gw_k0 = g.groupby("gw")["kick"].min()
        reg = regular_starters(g, gw_k0)
        reg["regular"] = 1
        a = a.merge(reg, on=["gw", "team_id", "player_id"], how="left")
        a["regular"] = a["regular"].fillna(0).astype(int)
        # suspended for at least one of the gameweek's fixtures
        sus = suspensions(g)
        fx = g[["gw", "team_id", "player_id", "fixture_id"]].drop_duplicates()
        fx["sus"] = [(t, p, f) in sus for t, p, f in zip(fx.team_id, fx.player_id, fx.fixture_id)]
        sus_gw = fx.groupby(["gw", "team_id", "player_id"])["sus"].max().reset_index()
        a = a.merge(sus_gw, on=["gw", "team_id", "player_id"], how="left")
        a["sus"] = a["sus"].fillna(False).astype(bool)
        frames.append(a)
    a = pd.concat(frames, ignore_index=True)
    a = a[a["n_fixtures"] == 1]                       # single-fixture gameweeks only
    a["absent"] = (a["regular"] == 1) & (a["minutes"] == 0)
    a["suspended"] = (a["regular"] == 1) & a["sus"]
    print(f"rows {len(a)} | regular-starter rows {int(a.regular.sum())} | "
          f"absent regulars {int(a.absent.sum())} | suspended regulars {int(a.suspended.sum())} "
          f"(of whom played 0 min: {int((a.suspended & (a.minutes == 0)).sum())})")
    for c in ("starts", "minutes", "a_total"):
        a[f"r_{c}"] = a[c] - a[{"starts": "p_start", "minutes": "e_min", "a_total": "prediction"}[c]]

    key = ["season", "gw", "team_id", "position"]
    for label in ("absent", "suspended"):
        n_abs = a.groupby(key)[label].sum().rename("n_abs").reset_index()
        b = a.merge(n_abs, on=key)
        b["treated"] = b["n_abs"] > 0
        print(f"\n=== spillover when a regular starter is {label.upper()} at the same position "
              f"(team-position-gameweeks treated: {int(n_abs.n_abs.gt(0).sum())})")
        # team-mates only: exclude the absent men themselves
        mates = b[~b[label]]
        for who, sel in (("depth players (not regular)", mates.regular == 0),
                         ("other regular starters", mates.regular == 1)):
            m = mates[sel]
            t = m.groupby("treated")[["r_starts", "r_minutes", "r_a_total", "p_start", "starts",
                                      "prediction", "a_total"]].mean()
            n = m.groupby("treated").size()
            print(f"  {who}:  n untreated {n.get(False, 0)}  n treated {n.get(True, 0)}")
            print(t.round(3).to_string())
            if n.get(True, 0) > 30:
                from scipy import stats
                for c in ("r_starts", "r_minutes", "r_a_total"):
                    x, y = m.loc[m.treated, c], m.loc[~m.treated, c]
                    tt = stats.ttest_ind(x, y, equal_var=False)
                    print(f"    {c:<10} treated-untreated {x.mean()-y.mean():+.3f}  p={tt.pvalue:.4f}")
        # the next man in line: the highest-p_start depth player at that position
        d = mates[mates.regular == 0].sort_values("p_start", ascending=False)
        nxt = d.groupby(key).head(1)
        t = nxt.groupby("treated")[["p_start", "starts", "r_starts", "e_min", "minutes", "r_minutes",
                                    "prediction", "a_total", "r_a_total"]].mean()
        print("  next in line (highest P(start) depth player):")
        print(t.round(3).to_string())
        if label == "absent":
            # by position
            print("  next in line, by position (treated only): residuals")
            print(nxt[nxt.treated].groupby("position")[["r_starts", "r_minutes", "r_a_total"]]
                  .agg(["mean", "size"]).round(3).to_string())
    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=os.path.join(config.DATA_DIR, "bt_base_pre_r17"))
    args = ap.parse_args()
    run(args.audit)
