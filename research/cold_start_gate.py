"""Round 23 gate — can a new signing's rates be seeded from his other-league
Understat history instead of the position prior?

`understat_player_match` keeps every match Understat holds for a resolved
player, including seasons BEFORE he joined FPL (other top leagues). For a
player whose first FPL season is S, the "foreign prior" is his decayed
npxG/90 and xA/90 over those earlier rows. The engine's audit rows give what
the shipped engine expected of him in his first six EPL appearances; the
question is whether the foreign prior predicts his realised attacking
returns better than the engine's cold-start rate did.

    python research/cold_start_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402

FIRST_N = 6


def main():
    conn = db.connect(config.DB_PATH)
    pl = pd.read_sql_query("SELECT season, player_id, code, understat_id, position FROM player "
                           "WHERE understat_id IS NOT NULL", conn)
    first = pd.read_sql_query("SELECT player_code, MIN(season) AS first_fpl FROM player_gw GROUP BY player_code", conn)
    pl = pl.merge(first, left_on="code", right_on="player_code")
    new = pl[(pl["season"] == pl["first_fpl"]) & pl["season"].isin(["2024-25", "2025-26"])].copy()
    # the EPL ingest drops other-league matches; the cold-start collector keeps
    # them in `understat_foreign_match` (seasons before the player's first FPL one)
    um = pd.read_sql_query("SELECT season, understat_id, match_date, minutes, npxg, xa, goals, assists "
                           "FROM understat_foreign_match", conn)
    um["understat_id"] = um["understat_id"].astype(str)
    new["understat_id"] = new["understat_id"].astype(str)
    rows = []
    for r in new.itertuples():
        h = um[(um["understat_id"] == r.understat_id) & (um["season"] < r.season)]
        h = h[h["minutes"].fillna(0) > 0]
        if h["minutes"].sum() < 900:            # at least ten full matches abroad
            continue
        # decay by season distance (SEASON_BREAK_DECAY-like) so 2019 matters less than 2023
        yrs = int(r.season[:4]) - h["season"].str[:4].astype(int)
        w = 0.7 ** yrs.clip(lower=1)
        mins = (w * h["minutes"]).sum()
        rows.append({"season": r.season, "player_id": r.player_id, "position": r.position,
                     "foreign_mins": float(h["minutes"].sum()),
                     "f_npxg90": float((w * h["npxg"].fillna(0)).sum() / mins * 90),
                     "f_xa90": float((w * h["xa"].fillna(0)).sum() / mins * 90)})
    seeds = pd.DataFrame(rows)
    print(f"new signings with >= 900 foreign Understat minutes: {len(seeds)} "
          f"({seeds.groupby('season').size().to_dict()})")
    auds = []
    for season in ("2024-25", "2025-26"):
        a = pd.read_csv(os.path.join(config.DATA_DIR, "bt_base_pre_r21", f"audit_{season}.csv"),
                        usecols=["gw", "player_id", "position", "minutes", "e_goals", "e_assists", "a_goals",
                                 "a_assists", "e_min", "n_fixtures"])
        a["season"] = season
        auds.append(a)
    aud = pd.concat(auds, ignore_index=True)
    d = aud.merge(seeds, on=["season", "player_id", "position"])
    d = d[d["minutes"] > 0].sort_values(["season", "player_id", "gw"])
    d["app_no"] = d.groupby(["season", "player_id"]).cumcount() + 1
    d = d[d["app_no"] <= FIRST_N]
    # the engine's expected attacking events per 90 of REALISED minutes, vs the foreign seed
    d["e_g90"] = d["e_goals"] / d["e_min"].clip(lower=1) * 90
    d["e_a90"] = d["e_assists"] / d["e_min"].clip(lower=1) * 90
    d["a_g90"] = d["a_goals"] / d["minutes"] * 90
    d["a_a90"] = d["a_assists"] / d["minutes"] * 90
    print(f"\nfirst {FIRST_N} appearances of each new signing: {len(d)} player-matches, "
          f"{d.groupby(['season','player_id']).ngroups} players")
    for pos in ("DEF", "MID", "FWD"):
        m = d[d["position"] == pos]
        if len(m) < 30:
            continue
        print(f"\n{pos} (n={len(m)}): realised G/90 {m.a_g90.mean():.3f}  engine {m.e_g90.mean():.3f}  "
              f"foreign npxG/90 {m.f_npxg90.mean():.3f} | realised A/90 {m.a_a90.mean():.3f}  "
              f"engine {m.e_a90.mean():.3f}  foreign xA/90 {m.f_xa90.mean():.3f}")
        # does the foreign seed explain realised output beyond the engine's rate? per player
        p = m.groupby(["season", "player_id"]).agg(a_g=("a_goals", "sum"), a_a=("a_assists", "sum"),
                                                   mins=("minutes", "sum"), e_g=("e_goals", "sum"),
                                                   e_a=("e_assists", "sum"), f_g=("f_npxg90", "first"),
                                                   f_a=("f_xa90", "first"), e_min=("e_min", "sum")).reset_index()
        p["a_g90"] = p.a_g / p.mins * 90; p["e_g90"] = p.e_g / p.e_min.clip(lower=1) * 90
        p["a_a90"] = p.a_a / p.mins * 90; p["e_a90"] = p.e_a / p.e_min.clip(lower=1) * 90
        from scipy import stats
        for what, a, e, f in (("goals", "a_g90", "e_g90", "f_g"), ("assists", "a_a90", "e_a90", "f_a")):
            re = stats.pearsonr(p[e], p[a]); rf = stats.pearsonr(p[f], p[a])
            # partial: foreign on top of engine
            X = np.column_stack([np.ones(len(p)), p[e], p[f]]); b, *_ = np.linalg.lstsq(X, p[a], rcond=None)
            resid = p[a] - X @ b; s2 = (resid @ resid) / (len(p) - 3); cov = s2 * np.linalg.inv(X.T @ X)
            t_f = b[2] / np.sqrt(cov[2, 2]); t_e = b[1] / np.sqrt(cov[1, 1])
            mae_e = np.abs(p[a] - p[e]).mean(); mae_f = np.abs(p[a] - p[f]).mean()
            mae_h = np.abs(p[a] - 0.5 * (p[e] + p[f])).mean()
            print(f"  {what:<8} players={len(p)}  corr engine {re[0]:+.2f}  corr foreign {rf[0]:+.2f}  | "
                  f"joint: t_engine {t_e:+.2f} t_foreign {t_f:+.2f} | MAE engine {mae_e:.3f} foreign {mae_f:.3f} half/half {mae_h:.3f}")
    conn.close()


if __name__ == "__main__":
    main()
