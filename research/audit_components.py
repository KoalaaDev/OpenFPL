"""Round 17a — component calibration audit of the xPts engine on replayed seasons.

The backtest scores RANK. A level error in one component (say clean sheets
15% too low for everyone) is invisible to `spearman_played` within a position
and only faintly visible to top-30, yet it decides DEF-vs-MID in a starting XI
and it decides the captain. This script replays a season point-in-time and
compares, per component, the points the engine expected with the points that
were actually scored — overall and for the players who really played 60+.

It also checks two conservation laws the minutes model is not told about:
a club fields exactly 11 starters and exactly 990 minutes per fixture.

    python research/audit_components.py 2024-25 2025-26 [--gws 2 38]

Writes data/bt_base/audit_<season>.csv (one row per player-gameweek) and
prints the calibration tables.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpl_engine import config, db, progress, scoring  # noqa: E402
from fpl_engine.xpts import engine, minutes_model  # noqa: E402

COMPS = ("goals", "assists", "cs", "conceded", "saves", "bonus", "cards",
         "defcon", "appearance", "residual")


def replay(conn, season: str, gws: list[int], tag: str) -> pd.DataFrame:
    clf, meta = minutes_model.load(tag)
    if clf is None:
        train = [s for s in config.BACKFILL_SEASONS if s < season]
        progress.step(f"training minutes model for {season} on {train}")
        minutes_model.train(conn, seasons=train, tag=tag)
        clf, meta = minutes_model.load(tag)
    rules = scoring.load_rules()
    actual = pd.read_sql_query(
        "SELECT pg.gw, pg.player_id, pg.team_id, pg.fixture_id, pg.minutes, "
        "pg.total_points, pg.goals_scored, pg.assists, pg.clean_sheets, "
        "pg.goals_conceded, pg.saves, pg.bonus, pg.yellow_cards, pg.red_cards, "
        "pg.defcon, pg.starts, p.position FROM player_gw pg JOIN player p "
        "ON p.season=pg.season AND p.player_id=pg.player_id WHERE pg.season=? "
        "AND p.position IN ('GK','DEF','MID','FWD')", conn, params=(season,))
    frames = []
    for g in gws:
        as_of = engine.first_kickoff(conn, season, g)
        if not as_of:
            continue
        x = engine.xpts_predict_gw(conn, season, g, as_of=as_of,
                                   use_availability=False,
                                   minutes_bundle=(clf, meta), rules=rules)
        if x.empty:
            continue
        a = actual[actual["gw"] == g].copy()
        # realised points per component, via the engine's own definition
        for c in COMPS:
            a[f"a_{c}"] = [engine._realised(c, r, r["position"], rules) or 0.0
                           for r in a.to_dict("records")]
        # a double gameweek has two rows: sum the actuals, keep max minutes
        agg = {f"a_{c}": "sum" for c in COMPS}
        agg.update(minutes="sum", total_points="sum", starts="sum",
                   fixture_id="count")
        a = a.groupby("player_id").agg(agg).rename(
            columns={"fixture_id": "n_played_fx"}).reset_index()
        j = x.merge(a, on="player_id", how="left")
        j["gw"] = g
        j["a_total"] = j["total_points"].fillna(0.0)
        j["minutes"] = j["minutes"].fillna(0.0)
        j["starts"] = j["starts"].fillna(0.0)
        for c in COMPS:
            j[f"a_{c}"] = j[f"a_{c}"].fillna(0.0)
        frames.append(j)
        progress.step(f"{season} GW{g}: {len(j)} players")
    return pd.concat(frames, ignore_index=True)


def report(df: pd.DataFrame, season: str) -> dict:
    out = {"season": season}
    print(f"\n=== {season}: component calibration (Σ predicted vs Σ actual points) ===")
    print(f"{'component':<12}{'all pred':>10}{'all act':>10}{'ratio':>7}   "
          f"{'60+ pred':>10}{'60+ act':>10}{'ratio':>7}")
    sixty = df[df["minutes"] >= 60]
    for c in COMPS + ("total",):
        pc, ac = (f"c_{c}", f"a_{c}") if c != "total" else ("prediction", "a_total")
        p_all, a_all = df[pc].sum(), df[ac].sum()
        p60, a60 = sixty[pc].sum(), sixty[ac].sum()
        out[c] = {"pred": p_all, "act": a_all, "pred60": p60, "act60": a60}
        r1 = p_all / a_all if a_all else float("nan")
        r2 = p60 / a60 if a60 else float("nan")
        print(f"{c:<12}{p_all:>10.0f}{a_all:>10.0f}{r1:>7.3f}   "
              f"{p60:>10.0f}{a60:>10.0f}{r2:>7.3f}")

    print(f"\n--- P(60+) calibration by band, early (GW<=6) vs later ---")
    df["band"] = pd.cut(df["p_60"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01])
    df["got60"] = (df["minutes"] >= 60).astype(float)
    for label, sub in (("GW2-6", df[df["gw"] <= 6]), ("GW7+", df[df["gw"] > 6])):
        print(f"  {label}")
        for b, d in sub.groupby("band", observed=True):
            print(f"    {str(b):<14} n={len(d):>5}  pred {d['p_60'].mean():.3f}  "
                  f"real {d['got60'].mean():.3f}  diff {d['got60'].sum() - d['p_60'].sum():+.0f}")
    out["band_early"] = {str(b): (len(d), float(d["p_60"].mean()), float(d["got60"].mean()))
                         for b, d in df[df["gw"] <= 6].groupby("band", observed=True)}
    out["band_late"] = {str(b): (len(d), float(d["p_60"].mean()), float(d["got60"].mean()))
                        for b, d in df[df["gw"] > 6].groupby("band", observed=True)}

    print(f"\n--- conservation: per club-gameweek sums (single-fixture gws only) ---")
    one = df[df["n_fixtures"] == 1]
    club = one.groupby(["gw", "team_id"]).agg(
        s_start=("p_start", "sum"), s_full=("p_60", "sum"), s_min=("e_min", "sum"),
        a_start=("starts", "sum"), a_full=("got60", "sum"), a_min=("minutes", "sum"))
    for k, want in (("s_start", 11), ("s_full", None), ("s_min", 990)):
        a = {"s_start": "a_start", "s_full": "a_full", "s_min": "a_min"}[k]
        print(f"  Σ{k[2:]:<6} model mean {club[k].mean():7.1f} (sd {club[k].std():5.1f})"
              f"   actual mean {club[a].mean():7.1f}"
              + (f"   law {want}" if want else ""))
    out["club_sums"] = {k: float(club[k].mean()) for k in
                        ("s_start", "s_full", "s_min", "a_start", "a_full", "a_min")}
    early = club[club.index.get_level_values("gw") <= 6]
    print(f"  early-season (GW<=6): Σstart {early['s_start'].mean():.2f}  "
          f"Σ60+ {early['s_full'].mean():.2f} vs actual {early['a_full'].mean():.2f}  "
          f"Σmin {early['s_min'].mean():.0f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("seasons", nargs="+")
    ap.add_argument("--gws", nargs=2, type=int, default=(2, 38))
    ap.add_argument("--out-dir", default=os.path.join(config.DATA_DIR, "bt_base"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    conn = db.connect(config.DB_PATH)
    try:
        for season in args.seasons:
            path = os.path.join(args.out_dir, f"audit_{season}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                df = replay(conn, season, list(range(args.gws[0], args.gws[1] + 1)),
                            tag=f"bt{season}")
                df.to_csv(path, index=False)
            report(df, season)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
