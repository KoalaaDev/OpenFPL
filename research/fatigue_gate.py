"""Round 20 gates — (a) does REST change a player's output per minute, and
(b) does a regular starter's ABSENCE change the club's goals for/against?

Both within-unit, so squad quality differences out:

 (a) starters who lasted 60+ minutes, output per 90 (xG+xA, points, BPS)
     demeaned by player-season, against: days since his own last league
     appearance, his own league minutes in the previous 7 and 14 days, the
     club's matches in the previous 7 days across every competition
     (`acq_bbc_calendar`) and a European tie in the previous 4 days.
     The minutes model already has rest/congestion; this asks the RATE
     question, which nothing in the engine scales.

 (b) team goals for and against, demeaned by team-season, when >= 1 regular
     starter (>= 4 of the club's last 5 starts) plays no minutes, split by the
     absent man's position, and when he is SUSPENDED (knowable pre-deadline).

    python research/fatigue_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402
from fpl_engine.xpts import absence_features as ab, calendar_features as cf  # noqa: E402

SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")


def ols(y, X, names):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    from scipy import stats
    for n, b, e in zip(["const"] + names, beta, se):
        t = b / e if e > 0 else 0
        print(f"      {n:<18} {b:+.4f}  se {e:.4f}  t {t:+.2f}  p {2 * stats.t.sf(abs(t), dof):.4f}")


def rest_test(conn):
    g = pd.read_sql_query(
        "SELECT pg.season, pg.gw, pg.player_id, pg.player_code, pg.team_id, pg.fixture_id, pg.kickoff_utc, "
        "pg.minutes, pg.total_points, pg.bps, pg.xg, pg.xa, pg.goals_scored, pg.assists, p.position "
        "FROM player_gw pg JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        "WHERE pg.season IN (%s) AND p.position IN ('GK','DEF','MID','FWD')" % ",".join("?" * len(SEASONS)),
        conn, params=list(SEASONS))
    g["kick"] = pd.to_datetime(g["kickoff_utc"], utc=True, errors="coerce")
    g = g.dropna(subset=["kick"]).sort_values(["player_code", "kick"])
    for c in ("minutes", "total_points", "bps", "xg", "xa"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    # own rest, from league appearances only
    played = g[g["minutes"] > 0][["player_code", "kick", "minutes"]].rename(columns={"kick": "pk", "minutes": "pm"})
    played = played.sort_values("pk")
    left = g[["player_code", "kick"]].copy()
    left["_i"] = np.arange(len(left))
    m = pd.merge_asof(left.sort_values("kick"), played, left_on="kick", right_on="pk",
                      by="player_code", direction="backward", allow_exact_matches=False)
    m = m.set_index("_i").sort_index()
    g["own_days_rest"] = ((g["kick"].to_numpy() - m["pk"].to_numpy()) / pd.Timedelta(days=1))
    # own minutes in the previous 7 / 14 days
    ks = g["kick"].to_numpy(); codes = g["player_code"].to_numpy(); mins = g["minutes"].fillna(0).to_numpy()
    o7 = np.zeros(len(g)); o14 = np.zeros(len(g))
    start = 0
    for i in range(len(g)):
        if i > 0 and codes[i] != codes[i - 1]:
            start = i
        j = i - 1
        while j >= start and (ks[i] - ks[j]) <= np.timedelta64(14, "D"):
            if ks[j] < ks[i]:
                o14[i] += mins[j]
                if (ks[i] - ks[j]) <= np.timedelta64(7, "D"):
                    o7[i] += mins[j]
            j -= 1
    g["own_min_7d"], g["own_min_14d"] = o7, o14
    # club calendar across competitions
    cal = cf.load(conn)
    g["club_prev7"] = np.nan; g["europe_prev4"] = np.nan
    if not cal.empty:
        cal = cal.sort_values("kick")
        for (season, tid), d in g.groupby(["season", "team_id"]):
            c = cal[(cal.season == season) & (cal.team_id == tid)]
            if c.empty:
                continue
            ck = c["kick"].to_numpy(); eu = c["europe"].to_numpy().astype(bool)
            rows = d.index
            kk = d["kick"].to_numpy()
            p7 = np.zeros(len(rows)); e4 = np.zeros(len(rows))
            for n, k in enumerate(kk):
                w = (ck < k) & (ck >= k - np.timedelta64(7, "D"))
                p7[n] = w.sum()
                e4[n] = (eu & (ck < k) & (ck >= k - np.timedelta64(4, "D"))).any()
            g.loc[rows, "club_prev7"] = p7
            g.loc[rows, "europe_prev4"] = e4
    d = g[(g["minutes"] >= 60) & g["own_days_rest"].notna() & (g["own_days_rest"] <= 30)].copy()
    d["xgi90"] = (d["xg"].fillna(0) + d["xa"].fillna(0)) / d["minutes"] * 90
    d["pts90"] = d["total_points"] / d["minutes"] * 90
    d["bps90"] = d["bps"] / d["minutes"] * 90
    d = d[d["position"] != "GK"]
    for c in ("xgi90", "pts90", "bps90"):
        d[c + "_w"] = d[c] - d.groupby(["season", "player_code"])[c].transform("mean")
    print(f"=== (a) REST vs output per 90, starters who lasted 60+ (non-GK), within player-season. rows {len(d)}")
    d["rest_b"] = pd.cut(d["own_days_rest"], [0, 3.5, 5.5, 8.5, 14.5, 31],
                         labels=["<=3d", "4-5d", "6-8d", "9-14d", "15-30d"])
    print(d.groupby("rest_b", observed=True)[["xgi90_w", "pts90_w", "bps90_w"]].agg(["mean", "size"]).round(4).to_string())
    d["m7_b"] = pd.cut(d["own_min_7d"], [-1, 0, 90, 180, 400], labels=["0", "1-90", "91-180", "181+"])
    print("  by own league minutes in the previous 7 days:")
    print(d.groupby("m7_b", observed=True)[["xgi90_w", "pts90_w", "bps90_w"]].agg(["mean", "size"]).round(4).to_string())
    dd = d.dropna(subset=["club_prev7"])
    if len(dd):
        print("  by club matches in the previous 7 days (all competitions):")
        print(dd.groupby("club_prev7")[["xgi90_w", "pts90_w", "bps90_w"]].agg(["mean", "size"]).round(4).to_string())
        print("  European tie in the previous 4 days:")
        print(dd.groupby("europe_prev4")[["xgi90_w", "pts90_w", "bps90_w"]].agg(["mean", "size"]).round(4).to_string())
    X = d[["own_days_rest", "own_min_7d"]].copy()
    X["own_days_rest"] = X["own_days_rest"].clip(upper=14)
    X["own_min_7d"] = X["own_min_7d"] / 90.0
    for y in ("xgi90_w", "pts90_w"):
        print(f"  OLS {y} ~ own_days_rest(<=14) + own_min_7d/90:")
        ols(d[y].to_numpy(), X.to_numpy(float), list(X.columns))
    if len(dd):
        X2 = dd[["own_days_rest", "club_prev7", "europe_prev4"]].copy()
        X2["own_days_rest"] = X2["own_days_rest"].clip(upper=14)
        for y in ("xgi90_w", "pts90_w"):
            print(f"  OLS {y} ~ own_days_rest + club_prev7 + europe_prev4  (rows {len(dd)}):")
            ols(dd[y].to_numpy(), X2.to_numpy(float), list(X2.columns))
    return g


def team_test(conn, g):
    tm = pd.read_sql_query(
        "SELECT season, team_id, fixture_id, goals_for, goals_against, xg, xga FROM team_match "
        "WHERE season IN (%s)" % ",".join("?" * len(SEASONS)), conn, params=list(SEASONS))
    for c in ("goals_for", "goals_against", "xg", "xga"):
        tm[c] = pd.to_numeric(tm[c], errors="coerce")
        tm[c + "_w"] = tm[c] - tm.groupby(["season", "team_id"])[c].transform("mean")
    # regular starters: >= 4 starts in his previous 5 rows
    st = pd.read_sql_query("SELECT season, player_code, fixture_id, starts FROM player_gw WHERE season IN (%s)"
                           % ",".join("?" * len(SEASONS)), conn, params=list(SEASONS))
    g = g.merge(st, on=["season", "player_code", "fixture_id"], how="left")
    g = g.sort_values(["player_code", "kick"])
    g["starts_l5"] = g.groupby("player_code")["starts"].transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum())
    g["regular"] = (g["starts_l5"] >= 4)
    g["absent"] = g["regular"] & (g["minutes"].fillna(0) == 0)
    sus = ab.load(conn)
    sus["sus"] = True
    g = g.merge(sus, on=["season", "team_id", "player_code", "fixture_id"], how="left")
    g["sus"] = g["sus"].fillna(False).astype(bool)
    g["suspended"] = g["regular"] & g["sus"]
    key = ["season", "team_id", "fixture_id"]
    agg = g.groupby(key).agg(n_abs=("absent", "sum"), n_sus=("suspended", "sum"),
                             abs_def=("absent", lambda s: s[g.loc[s.index, "position"].isin(["GK", "DEF"])].sum()),
                             abs_att=("absent", lambda s: s[g.loc[s.index, "position"].isin(["MID", "FWD"])].sum()))
    t = tm.merge(agg.reset_index(), on=key, how="left").fillna({"n_abs": 0, "n_sus": 0, "abs_def": 0, "abs_att": 0})
    print(f"\n=== (b) TEAM goals for/against (demeaned by team-season) when regular starters are missing. team-matches {len(t)}")
    from scipy import stats
    for label, col in (("any regular absent", "n_abs"), ("regular SUSPENDED", "n_sus"),
                       ("GK/DEF regular absent", "abs_def"), ("MID/FWD regular absent", "abs_att")):
        a, b = t[t[col] > 0], t[t[col] == 0]
        print(f"  {label}: n {len(a)} vs {len(b)}")
        for c in ("goals_for_w", "goals_against_w", "xg_w", "xga_w"):
            x, y = a[c].dropna(), b[c].dropna()
            p = stats.ttest_ind(x, y, equal_var=False).pvalue if len(x) > 10 else float("nan")
            print(f"      {c:<16} {x.mean():+.3f} vs {y.mean():+.3f}  diff {x.mean()-y.mean():+.3f}  p={p:.4f}")
    print("  by number of regulars absent:")
    print(t.groupby(t["n_abs"].clip(upper=3))[["goals_for_w", "goals_against_w", "xg_w", "xga_w"]].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    conn = db.connect(config.DB_PATH)
    g = rest_test(conn)
    team_test(conn, g)
    conn.close()
