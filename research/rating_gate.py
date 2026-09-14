"""Round 20 gate — is the crowd's eye test (BBC player ratings) information?

Two questions, both forward-in-time, both against what the engine already
knows, on the replayed seasons the ratings cover (2024-25, 2025-26):

  1. Minutes: does a player's rating in his LAST match predict whether he
     starts the NEXT one, beyond the shipped minutes model's P(start)?
     Logistic regression  started ~ logit(p_start) + rating_last(+ relative)
     fitted on 2024-25, scored on 2025-26 (and the reverse), log-loss delta.
  2. Points: does it predict next-match points beyond the engine's own
     projection?  pts ~ prediction + rating_last, same protocol, RMSE delta.

A rating is post-match, so the standing rule applies: anything measured from
realised outcomes tends to be inside rates fitted on them. The minutes
question is where it could still bite — a manager reads the same match.

    python research/rating_gate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_engine import config, db  # noqa: E402
from fpl_engine.xpts import bbc_role_features as br  # noqa: E402


def ratings(conn) -> pd.DataFrame:
    r = pd.read_sql_query(
        "SELECT r.event_urn, r.player_urn, r.rating, r.is_starter, m.kickoff_utc, m.season "
        "FROM acq_bbc_rating r JOIN acq_bbc_match m ON m.event_urn=r.event_urn "
        "WHERE r.rating IS NOT NULL", conn)
    pm = br.player_map(conn)
    r["player_code"] = r["player_urn"].map(pm)
    r = r.dropna(subset=["player_code"])
    r["player_code"] = r["player_code"].astype(int)
    r["kick"] = pd.to_datetime(r["kickoff_utc"], utc=True, errors="coerce")
    # relative to the match average (a 6.5 in a 4-0 win is not a 6.5 in a 0-4)
    r["rel"] = r["rating"] - r.groupby("event_urn")["rating"].transform("mean")
    return r.sort_values(["player_code", "kick"])


def attach_last_rating(a: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """For each audit row, the player's rating in his latest prior match."""
    a = a.copy()
    a["kick"] = pd.to_datetime(a["kick"], utc=True, errors="coerce")
    codes = pd.read_sql_query("SELECT season, player_id, code FROM player", CONN)
    a = a.merge(codes, on=["season", "player_id"], how="left")
    a = a.dropna(subset=["code", "kick"])
    a["player_code"] = a["code"].astype(int)
    left = a[["player_code", "kick"]].copy()
    left["_t"] = left["kick"] - pd.Timedelta(hours=6)
    left["_i"] = np.arange(len(left))
    rr = r[["player_code", "kick", "rating", "rel"]].rename(columns={"kick": "rkick"}).sort_values("rkick")
    rr["rating_l3"] = rr.groupby("player_code")["rating"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    m = pd.merge_asof(left.sort_values("_t"), rr, left_on="_t", right_on="rkick",
                      by="player_code", direction="backward", allow_exact_matches=False)
    m = m.set_index("_i").sort_index()
    for c in ("rating", "rel", "rating_l3"):
        a[c] = m[c].to_numpy()
    return a


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def minutes_test(a: pd.DataFrame) -> None:
    from sklearn.linear_model import LogisticRegression
    d = a.dropna(subset=["rating", "p_start"]).copy()
    d["started"] = (d["starts"] > 0).astype(int)
    d["lp"] = _logit(d["p_start"])
    print(f"\n=== Minutes: started ~ logit(P(start)) [+ last rating]   rows with a prior rating: {len(d)}")
    for train, test in (("2024-25", "2025-26"), ("2025-26", "2024-25")):
        tr, te = d[d.season == train], d[d.season == test]
        if len(tr) < 500 or len(te) < 500:
            print(f"  {train}->{test}: too few rows"); continue
        base = LogisticRegression(C=10).fit(tr[["lp"]], tr["started"])
        full = LogisticRegression(C=10).fit(tr[["lp", "rating", "rel", "rating_l3"]], tr["started"])
        from sklearn.metrics import log_loss
        lb = log_loss(te["started"], base.predict_proba(te[["lp"]])[:, 1])
        lf = log_loss(te["started"], full.predict_proba(te[["lp", "rating", "rel", "rating_l3"]])[:, 1])
        print(f"  train {train} -> test {test}: log-loss {lb:.4f} -> {lf:.4f}  ({(lf/lb-1)*100:+.2f}%)  "
              f"coef rating {full.coef_[0][1]:+.3f} rel {full.coef_[0][2]:+.3f} l3 {full.coef_[0][3]:+.3f}")
    # the ambiguous band, where a manager's mind is not made up
    band = d[(d.p_start >= 0.3) & (d.p_start < 0.7)]
    q = pd.qcut(band["rel"], 4, labels=["worst", "q2", "q3", "best"])
    print("  ambiguous band, started rate by last-match rating quartile (relative):")
    print(band.groupby(q, observed=True).agg(n=("started", "size"), model=("p_start", "mean"),
                                             started=("started", "mean")).round(3).to_string())


def points_test(a: pd.DataFrame) -> None:
    from sklearn.linear_model import Ridge
    d = a.dropna(subset=["rating"]).copy()
    d = d[d.minutes > 0]
    print(f"\n=== Points (players who played): pts ~ projection [+ last rating]   rows: {len(d)}")
    for train, test in (("2024-25", "2025-26"), ("2025-26", "2024-25")):
        tr, te = d[d.season == train], d[d.season == test]
        if len(tr) < 500 or len(te) < 500:
            continue
        base = Ridge(1.0).fit(tr[["prediction"]], tr["a_total"])
        full = Ridge(1.0).fit(tr[["prediction", "rating", "rel", "rating_l3"]], tr["a_total"])
        rb = np.sqrt(((base.predict(te[["prediction"]]) - te["a_total"]) ** 2).mean())
        rf = np.sqrt(((full.predict(te[["prediction", "rating", "rel", "rating_l3"]]) - te["a_total"]) ** 2).mean())
        print(f"  train {train} -> test {test}: RMSE {rb:.4f} -> {rf:.4f}  ({(rf/rb-1)*100:+.2f}%)  "
              f"coef rating {full.coef_[1]:+.3f} rel {full.coef_[2]:+.3f} l3 {full.coef_[3]:+.3f}")


if __name__ == "__main__":
    CONN = db.connect(config.DB_PATH)
    r = ratings(CONN)
    print("ratings rows mapped to FPL players:", len(r), "| matches:", r.event_urn.nunique(),
          "| per season:", r.groupby("season").size().to_dict())
    frames = []
    for season in ("2024-25", "2025-26"):
        path = os.path.join(config.DATA_DIR, "bt_base", f"audit_{season}.csv")
        a = pd.read_csv(path, usecols=["gw", "player_id", "position", "p_start", "starts", "minutes",
                                       "a_total", "prediction"])
        a["season"] = season
        ko = pd.read_sql_query("SELECT gw, MIN(kickoff_utc) kick FROM team_match WHERE season=? "
                               "GROUP BY gw", CONN, params=(season,))
        # the audit has no per-row kickoff; use the gameweek's first kickoff,
        # which is before every match in it (strictly point-in-time)
        a = a.merge(ko, on="gw", how="left")
        frames.append(a)
    a = attach_last_rating(pd.concat(frames, ignore_index=True), r)
    print("audit rows with a prior rating:", int(a["rating"].notna().sum()), "of", len(a))
    minutes_test(a)
    points_test(a)
    CONN.close()
