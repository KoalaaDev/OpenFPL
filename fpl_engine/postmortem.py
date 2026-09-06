"""Gameweek post-mortem: what the model believed at the deadline vs reality.

Recomputes the xPts prediction point-in-time (as_of = the gameweek's first
kickoff, no availability overlay — the stored status is today's, not that
week's) and decomposes every big miss into three buckets:

  MINUTES  — the player didn't play as expected (or started unexpectedly)
  VARIANCE — right process, wrong outcome (the xG arrived, the goals didn't)
  RATE     — the underlying per-90 / team assumption looks wrong

Run after each gameweek's ``pull``:

    python -m fpl_engine postmortem            # latest finished gw
    python -m fpl_engine postmortem --gw 3

The report is printed and saved to data/postmortem_{season}_gw{N}.json so the
season's error history accumulates alongside the backtests.
"""
from __future__ import annotations

import json
import os

import pandas as pd
from scipy.stats import spearmanr

from . import config
from .xpts import engine


def classify_miss(r) -> str:
    """Bucket one over-prediction row (needs p_play, mins, e_goals, e_assists,
    axg, axa, goals, assists, p_cs, conceded attributes)."""
    if r.p_play >= 0.45 and r.mins == 0:
        return "MINUTES (didn't play)"
    if r.p_play < 0.30 and r.mins >= 60:
        return "MINUTES (surprise start)"
    if r.mins >= 45:
        got_xgi = (r.axg or 0) + (r.axa or 0)
        exp_xgi = r.e_goals + r.e_assists
        if exp_xgi >= 0.35 and got_xgi >= 0.55 * exp_xgi and \
                (r.goals + r.assists) == 0:
            return "VARIANCE (chances came, no returns)"
        if exp_xgi >= 0.35 and got_xgi < 0.4 * exp_xgi:
            return "RATE (chances never came)"
        if r.p_cs >= 0.35 and (r.conceded or 0) >= 2:
            return "RATE/TEAM (defence overrated)"
    return "mixed"


def latest_finished_gw(conn, season: str) -> int | None:
    r = conn.execute(
        "SELECT MAX(gw) g FROM player_gw WHERE season=?", (season,)).fetchone()
    return int(r["g"]) if r and r["g"] is not None else None


def run(conn, season: str | None = None, gw: int | None = None,
        top: int = 12) -> dict:
    season = season or config.CURRENT_SEASON
    gw = gw or latest_finished_gw(conn, season)
    if gw is None:
        raise SystemExit(f"No {season} results in the database yet — run "
                         "`python -m fpl_engine pull` after the gameweek.")

    actual = pd.read_sql_query(
        "SELECT player_id, SUM(total_points) pts, SUM(minutes) mins, "
        "SUM(goals_scored) goals, SUM(assists) assists, SUM(xg) axg, "
        "SUM(xa) axa, SUM(clean_sheets) cs, SUM(goals_conceded) conceded, "
        "SUM(bonus) bonus FROM player_gw WHERE season=? AND gw=? "
        "GROUP BY player_id", conn, params=(season, gw))
    if actual.empty:
        raise SystemExit(f"No results stored for {season} GW{gw} — pull first.")

    as_of = engine.first_kickoff(conn, season, gw)
    pred = engine.xpts_predict_gw(conn, season, gw, as_of=as_of,
                                  use_availability=False)
    meta = {r["player_id"]: (r["web_name"], r["position"], r["name"])
            for r in conn.execute(
                "SELECT p.player_id, p.web_name, p.position, t.name FROM player p "
                "LEFT JOIN team t ON t.season=p.season AND t.team_id=p.team_id "
                "WHERE p.season=?", (season,))}

    j = actual.merge(pred, on="player_id", how="outer").fillna(
        {"pts": 0, "mins": 0, "prediction": 0, "e_min": 0, "e_goals": 0,
         "e_assists": 0, "p_cs": 0, "p_60": 0, "p_play": 0})
    j["err"] = j["prediction"] - j["pts"]

    def row_out(r, with_class=False):
        nm, pos, team = meta.get(r.player_id, ("?", "?", ""))
        d = {"player": nm, "position": pos, "team": team,
             "pred": round(r.prediction, 2), "actual": float(r.pts),
             "e_min": round(r.e_min), "mins": float(r.mins),
             "exp_xgi": round(r.e_goals + r.e_assists, 2),
             "actual_xgi": round((r.axg or 0) + (r.axa or 0), 2)}
        if with_class:
            d["why"] = classify_miss(r)
        return d

    cap = j.loc[j["prediction"].idxmax()]
    played = j[j.mins >= 60]
    gk_def = played[played.position.isin(["GK", "DEF"])]
    mm = j[j.p_play > 0.02].assign(
        bucket=pd.cut(j.p_60, [0, .2, .4, .6, .8, 1.01]))
    calib = mm.groupby("bucket", observed=True).apply(
        lambda d: {"n": int(len(d)), "pred_p60": round(float(d.p_60.mean()), 2),
                   "real_60share": round(float((d.mins >= 60).mean()), 2)},
        include_groups=False).to_dict()

    top_p = set(j.nlargest(20, "prediction")["player_id"])
    top_a = set(j.nlargest(20, "pts")["player_id"])
    report = {
        "season": season, "gw": gw,
        "spearman": round(float(spearmanr(j.prediction, j.pts).statistic), 3),
        "top20_hits": len(top_p & top_a),
        "captain_pick": meta.get(cap.player_id, ("?",))[0],
        "captain_actual": float(cap.pts),
        "captain_best": float(j.pts.max()),
        "predicted_total": round(float(j.prediction.sum())),
        "actual_total": round(float(j.pts.sum())),
        "over_predictions": [row_out(r, True) for r in
                             j.nlargest(top, "err").itertuples()],
        "under_predictions": [row_out(r) for r in
                              j.nsmallest(top, "err").itertuples()],
        "components_60plus": {
            "n": int(len(played)),
            "goals": {"predicted": round(float(played.e_goals.sum()), 1),
                      "actual": int(played.goals.sum()),
                      "actual_xg": round(float(played.axg.sum()), 1)},
            "assists": {"predicted": round(float(played.e_assists.sum()), 1),
                        "actual": int(played.assists.sum()),
                        "actual_xa": round(float(played.axa.sum()), 1)},
            "clean_sheets_gk_def": {"predicted": round(float(gk_def.p_cs.sum()), 1),
                                    "actual": int(gk_def.cs.sum())},
        },
        "minutes_calibration": {str(k): v for k, v in calib.items()},
    }
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"postmortem_{season}_gw{gw}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    report["saved_to"] = path
    return report


def print_report(rep: dict) -> None:
    # player names carry diacritics; a cp1252 console must not abort the report
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    print(f"\nPost-mortem {rep['season']} GW{rep['gw']}: "
          f"spearman {rep['spearman']} · top-20 hits {rep['top20_hits']}/20 · "
          f"captain {rep['captain_pick']} -> {rep['captain_actual']:.0f} pts "
          f"(best {rep['captain_best']:.0f})")
    print(f"calibration: predicted {rep['predicted_total']} vs actual "
          f"{rep['actual_total']} "
          f"({rep['predicted_total'] / max(1, rep['actual_total']) * 100:.0f}%)")

    def block(title, rows, with_why):
        print(f"\n--- {title} ---")
        for d in rows:
            line = (f"  {d['player'][:16]:<16} {d['position']:<3} "
                    f"{(d['team'] or '')[:12]:<12} pred {d['pred']:4.1f} "
                    f"got {d['actual']:3.0f} | e_min {d['e_min']:3.0f} "
                    f"min {d['mins']:3.0f} | exGI {d['exp_xgi']:.2f} "
                    f"axGI {d['actual_xgi']:.2f}")
            if with_why:
                line += f" | {d['why']}"
            print(line)

    block("biggest OVER-predictions", rep["over_predictions"], True)
    block("biggest UNDER-predictions (missed hauls)", rep["under_predictions"], False)
    c = rep["components_60plus"]
    print(f"\n--- component calibration (60+ mins, n={c['n']}) ---")
    print(f"  goals:   predicted {c['goals']['predicted']}  actual "
          f"{c['goals']['actual']}  (xG {c['goals']['actual_xg']})")
    print(f"  assists: predicted {c['assists']['predicted']}  actual "
          f"{c['assists']['actual']}  (xA {c['assists']['actual_xa']})")
    print(f"  clean sheets GK/DEF: predicted "
          f"{c['clean_sheets_gk_def']['predicted']}  actual "
          f"{c['clean_sheets_gk_def']['actual']}")
    print("\n--- minutes calibration P(60+) predicted vs realised ---")
    for k, v in rep["minutes_calibration"].items():
        print(f"  {k:<12} n {v['n']:>4}  pred {v['pred_p60']:.2f}  "
              f"real {v['real_60share']:.2f}")
    print(f"\nsaved: {rep['saved_to']}")
