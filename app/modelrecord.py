"""The model's season record: what it picked before each deadline, and how
that went against the average manager and against perfect hindsight.

For every finished gameweek this builds, once, and saves under
``data/model_record_<season>.json``:

  * the model's PICKS, from the projections it held before the deadline —
      - an unlimited-budget best XI (who the model would start), and
      - a £100m squad of fifteen with a bench, scored with FPL's autosubs and
        the armband passing to the vice: the fair comparison with a manager;
  * the references — FPL's average and highest score, the crowd's most
    captained player, and the best possible XI in hindsight under the SAME
    rules (so "captured x% of the ceiling" compares like with like);
  * accuracy — rank correlation, error and bias by position, a calibration
    table by projection size, and the engine's components (goals, assists,
    clean sheets, DefCon crossings, bonus) against what happened.

WHERE THE PICKS COME FROM, and it matters more than anything else on the page.
A gameweek is scored on what the model believed BEFORE its deadline, never on
a projection rebuilt afterwards:

  * ``live``   — the last archived projection snapshot built before the
                 deadline (``services._load_history``). Exactly what the site
                 showed. Snapshots begin 2026-09-11, so GW4 onward.
  * ``replay`` — for earlier gameweeks: the engine re-run strictly point in
                 time (data before the first kickoff; a minutes model trained
                 on seasons that never saw this one) with availability as the
                 change log stood AT THE DEADLINE. The data is honest; the
                 code is today's, which is why it is labelled.

A gameweek that has both reports how closely they agree, which is the check
that a replay is a fair stand-in for a live week.

What the £100m squad is NOT: a season-long team. It is rebuilt from scratch
each week — a free wildcard every week, no chips, no hits — which flatters it
against a real manager who carries last week's team. The page says so.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fpl_engine import config, db

RECORD_VERSION = 2
BUDGET = 100.0
POS = ("GK", "DEF", "MID", "FWD")
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
CALIBRATION_EDGES = [0.3, 1, 2, 3, 4, 5, 6, 8, 99]


def _path(season: str) -> str:
    return os.path.join(config.DATA_DIR, f"model_record_{season}.json")


def load(season: str | None = None) -> dict:
    season = season or config.CURRENT_SEASON
    try:
        with open(_path(season), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"season": season, "version": RECORD_VERSION, "gws": {}}


def _save(doc: dict) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = _path(doc["season"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)
    os.replace(tmp, _path(doc["season"]))


# ------------------------------------------------------------------ inputs --
def _events() -> dict[int, dict]:
    from . import services
    try:
        return {int(e["id"]): e for e in services.bootstrap().get("events", [])}
    except Exception:  # noqa: BLE001
        return {}


def _meta(conn, season: str) -> dict[int, dict]:
    return {int(r[0]): {"name": r[1], "pos": r[2], "team_id": int(r[3]) if r[3] is not None else None}
            for r in conn.execute(
                "SELECT player_id, web_name, position, team_id FROM player "
                "WHERE season = ? AND position IN ('GK','DEF','MID','FWD')", (season,))}


def _actuals(conn, season: str, gw: int, thresholds: dict) -> pd.DataFrame:
    """Per player for the gameweek, doubles summed. DefCon crossings are
    counted per FIXTURE (a double gameweek can cross twice)."""
    rows = pd.read_sql_query(
        "SELECT player_id, fixture_id, MAX(total_points) pts, MAX(minutes) mins, "
        "MAX(goals_scored) goals, MAX(assists) assists, MAX(clean_sheets) cs, "
        "MAX(bonus) bonus, MAX(defcon) defcon FROM player_gw "
        "WHERE season = ? AND gw = ? GROUP BY player_id, fixture_id",
        conn, params=(season, gw))
    if rows.empty:
        return rows
    pos = {r[0]: r[1] for r in conn.execute(
        "SELECT player_id, position FROM player WHERE season = ?", (season,))}
    rows["crossed"] = [
        1 if (d is not None and not pd.isna(d) and pos.get(p) in thresholds
              and d >= thresholds[pos[p]]) else 0
        for p, d in zip(rows["player_id"], rows["defcon"])]
    return rows.groupby("player_id", as_index=False).agg(
        pts=("pts", "sum"), mins=("mins", "sum"), goals=("goals", "sum"),
        assists=("assists", "sum"), cs=("cs", "sum"), bonus=("bonus", "sum"),
        crossed=("crossed", "sum"))


def _prices(conn, season: str, gw: int) -> dict[int, float]:
    """£m at that gameweek: player_gw.price is tenths; fall back to the last
    earlier gameweek, then to today's price for a player with no row yet."""
    out = {int(r[0]): float(r[1]) / 10.0 for r in conn.execute(
        "SELECT player_id, price FROM ("
        "  SELECT player_id, price, ROW_NUMBER() OVER (PARTITION BY player_id "
        "    ORDER BY gw DESC) rn FROM player_gw "
        "  WHERE season = ? AND gw <= ? AND price IS NOT NULL) WHERE rn = 1",
        (season, gw))}
    for pid, cost in conn.execute(
            "SELECT player_id, now_cost FROM player WHERE season = ?", (season,)):
        if int(pid) not in out and cost:
            out[int(pid)] = float(cost)
    return out


def _live_snapshot(gw: int, deadline: float) -> tuple[dict[int, float], float] | None:
    from . import services
    snaps = [s for s in services._load_history()
             if s.get("built_at", 0) < deadline and str(gw) in (s.get("gws") or {})]
    if not snaps:
        return None
    last = max(snaps, key=lambda s: s["built_at"])
    return ({int(pid): float(ep) for pid, ep in last["gws"][str(gw)].items()},
            float(last["built_at"]))


def _replay(conn, season: str, gw: int, deadline_iso: str) -> pd.DataFrame:
    """The engine at the first kickoff, availability as it stood at the
    deadline. Components are scaled by the same factor as the projection."""
    from fpl_engine import lineup_feed as lf
    from fpl_engine.xpts import engine
    as_of = engine.first_kickoff(conn, season, gw)
    pred = engine.xpts_predict_gw(conn, season, gw, as_of=as_of, use_availability=False)
    if pred.empty:
        return pred
    avail = lf.availability_at(conn, season, deadline_iso)
    f = pred["player_id"].map(lambda p: avail.get(int(p), 1.0)) if avail else 1.0
    pred = pred.copy()
    for col in ["prediction", "e_goals", "e_assists", "p_cs"] + [
            c for c in pred.columns if c.startswith("c_")]:
        if col in pred.columns:
            pred[col] = pred[col] * f
    pred.attrs["availability"] = bool(avail)
    return pred


# -------------------------------------------------------------- selection --
def _select(rows: list[dict], value: str, budget: float | None = None,
            bench_weight: float = 0.1) -> dict | None:
    """The best legal XI by `value` — or, with a budget, the best fifteen
    whose XI scores most. Captain counted twice. Exact (PuLP/CBC)."""
    import pulp
    ps = [r for r in rows if r.get(value) is not None and r["pos"] in POS]
    if budget is not None:
        ps = [r for r in ps if r.get("price")]
    if len(ps) < (15 if budget is not None else 11):
        return None
    prob = pulp.LpProblem("pick", pulp.LpMaximize)
    ids = [r["player_id"] for r in ps]
    by = {r["player_id"]: r for r in ps}
    x = pulp.LpVariable.dicts("x", ids, cat="Binary")          # in the XI
    c = pulp.LpVariable.dicts("c", ids, cat="Binary")          # captain
    obj = pulp.lpSum(by[i][value] * (x[i] + c[i]) for i in ids)
    prob += pulp.lpSum(x.values()) == 11
    prob += pulp.lpSum(c.values()) == 1
    for i in ids:
        prob += c[i] <= x[i]
    for pos, (lo, hi) in XI_LIMITS.items():
        n = pulp.lpSum(x[i] for i in ids if by[i]["pos"] == pos)
        prob += n >= lo
        prob += n <= hi
    clubs = {by[i]["team_id"] for i in ids}
    if budget is None:
        for club in clubs:
            prob += pulp.lpSum(x[i] for i in ids if by[i]["team_id"] == club) <= 3
    else:
        s = pulp.LpVariable.dicts("s", ids, cat="Binary")      # in the fifteen
        obj += bench_weight * pulp.lpSum(by[i][value] * (s[i] - x[i]) for i in ids)
        for i in ids:
            prob += x[i] <= s[i]
        for pos, n in SQUAD_SIZE.items():
            prob += pulp.lpSum(s[i] for i in ids if by[i]["pos"] == pos) == n
        for club in clubs:
            prob += pulp.lpSum(s[i] for i in ids if by[i]["team_id"] == club) <= 3
        prob += pulp.lpSum(by[i]["price"] * s[i] for i in ids) <= budget
    prob += obj
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=20))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    xi = [by[i] for i in ids if x[i].value() > 0.5]
    cap = next(by[i] for i in ids if c[i].value() > 0.5)
    vice = max((p for p in xi if p is not cap), key=lambda p: p[value])
    out = {"xi": xi, "captain": cap, "vice": vice}
    if budget is not None:
        bench = [by[i] for i in ids if s[i].value() > 0.5 and x[i].value() < 0.5]
        # FPL bench order: the keeper first, then by the value that picked them
        bench.sort(key=lambda p: (p["pos"] != "GK", -p[value]))
        out["bench"] = bench
    return out


def _score(sel: dict, with_autosubs: bool) -> float:
    """Actual points for a selection: armband to the vice if the captain did
    not play; FPL autosubs from the bench when there is one."""
    xi, cap, vice = sel["xi"], sel["captain"], sel["vice"]
    if not with_autosubs:
        base = sum(p["pts"] for p in xi)
        arm = cap["pts"] if cap["mins"] > 0 else (vice["pts"] if vice["mins"] > 0 else 0)
        return float(base + arm)
    from fpl_engine.xpts.rank_utility import autosub_points
    squad = xi + sel["bench"]
    pts = np.array([[float(p["pts"]) for p in squad]])
    played = np.array([[p["mins"] > 0 for p in squad]])
    idx = {p["player_id"]: k for k, p in enumerate(squad)}
    return float(autosub_points(
        pts, played, [p["pos"] for p in squad],
        [idx[p["player_id"]] for p in xi], idx[cap["player_id"]], idx[vice["player_id"]],
        [idx[p["player_id"]] for p in sel["bench"]])[0])


def _brief(sel: dict, value: str) -> dict:
    order = {p: k for k, p in enumerate(POS)}
    def one(p):
        return {"player_id": p["player_id"], "name": p["name"], "pos": p["pos"],
                "team_id": p["team_id"], "price": p.get("price"),
                "ep": round(float(p.get("ep") or 0), 2), "pts": int(p["pts"]),
                "mins": int(p["mins"]),
                "captain": p is sel["captain"], "vice": p is sel["vice"]}
    xi = sorted(sel["xi"], key=lambda p: (order[p["pos"]], -p[value]))
    out = {"xi": [one(p) for p in xi],
           "formation": "-".join(str(sum(p["pos"] == q for p in xi)) for q in POS[1:]),
           "projected": round(sum(float(p.get("ep") or 0) for p in xi)
                              + float(sel["captain"].get("ep") or 0), 1)}
    if "bench" in sel:
        out["bench"] = [one(p) for p in sel["bench"]]
        out["cost"] = round(sum(p["price"] for p in sel["xi"] + sel["bench"]), 1)
    return out


# --------------------------------------------------------------- one week --
def build_gw(conn, season: str, gw: int, event: dict) -> dict | None:
    thresholds = _defcon_thresholds()
    act = _actuals(conn, season, gw, thresholds)
    if act.empty:
        return None
    meta = _meta(conn, season)
    deadline = float(event.get("deadline_time_epoch") or 0)
    deadline_iso = event.get("deadline_time") or datetime.fromtimestamp(
        deadline, timezone.utc).isoformat()

    replay = _replay(conn, season, gw, deadline_iso)
    rep_ep = dict(zip(replay["player_id"].astype(int), replay["prediction"].astype(float))) \
        if not replay.empty else {}
    snap = _live_snapshot(gw, deadline)
    if snap:
        ep, source, built = snap[0], "live", snap[1]
    else:
        ep, source, built = rep_ep, "replay", None

    prices = _prices(conn, season, gw)
    a = act.set_index("player_id")
    rows = []
    for pid, m in meta.items():
        r = a.loc[pid] if pid in a.index else None
        rows.append({"player_id": pid, "name": m["name"], "pos": m["pos"],
                     "team_id": m["team_id"], "price": prices.get(pid),
                     "ep": ep.get(pid, 0.0),
                     "pts": int(r["pts"]) if r is not None else 0,
                     "mins": int(r["mins"]) if r is not None else 0})

    free = _select(rows, "ep")
    budget = _select(rows, "ep", budget=BUDGET)
    best_free = _select(rows, "pts")
    best_budget = _select(rows, "pts", budget=BUDGET)

    df = pd.DataFrame(rows)
    played = df[df["mins"] > 0]
    spearman = float(df["ep"].corr(df["pts"], method="spearman"))
    sp_played = float(played["ep"].corr(played["pts"], method="spearman")) if len(played) > 10 else None
    top20 = len(set(df.nlargest(20, "ep")["player_id"]) & set(df.nlargest(20, "pts")["player_id"]))
    by_pos = {}
    for pos in POS:
        d = played[played["pos"] == pos]
        if len(d):
            err = d["ep"] - d["pts"]
            by_pos[pos] = {"n": int(len(d)), "mae": round(float(err.abs().mean()), 2),
                           "bias": round(float(err.mean()), 2)}
    calib = []
    for lo, hi in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:]):
        d = df[(df["ep"] >= lo) & (df["ep"] < hi)]
        calib.append({"lo": lo, "hi": hi, "n": int(len(d)),
                      "pred": round(float(d["ep"].sum()), 2), "act": float(d["pts"].sum())})

    components = None
    if not replay.empty:
        j = replay.merge(act, on="player_id", how="left").fillna(0)
        j["pos"] = j["player_id"].map(lambda p: (meta.get(int(p)) or {}).get("pos"))

        def pair(d, pred_col, act_col, k=1.0):
            return {"pred": round(float(d[pred_col].sum()) * k, 1) if pred_col in d else None,
                    "act": float(d[act_col].sum())}
        gkdef = j[j["pos"].isin(["GK", "DEF"])]
        components = {
            "goals": pair(j, "e_goals", "goals"),
            "assists": pair(j, "e_assists", "assists"),
            # where a clean sheet is worth 4-6 points; a midfielder's 1 point
            # and a forward's 0 are noise in the same count
            "clean_sheets": pair(gkdef, "p_cs", "cs"),
            "defcon_crossings": pair(j, "c_defcon", "crossed", 1.0 / POINTS_PER_DEFCON),
            "bonus": pair(j, "c_bonus", "bonus"),
        }
        team_cs = _team_clean_sheets(conn, season, gw, replay)
        if team_cs:
            components["team_clean_sheets"] = team_cs

    agree = None
    if source == "live" and rep_ep:
        both = pd.DataFrame({"live": pd.Series(ep), "replay": pd.Series(rep_ep)}).dropna()
        both = both[(both["live"] > 0.2) | (both["replay"] > 0.2)]
        if len(both) > 20:
            agree = round(float(both["live"].corr(both["replay"], method="spearman")), 3)

    misses = []
    if free:
        for p in free["xi"]:
            misses.append({"name": p["name"], "pos": p["pos"], "team_id": p["team_id"],
                           "ep": round(p["ep"], 2), "pts": int(p["pts"]),
                           "delta": round(p["pts"] - p["ep"], 2)})
    mc = event.get("most_captained")
    crowd = next((r for r in rows if r["player_id"] == mc), None) if mc else None
    top = max(rows, key=lambda r: r["pts"])

    return {
        "gw": gw, "source": source, "snapshot_built_at": built,
        "deadline": deadline_iso, "availability_at_deadline": bool(
            getattr(replay, "attrs", {}).get("availability")) if not replay.empty else False,
        "live_vs_replay_spearman": agree,
        "fpl": {"average": event.get("average_entry_score"),
                "highest": event.get("highest_score")},
        "model_xi": {**_brief(free, "ep"), "actual": _score(free, False)} if free else None,
        "model_squad": {**_brief(budget, "ep"), "actual": _score(budget, True)} if budget else None,
        "hindsight_xi": {**_brief(best_free, "pts"), "actual": _score(best_free, False)} if best_free else None,
        "hindsight_squad": {**_brief(best_budget, "pts"), "actual": _score(best_budget, True)} if best_budget else None,
        "captain": {
            "model": {"name": free["captain"]["name"], "pts": int(free["captain"]["pts"]),
                      "ep": round(free["captain"]["ep"], 2)} if free else None,
            "crowd": {"name": crowd["name"], "pts": int(crowd["pts"])} if crowd else None,
            "best": {"name": top["name"], "pts": int(top["pts"])},
        },
        "accuracy": {"spearman": round(spearman, 3),
                     "spearman_played": round(sp_played, 3) if sp_played is not None else None,
                     "top20_hits": top20, "n_played": int(len(played)),
                     "mae_played": round(float((played["ep"] - played["pts"]).abs().mean()), 2),
                     "bias_played": round(float((played["ep"] - played["pts"]).mean()), 2),
                     "by_pos": by_pos},
        "calibration": calib,
        "components": components,
        "xi_deltas": sorted(misses, key=lambda m: m["delta"]),
        "built_at": time.time(),
    }


POINTS_PER_DEFCON = 2


def _team_clean_sheets(conn, season: str, gw: int, replay: pd.DataFrame) -> dict | None:
    """Expected vs actual TEAM clean sheets: the cleaner check, because a
    player-level count also depends on who lasted 60 minutes. The expectation
    is exp(-lambda_against) from the replay, one per club with a fixture."""
    if "lam_against" not in replay.columns or "team_id" not in replay.columns:
        return None
    lam = (replay[replay["lam_against"] > 0].groupby("team_id")["lam_against"].max())
    rows = conn.execute(
        "SELECT team_h, team_a, team_h_score, team_a_score FROM fixture "
        "WHERE season = ? AND gw = ? AND finished = 1", (season, gw)).fetchall()
    if not rows or lam.empty:
        return None
    actual = sum((r[3] == 0) + (r[2] == 0) for r in rows)
    teams = {r[0] for r in rows} | {r[1] for r in rows}
    pred = sum(float(np.exp(-lam[t])) for t in teams if t in lam.index)
    return {"pred": round(pred, 1), "act": float(actual)}


def _defcon_thresholds() -> dict:
    try:
        from fpl_engine import scoring
        rules = scoring.load_rules()
        return dict((rules.get("defensive_contribution") or {}).get("threshold") or {})
    except Exception:  # noqa: BLE001
        return {"DEF": 10, "MID": 12, "FWD": 12}


# ------------------------------------------------------------- the season --
def refresh(season: str | None = None, *, force: bool = False, progress=None) -> dict:
    """Build the record for every finished gameweek that does not have one.
    Idempotent: a gameweek is built once (or again when RECORD_VERSION moves)."""
    season = season or config.CURRENT_SEASON
    doc = load(season)
    if doc.get("version") != RECORD_VERSION:
        doc = {"season": season, "version": RECORD_VERSION, "gws": {}}
    events = _events()
    done = []
    conn = db.connect(config.DB_PATH)
    try:
        for gw, ev in sorted(events.items()):
            if not ev.get("finished") or (str(gw) in doc["gws"] and not force):
                continue
            if progress:
                progress(f"Recording the model's GW{gw} picks against what happened…")
            rec = build_gw(conn, season, gw, ev)
            if rec:
                doc["gws"][str(gw)] = rec
                done.append(gw)
    finally:
        conn.close()
    if done:
        doc["updated_at"] = time.time()
        _save(doc)
    return {"built": done, "total": len(doc["gws"])}
