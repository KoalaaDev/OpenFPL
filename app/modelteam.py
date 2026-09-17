"""The model's own team: one £100m squad, carried through the season.

The season record (`modelrecord.py`) rebuilds a best squad from scratch every
gameweek — a free wildcard weekly, which flatters it against a real manager.
This is the honest benchmark: the model picks fifteen before GW1, then every
week decides its transfers the way a manager has to —

  * from what it knew BEFORE that deadline (the live projection snapshot
    where one exists, otherwise a strictly point-in-time replay: data before
    that gameweek's first kickoff, injury flags as they stood at the deadline,
    and bookmaker odds only for the gameweek being decided — later weeks' odds
    were published after the deadline, so they are left out);
  * with free transfers banking by FPL's rules (one a week, up to five) and
    extra moves costing four points each, priced into the decision by the
    solver itself;
  * selling at purchase price plus half the profit, rounded down;
  * playing the XI and armband it chose, scored with FPL's autosubs, hits
    deducted.

No chips, on purpose: the comparison with the average manager should be the
model's projections and transfer decisions, not the timing of four chips.

Built once per finished gameweek by the scheduled refresh and saved to
``data/model_team_<season>.json``, with the state carried between weeks (the
fifteen and what was paid for them, the bank, the free transfers), so a new
gameweek extends the season rather than replaying it. The plan for the next
deadline is refreshed each time from the current projections.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from fpl_engine import config, db

TEAM_VERSION = 1
BUDGET = 100.0
HORIZON = 3
SOLVE = {"decay": 0.85, "ft_value": 1.5, "hit_cost": 4.0, "bench_weight": 0.1,
         "max_transfers_per_gw": 3, "time_limit": 45}
MAX_FT = 5


def _path(season: str) -> str:
    return os.path.join(config.DATA_DIR, f"model_team_{season}.json")


def load(season: str | None = None) -> dict:
    season = season or config.CURRENT_SEASON
    try:
        with open(_path(season), encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("version") == TEAM_VERSION:
            return doc
    except (OSError, ValueError):
        pass
    return {"season": season, "version": TEAM_VERSION, "weeks": {}, "state": None,
            "replay_cache": {}, "next": None}


def _save(doc: dict) -> None:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tmp = _path(doc["season"]) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, default=float)
    os.replace(tmp, _path(doc["season"]))


# ------------------------------------------------------------ projections --
def _horizon(conn, season: str, gw: int) -> list[int]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT gw FROM fixture WHERE season = ? AND gw >= ? AND gw IS NOT NULL "
        "ORDER BY gw", (season, gw))][:HORIZON]


def _believed(conn, season: str, gw: int, event: dict, doc: dict, progress=None
              ) -> tuple[dict[int, dict[int, float]], str]:
    """{target_gw: {player_id: ep}} as the model held it before `gw`'s deadline."""
    from . import modelrecord
    gws = _horizon(conn, season, gw)
    deadline = float(event.get("deadline_time_epoch") or 0)
    snap = _snapshot_covering(gws, deadline)
    if snap is not None:
        return snap, "live"
    from fpl_engine import lineup_feed as lf
    from fpl_engine.xpts import engine
    as_of = engine.first_kickoff(conn, season, gw)
    avail = lf.availability_at(conn, season, event.get("deadline_time") or "")
    cache = doc.setdefault("replay_cache", {})
    out: dict[int, dict[int, float]] = {}
    for t in gws:
        key = f"{gw}:{t}"
        if key not in cache:
            if progress:
                progress(f"Replaying the model's GW{t} projection as it stood before the GW{gw} deadline…")
            pred = engine.xpts_predict_gw(
                conn, season, t, as_of=as_of, use_availability=False,
                # later weeks' odds were published after this deadline
                odds_weight=None if t == gw else 0.0)
            cache[key] = {} if pred.empty else {
                str(int(p)): round(float(v) * (avail.get(int(p), 1.0) if avail else 1.0), 3)
                for p, v in zip(pred["player_id"], pred["prediction"]) if v > 0.005}
        out[t] = {int(p): v for p, v in cache[key].items()}
    return out, "replay"


def _snapshot_covering(gws: list[int], deadline: float) -> dict | None:
    from . import services
    snaps = [s for s in services._load_history()
             if s.get("built_at", 0) < deadline and str(gws[0]) in (s.get("gws") or {})]
    if not snaps:
        return None
    last = max(snaps, key=lambda s: s["built_at"])
    return {t: {int(p): float(v) for p, v in (last["gws"].get(str(t)) or {}).items()}
            for t in gws if str(t) in last["gws"]}


def _frame(meta: dict, prices_m: dict, eps: dict[int, dict[int, float]]) -> pd.DataFrame:
    gws = sorted(eps)
    rows = []
    for pid, m in meta.items():
        price = prices_m.get(pid)
        if not price:
            continue
        row = {"player_id": pid, "player": m["name"], "position": m["pos"],
               "team_id": m["team_id"], "team": None, "price": price, "available": 1.0}
        tot = 0.0
        for i, g in enumerate(gws):
            v = eps[g].get(pid, 0.0)
            row[f"ep_gw{g}"] = v
            tot += (SOLVE["decay"] ** i) * v
        row["ep_total"] = tot
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- decision --
def _decide(conn, season: str, gw: int, eps: dict, state: dict | None,
            prices_m: dict, meta: dict) -> dict:
    """One gameweek's decision from `state`. Returns the decision and the
    state it leaves for the next week."""
    from fpl_engine import manager
    from fpl_engine.optimise import chips, project
    gws = sorted(eps)
    frame = _frame(meta, prices_m, eps)
    tenths = {p: int(round(v * 10)) for p, v in prices_m.items()}
    if state is None:
        initial, bank, ft = None, BUDGET, 0
    else:
        squad = {int(p): int(v) for p, v in state["squad"].items()}
        # an owned player with no price this week (left the league) is still
        # sellable: fall back to what was paid rather than dropping him
        prices_m = {**{p: v / 10.0 for p, v in squad.items()}, **prices_m}
        frame = _frame(meta, prices_m, eps)
        tenths = {p: int(round(v * 10)) for p, v in prices_m.items()}
        initial = {p: manager.selling_price(paid, tenths.get(p, paid)) / 10.0
                   for p, paid in squad.items()}
        bank, ft = float(state["bank"]), int(state["ft"])
    pruned = project.prune(frame, keep_per_position=30, must_keep=set(initial or {}))
    plans = chips.optimise_with_chips(
        pruned, gws, initial=initial, bank=bank, free_transfers=max(ft, 0) if initial else 1,
        budget=BUDGET, chip_gws={}, n_plans=1, **SOLVE)
    if not plans:
        raise RuntimeError(f"no feasible plan for GW{gw}")
    first = plans[0].per_gw[0]

    build = initial is None
    ins = [t["player_id"] for t in first.get("transfers_in") or []]
    outs = [t["player_id"] for t in first.get("transfers_out") or []]
    used = int(first.get("free_used") or 0)
    hits = 0 if build else int(first.get("hits") or 0)
    if build:
        new_squad = {r["player_id"]: tenths[r["player_id"]] for r in first["squad"]}
        ft_next = 1                             # FPL: one free transfer for GW2
    else:
        new_squad = {p: v for p, v in squad.items() if p not in outs}
        for p in ins:
            new_squad[p] = tenths[p]
        ft_next = min(MAX_FT, ft - used + 1)
    squad_rows = first["squad"]
    xi = [r for r in squad_rows if r["in_xi"]]
    bench = sorted((r for r in squad_rows if not r["in_xi"]),
                   key=lambda r: (r["position"] != "GK", -r["ep"]))
    cap = next(r for r in xi if r["is_captain"])
    vice = next((r for r in xi if r["is_vice"]), None)

    def brief(r):
        return {"player_id": r["player_id"], "name": meta.get(r["player_id"], {}).get("name") or r["name"],
                "pos": r["position"], "team_id": r["team_id"], "price": r["price"],
                "ep": round(float(r["ep"]), 2), "captain": r["is_captain"], "vice": r["is_vice"]}

    def moved(pid, sell=False):
        m = meta.get(pid, {})
        price = (initial or {}).get(pid) if sell else prices_m.get(pid)
        return {"player_id": pid, "name": m.get("name"), "pos": m.get("pos"),
                "team_id": m.get("team_id"), "price": round(float(price or 0), 1),
                "ep_horizon": round(sum(eps[g].get(pid, 0.0) for g in gws), 1)}

    return {
        "decision": {
            "gw": gw, "build": build, "ft_before": 0 if build else ft,
            "transfers": [] if build else [
                {"out": moved(o, sell=True), "in": moved(i)} for o, i in zip(outs, ins)],
            "free_used": 0 if build else used, "hits": hits,
            "xi": [brief(r) for r in xi], "bench": [brief(r) for r in bench],
            "captain": cap["player_id"], "vice": vice["player_id"] if vice else None,
            "projected": round(sum(float(r["ep"]) for r in xi) + float(cap["ep"]) - 4 * hits, 1),
            "bank": round(float(first.get("bank") or 0.0), 1),
            "horizon": gws,
        },
        "state": {"squad": {str(p): v for p, v in new_squad.items()},
                  "bank": round(float(first.get("bank") or 0.0), 1), "ft": ft_next},
    }


def _score(decision: dict, actual: pd.DataFrame) -> dict:
    from fpl_engine.xpts.rank_utility import autosub_points
    a = actual.set_index("player_id") if not actual.empty else pd.DataFrame()
    squad = decision["xi"] + decision["bench"]
    pts = np.array([[float(a.loc[p["player_id"], "pts"]) if p["player_id"] in a.index else 0.0
                     for p in squad]])
    mins = np.array([[float(a.loc[p["player_id"], "mins"]) if p["player_id"] in a.index else 0.0
                      for p in squad]])
    idx = {p["player_id"]: k for k, p in enumerate(squad)}
    n_xi = len(decision["xi"])
    vice = decision["vice"] if decision["vice"] is not None else decision["xi"][0]["player_id"]
    total = float(autosub_points(
        pts, mins > 0, [p["pos"] for p in squad], list(range(n_xi)),
        idx[decision["captain"]], idx[vice], list(range(n_xi, len(squad))))[0])
    for k, p in enumerate(squad):
        p["pts"] = int(pts[0, k])
        p["mins"] = int(mins[0, k])
    return {"points": round(total, 1), "hits_cost": 4 * decision["hits"],
            "net": round(total - 4 * decision["hits"], 1)}


# ---------------------------------------------------------------- season --
def build(season: str | None = None, *, force: bool = False, progress=None) -> dict:
    """Extend the season through every finished gameweek, then refresh the
    plan for the next deadline. Idempotent; saves after every week."""
    from . import modelrecord, services
    season = season or config.CURRENT_SEASON
    doc = load(season)
    if force:
        doc = {"season": season, "version": TEAM_VERSION, "weeks": {}, "state": None,
               "replay_cache": doc.get("replay_cache", {}), "next": None}
    events = modelrecord._events()
    finished = sorted(g for g, e in events.items() if e.get("finished"))
    built = []
    conn = db.connect(config.DB_PATH)
    try:
        meta = modelrecord._meta(conn, season)
        thresholds = modelrecord._defcon_thresholds()
        state = doc.get("state")
        for gw in finished:
            if str(gw) in doc["weeks"]:
                continue
            # the season is a chain: a week can only follow the one before it
            if doc["weeks"] and str(gw - 1) not in doc["weeks"]:
                break
            eps, source = _believed(conn, season, gw, events[gw], doc, progress)
            if not eps or not eps.get(gw):
                break
            prices = modelrecord._prices(conn, season, gw)
            if progress:
                progress(f"Model's team: deciding GW{gw} ({source})…")
            step = _decide(conn, season, gw, eps, state, prices, meta)
            week = {**step["decision"], "source": source,
                    **_score(step["decision"], modelrecord._actuals(conn, season, gw, thresholds)),
                    "average": events[gw].get("average_entry_score"),
                    "highest": events[gw].get("highest_score")}
            doc["weeks"][str(gw)] = week
            state = doc["state"] = step["state"]
            built.append(gw)
            _save(doc)

        # the plan for the next deadline, from what the site is showing now
        nxt = next((g for g in sorted(events) if not events[g].get("finished")
                    and float(events[g].get("deadline_time_epoch") or 0) > time.time()), None)
        if nxt is not None and state is not None and (not finished or str(max(finished)) in doc["weeks"]):
            cache = services._load_proj_cache()
            gws = [g for g in _horizon(conn, season, nxt) if str(g) in (cache.get("gws") or {})]
            if gws:
                eps = {g: {int(r["player_id"]): float((r.get("ep") or {}).get(str(g)) or 0.0)
                           for r in cache["players"].values()} for g in gws}
                prices = {int(r["player_id"]): float(r["price"]) for r in cache["players"].values()
                          if r.get("price")}
                step = _decide(conn, season, nxt, eps, state, prices, meta)
                doc["next"] = {**step["decision"], "source": "current",
                               "built_at": cache.get("updated_at")}
                _save(doc)
    finally:
        conn.close()
    return {"built": built, "weeks": len(doc["weeks"]), "next": (doc.get("next") or {}).get("gw")}


def summary(season: str | None = None) -> dict:
    """What the Model tab needs: the weeks (no replay cache) and the totals."""
    doc = load(season)
    weeks = [doc["weeks"][k] for k in sorted(doc["weeks"], key=int)]
    net = sum(w["net"] for w in weeks)
    avg = sum((w.get("average") or 0) for w in weeks)
    return {"weeks": weeks, "next": doc.get("next"),
            "totals": {"net": round(net, 1), "average": avg,
                       "transfers": sum(len(w["transfers"]) for w in weeks),
                       "hits": sum(w["hits"] for w in weeks),
                       "weeks_above_average": sum(1 for w in weeks if w.get("average") is not None
                                                  and w["net"] > w["average"])},
            "state": doc.get("state")}
