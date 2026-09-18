"""The Live desk: everything that moves between now and the deadline.

Unlike the admin deadline desk (``app/deadline.py``), which is the operator's
view of the machinery, this is for everyone and it exists for one window:

    24 h before a deadline  ->  the tab appears and updates
    the deadline passes     ->  it stays 6 h more, marked over
    otherwise               ->  it is not there at all

That schedule is the whole design. A permanent "deadline" tab is furniture
nobody reads; a tab that shows up when there is something to do is a signal.
Everything on it is already archived by the pipeline — team news, Friday's
manager quotes, the three predicted-lineup feeds, price pressure and the
model's own biggest changes of mind — and none of it changes a projection.
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone

import pandas as pd

from fpl_engine import config, db

from . import deadline as dd
from . import services

_cache: dict = {"t": 0.0, "v": None}
TTL = 60.0

OPEN_BEFORE = 24 * 3600.0     # the tab appears this long before a deadline
STAY_AFTER = 6 * 3600.0       # ...and lingers this long after it, marked over


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def window(deadlines: dict, now: float | None = None) -> dict:
    """Which gameweek the live desk is about, and what phase it is in.

    ``deadlines`` is ``status.deadlines`` — gameweek -> ISO timestamp. Phases:
    ``open`` (inside the 24 h run-up), ``closed`` (the deadline has gone but
    within 6 h, so the gameweek is being played), ``idle`` (no tab).
    """
    now = time.time() if now is None else now
    stamped = sorted((int(g), t) for g, t in
                     ((g, _ts(v)) for g, v in (deadlines or {}).items())
                     if t is not None)
    nxt = next(((g, t) for g, t in stamped if t > now), None)
    prev = next(((g, t) for g, t in reversed(stamped) if t <= now), None)
    if prev and now - prev[1] <= STAY_AFTER:
        return {"phase": "closed", "gw": prev[0], "deadline": prev[1],
                "seconds": now - prev[1]}
    if nxt and nxt[1] - now <= OPEN_BEFORE:
        return {"phase": "open", "gw": nxt[0], "deadline": nxt[1],
                "seconds": nxt[1] - now}
    return {"phase": "idle", "gw": nxt[0] if nxt else None,
            "deadline": nxt[1] if nxt else None,
            "seconds": (nxt[1] - now) if nxt else None}


def _fixtures(conn, season: str, gw: int, tm: dict) -> list[dict]:
    rows = conn.execute(
        "SELECT fixture_id, kickoff_utc, team_h, team_a, team_h_score, "
        "team_a_score, finished FROM fixture WHERE season=? AND gw=? "
        "ORDER BY kickoff_utc", (season, gw)).fetchall()
    out = []
    for r in rows:
        out.append({
            "fixture_id": int(r["fixture_id"]),
            "kickoff": r["kickoff_utc"],
            "home": tm.get(int(r["team_h"])), "away": tm.get(int(r["team_a"])),
            "home_id": int(r["team_h"]), "away_id": int(r["team_a"]),
            "score": (None if r["team_h_score"] is None else
                      [int(r["team_h_score"]), int(r["team_a_score"] or 0)]),
            "finished": bool(r["finished"]),
        })
    return out


# ---------------------------------------------------------------- picks --
#
# Two sections replace the old "top 12 by projected points" board, which
# answered "who scores most" and nothing about WHY: a striker against the worst
# defence and a centre-back who crosses the DefCon threshold every week sat in
# one list with no way to tell them apart.

POINTS_PER_DEFCON = 2       # scoring_rules: defensive_contribution.points
LEAGUE_GOALS_PER_TEAM = 1.4  # fallback goals-against for an unpriced fixture


def _tag(pos: str, c: dict) -> str:
    """What kind of pick a projection is, from the engine's own components:
    an attacking return, a DefCon crossing, a clean sheet, or (keepers) saves."""
    att = c.get("goals", 0) + c.get("assists", 0)
    options = {"ATT": att, "DEFCON": c.get("defcon", 0), "CS": c.get("cs", 0)}
    if pos == "GK":
        options = {"CS": c.get("cs", 0), "SAVES": c.get("saves", 0)}
    return max(options, key=options.get)


def _pool(gw: int, pl: dict) -> list[dict]:
    """Every player with a projection AND a component breakdown for `gw`.
    A cache built before the breakdown existed yields an empty pool, which the
    page reports rather than guessing at a split."""
    cache = services._load_proj_cache()
    try:
        extra = {p["id"]: p for p in services.players_payload().get("players", [])}
    except Exception:  # noqa: BLE001 - the live bootstrap is a nicety here
        extra = {}
    out = []
    for rec in cache.get("players", {}).values():
        ep = (rec.get("ep") or {}).get(str(gw))
        comp = (rec.get("comp") or {}).get(str(gw))
        if ep is None or not comp or rec.get("position") not in ("GK", "DEF", "MID", "FWD"):
            continue
        pid = int(rec["player_id"])
        x = extra.get(pid, {})
        out.append({
            "player_id": pid, "name": (pl.get(pid) or {}).get("name") or rec.get("player"),
            "team_id": int(rec["team_id"]), "pos": rec["position"],
            "price": rec.get("price"), "ep": round(float(ep), 2),
            "xmins": (rec.get("xm") or {}).get(str(gw)),
            # first-choice taker only: _pk_share gives second and third choices
            # their (small) share too, and flagging them all said nothing
            "own": x.get("own"), "pk": (x.get("pk_share") or 0) >= 0.5,
            "status": x.get("status"), "chance": x.get("chance"),
            "tag": _tag(rec["position"], comp),
            "att": round(comp.get("goals", 0) + comp.get("assists", 0), 2),
            "xgi": round(comp.get("eg", 0) + comp.get("ea", 0), 2),
            "eg": comp.get("eg", 0), "ea": comp.get("ea", 0),
            "defcon": comp.get("defcon", 0),
            # E[DefCon points] / 2 is the chance he crosses the threshold in a
            # single fixture; capped because a double gameweek can exceed one
            "p_defcon": round(min(1.0, comp.get("defcon", 0) / POINTS_PER_DEFCON), 2),
            "cs_pts": comp.get("cs", 0), "p_cs": comp.get("pcs", 0),
            "bonus": comp.get("bonus", 0), "saves": comp.get("saves", 0),
        })
    return out


def best_xi(pool: list[dict]) -> dict | None:
    """The best legal XI the model can see for the gameweek.

    Legal the way FPL means it: one keeper, 3-5 defenders, 2-5 midfielders,
    1-3 forwards, at most three from a club, and the captain counted twice.
    Solved exactly (it is 600-odd binaries) rather than greedily — a greedy
    pick takes the fourth-best Man City player and then cannot fit the fifth.
    No budget: this is "who would the model start", not a squad it can afford;
    the Planner and Solver are where money is.
    """
    import pulp
    ps = [p for p in pool if p["ep"] > 0]
    if len(ps) < 11:
        return None
    prob = pulp.LpProblem("best_xi", pulp.LpMaximize)
    x = {p["player_id"]: pulp.LpVariable(f"x{p['player_id']}", cat="Binary") for p in ps}
    c = {p["player_id"]: pulp.LpVariable(f"c{p['player_id']}", cat="Binary") for p in ps}
    prob += pulp.lpSum(p["ep"] * (x[p["player_id"]] + c[p["player_id"]]) for p in ps)
    prob += pulp.lpSum(x.values()) == 11
    prob += pulp.lpSum(c.values()) == 1
    for p in ps:
        prob += c[p["player_id"]] <= x[p["player_id"]]
    for pos, lo, hi in (("GK", 1, 1), ("DEF", 3, 5), ("MID", 2, 5), ("FWD", 1, 3)):
        n = pulp.lpSum(x[p["player_id"]] for p in ps if p["pos"] == pos)
        prob += n >= lo
        prob += n <= hi
    for club in {p["team_id"] for p in ps}:
        prob += pulp.lpSum(x[p["player_id"]] for p in ps if p["team_id"] == club) <= 3
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    xi = [p for p in ps if x[p["player_id"]].value() > 0.5]
    cap = next(p for p in ps if c[p["player_id"]].value() > 0.5)
    vice = max((p for p in xi if p is not cap), key=lambda p: p["ep"])
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    xi.sort(key=lambda p: (order[p["pos"]], -p["ep"]))
    rows = [[{**p, "captain": p is cap, "vice": p is vice} for p in xi if p["pos"] == pos]
            for pos in ("GK", "DEF", "MID", "FWD")]
    return {"rows": rows,
            "formation": "-".join(str(len(r)) for r in rows[1:]),
            "points": round(sum(p["ep"] for p in xi) + cap["ep"], 1),
            "cost": round(sum(p["price"] or 0 for p in xi), 1),
            "captain": cap["name"]}


def fixture_picks(gw: int, pool: list[dict], limit_teams: int = 8) -> list[dict]:
    """Clubs ranked by how favourable their fixture is, and who to own there.

    Favourability is the MARKET's read of the fixture — expected goals for and
    against, from the bookmaker (or prediction-market) price the model already
    blends in — ranked by expected goal difference. Inside each club, attacking
    picks come first, then DefCon and clean-sheet picks: an attacking return is
    the bigger swing, and the owner asked for them on top.
    """
    try:
        grid = services.fixtures_payload().get("grid", {})
    except Exception:  # noqa: BLE001
        grid = {}
    by_team: dict[int, list] = {}
    for p in pool:
        by_team.setdefault(p["team_id"], []).append(p)
    clubs = []
    for tid, players in by_team.items():
        cells = grid.get(str(tid), {}).get(str(gw)) or []
        if not cells:
            continue                          # a blank gameweek: nothing to pick
        xg = xga = cs = 0.0
        priced = True
        for f in cells:
            o = f.get("odds") or {}
            if o.get("xg") is not None:
                gf, ga = o["xg"], o["xg_against"]
            else:
                # no bookmaker price: the club's own expected goals from the
                # engine, and a league-average goals-against
                priced = False
                gf = sum(q["eg"] for q in players) / max(1, len(cells))
                ga = LEAGUE_GOALS_PER_TEAM
            xg += gf
            xga += ga
            cs += math.exp(-ga)               # P(clean sheet) per fixture
        start = [q for q in players if (q["xmins"] or 0) >= 45 and q["ep"] >= 2.0]
        attack = sorted((q for q in start if q["tag"] == "ATT"), key=lambda q: -q["ep"])[:3]
        defence = sorted((q for q in start if q["tag"] in ("DEFCON", "CS") and q["pos"] != "GK"),
                         key=lambda q: -q["ep"])[:2]
        keeper = max((q for q in start if q["pos"] == "GK"), key=lambda q: q["ep"], default=None)
        clubs.append({
            "team_id": tid,
            "fixtures": [{"opp": f.get("opp"), "home": f.get("home")} for f in cells],
            "xg": round(xg, 2), "xga": round(xga, 2),
            "p_cs": round(min(1.0, cs), 2), "priced": priced,
            "attack": attack, "defence": defence, "keeper": keeper,
        })
    clubs.sort(key=lambda c: -(c["xg"] - c["xga"]))
    return clubs[:limit_teams]


def _price_pressure(limit: int = 8) -> dict:
    """Who is about to move before the deadline, at the price the Round-7
    conversion puts on it — a tie-breaker, said out loud."""
    try:
        p = services.prices_payload(limit=limit)
    except Exception as exc:            # noqa: BLE001
        return {"ok": False, "error": str(exc), "risers": [], "fallers": []}
    return p


def payload(force: bool = False) -> dict:
    now = time.time()
    if not force and _cache["v"] is not None and now - _cache["t"] < TTL:
        return _cache["v"]
    season = config.CURRENT_SEASON
    status = services.status_payload()
    win = window(status.get("deadlines") or {}, now)
    gw = win["gw"]
    if gw is None:
        out = {"season": season, "window": win, "built_at": now}
        _cache["t"], _cache["v"] = now, out
        return out
    conn = db.connect(config.DB_PATH)
    try:
        pl, tm = dd._names(conn, season)
        start_p = services._model_start_probs(season)
        pool = _pool(gw, pl)
        out = {
            "season": season,
            "window": win,
            "gw": gw,
            "fixtures": _fixtures(conn, season, gw, tm),
            # FPL's own change log: the only team news that is both free and
            # timestamped by the source, and the one signal the live model
            # actually consumes
            "news": dd._news(conn, season, pl, tm, days=7.0)[:40],
            "pressers": dd._pressers(conn, season, gw, pl, tm)[:40],
            "quotes": dd._presser_quotes(conn, season, gw, pl, tm),
            "lineups": dd._lineups(conn, season, gw, pl, tm, start_p),
            "movers": dd._movers(gw, pl, tm),
            "best_xi": best_xi(pool),
            "fixture_picks": fixture_picks(gw, pool),
            "components": bool(pool),
            "prices": _price_pressure(),
            "proj_updated_at": status.get("proj_updated_at"),
            "built_at": now,
        }
    finally:
        conn.close()
    _cache["t"], _cache["v"] = now, out
    return out
