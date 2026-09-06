"""Fetch a manager's FPL entry (squad ID) and derive current squad state.

Reads the free, public entry endpoints:
  * ``entry/{id}/``               basic info + current event
  * ``entry/{id}/history/``       per-gameweek history + chips
  * ``entry/{id}/event/{gw}/picks/``  the 15 picks
  * ``entry/{id}/transfers/``     every transfer made (with the price paid)
  * ``bootstrap-static/``         current + starting price of every player

From these we derive: the current 15-man squad (with per-player selling
prices), money in the bank, and an estimate of free transfers available for the
next gameweek. When the entry has no squad yet (pre-season, before the first
deadline), :func:`current_squad` returns ``None`` so the optimiser builds a
fresh squad from the budget instead.

**Selling prices.** The public picks endpoint does *not* carry
``selling_price``/``purchase_price`` — only the authenticated ``my-team/{id}/``
endpoint does. Treating the absent field as £0.0 silently tells the optimiser
that every sale raises nothing, which makes all transfers unaffordable and
yields a "do nothing" plan. We therefore reconstruct the true selling price
from public data instead: the price paid comes from ``entry/{id}/transfers/``
(or, for a player never transferred in, his season-start price
``now_cost - cost_change_start``), and FPL's sell rule is applied on top.
"""
from __future__ import annotations

import json

from .http import get_text

BASE = "https://fantasy.premierleague.com/api"
DEFAULT_ENTRY = 883566  # https://fantasy.premierleague.com/en/entry/883566/history/
MAX_FREE_TRANSFERS = 5


def _get(path: str, use_cache: bool = False) -> dict | list:
    return json.loads(get_text(f"{BASE}/{path}", use_cache=use_cache))


def fetch_entry(entry_id: int, use_cache: bool = False) -> dict:
    return _get(f"entry/{entry_id}/", use_cache=use_cache)


def fetch_history(entry_id: int, use_cache: bool = False) -> dict:
    return _get(f"entry/{entry_id}/history/", use_cache=use_cache)


def fetch_transfers(entry_id: int, use_cache: bool = False) -> list:
    """Every transfer the entry has made (public endpoint). [] if none."""
    try:
        return _get(f"entry/{entry_id}/transfers/", use_cache=use_cache) or []
    except Exception:
        return []


def fetch_bootstrap(use_cache: bool = True) -> dict:
    return _get("bootstrap-static/", use_cache=use_cache)


def price_tables(use_cache: bool = True) -> tuple[dict[int, int], dict[int, int]]:
    """(now, start) price per element id, both in tenths of £m.

    ``cost_change_start`` is the change since the season began, so
    ``now_cost - cost_change_start`` is exactly what the player cost at the
    start — the purchase price of anyone still in his owner's original squad.
    """
    boot = fetch_bootstrap(use_cache=use_cache)
    now: dict[int, int] = {}
    start: dict[int, int] = {}
    for e in boot.get("elements", []) or []:
        eid = int(e["id"])
        nc = int(e.get("now_cost") or 0)
        now[eid] = nc
        start[eid] = nc - int(e.get("cost_change_start") or 0)
    return now, start


def selling_price(purchase: int, now: int) -> int:
    """FPL's sell rule, in tenths of £m.

    You get back what you paid plus half of any profit, rounded down to the
    nearest £0.1m; if the price has fallen you take the full loss.
    """
    if now <= purchase:
        return now
    return purchase + (now - purchase) // 2


def reconstruct_prices(entry_id: int, elements: list[int], *,
                       use_cache: bool = False) -> dict[int, dict[str, float]]:
    """{element: {purchase_price, selling_price}} in £m, from public data only.

    Raises if the bootstrap prices cannot be read — a silent £0.0 would make
    every transfer look unaffordable to the optimiser (CLAUDE.md principle #5).
    """
    now, start = price_tables()
    if not now:
        raise RuntimeError("bootstrap prices unavailable — cannot reconstruct "
                           "selling prices")
    # last transfer-in wins (a player can be sold and bought back)
    paid: dict[int, int] = {}
    for t in sorted(fetch_transfers(entry_id, use_cache=use_cache),
                    key=lambda t: (t.get("event") or 0, t.get("time") or "")):
        if t.get("element_in") is not None and t.get("element_in_cost"):
            paid[int(t["element_in"])] = int(t["element_in_cost"])
    out: dict[int, dict[str, float]] = {}
    for eid in elements:
        eid = int(eid)
        n = now.get(eid)
        if n is None:
            continue                      # unknown element: caller guards
        purchase = paid.get(eid, start.get(eid, n))
        out[eid] = {"purchase_price": purchase / 10.0,
                    "selling_price": selling_price(purchase, n) / 10.0}
    return out


def fetch_picks(entry_id: int, gw: int, use_cache: bool = False) -> dict | None:
    try:
        return _get(f"entry/{entry_id}/event/{gw}/picks/", use_cache=use_cache)
    except Exception:
        return None  # no picks for that gw (e.g. before the first deadline)


def estimate_free_transfers(history: dict) -> int:
    """Estimate FTs available for the *next* gameweek from transfer history.

    Rule (2026-27): each gameweek grants +1 free transfer, bankable up to 5,
    minus the transfers actually made that week (extras were paid hits).

    The stock starts at **0**, not 1: transfers before the GW1 deadline are
    unlimited and free, and nothing banks out of them — the first free
    transfer is the one granted *after* GW1 completes. Seeding at 1 would
    hand out a phantom extra FT for the whole season (after a quiet GW1 it
    reports 2 when the true answer is 1), which makes the optimiser plan
    -4 hits believing they are free.
    """
    events = history.get("current", []) or []
    ft = 0
    for ev in events:
        made = ev.get("event_transfers", 0) or 0
        ft = min(MAX_FREE_TRANSFERS, max(0, ft - made) + 1)
    return max(1, ft)


def current_squad(entry_id: int, *, use_cache: bool = False) -> dict | None:
    """Return the manager's current squad state, or None if none exists yet.

    Returns dict with: entry_id, name, gw (the gw the picks are from),
    bank (£m), squad (list of {element, selling_price, purchase_price,
    is_captain, is_vice, multiplier}), free_transfers.
    """
    entry = fetch_entry(entry_id, use_cache=use_cache)
    history = fetch_history(entry_id, use_cache=use_cache)
    events = history.get("current", []) or []
    if not events:
        return None  # pre-season / no gameweek played yet

    last_gw = events[-1]["event"]
    picks = fetch_picks(entry_id, last_gw, use_cache=use_cache)
    if not picks or "picks" not in picks:
        return None

    et = picks.get("entry_history", {})
    squad = [{
        "element": p["element"],
        "selling_price": p.get("selling_price", 0) / 10.0,
        "purchase_price": p.get("purchase_price", 0) / 10.0,
        "is_captain": bool(p.get("is_captain")),
        "is_vice": bool(p.get("is_vice_captain")),
        "multiplier": p.get("multiplier", 1),
    } for p in picks["picks"]]

    # Public picks carry no prices; reconstruct them rather than leaving £0.0.
    if any(not p["selling_price"] for p in squad):
        rec = reconstruct_prices(entry_id, [p["element"] for p in squad],
                                 use_cache=use_cache)
        for p in squad:
            r = rec.get(p["element"])
            if r is None:
                raise RuntimeError(
                    f"no price for element {p['element']} — refusing to hand "
                    "the optimiser a £0.0 selling price")
            p["selling_price"] = p["selling_price"] or r["selling_price"]
            p["purchase_price"] = p["purchase_price"] or r["purchase_price"]

    return {
        "entry_id": entry_id,
        "name": entry.get("name"),
        "gw": last_gw,
        "bank": et.get("bank", 0) / 10.0,
        "squad": squad,
        "free_transfers": estimate_free_transfers(history),
    }


# --------------------------------------------------------------------------
# chips
#
# FPL names chips differently from the optimiser ("bboost" vs "bench_boost"),
# and since 2024-25 each chip is granted TWICE — once per half of the season,
# with the windows published in the bootstrap's own ``chips`` table. Reading
# that table rather than hardcoding the halves means a rule change (or a
# one-off extra chip) is picked up for free; the fallback below only covers a
# bootstrap that has no table at all.
#
# What the public API can and cannot see is the whole design constraint here:
# ``entry/{id}/history/`` lists chips whose gameweek has PASSED, so a chip
# activated for the upcoming deadline is invisible to it. That state lives
# only in the authenticated ``my-team/{id}/`` response, which is why
# :func:`chip_state` takes an optional my-team document and prefers it.
# --------------------------------------------------------------------------

FPL_CHIP_NAMES = {"wildcard": "wildcard", "freehit": "freehit",
                  "bboost": "bench_boost", "3xc": "triple_captain"}
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit",
               "bench_boost": "Bench Boost", "triple_captain": "Triple Captain"}
HALVES = ((1, 19), (20, 38))


def _default_windows() -> list[dict]:
    return [{"chip": c, "start": a, "stop": b}
            for a, b in HALVES for c in CHIP_LABELS]


def chip_windows(boot: dict | None = None) -> list[dict]:
    """Every chip slot this season offers: {chip, start, stop}, engine names.

    Unknown chips (2024-25's assistant manager, anything FPL adds later) are
    dropped rather than guessed at — the optimiser can only model the four it
    implements, and silently mapping a fifth onto one of them would be worse
    than not seeing it.
    """
    try:
        boot = fetch_bootstrap() if boot is None else boot
    except Exception:
        return _default_windows()
    out = []
    for c in (boot.get("chips") or []):
        chip = FPL_CHIP_NAMES.get(c.get("name"))
        if not chip:
            continue
        out.append({"chip": chip,
                    "start": int(c.get("start_event") or 1),
                    "stop": int(c.get("stop_event") or 38)})
    return out or _default_windows()


def _next_gw_from(boot: dict) -> int | None:
    for e in boot.get("events", []) or []:
        if e.get("is_next"):
            return int(e["id"])
    for e in boot.get("events", []) or []:
        if not e.get("finished"):
            return int(e["id"])
    return None


def my_team_chips(my_team: dict | None) -> list[dict]:
    """Normalise the ``chips`` block of a my-team response.

    Shape per entry: ``{"chip": ..., "status": available|active|played,
    "start": gw, "stop": gw, "played_gw": gw|None}``. This is the only source
    that knows a chip is ACTIVE for the upcoming deadline.
    """
    out = []
    for c in ((my_team or {}).get("chips") or []):
        chip = FPL_CHIP_NAMES.get(c.get("name"))
        if not chip:
            continue
        played = [int(g) for g in (c.get("played_by_entry") or [])]
        status = str(c.get("status_for_entry") or
                     ("played" if played else "available"))
        out.append({"chip": chip, "status": status,
                    "start": int(c.get("start_event") or 1),
                    "stop": int(c.get("stop_event") or 38),
                    "played_gw": played[0] if played else None})
    return out


def chip_state(entry_id: int | None = None, *, history: dict | None = None,
               my_team_chips_: list[dict] | None = None,
               boot: dict | None = None, next_gw: int | None = None,
               use_cache: bool = False) -> dict:
    """Which chips this entry has spent, holds, and has active right now.

    ``history`` (public) supplies chips played in gameweeks that have already
    passed; ``my_team_chips_`` (from an authenticated import) supplies the
    active one and confirms the rest. A played chip is charged to the first
    unused window containing its gameweek, which is what makes "wildcard used
    in GW3" leave the GW20-38 wildcard still available.
    """
    boot = boot if boot is not None else fetch_bootstrap()
    if next_gw is None:
        next_gw = _next_gw_from(boot)
    if history is None and entry_id is not None:
        try:
            history = fetch_history(entry_id, use_cache=use_cache)
        except Exception:
            history = None

    played: list[dict] = []
    for c in ((history or {}).get("chips") or []):
        chip = FPL_CHIP_NAMES.get(c.get("name"))
        if chip and c.get("event") is not None:
            played.append({"chip": chip, "gw": int(c["event"])})
    mine = my_team_chips_ or []
    for c in mine:                      # my-team knows chips history misses
        if c["status"] == "played" and c.get("played_gw") is not None:
            if not any(p["chip"] == c["chip"] and p["gw"] == c["played_gw"]
                       for p in played):
                played.append({"chip": c["chip"], "gw": int(c["played_gw"])})
    played.sort(key=lambda p: p["gw"])
    active = next((c["chip"] for c in mine if c["status"] == "active"), None)

    windows = [dict(w, used_gw=None) for w in chip_windows(boot)]
    for p in played:
        slot = next((w for w in windows
                     if w["chip"] == p["chip"] and w["used_gw"] is None
                     and w["start"] <= p["gw"] <= w["stop"]), None)
        if slot is None:                # window table disagrees with reality
            slot = next((w for w in windows if w["chip"] == p["chip"]
                         and w["used_gw"] is None), None)
        if slot is not None:
            slot["used_gw"] = p["gw"]

    gw = next_gw or 1
    available = sorted({w["chip"] for w in windows
                        if w["used_gw"] is None and w["stop"] >= gw
                        and w["chip"] != active})
    return {"played": played, "active": active, "available": available,
            "windows": windows, "next_gw": next_gw,
            "source": "my-team" if mine else "public"}
