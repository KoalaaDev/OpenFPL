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
