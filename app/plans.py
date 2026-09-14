"""Plans and entitlements — the seam for a free / paid split later.

Nothing is charged for today: `FPLABS_ENFORCE_PLANS` is off, so every
visitor gets the `pro` entitlements. What ships is the *mechanism*, so that
turning on a paid tier is a billing webhook that flips `user.plan` and one
environment variable, not a refactor:

* every account has a `plan` (`free` / `pro`), anonymous visitors are `free`
* `entitlements(plan)` is the single table of what each plan may do
* `check(entitlements, params)` clamps a solve request to the plan, and says
  which knobs it clamped so the UI can show an upgrade hint instead of a
  silently smaller answer

The split proposed in docs/MONETISATION.md: the *reading* surface (projections,
fixtures, prices, a 3-gameweek plan) stays free because it is what makes the
product discoverable; the *decision* surface that costs CPU or is genuinely
differentiated (long horizons, all three playstyles, chip planning, mini-league
analysis, the simulator, projection history) is the paid tier.
"""
from __future__ import annotations

import os

PLANS = ("free", "pro")

ENTITLEMENTS = {
    "free": {
        "max_horizon": 3,
        "playstyles": 1,
        "chips": False,
        "drafts": 2,
        "league": False,
        "history": False,
        "time_limit": 30,
    },
    "pro": {
        "max_horizon": 8,
        "playstyles": 3,
        "chips": True,
        "drafts": 50,
        "league": True,
        "history": True,
        "time_limit": 120,
    },
}


def enforced() -> bool:
    return os.environ.get("FPLABS_ENFORCE_PLANS", "0") == "1"


def plan_for(user: dict | None) -> str:
    if not enforced():
        return "pro"
    p = (user or {}).get("plan") or "free"
    return p if p in PLANS else "free"


def entitlements(plan: str) -> dict:
    return dict(ENTITLEMENTS.get(plan, ENTITLEMENTS["free"]))


def check(ent: dict, params: dict) -> tuple[dict, list[str]]:
    """Clamp solve params to the plan; return (params, clamped_knobs)."""
    p = dict(params)
    clamped: list[str] = []
    h = int(p.get("horizon") or 5)
    if h > ent["max_horizon"]:
        p["horizon"] = ent["max_horizon"]
        clamped.append("horizon")
    n = int(p.get("n_plans") or 1)
    if n > ent["playstyles"]:
        p["n_plans"] = ent["playstyles"]
        clamped.append("playstyles")
    if p.get("playstyles") and len(p["playstyles"]) > ent["playstyles"]:
        p["playstyles"] = list(p["playstyles"])[:ent["playstyles"]]
        clamped.append("playstyles")
    if not ent["chips"] and p.get("chips"):
        p["chips"] = {}
        clamped.append("chips")
    tl = int(p.get("time_limit") or 60)
    if tl > ent["time_limit"]:
        p["time_limit"] = ent["time_limit"]
        clamped.append("time_limit")
    return p, clamped
