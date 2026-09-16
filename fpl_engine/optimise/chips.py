"""Chip-aware multi-period FPL optimiser (superset of :mod:`milp`).

Extends the base multi-period MILP with the four FPL chips, solver-style user
constraints and alternative-plan generation, for use by the web app:

* **Wildcard** — unlimited free transfers that gameweek (changes permanent).
* **Free Hit** — unlimited squad change for one gameweek, reverting after.
* **Bench Boost** — all 15 players score that gameweek.
* **Triple Captain** — captain scores 3x instead of 2x.
* Target/hold players, avoid/sell players, do-not-buy clubs, forced moves,
  minimum-banked-FT targets, a terminal value per banked free transfer, and
  N alternative plans via no-good cuts.

The base module's guarantees (budget recursion on the actual bank, FT accrual
+1/gw bankable to 5, hits net of -4, 2/5/5/3 squad, <=3 per club, legal XI)
all carry over. :mod:`milp` itself is left untouched so its tests and CLI
behaviour are unchanged.
"""
from __future__ import annotations

import math

import warnings
from dataclasses import dataclass, field

import pandas as pd
import pulp

from .milp import (MAX_FREE_TRANSFERS, MAX_PER_CLUB, POSITION_QUOTA, SQUAD_SIZE,
                   XI_MAX, XI_MIN, XI_SIZE)


# --- playstyles -------------------------------------------------------------
# Three strategies a manager actually chooses between, not three near-identical
# optima. Each varies only *preferences* (how far ahead to look, how much a
# banked transfer is worth, whether hits are acceptable at all) — never the
# rules: the -4 is always priced at -4, and "no hits" forbids them outright
# rather than pretending they are cheap.
PLAYSTYLES: dict[str, dict] = {
    "aggressive": {
        "label": "Win now",
        "note": "Chases the next gameweek or two and will pay a hit to do it.",
        "params": {"decay": 0.70, "ft_value": 0.5, "allow_hits": True,
                   "bench_weight": 0.05},
    },
    "balanced": {
        "label": "Balanced",
        "note": "Takes a hit only when it clearly pays; even weight across the horizon.",
        "params": {"decay": 0.85, "ft_value": 1.5, "allow_hits": True,
                   "bench_weight": 0.10},
    },
    "patient": {
        "label": "Patient",
        "note": "Never takes a hit; banks free transfers and plans the long game.",
        "params": {"decay": 0.95, "ft_value": 2.5, "allow_hits": False,
                   "bench_weight": 0.15},
    },
}
DEFAULT_PLAYSTYLES = ["aggressive", "balanced", "patient"]


def optimise_playstyles(proj, gws: list[int], *, styles: list[str] | None = None,
                        on_progress=None, **kw) -> list[ChipPlan]:
    """One plan per playstyle, so the user picks a strategy rather than a
    near-duplicate of the same optimum.

    Style presets override any matching keyword (that is the point of a
    style); everything else — squad, bank, chips, locks, club rules — is
    shared. Plans come back ordered by the requested styles, each tagged with
    ``style``/``style_label``/``style_note``. Compare them on ``total_ep``,
    never on ``objective`` (the styles weight gameweeks differently).
    """
    styles = [s for s in (styles or DEFAULT_PLAYSTYLES) if s in PLAYSTYLES]
    out: list[ChipPlan] = []
    for i, key in enumerate(styles):
        spec = PLAYSTYLES[key]
        if on_progress:
            on_progress(f"Solving {spec['label']} plan ({i + 1}/{len(styles)})…")
        params = {**kw, **spec["params"], "n_plans": 1}
        plans = optimise_with_chips(proj, gws, on_progress=None, **params)
        if not plans:
            continue
        plan = plans[0]
        plan.style = key
        plan.style_label = spec["label"]
        plan.style_note = spec["note"]
        out.append(plan)
    return out


def _solve(prob: pulp.LpProblem, time_limit: int) -> None:
    """CBC/HiGHS with a 1% optimality gap — plans converge far faster and a
    1% gap is far below projection noise."""
    for name in ("HiGHS_CMD", "PULP_CBC_CMD"):
        if name in pulp.listSolvers(onlyAvailable=True):
            solver = getattr(pulp, name)(msg=False, timeLimit=time_limit,
                                         gapRel=0.01)
            prob.solve(solver)
            return
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.01))

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pulp")

CHIPS = ("wildcard", "freehit", "bench_boost", "triple_captain")

# What a chip is worth if you *save* it, in points. A rolling-horizon optimiser
# sees a chip as free upside — it only ever adds points — so with no opportunity
# cost it burns every available chip inside the horizon (GW2-6 of a season, for
# single-digit gains). These are the option value of holding the chip back for
# the week it is actually meant for: a Triple Captain on a premium in a double
# gameweek, a Bench Boost with a full bench playing twice. A chip is played only
# when its gain here beats what it would be worth saved.
#
# No horizon optimiser can derive these: the reason to hold a chip is the double
# and blank gameweeks of GW25+, which sit outside a 5-week horizon and are not
# even scheduled yet. So this is the only lever, and it is a HEURISTIC — it
# encodes ordinary FPL practice (a well-timed TC/BB is worth ~15-25 pts), not a
# fitted season-long simulation. Raise them to be stricter early in a season,
# lower them (or set 0) once a chip genuinely has to be spent.
DEFAULT_CHIP_RESERVE: dict[str, float] = {
    "wildcard": 20.0,
    "freehit": 15.0,
    "bench_boost": 15.0,
    "triple_captain": 15.0,
}

# What a *saved* chip is worth is an option value: holding it pays
# E[max over the gameweeks left], while playing it now pays this week's value.
# The reserve is the premium between them, and it shrinks as the season runs
# out — a flat constant cannot express that, and 15.0 turned out to be roughly
# twice the truth for Triple Captain at a realistic horizon.
#
# Measured by simulating all 74 replayed gameweeks (xpts/simulate.py) and
# taking E[max] - max E over rolling windows:
#
#     gameweeks left        5      10      20
#     Triple Captain     5.40    7.00    8.06
#     Bench Boost        8.04   10.54   12.12
#
# The fitted curves below reproduce those to within 0.5 pts. The simulator's
# chip payoffs are themselves calibrated against what the chips actually paid
# (TC predicted 6.85 / realised 7.45; BB predicted 18.57 / realised 18.81),
# and E[max] tracks the realised best week (TC 13.7 vs 14.5 over 5 gameweeks).
#
# Wildcard and Free Hit have no equivalent measurement — their payoff is a
# whole-squad rebuild, not a one-week total — so they keep the flat heuristic.
MEASURED_RESERVE_CURVE: dict[str, tuple[float, float]] = {
    "triple_captain": (2.31, 1.92),      # premium ~ a + b * ln(gws remaining)
    "bench_boost": (3.31, 2.94),
}


def chip_reserve_for(chip: str, gws_remaining: int) -> float:
    """Points a chip is worth kept back, given how much season is left."""
    curve = MEASURED_RESERVE_CURVE.get(chip)
    if not curve or gws_remaining is None or gws_remaining < 1:
        return DEFAULT_CHIP_RESERVE.get(chip, 0.0)
    a, b = curve
    return max(0.0, a + b * math.log(max(1, int(gws_remaining))))


def default_reserve(gws_remaining: int | None = None) -> dict[str, float]:
    """The full reserve map, measured where a measurement exists."""
    if gws_remaining is None:
        return dict(DEFAULT_CHIP_RESERVE)
    return {c: chip_reserve_for(c, gws_remaining)
            for c in DEFAULT_CHIP_RESERVE}


@dataclass
class ChipPlan:
    gws: list[int]
    objective: float
    status: str = ""
    per_gw: list[dict] = field(default_factory=list)
    style: str = ""            # playstyle key, when produced by a preset
    style_label: str = ""      # human-readable name for the UI
    style_note: str = ""       # one line on what this style optimises for

    @property
    def total_ep(self) -> float:
        """Undecayed projected XI points over the horizon, net of hits.

        ``objective`` is NOT comparable between playstyles (each uses its own
        decay and free-transfer valuation); this is, so the UI ranks on it.
        """
        return round(sum(g.get("xi_points", 0.0) - 4.0 * (g.get("hits", 0) or 0)
                         for g in self.per_gw), 2)



def _check_selling_prices(initial: dict[int, float] | None) -> None:
    """Refuse a squad whose selling prices are missing.

    FPL's cheapest player costs £3.5m+, so a £0.0 selling price is never real
    — it means the price was absent from the source (the public picks endpoint
    carries none) and got defaulted to zero. Left alone it silently makes every
    sale raise nothing, so no transfer is affordable and the optimiser returns
    a "do nothing" plan that looks legitimate. Fail loud instead (CLAUDE.md #5).
    """
    if not initial:
        return
    bad = sorted(pid for pid, v in initial.items() if not v or float(v) <= 0.0)
    if bad:
        raise ValueError(
            f"{len(bad)} squad player(s) have a £0.00 selling price "
            f"(ids {bad[:5]}{'…' if len(bad) > 5 else ''}). The public picks "
            "endpoint does not expose selling prices — use "
            "fpl_engine.manager.reconstruct_prices() to derive them.")


def optimise_with_chips(
    proj: pd.DataFrame,
    gws: list[int],
    *,
    initial: dict[int, float] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
    budget: float = 100.0,
    decay: float = 0.85,
    chip_decay: float = 1.0,
    hit_cost: float = 4.0,
    bench_weight: float = 0.1,
    ft_value: float = 1.5,
    max_transfers_per_gw: int = 3,
    time_limit: int = 60,
    chip_gws: dict[str, list[int]] | None = None,
    chip_force: dict[str, int] | None = None,
    chip_reserve: dict[str, float] | None = None,
    gws_remaining: int | None = None,
    locked: set[int] | None = None,
    avoid: set[int] | None = None,
    banned_clubs: set[int] | None = None,
    sell_clubs: set[int] | None = None,
    allow_hits: bool = True,
    concentration: float = 0.0,
    forced_in: dict[int, list[int]] | None = None,
    forced_out: dict[int, list[int]] | None = None,
    min_ft: dict[int, int] | None = None,
    n_plans: int = 1,
    unlimited_first: bool = False,
    on_progress=None,
) -> list[ChipPlan]:
    """Solve the chip-aware plan; return up to ``n_plans`` alternatives.

    ``chip_gws`` maps chip name -> gameweeks it may be played in (absent or
    empty = chip unavailable). ``chip_force`` pins a chip to a gameweek.
    ``banned_clubs`` blocks *buying* from those clubs; ``sell_clubs`` also
    forces players already owned from those clubs out of the squad by the end
    of the horizon (the solver picks the cheapest gameweek to do it, so a
    "get off this club" instruction still uses free transfers where possible).
    ``chip_reserve`` maps chip -> the points it is worth if saved for a better
    gameweek beyond the horizon; a chip is played only when its gain here beats
    that (see ``DEFAULT_CHIP_RESERVE``). Without it the optimiser burns every
    available chip inside the horizon, because within a 5-gameweek window a chip
    is pure upside with no opportunity cost.
    ``chip_decay`` discounts the *chip-specific* payoff (Triple Captain's extra
    captain haul, Bench Boost's bench) separately from ``decay``. The horizon
    decay exists to discount uncertain far-future transfer planning, but
    applying it to a one-week chip makes every chip drift to the first
    gameweek — at ``decay=0.85`` a chip worth 10 pts in GW5 scores 6.1 against
    8 pts in GW2, so the model plays it early rather than where it actually
    pays. Default 1.0 times those chips on their real payoff; lower it to
    re-introduce caution about distant projections.
    ``allow_hits=False`` forbids paid transfers outright — the honest way to
    express a no-hits playstyle (the -4 itself is never re-priced).
    ``forced_in``/``forced_out`` map gw -> player_ids that must move that gw.
    ``min_ft`` maps gw -> minimum banked FTs to hold *after* that gw's moves.
    ``unlimited_first`` models the pre-GW1-deadline state: the first gw's
    transfers are free (no hits, no FTs consumed, any number of moves) and
    the FT stock resets to 1 afterwards — like a scratch build, but from an
    existing squad so the plan still reads as transfers.
    """
    _check_selling_prices(initial)
    scratch = initial is None
    free0 = scratch or unlimited_first     # first gw's moves are free
    P = proj.reset_index(drop=True)
    ids = list(P["player_id"])
    pos = dict(zip(P["player_id"], P["position"]))
    club = dict(zip(P["player_id"], P["team_id"]))
    price = dict(zip(P["player_id"], P["price"]))
    name = dict(zip(P["player_id"], P["player"]))
    ep = {g: dict(zip(P["player_id"], P[f"ep_gw{g}"])) for g in gws}

    prev = {pid: (0 if scratch else (1 if pid in initial else 0)) for pid in ids}
    sell = {pid: (initial.get(pid, price[pid]) if not scratch else price[pid])
            for pid in ids}
    start_bank = budget if scratch else bank

    chip_gws = {c: set(v) for c, v in (chip_gws or {}).items() if v}
    chip_force = chip_force or {}
    # how much season is left decides what a saved chip is worth; with three
    # gameweeks to go there is almost nothing left to save it for
    _left = gws_remaining if gws_remaining is not None else (
        max(0, 38 - max(gws)) + len(gws) if gws else None)
    reserve = {**default_reserve(_left), **(chip_reserve or {})}
    tc_on = bool(chip_gws.get("triple_captain"))
    bb_on = bool(chip_gws.get("bench_boost"))
    fh_on = bool(chip_gws.get("freehit"))
    locked = locked or set()
    avoid = avoid or set()
    banned_clubs = banned_clubs or set()
    sell_clubs = sell_clubs or set()
    forced_in = forced_in or {}
    forced_out = forced_out or {}
    min_ft = min_ft or {}

    T = list(range(len(gws)))
    gw_of = dict(enumerate(gws))
    BIG_MONEY = budget + sum(sorted(price.values())[-15:]) + 10

    prob = pulp.LpProblem("fpl_chips", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("sq", (ids, T), cat="Binary")   # squad PLAYED at t
    # squad carried out of t (differs from played squad only on a Free Hit week)
    carry = (pulp.LpVariable.dicts("cy", (ids, T), cat="Binary") if fh_on
             else squad)
    xi = pulp.LpVariable.dicts("xi", (ids, T), cat="Binary")
    cap = pulp.LpVariable.dicts("cap", (ids, T), cat="Binary")
    tin = pulp.LpVariable.dicts("in", (ids, T), cat="Binary")
    tout = pulp.LpVariable.dicts("out", (ids, T), cat="Binary")
    fused = pulp.LpVariable.dicts("fused", T, lowBound=0, cat="Integer")
    paid = pulp.LpVariable.dicts("paid", T, lowBound=0, cat="Integer")
    ftv = pulp.LpVariable.dicts("ft", T, lowBound=0, upBound=MAX_FREE_TRANSFERS,
                                cat="Integer")
    bankv = pulp.LpVariable.dicts("bank", T, lowBound=0)

    # chip vars exist only for enabled chips; disabled chips are the constant 0
    chip = {c: (pulp.LpVariable.dicts(f"chip_{c}", T, cat="Binary")
                if chip_gws.get(c) else dict.fromkeys(T, 0)) for c in CHIPS}
    tcp = (pulp.LpVariable.dicts("tcp", (ids, T), cat="Binary") if tc_on
           else None)                                          # cap AND TC
    bbp = (pulp.LpVariable.dicts("bbp", (ids, T), cat="Binary") if bb_on
           else None)                                          # bench AND BB

    clubs = set(club.values())

    # --- chip availability / forcing ---
    for c in CHIPS:
        allowed = chip_gws.get(c)
        if not allowed:
            continue
        for t in T:
            if gw_of[t] not in allowed:
                prob += chip[c][t] == 0
        prob += pulp.lpSum(chip[c][t] for t in T) <= 1
        if c in chip_force:
            fg = chip_force[c]
            if fg in gws:
                prob += chip[c][gws.index(fg)] == 1
    if chip_gws:
        for t in T:
            prob += pulp.lpSum(chip[c][t] for c in CHIPS) <= 1
    wc, fh, bb, tc = (chip["wildcard"], chip["freehit"],
                      chip["bench_boost"], chip["triple_captain"])

    # --- objective ---
    obj = []
    for t, g in enumerate(gws):
        d = decay ** t
        cd = chip_decay ** t          # one-week chips are timed on real payoff
        for p in ids:
            e = ep[g][p]
            obj.append(d * e * (xi[p][t] + cap[p][t]))
            obj.append(d * bench_weight * e * (squad[p][t] - xi[p][t]))
            if tc_on:
                obj.append(cd * e * tcp[p][t])
            if bb_on:
                obj.append(cd * (1.0 - bench_weight) * e * bbp[p][t])
        obj.append(-hit_cost * paid[t])
        # opportunity cost of spending a chip now instead of saving it
        for c in CHIPS:
            if chip_gws.get(c) and reserve.get(c):
                obj.append(-float(reserve[c]) * chip[c][t])
    obj.append(ft_value * ftv[T[-1]])
    # --- portfolio concentration (experimental; 0.0 is an exact no-op) ---
    # Players from one club share a fixture, so their returns are correlated:
    # independence understates a one-club triple-up's spread by 6% and a whole
    # XI's by 8%. A positive weight prices that correlation as a cost (spread
    # out), a negative one as a benefit (concentrate deliberately).
    #
    # The excess must be pinned EXACTLY, not merely bounded below. A one-sided
    # `z >= n - 1` is correct for a penalty but leaves the objective unbounded
    # the moment the weight goes negative, because nothing stops z running to
    # infinity. So y_k is a true indicator for "at least k from this club":
    #     n >= k * y_k         y_k can only be 1 when the players are there
    #     n <= (k-1) + M * y_k y_k must be 1 once they are
    # and the excess is y_2 + y_3, exact for every n in 0..MAX_PER_CLUB.
    if concentration:
        for t_i, g in enumerate(gws):
            d = decay ** t_i
            for cl in sorted(set(club.values())):
                members = [p for p in ids if club[p] == cl]
                if len(members) < 2:
                    continue
                n = pulp.lpSum(xi[p][t_i] for p in members)
                for k in range(2, MAX_PER_CLUB + 1):
                    y = pulp.LpVariable(f"conc_{cl}_{t_i}_{k}", cat="Binary")
                    prob += n >= k * y
                    prob += n <= (k - 1) + MAX_PER_CLUB * y
                    obj.append(-concentration * d * y)

    prob += pulp.lpSum(obj)

    # --- per-gameweek structure (played squad) ---
    for t in T:
        prob += pulp.lpSum(squad[p][t] for p in ids) == SQUAD_SIZE
        for pp, q in POSITION_QUOTA.items():
            prob += pulp.lpSum(squad[p][t] for p in ids if pos[p] == pp) == q
        for cl in clubs:
            prob += pulp.lpSum(squad[p][t] for p in ids if club[p] == cl) <= MAX_PER_CLUB
        prob += pulp.lpSum(xi[p][t] for p in ids) == XI_SIZE
        prob += pulp.lpSum(cap[p][t] for p in ids) == 1
        for p in ids:
            prob += xi[p][t] <= squad[p][t]
            prob += cap[p][t] <= xi[p][t]
            # chip product linearisations (maximisation pulls them up, so the
            # <= pair suffices — no lower bounds needed)
            if tc_on:
                prob += tcp[p][t] <= cap[p][t]
                prob += tcp[p][t] <= tc[t]
            if bb_on:
                prob += bbp[p][t] <= bb[t]
                prob += bbp[p][t] <= squad[p][t] - xi[p][t]
        for pp in POSITION_QUOTA:
            n = pulp.lpSum(xi[p][t] for p in ids if pos[p] == pp)
            prob += n >= XI_MIN[pp]
            prob += n <= XI_MAX[pp]

    # --- squad continuity, transfers, carry (Free Hit reverts) ---
    for t in T:
        for p in ids:
            base = prev[p] if t == 0 else carry[p][t - 1]
            if fh_on:
                # played squad follows base + transfers, unless FH (then free)
                prob += squad[p][t] - base - tin[p][t] + tout[p][t] <= 2 * fh[t]
                prob += squad[p][t] - base - tin[p][t] + tout[p][t] >= -2 * fh[t]
                prob += tin[p][t] <= 1 - fh[t]     # FH week: no real transfers
                prob += tout[p][t] <= 1 - fh[t]
                # carry = played squad normally; on FH, carry = base (revert)
                prob += carry[p][t] <= squad[p][t] + fh[t]
                prob += carry[p][t] >= squad[p][t] - fh[t]
                prob += carry[p][t] <= base + (1 - fh[t])
                prob += carry[p][t] >= base - (1 - fh[t])
            else:
                prob += squad[p][t] == base + tin[p][t] - tout[p][t]
            prob += tin[p][t] + tout[p][t] <= 1

    # --- bank recursion (actual bank only) + FH affordability ---
    for t in T:
        prev_bank = start_bank if t == 0 else bankv[t - 1]
        prob += bankv[t] == (prev_bank
                             + pulp.lpSum(sell[p] * tout[p][t] for p in ids)
                             - pulp.lpSum(price[p] * tin[p][t] for p in ids))
        if fh_on:
            # Free-Hit squad must be affordable from carried squad value + bank
            base_val = (pulp.lpSum(sell[p] * prev[p] for p in ids) if t == 0
                        else pulp.lpSum(sell[p] * carry[p][t - 1] for p in ids))
            prob += (pulp.lpSum(sell[p] * squad[p][t] for p in ids)
                     <= base_val + prev_bank + BIG_MONEY * (1 - fh[t]))

    # --- transfer counts, hits, free-transfer stock ---
    for t in T:
        nt = pulp.lpSum(tin[p][t] for p in ids)
        cap_n = SQUAD_SIZE if (free0 and t == 0) else max_transfers_per_gw
        prob += nt <= cap_n + SQUAD_SIZE * wc[t]      # WC lifts the per-gw cap
        prob += fused[t] <= nt
        prob += fused[t] <= ftv[t]
        # WC/FH weeks consume no FTs and cost no hits
        prob += fused[t] <= SQUAD_SIZE * (1 - wc[t] - fh[t]) + 0
        if free0 and t == 0:
            # building the initial 15 / pre-deadline moves are free: no hits,
            # no FTs consumed
            prob += paid[t] == 0
            prob += fused[t] == 0
        else:
            prob += paid[t] >= nt - fused[t] - SQUAD_SIZE * (wc[t] + fh[t])
        # FT stock recursion
        if t == 0:
            prob += ftv[t] == (1 if scratch else free_transfers)
        elif free0 and t == 1:
            prob += ftv[t] == 1
        else:
            # A Wildcard or Free Hit week PRESERVES the stock, it does not
            # add to it. FPL: "any saved free transfers are maintained for the
            # following Gameweek. If you had 2 saved free transfers, you will
            # still have 2 saved free transfers the Gameweek after playing the
            # chip." Two in, two out - so the usual +1 accrual is suspended for
            # a gameweek in which a chip was played. wc/fh are literal 0 for
            # any chip the manager does not hold, so this is inert then.
            prob += (ftv[t] <= ftv[t - 1] - fused[t - 1] + 1
                     - wc[t - 1] - fh[t - 1])
            prob += ftv[t] >= 1
        g = gw_of[t]
        if g in min_ft:
            prob += ftv[t] - fused[t] >= min_ft[g]

    # --- user constraints ---
    for p in locked:
        if p not in pos:
            continue
        if prev.get(p):
            for t in T:
                prob += squad[p][t] == 1          # hold: never sold
        else:
            prob += squad[p][T[-1]] == 1          # target: owned by horizon end
    for p in avoid:
        if p not in pos:
            continue
        for t in T:
            prob += tin[p][t] == 0
        prob += squad[p][T[-1]] == 0              # gone (or never bought) by end
        if not prev.get(p):
            for t in T:
                prob += squad[p][t] == 0
    for p in ids:
        if club[p] in banned_clubs and not prev.get(p):
            for t in T:
                prob += squad[p][t] == 0
        if club[p] in sell_clubs:
            for t in T:
                prob += tin[p][t] == 0        # never buy into the club
            prob += squad[p][T[-1]] == 0      # and be out of it by the end
    if not allow_hits:
        for t in T:
            prob += paid[t] == 0
    for g, plist in forced_in.items():
        if g in gws:
            for p in plist:
                if p in pos:
                    prob += tin[p][gws.index(g)] == 1
    for g, plist in forced_out.items():
        if g in gws:
            for p in plist:
                if p in pos:
                    prob += tout[p][gws.index(g)] == 1

    # --- solve, extract, cut, repeat for alternative plans ---
    plans: list[ChipPlan] = []
    for k in range(max(1, n_plans)):
        if on_progress:
            on_progress(f"Solving plan {k + 1}/{n_plans}…")
        _solve(prob, time_limit)
        if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
            break
        plan = _extract(prob, ids, gws, name, pos, club, price, sell, ep,
                        squad, xi, cap, tin, tout, fused, paid, ftv, bankv,
                        chip, carry, prev)
        if plan is None:
            break
        plans.append(plan)
        if k + 1 >= n_plans:
            break
        # no-good cut over the decision signature (transfers + chips)
        chosen, others = [], []
        for t in T:
            for p in ids:
                (chosen if _v(tin[p][t]) else others).append(tin[p][t])
                (chosen if _v(tout[p][t]) else others).append(tout[p][t])
            for c in CHIPS:
                (chosen if _v(chip[c][t]) else others).append(chip[c][t])
        prob += (pulp.lpSum(chosen) - pulp.lpSum(others) <= len(chosen) - 1)
    return plans


def _v(var) -> int:
    return int(round(pulp.value(var) or 0))


def _extract(prob, ids, gws, name, pos, club, price, sell, ep,
             squad, xi, cap, tin, tout, fused, paid, ftv, bankv, chip,
             carry=None, prev=None):
    def _by_position(p):
        """Order transfers so the IN and OUT lists pair up by index."""
        return (list(POSITION_QUOTA).index(pos[p]), -price[p], p)

    plan = ChipPlan(gws=gws, objective=pulp.value(prob.objective) or 0.0,
                    status=pulp.LpStatus[prob.status])
    for t, g in enumerate(gws):
        squad_ids = [p for p in ids if _v(squad[p][t])]
        if len(squad_ids) != SQUAD_SIZE:
            return None  # infeasible/timeout garbage
        xi_ids = [p for p in ids if _v(xi[p][t])]
        cap_id = next((p for p in ids if _v(cap[p][t])), None)
        chips_played = [c for c in CHIPS if _v(chip[c][t])]
        chip_now = chips_played[0] if chips_played else None
        # vice = best remaining XI player by ep
        vice_id = None
        rest = sorted((p for p in xi_ids if p != cap_id),
                      key=lambda p: -ep[g][p])
        if rest:
            vice_id = rest[0]
        mult = 3 if chip_now == "triple_captain" else 2
        xi_pts = sum(ep[g][p] for p in xi_ids)
        if cap_id is not None:
            xi_pts += (mult - 1) * ep[g][cap_id]
        bench_ids = [p for p in squad_ids if p not in xi_ids]
        fh_in = fh_out = []
        if chip_now == "freehit" and prev is not None:
            base_ids = {p for p in ids
                        if (prev[p] if t == 0 else _v(carry[p][t - 1]))}
            fh_in = [{"player_id": p, "name": name[p], "position": pos[p],
                      "ep": round(ep[g][p], 2)}
                     for p in sorted(set(squad_ids) - base_ids, key=_by_position)]
            fh_out = [{"player_id": p, "name": name[p], "position": pos[p],
                       "ep": round(ep[g][p], 2)}
                      for p in sorted(base_ids - set(squad_ids), key=_by_position)]
        if chip_now == "bench_boost":
            xi_pts += sum(ep[g][p] for p in bench_ids)
        plan.per_gw.append({
            "gw": g,
            "chip": chip_now,
            "squad": [{
                "player_id": p, "name": name[p], "position": pos[p],
                "team_id": club[p], "price": price[p], "sell": sell[p],
                "ep": round(ep[g][p], 2),
                "in_xi": p in xi_ids,
                "is_captain": p == cap_id, "is_vice": p == vice_id,
            } for p in sorted(squad_ids,
                              key=lambda p: (["GK", "DEF", "MID", "FWD"].index(pos[p]),
                                             -ep[g][p]))],
            "captain": name.get(cap_id), "captain_id": cap_id,
            "vice": name.get(vice_id), "vice_id": vice_id,
            "xi_points": round(xi_pts, 2),
            # Both lists are ordered by POSITION, which makes pairing them by
            # index legal. The squad quota is enforced every gameweek, so the
            # number leaving a position always equals the number arriving in
            # it; without the sort these are two id-ordered sets and a consumer
            # zipping them shows moves like "DEF -> FWD" that the solver never
            # made and the rules would not allow.
            "transfers_in": [{"player_id": p, "name": name[p],
                              "position": pos[p]}
                             for p in sorted((p for p in ids if _v(tin[p][t])),
                                             key=_by_position)],
            "transfers_out": [{"player_id": p, "name": name[p],
                               "position": pos[p]}
                              for p in sorted((p for p in ids if _v(tout[p][t])),
                                              key=_by_position)],
            # A Free Hit is not a transfer - the rules make it free and it
            # reverts - so tin/tout are pinned to zero that week and the plan
            # would otherwise report "no changes" for the one gameweek that
            # changes most. These say what the chip actually does: the played
            # squad against the squad you would have fielded without it.
            "fh_in": fh_in, "fh_out": fh_out,
            "n_transfers": sum(_v(tin[p][t]) for p in ids),
            "free_used": _v(fused[t]), "hits": _v(paid[t]),
            "free_after": _v(ftv[t]) - _v(fused[t]),
            "bank": round(pulp.value(bankv[t]) or 0.0, 1),
        })
    return plan
