"""How a playstyle bends the projections it optimises over.

Three things a manager means by "aggressive" that a horizon decay alone
cannot express:

  * **when** the points land — a short horizon. That is ``decay``, and it was
    already the only thing the styles varied.
  * **what kind** of points they are. Two players projected at 5.0 are not the
    same bet: one is a striker whose 5.0 is mostly goals (a blank or a haul),
    the other a defender whose 5.0 is appearance points, a clean sheet and
    DefCon (5.0 most weeks). Chasing a rank needs the first; protecting one
    needs the second. ``upside`` prices that preference against the explosive
    part of each projection (``ex_gw{g}`` — goals, assists and their bonus),
    which ``optimise/project.py`` carries from the component engine.
  * **what the squad is worth afterwards.** A price rise is realised only on
    sale and FPL returns half the profit, so it is worth *points* — about 0.16
    per £1m per gameweek held (Round 7). ``price`` weights that conversion.

Both are PREFERENCES, exactly as the existing style parameters are, and both
default to zero so the shipped Balanced style optimises the same objective it
always did. Neither is claimed to raise expected points: the measured facts
are that a price rise converts at ~0.2 points (so it is a tie-breaker, and is
weighted as one) and that rank-tilting the objective did not beat the mean
(Rounds 13-14) — which is why the aggressive tilt is small, opt-in per style,
and never applied to the Balanced default.
"""
from __future__ import annotations

import pandas as pd


def tilt_projection(proj: pd.DataFrame, gws: list[int], *,
                    upside: float = 0.0, price: float = 0.0) -> pd.DataFrame:
    """Return a copy of ``proj`` with each gameweek's EP bent by the style.

        ep' = ep + upside * explosive + price * price_points / len(gws)

    ``upside`` > 0 prefers players whose points arrive as goals and assists,
    < 0 prefers steady accumulators. ``price_points`` is already a points
    figure for the whole hold, so it is spread across the horizon rather than
    counted once per gameweek. ``ep_total`` is left alone: it is the pruning
    and reporting key, not the objective.
    """
    if not upside and not price:
        return proj
    out = proj.copy()
    n = max(1, len(gws))
    pp = (out["price_points"] if "price_points" in out.columns else 0.0)
    for g in gws:
        col = f"ep_gw{g}"
        if col not in out.columns:
            continue
        adj = 0.0
        if upside and f"ex_gw{g}" in out.columns:
            adj = adj + upside * out[f"ex_gw{g}"].fillna(0.0)
        if price:
            adj = adj + price * pp / n
        out[col] = (out[col] + adj).clip(lower=0.0)
    return out
