"""Decision sensitivity: which choices are fragile, and to what.

Phase 2.5: instead of a better objective (there isn't one — Round 14), find
where the champion's decision hangs by a thread, because those are the only
places where new information (team news, a lineup leak, an odds move) can
change the decision at all. A captain 1.4 xP clear of the field is
information-robust: nothing short of an injury moves him. Two midfielders
0.03 apart are information-sensitive: any signal that shifts either by a
hair flips the pick, so that is where information acquisition pays.

Two measures per decision, both from the simulator's joint draws:

* **margin** — expected-points gap to the best alternative (captain: next
  best armband; XI slot: best legal swap with a benched player).
* **stability** — P(the choice remains optimal under estimation
  uncertainty): bootstrap the draws (resample gameweeks the match could
  have gone), recompute the argmax, count how often it survives. A margin
  of 0.2 on a volatile pair can be far less stable than 0.2 on a quiet
  one, which is why both numbers are reported.

This is reporting, not a new objective: the decision itself stays max-xP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rank_utility import POS_MIN, legal_xis

BOOTSTRAPS = 400


def _best_xi(mean: np.ndarray, positions: list[str]) -> list[int]:
    by_pos = {p: [i for i, q in enumerate(positions) if q == p]
              for p in POS_MIN}
    return list(max(legal_xis(by_pos),
                    key=lambda xi: float(mean[list(xi)].sum())))


def analyse_squad(draws: np.ndarray, players: pd.DataFrame,
                  squad_ids: list[int], *, n_boot: int = BOOTSTRAPS,
                  seed: int = 0) -> dict:
    """Margins and bootstrap stability for one squad's max-xP decision.

    ``draws`` (n_sims, n_players) aligned with ``players`` (player_id,
    position); the 15 ``squad_ids`` must all be present.
    """
    col = {p: i for i, p in enumerate(players["player_id"])}
    missing = [p for p in squad_ids if p not in col]
    if missing:
        raise ValueError(f"players missing from draws: {missing}")
    cols = [col[p] for p in squad_ids]
    sub = np.asarray(draws, dtype=np.float64)[:, cols]
    positions = [players["position"].iloc[col[p]] for p in squad_ids]
    mean = sub.mean(axis=0)

    xi = _best_xi(mean, positions)
    order = sorted(xi, key=lambda i: -mean[i])
    cap, vice = order[0], order[1]
    cap_margin = float(mean[cap] - mean[vice])

    # XI slot margins: for each starter, the best legal swap with a benched
    # outfielder (GK swaps with the bench GK only)
    bench = [i for i in range(15) if i not in xi]
    swaps = []
    for i in xi:
        best_alt, best_gain = None, -np.inf
        for j in bench:
            trial = [k for k in xi if k != i] + [j]
            by_pos = {p: sum(1 for k in trial if positions[k] == p)
                      for p in POS_MIN}
            if any(by_pos[p] < POS_MIN[p] for p in POS_MIN) or by_pos["GK"] != 1:
                continue
            gain = float(mean[j] - mean[i])
            if gain > best_gain:
                best_alt, best_gain = j, gain
        if best_alt is not None:
            swaps.append({"out": squad_ids[i], "in": squad_ids[best_alt],
                          "margin": round(-best_gain, 3)})
    swaps.sort(key=lambda s: s["margin"])

    # bootstrap stability: resample the draws, redo the argmaxes
    rng = np.random.default_rng(seed)
    n = sub.shape[0]
    cap_stable = 0
    xi_stable = 0
    xi_set = set(xi)
    for _ in range(n_boot):
        m = sub[rng.integers(0, n, n)].mean(axis=0)
        b_xi = _best_xi(m, positions)
        xi_stable += set(b_xi) == xi_set
        cap_stable += int(np.argmax(np.where(
            np.isin(np.arange(15), b_xi), m, -np.inf))) == cap
    return {
        "xi": [squad_ids[i] for i in xi],
        "captain": squad_ids[cap], "vice": squad_ids[vice],
        "captain_margin": round(cap_margin, 3),
        "captain_stability": round(cap_stable / n_boot, 3),
        "xi_stability": round(xi_stable / n_boot, 3),
        "tightest_swaps": swaps[:5],
        "fragile": cap_margin < 0.3 or (swaps and swaps[0]["margin"] < 0.15),
    }
