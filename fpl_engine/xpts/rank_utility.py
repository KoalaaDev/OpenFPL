"""MFRU — Mean-Field Rank Utility: FPL decisions scored against the crowd.

The engine predicts E[points]; this layer prices what those points are WORTH
to a manager's rank. FPL is a game against six million opponents whose
aggregate squad is public (ownership), so a manager's weekly rank return is
not their score S but the differential against the ownership mean-field:

    Delta = S(d) - S_field,      S_field = sum_i eo_i * X_i

where X is the JOINT points vector (from ``xpts.simulate``, so teammate and
scoreline correlations are kept), eo is effective ownership (fielded share
plus the crowd captain's doubled mass), and d is the full decision — XI,
captain, vice-captain and bench order, pushed through FPL's automatic
substitutions *inside each draw*.

The objective is a utility over the differential, not the score:

    U_gamma(d) = E[Delta] + gamma * sd(Delta)

Three consequences fall out of the algebra rather than being modelled ad hoc:

1. **E[Delta] = E[S] - const.** The mean-field term does not depend on the
   decision, so a risk-NEUTRAL manager should ignore effective ownership
   entirely — the max-xP pick already maximises expected rank return. Any EO
   "edge" lives exclusively in the gamma term. This is a theorem, not a
   modelling choice, and it is why chasing differentials for their own sake
   costs points in expectation.
2. **gamma prices the EO cascade.** sd(Delta) is computed with a - eo inside
   the (co)variance, so owning the template hedges rank risk and a
   differential adds it: gamma > 0 (rank chasing, e.g. needing a top-1k
   push) buys variance, gamma < 0 (rank protecting) shadows the crowd.
   Both use the same E[points]; only the risk price moves.
3. **The bench dilemma and the armband are handled per draw, not in
   expectation.** A fringe starter's value includes the P(0 minutes) branch
   where his bench replacement's points arrive via the autosub operator; the
   captain's includes the branch where the armband passes to the vice. Both
   are nonlinear in the joint minutes draw, which is exactly what a
   per-player expectation cannot see.

``decide`` is exhaustive over legal formations from a 15 (~3k XIs), coarse-
scored without autosubs, then the top candidates are re-scored exactly
(autosubs, all 11 armbands, all 6 bench orders).
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd

# XI formation limits (GK is exactly 1; outfield mins/maxes are FPL's).
POS_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
POS_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
AVG_SQUAD_SIZE = 15.0      # normalises raw `selected` counts to manager share
FINE_TOP_K = 25            # candidates re-scored exactly after the coarse pass


# ------------------------------------------------------- effective ownership
def effective_ownership(selected: pd.Series, fielded_weight: float = 11.0 / 15.0,
                        captain_extra: bool = True) -> pd.Series:
    """Effective ownership per player from raw `selected` counts.

    ``selected`` (index player_id) are FPL's "number of squads owning him"
    counts. The manager population is not stored anywhere, but every manager
    owns exactly 15 players, so N_managers = sum(selected)/15 and the raw
    ownership share needs no external input. The fielded share is approximated
    by a flat 11/15 (the crowd benches someone, and which four is not
    observable historically), and the crowd captain — proxied by the single
    most-owned player, which in practice is the template premium — carries his
    ownership again for the doubling.

    The proxy's absolute level only scales U_gamma; decisions depend on
    relative EO, and both arms of any comparison see the same vector.
    """
    total = float(selected.sum())
    if total <= 0:
        return selected * 0.0
    share = selected * (AVG_SQUAD_SIZE / total)
    eo = share.clip(upper=1.0) * fielded_weight
    if captain_extra and len(eo):
        eo.loc[eo.idxmax()] += float(eo.max())
    return eo


def legal_xis(by_pos: dict[str, list[int]]):
    """Every legal XI from a 2/5/5/3 squad, as tuples of squad indices."""
    out = []
    for gk in by_pos["GK"]:
        for d in range(POS_MIN["DEF"], min(POS_MAX["DEF"], len(by_pos["DEF"])) + 1):
            for m in range(POS_MIN["MID"], min(POS_MAX["MID"], len(by_pos["MID"])) + 1):
                f = 10 - d - m
                if not (POS_MIN["FWD"] <= f <= min(POS_MAX["FWD"], len(by_pos["FWD"]))):
                    continue
                for dd in combinations(by_pos["DEF"], d):
                    for mm in combinations(by_pos["MID"], m):
                        for ff in combinations(by_pos["FWD"], f):
                            out.append((gk,) + dd + mm + ff)
    return out


# ------------------------------------------------------------ autosub layer
def autosub_points(pts: np.ndarray, played: np.ndarray, positions: list[str],
                   xi: list[int], captain: int | None, vice: int | None,
                   bench: list[int]) -> np.ndarray:
    """Per-draw squad score under FPL's automatic-substitution rules.

    ``pts``/``played`` are (n_draws, n_squad) aligned with ``positions``;
    ``xi``/``bench`` are column indices (bench in priority order, its GK
    first). A player with 0 minutes scores 0, so non-playing starters need no
    removal — the operator only *adds* bench players who legally come on:

    * the bench GK replaces the starting GK only (like for like),
    * an outfield bench player comes on for a non-playing starter provided
      the resulting named-XI composition stays a legal formation; same
      position is always legal, otherwise the starter's slot must be
      droppable (above its position minimum) and the sub's addable,
    * the armband passes to the vice when the captain does not play
      (``captain=None`` skips the armband, for callers that add it
      separately — the autosub part does not depend on it).

    Vectorised over draws; the loop is over the three outfield bench slots.
    """
    n = pts.shape[0]
    # a player with 0 minutes scores 0 by rule; zero it explicitly rather
    # than assuming the caller's points array already does
    live = pts * played
    score = live[:, xi].sum(axis=1)

    pos = np.asarray(positions)
    xi_arr = np.asarray(xi)
    gk_xi = [i for i in xi if positions[i] == "GK"]
    bench_gk = [i for i in bench if positions[i] == "GK"]
    if gk_xi and bench_gk:
        swap = ~played[:, gk_xi[0]] & played[:, bench_gk[0]]
        score += np.where(swap, live[:, bench_gk[0]], 0.0)

    # per-draw state: composition of the named XI and, per position, how many
    # non-playing starters are still waiting for a replacement
    comp = {p: np.full(n, int((pos[xi_arr] == p).sum())) for p in POS_MIN}
    need = {p: (~played[:, [i for i in xi if positions[i] == p]]).sum(axis=1)
            for p in ("DEF", "MID", "FWD")}

    for j in [i for i in bench if positions[i] != "GK"]:
        q = positions[j]
        avail = played[:, j].copy()
        came_on = avail & (need[q] > 0)          # like for like: always legal
        need[q] = need[q] - came_on
        avail &= ~came_on
        for p in ("DEF", "MID", "FWD"):          # cross-position, if legal
            if p == q:
                continue
            ok = (avail & (need[p] > 0) & (comp[p] - 1 >= POS_MIN[p])
                  & (comp[q] + 1 <= POS_MAX[q]))
            need[p] = need[p] - ok
            comp[p] = comp[p] - ok
            comp[q] = comp[q] + ok
            came_on |= ok
            avail &= ~ok
        score += np.where(came_on, live[:, j], 0.0)

    if captain is not None:
        cap_on = played[:, captain]
        score += np.where(cap_on, live[:, captain], live[:, vice])
    return score


# ---------------------------------------------------------------- the model
class MeanFieldRankUtility:
    """Choose XI, captain, vice and bench order by rank utility.

    Parameters
    ----------
    draws : (n_sims, n_players) joint points samples from ``simulate_gw``
    mins : matching per-draw minutes (``simulate_gw``'s ``mins``)
    players : DataFrame with player_id, position aligned to draw columns
    eo : Series player_id -> effective ownership (missing ids read as 0)
    gamma : risk price on sd(Delta); 0 = risk-neutral (max expected rank
        return, provably = max xP up to the autosub/armband channels)
    objective : which functional of the per-draw differential to maximise —
        "mean_sd"  E[Delta] + gamma sd(Delta)          (default)
        "p_beat"   P(Delta > 0), the beat-the-field frequency (a tiny mean
                   tiebreak resolves the 1/n_sims granularity)
        "cvar20"   mean of the worst 20% of Delta      (downside protection)
        "q80"      80th percentile of Delta            (upside chasing)
    The non-default objectives exist to test the rank question directly:
    points are a proxy, rank return is the game, and these are the honest
    candidates for "optimise rank, not points" that per-draw samples allow.
    """

    OBJECTIVES = ("mean_sd", "p_beat", "cvar20", "q80")

    def __init__(self, draws: np.ndarray, mins: np.ndarray,
                 players: pd.DataFrame, eo: pd.Series, gamma: float = 0.0,
                 objective: str = "mean_sd"):
        if objective not in self.OBJECTIVES:
            raise ValueError(f"unknown objective {objective!r}")
        self.objective = objective
        self.draws = np.asarray(draws, dtype=np.float32)
        self.played = np.asarray(mins) > 0
        self.ids = players["player_id"].to_numpy()
        self.position = players["position"].fillna("MID").tolist()
        self.col = {p: i for i, p in enumerate(self.ids)}
        self.gamma = float(gamma)
        eo_vec = np.array([float(eo.get(p, 0.0)) for p in self.ids],
                          dtype=np.float32)
        # the mean-field opponent: one number per draw, shared by every
        # candidate decision — this is what makes E[Delta] decision-free
        self.field = self.draws @ eo_vec

    # -- utility over per-draw squad scores --------------------------------
    def _obj(self, delta: np.ndarray) -> np.ndarray:
        """Objective per candidate; ``delta`` is (n_sims, n_candidates)."""
        if self.objective == "p_beat":
            return (delta > 0).mean(axis=0) + 1e-6 * delta.mean(axis=0)
        if self.objective == "cvar20":
            k = max(1, int(0.2 * delta.shape[0]))
            return np.partition(delta, k - 1, axis=0)[:k].mean(axis=0)
        if self.objective == "q80":
            return np.quantile(delta, 0.8, axis=0)
        return delta.mean(axis=0) + self.gamma * delta.std(axis=0)

    def utility(self, score: np.ndarray) -> float:
        return float(self._obj((score - self.field)[:, None])[0])

    def decide(self, squad_ids: list[int]) -> dict:
        """Full decision for a given 15: XI, captain, vice, bench order."""
        cols = [self.col[p] for p in squad_ids if p in self.col]
        if len(cols) < 15:
            missing = [p for p in squad_ids if p not in self.col]
            raise ValueError(f"players missing from the simulation: {missing}")
        sub = self.draws[:, cols]
        played = self.played[:, cols]
        positions = [self.position[c] for c in cols]
        by_pos = {p: [i for i, q in enumerate(positions) if q == p]
                  for p in POS_MIN}
        if any(len(by_pos[p]) < POS_MIN[p] for p in POS_MIN):
            raise ValueError("squad cannot field a legal XI")

        mean = sub.mean(axis=0)
        xis = legal_xis(by_pos)

        # coarse pass: no autosubs, captain = best mean in the XI (the armband
        # rarely changes between candidates sharing their best player)
        A = np.zeros((len(xis), 15), dtype=np.float32)
        caps = np.empty(len(xis), dtype=np.int64)
        for k, xi in enumerate(xis):
            A[k, list(xi)] = 1.0
            caps[k] = xi[int(np.argmax(mean[list(xi)]))]
        totals = sub @ A.T + sub[:, caps]                    # (n_sims, n_cand)
        u = self._obj(totals - self.field[:, None])
        order = np.argsort(-u)[:min(FINE_TOP_K, len(xis))]

        # fine pass: exact autosubs, every armband, every bench order — the
        # autosub score is captain-independent, so it is computed once per
        # bench order and the armband is layered on per candidate captain
        best = None
        for k in order:
            xi = list(xis[k])
            bench = [i for i in range(15) if i not in xi]
            bench_gk = [i for i in bench if positions[i] == "GK"]
            outfield = [i for i in bench if positions[i] != "GK"]
            for perm in permutations(outfield):
                border = bench_gk + list(perm)
                base = autosub_points(sub, played, positions, xi, None, None,
                                      border)
                for cap in xi:
                    vice_pool = [i for i in xi if i != cap]
                    vice = vice_pool[int(np.argmax(mean[vice_pool]))]
                    s = base + np.where(played[:, cap], sub[:, cap],
                                        sub[:, vice])
                    uu = self.utility(s)
                    if best is None or uu > best["u"]:
                        best = {"u": uu, "xi": xi, "cap": cap, "vice": vice,
                                "bench": border, "score": s}
        delta = best["score"] - self.field
        return {
            "xi": [squad_ids[i] for i in best["xi"]],
            "captain": squad_ids[best["cap"]],
            "vice": squad_ids[best["vice"]],
            "bench": [squad_ids[i] for i in best["bench"]],
            "utility": best["u"],
            "e_points": float(best["score"].mean()),
            "e_delta": float(delta.mean()),
            "sd_delta": float(delta.std()),
            "p_beat_field": float((delta > 0).mean()),
        }
