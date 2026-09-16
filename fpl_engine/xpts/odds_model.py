"""Betting-odds → Poisson goal rates.

Bookmaker odds are the sharpest free forecast of match outcomes — they price
team news, rotation, motivation and everything else the trailing-form team
model cannot see. This module turns a match's 1X2 (+ optional over/under 2.5)
odds into implied Poisson goal rates (λ_home, λ_away) that the xpts engine
blends with its own team model:

1. **De-margin**: decimal odds imply probabilities that sum to >1 (the
   bookmaker's overround). Proportional normalisation removes it.
2. **Invert**: find the (λ_home, λ_away) whose independent-Poisson outcome
   probabilities best match the de-margined market — least squares over a
   precomputed λ grid, then local refinement. The totals market, when
   present, pins down the overall goal level that 1X2 alone leaves loose.

Everything here is a pure function of the odds — no I/O, no state. The
engine-side blend weight lives in ``ODDS_WEIGHT`` (fitted by the backtest
sweep; 0 disables odds entirely).
"""
from __future__ import annotations

import math
from functools import lru_cache

ODDS_WEIGHT = 0.85      # λ_final = (1-w)·team model + w·odds. Backtest sweep
                        # (2024-25 + 2025-26): active-player spearman and
                        # captain improve monotonically in w; 0.7-1.0 are
                        # within noise, 0.85 keeps a team-model floor for
                        # fixtures whose odds are missing or stale.

MAX_GOALS = 10          # truncation for Poisson outcome sums
GRID = [round(0.2 + 0.05 * i, 2) for i in range(77)]   # λ ∈ [0.2, 4.0]
_GRID_PROBS: list[tuple[float, float, float, float, float, float]] = []


def demargin(*odds: float) -> list[float]:
    """Decimal odds -> probabilities, proportionally stripped of overround.

    Any non-positive/missing odd invalidates the set (returns [])."""
    if not odds or any(o is None or o <= 1e-9 for o in odds):
        return []
    raw = [1.0 / o for o in odds]
    s = sum(raw)
    return [r / s for r in raw]


@lru_cache(maxsize=8192)
def _pois_pmf(lam: float) -> tuple[float, ...]:
    p, out = math.exp(-lam), []
    for k in range(MAX_GOALS + 1):
        out.append(p)
        p *= lam / (k + 1)
    return tuple(out)


def outcome_probs(lam_h: float, lam_a: float) -> tuple[float, float, float, float]:
    """(P(home), P(draw), P(away), P(total>2.5)) under independent Poissons."""
    ph = pa = pd_ = po = 0.0
    h, a = _pois_pmf(lam_h), _pois_pmf(lam_a)
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            p = h[i] * a[j]
            if i > j:
                ph += p
            elif i == j:
                pd_ += p
            else:
                pa += p
            if i + j >= 3:
                po += p
    return ph, pd_, pa, po


def _grid_probs():
    if not _GRID_PROBS:
        for lh in GRID:
            for la in GRID:
                ph, pd_, pa, po = outcome_probs(lh, la)
                _GRID_PROBS.append((lh, la, ph, pd_, pa, po))
    return _GRID_PROBS


def implied_rates(p_home: float, p_draw: float, p_away: float,
                  p_over25: float | None = None) -> tuple[float, float]:
    """Solve (λ_home, λ_away) matching the de-margined market probabilities."""
    use_o = p_over25 is not None
    best, best_l = (1.4, 1.2), float("inf")
    for lh, la, ph, pd_, pa, po in _grid_probs():
        l = (ph - p_home) ** 2 + (pd_ - p_draw) ** 2 + (pa - p_away) ** 2
        if use_o:
            l += (po - p_over25) ** 2
        if l < best_l:
            best, best_l = (lh, la), l

    def loss(lh: float, la: float) -> float:
        ph, pd_, pa, po = outcome_probs(lh, la)
        l = (ph - p_home) ** 2 + (pd_ - p_draw) ** 2 + (pa - p_away) ** 2
        if use_o:
            l += (po - p_over25) ** 2
        return l

    lh, la = best
    step = 0.025
    for _ in range(3):                   # local refinement below grid pitch
        cands = [(max(lh + dh, 0.05), max(la + da, 0.05))
                 for dh in (-step, 0.0, step) for da in (-step, 0.0, step)]
        lh, la = min(cands, key=lambda c: loss(*c))
        step /= 2
    return round(lh, 4), round(la, 4)


def fixture_odds_map(conn, season: str, fixture_ids: list[int],
                     model_totals: dict[int, float] | None = None,
                     sources: dict[int, str] | None = None) -> dict[int, tuple[float, float]]:
    """{fixture_id: (λ_home, λ_away)} for the fixtures that have a market price.

    Bookmaker odds (`match_odds`: football-data for played matches, The Odds
    API for upcoming ones) come first. Where a fixture has none — the live
    path whenever the Odds API key is missing or rejected, which was the
    case for the whole of 2026-27 to GW4 — the latest Polymarket 1X2 quote
    (`market_quote`, keyless, refreshed on every pull) stands in. 1X2 alone
    leaves the goal LEVEL loose, so ``model_totals`` (the team model's
    λ_home+λ_away per fixture) pins it through the implied P(over 2.5): the
    market supplies the split and the strength, the model the total. The
    blend weight is unchanged (`ODDS_WEIGHT`), so a fixture with a Polymarket
    quote is treated exactly like one with a bookmaker price. ``sources``,
    if given, is filled with "bookmaker" / "polymarket" per fixture.
    """
    if not fixture_ids:
        return {}
    qs = ",".join("?" * len(fixture_ids))
    rows = conn.execute(
        f"SELECT fixture_id, lam_home, lam_away FROM match_odds "
        f"WHERE season=? AND fixture_id IN ({qs}) "
        f"AND lam_home IS NOT NULL AND lam_away IS NOT NULL",
        [season] + [int(f) for f in fixture_ids]).fetchall()
    out = {int(r["fixture_id"]): (float(r["lam_home"]), float(r["lam_away"]))
           for r in rows}
    if sources is not None:
        sources.update({k: "bookmaker" for k in out})
    missing = [int(f) for f in fixture_ids if int(f) not in out]
    if not missing:
        return out
    try:
        qm = ",".join("?" * len(missing))
        quotes = conn.execute(
            f"SELECT fixture_id, p_home, p_draw, p_away FROM market_quote "
            f"WHERE season=? AND fixture_id IN ({qm}) AND p_home IS NOT NULL "
            f"ORDER BY observed_utc", [season] + missing).fetchall()
    except Exception:      # noqa: BLE001 - table absent on an old database
        quotes = []
    latest = {int(r["fixture_id"]): r for r in quotes}       # last row = newest
    for fid, r in latest.items():
        ph, pd_, pa = float(r["p_home"]), float(r["p_draw"]), float(r["p_away"])
        tot = ph + pd_ + pa
        if tot <= 0:
            continue
        ph, pd_, pa = ph / tot, pd_ / tot, pa / tot
        p_over = None
        if model_totals and fid in model_totals and model_totals[fid] > 0:
            lam = float(model_totals[fid])
            pm = _pois_pmf(lam)                                # Poisson total
            p_over = max(0.0, 1.0 - (pm[0] + pm[1] + pm[2]))
        out[fid] = implied_rates(ph, pd_, pa, p_over)
        if sources is not None:
            sources[fid] = "polymarket"
    return out


# ---------------------------------------------------------------------------
# The market stretch: what the team model does to fixtures NO market has priced
# ---------------------------------------------------------------------------
# Over every fixture that both the team model and a bookmaker priced
# (2023-24 to 2025-26, ~2,160 fixtures), log(market lambda) on log(model
# lambda) has a slope of 1.23 / 1.23 / 1.31 with r = 0.92-0.95: the model
# RANKS fixtures as the market does and COMPRESSES the gaps by a quarter to
# a third, every season (ridge shrinkage and a 240-day window will do that).
# Where a fixture carries a price the blend fixes it; beyond the bookmakers'
# two-round horizon the far-horizon projections were left with the
# compressed spread — City vs Sunderland looked like a mid-table match. The
# stretch applies the fitted mapping to UNPRICED fixtures only. It is refit
# on every pull from the seasons on hand (point-in-time: model rates as of
# each gameweek's first kickoff), stored in models/xpts/market_stretch.json,
# and never touches a replay (every replayed fixture has football-data odds,
# and backtest.run passes market_stretch=False regardless).
STRETCH_PATH = None      # resolved lazily from config.MODELS_DIR
STRETCH_MIN_R = 0.8      # a season whose fit is this weak is left out (2022-23: r=0.60)


def _stretch_path() -> str:
    import os
    from .. import config
    return STRETCH_PATH or os.path.join(config.MODELS_DIR, "xpts", "market_stretch.json")


def fit_market_stretch(conn, seasons: list[str], *, path: str | None = None) -> dict | None:
    """Fit log(lambda_market) = a + b log(lambda_model) on priced fixtures of
    ``seasons`` and store it. Returns the record, or None when too few."""
    import json
    import os
    import numpy as np
    from . import engine as _eng, team_model as _tm
    lm, lk, used = [], [], {}
    for season in seasons:
        odds = {int(r["fixture_id"]): (r["lam_home"], r["lam_away"]) for r in conn.execute(
            "SELECT fixture_id, lam_home, lam_away FROM match_odds WHERE season=? AND lam_home IS NOT NULL",
            (season,))}
        if len(odds) < 60:
            continue
        s_lm, s_lk = [], []
        for gw in range(3, 39):
            as_of = _eng.first_kickoff(conn, season, gw)
            if not as_of:
                continue
            fx = _eng._gw_fixtures(conn, season, gw)
            if not fx or not any(f["fixture_id"] in odds for f in fx):
                continue
            tm = _tm.fit(conn, as_of)
            for f in fx:
                od = odds.get(f["fixture_id"])
                if not od or not od[0] or not od[1]:
                    continue
                lh, la = tm.fixture(f["hcode"], f["acode"])
                if lh > 0 and la > 0:
                    s_lm += [np.log(lh), np.log(la)]
                    s_lk += [np.log(od[0]), np.log(od[1])]
        if len(s_lm) < 100:
            continue
        r = float(np.corrcoef(s_lm, s_lk)[0, 1])
        if r < STRETCH_MIN_R:
            continue
        used[season] = {"n": len(s_lm), "r": round(r, 3)}
        lm += s_lm
        lk += s_lk
    if len(lm) < 300:
        return None
    b, a = np.polyfit(np.asarray(lm), np.asarray(lk), 1)
    rec = {"a": float(a), "b": float(b), "n": len(lm), "seasons": used,
           "r": round(float(np.corrcoef(lm, lk)[0, 1]), 3)}
    out = path or _stretch_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def load_market_stretch(path: str | None = None) -> dict | None:
    import json
    import os
    if os.environ.get("FPL_MARKET_STRETCH", "1") in ("0", "off", "false"):
        return None
    try:
        with open(path or _stretch_path(), encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if not rec or rec.get("b") is None or rec.get("a") is None:
        return None
    return rec


def apply_stretch(lam: float, rec: dict) -> float:
    """lambda' = exp(a + b log lambda), clipped to a sane goal rate."""
    import math
    if lam <= 0:
        return lam
    return float(min(4.0, max(0.2, math.exp(rec["a"] + rec["b"] * math.log(lam)))))
