# Research log — FPL decision engine

Every experiment, including the failures, per the autonomous-research mandate.
Convention: an experiment "ships" only if it survives a paired out-of-sample
test; p > 0.05 is unproven regardless of the mean. With N arms compared
against one baseline, only p < 0.05/N survives multiple testing — stated per
entry. Earlier modelling rounds (projection side) are logged in CLAUDE.md
Rounds 1–12 and are not repeated here; their standing conclusions
(projection is minutes-limited; effects measured in realised outcomes are
already absorbed by rates estimated from realised outcomes) gate what gets
re-tested.

Replay design shared by all decision-layer entries: squad fixed to the
lagged-ownership template 15, arms differ only in XI/captain/vice/bench
order; realised scoring applies FPL's real autosub + armband rules to actual
minutes; ownership lagged one gameweek; availability overlay off (stored
status is today's, not historical). `python -m fpl_engine rank-backtest`,
pooled with `compare-rank-backtests`.

---

## E1. MFRU decision layer vs max-xP and the crowd (Round 13)

* **Hypothesis.** Scoring the full decision on the differential against the
  ownership mean-field (Delta), with autosubs/armband applied per simulator
  draw, beats ranking by expected points.
* **Design.** 2024-25 + 2025-26, 74 paired gameweeks, 3000 draws/gw,
  minutes model trained only on prior seasons per replay.
* **Result.** Every model arm beats the crowd (max-xP +2.11/gw p=0.027;
  MFRU g0 +2.38/gw p=0.008). MFRU g0 vs max-xP: +0.27/gw, p=0.63, sign
  flips between seasons. gamma in ±{0.15,0.3}: all |Δ|≤0.28, p≥0.51.
* **Verdict.** Decision layer vs crowd: **real and replicated**. MFRU vs
  max-xP: **unproven**. Theorem: E[Delta] is EO-free, so no realised-mean
  test can validate gamma≠0; it can only bound its cost (≤0.3 pts/gw here).
* **Status.** Shipped as measurement infrastructure; gamma=0 in any live use.

## E2. Third replay season + strong baselines + rank objectives (E1 scaled)

* **Hypothesis bundle** (pre-registered here before the run finished):
  (a) with n≈111 gameweeks (adding 2023-24), does mfru_g0's +0.27 vs xp
  survive or shrink; (b) does any rank functional — P(beat field), CVaR20,
  Q80, gamma grid ±{0.15,0.3,0.5,1} — beat max-xP on realised points or on
  realised beat-the-field frequency; (c) do all model arms clear the
  required baselines (random-valid, competent-human heuristic, crowd
  template, FPL's own site-published expected points).
* **Design.** 2023-24 (minutes model trained on 2022-23 only — weaker, so a
  conservative test), 2024-25, 2025-26; 19 arms; multiple-testing alpha
  0.05/18 ≈ 0.0028 for the vs-xp family.
* **Baseline mapping to the mandate's list:** random valid = `random`;
  highest-xP XI/captain = `xp`; ownership template = `crowd`;
  human heuristic = `human` (form × nailedness + opponent leakiness + venue
  + premium-captain bias, deadline-public info only); conservative template
  = `mfru_g-1/-0.5`; aggressive differential = `mfru_g+0.5/+1` (within a
  fixed template squad, differential strategies exist only in the captain /
  XI-variance channel — free-squad differentials are a transfer-layer
  question, out of scope for this controlled design); public benchmark =
  `site_ep` (FPL's own published ep_this, archived by vaastav; capture
  timing is the archiver's — benchmark only, never an input).
* **Result.** See CLAUDE.md Round 14 (filled after the run).

## E3. Commercial benchmark search

* **Hypothesis.** A commercial/public model's historical per-gameweek
  projections can be legally obtained for 2023-24 … 2025-26 and benchmarked.
* **Findings.** theFPLkiwi's public GitHub archive ends 2023-12-22 and holds
  only 4 gameweek files for 2023-24 (GW1/3/4/18) — too sparse to pair.
  FPL Review / Fantasy Football Hub publish no historical projection
  archives; their live outputs are paywalled and scraping their sites is not
  a legal acquisition route. FPL's own `ep_this` (see E2) is the one
  season-long, freely archived public projection; the proven-elite-manager
  panel (`acquire panel/picks`) is the forward-collection route to a
  live-manager benchmark, testable ~10-15 gameweeks after collection starts.
* **Verdict.** Season-long historical commercial benchmarks: **wall**, not
  skipped — documented. `site_ep` stands in as the public benchmark.

## Standing walls (do not re-attempt without a new data source)

* Historical captaincy/XI **distributions of the field** exist in no free
  feed — the mean-field's captain mass is a proxy (most-owned player), and
  segment/mixture models of the field (casual vs top-10k) cannot be fitted
  historically. Forward collection via the elite panel is the only route.
* **Rank distributions** (validating gamma≠0 or any rank-tail objective on
  realised rank) need thousands of gameweeks; two–three seasons cannot do
  it. The theorem in E1 says realised-mean tests are structurally unable to.
* Historical **availability/team-news** state: overwritten before the
  acquire change log existed; the overlay stays live-only.
* Everything in CLAUDE.md "Tested and rejected" and Rounds 8/8b (projection
  sharpening) stands.
