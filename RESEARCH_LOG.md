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
* **Result** (111 paired gameweeks; vs the `xp` baseline, pts/gw, 95% CI):
  `random` −6.08 [−7.65,−4.50] p<0.0001; `human` −1.77 [−3.44,−0.10]
  p=0.041, same sign all three seasons; `crowd` −2.17 [−3.56,−0.78]
  p=0.0027 (survives Bonferroni). `mfru_g0` +0.13 [−0.76,+1.01] p=0.78 —
  the +0.27 seen at n=74 *shrank* with more data. `xp_bench` −0.05 p=0.60:
  the bench-order channel alone is worthless. Every gamma in ±{0.15…1}:
  |Δ| ≤ 0.41, p ≥ 0.32. Rank functionals: `p_beat` −0.61 p=0.33,
  `cvar20` −0.58 p=0.36, `q80` −0.37 p=0.56 — all flat-to-worse.
  Realised beat-the-field share is 0.95–0.99 for every non-random arm: the
  EO-weighted field proxy sits below any real XI, so that metric cannot
  discriminate — noted as a limitation, not evidence.
* **Verdict.** (a) mfru_g0's edge over max-xP is **dead**, not merely
  unproven — it shrank toward zero as n grew. (b) No implementable rank
  functional beats the mean at the weekly fixed-squad layer; combined with
  the E1 theorem, max-xP ≈ max expected rank return empirically *and*
  structurally here. (c) The model clears random by ~6 pts/gw, the
  competent-human heuristic by ~1.8, the crowd template by ~2.2
  (Bonferroni-proof) — the mandate's "minimum" tier is met with evidence;
  "strong" is met trivially because max-xP *is* the champion and no
  challenger displaced it.
* **Status.** max-xP stays the production decision rule. MFRU retained as
  measurement infrastructure only.

## E4. The site_ep benchmark is leaky — excluded

* **Hypothesis.** vaastav's archived `xP` (FPL's published ep_this) is a
  deadline-honest public benchmark.
* **Test.** spearman(xP, actual) among players who played, and the
  precision of xP ≤ 0.05 at predicting 0 minutes, per season.
* **Result.** 2023-24 / 2024-25: spearman_played 0.581 / 0.511 — at the
  repo's measured perfect-minutes-oracle ceiling (~0.59) — and 93.5% / 89.3%
  of xP≈0 rows are exact non-players. That is post-hoc knowledge of who
  played, not a deadline projection. 2025-26: spearman_played 0.074 —
  near-random among starters. The arm "beat" xp by +2.6/gw pooled
  (+5.5 in 2024-25) — an artifact of the leak, and the reason a benchmark
  that looks too good gets audited before it gets believed.
* **Verdict.** Excluded from the arms. No season-long, deadline-honest,
  legally accessible public projection archive exists for these seasons
  (E3); the forward-collected elite-manager panel remains the only honest
  external benchmark, testable ~10-15 gameweeks after collection begins.

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

---

# Phase 2 — information over objectives

Mandate: max-xP is the immutable champion; every change is a challenger
needing robust out-of-sample evidence. The identified bottleneck is minutes
uncertainty, and the ledger says the model is saturated on *historical*
features (six attempts ≤0.25% log-loss, Rounds 8/8b) — so the lever is new
information at the deadline, which must first be archived point-in-time.

## E5. Scheduled collection: the repo as the archive

* **What.** `acquire/actions.py` + `.github/workflows/collect.yml`: 4x-daily
  stdlib-only Actions runs committing to `data/collected/` — availability
  change log (P2 dataset), per-run market snapshots (deadline-decay, §8 of
  the mandate), frozen pre-deadline ownership per gw, panel picks per
  deadline. Importer replays files into the `acq_*` tables.
* **Why files, not the DB.** The SQLite is gitignored; two seasons of
  availability died with it once already. Files in git survive clones,
  diff cleanly, and timestamp themselves through commit history as a
  secondary audit trail.
* **Panel caveat (pre-registered).** The 2026-27 panel was enumerated from
  the overall table after GW2; membership is graded on past seasons only,
  but reachability through today's table is conditioned on a good GW1-2.
  Therefore: GW1-2 picks are archived but excluded from differential
  analyses; panel-vs-model evaluation starts at GW4 and needs ~10-15
  gameweeks of picks to be testable at all (E10-style power caveats apply).
* **What unlocks when.** Deadline-decay curves and the late-news change
  engine: after a few gameweeks of snapshots. Availability -> xMins
  challenger: needs roughly a season of change-log history to backtest
  honestly (the stored FPL status is only "today's"). Elite-manager
  disagreement signal: ~10-15 gws. Each will be run as champion/challenger
  with paired tests when its data exists — not before.

## Phase-2 priority gating (what is NOT being done yet, and why)

* Press conferences / journalist lineups: no free archived, timestamped
  source found (Sportmonks free tier 403s expectedLineups — audited in
  CLAUDE.md); revisit only with a source whose historical predictions are
  archived per fixture with timestamps.
* Bookmaker info beyond what ships: already measured — market encompasses
  the team model at team level, ODDS_WEIGHT insensitive at player level;
  anytime-scorer/clean-sheet player props have no free historical archive.
  Do not re-run without new data.
* Transfer-layer optimisation (§16): explicitly deferred by the mandate
  until the weekly layer is stable; the MILP already handles the
  mechanics, and decision-layer evidence says its projections are the
  binding input.

## E6. Phase 2.5 exploitation infrastructure (built ahead of the data)

Champion unchanged; these systems find where it is wrong and what
information could have corrected it.

* **Decision sensitivity** (`fpl_engine/xpts/sensitivity.py`, `sensitivity`
  CLI): per-decision margins (captain vs next armband; each starter vs his
  best legal bench swap) plus bootstrap stability over the simulator draws
  — P(the choice survives estimation uncertainty). Fragile decisions
  (margin < ~0.3 xP or unstable) are where information acquisition pays;
  robust ones cannot be changed by any news short of an injury. First live
  reading (2026-27 GW3, template): Haaland captain margin +1.03, 100%
  stable; tightest XI call Mbeumo-over-Gabriel at 0.37 — a robust week.
* **Model-error database** (`fpl_engine/errors.py`, `errors` CLI): every
  gw's per-player prediction vs realised outcome persisted in
  `model_error` with a causal-shaped classification (did_not_play /
  unexpected_appearance / under_minutes / haul_missed /
  blank_despite_minutes / ok / other). Seeded point-in-time from 2024-25 +
  2025-26 replays; the live season records after each gw. Joins to the
  availability change log on player/time — that join IS the info-value
  framework's spine (mandate §2): information event -> did the error class
  it should have prevented occur?
* **Deadline-decay pipeline** (`fpl_engine/decay.py`, `decay` CLI): grades
  every archived pre-deadline snapshot's availability field against
  realised appearances, bucketed by hours-to-deadline. Auto-populates as
  the scheduled collection accumulates over finished gameweeks; today it
  correctly reports insufficient overlap. This also floors any future
  source: a paid feed must beat the free availability field at the same
  lead time.
* **Mandate §6 (portfolio/joint XI optimisation) is already answered**:
  mfru_g0 IS the joint-distribution XI optimiser (covariance, bench
  insurance, formation interaction, per-draw autosubs) and measured +0.13
  vs max-xP, p=0.78, n=111 (E2). Not rebuilt.
* **Mandate §9**: MFRU frozen as-is; conditional questions (rank-state,
  template-heavy weeks, mini-leagues) wait for rank-state data that does
  not exist yet.
