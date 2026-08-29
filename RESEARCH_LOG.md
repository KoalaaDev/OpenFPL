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

## E6. The transfer layer, replayed (deferred by the mandate; run anyway)

* **Status of the question.** Section 16 defers transfer-layer optimisation
  until the weekly layer is stable, on the grounds that "the MILP already
  handles the mechanics". That premise had never been tested - the machinery
  was assumed correct, not scored. It is now, and the premise was wrong in
  three places (below).
* **Design.** Differs deliberately from E1-E2: the squad is NOT fixed.
  Point-in-time replay of 2024-25 + 2025-26, projections for gw *t* built with
  `as_of` = the first kickoff of the DECISION gameweek, real prices and FPL
  selling rules, FT accrual, realised points with autosubs and the vice
  fallback. Chips off. **Every policy starts from the same opening squad**, so
  the comparison is transfer policy alone.
* **Result.** Transferring beats never-transferring by **+283 to +354
  pts/season**, every arm p<0.0001. Horizon 1 vs 3 vs 5 vs 8 is **not
  resolvable**: +28 / +43 / +70 a season, p >= 0.24, against a design that
  cannot resolve less than ~120. Point estimates are monotone in horizon and
  the mechanism shows in the non-outcome columns (a longer horizon ends with a
  more valuable squad on no more hits).
* **Verdict.** Transfers: **real and replicated**. Horizon: **unproven**, and
  unprovable at this n - once two policies transfer differently they own
  different players, their weeks decorrelate, and the paired design stops
  pairing. Default horizon 5 stays.
* **Three rule defects the replay exposed**, all silent, all now fixed with
  tests: free transfers accrued +1 in a chip gameweek (FPL preserves the stock
  instead); `transfers_in`/`transfers_out` were unordered, so consumers pairing
  them by index rendered illegal moves; a Free Hit reported no changes at all,
  because it correctly pins tin/tout to zero.

## E7. Rank tilt at the transfer layer (E1/E2 one layer up)

* **Hypothesis.** E1's theorem kills gamma on the realised *mean*, but
  ownership does not drop out of the *variance*:
  `Var(Delta) ~ sum m_i sigma_i^2 (1 - 2 EO_i)`, which is linear in the
  decision variables. At the weekly layer a tilt reshuffles 11 of a fixed 15;
  driving *transfers* it changes which 15 you own, so the channel is wider.
* **Design.** E6's harness, tilt folded into the projections as
  `ep' = ep + gamma sigma^2 (1 - 2 EO)`; ownership lagged one gw, sigma
  empirical-Bayes and point-in-time. Plus a separate club-concentration sweep
  (lambda in +/-1) priced exactly in the MILP.
* **Result.** gamma -0.30 / -0.10 / +0.10 / +0.30 vs 0: **-121 / -125 / -191 /
  -509 pts/season**, p = 0.045 / 0.022 / 0.005 / <0.0001. gamma=0 also wins on
  beat-the-crowd rate (72% vs 49-57%). Margin sd does move (13.7 -> 15.3), so
  the channel is real; the exchange rate is ~6% more spread for ~180 points of
  mean. Club concentration: every arm inside noise (p > 0.32) and margin sd
  **flat** while stacking varies five-fold - FPL's own squad rules cap the
  achievable correlation.
* **Verdict.** `gamma = 0` and `concentration = 0`. Stronger than E2's bound:
  at the weekly layer a tilt is merely unprovable, here it is significantly
  **harmful** in both directions. The concentration knob ships at 0.0 only
  because it is tested and free.
* **One implementation note.** A one-sided `z >= n-1` prices a penalty
  correctly but leaves the objective unbounded once the weight goes negative;
  CBC returns garbage, not an error. Pinned with two-sided indicators, with a
  test asserting the negative arm stays bounded.

---

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

## E8. Pricing a lineup feed before buying one

* **Question.** Round 8 located the minutes error in the band where the model
  says P(start) is 0.30-0.70 - 11.5% of rows, where it is *correctly
  calibrated* because the manager has not decided. What is resolving only that
  band worth?
* **Design.** Oracle restricted to the band, at 25/50/75/100% resolution, 74
  paired gameweeks; then replayed through E6's harness with the feed modelled
  honestly - it resolves the imminent gameweek's XI and says nothing about
  GW+3, because that information does not exist.
* **Result.** Band at 100%: `spearman_played` **+0.0814**, top-30 +0.26, both
  p<0.001 - **40% of the entire minutes ceiling from 11% of rows** - and value
  is **linear** in the fraction resolved. Through the decision harness: +2.35
  pts/gw = **+89/season**, p=0.055, CI [-1, +180].
* **Verdict.** The *information* gain is proven; the *points* conversion is the
  best available estimate and hits the E6 wall (a metric over 600 players is
  well powered, the same effect through 11 squad slots is not).
* **What it changes operationally.** Because value is linear in resolution, a
  vendor is judged on its hit rate over ~90 ambiguous rows a gameweek - five
  gameweeks gives ~450 observations - not on a season-long decision backtest.
  Bar: the model's own ~55% in that band; each point above is worth roughly
  `(acc - 0.55)/0.45 x 89` points a season.
* **Negative result attached.** The obvious free substitute is not one. Crowd
  transfer flow into gw G is legally timed (corr +0.37/+0.39 with G-1 points vs
  +0.15/+0.18 with G's), but among players who did NOT start G-1 its
  correlation with starting now is **-0.002 / +0.010**. The crowd chases last
  week's hauls; it does not relay team news. `shift(1)` stays.

## Phase-2 priority gating (what is NOT being done yet, and why)

* Press conferences / journalist lineups: no free archived, timestamped
  source found (Sportmonks free tier 403s expectedLineups — audited in
  CLAUDE.md); revisit only with a source whose historical predictions are
  archived per fixture with timestamps.
* Bookmaker info beyond what ships: already measured — market encompasses
  the team model at team level, ODDS_WEIGHT insensitive at player level;
  anytime-scorer/clean-sheet player props have no free historical archive.
  Do not re-run without new data. *Update:* player props ARE reachable live —
  The Odds API serves `player_goal_scorer_anytime` / `player_shots` /
  `player_assists` for EPL at **2 credits per fixture per market** (~20 a
  gameweek against 500/month free), from six books including Pinnacle. Still no
  historical archive, so it is forward-collection only and belongs with E5.
  Polymarket was audited and rejected for the model: 18 EPL fixtures with real
  liquidity, but **every market is team level** — no goalscorer or assist
  market exists — so it is displayed, never modelled.
* Transfer-layer optimisation (§16): was deferred on the premise that "the
  MILP already handles the mechanics". E6 tested that premise rather than
  assuming it, and found three silent rule/reporting defects — so the layer is
  now scored and correct. The deferral otherwise stands: E7 shows the
  projections really are the binding input, since tilting them is harmful and
  the horizon itself is unresolvable at this n.
