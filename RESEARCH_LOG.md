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

## E9. Transfermarkt injury history — the first exogenous minutes signal

* **Why it is not attempt number seven.** The six prior sharpening attempts all
  returned <=0.25% and all were re-arrangements of data the model already had.
  Injury history is exogenous (minutes record THAT a player was absent, never
  that it was a hamstring) and, unlike availability or lineups, **dated** — so
  it clears the archive wall that stopped the other three.
* **Corpus.** 3,222 spells, 473 players, 2012-2026, scraped directly (the
  `transfermarkt-api` wrapper is unmaintained and 500s on every endpoint).
  Typed: Hamstring 371, Knee 195, Ankle 191, Muscle 177, Calf 86, Groin 76.
  569 spells fall in 2024-25 and 598 in 2025-26.
* **Result** (train on prior seasons, held-out season, ~57k player-matches):
  baseline 0.4962 / 0.4425; + currently-out −5.11% / −7.33%; + history only
  −1.39% / −1.96%; + both −7.08% / −9.25%. **History on top of currently-out:
  −2.08% in BOTH seasons.**
* **The headline is a handicapped-baseline artefact.** Replays disable the
  availability overlay, so the baseline knows nothing about who is fit; most of
  the −7/−9% re-derives what the live model already gets from FPL `status`.
  Quoting it would be E4's `site_ep` mistake in a new coat. The defensible
  number is −2.08%.
* **Leak audit.** The failure mode is an injury sustained during a match being
  credited to it. 133 spells start on a fixture day and 85% of those players
  played, so it is not driving the result. Boundary enforced in
  `xpts/injury_features.py`: ended spells fully usable, ongoing spells
  contribute only the fact of absence.
* **Verdict.** **Not shipped.** Promising and replicated, but log-loss is not
  this repo's bar. Two gates before it can: (a) score on decision metrics —
  Understat was 3.9% better at rate estimation and moved nothing; (b) check TM
  against FPL `status` where both exist, because a noisier copy of a flag we
  already read live is worth nothing live.
* **Possible second prize.** A dated historical availability record is exactly
  what E5 is collecting forward for. If it survives (a) and (b), the
  availability→xMins challenger becomes testable now rather than after a season
  of collection.

## E10. Transfer rumours — a correctness fix, priced in automatically

* **Problem.** FPL reclassifies a player only when a move completes, so the
  engine recommends players onto fixtures they will never play. Not a
  mis-rating — the wrong club.
* **Source.** Transfermarkt's PL rumour board: player, club, interested club,
  that club's league, source date, and its own assessment. 25 rumours, 15
  resolve to FPL players, 8 of 9 strong ones are exits.
* **Applied**, weighted by the assessment rather than switched at a threshold:
  `ep' = p * ep_destination + (1-p) * ep_current`, destination value zero for a
  move out of the league. Enabled at >=50%. `engine.xpts_predict_gw` gained a
  `team_override`, which is sufficient for reprojection since everything
  downstream keys off `team_id`.
* **Not a measured edge and not claimed as one.** Transfermarkt's percentage is
  a forum-sourced opinion; this ships as a correctness fix, on the same footing
  as excluding Assistant Managers from the backtest actuals.
* **Two parsing traps**, both producing plausible wrong output and both now
  tested: the club regex could not cross `"><img` to reach the title (parsed
  zero rows), and substring club matching filed "Manchester City" as *leaving
  the league* because it does not contain "Man City".

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

## E11. Transfermarkt, scored on decisions — both E9 gates close negative

E9 left injury history unshipped behind two explicit gates. This round collected
the rest of Transfermarkt's dated data, built every family the brief asked for,
and put all of it through the same forward-in-time protocol. **Nothing enters
the model.** The round's value is four silent defects and two closed questions.

### E11a. The identity join was wrong (found before any measurement)

`injury_features.spells()` resolved a Transfermarkt player to
`tm_player.player_id` and joined it to a *different* season's `player` table.
FPL reassigns element ids every summer. Measured on this database:

| joined to | shared ids | still the same footballer |
|---|---|---|
| 2022-23 | 626 | **0.3%** |
| 2023-24 | 625 | 0.2% |
| 2024-25 | 626 | 0.3% |
| 2025-26 | 626 | 0.8% |

Id 1 is Raya now, Cédric in 2022-23, Balogun in 2023-24, Fábio Vieira in
2024-25. Identity now travels on `player.code` with the Understat collision
rule, and `verify` gained the invariant (fifth of its kind).

### E11b. Two unit bugs, one of them in the shipped model

pandas keeps whatever resolution a timestamp was parsed at, and since 2.0 an
ISO8601 string parses to **microseconds** — so `astype("int64") / 86_400e9`
silently returns days/1000. In `_team_congestion` that left `days_rest`
harmless (a tree reads only the order) but turned `team_matches_14d` into a
gameweek counter: a 14 meaning 14,000 days counts every previous match of the
season. Median 20, max 37, against a true median of 1 and max of 4.

**Repairing it changes no decision** (74 paired gameweeks, fixed vs buggy):
spearman −0.0002, spearman_played +0.0005 (p=0.21), top30 −0.019 (p=0.40),
captain +0.50 (p=0.069). A correctness fix, not an improvement — consistent
with §8's six failed attempts at sharpening minutes.

### E11c. FPL publishes `birth_date`, and age is the cold-start feature

It is in the bootstrap and in vaastav's `players_raw.csv` from 2024-25 onward,
and the pipeline discarded it. Age separates the two kinds of player the
trailing features *cannot* tell apart — among men with under five career
Premier League appearances every history feature is identical:

| age | n | P(60+) |
|---|---|---|
| 15-18 | 394 | **0.000** |
| 18-20 | 1456 | 0.027 |
| 20-22 | 1205 | 0.159 |
| 22-24 | 1036 | 0.343 |
| **24-27** | 1425 | **0.401** |
| 27-30 | 910 | 0.382 |
| 30-40 | 845 | 0.351 |

**But FPL's own column does not replicate, and the reason is coverage, not
noise.** FPL began publishing in 2024-25, so coverage runs 56% / 60% / 88% /
99% across seasons: a tree left to learn the missing branch learns it on a
training population that has all but vanished by serve time. Log-loss vs
baseline on the cold-start segment, three seeds, two held-out seasons:

| arm | 2024-25 | 2025-26 |
|---|---|---|
| coverage indicator only (falsification) | −0.74% | −1.75% |
| FPL age, raw | −2.22% | **+0.13%** |
| FPL age, position-median imputed | −1.36% | **−0.50%** |
| Transfermarkt age | **−3.73%** | **−3.98%** |
| TM-filled + imputed + coverage flag | −3.46% | −3.79% |

Only the Transfermarkt-covered variants replicate, and **the two sources carry
the same dates**: median disagreement 0.0000 years, 0.12% differ by more than
30 days. Transfermarkt is not a better signal here, it is coverage for the
seasons FPL had not started publishing. FPL's column becomes sufficient on its
own once 2024-25 and later are the training seasons.

### E11d. Every family, on log-loss — and then on decisions

Log-loss vs baseline, `age` = the imputed column, three seeds:

| arm | all 24-25 | all 25-26 | cold start 24-25 | cold start 25-26 |
|---|---|---|---|---|
| age + TM transfers | −0.95% | −1.16% | −6.20% | −7.33% |
| age + TM market value | −0.42% | −0.51% | −5.18% | −5.81% |
| age + TM squad depth | −0.51% | −0.64% | −4.51% | −4.89% |
| **age + every TM family** | **−1.16%** | **−1.68%** | −6.57% | −8.62% |
| age + injury flag | −9.59% | −9.93% | −8.22% | −7.65% |
| age + injury flag + history | −11.00% | −11.31% | −9.11% | −8.52% |
| everything | −12.08% | −13.08% | −13.85% | −14.51% |

Injury history on top of the flag is **−1.5%** in both seasons (E9 reported
−2.08% on the corrupted identity join; the shape survives, the size shrinks).

Now the decision metrics, 74 paired gameweeks, `data/bt_*` arms:

| arm vs | spearman | spearman_played | prec@20 | top30 | captain | rmse |
|---|---|---|---|---|---|---|
| age — baseline | +0.0008*** | **−0.0011*** | +0.003 | +0.02 | −0.27 | +0.000 |
| age+TM — baseline | +0.0030*** | −0.0007 | +0.001 | −0.01 | **−0.49*** | −0.0025*** |
| history — flag | +0.0026*** | −0.0013 | −0.003 | −0.00 | +0.11 | −0.0022*** |
| everything — flag | +0.0045*** | −0.0022 | −0.001 | +0.00 | +0.11 | −0.0039*** |

The pattern is E11's whole result and it is the Understat pattern exactly:
`spearman` and `rmse` improve significantly because the model got better at
ranking **who plays at all** — which is what age, transfer recency and injury
history inform — while every metric that decides a squad sits still or drifts
negative. In the opening gameweeks injury history is significantly *worse*
(`spearman_played` −0.0042, p=0.004).

### E11e. Gate (b): the Transfermarkt flag is a weaker copy of FPL `status`

On the live season, 596 of 626 players mapped:

| | FPL says out | FPL says available |
|---|---|---|
| **TM says out** | 38 | 3 |
| **TM says available** | **81** | 474 |

**31.9% recall, 92.7% precision.** FPL flags 119; Transfermarkt catches 38.
The misses are structural, not noise: FPL's `status` also carries suspensions
(`s`), players who have left or are unregistered (`u`) and doubts (`d`), none
of which an injury table can see. So Transfermarkt's flag adds nothing live —
and, more importantly, it was the *control* in E9's −2.08%. A 32%-recall
control makes that figure an upper bound on the live gain, not an estimate.

### Verdict

**Nothing ships into the model.** Both E9 gates close negative: injury history
moves no decision (a), and its availability channel is a strictly weaker copy
of a flag already read live (b). The TM families are 1.2-2.0% better at
log-loss on top of everything else and change no decision either.

What ships is the data layer and the repairs: three dated datasets, identity on
`player.code`, the fifth `verify` invariant, `birth_date` ingest, the two unit
fixes, a 270x faster injury builder, and tests for every parsing trap. The
optional blocks stay switchable through `$FPL_MINUTES_EXTRA` so the experiment
is reproducible rather than described.

**One number worth keeping.** Against a baseline denied availability entirely,
the flag alone is worth +0.0085 `spearman_played`, +0.011 prec@20 and **+0.134
top-30 pts/pick**, all p < 0.003. That is the first *decision-metric* price
this repo has been able to put on availability at all (§8: "no replay here can
measure it"), and since FPL's live `status` is strictly stronger, it is a lower
bound on what the availability overlay is already earning.

## E12. The Tactics/Manager expert — rejected, except the part that is not tactics

Pre-registered question: *does manager/tactical context provide independent
predictive information the model is currently missing?* Built as one isolated
expert with six separable families, tested incrementally, on genuinely
held-out seasons. **The managerial answer is no.**

### Sources, and what does not exist

| source | verdict |
|---|---|
| **StatsBomb open data** | Premier League **2003/04 and 2015/16 only** — checked against `competitions.json`. Neither overlaps a replayed season. No event-level tactical data is reachable. |
| **Transfermarkt staff history** | 927 dated managerial spells across the 27 clubs in this database: appointment date, departure date. Usable. |
| **Understat** | PPDA and deep completions per team-match (3,078 rows), and the per-match ROLE a player occupied (40,116 rows) — the only free per-match role feed. Pulled for every backfilled season for the first time; the pipeline had only ever fetched the current season and one before. |

So possession, field tilt, crossing frequency, attacking width and build-up
style have **no free per-match history here**. PPDA and deep completions are
the two style axes that exist. Approximating the rest from articles about
managers would be inventing data, so it was not done.

### Family ablation, three seeds, two independently held-out seasons

Log-loss vs baseline (negative is better):

| family | 2024-25 | 2025-26 | verdict |
|---|---|---|---|
| manager identity, tenure, continuity | **+0.61%** | **+0.38%** | worse in both — reject |
| formation (shape, stability, changes) | **+0.23%** | **+0.25%** | worse in both — reject |
| playing style (PPDA, deep, opponent's) | −0.07% | −0.13% | ~zero — reject |
| manager x opponent | −0.09% | +0.03% | ~zero — reject |
| manager x player role | −0.85% | −1.21% | replicates |
| **the player's own line** | **−1.31%** | **−1.42%** | replicates |
| the whole expert | −1.08% | −1.51% | |
| **on top of everything else** | **−0.88%** | **−1.45%** | replicates |

### The falsification localises it, and it is not managerial

| decomposition | 2024-25 | 2025-26 |
|---|---|---|
| coverage indicator alone | −0.38% | −0.13% |
| squad competition (slots, share of the line) | −0.31% | −0.19% |
| **which line he plays** | **−1.26%** | **−1.25%** |
| both together | −1.31% | −1.42% |
| + coverage indicator on top | −1.40% | −1.34% |

Competition for the shirt is worth barely more than the coverage indicator.
The whole effect is **which line he actually plays**, and `manager x role` only
worked because it partly re-encodes the same fact. This is not a discovery
about managers. It is that **FPL's four-way position label is too coarse**: it
calls a DMC and an AMC both "MID" and their minutes differ.

### A defect worth recording, because it cost points at the top of the board

The first version coded an unresolved role `0`, which asserts "he is neither an
attacking nor a defensive midfielder" rather than "we have not seen his line".
Understat resolves ~65% of players and names a role only for a starter, so a
third of rows carried that false denial. Decision metrics, 74 paired gameweeks:

| | unknown coded 0 | unknown coded NaN |
|---|---|---|
| spearman_played | +0.0055*** | +0.0046*** |
| **top11** | **−0.141 (p=0.008)** | −0.050 (p=0.39) |
| captain | −0.473 (p=0.084) | −0.351 (p=0.13) |

Significant harm to the starting XI, in both seasons independently, from a
missing-value convention. Generalise it: **an absent observation is not a
negative one**, and a tree will happily learn the difference if you let it.

### Decision metrics for the survivor — and what ships

Three features (`role_is_am`, `role_is_dm`, `role_vs_fpl_line`) reproduce the
whole family, so that is the shipped unit. 74 paired gameweeks vs baseline:

| | pooled | 2024-25 | 2025-26 |
|---|---|---|---|
| spearman | +0.0024*** | +0.0028*** | +0.0020*** |
| **spearman_played** | **+0.0047 (p=0.0007)** | +0.0039** | +0.0055** |
| rmse | −0.0049*** | −0.0042*** | −0.0056*** |
| p_at_20 | +0.0020 | +0.0068 | −0.0027 |
| top11 / top30 | −0.055 / +0.029 | +0.010 / +0.022 | −0.120 / +0.037 |
| captain | −0.351 | −0.351 | −0.351 |

**The captain column is three observations, not a finding.** The pick changes
in 3 of 74 gameweeks (one in 2024-25 worth −13, two in 2025-26 worth −13
between them), and the identical −0.3514 in both seasons is arithmetic
coincidence. It carries no information either way.

**Shipped**, into `minutes_model.FEATURES`. `spearman_played` is the metric
§8b used to ship the hybrid E[minutes] estimator (+0.0010, p=0.010) and to
reject isotonic calibration (−0.0012); this is **4.6x** the size of the change
that earned its place, significant in each season separately. Stated honestly:
it is a rank-quality gain and **not** a proven points gain — top11, top30 and
prec@20 are all null. It is NaN without `pull --understat`, which the
classifier tolerates by behaving as it did before.

### The standing wall this removes, and the one it leaves

CLAUDE.md listed manager identity as unreachable. It is reachable, dated, and
**worth nothing** — which is a better outcome than leaving it as an open
question. What remains untestable is the interaction the brief cared most
about: with no event data overlapping a replayed season, "does this manager
play inverted wingers differently" cannot be asked here at all.

### E12a. A sixth identity defect, found by the invariant it was written for

Pulling Understat for every backfilled season (which the pipeline had never
done) tripped `identity.stable_across_seasons`. FPL's **Amad Diallo** resolved
to Understat **8127** ("Amad Diallo Traore", Manchester United) in two seasons
and to **12200** ("Amadou Diallo", Newcastle United) in three — a different
footballer, whose shots and match roles were attached to him across most of the
database. The fuzzy name pass fired before the club check and matched the wrong
man; the cross-season fill only touches NULLs, so nothing reconciled the two.

**There is no safe automatic tiebreak.** The wrong id won on the number of
seasons (3 vs 2) *and* on stored match volume (1 row vs 0), so both obvious
rules pick the impostor. `pipeline.reconcile_understat_ids` therefore unsets
every season for a code that claims more than one id and reports it — the
resolver's existing collision rule, applied in the other direction. An
`entity_override` row remains the way to pin a case a human has checked.

Worth noting where this sits: the invariant was added in Round 9 after four
identity defects, and has now caught its second and third (the Transfermarkt
`player_id` join in E11a, and this). That is the argument for the `verify`
command in one line.

## E13. Where the upgrade comes from — every component, replayed against truth

Asked directly rather than inferred: replace ONE component of the engine with
what actually happened, leave everything else alone, and score it on the
decision metrics. `xpts_predict_gw` gained `minutes_override` and `oracle`,
both None on every shipped path (asserted bit-identical on a real gameweek).
74 paired gameweeks; the minutes row reproduces §8's known +0.21, which is the
harness's calibration check.

| perfect knowledge of | spearman_played | p@20 | top11 | top30 | captain |
|---|---|---|---|---|---|
| everything | +0.605 | +0.760 | +7.30 | +5.28 | +10.18 |
| **attack (goals+assists)** | **+0.228** | +0.457 | **+6.36** | +4.35 | **+9.46** |
| goals | +0.134 | +0.400 | +5.68 | +3.80 | +8.72 |
| **bonus** | +0.130 | +0.478 | **+5.49** | +3.62 | +5.97 |
| clean sheets | +0.184 | +0.077 | +2.35 | +2.25 | +0.91 |
| assists | +0.099 | +0.087 | +2.54 | +1.66 | +4.00 |
| **minutes** | **+0.200** | +0.038 | +0.76 | +0.58 | +0.34 |
| 60-minute class only | +0.170 | +0.025 | +0.58 | +0.50 | +0.39 |
| appearance points | +0.149 | +0.012 | +0.46 | +0.43 | +0.36 |
| DefCon | +0.057 | +0.023 | +0.70 | +0.47 | −0.09 |
| conceded | +0.074 | +0.005 | +0.07 | +0.25 | 0.00 |
| cards | +0.046 | +0.003 | +0.02 | +0.08 | 0.00 |
| saves | +0.012 | +0.012 | +0.09 | +0.11 | 0.00 |
| **availability (who sits out)** | **+0.000** | +0.009 | +0.28 | +0.23 | +0.26 |

### Three findings

**1. §8's "nothing else is worth a fraction of that" was never tested.** It is
right on rank — minutes at +0.200 is second only to attack at +0.228 — and
wrong on points by a factor of eight: +0.76 against +6.36 points per pick.
Minutes is the largest REACHABLE lever, not the largest lever. The claim in
CLAUDE.md is corrected.

**2. `spearman_played` cannot see availability at all.** Perfect knowledge of
who does not play scores **exactly +0.0000** on it — by construction, since it
only moves predictions for players the metric excludes — while being worth
+0.28 points per pick. (Consistency check: E11's Transfermarkt injury FLAG did
move `spearman_played` +0.0085, because it also reshuffles the men who did
play through their exposure. The pure availability oracle does not.) Any test
of an availability signal on that metric alone is blind to its own channel.

**3. Roughly 5.6 of the 7.3 points-per-pick ceiling is not knowable before
kickoff.** Attack, bonus and clean sheets dominate the total and none of them
can be resolved by any pre-deadline information; the reachable components sum
to under 2. **The free-data points ceiling is close to exhausted.** The one
paid lever remains a predicted-lineup feed, priced at ~+89/season in E8 — and
the decomposition sharpens what to buy: the 60-minute class carries **85% of
the minutes rank gain and 77% of its points gain**, so the product to price is
"does he start and last an hour", not "how many minutes".

### Two rejections re-tested on the metric that could finally see them

The decomposition prices the clean-sheet channel at +2.35 points per pick and
DefCon at +0.70. Both had been dismissed on metrics that under-weight them —
`ODDS_WEIGHT` on `spearman_played`/top30, DefCon at "0.011 pts per
player-gameweek". Re-run on the full metric set:

| arm | spearman_played | top11 | top30 | captain |
|---|---|---|---|---|
| `ODDS_WEIGHT` 0 vs 0.85 | −0.0016 | −0.036 | −0.021 | −0.46 |
| `ODDS_WEIGHT` 0.5 vs 0.85 | +0.0001 | +0.007 | −0.002 | +0.12 |
| `ODDS_WEIGHT` 1.0 vs 0.85 | −0.0002 | +0.030 | +0.005 | +0.05 |
| DefCon rate x1.13 | +0.0001 | −0.015 | −0.018 | 0.00 |
| DefCon rate x1.26 | +0.0002 | +0.014 | −0.007 | 0.00 |

Nothing significant. Odds lean the right way — switching them off costs 0.036
points per pick and half a captain point in both seasons — but not resolvably.
**Both standing conclusions survive.**

*A methodological trap, recorded because it produced a perfect null.* The first
odds run returned **exact zeros on every metric for every weight**. This
database carried odds for the live season only, so `fixture_odds_map` was empty
and every weight was really zero. An A/B of a parameter with no data behind it
is indistinguishable from a parameter that does not matter — and the zeros are
the tell, since a real null is noisy. `ingest_football_data` over the backfill
seasons was the fix.

### What this says to do next

Nothing in the engine. The three components worth most are luck; the reachable
ones are worth under 2 points per pick between them and two of the three have
now been re-tested and held. The next real gain is an external information
source for the 60-minute class, and E8 has already priced it.
