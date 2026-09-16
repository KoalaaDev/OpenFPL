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

## E14. The ceiling, split: estimator error vs irreducible variance

E13's oracle substitutes what ACTUALLY HAPPENED, so it measures clairvoyance —
+6.36 points per pick for knowing this week's goals. That is not a work item,
because nobody can know it, and it does not answer the question the repo needs
answered: **how good could an estimator get?**

So this substitutes a RATE instead of an outcome: each player's
leave-one-gameweek-out season rate, computed with hindsight across the whole
season but with the gameweek being predicted removed. It knows his true rate
and nothing about the match. Three tiers:

    baseline --(estimator error)--> rate oracle --(variance)--> outcome oracle

74 paired gameweeks, both seasons:

| perfect ESTIMATE of | spearman_played | top11 | top30 | rmse |
|---|---|---|---|---|
| attacking rates (season, shrunk) | −0.0017 | −0.02 | −0.07 | +0.0058*** |
| all player rates | −0.0002 | +0.06 | −0.04 | +0.0047** |
| team lambda | **+0.0000** | −0.02 | −0.00 | −0.0012 |
| everything | −0.0006 | −0.01 | −0.08 | +0.0032* |
| DefCon rate | **+0.0014**** | −0.00 | −0.01 | −0.0006* |
| *(for scale) attacking OUTCOME* | *+0.134* | *+5.68* | *+3.80* | *−0.465* |

**A perfect rate estimate is worth nothing.** Not "a little" — nothing, and on
`spearman_played` and rmse it is significantly worse, while knowing the outcome
is worth +5.68 points per pick. The attacking ceiling is **irreducible
match-to-match variance, not estimator error.**

That single measurement explains four earlier results at once: Understat's
3.9%-better rates moving nothing, the 24-variant constant sweep moving nothing,
the set-piece decomposition moving nothing, and adaptive shrinkage moving
nothing. All four were improving an estimator already at the variance-limited
optimum.

### The local arm closes the form question too

A season rate is stationary by construction, so if a player's true rate really
moves — a role change, a genuine hot streak — a season oracle cannot see it and
would understate the ceiling. So the same oracle was rebuilt over a window
centred on the gameweek (±4, excluding it):

| arm | spearman_played | top11 | captain | rmse |
|---|---|---|---|---|
| perfect LOCAL attacking rate | **−0.0050**** | −0.11 | −0.81 | +0.0104*** |
| perfect LOCAL rates, all | **−0.0061**** | −0.20 | −0.80 | +0.0109*** |

**Significantly worse.** Even with perfect hindsight, a local rate degrades the
model, because a ±4-gameweek window of goals is dominated by sampling noise.
There is no exploitable non-stationarity: "form" at this resolution is noise,
and chasing it is harmful even when you know it exactly. This is the strongest
available statement of a result the repo had only ever seen indirectly.

*A trap worth recording.* The first version used the RAW leave-one-out rate and
made the model clearly worse (`spearman_played` −0.0084***, rmse +0.0158***).
That is not a finding about rates: a striker's 10 goals in 25 90s carries a
sampling sd of ~0.13 on a 0.40 rate, so the "oracle" was a NOISIER estimator
that merely happened to see the future. Shrinking it with the same
empirical-Bayes constant the shipped estimator uses is the fair test — and the
gap between the two rows is a clean measure of how much the shipped shrinkage
is worth.

### What is left, and the collector that follows from it

Every rate channel is closed and the team-lambda channel scores exactly
+0.0000. The only component with reachable headroom is whether a player starts
and lasts an hour — and E8 established the model is already CORRECTLY
calibrated in the band where it is unsure, because the manager has not decided.
Only an outside forecast resolves it.

`acquire/sources/predicted_lineups.py` collects one: RotoWire publishes
confirmed and predicted Premier League XIs, server-rendered, free, with a
per-player position in the same vocabulary Understat uses. Each side carries
its own status, and the distinction is load-bearing — a CONFIRMED XI lands
about an hour before kickoff, i.e. AFTER the deadline, so it is ground truth to
score forecasts against and never an input. Conflating the two would
manufacture an oracle out of a feed.

Nobody archives past predictions, so this cannot be backtested from history;
forward collection through the scheduled Action is the only route, and E8b says
it becomes priceable after ~5 gameweeks (~90 ambiguous rows each, ~450
observations) rather than the season-plus a decision backtest would need. The
bar is the model's own ~55% in that band, and each point above it is worth
roughly `(accuracy − 0.55)/0.45 × 89` points a season.

## E15. The forward tests at 2026-27 GW3 — what is runnable, and the first lineup-feed reading

* **Question.** Three collectors were started so that champion/challenger
  tests could run once their data existed (E5, E8, the manager panel). With
  results through GW3 (8 of 10 GW3 matches at the time of writing), which of
  them can be run, and what does the first reading say?
* **Inventory, checked against the files rather than assumed.**
  - Predicted lineups (RotoWire): collection began 2026-08-31, so **GW3 is the
    first gameweek with a pre-deadline forecast** — 19 of 20 clubs had one
    before the 17:30Z cutoff. One gameweek of the ~5-10 E8b asks for.
  - Elite-manager panel: GW3 picks are the **first out-of-sample gameweek**
    (the panel was enumerated after GW2, so GW1-2 are conditioned). 10-15
    needed. Not testable.
  - Availability change log: 1,606 observations back to 2026-07-26 via the
    raw snapshots; the availability->xMins challenger needs a season. Not
    testable. It DID serve as the point-in-time overlay for this test.
  - Market snapshots (deadline decay): 32 snapshots over GW3-4. Not testable.
* **Three defects had to be fixed before any number could be trusted**, all
  silent, all found by reading the GW2 post-mortem (see CLAUDE.md, "Three
  silent live-season defects"): the cached minutes regressor was loading with
  no intercept (xgboost 3.2-written file read by 3.0.2 — E[min | plays] of 23
  for Haaland; the whole live board deflated ~4x), a pull during Monday's
  kickoff had written phantom 0-minute GW2 rows for 29 Arsenal/Villa players,
  and the lineup archive filed every post-deadline GW3 XI under GW4 because it
  used FPL's `is_next`. Guards: a self-check probe in the model cache, a
  finished-fixture filter + `verify` invariant, and kickoff-derived labelling.
  FPL's `finished` flag was also observed lagging full time by two days;
  `finished_provisional` is now the boundary between a result and a match in
  progress.
* **Design of the reading** (`python -m fpl_engine lineup-feed --gw 3`,
  `fpl_engine/lineup_feed.py`). Forecast = last *predicted* XI observed
  strictly before first kickoff - 90 min. Model = the shipped minutes model
  re-run as of first kickoff with the availability overlay taken from the
  change log **as it stood at the deadline** (652 rows), not today's status.
  Truth = `player_gw.starts` for the 16 clubs whose fixture had finished
  (falls back to the confirmed XI until results are pulled; the 12-club
  confirmed-XI reading earlier the same day gave 0.750 vs 0.583 on n=36). All
  RotoWire names resolved; none dropped.
* **Result, GW3 only (n = 46 rows in the 0.30-0.70 band, 533 overall).**

  | | n | feed acc | model acc (p>=0.5) | Brier feed / model |
  |---|---|---|---|---|
  | ambiguous band | 46 | **0.761** [0.62, 0.86] | 0.609 | 0.239 / 0.230 |
  | all rows | 533 | 0.925 | 0.917 | 0.075 / 0.066 |
  | XI precision (16 clubs) | | feed 0.886 | model top-11 0.881 | |

  Where the two disagree in the band (15 rows) the feed is right 73% of the
  time. On the E8b line that is **+41.7 points a season [+14, +62]** — and the
  model's own band accuracy of 0.61 is consistent with E8's ~0.55.
* **Verdict: promising, not priceable.** One gameweek; 46 band rows against
  the ~450 the pre-registration asks for (the band is ~45-60 rows a gameweek
  once availability zeroes the ruled-out, not the ~90 E8b estimated, so budget
  ~8-10 gameweeks). The Brier scores say the model's *probabilities* are
  still slightly better calibrated than the feed's hard 0/1 in the band, so
  the eventual integration is a blend, not a replacement. Nothing is wired
  into the engine. The reading accumulates in `data/lineup_feed_2026-27.json`
  with row-level records, so the pooled estimate is exact each week.
* **Also run.** GW2 post-mortem on the repaired cache and completed results:
  predicted 895 vs actual 891 (100%), Spearman 0.698, model captain Bruno
  Fernandes = the week's top scorer (23).

## E16. Leaky defences: are defenders on easy fixtures over-projected when their own club keeps conceding?

* **Hypothesis (owner's, pre-registered before any number).** The engine
  takes a defender on an easy fixture whenever his own numbers are good,
  but a club that has *shown* it concedes should make that defender worse
  than projected however good his stats are. Test it on the scoreline
  record, not on xG.
* **Where leakiness already enters.** P(CS) = P(60+) x exp(-lambda_against)
  and E[conceded] is a Poisson floor-division on the same lambda, where
  lambda_against is the market's implied goals for the opponent (85%)
  blended with the team model's rate, whose defence rating is fitted on
  realised goals against blended 50/50 with xGA over a 180-day half-life. So
  the question is whether that lambda is CALIBRATED with respect to the
  club's own record, not whether the record is used.
* **Design.** 2024-25 + 2025-26, replayed point-in-time on the shipped
  engine (no Understat, no OpenFPL arm; bit-identical elsewhere). Trailing
  scoreline features per club, 10 matches, cross-season by `team.code`,
  shrunk toward the league mean by 3 pseudo-matches: goals against (`ga`),
  clean-sheet share, share conceding 2+ (`two_plus`, the scoreline tail),
  goals against minus xGA (`ga_xga`, "concedes more than the chances").
  `xpts/leaky.py`; `research/leaky_defence.py`.
  Diagnostics on 7,069 GK/DEF single-fixture rows who played 60+:
  D1 calibration of P(CS | 60+) by decile and inside leakiness terciles;
  D2 logistic regression on realised CS with the engine's own logit as an
  OFFSET (so a coefficient is what leakiness adds beyond lambda), standard
  errors clustered by gameweek; D3 the decision view, the 10 highest-
  projected GK/DEF each gameweek, projected vs realised, split leaky x easy.
  Arms, each scaling lambda_against for the club's DEFENSIVE components
  only (its opponent's attack untouched), paired over 74 gameweeks through
  the standard backtest with new defender metrics (`def_top5`, `def_top10`,
  `def_spearman_played`): `goals_def` (defence fitted on realised goals,
  xGA ignored), `ga` at alpha 0.5 / 1.0 (lambda x (ga/league_ga)^alpha),
  `tail` (lambda x (1 + two_plus - league)). Four arms, alpha 0.05/4.

* **D1: the clean-sheet probability is calibrated, including by leakiness.**
  Slope on the engine's own logit **0.997** (se 0.13), intercept -0.01.

  | own-club tercile | n | trailing GA/match | predicted P(CS) | realised | gap |
  |---|---|---|---|---|---|
  | solid | 2,358 | 1.11 | 0.294 | 0.292 | -0.002 |
  | mid | 2,363 | 1.46 | 0.250 | 0.264 | +0.014 |
  | leaky | 2,348 | 1.91 | 0.207 | 0.190 | **-0.016** |

  The leaky tercile keeps 1.6 percentage points fewer clean sheets than
  projected, on a base of 20.7%: about 0.06 points per defender-gameweek at
  4 points a clean sheet. Inside P(CS) quartiles the sign flips around
  (leaky is -0.024 / +0.028 / -0.077 / +0.020 from the lowest quartile up),
  so it is not a monotone bias.
* **D2: nothing the record adds survives clustering.** Coefficients on a
  standardised feature, offset = engine logit, SE clustered by gameweek:

  | covariate | coef | z (naive) | z (clustered) |
  |---|---|---|---|
  | goals against | -0.049 | -1.64 | -0.86 |
  | share conceding 2+ (the scoreline tail) | -0.060 | -2.03 | -1.12 |
  | clean-sheet share | -0.009 | -0.32 | -0.18 |
  | GA minus xGA | **+0.064** | +2.20 | +1.07 |
  | GA x easy-fixture interaction | +0.168 | +2.70 | +1.38 |
  | joint: GA / tail / GA-xGA | -0.096 / -0.073 / +0.167 | | -0.68 / -0.58 / +2.23 |

  Two things worth keeping. The naive z-scores look like a finding and the
  clustered ones do not: defenders of one club in one gameweek share a
  scoreline, so 7,069 rows are a few hundred effective observations. And
  the one term that is even borderline runs the OTHER way: a club that has
  been conceding more than its xGA keeps *more* clean sheets than lambda
  says, i.e. finishing luck against it reverts. "Time has shown it likes to
  concede" is, at that margin, the trap rather than the signal. Where the
  goals-against term is negative at all it is in HARD fixtures (main effect
  -0.155, interaction +0.168, net ~0 on easy ones), the opposite location
  to the hypothesis.
* **D3: the engine rarely picks them, and when it does they score.** Of 740
  top-10 GK/DEF picks over 74 gameweeks, **94 (12.7%)** were from a club
  conceding above the league rate.

  | picks | n | projected | realised | gap |
  |---|---|---|---|---|
  | solid club | 646 | 4.41 | 4.31 | -0.10 |
  | leaky club | 94 | 4.18 | 4.26 | +0.08 |
  | leaky club, easy fixture | 25 | 4.24 | 4.52 | +0.28 |
  | solid club, easy fixture | 294 | 4.47 | 4.24 | -0.23 |

  Paired per gameweek, leaky picks' error minus solid picks' error: +0.24,
  p = 0.65 (43 gameweeks with both). The cell the hypothesis names,
  leaky x easy, is 25 picks and over-performed. If anything is over-
  projected in easy fixtures it is the SOLID clubs' defenders (-0.23 on
  n = 294, 1.3 standard errors, noise).
* **Arms (74 paired gameweeks vs the shipped engine).**

  | arm | spearman_played | top30 | captain | rmse | def_top5 | def_top10 | def_spearman_played |
  |---|---|---|---|---|---|---|---|
  | goals_def (defence on realised goals) | -0.0001 | -0.008 | 0.00 | +0.0002* | +0.05 | -0.03 | -0.0004 |
  | ga alpha 0.5 | -0.0018 | -0.028 | -0.24 | +0.0019 | -0.15 | -0.11 | -0.0006 |
  | ga alpha 1.0 | **-0.0056** (p=0.010) | -0.049 | -0.28 | **+0.0075*** | -0.28 (p=0.055) | **-0.19** (p=0.049) | -0.0066 |
  | tail (2+ conceded share) | -0.0021 | -0.024 | -0.20 | +0.0018 | -0.11 | -0.10 | -0.0022 |
  | xga_def (post-hoc, defence on xGA alone) | +0.0002 | +0.018 | +0.04 | -0.0002 (p=0.024) | +0.04 | +0.02 | +0.0002 |

  Every scoreline arm is flat to worse on every metric, in both seasons
  separately, and the effect is monotone in the strength: alpha 1.0 costs
  0.19 points per defender pick and 0.006 of rank quality among players who
  played, both at the edge of significance and both the wrong sign. Fitting
  the defence on realised goals instead of the goals/xGA blend changes
  nothing (rmse +0.0002 is significant and negligible). The post-hoc xGA arm,
  run after D2 to check the reverted sign, leans the way D2 said, positive on
  every metric in the pooled set, but only rmse reaches p < 0.05 and the
  points metrics are +0.02 per pick at p 0.13-0.19: the team model is 15%
  of lambda, so the most it can move is small, and it is not shippable on
  this evidence. Direction noted, not acted on.
* **Verdict: rejected, on the metric that could see it.** The engine is
  not over-projecting leaky clubs' defenders on easy fixtures. Their P(CS)
  is calibrated to within 1.6 points, they are 13% of its defender picks,
  those picks score as projected, and pushing lambda toward the scoreline
  record costs points monotonically. The mechanism is the standing one:
  lambda_against is 85% bookmaker, and the bookmaker has watched the same
  scorelines. What the record adds beyond a market price is the part the
  market has correctly discounted, luck.
* **What the owner is seeing, then.** A defender on an easy fixture from a
  club conceding 1.9 a match is projected with P(CS) around 0.19-0.21, i.e.
  about 0.8 clean-sheet points, against 1.2 for a solid club's defender.
  His own attacking and DefCon rates can legitimately outweigh that 0.4, and
  D3 says when they do the pick pays. The recommendation to read is his
  `p_cs` next to his `prediction`, which the projections table already
  shows; the number is honest.
* **Status.** Nothing ships to the engine. `defence_leak` stays as a research
  hook (None on every shipped path, pinned by `tests/test_leaky.py`); the
  defender metrics stay in the backtest harness because the DefCon result
  showed a defensive change can hide inside board-wide points per pick.

## E17. "The model struggles to pick defenders": where, by how much, and ten hypotheses

* **Owner's observation.** The engine's defender picks feel weak. Ten
  hypotheses requested, each backtested.
* **Premise, measured first.** Rank quality among players who played and
  points per pick of the engine's top-10 by position, 74 gameweeks, against
  the naive baselines on the same rows:

  | position | engine spearman_played | engine top-10 pts/pick (projected) | realised best-10 | ppg baseline | trail-4 baseline |
  |---|---|---|---|---|---|
  | DEF | 0.324 | 4.24 (4.31) | 9.27 | 0.21 / 3.0 | 0.20 / 3.3 |
  | MID | 0.417 | 5.15 (5.32) | 11.09 | 0.34 / 3.9 | 0.34 / 4.2 |
  | FWD | 0.477 | 4.41 (4.44) | 7.28 | 0.42 / 3.9 | 0.40 / 3.9 |
  | GK | 0.140 | 3.49 (3.50) | 5.21 | 0.06 / 2.5 | 0.07 / 3.2 |

  So the engine ranks defenders worse than midfielders and forwards, and
  that is true of every predictor here: defenders are a harder position,
  because a 4-point clean sheet is a coin toss decided by eleven other
  people. Against the naive rules the engine's defender edge is the LARGEST
  of any position (+0.11 to +0.12 Spearman, +0.9 to +1.2 points per pick),
  and its top-10 projections are honest (4.31 projected, 4.24 realised).
* **Where the defender error lives.** Modelled vs realised points per
  component, DEF, from the engine's own `c_*` columns:

  | component | near-certain starters (P(60+) > 0.85, n = 2,394): model / real / ratio | top-10 picks: model / real |
  |---|---|---|
  | goals | 0.302 / 0.246 / **0.81** | 0.516 / 0.349 |
  | assists | 0.158 / 0.199 / **1.26** | 0.272 / 0.320 |
  | clean sheet | 0.919 / 0.954 / 1.04 | 1.390 / 1.427 |
  | conceded | -0.456 / -0.419 / 0.92 | -0.258 / -0.278 |
  | bonus | 0.205 / 0.206 / 1.01 | 0.346 / 0.326 |
  | DefCon | 0.263 / 0.281 / 1.07 | 0.321 / 0.311 |
  | appearance | 1.841 / 1.854 / 1.01 | 1.873 / 1.828 |
  | total | 3.076 / 3.134 / 1.02 | 4.308 / 4.235 |

  Two calibration defects and one non-defect. Defenders score **19% fewer
  goals than their xG says** and this is not a defender thing: midfielders
  and forwards convert at 0.89 of their modelled xG too, and every position
  is under-projected on assists (DEF 1.26, MID 1.07, FWD 1.30). Among the
  top-10 defender picks the goals gap widens to a third, which is the
  winner's curse on a noisy rate. Everything else is calibrated: P(60+) for
  defenders to within 1.5 points in every band, clean sheets to 4%, bonus
  and DefCon to within 7%. Share of the top-10 pick error variance: clean
  sheet **39%**, goals 21%, bonus 16%, assists 8%, everything else under 5%.
  Home clean sheets read 1.6 points high (0.285 vs 0.269) and away 1.8 low
  (0.215 vs 0.233), n ~ 2,800 each.
* **Pre-registered arms** (written before any ran; `research/defender_arms.py`;
  10 arms against one baseline, alpha 0.05/10 = 0.005; judged on
  `def_top5`, `def_top10`, `def_spearman_played` first and the board-wide
  metrics second; the 2025-26 GW12 no-op check confirmed every arm moves
  defender projections):

  | arm | hypothesis | mechanism |
  |---|---|---|
  | H1 `xg_cal` | finishing calibration: xG90 and xA90 scaled by the position's point-in-time realised/expected ratio | `rates.calibrate_by_pos` |
  | H2 `def_att_exp` | a defender's goals are set pieces, which scale less with the fixture | attack scaler ** 0.5 for DEF |
  | H3 `conc_emin` | conceded goals count while on the pitch: Poisson mean on E[min]/90 not P(plays) | `conceded_exposure` |
  | H4 `cs_nb` | team goals are overdispersed (var/mean 1.078): P(0) from a negative binomial | `cs_dispersion` 0.056 |
  | H5 `bonus_dc` | DefCon actions earn BPS: crossings enter the bonus regression | `rates.bonus_defcon` |
  | H6 `defcon_113` | the measured 13% DefCon shortfall, re-judged on defender metrics | `rate_scale` |
  | H7 `odds_10` | market-only lambda for the clean-sheet channel | `odds_weight` 1.0 |
  | H8 `venue` | the home/away clean-sheet gap in the calibration | home lambda x1.06, away x0.94 |
  | H9 `def_k12` | defenders' attacking rates shrunk twice as hard | `rates.k_by_pos` DEF 12 |
  | H10 `hl90` | recent defensive form: team-model half-life 90 days | `team_model.HALF_LIFE_DAYS` |
* **Prior.** H1 is the one aimed at a measured defect with a mechanism the
  engine lacks (a level calibration, the Understat lesson). H3, H4 and H5
  are structurally more correct forms of components that are already
  calibrated on average, so the E14 rule (a better estimator of a
  calibrated quantity moves nothing) says null. H6, H7 and H10 re-test
  standing rejections on the metric that could see them. H8 is generated
  from the calibration table itself and is the one most at risk of being
  noise. Results follow.
* **Results, 74 paired gameweeks vs the shipped engine** (family alpha
  0.005; `def_*` are GK+DEF pooled, the clean-sheet positions):

  | arm | def_top5 | def_top10 | def_spearman_played | spearman_played | top30 | rmse |
  |---|---|---|---|---|---|---|
  | H1 xg_cal | -0.03 | -0.06 (p=0.12) | **+0.0009 (p=0.001)** | +0.0007 (p=0.036) | +0.00 | +0.0010 (p=0.017, worse) |
  | H2 def_att_exp | +0.02 | **-0.11 (p=0.025)** | -0.0014 (p=0.060) | -0.0005 | +0.02 | -0.0004 |
  | H3 conc_emin | -0.01 | -0.00 | **-0.0018 (p<0.001)** | **-0.0015 (p<0.001)** | +0.02 (p=0.048) | -0.0001 |
  | H4 cs_nb | +0.02 | +0.01 | -0.0003 | **-0.0007 (p=0.001)** | +0.01 | -0.0001 |
  | H5 bonus_dc | +0.10 (p=0.055) | -0.01 | -0.0005 | +0.0008 (p=0.040) | -0.04 (p=0.073) | -0.0013 (p=0.010) |
  | H6 defcon_113 | 0.00 | +0.01 | -0.0006 | +0.0001 | -0.02 | -0.0002 |
  | H7 odds_10 | +0.06 | -0.02 | +0.0003 | -0.0002 | +0.02 | +0.0005 |
  | H8 venue | +0.10 (p=0.17) | +0.05 | -0.0013 | -0.0004 | +0.05 (p=0.051) | -0.0007 |
  | H9 def_k12 | +0.02 | -0.04 | +0.0004 | +0.0001 | +0.02 | +0.0001 |
  | H10 hl90 | +0.02 | -0.01 | -0.0005 (p=0.029) | +0.0000 | -0.01 | -0.0002 |

  Per season, the two arms that looked alive pooled do not replicate:
  H8 `venue` is def_top5 **+0.21 (p=0.016)** in 2024-25 and -0.01 in
  2025-26, with def_spearman_played -0.0047 (p=0.049) in 2025-26; H1's
  top11 is -0.11 (p=0.010) in 2024-25 and +0.17 (p=0.015) in 2025-26.
  H6 is an exact zero in 2024-25 because DefCon counts only exist from
  the 2025-26 rule era, which is the harness working, not a bug.
* **Verdict: none of the ten survives.** Not one arm improves defender
  points per pick at any conventional level, let alone the family alpha.
  The two significant results are the wrong way: H3 and H4, the
  structurally *more correct* forms of conceded exposure and clean-sheet
  probability, both lower rank quality significantly. That is E14's rule
  a fifth time: a component that is already calibrated on average is not
  improved by a better formula for it, because the noise is in the
  outcome, not the estimator. H1 earns a +0.0009 rank gain among
  defenders who played at p=0.001, which is real, a tenth of the role
  features' gain, and paid for with worse rmse and no points. Not shipped.
* **What the diagnostic did establish, which is the actual answer to the
  owner's observation.** The engine's defenders ARE worse picks than its
  midfielders and forwards (4.2 against 5.2 and 4.4 points per pick), and
  every predictor shares that ordering, because a defender's score is a
  4-point clean sheet decided by his whole team plus rare attacking
  returns. The engine's projections of them are honest (4.31 projected,
  4.24 realised across 740 picks) and its edge over a points-per-game
  rule is +1.2 points per defender pick, the largest of any position.
  Clean-sheet luck is 39% of the pick error and E13 already priced perfect
  clean-sheet knowledge at +2.35 points per pick: that is the ceiling,
  and it is not knowable at the deadline. The two measured calibration
  defects (goals 19% over xG, assists 26% under, in every position) are
  level errors that do not reorder players, which is why correcting them
  (H1) moves rank a hair and points not at all.
* **Status.** All hooks stay as research affordances, None on every
  shipped path (`tests/test_defender_hooks.py`). The `c_*` component
  columns on the engine output ship, because a post-mortem should be able
  to say which component missed. Diagnostic:
  `python research/defenders.py diagnose --frame <frame.csv>`.

## E18. A better clean-sheet engine: a learned P(clean sheet) against the Poisson zero

* **Owner's request.** "We need a better clean sheet engine." E17 found
  clean-sheet luck to be 39% of the defender pick error and E13 priced
  perfect clean-sheet knowledge at +2.35 points per pick, so this is the
  component where a better estimator would be worth the most if one exists.
* **What "better" has to beat.** The shipped P(no goals conceded) is the
  Poisson zero exp(-lambda_against), lambda being the bookmaker's implied
  goals for the opponent (85%) blended with the team model (15%). E16
  showed it calibrated (slope 0.997 on its own logit) and unimproved by any
  scoreline feature. A better engine therefore has to be a different
  functional form or a different information set, judged on the direct
  question first: held-out log-loss of P(clean sheet) on team-matches,
  740 a season, before any decision backtest.
* **Design (pre-registered).** `xpts/cs_model.py`: one row per club and
  fixture, features strictly point-in-time at the gameweek's first kickoff:
  the market lambda, the model lambda and the engine's blend (log scale),
  venue, the club's trailing 10-match xGA, goals against, clean-sheet share
  and goals-against-minus-xGA, and the opponent's trailing xG, goals for,
  blank share and goals-for-minus-xG (all shrunk by 3 pseudo-matches).
  Trained on the seasons before the one scored (2022-23 + 2023-24 for
  2024-25; those plus 2024-25 for 2025-26). Forms: a logistic regression
  on lambdas + venue only (a market/model re-weighting), a logistic
  regression on everything, a small gradient-boosted classifier, and an
  offset logit with NO intercept (the engine's own logit as a fixed
  offset, ridge-regularised coefficients on the trailing record). Decision
  arms through the standard 74-gameweek harness for each form, judged on
  the defender metrics; family alpha 0.05/4.
* **Stage 1: held-out team-match log-loss (lower is better).**

  | P(clean sheet) from | 2024-25 log-loss / Brier / mean p | 2025-26 log-loss / Brier / mean p |
  |---|---|---|
  | **Poisson zero, engine blend (shipped)** | **0.5193** / 0.1710 / 0.243 | 0.5293 / 0.1755 / 0.262 |
  | Poisson zero, market lambda only | 0.5200 / 0.1712 / 0.244 | **0.5286** / 0.1753 / 0.264 |
  | Poisson zero, team-model lambda only | 0.5194 / 0.1708 / 0.243 | 0.5379 / 0.1783 / 0.257 |
  | logit: lambdas + venue | 0.5239 / 0.1725 / 0.209 | 0.5309 / 0.1760 / 0.232 |
  | logit: full features | 0.5250 / 0.1731 / 0.210 | 0.5301 / 0.1759 / 0.234 |
  | gradient boosting: full features | 0.5332 / 0.1748 / 0.210 | 0.5411 / 0.1813 / 0.241 |
  | offset logit, no intercept: trailing record | 0.5194 / 0.1711 / 0.234 | 0.5284 / 0.1755 / 0.251 |
  | offset logit, no intercept: record + lambdas | 0.5216 / 0.1720 / 0.236 | 0.5294 / 0.1757 / 0.255 |
  | realised clean-sheet rate | 0.232 | 0.250 |

  Every free-intercept model is WORSE than the Poisson zero, in both
  seasons, and the mean-p column says why: the league clean-sheet rate
  moves from season to season (0.272, 0.207, 0.234, 0.255 across the four
  in the database), a fitted intercept inherits the training seasons' rate
  and carries it into a season with a different one (0.210 predicted
  against 0.232 realised), and the gradient-boosted model pays that price
  and a variance price on top. The Poisson zero has no intercept to
  inherit: its level comes from this week's lambda, which the market
  re-prices every week. Removing the intercept (the offset form) removes
  the loss and buys nothing: the trailing record on top of the engine's own
  logit is worth +0.0001 log-loss in 2024-25 and -0.0009 in 2025-26, a tie.
  Adding the lambdas as free features to that form makes it worse again.
  The logit coefficients say the same thing as E16's regression: after the
  blend, own xGA and opponent xG carry the only weight and goals-against-
  minus-xGA enters with the sign of luck reverting.
* **Stage 2: decision arms.** See the results block below.
* **Stage 2 results, 74 paired gameweeks vs the shipped engine** (GK+DEF
  metrics first; family alpha 0.0125):

  | arm | def_top5 | def_top10 | def_spearman_played | spearman_played | top11 | rmse |
  |---|---|---|---|---|---|---|
  | logit: lambdas + venue | -0.04 | +0.01 | -0.0015 (p=0.043) | +0.0004 | +0.03 | +0.0013 (p=0.017, worse) |
  | logit: full features | -0.02 | -0.02 | -0.0021 (p=0.14) | +0.0003 | +0.04 | +0.0015 (p=0.018, worse) |
  | gradient boosting | -0.12 | -0.12 (p=0.21) | -0.0019 | +0.0004 | +0.10 (p=0.10) | +0.0019 |
  | offset logit, no intercept | +0.08 (p=0.34) | -0.03 | -0.0009 | +0.0001 | +0.03 | +0.0002 |

  Nothing reaches the family alpha in the right direction; the two forms
  that reach p < 0.05 at all do so on rmse, and worse. Every defender
  points metric is inside noise and flips sign between seasons (the offset
  form is def_top5 +0.18 in 2024-25 and -0.02 in 2025-26). The decision
  layer agrees with the team-match layer, which is the better-powered one.
* **Verdict: the Poisson zero on the market-blended lambda IS the better
  clean-sheet engine, and now it has been shown to be.** Four learned forms,
  two seasons, 1,480 held-out team-matches and 74 paired gameweeks, and the
  best of them ties. The reason is structural rather than a shortage of
  features: P(no goals) is one number per fixture, the bookmaker re-prices
  that number every week with more information than any trailing record
  carries (E13's encompassing test, market coefficient 0.90 against the
  model's 0.00), and the Poisson zero converts it without a fitted level.
  The trailing record adds +0.0001 / -0.0009 log-loss on top of it, which is
  the E16 regression's null again in a form that could not be more
  generous to the features. The 39% of defender pick error that is
  clean-sheet variance is variance, not estimator error.
* **The one thing that is not closed.** All of this is a marginal P(no
  goals) from a marginal lambda. A source that carries the joint scoreline
  distribution directly, Polymarket's exact-score market (recorded in the
  Polymarket section as the one thing it has that the bookmaker feed does
  not), prices P(0 goals) without going through a Poisson at all. It is
  live-only and un-backtestable here, so it is a forward-collection
  question, not a modelling one.
* **Status.** `xpts/cs_model.py` stays as a research module; the engine's
  `cs_model` tweak is None on every shipped path (`tests/test_cs_model.py`
  pins the hook off, the rows point-in-time, and the offset form's
  reduction to the Poisson zero at zero coefficients). Stage 1 reproduces
  with `python research/cs_engine.py`.

## Numbering note

Two sessions worked in parallel on 2026-09-14. The defender studies merged
first as E16-E18 (below); this branch's Round 17-20 entries, written as
E16-E20, are renumbered E19-E23 here. CLAUDE.md's Round 17-20 sections and
the memory notes refer to the new numbers.

## E19. Level, not rank: a calibration audit and seven pre-registered arms

* **Why this round exists.** Every metric the backtest reports is a *rank*
  metric or an aggregate. A level error inside one component — clean sheets
  15% too generous for everyone, assists 15% too mean — is invisible to
  `spearman_played` within a position and shows up in top-30 only faintly,
  yet it is exactly what decides a defender against a midfielder in an XI
  and who wears the armband. Nobody had ever asked the engine "do your
  component sums match what was scored?" on a replayed season. Two
  operational fixes came first: Understat club data now covers every
  backfill season (2022-23 on; it had only covered the live season and the
  one before, so the shipped role features were NaN in any replay of
  2024-25), and the engine now emits its per-component points (`c_*`
  columns) so an audit is a join, not a reconstruction.
* **The audit** (`research/audit_components.py`, three seasons replayed
  point-in-time with the availability overlay off and the season-tagged
  minutes model, exactly as the backtest does; one row per player-gameweek,
  32k / 30k / 32k rows). Sum of expected points over sum of realised, all
  players:

  | component | 2023-24 | 2024-25 | 2025-26 |
  |---|---|---|---|
  | goals | 0.92 | 1.07 | 1.07 |
  | **assists** | **0.82** | **0.91** | **0.87** |
  | clean sheets | **1.18** | 1.01 | 0.95 |
  | conceded | 0.86 | 1.06 | 1.08 |
  | saves | 0.81 | 0.91 | 0.89 |
  | bonus | 1.00 | 1.01 | 0.99 |
  | appearance | 1.03 | 1.02 | 1.01 |
  | DefCon | — | — | 0.94 |
  | **total** | 1.02 | 1.02 | 0.99 |

  and by position (total): DEF **1.12 / 1.06 / 0.98**, GK 1.09 / 1.00 / 0.99,
  MID 0.97 / 1.01 / 1.00, FWD 0.95 / 1.00 / 1.00.
* **Four things the audit settles.**
  1. **Assists are under-predicted every season, by 9-18%.** Exposure
     divided out (expected starters who did play 60+), the assist *rate* is
     low at the level (log-calibration intercept −0.11, slope 0.86) while
     the goal rate is fine (−0.03, 0.94). Cause, measured raw: FPL assists
     per Opta xA are **DEF ~1.2, MID ~1.35, FWD ~2.1**, stable across four
     seasons. The 50/50 xA/assists blend in `rates.py` mixes two units.
  2. **Defenders convert below their xG, and increasingly so:** goals per xG
     DEF 0.93 → 0.85 → 0.76 (2023-24 → 2025-26) against MID/FWD ≈ 1.0. That
     is the DEF over-prediction in the xG-era seasons (DEF goals 1.22 /
     1.19); in 2023-24 the same DEF over-prediction came from clean sheets
     instead (1.22), a goal-glut season the 240-day team-model window under-
     reacts to. Same symptom, different cause each year — the signature of
     something a *static* correction gets wrong and an *adaptive* one can
     follow.
  3. **The "60+ under-prediction" in the live post-mortems is mostly the
     minutes channel.** Conditioning on players who played 60+ selects the
     surprise starters (e_min ≈ 0), whose components are near zero by
     construction. Once exposure is divided out the goal channel is
     calibrated. The post-mortem's component table should be read with
     that in mind; it is not evidence of a rate bias.
  4. **Conservation laws.** Σ P(start) per club-fixture is 11.0 in every
     season (sd 0.6-0.8); Σ E[min] is 1015 / 997 / — against the law of 990
     and **1057 in the first six gameweeks of 2023-24** (a model trained on
     one season). The minutes model is calibrated by position to within
     ±3% in all three seasons.
* **Seven arms, all env-gated (`$FPL_XPTS_VARIANT`), shipped path bit-
  identical; 74 paired gameweeks (2024-25 + 2025-26) against a fresh
  baseline in `data/bt_base/`, minutes model shared across arms.**

  | arm | hypothesis | spearman_played | top30 pts/pick | verdict |
  |---|---|---|---|---|
  | `nb_cs` | team goals are over-dispersed (var/mean 1.078), so P(GA=0) should be the gamma-Poisson zero, not exp(−λ) | **−0.0007 (p<0.001)** | +0.00 | **rejected** — CS is already over-called; more zero mass makes it worse, as the audit predicted |
  | `bonus_gd` | the winning side collects more BPS: add expected goal margin to the league bonus regression | +0.0004 (p=0.33) | −0.02 | rejected; rmse significantly worse |
  | `budget_min` | rescale each club's exposure to Σ E[min] = 990 | −0.0002 | +0.06 (p=0.11) | unproven — the 7% overshoot it corrects is a 2023-24 phenomenon; 1% in the test seasons |
  | `online_calib` | one multiplier per component, shrunk ratio of season-to-date actual/predicted (8 gameweeks of prior on 1.0) | +0.0000 | **+0.035 (p=0.023)** | small, consistent in sign both seasons |
  | `online_calib_pos` | the same per position × component | +0.0002 (p=0.51); 2025-26 alone −0.0007 (p=0.04) | **+0.071 (p=0.004)**; +0.105 / +0.036 | the largest points-per-pick gain in this file; rank flat |
  | `xa_scaled` | xA rescaled into FPL-assist units by the league-wide ratio before the blend | +0.0002 (p=0.20) | **+0.037 (p=0.023)**; +0.032 / **+0.042 (p=0.03)** | level fix works; rank untouched |
  | `xa_scaled_pos` | the same, per position | **+0.0004 (p=0.050)** | +0.030 (p=0.13) | rank +0.0004 (p=0.05); the per-position form buys rank, the league-wide form buys level |
  | `xg_conv_pos` | goal rate = xG × per-position conversion (shrunk to 1) | **+0.0005 (p=0.001)** | **+0.043 (p=0.016)**; top-11 +0.06 (p=0.06) | best single arm: rank AND points, same sign both seasons |
  | `xa_scaled_pos,xg_conv_pos` | both structural fixes together | **+0.0009 (p=0.002)** | +0.042 (p=0.078); prec@20 +0.004 | **SHIPPED** — rank +0.0014 (p=0.001) / +0.0004 in the two seasons; the xA half adds +0.0004 (p=0.039) on top of the xG half; rmse identical |
  | `…,online_calib_pos` | the shipped pair plus per-position online multipliers | −0.0002 vs shipped (p=0.46) | +0.006 vs shipped (p=0.78) | **nothing left to correct** — the structural fix absorbs what the online arm was catching; online calibration stays a research tool |

* **Falsified as a cause, recorded so it is not re-derived.** Midfielder
  clean sheets are under-predicted 12% in both xG-era seasons. The natural
  story — a midfielder is hooked when chasing and kept on when protecting a
  lead, so "60+" and "clean sheet" are positively dependent and
  P(60+)·P(CS) understates the joint — is *true* (team CS 0.18 when a
  starting midfielder is hooked before 60, 0.25 when he lasts) and *too
  small*: the implied E[60+ ∧ CS] / (P(60+)P(CS)) is 1.025 for MID, 1.02
  for DEF/FWD, 1.00 for GK. A 2% effect cannot carry a 12% gap.
* **What the two significant arms say together.** A cross-position level
  correction buys points per pick without buying rank: `online_calib_pos`
  moves the top-30 by +0.07 (≈ +1 point a gameweek over a fifteen) with
  `spearman_played` flat pooled and slightly *worse* in 2025-26. That is the
  expected shape — it re-weights DEF against MID/FWD, which reshuffles the
  top of the board across positions without changing the order inside one.
  It is also the first time in this file a change has cleared p < 0.01 on
  top-30 in both seasons' direction. 

  The structural arms say the same thing with fewer moving parts. Both
  online forms are *adaptive corrections of a symptom*; the two rate arms fix
  the *cause* at the estimator, need no in-season state, and clear the primary
  rank metric where the online arms did not. **What ships is the pair of
  per-position unit conversions in `rates.py`** (goal rate = xG × conversion,
  xA rescaled into FPL-assist units before the blend, both point-in-time,
  both shrunk toward 1), with `$FPL_XPTS_VARIANT=legacy_rates` restoring the
  old estimator for any future A/B. The online calibration stays a research
  arm: whether it still adds anything once the causes are fixed is the
  `combo_online` row below.

  This is also the first counter-example to E14's "no estimator headroom in
  the attacking channel". E14 substituted a perfect *rate* for each player and
  found nothing — but its oracle was a rate in the *same units the engine
  already used*, so a level error common to every defender (or every forward's
  assists) was invisible to it by construction. Headroom in the level of a
  unit is not the same thing as headroom in the estimate of a player, and the
  rank metrics cannot see the former inside a position. The audit is the tool
  that can, and it should be re-run whenever a component's definition changes
  (a scoring-rule change, a new xG provider).

## E20. Press conferences, at last: a pre-deadline text feed that is archived

* **The source, found by the owner.** BBC Sport runs a Friday "Premier
  League news conferences" live blog: one post per manager quote, each
  labelled with the fixture it concerns and timestamped, from ~08:00 to
  ~14:30 UK — the day before the deadline. The page the owner pointed at
  (2026-09-11) had 71 posts, 20 with concrete availability ("Collins ...
  calf injury, won't be involved", "Maddison is available", "Caicedo will
  not be available"). Past weeks are enumerable through BBC's search
  container; six search terms and a long patience window found **83 pages
  back to October 2023, 9,699 posts** — 21 / 32 / 25 / 5 Fridays across
  2023-24 / 2024-25 / 2025-26 / 2026-27. Search is relevance-ordered and
  does not surface every week; that is the ceiling of this discovery route
  and it is stated rather than hidden. `acquire/sources/bbc_pressers.py`,
  stored verbatim, never interpreted at acquisition. The same collector
  family archives every match's lineups (formation, pitch slot, captain,
  per-player stats) and live text (`acquire/sources/bbc.py`; 799 matches,
  31,937 lineup rows, 86,145 posts).
* **Stage 0 extractor** (`fpl_engine/pressers.py`): rules only, on purpose.
  Fixture label → the two clubs → their FPL squads for that season; a
  mention resolves on full name, web name or surname only when exactly one
  player across both squads answers to it (ambiguity is skipped, never
  guessed); the mention's sentence is classified by an ordered lexicon —
  out > doubt > rested > available. In-match commentary gives an `injury`
  class ("... because of an injury") for the player's next gameweek. Yield:
  3,779 Premier-League-labelled posts → **1,384 player-level observations**
  (out 481, available 533, doubt 311, rested 31) plus 2,770 commentary
  injuries. It is noisy by design ("no doubt we will see him" reads as a
  doubt); the gates measure what survives the noise.
* **Gate 2 — information the replay engine does not have.** Joining each
  observation to the replayed minutes model (availability overlay off, as
  the backtest runs) over 2024-25 + 2025-26:

  | class | n | model P(start) | actually started | 60+ |
  |---|---|---|---|---|
  | (no mention) | 57,454 | 0.252 | 0.256 | 0.239 |
  | **out** | 280 | 0.273 | **0.175** | 0.168 |
  | doubt | 160 | 0.400 | 0.381 | 0.350 |
  | available | 219 | 0.333 | 0.388 | 0.374 |
  | injury (commentary) | 2,070 | 0.611 | 0.597 | 0.563 |

  Restricted to players the model rated likely starters (P(start) ≥ 0.6):
  **"out" 0.830 → 0.491 started (n=55)**, "doubt" 0.847 → 0.717 (n=46),
  "available" 0.838 → 0.814, commentary injury 0.828 → 0.803 (n=1,170).
  So a manager's "out" halves a likely starter's real chance and the
  replay engine cannot see it; a manager's "doubt" takes 13 points off;
  commentary injuries and "available" carry almost nothing (the trailing
  history and the overlay already know). Cross-fitted exposure factors
  (realised 60+ over modelled, per class): out **0.69** (2024-25) / **0.48**
  (2025-26), doubt 0.89 / 1.00, available 1.18 / 1.20.
* **Gate 1 — timing against FPL's own status log** (2026-27 only, the one
  season with a stored change log; n=29 statements, so a reading, not a
  result): when the manager spoke, FPL had already flagged 59% of the
  "out" players and 75% of the "doubt" ones; **by the deadline 65% and
  100%**. A third of "out" statements were still unflagged at the deadline —
  some will be extractor false positives, but the live overlay is
  demonstrably not a superset of what the manager said.
* **The decision test — first reading was against a STALE baseline, and it
  was wrong.** `data/bt_base` had been replayed before the Round 17 rate
  units shipped; a null arm (same features, retrained) reproduced the Round
  17 `combo` numbers bit for bit, which located the error. Rule, now
  standing: **re-run the baseline after any shipped change before comparing
  an arm** (`data/bt_base` is the post-Round-17 replay; the old one is kept
  as `bt_base_pre_r17`). Against the correct baseline, the arm that scaled
  exposure by the statements (v1 extractor, factors fitted on the other
  season), 74 paired gameweeks:

  | arm | spearman | spearman_played | p@20 | top11 | top30 |
  |---|---|---|---|---|---|
  | pressers, pooled | +0.0006 (p=0.009) | **−0.0012 (p=0.015)** | −0.002 | −0.04 | −0.00 |
  | 2024-25 alone | +0.0008 | **−0.0022 (p=0.018)** | −0.005 (p=0.044) | −0.09 | −0.02 |
  | 2025-26 alone | +0.0003 | −0.0002 | +0.001 | +0.00 | +0.01 |

  It improves the ranking of who plays at all and **significantly worsens
  the ranking of those who do**: the extractor is right about half the time
  when it calls a likely starter "out", and halving a real starter's
  exposure costs more rank than zeroing an absent one gains. Sparse AND
  imprecise is the worst combination for an overlay.
* **Stage 0.5 — clause-level classification.** Reading the class from the
  clause that names the player rather than the sentence ("Saka is suspended
  but Odegaard returns") raised "out" precision on likely starters (started
  0.49 → 0.41) at half the recall (55 → 27 such calls over two seasons).
  Cross-fitted "out" factors 0.63 / 0.20. Precision, not recall, is the
  binding constraint, and it is what a learned extractor is for.
* **What ships: shown, not modelled.** The Friday page is archived and
  extracted by the scheduled pre-deadline refresh and the statement appears
  on the player card with its class and timestamp — the same treatment as
  Polymarket and the Transfermarkt dossier — but `pressers.LIVE_FACTORS` are
  all 1.0: nothing scales a projection until an extractor clears the
  precision bar. That bar is now explicit: on likely starters called "out",
  fewer than a quarter may go on to start (the rules pass is at 41-49%).
  Every observation is stored, so precision accrues week by week against
  realised starts and the bar can be checked without another replay.
* **Stage 1 is scoped, not built.** A small learned extractor needs labels;
  the honest route is LLM-labelled clauses distilled into a small encoder
  (`research/presser_label.py` is the labelling scaffold; it needs an
  `ANTHROPIC_API_KEY`, which this machine does not have, and `torch`, which
  is not installed). With ~700-1,400 clauses a season it is a small job.
  "Sentiment" is not the target; (player, status class) is.
* **Also settled in passing.** In-match commentary injuries ("… because of
  an injury", 2,770 observations) carry almost nothing for the next
  gameweek: likely starters 0.828 modelled → 0.803 started. The trailing
  history already knows who limped off.

## E21. The BBC archive as model input: roles ship, the calendar is measured, a stale baseline is caught

* **Three blocks, one methodological catch.** The Round-18 archive (every
  match's lineups with formation slot and played position since 2022-23, and
  every competition's fixtures) was turned into two optional minutes-model
  blocks, `bbcrole` and `cal`, and replayed. The first comparison of any arm
  came out implausibly strong — until a **null arm** (the baseline's own
  features, simply retrained) reproduced Round 17's `combo` numbers bit for
  bit. `data/bt_base` had been replayed BEFORE Round 17 shipped its rate
  units, so every arm was being credited with that change. Standing rule:
  re-run the baseline after any shipped change; the null arm is the check.
  `bt_base` is now the post-Round-17 replay (`bt_base_pre_r17` kept).
* **A second catch, in coverage.** The first role arm trained on seasons
  that had no BBC rows at all (only 2024-25 onward had been archived) and
  still "gained" — which the null arm explained. The 2022-23 and 2023-24
  lineups were backfilled (all four seasons now 380/380) before the clean
  runs below; a feature present at serve time and absent in training is the
  birth-date lesson (E11c) again.
* **BBC roles — SHIPPED.** Played position on one attacking axis (GK 0 …
  Striker 4, with Wing Back, Defensive and Attacking Midfielder in between),
  row in the formation graphic, and AM/DM flags, from strictly prior starts;
  BBC player URNs resolve to `player.code` by accent-aware, token-overlap
  name matching (99.7% of starter rows). Against the corrected baseline, 74
  paired gameweeks:

  | arm | spearman_played | rmse | p@20 | top30 |
  |---|---|---|---|---|
  | + BBC roles (on top of Understat lines) | **+0.0031 (p<0.001)**; +0.0041*** / +0.0020 (p=0.06) | **−0.0019 (p=0.001)** | +0.0068 (p=0.049) | +0.004 |
  | BBC roles REPLACING Understat lines | **−0.0041 (p=0.004)** | +0.0046*** | −0.004 | −0.02 |

  They add to Understat's line rather than substitute for it, so the
  Understat dependency stays; the block is in `minutes_model.FEATURES`
  (`BBC_ROLE_FEATURES`), NaN without the archive, and the scheduled refresh
  pulls the last eight days of lineups so the live model sees last weekend's
  roles. This is the largest `spearman_played` gain since the line features
  themselves (+0.0047).
* **All-competition calendar (`cal`).** Real rest days, matches in the
  surrounding week and a European tie within four days, from BBC's collated
  fixtures (women's and youth sides filtered by competition — BBC names
  them exactly like the men's club). 2024-25: `spearman_played` +0.0026
  (p=0.004), rmse −0.0016 (p=0.05); pooled on the pre-roles baseline
  `spearman_played` +0.0014 (p=0.033), top-11 +0.11 (p=0.041) — it cleared
  the bar against the model it was designed against. **Re-run on top of the
  shipped roles (the standing rule), it does not**: `spearman_played` +0.0003
  (p=0.63), top-11 +0.06 (p=0.18), rmse −0.0012 (p=0.024). The role block
  absorbed most of what the calendar carried (a club's European midweek is
  visible in who started the previous match and where). Not shipped; kept
  env-gated (`$FPL_MINUTES_EXTRA=cal`) with its archive maintained, since an
  rmse-only gain is the shape that becomes a rank gain when a bigger change
  lands.
* **Captaincy + slot competition (`bbcx`) — rejected.** The club captain
  flag and the player's share of his own formation slot over the club's last
  five matches (plus how many rivals held it), on top of the shipped roles:
  `spearman_played` −0.0002 (p=0.71), top-11 −0.07, rmse ±0.0000, both
  seasons flat. The shipped role and depth features already carry it.

## E22. Absences, fatigue and the crowd's eye test: three owner hypotheses, gated

All four tests below are against the post-Round-19 baseline (`data/bt_base`,
whose per-player audits were regenerated first), so nothing is credited with
an earlier shipped change.

* **Does a starter's absence spill onto his team-mates? Yes, measurably.**
  `research/absence_gate.py`, three replayed seasons, 88,548 single-fixture
  player-gameweeks. When a REGULAR starter (>=4 of the club's last 5 starts)
  plays no minutes, the highest-P(start) depth player at his position starts
  **73% of the time against the model's 61%**, plays **+11 minutes** and
  scores **+0.31 points** over his projection (goalkeepers +0.51 starts,
  +44 minutes, +1.44 points). Restricted to SUSPENSIONS — derivable from the
  card log before the deadline, so knowable in a replay — the next man in
  line is +0.05 starts, +6 minutes, +0.19 points, and the club's other
  regulars gain +0.25 points (p=0.049), i.e. shots and set pieces move too.
  The live availability overlay zeroes the absent man and redistributes
  nothing; this is the size of what it leaves on the table per instance.
* **As a minutes-model block (`absent`) — measured, not shipped.**
  `xpts/absence_features.py`: `sus_self`, `pos_regulars_out`,
  `team_regulars_out` from the card log (second yellow 1, straight red 3,
  5/10/15 yellows by matchday 19/32/38 -> 1/2/3; 697 bans over five seasons)
  and the fixture calendar (`team_match` for replayed seasons, `fixture` for
  the live one — the first run read `fixture` only, which holds the live
  season, and was a null arm). 74 paired gameweeks:

  | arm | spearman | spearman_played | p@20 | top30 | captain | rmse |
  |---|---|---|---|---|---|---|
  | + absent | **+0.0024 (p<0.0001)** | −0.0003 (p=0.67) | −0.0007 | +0.008 | −0.28 (p=0.16) | **−0.0035 (p=0.0003)** |

  The who-plays-at-all metrics improve and every decision metric sits
  still, in both seasons — the availability signature (E11c, Transfermarkt
  injury history) for the fourth time. Suspensions are ~5 players a
  gameweek, and a per-instance +0.2 points on five men cannot move a rank
  over 600. The zeroing half (`sus_self`) is what the live overlay already
  does; the spillover half is real but too sparse to prove here. Kept
  env-gated (`$FPL_MINUTES_EXTRA=absent`), tested
  (`tests/test_absence_features.py`). It becomes provable only when injuries
  are dated point-in-time for a replayed season — the availability change
  log accruing in `data/collected/` is that dataset, roughly a season out.
* **Does a missing starter weaken the club? Barely, and the market prices
  it.** Team goals demeaned by team-season: with a regular suspended the
  club scores −0.10 goals (p=0.12) and allows +0.09 xGA (p=0.07), n=323; with
  any regular absent −0.08 goals for (p=0.06) while BOTH xG and xGA rise
  (+0.18/+0.20, p<0.0001 — a season-timing artefact: depleted squads meet in
  more open matches, both ways). About 7% of a club's attack at most, not
  resolvable, second-order for player points (E13), and the odds blend
  already moves live fixture rates on team news. No arm built.
* **Fatigue as a PERFORMANCE effect — rejected at the gate.**
  `research/fatigue_gate.py`, 26,721 outfield starters who lasted 60+,
  within player-season. Own days since last appearance: xGI/90 coefficient
  **0.0000** (p=0.95), points/90 −0.013/day (p=0.11, wrong sign for fatigue).
  More own minutes in the previous 7 days -> slightly MORE output (+0.008
  xGI/90 per 90 minutes, p=0.04): selection, not rest. The one process
  effect is after a European tie within four days, xGI/90 −0.024 (p<0.0001)
  — while points/90 (+0.16, p=0.02) and BPS/90 (+0.57) go the other way. A
  rate scaler needs a consistent sign and there is none. Rest still enters
  through the minutes model (`days_rest`, `team_matches_14d`, and the
  all-competition `cal` block, measured in E21).
* **BBC crowd ratings ("the eye test in numbers") — rejected at the gate.**
  Collected after all: the averages are server-rendered into the match
  page's `__INITIAL_DATA__` as a `playerRater` block (2024-25 onward; 753
  matches, 22,782 rated player-matches; `acquire backfill --source
  bbc_ratings`). `research/rating_gate.py`, forward in time both ways:
  points beyond the projection **+0.01% / −0.00% RMSE** (nothing — a
  post-match rating is an outcome, and outcomes are already in the rates,
  the standing rule); starts beyond P(start) **−0.23% / −0.34% log-loss**,
  and in the 0.3-0.7 band the worst-rated quartile starts 46% against the
  best quartile's 57% (model 50-52%). A real whisper on rotation, below the
  1-2% log-loss changes that have never moved a decision here (E11c).
  Archive maintained by the scheduled pull; no arm.

## E23. Absences with injuries, referees, set plays, redistribution: four owner asks, one control that mattered

Everything against `data/bt_base` (post-Round-19), 74 paired gameweeks.
Three new archives: Transfermarkt squads + 9,003 dated injury spells
(re-crawled; the first run was rolled back by a lock while another writer
held the file), BBC `match-stats` for every archived match (1,559; Opta xG
split into open play / set play exists from mid-December 2024 only) and the
match officials parsed out of the 1,559 lineup payloads already on disk.

* **The ban derivation was wrong past the first match.** Checked against
  what happened over 697 derived rows: after a red the player was out for
  90% of first matches but PLAYED 54% of second and 62% of third ones (FPL's
  log does not separate a second yellow from a straight red, and most reds
  are one-match bans); five-yellow bans were 98.5% right. Every red is now
  one match: 344 bans, 94% precise. Imposing the old set as hard zeros
  (`absent_zero`, sus only) cost `spearman` −0.0047*** and `spearman_played`
  −0.0041*** — a half-right zero is worse than no information.
* **Redistribution alone is worth nothing.** `spill` vs `absent_zero` on the
  identical absence set (the rate-side spillover: the absent man's expected
  xG+xA handed to club-mates in proportion, f=1): every metric flat (top-11
  +0.06, p=0.18; the rest ±0.000). With Friday "out" statements added to the
  absence set the imposed arm is significantly worse in both seasons
  (`spearman_played` −0.0058***): those calls are right about half the time.
* **Referee card rate — rejected.** Prior decayed yellows per match, shrunk
  (k0=15), relative to the league; factors spread sd 0.05. Flat everywhere
  (`spearman_played` +0.0001, p=0.70; top-30 +0.006). A yellow is one point
  on a component worth a tenth of a point.
* **Set-play split — rejected.** The opponent's prior set-play share of xGA
  (BBC/Opta, shrunk k0=8) reshaping xG by the position's set-play share
  (mean-preserving). Fully informed in 2025-26: `spearman_played` −0.0003
  (p=0.63), top-30 +0.002. The fixture channel is second order (E13) and a
  per-player share would not change that.
* **Learned absence block, bans + injuries — the headline that needs its
  control.** `$FPL_MINUTES_EXTRA=absent`, `$FPL_ABSENCE_KINDS=sus,inj`:
  ~3,500 known player-fixture absences a season (0.6% of them played), i.e.
  35x the suspension-only arm. Against the baseline: `spearman_played`
  **+0.0082***** (both seasons), top-30 **+0.156***** pts/pick, top-11
  +0.155 (p=0.014), rmse −0.045***. The largest paired gain this project has
  measured — and most of it is not the spillover. The CONTROL (`absent_self`:
  the same absence data, the player's own flag only, no team-mate features)
  scores `spearman_played` +0.0063***, top-30 +0.148***, top-11 +0.18**.
  That is the value of AVAILABILITY to a replay that is denied it, and it
  reproduces E11c's valuation (+0.0085 / +0.134) — the live model already
  earns it through FPL's status overlay. **Spillover features alone** (full
  block − control): `spearman_played` +0.0019 (p=0.018 pooled; p=0.14 and
  0.07 by season), rmse −0.0021**, **prec@20 −0.0095 (p=0.005, worse)**,
  top-11 −0.03, top-30 +0.008. A small rank gain that neither season carries
  alone and a significant hit-rate loss: **not shipped**. Kept env-gated with
  the three-kind loader (`xpts/absence.py`), tests in
  `tests/test_round20_context.py`.
* **Standing rule from this entry.** A replay runs with the availability
  overlay OFF, so any feature that re-supplies "he is out" is credited with
  the whole availability channel. Always run the own-flag control and ship
  only the increment over it. The injury boundary is E11c's (spell began a
  day before kickoff and had not ended), which is knowable at the deadline
  in kind but not always in extent.

## E24. The manager's selection process: nine owner hypotheses gated, one block shipped

Against `data/bt_base` (post-Round-19), 74 paired gameweeks. The owner
listed nine selection mechanisms plus a set of "weird football" ones
(2026-09-14, evening). Flagged first: consecutive starts, minutes when he
starts, home/away and congestion are already features; days since last
START, the calendar ahead, European ties with a rest control, XI churn,
formation changes and manager identity were tested in E8b, E12 and E21;
slot competition in E21; absence spillover in E22-E23.

**Gate** (`research/selection_gate.py`): the `sel` block built by the
model's own frame builder, joined to the baseline's audit (`p_start`) on
54,177 player-gameweeks; per feature, does it predict the START residual
(actual − P(start)), overall and in the 0.3-0.7 band?

| feature | overall t | band t | reading |
|---|---|---|---|
| consecutive starts (0 … 13+) | — | — | residual within ±0.01 at every run length, with and without congestion: **no rotation threshold** |
| `self_returning` (first match after a known absence) | **+19.7** (+0.15) | **+8.6** (+0.29) | the model sees his zeros and under-rates him |
| `pos_regular_returning` | −3.8 | **−3.4** (−0.05) | the stand-in's chance falls when the regular is back |
| `prev_unused` (BBC bench, no minutes) | **+7.4** (+0.024) | +1.3 | unused subs start more than modelled |
| `prev_absent` (not in the squad) | −5.9 | **−2.9** (−0.06) | |
| `pts_l1` / `gi_l1` / `pts_l1_vs_avg` | **+8.8 / +6.5 / +8.1** | +5.3 / +3.9 / +3.2 | last match's returns drive selection; the model had no points features |
| `yellows_todate` / `ban_threat` | +4.5 / +3.1 | +2.8 / +1.3 | more bookings, MORE starts (they are the regulars who tackle); no resting before a ban |
| subbed off early, sub minutes, consec × congestion, XI kept, last result, 3+ conceded/scored, clean sheet, home/away bias | n.s. | n.s. | killed |

**Replays.** Full block (19 features) and the gate's survivors (`CORE`, 9):

| arm | spearman | spearman_played | p@20 | top11 | top30 | rmse |
|---|---|---|---|---|---|---|
| `sel` (all 19) | +0.0128*** | **+0.0058 (p=0.002)**; +0.0067** / +0.0049 (p=0.10) | +0.0007 | −0.04 | −0.01 | −0.0139*** |
| `sel_core` (9 survivors) | +0.0123*** | **+0.0050 (p=0.004)**; +0.0065** / +0.0036 (p=0.18) | −0.002 | −0.00 | +0.01 | −0.0145*** |
| full − core | +0.0005 | +0.0008 (p=0.31) | | | | |

The null features add nothing; the core ships. Points-per-pick metrics are
flat, as for every minutes gain before it (E12, E21): this is a rank-quality
gain of the size of the Understat line features (+0.0047) and above the BBC
roles (+0.0031). **Ablation** (`sel_noret`: the core without the two
returning features): `spearman_played` +0.0020 (p=0.17), rmse −0.0103***;
the returning features alone (core − noret) are **+0.0030 (p=0.005)**, rmse
−0.0043***. So the proven rank gain is mostly the returning-player
mechanism; bench status, last-match returns and bookings add who-plays
accuracy and an unproven +0.002 of rank. The block ships whole (it clears
the bar as a whole and the extra features cost nothing), stated that way.

**Why this is not the availability channel again (E23's control).** The
returning flag re-supplies "he is fit again", and the live overlay has no
equivalent: it only zeroes players who are out and does nothing for a man
whose trailing window is zeros because he was injured. So the increment is
real at serve time. At serve time the absence source is FPL's own change log
(`xpts/absence.py` kind `fpl`: status i/s/u or chance <= 25%, from the
change that opened the spell to the one that closed it), bans from the card
log, and Transfermarkt spells as crawled; in a replay it is bans + dated
injuries. Transfermarkt's `until_date` typically sits days before the first
match back, so the replay's returning flag is knowable in kind at the
deadline, as FPL's status flip is live.

**Round 21b, the owner's refinements, gated the same way.** (B) *The signal
lives in the specific stand-in*, as predicted: the player who held the
returning regular's modal pre-absence formation slot in the last match of
the absence (`replacement_for_returning`, from BBC lineups) carries
**−0.046 overall (t=−6.8) and −0.089 in the band (p=0.0001)**, while the
position-level count WITHOUT that man is −0.002 (p=0.11). Replayed as an
arm on top of the shipped block (`$FPL_MINUTES_EXTRA=sel_rep`) against the
new baseline: `spearman_played` +0.0002 (p=0.59), rmse +0.0003, every
decision metric flat in both seasons — the shipped block (his own return
flag, the position count, and the depth features the tree interacts them
with) already absorbs the specific stand-in. Not shipped; kept as a
research feature. (C) Home/away start bias with a minimum of eight rows each way: coef
+0.0095 (p=0.46), band −0.07 (p=0.12) — null. (D) Suspension threat split by
importance: regulars one booking from a ban +0.028 (p=0.09), fringe +0.057
(p=0.13), regulars on exactly four yellows +0.019 (p=0.06) — every sign is
MORE starts, managers do not rest them; killed. (E) Manager reaction to a
shock, by position: conceding 3+, losing by 3+, winning by 3+ and losing
move no position's residual (all p > 0.13; FWD after a loss +0.009,
p=0.048, noise), and at team level a club that conceded 3+ keeps 78.3% of
its XI against 80.1% otherwise — a fifth of a player. Killed.

**Shipped:** `SELECTION_FEATURES` in `minutes_model.FEATURES` (48 features),
attached in `_frame` from `xpts/selection_features.py`; live model retrained
on all four seasons (holdout 2025-26 accuracy 0.836), projections rebuilt
GW4-9, server restarted 2026-09-14 evening. Baseline rotated:
`data/bt_base` = the `sel_core` replay, previous baseline in
`data/bt_base_pre_r21` (audits there; regenerate for the new baseline before
the next audit-based gate). Tests: `tests/test_selection_features.py`.

## E25. Does the opponent's style reach a player? DefCon and possession

The owner asked (2026-09-15) whether the timestamped in-match data (touches,
attacks, possession over time) could make the model "understand each club's
gameplay". What is held: per-team match totals for every archived match
(possession, shots split, box touches, crosses, corners, tackles,
clearances, distance, sprint share; Opta xG open/set from Dec 2024), per-
player match stats and formation slots, the timestamped live text; not yet
collected: BBC's per-minute momentum series; empty: Understat's shot log.
Every team-STYLE feature tried so far (E12: pressing, deep completions,
formation, manager; E9-round game state) moved nothing, because style
reaches a player through the fixture rate and the market encompasses the
team model there. The one place it can bite is a component whose mechanism
depends on the opponent — and DefCon is the rate E14 found headroom in.

**Gate (mechanism).** 7,711 player-matches with DefCon counts (2025-26,
2026-27), 60+ minutes, within player-season: a defender's threshold
crossing runs **16% against opponents under 40% possession to 37% over
60%**, +4.2 points per 10 points of possession (p=5e-12; +0.42 actions/90);
midfielders +1.9 per 10 (p=2e-4). Opponent possession is predictable from
its own prior matches (r=0.48). The engine's DefCon term is a flat per-90
rate x exposure with no opponent term.

**Arm `defcon_style`** (`bbc_context.possession_factors`, engine variant):
the opponent's decayed prior possession (k0=5 toward 50) times a per-
position slope fitted WITHIN PLAYER on rows strictly before as_of (shrunk,
n0=500) — point-in-time throughout; multipliers 0.82-1.19 on the crossing
rate. DefCon exists only from 2025-26, so the paired test is **37
gameweeks**, not 74 (2024-25 is bit-identical by construction):

| metric | delta | p |
|---|---|---|
| spearman_played | +0.0009 | 0.73 |
| top-30 pts/pick | **+0.106** | 0.077 |
| top-11 pts/pick | +0.135 | 0.14 |
| captain | +0.41 | 0.20 |
| def_spearman_played | +0.0076 | 0.11 |
| spearman | −0.0008 | 0.003 |

Every decision metric leans the right way and none clears the bar at
n=37; the top-30 gain is the size of a shipped change (Round 17's was
+0.04) with half the sample. **Not shipped**; kept env-gated
(`$FPL_XPTS_VARIANT=defcon_style`, `tests/test_round20_context.py`) and
re-tested when 2026-27 has accrued — every gameweek adds power, and this
is the first arm whose verdict is a sample-size wait rather than a null.
The same construction (opponent style x component mechanism) applies to
GK saves vs opponent shot volume and defenders' clearances vs crosses,
untested.

## E26. Opponent style for keepers, forwards and midfielders (E25 extended)

The owner asked for the DefCon construction on keepers and forwards
(2026-09-15). Gate (`research/style_gate.py`, within player-season, errors
clustered by fixture, the strength control = the opponent's prior xG for
keepers / xG against for attackers):

* **Keepers**: the opponent's prior shots on target add saves with its xG
  present (+0.21 saves/90 per SoT, p=0.044) and xG then loses significance;
  top-quartile shot-volume opponents give +0.6 saves/90 over the bottom
  quartile (about 0.2 points). The engine scales saves by lambda_against^0.6
  only.
* **Forwards**: nothing beyond the opponent's xGA — possession p=0.88,
  shots allowed p=0.12, box touches allowed p=0.10 (n=1,217). Killed.
* **Midfielders**: the box touches an opponent allows raise xG/90 beyond its
  xGA (t=4.1, p<0.0001; shots allowed t=3.6) and carry to points/90
  (p=0.004). Nothing in the engine reacts to it.

**Arm `style`** (`bbc_context.style_factor_map`): the opponent's decayed
prior shots on target (keepers' saves) and box touches allowed (midfielders'
xG), relative to the league, times a slope fitted within player on rows
before as_of (shrunk, n0=500); multipliers 0.65-1.28; forwards excluded.
74 paired gameweeks, combined and ablated:

| arm | spearman_played | top11 | top30 | captain | def_top5 | spearman |
|---|---|---|---|---|---|---|
| style (both) | +0.0009 (0.61) | **+0.17 (0.046)**; 25-26 +0.34** / 24-25 +0.01 | +0.02 | **+0.69 (0.048)**; 24-25 +0.97* / 25-26 +0.41 | +0.25 (0.028) | −0.0009*** |
| keepers only | +0.0011 | +0.06 (0.45); 25-26 +0.22* / 24-25 −0.11 | +0.04 | +0.36 | +0.25 (0.028) | −0.0007** |
| midfielders only | +0.0008 | **+0.17 (0.048)**; 25-26 +0.34** / 24-25 +0.01 | +0.01 | **+0.69 (0.048)** | +0.17 | −0.0009*** |

The midfielder half carries the pair. Read with the standing rules: the
bar is `spearman_played` or top-30 and neither moves; the top-11 gain
lives in one season and the captain gain in the other, and captain points
need ~2,000 gameweeks to resolve a 0.4-point edge (E1). Two p~0.048 results
among eight metrics and three arms are what noise looks like. The
`spearman` loss (−0.0009, p<0.001) is real but a reshuffle among fringe
players. **Not shipped**; env-gated (`$FPL_XPTS_VARIANT=style`,
`style_gk`, `style_mid`) and queued with `defcon_style` for re-test as
2026-27 accrues. The mechanism for midfielders is measured and the arm is
the right shape; the decision value is unresolved at n=74.

## E27. An outside architecture review, sorted; two new lineup feeds; cross-league seeding gated

The owner brought a separate research summary (2026-09-16: the academic
OpenFPL paper, 2PM-Transformer, hierarchical Bayes, GNNs, commercial stacks,
bookmaker props, predicted-lineup accuracy). Sorted against this log:

* **Confirms, already measured here:** minutes/lineups are the binding
  constraint (E8, E8b, E16 pricing), the market is at least as good as any
  team model (the encompassing regression), the engine is already the
  hurdle/compound structure it recommends (P(0 min) → conditional rates →
  scoring rule), and a new regressor over the same features cannot beat the
  variance floor (E14: perfect rates are worth nothing).
* **Refuted here:** market-as-calibration-target (E17 market-only lambda,
  null), tail-calibration objectives (E1-E2 rank functionals, null),
  Bayesian time-varying ability (adaptive shrinkage, worse).
* **Genuinely new, acted on:**
  1. **Two more predicted-lineup feeds, forward-collected.** Fantasy Football
     Scout's free page serves every club's predicted XI (formation, rows,
     full names) AND Out / Doubts-with-percentage / Banned / latest-news
     lists (`acquire/sources/ffscout_lineups.py`); SportsGambler serves
     predicted or confirmed XIs per fixture through its own AJAX endpoint
     (`acquire/sources/sportsgambler_lineups.py`). Both keyless, stdlib
     parsed, wired into the scheduled collector (`acquire/actions.py`:
     `data/collected/lineups_ffscout/`, `team_news_ffscout/`,
     `lineups_sportsgambler/`, append-only, change-detected per club) and
     seeded 2026-09-16 (220 + 108 + 264 rows). The lineup-tester evidence
     says the best provider varies by club, so an ensemble of three feeds
     is the right thing to price in E15's harness once the band rows accrue;
     the Scout doubts-with-percentage list is the first pre-deadline injury
     feed with a stated probability, to be gated against FPL's flag.
  2. **Cross-league seeding for new signings — killed at the gate.** The EPL
     ingest drops a player's other-league Understat matches; a collector now
     keeps them (`understat_foreign_match`, seasons before his first FPL
     one; 18,691 rows over 249 players). For 105 new signings with >= 900
     foreign minutes, the foreign npxG/90 and xA/90 predict the first six
     EPL appearances no better than the engine's cold-start rate: joint
     regression t_foreign (goals / assists) = −0.14 / −0.18 (DEF, 42
     players), +1.99 / +0.68 (MID, 38), −0.38 / −0.80 (FWD, 18), with the
     engine term significant in most cells; MAE within 0.015 of the
     engine's everywhere. One borderline cell in six is noise.
     `research/cold_start_gate.py`. No arm.
  3. **Player-prop odds** (anytime scorer, clean sheet) are the one market
     signal that reaches a player directly; forward logging needs a working
     Odds API key (currently 401). Queued.

## E28. Oddschecker: keyless bookmaker odds, the exact-score market and player props

Found by the owner (2026-09-16). Each Premier League match page embeds a
`subeventmarkets` JSON with every populated market's decimal odds per
bookmaker (~24) and Oddschecker's own implied probability per selection;
three markets are populated server-side: Win Market, Correct Score (41
scorelines) and Anytime Goalscorer (~44 players). `ingest/oddschecker.py`
de-margins the median 1X2 price, turns the correct-score distribution into
P(over 2.5) / P(clean sheet) per side / P(both score), inverts to Poisson
rates through the shipped `implied_rates`, and writes `match_odds` (one row
per fixture, source `oddschecker`) plus `market_prop` (clean sheets,
both-to-score, anytime scorers per player); all rows are appended to
`data/collected/oddschecker/<season>.csv`. Cloudflare challenges Python's
TLS fingerprint (`requests` → 403, `cf-mitigated: challenge`) while the
system `curl` with a cookie jar and a Referer is served, so the client
shells out to curl, one request a second, and retries the first cookie-less
hit. GW5 2026-27: all 10 fixtures priced, 441 anytime-scorer prices.

What it changes: the live model blends BOOKMAKER prices for upcoming
fixtures again (the E27 finding was that it had blended nothing all
season), with the goal level pinned by the exact-score market rather than
the team model's total. Not a model change — no replay. What it opens:
`market_prop` is the first per-player market signal here; from GW5 the
anytime-scorer probability accrues beside the engine's `e_goals`, and the
gate is the usual one — does the market's P(scores) predict realised goals
beyond the engine's expectation? — runnable after ~8 gameweeks. The
correct-score P(clean sheet) answers CLAUDE.md's open question 1 (the
exact-score market as a direct P(no goals)) once the same window has
accrued, scored on log-loss next to `exp(-lambda)` in
`research/cs_engine.py`.

**E28 addendum — the horizon beyond the market, and the prices on the card.**
The owner asked whether Polymarket could substitute for gameweeks no
bookmaker has priced; it lists only the current round (10 fixtures), fewer
than Oddschecker's two. Measured instead: the team model compresses the
market's strength gaps by 23-31% in every full season (log-log slope
1.23/1.23/1.31, r 0.92-0.95, 2,162 fixture-sides; 2022-23 excluded at
r=0.60), so `odds_model.fit_market_stretch` fits that mapping on every pull
and the engine applies it to UNPRICED fixtures only (replays untouched:
`backtest` passes `market_stretch=False`). Per-player anytime-scorer and
club clean-sheet prices resolve to FPL players at 97-99%
(`oddschecker.match_player`; the rest are non-FPL squad members) and are
shown on the player card, not modelled.
