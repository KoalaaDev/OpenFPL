# CLAUDE.md — fpl_engine data pipeline

Guidance for working on the automatic data pipeline that feeds OpenFPL.

## What this is

`fpl_engine/` is a free, automatic data pipeline that pulls Fantasy Premier
League data into a **local SQLite database** and builds the point-in-time
feature samples the pre-trained OpenFPL models consume — so predictions run
end-to-end with no hand-built `samples.csv`.

Store: one SQLite file (`data/fpl.sqlite`, override with `$FPL_DB_PATH`). No
server, no cloud, no API keys.

## Architecture (layers)

```
Free sources ─▶ ingest ─▶ SQLite ─▶ features (point-in-time) ─▶ OpenFPL models ─▶ predictions
```

| Module | Responsibility |
|---|---|
| `fpl_engine/db.py` | SQLite schema + connection/upsert helpers (the only store) |
| `fpl_engine/http.py` | cached, retrying, polite HTTP (requests or urllib) |
| `fpl_engine/ingest/fpl_api.py` | official FPL API (free): bootstrap, fixtures, per-player history |
| `fpl_engine/ingest/vaastav.py` | free historical backfill (cross-season form for early GWs) |
| `fpl_engine/ingest/understat.py` | best-effort Understat advanced stats (degrades gracefully) |
| `fpl_engine/resolve.py` | FPL↔Understat entity resolution (override table, fails loud) |
| `fpl_engine/scoring.py` | canonical FPL points calculator (YAML-driven) |
| `fpl_engine/features.py` | point-in-time 228-feature builder (exact OpenFPL columns) |
| `fpl_engine/predict.py` | OpenFPL ensemble inference (refactor of `play.ipynb`), optional blend |
| `fpl_engine/train.py` | optional GPU retrain of per-position models + blend weight |
| `fpl_engine/manager.py` | fetch an FPL entry (squad id): current squad, bank, FTs |
| `fpl_engine/price_model.py` | who rises/falls in price before the next gameweek (transfer flow ÷ ownership + price history + form); ranks the movers, does not feed the objective |
| `fpl_engine/optimise/project.py` | per-player projections across the horizon |
| `fpl_engine/optimise/milp.py` | multi-period squad/transfer/captain optimiser (PuLP+CBC) |
| `fpl_engine/pipeline.py` / `__main__.py` | orchestration + CLI |

## Non-negotiable engineering principles

1. **Point-in-time discipline.** Every feature uses only matches with
   `kickoff_utc < as_of` (the target GW's first kickoff). The builder filters
   physically; never relax this. This is the highest-risk failure mode.
2. **The scoring engine is the single source of truth.** All FPL points come
   from `scoring.points_from_events`, driven by `config/scoring_rules_*.yaml`.
3. **Forward-in-time validation only.** No random splits. Reconcile and backtest
   on point-in-time data.
4. **Data contracts / idempotency.** Ingestors `INSERT OR REPLACE` on stable
   keys; re-running never duplicates. `player_gw` is keyed per *fixture* so
   double-gameweeks are not silently collapsed.
5. **Fail loud on entity-resolution misses**; never silently drop players.
6. **No hardcoded scoring constants** outside the YAML rules file.

Current scoring rules: `config/scoring_rules_2026_27.yaml` (version `2026-27`).

## Adapting to new results

Two independent mechanisms:

1. **Form (inputs) update every week for free.** Re-running `pull` after a
   gameweek writes the new matches into `player_gw`/`team_match`; the next
   `build`/`predict` recomputes the trailing-window features from them. The
   frozen OpenFPL models then see fresh form. This is the primary adaptation and
   needs no retraining.
2. **Weights (optional) via `train.py`.** `python -m fpl_engine train` refits
   per-position XGBoost regressors on the point-in-time feature store (GPU auto-
   detected via `device`; override with `$FPL_DEVICE`), validates forward-in-time
   (holds out the latest season, reports stratified RMSE), and saves to
   `models/retrained/`. Inference blends them with OpenFPL:
   `predict/optimise --blend auto` weights the fresh model up as the new season
   accrues data (`season_blend_weight`); `--blend 0` (default) is pure OpenFPL.
   The retrained models reuse OpenFPL's scaler + per-position feature subset, so
   the blend is in one consistent space. Training MUST stay forward-in-time (no
   random splits, no look-ahead) — the frame is built with the same `as_of`
   builder used for prediction.

## xPts component engine (fpl_engine/xpts/)

A structural alternative to the monolithic OpenFPL regression: FPL points are
*assembled* from modelled processes, combined through the scoring YAML (which
stays the single source of truth):

| Module | Component |
|---|---|
| `xpts/team_model.py` | time-decayed Poisson attack/defence (fit on `team_match`, teams matched across seasons by `code`, goals blended with xG, promoted clubs shrunk to a below-average prior). Each rating update is a **ridge-regularised fixed point** (`RIDGE_GOALS` pseudo-goals on both sides) and the ratings are centred on effective matches: without that, a club with no goals yet drives `log(0)` and diverges, which silently collapses `league_rate` — see the note below |
| `xpts/minutes_model.py` | XGBoost P(0 / 1-59 / 60+ minutes), plus a separate regressor for E[minutes \| plays] so exposure is estimated rather than rebuilt from class means, from four feature blocks: trailing history, role/depth (minutes *when he starts*, consecutive starts, share of the team+position minutes, price rank inside team+position — raw price is deliberately excluded as a fame bias), fixture context (rest days, 14-day congestion, home, fixtures this gw) and crowd (last gameweek's ownership and net transfers). **One feature builder serves training and prediction** — the target gameweek's rows are appended to the history frame, so there is no train/serve skew. Trained on *every* prior season; cached in `models/xpts/` and retrained automatically when the cache predates a feature change (`ensure`). Live availability (status/chance_next) is applied on top |
| `xpts/rates.py` | empirical-Bayes-shrunk per-90 rates (xG, xA, saves, cards), an event-conditional **bonus model** (league per-position WLS `bonus ~ goals+assists+cs`; players keep only their deviation as a flat rate, so E[bonus] scales with the fixture), a **DefCon threshold-crossing rate** (raw `defcon` counts are ingested from vaastav/FPL for the rule era — `defensive_contribution.since` in the scoring YAML; rate = shrunk crossings per 90) + a **residual rate** = actual points minus full reconstruction (including actual DefCon), i.e. genuinely unmodelled scraps only. Player histories get an extra ×`SEASON_BREAK_DECAY` per season boundary so an outlier season is not carried whole across a summer |
| `xpts/engine.py` | E[points] per player per gw: exposure x rates x fixture scalers, P(CS)=P(60+)*exp(-λ_opp), Poisson floor-division expectations for conceded/saves (GK saves scale sublinearly with opponent threat), E[bonus] from the league event coefficients applied to the player's own expected events; DGWs sum over fixtures |
| `xpts/odds_model.py` | betting odds → implied Poisson goal rates: de-margin 1X2 (+ over/under 2.5) odds, invert to (λ_home, λ_away), blend into the team model's fixture rates with `ODDS_WEIGHT` (0.85; fitted by backtest sweep on 2024-25 + 2025-26 — improves active-player rank and captain picks, 0 disables) |
| `ingest/odds.py` | two free odds sources into `match_odds`: football-data.co.uk CSVs (no key; historical + in-season, early-snapshot `Avg` columns preferred over closing for point-in-time honesty) and The Odds API upcoming fixtures (`$ODDS_API_KEY`, free tier — one call = 2 credits of 500/month). Team names resolve via explicit maps and fail loud |
| `xpts/set_pieces.py` | penalty duty as a correction toward FPL's published order, not a flat bonus — zero when today's duty already matches the history, and zero in any replayed season |
| `xpts/simulate.py` | correlated match simulator: draws the scoreline first, then hands out goals/assists/CS/saves/bonus conditional on it. For **joint risk** (floors, ceilings, P(haul), portfolio variance) — measured NOT to improve ranking, see below |
| `backtest.py` | forward-in-time replay of a past season scoring xpts vs OpenFPL vs naive baselines. Metrics are decision-relevant: `spearman_played` (rank quality *among players who got on the pitch* — plain Spearman mostly measures who plays at all), `top11`/`top30` (mean ACTUAL points of the highest-predicted players — points per pick), precision@20, captain points, RMSE. Per-gameweek series are kept in the JSON so `compare-backtests` can paired-t-test a change; the blend weight is fitted on the first half of the season and evaluated on the second, then written to `models/xpts/blend.json` |

`predict`/`optimise`/the web app automatically blend xPts with OpenFPL using
the weight in `models/xpts/blend.json` (absent or 0 = pure OpenFPL). Penalty
takers (`penalties_order` from the live bootstrap) get a small xG90 boost in
the web path. Backtests disable the availability overlay (stored status is
today's, not historical). Run:

```
python -m fpl_engine backtest --backtest-season 2025-26   # also (re)trains minutes model
python -m fpl_engine compare-backtests before/ data/      # paired A/B of a change
```

**Prove a change, do not eyeball it.** Season means of these metrics move by
more than a real improvement does. Snapshot `data/backtest_*.json`, make the
change, re-run the backtest for both 2024-25 and 2025-26, then
`compare-backtests` the two — 74 paired gameweeks and a p-value. Treat p > 0.05
as unproven no matter how good the mean looks.

## Where the error actually is

Replaying 2024-25 and 2025-26 with a **perfect** minutes model (realised
minutes substituted in, everything else untouched) gains:

| | Spearman | Spearman (played) | prec@20 | top-30 pts/pick | captain |
|---|---|---|---|---|---|
| oracle minutes − baseline | **+0.21** | **+0.21** | +0.05 | **+0.67 … +0.88** | +0.08 … +0.49 |

Nothing else in the engine is worth a fraction of that. Two consequences:

1. **Spend effort on minutes, not on rate constants.** A one-at-a-time sweep of
   every constant in `rates.py`, `team_model.py`, `engine.py` and
   `odds_model.py` (24 variants × 2 seasons) moved `spearman_played` by at most
   ±0.002 and top-30 by at most ±0.06 — all inside gameweek-to-gameweek noise.
   Those defaults are fine; leave them alone.
2. **Even a perfect minutes model leaves `spearman_played` at ~0.59.** FPL
   points are extremely noisy. Improvements of a few hundredths are real and
   worth having; anything claiming much more is a bug or a leak.

### Tested and rejected (do not re-litigate without new information)

Each was implemented, backtested forward-in-time on both seasons, and dropped
because the out-of-sample gain did not survive:

* **Blending a raw-form channel into xPts** (decay-weighted points-per-90 ×
  the engine's own expected minutes, so it adds form *rate* rather than a
  second minutes estimate; weight fitted on the first half of the season,
  judged on the second). Best case +0.03 top-30 in 2024-25, ≈0 or negative in
  2025-26, and `spearman_played` *falls* monotonically with the weight. The
  component engine already absorbs form through its rate estimates.
* **Bonus as a simulated within-match BPS rank** (per-position BPS regression
  on the engine's own expected events + a shrunk personal deviation, then
  1500 Monte-Carlo draws per fixture awarding 3/2/1 to the top three). It is
  better *calibrated* than the linear term — 6.00 expected bonus per fixture
  against the linear model's 6.85, versus 6.30 actually awarded — but it moved
  no decision metric consistently (2025-26: captain +0.35, prec@20 +0.008;
  2024-25: captain −0.62, top-11 −0.19). The linear term already gets the
  *ordering* right, which is all the ranking needs.
* **Market odds beyond what is already there.** With football-data odds loaded
  for every backtest season, `ODDS_WEIGHT` 0 / 0.5 / 0.85 / 1.0 are
  indistinguishable (≤0.001 Spearman, ≤0.02 top-30). The team model, fitted on
  xG, is already where the market is. Odds still earn their place for
  *upcoming* fixtures where team news beats trailing form — but do not expect
  the backtest to show it.
* **Recency-weighted minutes training** (550-day half-life on the sample
  weights): log-loss got slightly *worse*. Rotation patterns from three
  seasons ago still generalise.
* **Adaptive (change-point) Bayesian shrinkage on the player rates.** Two
  estimates per player — fast (70-day half-life, weak shrinkage) and slow
  (420-day, strong) — blended by how much evidence there is that they differ,
  `w = z²/(z²+C)` with `z` the divergence in standard errors. The idea is to
  believe a genuine role change sooner without chasing a hot streak. At
  C = 1 / 4 / 16 and at pure-fast, every variant is flat-to-worse: top-30
  −0.02 to −0.03 pts/pick (p ≈ 0.07–0.17), `spearman_played` −0.0001. The
  fixed 240-day half-life with empirical-Bayes shrinkage is already doing this
  job; a role change large enough to matter also brings the exposure that
  moves the shrunk estimate on its own.

### Team-model identifiability (a live bug, fixed)

`log E[goals] = mu + home + attack(i) − defence(j)` is only identified up to a
constant; the fit pins it down by mean-centring attack and defence. An
unregularised multiplicative Poisson update sends a club with **no goals in its
weighted history** to `log(0) = −inf` — which drags the centring, and with it
`mu`. The fixture rates still looked plausible (attack/defence absorbed the
shift) so this was invisible in the λ's, but `league_rate` collapsed, and
`league_rate` is the denominator of the engine's fixture attack scaler. In
2026-27 GW2 it read **0.299** instead of ~1.42, which pinned **90% of fixtures
at the scaler cap** — the fixture-quality signal was gone for the whole league.
It bit only the live season, where two promoted clubs had a single match and
one of them had not scored, so no historical backtest could have caught it.

The fix is `RIDGE_GOALS` pseudo-goals inside a fixed-point (not incremental)
update plus effective-match-weighted centring, and taking `league_rate`
straight from the data instead of from `exp(mu + home/2)`. It costs nothing on
healthy seasons (74 paired gameweeks: every metric within ±0.03) and is covered
by `tests/test_team_model_stability.py`. **If you touch `team_model.fit`, check
`league_rate` against the observed goals-per-team-match before trusting it.**

### Squad value: the price-change model

`price_model.py` is the one place where a *new* free signal turned into a
large, replicated edge. Held out forward in time, the top-10 ranked risers
actually rise **67% (2024-25) / 75% (2025-26)** of the time against a **2.0%**
base rate, and the top-30-minus-bottom-30 spread is **0.87-0.91 tenths of £m
per pick** against **0.40** for a naive net-transfers rule. Log-loss is 0.166
against 0.308 for the class prior.

Feature blocks, each adding on top of the last (2024-25 / 2025-26 log-loss):

| | 24-25 | 25-26 | P(rose \| top-10) |
|---|---|---|---|
| transfer flow ÷ ownership | 0.2062 | 0.1988 | 0.49 / 0.56 |
| + price history | 0.1924 | 0.1864 | 0.53 / 0.63 |
| + position, gameweek | 0.1863 | 0.1792 | 0.53 / 0.63 |
| + last gameweek's form | 0.1724 | 0.1677 | **0.69 / 0.74** |
| *deadline-only (form lagged a gw)* | 0.1845 | 0.1774 | 0.57 / 0.64 |

The last row is the leak check: strip everything about the most recent
gameweek's matches and the edge survives, so the gain is not the panel peeking
across the gameweek boundary.

**It is deliberately not wired into the solver's objective** — see Round 7
below, which measures the exchange rate rather than guessing it and finds a
price rise is worth about 0.2 points. Read it with
`python -m fpl_engine prices`, which now prints that conversion, and let it
break ties between transfer targets you already rate.

### The betting market encompasses the team model

Regressing realised team goals on both point-in-time rates over 2,102
team-matches (2023-24 to 2025-26):

```
goals ~ model + market      model  coef -0.009  t -0.05  p 0.96
                            market coef +0.898  t +6.79  p 0.0000
R2:  model only 0.1175   market only 0.1365   both 0.1365
resid_model ~ (model - market):  coef -0.806  t -7.02  p 0.0000
```

The team model contributes **nothing** once the market is in, and when the two
disagree the market is right about 80% of the way. So the "train a meta-model
on where the market is wrong" idea has no residual to find here — the model is
not wrong in a *predictable direction*, it is simply the weaker estimator.

This sits oddly next to the finding that `ODDS_WEIGHT` 0 / 0.5 / 0.85 / 1.0 are
indistinguishable on player metrics. Both are true: the market's team-level
edge is real but small in absolute terms (RMSE 1.166 → 1.153 on a quantity
whose sd is ~1.24), and it reaches a player only through the fixture attack
scaler, where it is swamped by minutes and rate noise. Do not read the
insensitivity as "odds are useless" — read it as "the fixture channel is a
second-order term for player points".

### The correlated match simulator

`sim.py` in the research scripts draws the scoreline first and hands out goals,
assists, clean sheets, saves, cards and top-3 BPS bonus *conditional on it*, so
the samples carry the joint structure a per-player expectation cannot. Its
mean tracks the analytic engine at r = 0.992.

It does **not** improve rankings, and that is measured, not assumed:

| criterion (74 gameweeks) | Δ spearman_played | Δ top30 | Δ captain |
|---|---|---|---|
| sim mean | **−0.0043** (p=0.012) | +0.05 (p=0.27) | +0.15 (p=0.72) |
| sim median | −0.0398 (p<0.0001) | −0.37 | −1.14 |
| P(haul ≥ 10) | −0.0304 (p<0.0001) | +0.02 (p=0.71) | +0.43 (p=0.38) |
| mean ± 0.25 sd | −0.005 | +0.04 | −0.20 |

And within the realistic captain choice set (top 10 by xP, **740** candidate
player-gameweeks — far better powered than one captain a week), every
criterion ranks the same: pooled Spearman 0.208 (xP), 0.218 (sim mean), 0.209
(P(haul)), 0.212 (90th percentile), all pairwise p > 0.2. **The extra moments
do not help pick a captain. The mean is sufficient.**

What the simulator *is* good for is the thing the metrics above cannot see —
joint risk. Independence understates the spread of a starting XI's total by
**8%** (sd 15.0 vs 13.9) and of a one-club triple-up by **6%**; across three
different clubs the ratio is 1.00, exactly as it should be. Any chip, rank or
differential calculation done player-by-player is therefore wrong by about
that much. Keep it for that; do not use it to rank.

### Beating the crowd's captain

With the template held fixed so only the captain differs (ownership from the
real per-gameweek `selected` counts, lagged one gameweek), over 74 gameweeks:

| captain rule | mean pts | vs the crowd | blanks ≤2 | hauls ≥10 |
|---|---|---|---|---|
| the crowd's own pick | 7.16 | — | 38% | 31% |
| max analytic xP | 7.87 | **+0.70** (p=0.11) | 28% | 37% |
| max sim mean | 8.01 | +0.85 | 22% | 37% |
| max P(haul ≥ 10) | 8.24 | +1.08 | 23% | 41% |
| max P(beat the field) | 8.07 | +0.91 | 26% | 37% |

The model's captain beats the template's by **+0.7 to +1.1 points a week** —
27 to 41 points across a season, from one decision. But the *differences
between the model rules* (+0.15 to +0.38) are not resolvable: captain points
have a standard deviation of 5.9, so at n=74 nothing short of a two-point gap
is significant, and separating +0.38 from zero would need roughly **2,000
gameweeks**. Anyone claiming a validated captaincy edge on two seasons of data
is reading noise. Rank against the crowd is where the measurable money is, not
the choice of criterion.

### Also measured, too small to ship

* **Manager rotation tendency** (churn in the XI between consecutive matches,
  trailing 5, as an xMins feature): log-loss −0.0011 in *both* seasons, so the
  effect is real, but every downstream metric is inside noise. Left out.
* **Opponent / own-team defensive strength as xMins features**: −0.0010
  log-loss in 2025-26, +0.0002 in 2024-25. Inconsistent. Left out.

### What cannot be settled with this repo's data

Stated plainly so nobody re-runs the experiment expecting a number:

* **Predicted-lineup, press-conference and injury-duration models.** No free
  feed here carries manager quotes, injury type or expected return date, and
  FPL's `status`/`chance_next` is a *current* snapshot with no history — so
  even the availability overlay cannot be backtested honestly.
* **Manager identity and manager-specific rotation.** Not in any feed the
  pipeline reads; the club-level proxy above is the closest reachable, and it
  is worth ~0.25% of log-loss.
* **Tactical regimes / positional role changes.** Needs Understat resolved
  (`pull --understat` — `player.understat_id` is otherwise NULL and every
  Understat feature is silently NaN) and really needs event data with pitch
  locations, which no free source here provides.
* **Reinforcement learning over transfer sequences.** Buildable, but note the
  captaincy result above: if a +0.4 pts/week captaincy edge needs 2,000
  gameweeks to distinguish from zero, an RL agent's advantage over the existing
  MILP cannot be validated on 4 seasons either. It would be an unfalsifiable
  addition, which is the opposite of the discipline this file is for.

### Understat: joined at last, and what it was worth

`player.understat_id` was NULL for **every** season, so every Understat column
the feature builder emits was NaN and no Understat idea had ever actually been
testable. Two defects caused it, both now fixed:

* `_pull_understat` resolved only the *current* season. It now resolves every
  backfilled season too (one cached request each) and then fills across
  seasons on the stable `player.code`. Coverage: **83%** of players per season.
* the per-match ingest dropped three fields the endpoint returns —
  **`npg`, `npxG`** (the non-penalty split) and **`position`** (the role he
  played in that match: AMR, DMC, FW…). All three are now stored.

Penalties are 5.5% of league xG but up to **32% of Cole Palmer's and 31% of
Bruno Fernandes's** — a premium asset whose xG the engine was scaling by the
fixture as if all of it were open play, and whose loss of penalty duty would
have been invisible. Understat's per-match `position` is also far richer than
FPL's label: only 15% of "MID" minutes are actually MC — 15% are DMC, 11% AMC,
7% AML, 7% AMR.

**Two-stage evaluation, and why it matters.** Stage 1 asked the information
question directly, away from the engine: predict a player's realised attacking
points per 90 over his *next six appearances* from point-in-time rate
estimates, ridge, trained forward in time, ~17k player-gameweeks. Blocks added
cumulatively, RMSE on the resolved subset, two independently held-out seasons:

| block | 2024-25 | 2025-26 | verdict |
|---|---|---|---|
| FPL xG/xA only | 1.0847 | 1.0044 | baseline |
| + Understat npxG, xA | 1.0615 | 0.9829 | **both** |
| + penalty split | 1.0556 | 0.9819 | **both** |
| + shots × npxG-per-shot, key passes × xA-per-key-pass | **1.0425** | **0.9684** | **both** |
| + xGChain / xGBuildup | 1.0436 | 0.9708 | worse in both — kill |
| + role share / role change | 1.0413 | 0.9723 | inconsistent — kill |

So the decomposition is **3.9% / 3.6% better RMSE** at estimating the rate,
replicated. The information is real.

Stage 2 put that estimate into the engine. **It changes no decision**:
`spearman_played` +0.0001, prec@20 −0.003, top-30 −0.03, captain −0.15 — every
one p > 0.19 over 74 paired gameweeks. A 4%-better attacking-rate estimate is
invisible under the engine's noise floor, which is minutes. That is a
quantitative answer to "how good does a rate estimate have to be to matter
here?" — better than 4%.

**The calibration trap (the most transferable lesson of the round).** The first
Stage-2 attempt made the engine significantly *worse* — `spearman_played`
−0.0029 (p=0.0005), prec@20 −0.0095 (p=0.015), captain −0.53. The cause was
not Understat: my rate module shrank every player toward a single league-wide
prior instead of a per-position one, which inflated goalkeeper xG by **~500×**
and defender xG by 1.8× while deflating forwards to 0.77×. Stage 1 never saw
it, because a fitted model rescales its inputs freely — **the engine consumes a
rate as an absolute quantity and cannot.** After per-position priors and a
level calibration (Understat supplies the ordering inside a position, FPL keeps
the level) the harm disappeared entirely.

Generalise it: *any* estimator injected into the structural engine must be
calibrated to the scale the engine assumes, not merely correlated with the
truth. `rates.py` blends xA 50/50 with realised FPL assists for exactly this
reason — FPL's assist definition is broader than Opta's xA — and substituting
pure Opta xA silently lowballs every creator.

**Not shipped into the engine.** The rates stay on FPL's own expected stats:
the Understat version is a better estimator that demonstrably changes nothing,
and the standing rule is that a change with no measurable gain does not earn
its dependency. The *data* fixes ship, because they cost nothing, repair a real
defect, and are the precondition for anything role- or set-piece-shaped later.

### xMins, round two: E[minutes] as its own estimator

Since a perfect minutes model is worth +0.21 Spearman and nothing else comes
close, exposure deserves to be estimated rather than reconstructed. The
classifier stays exactly as it is (its log-loss is unchanged); only the way
E[minutes] is built from it changed:

| E[minutes] estimator | MAE 24-25 | MAE 25-26 | Δ spearman_played |
|---|---|---|---|
| class means (what it was) | 14.10 | 12.89 | — |
| isotonic-calibrated classes | 13.89 | 12.92 | **−0.0012 (p=0.0004)** |
| a single minutes regressor | 14.00 | 12.64 | +0.0002 (p=0.72) |
| **P(plays) × E[min \| plays]** | **13.67** | **12.42** | **+0.0010 (p=0.010)** |

The hybrid ships. Calibrating the class probabilities does **not**: it buys
E[minutes] accuracy in one season and costs rank quality significantly, which
is a good reminder that the classifier's probabilities are used for clean
sheets and appearance points as well as for exposure.

### Set pieces and penalty duty

Understat's `getPlayerData` carries every shot a player has taken, with the
`situation` it came from (OpenPlay / FromCorner / SetPiece / DirectFreekick /
Penalty) and **who assisted it** — which is what identifies duty from play
rather than from a snapshot. `understat_shot` now holds 77,873 of them
(~100 penalties and ~1,650 corner shots a season, ~7,000 shots a season with a
named assister).

**Duty is only weakly inferable from history.** Standing at each penalty a club
won and predicting the taker from prior shots alone, over 257 events:

| rule | top-1 |
|---|---|
| decayed share of the club's penalties | **55.3%** |
| whoever has taken the most | 52.9% |
| whoever took the last one | 48.6% |
| the club's best open-play shooter | 33.5% |

The model wins, but only just — because **the taker changes between one
penalty and the next 35% of the time**. Against FPL's own published order for
the live season, the shot-derived estimate agrees on 12/18 clubs for penalties
(67%), 11/20 for corners (55%) and 9/20 for direct free kicks (45%).

The conclusion is the practical one: **FPL's `penalties_order` /
`corners_and_indirect_freekicks_order` / `direct_freekicks_order` are better
than anything the shot log can infer**, they are free, and the repo was not
storing them. They now sit on the `player` table. They are a current-season
snapshot with no history, so nothing built on them can be backtested — that
limitation is real and is why the shot-derived share exists at all.

**The set-piece decomposition does not improve the rate estimate.** Added as a
block to the provenance test (pen share, club penalty rate, open-play vs
set-piece xG, set-piece creation share): RMSE −0.8% in 2024-25 but **+0.4% in
2025-26**, and no gain in the segments it was built for (penalty-involved
+0.9% / +0.5%; on-duty +0.1% / +5.1%). Inconsistent, so it does not ship as a
rate input. The reason is straightforward once seen: a player's realised xG
already encodes his duty *for as long as he holds it*, and duty rarely changes
inside a six-appearance window.

**What did ship is a bug fix.** The live path added a flat `+0.10` xG90 to
every first-choice penalty taker — on top of a trailing xG that already
contained the penalties he had been taking. For a premium midfielder that
double-counts about **0.43 points a match**, against a true duty term of
0.13 penalties/match × 0.71 × 0.79 ≈ 0.35 points, most of which the history
already carried. It also did nothing whatsoever for a player who had just
*lost* the duty and whose trailing xG was still inflated.

`xpts/set_pieces.py` replaces it with a correction:

    delta = team_pen_rate × (published_share − historical_share) × PEN_XG

**zero whenever today's duty matches the history**, so it cannot reshape the
board; it moves only the 83 players (of 614) whose duty has actually changed —
44 up, 28 down. Every constant is measured over 358 penalties in 64
club-seasons, not chosen: a club wins 0.126 penalties per match, and its
first/second/third choice take 71% / 21% / 6% of them.

Two guards make it safe. It returns zero when the season has **no** published
order at all — in a replay every value is NULL, which would otherwise read as
"nobody is on duty" and strip the penalty component from every taker who ever
had it — and zero when there is no shot history to correct against, falling
back to the old flat boost. Backtests are therefore bit-identical.

### Two more defects the round turned up

* **Every `pull` was wiping the Understat resolution.** The bootstrap ingest
  wrote `understat_id: None` into an `INSERT OR REPLACE`, so a plain
  `python -m fpl_engine pull` silently undid the entity resolution. It now
  carries the existing value through.
* **The resolver could give two players the same Understat id.** Nothing
  stopped two different FPL players each matching the same id independently —
  "Gabriel"/"Gabriel Jesus", "McConnell"/"McDonnell", "Amad"/"Diallo". That
  hands one player another's shot history, which is worse than having none, so
  a collision now unsets **both** and reports them ambiguous. 27 assignments
  across the five seasons were affected.

### Round 6: what the distribution is actually worth

The simulator's value cannot be measured on chip decisions directly — one chip
a season is two observations. Its *calibration* can, on 55,263 player-gameweeks
and 74 squad-gameweeks, and every decision built on it inherits that.

**Where it was wrong.** The point-estimate version was underdispersed, which is
the classic plug-in error: it treated every shrunk rate as if it were known.

| | before | after | truth |
|---|---|---|---|
| P(haul ≥ 10), overall | 0.0146 | 0.0153 | **0.0186** |
| starting-XI sd | 15.17 | 15.70 | 16.44 |
| XI outcomes outside the predicted 10-90 band | 20.3% | 14.9% | 20.0% |

Two fixes. Team goals are drawn through a gamma mixture (measured var/mean
1.078 over 3,060 team-matches, so Poisson is close but slightly thin), and the
unmodelled residual is drawn rather than added as a constant — it is lumpy
events, not a trickle. The important one is the **goal and assist shares are
now drawn from a Dirichlet** centred on the modelled shares.

That last detail is worth keeping. The obvious construction — multiply each
player's weight by a mean-1 gamma and renormalise — is *not* a Dirichlet
unless every scale matches, and the Jensen gap shrinks the dominant player's
share. It moved the striker's mean while barely widening his spread: the exact
opposite of the intent. `_dirichlet_like` preserves E[share] exactly.

**What remains, and why.** P(haul ≥ 10) is still ~18% light overall, but split
by how certain the start was, the ratio of realised to predicted falls
monotonically:

| simulated P(plays) | n | predicted | realised | ratio |
|---|---|---|---|---|
| < 0.50 | 33,033 | 0.0014 | 0.0035 | **2.50** |
| 0.50-0.80 | 7,328 | 0.0152 | 0.0177 | 1.17 |
| 0.80-0.95 | 12,383 | 0.0405 | 0.0462 | 1.14 |
| ≥ 0.95 | 2,519 | 0.0746 | 0.0838 | **1.12** |

Most of the gap is the backtest's own doing: replays disable the availability
overlay, so the simulator hands chances to players who were in fact injured or
dropped, thinning the tails of the men who actually started. For a nailed
starter the residual error is ~12%, and that is the number to remember when
reading `p_haul` or `ceiling`.

**Chip payoffs are calibrated.** Independently of the above, over 74 replayed
gameweeks: Triple Captain predicted 6.85 against a realised 7.45 (sd 5.62 vs
5.78), Bench Boost predicted 18.57 against 18.81 (sd 8.32 vs 8.04).

**The one chip decision a distribution actually changes.** For a linear payoff
the distribution is irrelevant — Triple Captain's extra captain score and Bench
Boost's bench total are both linear, so E[gain this week] is the same whether
it comes from the samples or from the analytic engine. It matters only when the
objective is nonlinear, and "the best week out of the next N" is:

| chip | gws left | max E[gain] | E[max gain] | **premium** | realised best week |
|---|---|---|---|---|---|
| Triple Captain | 5 | 8.33 | 13.73 | **5.40** | 14.48 |
| | 10 | 9.13 | 16.13 | **7.00** | 18.07 |
| | 20 | 10.47 | 18.53 | **8.06** | 21.71 |
| Bench Boost | 5 | 20.53 | 28.56 | **8.04** | 26.61 |
| | 10 | 21.09 | 31.64 | **10.54** | 28.04 |
| | 20 | 22.38 | 34.50 | **12.12** | 30.71 |

E[max] tracks the realised best week closely, which is the check that matters.
The premium is exactly what `chip_reserve` was guessing at 15.0 — roughly
**twice** the truth for Triple Captain at a realistic horizon, and it should
shrink as the season runs out (4.4 with three gameweeks left, not 15). Over-
reserving makes the solver hoard a chip it should have played.
`chips.chip_reserve_for` now returns the measured curve; Wildcard and Free Hit
keep the flat heuristic, because a whole-squad rebuild is not a one-week total
and nothing here measures it.

The decision *gain* from that correction is still unmeasurable at two chips a
season. What is measurable, and now measured, is that the number it replaces
was wrong.

### Round 7: what £1m is actually worth

The price model discriminates strongly — the top-10 ranked risers rise 67-75%
of the time against a 2% base rate — but a signal is only worth what it
converts to, and converting it needs an exchange rate between £m and points.
Inventing one ("£1m = X points") would quietly distort every transfer, so it
was left out of the objective in Round 3. It does not need inventing: the
value of budget is the marginal value of that constraint in the squad-selection
problem, and that is computable.

Rebuilding the best legal 15 (2/5/5/3, at most 3 per club, a legal XI and a
captain) under a budget and then again with more, across 18 replayed
gameweeks:

| extra budget | +0.5m | +1.0m | +2.0m | +4.0m |
|---|---|---|---|---|
| points per £1m per gameweek | **0.163** | 0.117 | 0.107 | 0.093 |

Diminishing, as it must be. Price moves are small (±0.1-0.3m), so the marginal
0.163 is the rate that applies to them.

The realised check on the same squads is **-1.2 ± 1.0 points and settles
nothing**: with a per-gameweek sd of 4.1, confirming an 0.08-point effect would
need roughly 19,000 gameweeks. Worth stating plainly, because it is the same
wall as captaincy and chip timing — the modelled figure is the usable one.

**The chain, with every link measured.** A rise is realised only on sale, and
FPL returns the purchase price plus **half** the profit. So

    points = expected move (£m) x 0.5 x 0.163 x gameweeks remaining

`price_model.points_value` does exactly that, and `prices` now prints it. The
answer is deliberately deflating: the strongest riser the model can find this
week is worth **0.23 points** with 37 gameweeks left, and about 0.03 by the
run-in. Over a season of ~30 transfers, consistently taking the ranked riser
over a neutral alternative is worth on the order of **3 points** — real, but
two orders of magnitude below what the hit rate makes it look like.

That is the whole reason to do the conversion. A 67-75% hit rate against a 2%
base rate is a genuinely strong model; priced properly it is a tie-breaker.
It stays out of the solver's objective and is reported alongside it, which is
where a 0.2-point term belongs.

### Round 8: the minutes ceiling is mostly not reachable

Perfect minutes is worth +0.21 Spearman, which made xMins the obvious place to
push. This round asked the sharper question — *can minutes be predicted better
at the deadline?* — and the answer is largely no, for a reason worth writing
down rather than rediscovering.

**The start model is already at AUC 0.944-0.954, and it is calibrated.**

| P(start) band | n | actually started | MAE | share of all E[min] error | share of rows |
|---|---|---|---|---|---|
| ≤ 0.02 | 22,699 | 0.004 | 1.1 | 3.2% | 39.8% |
| 0.02-0.10 | 8,291 | 0.066 | 11.5 | 12.7% | 14.5% |
| 0.10-0.30 | 6,372 | 0.208 | 24.4 | 20.8% | 11.2% |
| **0.30-0.70** | 8,044 | **0.554** | 32.1 | **34.6%** | 14.1% |
| 0.70-0.90 | 8,756 | 0.867 | 20.9 | 24.5% | 15.4% |
| ≥ 0.90 | 2,868 | 0.947 | 11.0 | 4.2% | 5.1% |

Read the second and third columns together. The ambiguous middle is **14% of
rows but 35% of the error**, and in that band the model says 0.55 and reality
delivers 0.554. It is not mis-fitted; it is correctly reporting that the
manager has not decided. Nothing available at the deadline resolves that,
because the team sheet lands an hour before kickoff.

**Sharpening has been tried five times** and each attempt returned ≤0.25% of
log-loss: club rotation tendency, opponent strength, Understat match role,
isotonic calibration, and now an explicit start/sub decomposition. The last is
worth recording because it is the structurally *better* model —
P(start) x E[min|start] + P(start=0) x P(on) x E[min|sub], which stops
conflating a starter hooked at 55 with a substitute who came on at 60 — and it
still loses: E[minutes] MAE improves slightly (13.80→13.68, 12.46→12.42) but
P(60+) gets worse (Brier 0.0895→0.0934) and `spearman_played` drops 0.0025
(p=0.005). Clean sheets and appearance points need P(60+) more than exposure
needs the last half-minute of MAE.

**The one lever that is real is live-only.** FPL publishes `status` and
`chance_next` before the deadline, and that *is* team news. Backtests must
disable it (the stored status is today's, not that gameweek's), so no replay
here can measure it — but on the live GW2 it moves 80 of 614 players and 713
minutes of exposure, and correctly zeroes out the injured. The gap between
what the backtest measures and what the live model does is that overlay.

So: minutes remains the largest source of error, and most of what is left is
irreducible at the moment the decision is made. Treat "more work on xMins" as
answered unless a genuinely new information source appears — a predicted-lineup
feed being the obvious one.

### Round 9: invariants, because four identity bugs was enough

Every identity defect this project shipped was silent. The resolution never ran
for backfilled seasons; every `pull` blanked `understat_id`; one Understat id
was handed to two players in all four seasons; price columns mixed £m with
tenths. None of them raised anything — the pipeline produced NaNs, or worse,
plausible numbers.

`python -m fpl_engine verify` now checks them and exits non-zero:

* identity is one-to-one **both ways**, and stable across seasons on
  `player.code`
* `player_gw` rows resolve to a player; shots resolve to a shooter
* `player_gw.price` is tenths and `player.now_cost` is £m — the two are never
  mixed inside a team+position group, which is what corrupted the depth chart
* minutes ∈ [0,120], team goals ∈ [0,15]
* no played match without a kickoff time; nothing in the future already
  carrying minutes; no shot dated outside the season it is filed under

Each is covered by a test that breaks the data on purpose.

**And it immediately found a live measurement bug.** FPL's **Assistant
Managers** are selectable entries that score points and never play a minute —
20 of them in 2024-25, 322 rows, **1,861 points**. They were sitting in the
backtest's actuals, occupying slots in the "actual top 20" that no player model
can reach: in 16 of 38 gameweeks an average of 4.75 and up to **8 of the 20**
top-scoring slots. Every model in the comparison was being marked against an
unreachable benchmark. Excluding them (74 paired gameweeks):

| metric | before | after | delta | p |
|---|---|---|---|---|
| spearman | 0.7123 | 0.7221 | **+0.0099** | <0.0001 |
| prec@20 | 0.1899 | 0.1986 | **+0.0088** | 0.006 |
| rmse | 1.9889 | 1.9082 | **−0.0807** | <0.0001 |
| spearman_played | 0.3855 | 0.3854 | −0.0001 | 0.61 |

That last row is the check that the fix is real rather than a reshuffle:
`spearman_played` already filtered on `minutes > 0`, so it was the one metric
Assistant Managers could never contaminate — and it is the one that did not
move. `verify.PLAYER_POSITIONS` is now the single definition, applied in the
backtest actuals, the minutes frame and the price panel.

### Round 12: game state is real, and already priced in

A static Poisson draw says a team attacks the same way at 0-0 and at 3-0. It
does not. Reconstructing the running score from the shot log (every goal is a
shot with a minute; 1,160 matches where the reconstruction exactly matches the
true final score, 29,776 shots) and comparing each team against **its own**
output while level — so team strength, opponent and venue all difference out:

| score difference | xG/90 vs level | shots/90 | xG per shot |
|---|---|---|---|
| 2+ down | 1.251 | 1.333 | 0.939 |
| 1 down | **1.330** | 1.287 | 1.033 |
| level | 1.000 | 1.000 | 1.000 |
| 1 up | **0.729** | 0.760 | 0.960 |
| 2+ up | 0.762 | 0.755 | 1.009 |

A chasing side takes ~29% more shots of the *same quality*; a leading side
~24% fewer. It is a volume effect, not a quality one.

**The within-match design is the whole result.** The naive split — comparing
states across all teams — reports the opposite sign, that a team two goals up
generates 1.26x the xG. That is not game state, it is that a team two goals up
is disproportionately a good team playing a bad one. Differencing against the
team's own level-score output flips 1.26 to 0.76.

**And it changes nothing, for a reason worth keeping.** The team model's
lambda is fitted on *realised* goals, which by construction already contain
every game-state dynamic that happened. Layering the effect on top re-applies
what the estimate absorbed. Scored on 1,480 replayed team-matches, the
predicted distribution of a team's goals is equally calibrated either way —
PIT uniformity chi2 **9.9** static against **8.9** path-dependent, on a 16.9
critical value. Noise.

`simulate.GAME_STATE` is therefore **off**, with the draw kept, tested and
switchable: the effect is real, and the conclusion is specific to a lambda
estimated from goals. A team model built on a basis that does not already
absorb it would need this test re-run.

#### Round 13: MFRU — the rank objective, priced and measured

The projection side is saturated (Rounds 8/8b), so this round formalised the
*decision* side: what expected points are worth to a manager's **rank**.
`xpts/rank_utility.py` (MFRU, Mean-Field Rank Utility) scores a full decision
— XI, captain, vice, bench order — on the differential against the ownership
mean-field inside each of the simulator's correlated draws:

    Delta = S(d) - sum_i eo_i X_i,     U_gamma(d) = E[Delta] + gamma sd(Delta)

with FPL's automatic substitutions and the armband transfer applied per draw.
Three structural facts, the first a theorem rather than a finding:

1. **E[Delta] does not depend on effective ownership** — the mean-field term
   is decision-free, so a risk-neutral manager maximising expected rank
   return should pick exactly the max-xP team. Every "EO edge" lives in the
   variance term: gamma > 0 buys rank variance (differentials), gamma < 0
   shadows the template. Chasing differentials for their own sake costs
   points in expectation, provably.
2. The only channels where a risk-neutral decision can legally differ from
   max-xP are the **autosub branch** (a fringe starter is insured by a nailed
   first sub) and the **vice-captain branch**. Both are nonlinear in the
   joint minutes draw, which is what the simulator exists to price.
3. gamma reshapes the rank *distribution*, not its mean — so a realised-mean
   backtest cannot validate it even in principle, only bound its cost.

`rank-backtest` replays a season with the squad held fixed to the lagged-
ownership template — the same controlled design as the captaincy study — so
arms differ only in the objective; realised scoring applies the real autosub
rules. Over 74 paired gameweeks (2024-25 + 2025-26, `compare-rank-backtests`):

| arm vs the crowd's own XI+captain | pts/gw | p |
|---|---|---|
| max-xP (the current rule) | **+2.11** | 0.027 |
| MFRU gamma=0 | **+2.38** | 0.008 |
| MFRU gamma=+0.3 / -0.3 | +2.39 / +1.82 | 0.012 / 0.036 |

| arm vs max-xP | pts/gw | p |
|---|---|---|
| MFRU gamma=0 (autosub+armband channels) | +0.27 | 0.63 |
| every gamma in ±{0.15, 0.3} | -0.28 … +0.28 | ≥ 0.51 |
| sim-mean instead of analytic mean | -0.30 | 0.58 |

Read it plainly: **the decision layer's edge over the crowd is real,
replicated and now measured at the full-XI level** (~+2.1-2.4 pts/week, or
~80-90 points a season, consistent with the captaincy study's +0.7-1.1 from
the armband alone). **The incremental edge of MFRU over max-xP is unproven**:
+0.27/week pooled, sign flips between seasons (+1.6 in 2024-25, -1.1 in
2025-26). The autosub channel changes the XI or bench in a quarter of
gameweeks and the captain in ~28%, yet the realised means cannot separate it
from the current rule at n=74 — the same wall as captaincy criteria and chip
timing. And the gamma result is the theorem working as stated: risk pricing
is not supposed to move the mean. Validating a nonzero gamma needs rank
distributions over thousands of gameweeks, which this repo does not have.

So the model ships as measurement infrastructure and as the formal answer to
"where is the remaining edge": it is in playing the decision layer at all
(which max-xP already does), not in EO-adjusting it. MFRU's gamma stays 0 —
i.e. behaviourally the current rule — until a rank-tail objective can be
tested honestly. Unit tests pin the autosub operator's FPL rules
(like-for-like GK, formation-blocked subs, bench-order priority, armband)
and the gamma=0 = max-xP equivalence (`tests/test_rank_utility.py`).

### Round 14: the decision layer at n=111 — max-xP survives everything

Round 13 scaled up per the research mandate: a third replay season
(2023-24, minutes model trained on 2022-23 only), the full baseline battery
(random-valid, a competent-human heuristic from deadline-public information
only, the crowd template), gamma widened to ±1, and three direct rank
functionals — maximise P(beat the field), CVaR20 (downside), Q80 (upside) —
since points are a proxy and rank is the game. `RESEARCH_LOG.md` carries the
pre-registered hypotheses and every number; the short version, 111 paired
gameweeks vs the max-xP baseline:

| arm | pts/gw vs xp | p |
|---|---|---|
| random valid decision | **−6.08** | <0.0001 |
| competent-human heuristic | **−1.77** | 0.041 (same sign every season) |
| crowd template | **−2.17** | 0.0027 |
| mfru_g0 | +0.13 | 0.78 |
| every gamma in ±{0.15, 0.3, 0.5, 1} | −0.41 … +0.23 | ≥ 0.32 |
| P(beat field) / CVaR20 / Q80 objectives | −0.61 / −0.58 / −0.37 | ≥ 0.33 |
| bench-order channel alone (`xp_bench`) | −0.05 | 0.60 |

Three conclusions worth pinning:

1. **mfru_g0's +0.27 was noise** — it shrank to +0.13 (CI −0.76…+1.01) when
   n grew from 74 to 111. The autosub/armband channels are real mechanics
   with no measurable realised value at the weekly fixed-squad layer.
2. **No implementable rank objective beats the mean.** Combined with the
   Round 13 theorem (E[Delta] is EO-free), "maximise expected points" ≈
   "maximise expected rank return" holds both structurally and now
   empirically at this layer. The remaining rank question — tail objectives
   over a season — needs rank distributions no free data provides.
3. **The leak audit earns its keep.** vaastav's archived `xP` (FPL's own
   published ep_this) looked like the one season-long public benchmark and
   "beat" the engine by +2.6/gw — then failed the audit: spearman_played
   0.58/0.51 in 2023-24/2024-25, which is this repo's *perfect-minutes
   oracle* level, and 89-94% precision at zeroing exact non-players. It
   embeds realised minutes (and is near-random in 2025-26), so it is
   excluded; a benchmark that looks too good gets audited before it gets
   believed. No deadline-honest historical commercial archive exists
   (theFPLkiwi's stops Dec 2023 with 4 files) — the forward-collected elite
   panel remains the only honest external benchmark.

So the production decision rule stays **max-xP on the analytic engine**, now
defended at n=111 against seventeen challengers and three human-shaped
baselines rather than assumed. The measured hierarchy — random −6.1, human
−1.8, crowd −2.2 per week against it — is the honest answer to "would this
make a manager's decisions better": yes, by about 2 points a week over a
competent manual pick, roughly 80 points a season, before transfers.

## Sportmonks: audited, not usable

Two independent hard blockers on the free plan, both verified against the API
rather than inferred from documentation:

* **No Premier League.** The key resolves four leagues — Danish Superliga,
  Scottish Premiership and their play-offs. `/leagues/8` answers *"you don't
  have access to it via your current subscription"*.
* **No expected lineups.** The `expectedLineups` include returns **HTTP 403**
  even on a league the key *can* read, so the highest-priority feature is
  gated independently of the league restriction. `predictions`, `xgfixture`
  and `pressure` are likewise 403.

What the free tier does expose, for those two leagues: `lineups`,
`lineups.details`, `events`, `statistics`, `formations`, `referees`,
`weatherReport`, `odds`, `sidelined`, with seasons back to 2005/06.

**Before paying for a tier, verify one thing**: that the expected XI is
*archived per historical fixture with a timestamp*, not merely served for
upcoming matches. A prediction that is not preserved as it stood before each
past deadline cannot be backtested, and the Round 8 result means an
unbacktestable minutes feature is worth very little — the current start model
is already calibrated at AUC 0.95, so a new source has to be demonstrated
better, not assumed.

## Data acquisition (`acquire/`)

An independent collector. It writes raw responses and normalised observations
into the same SQLite file and stops there — it imports no model code, computes
no features, and a test enforces that (`test_the_acquirer_does_not_import_the_model`).
The modelling engine reads these tables; nothing reads back.

    python -m acquire pull --source fpl --season 2026-27
    python -m acquire backfill --source fpl     # replay archived payloads
    python -m acquire validate                  # non-zero exit on error
    python -m acquire status
    python -m acquire as-of --season 2026-27 --at 2026-08-28T17:00:00Z

### What the source survey actually found

Checked against the sites, not the documentation:

| source | finding |
|---|---|
| **FBref / Sports Reference** | returns **403 to `robots.txt` itself** for non-browser agents. Actively refusing automated access. Not scraped. |
| **Understat** | `robots.txt` is `User-agent: * / Disallow: /`. The repo **already** depends on it (`ingest/understat.py`), which predates this work — flagged here rather than quietly extended. No new Understat scraping was added. |
| **Transfermarkt** | permissive `robots.txt`, but its terms prohibit automated extraction. robots.txt is not a licence. Not scraped. |
| **Premier League official** | crawlable (24 disallow rules, almost all query-string patterns). Viable if a need appears. |
| **FPL API** | `robots.txt` present, no disallow rules, already used by the pipeline. |

### Why FPL's own feed was the thing to build first

The modelling rounds established that expected minutes is the ceiling, that the
start model is already calibrated at AUC ~0.95, and that **the one unexploited
lever is team news**. FPL publishes exactly that, before the deadline, and the
pipeline was throwing it away — `status` and `chance_next` were overwritten on
every pull, which is why backtests must disable the availability overlay.

`elements[]` carries `status`, `chance_of_playing_next_round`, `news` (the text:
*"Thigh injury - 75% chance of playing"*) and **`news_added`, a source-side
publication timestamp**. No scraping, no third party, no terms question.

It is stored as a **change log**, not snapshots: a row is written only when a
player's state moves, so `as-of` is one indexed lookup and re-ingesting a
payload writes nothing. `news` is kept verbatim — *"75% chance of playing"* must
never be flattened to *"available"* on the way in.

Two timestamps, and the distinction is the whole design:

* `observed_utc` — when we saw it. Always safe for a backtest.
* `source_published_utc` — FPL's `news_added`. Because it survives, a snapshot
  taken later still dates a standing item, which recovers *some* of the past
  without having been there.

`backfill` replays the bootstrap payloads `raw_snapshot` had been archiving all
along without anyone parsing them: **864 observations over 614 players, back to
2026-07-26**, from data already on disk.

### The limitation, stated plainly

**This is not backtestable yet, and no amount of scraping fixes that.**
Availability for 2024-25 and 2025-26 was overwritten and is gone; nobody can
recover it. The change log becomes testable roughly a season after it starts
running. Until then it is a live-only signal, exactly like the availability
overlay it feeds.

What it already shows is why archiving matters: **140 of 614 players changed
state within one month**, and the text moves too — J. Timber went from
*"Expected back 21 Aug"* to *"Unknown return date"*, which is a real signal that
a snapshot table destroys.

### Not integrated

Per the brief, the modelling engine is untouched. The first integration, when
there is enough history to judge it, is availability → xMins — and it must clear
the same bar as everything else: a paired backtest, or it does not ship.

### Round 8b: a predicted-lineup model of our own

Building P(start) from our own team-sheet history, rather than scraping
somebody else's prediction. The squad-state features not previously tried:
days since he last **started** (distinct from last appeared), the calendar
*ahead* (days to next match, matches in the next ten days), rest relative to
the opponent, and his share of the club's starts at his position.

| model | AUC 24-25 | AUC 25-26 |
|---|---|---|
| three-class proxy (already shipped) | 0.9446 | 0.9538 |
| dedicated P(start) | 0.9441 | 0.9529 |
| **+ the new squad-state features** | 0.9441 | 0.9536 |
| new features alone | 0.9144 | 0.9280 |

The new features carry real signal on their own and **nothing incremental**.
That is the sixth independent attempt at sharpening this model, all returning
≤0.25%. P(start) from this data is saturated at AUC ~0.95, and Round 8's
explanation still holds: the residual sits in the ambiguous middle, where the
model is already calibrated because the manager has not decided.

**What did ship is a reporting fix.** The web UI was displaying `start_rate` —
the share of a player's last ten matches he started. Against the model the
engine already runs:

| signal | AUC | log-loss | Brier |
|---|---|---|---|
| UI's trailing start rate | 0.901 / 0.905 | 0.574 / 0.535 | 0.116 / 0.111 |
| the engine's own model | 0.945 / 0.954 | **0.277 / 0.250** | 0.086 / 0.078 |

More than twice the log-loss. The number on screen was materially worse than
what the pipeline already knew — Tanaka Ao showed 70% (seven of his last ten)
against a model probability of 14%.

`p_start` is now published by the minutes model, carried through the engine's
per-player frame and shown in the UI. It is calibrated (predicted 0.503 →
realised 0.507 in the coin-flip band) and the availability overlay scales it,
so a player ruled out reads 0. It is **not** claimed to be more accurate than
the 60-minute class it sits beside; it answers the question people actually
ask, on a scale they can read.

### Do substitutes score differently per minute?

Yes, and it still must not be used. Within player, controlling for who gets
picked to start:

| | ratio, sub vs starting |
|---|---|
| points per 90 | 1.259 |
| …but appearance points inflate short outings, so **G+A per 90** | 1.233 |
| late-match minutes simply contain more goals | 1.105 |
| **residual role effect** | **1.116** |

A real ~12% edge in attacking output per minute, after controlling for both
squad quality and the clock. Applied to the engine as a multiplier on the
substitute share of expected attacking output it makes decisions **worse**:
`spearman_played` −0.0006 (p=0.0002) at 1.116x, −0.0011 (p=0.0005) at 1.25x.

The reason is the one this project keeps rediscovering: `xg90` is estimated
from *all* of a player's minutes, his substitute minutes included, so the
higher productivity is already inside the rate. Adding a multiplier
double-counts it — the same failure as Understat rates, the set-piece
decomposition and game state. **Any effect measured in realised outcomes is
already absorbed by rates estimated from realised outcomes.** Four independent
confirmations now; treat it as a standing rule before building the fifth.

### Tracking elite managers: buildable, not yet testable

The hypothesis is that skilled managers collectively hold forward-looking
information the model cannot derive from history — team news, the eye test,
something a beat reporter said on Friday. It is the same shape as the betting
market, which was measured to *encompass* our team model outright, so it
deserves a test rather than an assumption.

**Backtestable? No.** `entry/{id}/event/{gw}/picks/` takes no season parameter
and serves only the season in progress; previous seasons 404.
`entry/{id}/history/` keeps past seasons as totals and ranks, never squads. So
2024-25 and 2025-26 picks are gone, and the current season has one finished
gameweek. Forward collection is the only route — the third time this wall has
appeared, after availability and expected lineups.

**"The top 1000" is the wrong panel.** Sampled after one gameweek, the Overall
top 250 had a **median past-season rank of 2.6 million**; 78% had a career
median worse than a million, only 19% had *ever* finished inside the top 100k,
and 15% were brand-new accounts. That table ranks luck, not skill. The panel is
therefore graded on **past seasons only** — top 100k in at least two of them —
which found 59 proven managers among the 700 enumerated, or 8.4%.

**And the first result off it was an artefact of my own search.** The panel's
GW1 differentials looked spectacular: ownership correlating +0.44 with points
against +0.28 for overall ownership, and the 30 biggest differentials averaging
8.2 points against 2.0 for the most avoided. Then the obvious check: those 59
managers scored **103-116 in GW1, mean 107.1, against an FPL average of 50**.
Of course they did — they were found by walking the Overall table, and after
one gameweek you only sit there if that gameweek went well. The filter never
looked at the current season, but the *enumeration* did, and the sample is
conditioned on the very outcome being measured.

The fix is that the panel is fixed once built. From GW2 onward these managers
are followed regardless of results, so their picks precede the outcome and are
out of sample. Rebuild it in **pre-season only** — a mid-season rebuild would
re-introduce exactly this conditioning.

    python -m acquire panel --pages 20    # grade on past seasons, pre-season only
    python -m acquire picks --gw 2        # snapshot the panel, after each deadline

Roughly 10-15 gameweeks of collection before the differential signal can be
tested against the model at all.

## Optimiser

`optimise/milp.py` is a multi-period mixed-integer program (PuLP + bundled CBC,
free) over a rolling horizon. It jointly chooses squad, starting XI, captain and
transfers per gameweek to maximise **discounted expected points net of the -4
hit cost**. Free transfers accrue (+1/gw, bankable to 5; the stock starts at
**0** — transfers before the GW1 deadline are unlimited and bank nothing, so a
quiet GW1 leaves exactly 1 FT for GW2) and are modelled
explicitly, so the model decides whether a hit is worth it (a transfer is taken
only when its marginal XI gain over the displaced/benched player beats 4 points).
Constraints enforced as hard: £100m budget with the bank recursion, 2/5/5/3
squad, ≤3 per club, legal XI formation. Given an entry (squad) id it suggests
transfers from the current team; with no team yet (pre-season) it builds a fresh
squad from budget (initial selection is free). Default entry: `883566`.
Projections use current form applied to each horizon gameweek's fixture.

## Must stay green

* `scoring.points_without_defcon` reconciles **100%** of 2024-25 player-matches
  against actual FPL totals (the Phase-1 gate). Re-run before touching scoring.
* `features.build_samples` reproduces the FPL-sourced columns of
  `data/samples.csv` for windows 1/3/5/10 exactly, and window-38 within
  long-horizon tolerance (depends on how many backfill seasons are loaded).
* `predict.predict(samples.csv)` reproduces `data/predictions.csv` exactly.

Run: `python -m pytest tests/ -q`

## Known approximations (documented, not bugs)

* **`player relevant fpl points`** (5 columns): OpenFPL's exact definition is
  not reconstructable from this repo's artefacts, so a documented best-effort
  (`total_points - appearance_points`) is used. All other FPL columns match.
* **Understat resolution is 83% per season**, and the unresolved 17% keep
  `understat_id` NULL so their Understat features stay NaN (the FPL xG
  stand-ins apply). Resolution now runs for every backfilled season and
  fills across seasons on the stable `player.code`.
* **Understat** is ingested from its JSON endpoints (`getLeagueData/{league}/{year}`
  for every club's per-match xG/xGA/deep/PPDA in one call, `getPlayerData/{id}`
  for a player's per-match log across all seasons, `main/getPlayersStats/` for
  ids/names used in resolution; all need `X-Requested-With: XMLHttpRequest`).
  `pull --understat` (the web Data button has it on) covers the current and
  previous season; current-season players are fetched live, the rest cached.
  Understat features are NaN when it is unreachable; the models tolerate this
  via `np.nan_to_num`. To limit the damage, FPL's own (Opta) expected stats —
  stored per match in `player_gw.xg/xa/xgi/xgc` (plus per-gw `price`, the crowd signals `selected/transfers_in/transfers_out`, and raw DefCon counts `defcon/tackles/cbi/recoveries` from 2025-26 on) and `team_match.xg/xga` (team
  xG = summed player xG, xGA = opponent's) — stand in for the Understat
  metrics they map onto (`player xg/xa`, `team/opponent xg/xga`) whenever the
  Understat history is empty. Understat data takes precedence when present.
  Shots, key passes, xGChain/xGBuildup, deep and PPDA have no FPL equivalent
  and stay NaN.
* **Expected minutes** (`fpl_engine/minutes.py`) scale EP relative to the
  trailing-minutes baseline the features encode; across a season break a fit
  player is never down-weighted on stale absences (pre-season factor is in
  `[avail, 1.15·avail]`).
* **Clubs with no top-flight match log** (promoted) borrow a pooled prior from
  the previous season's relegated clubs for team/opponent features instead of
  NaN→0; clubs are matched across seasons by FPL's stable `team.code`.
* **Odds timing**: football-data's early-snapshot odds are collected days
  before kickoff — honest for a prediction made before the gameweek's first
  kickoff, though matches later in the gameweek carry a little extra market
  information. The Odds API covers upcoming fixtures only on the free tier
  (its historical endpoints are paid), so backtests rely on football-data.
* **Selling prices are reconstructed, not fetched.** The public
  `entry/{id}/event/{gw}/picks/` endpoint carries no `selling_price` — only the
  authenticated `my-team/{id}/` endpoint does. `manager.reconstruct_prices`
  derives them keylessly: price paid comes from `entry/{id}/transfers/`, or for
  a player never transferred in his season-start price
  (`now_cost - cost_change_start` from bootstrap), then FPL's rule (purchase +
  half the profit, rounded down; a fallen price sells at current value) is
  applied. Verified exact against an authenticated squad. Both optimisers
  refuse a £0.00 selling price outright — left silent it makes every sale raise
  nothing, so no transfer is affordable and the solver emits a plausible-looking
  "do nothing" plan.
* **League-rank / status-rank** columns are AM-only in OpenFPL and left NaN for
  player rows (matching the reference samples).

## Web app (FPL Review-style planner)

`app/` (FastAPI backend) + `web/` (React/Vite frontend) serve a local planner
UI on **http://127.0.0.1:8410** with four tabs: Planner (pitch + drafts),
Projections (per-GW model output table), Fixtures (FDR heatmap) and Solver
(chip-aware optimisation via `fpl_engine/optimise/chips.py` — a superset of
`milp.py` adding WC/FH/BB/TC chips, target/avoid/ban constraints, club-level
buy/sell rules and playstyle plans; `milp.py` itself stays untouched).

Solver specifics worth knowing:

* **Playstyles, not near-duplicates.** Asking for N plans returns one per
  preset in `chips.PLAYSTYLES` (Win now / Balanced / Patient) rather than N
  solutions of the same optimum, which differed only cosmetically. Styles vary
  *preferences* only — horizon `decay`, `ft_value`, and whether hits are
  allowed at all — never the rules: a -4 is always priced at -4, and the
  no-hits style forbids them via `allow_hits=False` instead of underpricing
  them. Compare plans on `ChipPlan.total_ep` (undecayed, net of hits), **never**
  on `objective` — each style weights gameweeks differently, so objectives are
  not comparable across them.
* **Club rules.** `banned_clubs` blocks *buying* from a club; `sell_clubs` also
  forces players already owned out by the end of the horizon, letting the
  solver choose the cheapest gameweek so the exit still uses free transfers.
* **Chips have option value.** A rolling-horizon optimiser sees a chip as pure
  upside and burns every available one inside the horizon (all four across
  GW2-6, for single-digit gains). `chip_reserve` prices what a chip is worth *saved*. For Triple Captain and
  Bench Boost that price is now **measured** by simulation rather than guessed
  (`chip_reserve_for`, see the Round 6 section) and decays as the season runs
  out; Wildcard and Free Hit keep the flat heuristic. Set a chip to 0 for the
  old always-play behaviour.
* **`chip_decay`** discounts one-week chip payoffs separately from `decay`
  (default 1.0). The horizon decay exists for uncertain far-future *transfer*
  planning; applying it to a chip made every chip drift to the first gameweek,
  since a 10-pt chip in GW5 scored 6.1 against 8 pts in GW2.

```
python -m app                      # serve the built site (needs app/static)
cd web && npm install && npm run build   # rebuild frontend -> app/static
cd web && npm run dev              # frontend dev server (proxies /api to 8410)
```

Projections are cached per (season, gw) in `data/web_cache/projections.json`
(invalidated by a data pull); drafts persist in `data/web_cache/drafts.json`.
A solve builds any missing projection gameweeks first, so the first solve of a
session is slow (model inference per GW) and later ones are fast.

## Commands

```
python -m fpl_engine init-db
python -m fpl_engine verify          # data invariants; non-zero exit on error
python -m fpl_engine pull            # FPL live + vaastav backfill + odds -> SQLite (free;
                                     #   set $ODDS_API_KEY for upcoming-fixture odds)
python -m fpl_engine predict --gw 1        # end-to-end predictions
python -m fpl_engine run --gw 1            # pull + build + predict
python -m fpl_engine optimise --entry 883566 --horizon 5   # transfers / squad
python -m fpl_engine prices                # who is about to rise / fall in price
python -m fpl_engine simulate --gw 2       # floors / ceilings / P(haul) + joint risk
python -m fpl_engine train                 # optional: retrain models (GPU-aware)
python -m fpl_engine predict --gw 1 --blend auto   # blend retrained + OpenFPL
python -m pytest tests/ -q
```
