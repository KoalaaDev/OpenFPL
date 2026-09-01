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

**That claim used to end "nothing else in the engine is worth a fraction of
that", and it was wrong — it had never been tested against any other oracle.**
Substituting each component in turn (see the decomposition below) puts perfect
attacking returns at **+0.23** `spearman_played`, ahead of minutes, and at
**+6.4 points per pick** against minutes' **+0.8**. Minutes is the largest
REACHABLE lever, not the largest lever. Two consequences still follow:

1. **Spend effort on minutes, not on rate constants.** A one-at-a-time sweep of
   every constant in `rates.py`, `team_model.py`, `engine.py` and
   `odds_model.py` (24 variants × 2 seasons) moved `spearman_played` by at most
   ±0.002 and top-30 by at most ±0.06 — all inside gameweek-to-gameweek noise.
   Those defaults are fine; leave them alone.
2. **Even a perfect minutes model leaves `spearman_played` at ~0.59.** FPL
   points are extremely noisy. Improvements of a few hundredths are real and
   worth having; anything claiming much more is a bug or a leak.

### The oracle decomposition: rank and points are different questions

Every component replaced with what actually happened, one at a time, 74 paired
gameweeks (`xpts_predict_gw(oracle=...)`, off on every shipped path). The
minutes row reproduces the +0.21 above, which is the harness's calibration
check.

| perfect knowledge of | spearman_played | top11 pts/pick | captain | reachable before the deadline? |
|---|---|---|---|---|
| everything | +0.605 | +7.30 | +10.18 | — |
| **attack (goals+assists)** | **+0.228** | **+6.36** | **+9.46** | no |
| goals alone | +0.134 | +5.68 | +8.72 | no |
| **bonus** | +0.130 | **+5.49** | +5.97 | no |
| clean sheets | +0.184 | +2.35 | +0.91 | partly |
| assists | +0.099 | +2.54 | +4.00 | no |
| **minutes** | **+0.200** | +0.76 | +0.34 | **yes — a lineup feed** |
| the 60-minute class alone | +0.170 | +0.58 | +0.39 | **yes** |
| DefCon | +0.057 | +0.70 | −0.09 | yes (it is a rate) |
| appearance points | +0.149 | +0.46 | +0.36 | yes |
| availability (who does not play) | **+0.000** | +0.28 | +0.26 | yes |
| conceded / saves / cards | +0.07 / +0.01 / +0.05 | ~0.1 | 0 | partly |

Three things fall out of it.

**1. The metric decides the answer.** On rank, minutes is joint-top. On points
per pick it is worth an eighth of attacking returns. A change judged on
`spearman_played` alone is being judged on the question where minutes dominates
— which is the question this repo happens to have been asking.

**2. `spearman_played` is structurally blind to availability.** Perfect
knowledge of who does not play scores exactly **+0.0000** on it, by
construction: it only changes predictions for players the metric excludes. It
is still worth +0.28 points per pick. Any test of an availability signal on
that metric alone cannot see its own channel.

**3a. A perfect RATE estimate is worth nothing — the ceiling really is luck.**
The oracle above substitutes outcomes, so it measures clairvoyance. Replacing
each rate with its leave-one-gameweek-out SEASON value instead — perfect
knowledge of the player's true rate, none of the match — moves nothing
(`spearman_played` −0.0017, top11 −0.02, rmse significantly *worse*), and a
perfect LOCAL rate over a ±4-gameweek window is significantly worse still
(−0.0050**, top11 −0.11, captain −0.81). So there is no estimator headroom in
the attacking channel and no exploitable non-stationarity: form at that
resolution is noise, and chasing it hurts even with hindsight. Team lambda
scores exactly +0.0000. DefCon is the one rate with any headroom, at +0.0014**.
This is why Understat's rates, the constant sweep, the set-piece decomposition
and adaptive shrinkage all moved nothing — they were improving an estimator
already at the variance-limited optimum. Use the SHRUNK oracle: the raw
leave-one-out rate is a noisier estimator that merely sees the future, and
scores −0.0084***.

**3. Most of the ceiling is luck, not modelling.** Attack, bonus and clean
sheets are ~5.6 of the 7.3 points-per-pick ceiling and none of them is knowable
before kickoff. The reachable components sum to under 2. **The free-data points
ceiling is close to exhausted**, and the one paid lever is a predicted-lineup
feed, already priced at ~+89 points a season (§16).

**The valuable part of minutes is the 60-minute class, not the exact figure.**
It carries 85% of the minutes rank gain and 77% of its points gain. Buy a feed
for "does he start and last an hour", not for "how many minutes".

**And that feed is now being collected.** `acquire/sources/predicted_lineups.py`
archives RotoWire's confirmed and predicted XIs on every scheduled run
(`data/collected/lineups/<season>.csv`, append-only, a line only when a
forecast changes). Each side carries its own status and the distinction is
load-bearing: a CONFIRMED XI is published about an hour before kickoff, i.e.
AFTER the deadline, so it is the ground truth to score predictions against and
never an input. No third party archives past predictions, so forward
collection is the only route — and per §16 it becomes priceable after roughly
five gameweeks rather than the season-plus a decision backtest would need.

### Two rejections re-tested on the metric that could see them, and both hold

The decomposition says the clean-sheet channel is worth +2.35 points per pick
and DefCon +0.70 — both previously judged on metrics that under-weight them.
Re-run on the full set, 74 paired gameweeks:

| arm | spearman_played | top11 | top30 | captain |
|---|---|---|---|---|
| `ODDS_WEIGHT` 0 (off) vs 0.85 | −0.0016 | −0.036 | −0.021 | −0.46 |
| `ODDS_WEIGHT` 1.0 vs 0.85 | −0.0002 | +0.030 | +0.005 | +0.05 |
| DefCon rate x1.13 (the measured shortfall) | +0.0001 | −0.015 | −0.018 | 0.00 |
| DefCon rate x1.26 | +0.0002 | +0.014 | −0.007 | 0.00 |

Nothing is significant. Odds lean the right way — switching them off costs
0.036 points per pick and half a captain point — but not resolvably at n=74.
Both standing conclusions survive their sharper test. *(The first attempt at
the odds arm returned exact zeros: this database had odds for the live season
only, so every weight was really zero. `ingest_football_data` over the backfill
seasons was the fix — a reminder that an A/B of a parameter with no data behind
it looks exactly like a parameter that does not matter.)*

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

* **Predicted-lineup and press-conference models.** No free feed here carries
  manager quotes, and FPL's `status`/`chance_next` is a *current* snapshot with
  no history — so the availability overlay cannot be backtested from FPL data.
  **Injury duration came off this list and has now been settled**:
  Transfermarkt dates every spell, which made the availability channel
  backtestable for the first time — and both gates closed negative. Its flag
  is a 32%-recall copy of FPL's `status`, and its history moves no decision
  metric. See the Transfermarkt section. What the exercise did buy is a price
  for availability itself: +0.134 top-30 pts/pick against a baseline denied it,
  which is a lower bound on what the live overlay already earns.
* **Manager identity and manager-specific rotation. Settled: reachable, and
  worth nothing.** Transfermarkt carries 927 dated managerial spells; added as
  a feature family they make the minutes model **worse** in both held-out
  seasons (+0.61% / +0.38% log-loss). Formation is likewise worse in both
  (+0.23% / +0.25%) and pressing style is zero. See the tactics section.
  Manager x player-role interactions cannot be tested properly here:
  **StatsBomb's open data holds only Premier League 2003/04 and 2015/16** —
  checked against `competitions.json`, not assumed — so no event-level source
  overlaps a replayed season.
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
| **Transfermarkt** | permissive `robots.txt`; its terms prohibit automated extraction, and robots.txt is not a licence. Originally not scraped for that reason. **Now scraped, at the owner's explicit instruction** for personal use — see the Transfermarkt section below. The `felipeall/transfermarkt-api` wrapper was evaluated first and rejected: unmaintained since April 2025, and its own issue #121 is "500 Error Status on all GET Endpoints" (confirmed — every endpoint 500s). |
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

### The repository is now the archive (scheduled collection)

The SQLite file is gitignored and rebuilt from free sources, so anything
collected only into it dies with the clone — which is how two seasons of
availability history were lost. `acquire/actions.py` +
`.github/workflows/collect.yml` fix the class of problem: a scheduled
GitHub Actions run (4x daily, stdlib-only, no secrets) fetches the official
bootstrap and commits to `data/collected/`:

* `availability.jsonl` — append-only team-news change log (status,
  chance_next, verbatim news, FPL's `news_added`), a line only when a
  player's state actually changed. This is the Phase-2 Priority-2 dataset
  accruing forward.
* `snapshots/<utc>.csv.gz` — one compact per-run market snapshot (price,
  ownership, transfers, ep_next, form) for the deadline-decay question
  ("how much is each hour of information worth?").
* `ownership/gw<n>.csv` — overwritten until that gameweek's deadline
  passes, then frozen: the last pre-deadline ownership, i.e. the lagged-EO
  input collected forward instead of reconstructed.
* `picks/<season>/gw<n>.csv` — the proven-manager panel's squads, fetched
  once per passed deadline (picks lock at the deadline, so post-deadline
  collection is point-in-time valid).
* `panel.json` — the fixed panel. Graded on PAST seasons only; the 2026-27
  panel was enumerated after GW2, so its GW1-2 picks are conditioned and
  excluded from any differential analysis (see RESEARCH_LOG).

`python -m acquire actions-import` replays the file archive into the
`acq_*` change-log tables so research keeps one query surface. Scheduled
workflows fire only on the default branch — the workflow must be merged to
`main` before the cron runs. Tests: `tests/test_acquire_actions.py`
(append-only idempotence, ownership freeze, importer idempotence, and the
no-model-import rule).

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

### Round 15: the optimiser itself, finally replayed

Every round so far measured the *projections*. The machinery that turns them
into decisions — the multi-period MILP, free-transfer accrual, the -4 hit
price, the horizon decay — had never been scored against what actually
happened. A model that ranks players well can still be given away by a
transfer policy that churns, or hoards, or plans into fixtures it cannot see.

The replay is point-in-time throughout: projections for gameweek *t* are built
with `as_of` = the first kickoff of the **decision** gameweek, so a plan made in
GW7 for GW11 sees only GW1-6 form. Prices move as they moved, selling prices
follow FPL's rule (purchase plus half the profit, rounded down), free transfers
accrue +1 and bank to 5, and points are realised points with auto-substitutions
and the vice-captain fallback applied. Chips are off, so what is measured is
transfers alone. **Every policy starts from the same opening squad**, so the
comparison is not decided by the luck of an opening fifteen.

| policy | pts/season | transfers | hits | end squad value |
|---|---|---|---|---|
| never transfer | 1815 | 0 | 0 | £99.2m |
| horizon 1 (greedy) | 2091 | 44.0 | 7.0 | £98.1m |
| horizon 3 | 2118 | 47.0 | 10.0 | £99.9m |
| horizon 5 (the default) | 2132 | 52.0 | 15.0 | £100.0m |
| horizon 8 | 2160 | 49.5 | 12.5 | £100.2m |

**Transferring earns its keep, decisively** (74 paired gameweeks):

| | pts/season | t | p | 95% CI |
|---|---|---|---|---|
| horizon 1 vs never | **+283** | 4.44 | <0.0001 | [+158, +409] |
| horizon 3 vs never | **+311** | 4.32 | <0.0001 | [+170, +452] |
| horizon 5 vs never | **+326** | 4.51 | <0.0001 | [+184, +468] |
| horizon 8 vs never | **+354** | 4.83 | <0.0001 | [+210, +497] |

Two different mechanisms, one per season, and both favour transferring. In
2024-25 a strong opening squad decayed (62.3 -> 48.0 pts/gw across the halves)
and transfers held it up (61.8 -> 56.8). In 2025-26 the opening squad was
simply weak and never decayed at all (42.5 -> 43.7); transfers rebuilt it to
~56. Squad rot and squad error are separate failure modes and the solver
repairs both.

**Whether the multi-period machinery beats a greedy one-week solve is not
resolvable here**, which is the deflating half:

| vs horizon 1 | pts/season | t | p | 95% CI |
|---|---|---|---|---|
| horizon 3 | +28 | 0.46 | 0.65 | [-91, +147] |
| horizon 5 | +43 | 0.73 | 0.47 | [-71, +157] |
| horizon 8 | +70 | 1.19 | 0.24 | [-46, +186] |

Read the CIs before concluding anything. This design cannot resolve less than
about **120 points a season**; separating +70 from zero at 80% power would need
roughly **11 seasons**. So this is emphatically *not* "the horizon does not
matter" — it is "two seasons cannot tell".

What the point estimates do is line up monotonically in the horizon, and the
*mechanism* is visible in the columns that are not the outcome: a longer
horizon finishes with a **more valuable squad** (£98.1m -> £100.2m) while
taking **no more hits** than horizon 5. That is exactly what planning ahead is
supposed to buy — seeing the transfer coming and rolling a free transfer to it
rather than paying four points for it. Direction, mechanism and structural
argument all agree; only the significance is missing, and it is missing for a
power reason rather than a null one. **The default horizon of 5 stays.**

**Why this test is so much weaker than the others in this file.** Once two
policies make different transfers they own different players, so their weekly
outcomes decorrelate and the paired design stops pairing. Every earlier
comparison here — rate estimators, chip reserves, bench weight — held the
squad nearly fixed and differenced a small perturbation. A policy comparison
cannot, which puts a hard floor under what any two-season replay of a
*decision rule* can show. Expect it again for anything that changes what the
squad owns.

### Auto-substitution: the bench is priced correctly

FPL replaces a starter who plays no minutes with the first eligible bench
player. Over 74 replayed gameweeks with a model-picked XI, 0.53 starters blank
a gameweek, 43% of gameweeks have at least one, and auto-subs recover **2.26
pts/gw (~86 a season)**. The optimiser assigns the bench almost nothing
(`bench_weight` 0.05-0.15), so it minimises bench spend by construction — which
looks like a large blind spot.

It is not, and comparing whole-squad points cannot show why: a ~0.5 point
effect under a 15-point standard deviation is invisible. Measuring the
**channel** instead — the auto-sub points the squad actually recovered, whose
paired sd is 3.5 rather than 15:

| bench valued | bench spend | auto-sub pts recovered |
|---|---|---|
| zero | £15.74m | 0.553/gw |
| half | £17.18m | 0.789/gw |

**+0.237 pts/gw** recovered, for **£1.43m** diverted from the XI, which costs
**0.234 pts/gw** at the budget exchange rate measured in Round 7. Net
**+0.003**. The two sides cancel because a £4.0m bench player scores about as
well as a £5.5m one *given* he is called upon, and the 86 pts/season accrues
either way. The benefit side is noisy (±0.56), so this is centred on zero
rather than proven zero — but the shipped defaults already sit in the flat
region, so nothing changes.

### DefCon: the convexity is real and the fix is worse

Crossing a threshold of 10-12 defensive actions is convex in minutes, and the
engine models it as a per-90 crossing rate times exposure, which is linear. The
first diagnostic looked damning — 134 predicted crossings against 1 realised
for 1-45 minute cameos — but it bucketed by **realised** minutes, i.e.
conditioned on the outcome. Re-bucketed on predicted P(60+), which is what the
model actually knew, against a Negative-Binomial posterior that carries the
uncertainty in the rate into the tail:

| P(60+) | realised | shipped | ratio | posterior | ratio |
|---|---|---|---|---|---|
| < 0.10 | 51 | 60.3 | 0.845 | 34.0 | 1.501 |
| 0.40-0.70 | 283 | 224.4 | 1.261 | 221.8 | 1.276 |
| 0.70-0.90 | 748 | 635.9 | 1.176 | 577.0 | 1.296 |
| **TOTAL** | **1418** | **1254.1** | **1.131** | **1122.4** | **1.263** |

The shipped form is better calibrated everywhere. The residual 13% under-
prediction is real but traces to an in-season upward trend in crossings
(0.111 -> 0.134 per appearance) that any trailing estimator lags, worth about
0.011 pts per player-gameweek. Not shipped.

### Round 16: pricing information — what a lineup feed is worth

Two questions: is the model at its ceiling because the *information* is
exhausted, and is maximising expected points the wrong objective? The first
turns out to be answerable in points; the second turns out to be answerable in
algebra.

**What a predicted-lineup feed is worth.** Round 8 located the minutes error in
the band where the model says P(start) is 0.30-0.70 — 11.5% of rows, where it
is *correctly calibrated* because the manager has not decided. Resolving only
that band, and nothing else, over 74 paired gameweeks:

| resolved | share of rows | spearman_played | top11 | top30 |
|---|---|---|---|---|
| band, 25% | 2.8% | +0.0229*** | +0.09 | +0.07* |
| band, 50% | 5.6% | +0.0445*** | +0.24** | +0.16*** |
| band, 75% | 8.4% | +0.0625*** | +0.24* | +0.21*** |
| **band, 100%** | **11.5%** | **+0.0814*** | **+0.36** | **+0.26*** |
| everything outside the band | 88.5% | +0.1308*** | +0.62*** | +0.56*** |
| full oracle | 100% | +0.2029*** | +0.73*** | +0.67*** |

The full oracle reproduces the known +0.21, so the harness is sound. The band
is **40% of the entire minutes ceiling from 11% of rows** — 3.6x the average
value density — and the value is **linear** in the fraction resolved, so a feed
that is half-useful is worth half. There is no threshold to clear.

Replayed through the Round 13 harness, with the feed modelled honestly (it
resolves the imminent gameweek's XI and says nothing about GW+3, because that
information does not exist):

| | baseline | with a perfect rotation feed | gain |
|---|---|---|---|
| 2024-25 | 2192 | 2298 | +106 |
| 2025-26 | 2073 | 2141 | +68 |
| pooled | | | **+2.35 pts/gw = +89/season** (p=0.055) |

Read the two layers separately. The *information* gain is proven — p<0.001 on
the rank metrics, replicated per season. The *points* conversion is the best
available estimate with a CI of [-1, +180], because of the Round 13 wall: once
the feed changes a transfer the arms own different players and the pairing
collapses. A metric over 600 players is well powered; the same effect seen
through 11 squad slots is not.

**Therefore: judge a feed on its band accuracy, not on a decision backtest.**
Because value is linear in resolution, a vendor is priced by its hit rate on
~90 ambiguous rows a gameweek — five gameweeks gives ~450 observations and a
tight estimate, against the season-plus a decision backtest would need. The bar
is the model's own ~55% in that band, and each point above it is worth roughly
`(accuracy - 0.55)/0.45 x 89` points a season.

**The crowd's transfer flow is not team news.** The obvious free substitute —
un-lagging the crowd features, since transfers into gameweek G close at G's
deadline — was checked before being built:

| | 2024-25 | 2025-26 |
|---|---|---|
| corr(net flow into GW, points in GW-1) | +0.367 | +0.390 |
| corr(net flow into GW, points in GW) | +0.146 | +0.175 |
| **among players who did NOT start GW-1: corr(flow, starts now)** | **-0.002** | **+0.010** |

The first two rows confirm the timing is legal (the flow is backward-looking,
so using it would not leak). The third says it is worthless for the only
population that needs help. Twelve million managers chase last week's hauls;
they do not relay team news. `shift(1)` stays.

### The same conclusion from the transfer layer

Rounds 13-14 close the decision layer at the **weekly** layer: the squad is
held to the lagged-ownership template and the arms differ only in
XI/captain/vice/bench order. That is where the EO-free theorem was derived
(`E[Delta]` does not depend on `EO`, so no realised-mean test can validate
`gamma != 0`) and where seventeen challengers were beaten by plain max-xP at
n=111. None of that is re-derived here.

What was not covered there is the layer above it: a rank tilt applied to the
**projections that drive transfers**, so the arms end up owning different
players. Two independent nonlinearities were swept through the Round 15 replay
harness.

**Club concentration.** Players from one club share a fixture, so their
returns are correlated. `concentration` prices that exactly (see
`optimise/chips.py`); pooled over 74 gameweeks:

| lambda | pts/season | margin sd | beat crowd | excess club stacking |
|---|---|---|---|---|
| -1.0 concentrate | 2108 | 13.36 | 61% | 5.95 |
| -0.5 | 2160 | 13.86 | 68% | 5.03 |
| **0.0 ships** | 2132 | 13.54 | 65% | 3.97 |
| +0.5 | 2147 | 13.04 | 70% | 2.61 |
| +1.0 spread out | 2084 | 14.86 | 61% | 1.14 |

Every arm is inside noise (p > 0.32) and — the actual finding — **margin sd is
flat while stacking varies five-fold**, in both seasons. The lever moves
composition and not risk, because FPL's rules already cap the achievable
concentration: at most 3 per club and 11 starters across 5+ clubs leaves too
little room for correlation to matter. The +1.0 arm even has the *highest*
variance, since forcing a spread pushes into worse and more volatile players.
The knob stays at 0.0 and is kept only because it is tested and free.

**Differential exposure.** The sharper version, because ownership does not drop
out of the *variance* even though it drops out of the mean:
`Var(Delta) ~ sum m_i * sigma_i^2 * (1 - 2 EO_i)`, which is linear in the
decision variables and so folds straight into the projections as
`ep' = ep + gamma * sigma^2 * (1 - 2 EO)`. Pooled over 74 gameweeks, ownership
lagged and sigma estimated point-in-time:

| gamma | pts/season | vs 0 | p | margin sd | beat crowd |
|---|---|---|---|---|---|
| -0.30 protect | 2056 | -121 | 0.045 | 14.04 | 55% |
| -0.10 | 2051 | -125 | 0.022 | 14.32 | 57% |
| **0.00 ships** | **2173** | — | — | 14.24 | **72%** |
| +0.10 chase | 1988 | -191 | 0.005 | 15.33 | 49% |
| +0.30 chase | 1678 | -509 | <0.0001 | 14.63 | 36% |

Unlike club concentration this lever *does* move margin sd (13.7 -> 15.3), so
the channel is real — but the exchange rate is ruinous: ~6% more spread costs
~180 points of mean. **Every deviation from the mean is significantly worse in
both directions**, which is a stronger statement than the weekly layer could
make, and it is stronger for a structural reason: at the weekly layer a tilt
only reshuffles eleven of a fixed fifteen, while here it changes which fifteen
you own. Round 14 bounds gamma's cost at the weekly layer; this shows it is
actively harmful once it drives squad selection. Both say `gamma = 0`.

**One implementation note, because the bug was silent.** A one-sided
`z >= n - 1` prices a penalty correctly but leaves the objective **unbounded**
as soon as the weight goes negative — CBC returns garbage, not an error, and
every solve produced an empty frame rather than raising. The excess is now
pinned exactly with two-sided indicators (`n >= k*y_k`, `n <= (k-1) + M*y_k`),
and `tests/test_concentration.py` asserts the negative arm stays bounded.

### The model against the field, with transfers

Round 14 measures the engine against the crowd with the **squad held fixed**
(+2.11 pts/gw for max-xP, p=0.027) — that isolates the weekly decision. This is
the same comparison one layer up, with transfers live, so it carries the
transfer value the fixed-squad design deliberately excludes.

The crowd's realised score is directly computable from `player_gw.selected` and
validates cleanly:
ownership sums to **15.00** players, the implied manager counts are 10.74m and
12.32m (the real FPL populations), and the crowd scores 52.2 / 51.5 a gameweek
against a published average of ~50-55. No synthetic field is needed.

Against that field, the engine's own squad under a horizon-5 policy:

| | margin | per gameweek | weeks beating the crowd |
|---|---|---|---|
| 2024-25 | **+263** | +7.11 | 65% |
| 2025-26 | **+171** | +4.61 | 65% |

About **+224 points a season over the average manager**, with an identical 65%
weekly hit rate in both seasons. It is conservative in one respect: this arm
plays **no chips**, while the crowd baseline includes theirs. Against Round
14's +2.11/gw at the fixed-squad layer, the gap between the two is roughly what
the transfer policy is worth on top of the weekly decision — consistent with
Round 15's finding that transferring at all is worth +283 to +354 a season.

### Polymarket: shown, not modelled

Free, keyless, `robots.txt` carries no disallow rules, and coverage is real —
**18 EPL fixtures** a gameweek with 1-cent spreads and seven-figure liquidity
(Palace-City: $595k traded, $2.76m in the book). Ingested by
`ingest/polymarket.py`, refreshed on every `pull`.

**Every EPL market it offers is team level**: match result, exact score,
halftime, second half, first *team* to score, corners. There is no
anytime-goalscorer or assist market — checked, not assumed. So nothing here
reaches a player except through the fixture attack scaler, and that channel is
already measured to be second order: `ODDS_WEIGHT` at 0 / 0.5 / 0.85 / 1.0 is
indistinguishable. We also already carry bookmaker prices for the same
fixtures, Pinnacle included. A second team-level source is therefore redundant
with something worth ~nothing at the player level, and it is **not wired into
the model**.

It earns its place on screen instead, the same way the price model does:
reported next to the recommendation, never inside the objective. A prediction
market disagreeing with a sportsbook is worth a human's attention on a
captaincy call even when it moves no model output, so the Fixtures grid marks
the fixtures where the two part company by 4+ points.

**Quotes live in `market_quote`, not `match_odds`, and that is load-bearing.**
`odds_model.fixture_odds_map` selects every row for a fixture and builds a dict
keyed by `fixture_id`, so the last row silently wins. Writing a second source
into that table would move lambda depending on row order — a model change with
no backtest behind it.

*One parsing trap, recorded because it produced plausible-looking wrong
numbers.* Polymarket splits a three-way match into separate Yes/No markets, and
the draw's label is `"Draw (Crystal Palace FC vs. Manchester City FC)"` — it
contains **both** club names. Matching a team name as a substring therefore also
matches the draw, and the away side inherits the draw's price. Every row still
summed to 1.0; the only symptom was `p_draw == p_away` on all 18 fixtures. Legs
are now classified on an exact cleaned label, with the draw tested first
(`tests/test_polymarket.py`).

**The one genuinely new thing it has is the Exact Score market.** The engine
currently inverts 1X2 plus over/under 2.5 into (lambda_home, lambda_away) — two
parameters fitted to four numbers. An exact-score market prices the full
scoreline distribution directly, which pins the joint structure rather than the
marginals. That is strictly more information than we use, and it would feed the
simulator's correlation rather than the ranking. Untested, and expected small
for the usual reason: it improves the fixture channel, which is second order.

### Transfermarkt: rumours and injury history

Scraped directly, at the owner's explicit instruction, after the third-party
wrapper (`felipeall/transfermarkt-api`) was found unmaintained since April 2025
with every endpoint returning 500 — its own open issue #121. Two datasets, and
they are worth very different amounts.

**Transfer rumours are a correctness fix, not an edge.** FPL reclassifies a
player only once a move COMPLETES, so between the deal being agreed and that
update the engine projects him onto a club's fixtures he will never play. That
is not a mis-rating, it is the wrong club. Nothing else reaches this: FPL tells
you afterwards (`status='u'`, "Has joined X permanently"), and Polymarket
prices only the superstar tier (13 open markets, all Alvarez/Rashford-sized).

The board gives player, current club, interested club **and that club's
league**, a source date and Transfermarkt's own assessment. The league is what
decides the consequence: a destination in `GB1` means a different Premier
League run (reproject); anything else means he leaves the game and is worth
zero. Of 25 rumours, 15 resolve to FPL players and 8 of 9 strong ones are
exits, so the dominant effect is players who should stop being recommended.

Rumours at or above 50% are priced in automatically, **weighted rather than
switched**:

    ep' = p * ep_at_destination + (1 - p) * ep_as_things_stand

A hard threshold would treat a 51% rumour and a 95% one identically. Gabriel
Jesus at 83% to Barcelona keeps 17% of his projection; Gakpo at 69% to Man City
is repriced on City's fixtures with his own rates. `engine.xpts_predict_gw`
takes a `team_override`, which is all the reprojection needs — everything
downstream keys off `team_id`, so fixtures, opponent strength and the
clean-sheet lambda follow. A manual watch remains for what the board misses.

*Two parsing traps, both of which produced plausible wrong output.* The club
regex `verein/(\d+)"[^>]*title=` cannot reach the title, because the markup is
`verein/631"><img src="…" title="Chelsea FC"` and `[^>]*` will not cross `>` —
it parsed zero rows. And substring club matching silently misclassified PL
destinations: "Manchester City" does not contain "Man City", so Gakpo's move
was filed as *leaving the league*, which is the one distinction the feature
exists to make.

**Injury history: measured, gated, and both gates closed negative.** 8,760
dated spells over 1,736 crawled players back to 2012, typed and with duration.
Round one reported −2.08% log-loss for history *on top of already knowing he is
out*, and refused to ship it behind two gates. Both have now been run.

*The identity join was wrong when that number was produced.* Spells were
attached through `tm_player.player_id` joined to a different season's `player`
table, and FPL reassigns element ids every summer — **99.7% of ids point to a
different footballer one season later**. Identity now travels on `player.code`.
Redone correctly the figure is **−1.5%**, replicated in both seasons: the shape
survives, the size shrinks.

*Gate (a) — decision metrics. Fails.* Injury history on top of the availability
flag, 74 paired gameweeks:

| | spearman | spearman_played | prec@20 | top30 | captain | rmse |
|---|---|---|---|---|---|---|
| history − flag | +0.0026*** | −0.0013 | −0.003 | −0.00 | +0.11 | −0.0022*** |
| everything − flag | +0.0045*** | −0.0022 | −0.001 | +0.00 | +0.11 | −0.0039*** |

`spearman` and `rmse` improve because the model got better at ranking **who
plays at all**, which is what injury history informs. Every metric that decides
a squad sits still, and in the opening gameweeks `spearman_played` is
significantly *worse* (−0.0042, p=0.004). This is the Understat result again:
a better estimator that changes no decision.

*Gate (b) — is it a noisier copy of FPL `status`? It is a weaker one.* On the
live season, 596 of 626 players mapped:

| | FPL says out | FPL says available |
|---|---|---|
| **TM says out** | 38 | 3 |
| **TM says available** | **81** | 474 |

**31.9% recall, 92.7% precision.** The misses are structural: FPL's `status`
also carries suspensions, players who have left or are unregistered, and
doubts, none of which an injury table can see. So it adds nothing live — and
since that flag was the *control* in the −2.08%/−1.5% measurement, a 32%-recall
control makes even that an upper bound rather than an estimate.

**Not shipped, and the question is closed.** The leakage boundary in
`xpts/injury_features.py` still holds (an ENDED spell is fully usable, an
ongoing one contributes only the fact of absence; audited for the
sustained-during-the-match failure and cleared at 133 spells on fixture days,
85% of whom played), and the builder is now vectorised — one sortable
(player, day) key turns a 68-second per-row scan into 0.25s, which is what
makes it affordable inside a backtest at all.

**The second prize is real and is the round's one keeper.** A dated
availability record makes the availability→minutes channel backtestable for the
first time — "no free feed here carries it" was a standing wall. Against a
baseline denied availability entirely, the flag alone is worth **+0.0085
`spearman_played`, +0.011 prec@20 and +0.134 top-30 pts/pick**, all p < 0.003.
That is the first decision-metric price this repo has put on availability, and
because FPL's live `status` is strictly stronger it is a *lower bound* on what
the availability overlay is already earning. Do not read it as a gain
available from Transfermarkt — it is a valuation of something the live model
already does.

### The rest of Transfermarkt: three dated datasets, no decision moved

`/kader/.../plus/1` (one request per club-season: date of birth, height, foot,
detailed position, joined date, signed-from club and fee, contract expiry,
market value), `/ceapi/transferHistory/list/{id}` and
`/ceapi/marketValueDevelopment/graph/{id}` — 3,691 squad rows, 15,118 dated
transfers, 30,215 dated valuations. `xpts/tm_features.py` builds four families
point-in-time; `$FPL_MINUTES_EXTRA` switches them on for one process, so the
shipped model stays bit-identical to the one every number above was measured on.

**Deliberately NOT ingested: Transfermarkt's appearance and minutes tables.**
They are a second copy of `player_gw`, and the standing rule from four
independent confirmations is that an effect measured in realised outcomes is
already absorbed by estimates fitted to realised outcomes.

**What is and is not point-in-time.** The two ceapi feeds date every row and are
filtered strictly before kickoff. The squad page cannot be: it serves TODAY's
contract expiry and market value even when asked for `saison_id` 2023, and
backdates a player's current joined-date onto the old squad (Raya reads
"04/07/2024" on Arsenal's 2023-24 page, when he was there on loan). Only date
of birth, height, foot and the name→id mapping are safe historically. Contract
expiry is live-only, on the same footing as FPL's `status`.

Log-loss vs baseline, three seeds, two independently held-out seasons:

| arm | all 24-25 | all 25-26 | cold start 24-25 | cold start 25-26 |
|---|---|---|---|---|
| age + TM transfers | −0.95% | −1.16% | −6.20% | −7.33% |
| age + TM market value | −0.42% | −0.51% | −5.18% | −5.81% |
| age + TM squad depth | −0.51% | −0.64% | −4.51% | −4.89% |
| **age + every TM family** | **−1.16%** | **−1.68%** | −6.57% | −8.62% |

Real, replicated, and worth **nothing in decisions**: against the baseline,
age + every TM family moves `spearman` +0.0030*** and `rmse` −0.0025*** while
`spearman_played` is −0.0007, prec@20 +0.001, top30 −0.01 and captain −0.49
(p=0.083, i.e. drifting the wrong way). The 4%-better-rate threshold from the
Understat round has its counterpart here: **a 1-2% better minutes log-loss is
also below the noise floor.**

### Age is the cold-start feature, and FPL was publishing it all along

`birth_date` is in the bootstrap and in vaastav's `players_raw.csv` from
2024-25; the pipeline discarded it. It separates the two kinds of player the
trailing features cannot tell apart — among men with under five career Premier
League appearances every history feature is identical (no minutes, no starts,
no appearances) and P(60+) still runs **0.000 at 15-18 to 0.401 at 24-27**,
falling again after 30.

**FPL's own column does not replicate yet, and the reason is coverage.** It
began in 2024-25, so coverage runs 56% / 60% / 88% / 99% across seasons: a tree
left to learn the missing branch learns it on a training population that has
all but vanished by serve time. Cold-start log-loss vs baseline:

| arm | 2024-25 | 2025-26 |
|---|---|---|
| coverage indicator alone (falsification) | −0.74% | −1.75% |
| FPL age, raw | −2.22% | **+0.13%** |
| FPL age, position-median imputed | −1.36% | −0.50% |
| Transfermarkt age | **−3.73%** | **−3.98%** |

Only the Transfermarkt-covered variants replicate — and the two sources carry
**the same dates** (median disagreement 0.0000 years, 0.12% differ by over 30
days). Transfermarkt is not a better signal, it is coverage for seasons FPL had
not started publishing, and FPL's column becomes sufficient on its own once
2024-25 and later are the training seasons. The ingest ships because it is free
and repairs a real omission; the *feature* does not, because it moves no
decision either (`spearman_played` −0.0011, p=0.09).

### A unit bug that made the congestion feature a gameweek counter

pandas keeps whatever resolution a timestamp was parsed at, and since 2.0 an
ISO8601 string parses to **microseconds** — so `astype("int64") / 86_400e9`
silently returns days/1000. `days_rest` survived it (a tree reads only the
order, and nothing then reached the 30-day clip); `team_matches_14d` did not,
because a 14 meaning 14,000 days counts every previous match of the season.
Median 20 and max 37 against a true median of 1 and max of 4 — the model had
been documenting a fixture-congestion signal it did not have.

**Repairing it changes no decision** (74 paired gameweeks): spearman −0.0002,
`spearman_played` +0.0005 (p=0.21), top30 −0.019 (p=0.40), captain +0.50
(p=0.069). A correctness fix, not an improvement, and consistent with Round 8.
Divide by a `Timedelta`, never by a magic constant against an integer view;
`tests/test_minutes_model.py` pins it.

## The tactics expert: six families, one survivor, and it is not tactics

Asked as a pre-registered question — *does manager/tactical context carry
information the model is missing?* — with six separable families, tested
incrementally on genuinely held-out seasons. `xpts/tactics_features.py`.

**Sources.** Transfermarkt's staff history gives 927 DATED managerial spells
across the 27 clubs in this database; the dates are the point, since a name is
a categorical with no history but an appointment date makes "who was in charge
that day, and for how long" point-in-time. Understat gives PPDA and deep
completions per team-match, and the per-match ROLE a player occupied (AMR, DMC,
FWL) — the only free per-match role feed there is, now pulled for every
backfilled season rather than the current one and one before.

**Not reachable, checked rather than approximated:** possession, field tilt,
crossing frequency, attacking width and build-up style have no free per-match
history here, and no event-level source overlaps a replayed season.

| family | 2024-25 | 2025-26 | verdict |
|---|---|---|---|
| manager identity, tenure, continuity | **+0.61%** | **+0.38%** | worse in both |
| formation (shape, stability, changes) | **+0.23%** | **+0.25%** | worse in both |
| playing style (PPDA, deep, opponent's) | −0.07% | −0.13% | ~zero |
| manager x opponent | −0.09% | +0.03% | ~zero |
| manager x player role | −0.85% | −1.21% | replicates |
| **the player's own line** | **−1.31%** | **−1.42%** | replicates |

**The falsification says it is not managerial.** The coverage indicator alone
is worth −0.38%/−0.13% and squad competition −0.31%/−0.19%, while *which line
he plays* carries −1.26%/−1.25% — the whole family. `manager x role` only
worked because it partly re-encodes the same fact. The finding is that **FPL's
four-way label is too coarse**: it calls a DMC and an AMC both "MID".

**A missing-value defect that cost real points.** Coding an unresolved role `0`
asserts "he is neither an attacking nor a defensive midfielder" rather than "we
have not seen his line", and a third of rows carried that false denial. It cost
**top11 −0.141 (p=0.008)** — significant harm to the starting XI in both
seasons — and vanished (−0.050, p=0.39) once unknown became NaN. *An absent
observation is not a negative one.*

**What ships:** three features (`role_is_am`, `role_is_dm`, `role_vs_fpl_line`)
that reproduce the whole family, now in `minutes_model.FEATURES`. Over 74
paired gameweeks: `spearman_played` **+0.0047 (p=0.0007)**, significant in each
season separately (+0.0039, +0.0055), `rmse` −0.0049***, and top11/top30/
prec@20 all null. That is 4.6x the `spearman_played` gain the hybrid E[minutes]
estimator shipped on — but it is a rank-quality gain, **not** a proven points
gain, and it is stated that way. NaN without `pull --understat`, which the
classifier tolerates by behaving as it did before.

*The captain column in that comparison is three observations.* The pick changes
in 3 of 74 gameweeks; the identical −0.35 in both seasons is arithmetic
coincidence and carries no information either way.

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
* **One player, one Understat id.** The cross-season fill only touches NULLs,
  so two seasons resolving the same man to different ids went unreconciled:
  FPL's Amad Diallo was Understat 8127 ("Amad Diallo Traore", Man Utd) in two
  seasons and 12200 ("Amadou Diallo", Newcastle) in three — a different player.
  Neither the season count (3 vs 2) nor the stored match volume (1 vs 0) picks
  the right one, so `pipeline.reconcile_understat_ids` unsets every season for
  a code claiming more than one id and reports it. `entity_override` pins a
  case a human has checked.
* **Understat resolution is 83% per season**, and the unresolved 17% keep
  `understat_id` NULL so their Understat features stay NaN (the FPL xG
  stand-ins apply). Resolution now runs for every backfilled season and
  fills across seasons on the stable `player.code`.
* **Understat** is ingested from its JSON endpoints (`getLeagueData/{league}/{year}`
  for every club's per-match xG/xGA/deep/PPDA in one call, `getPlayerData/{id}`
  for a player's per-match log across all seasons, `main/getPlayersStats/` for
  ids/names used in resolution; all need `X-Requested-With: XMLHttpRequest`).
  `pull --understat` (the web Data button has it on) covers the current and
  previous season by default; `_pull_understat(..., history_seasons=4)` covers
  every backfilled one, which is what the per-match ROLE features need.
  Understat features are NaN when it is unreachable; the models tolerate this
  via `np.nan_to_num`, and the three shipped role features degrade to NaN so
  the minutes classifier behaves as it did before they existed. To limit the damage, FPL's own (Opta) expected stats —
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
python -m fpl_engine transfermarkt   # squads, transfers, values, injuries, managers (rate limited)
python -m fpl_engine pull --understat      # needed for the shipped role features
python -m fpl_engine predict --gw 1        # end-to-end predictions
python -m fpl_engine run --gw 1            # pull + build + predict
python -m fpl_engine optimise --entry 883566 --horizon 5   # transfers / squad
python -m fpl_engine prices                # who is about to rise / fall in price
python -m fpl_engine simulate --gw 2       # floors / ceilings / P(haul) + joint risk
python -m fpl_engine train                 # optional: retrain models (GPU-aware)
python -m fpl_engine predict --gw 1 --blend auto   # blend retrained + OpenFPL
python -m pytest tests/ -q
```

### Round 17: minutes uncertainty, new information only — screen first, always

Seven pre-registered hypotheses on the P(start) middle band, every observation
strictly as-of the deadline, nothing judged on log-loss alone. Full record in
`RESEARCH_LOG.md` E15; what a future round needs to know:

* **Return-from-injury dynamics is the first genuinely miscalibrated pocket
  found in nine attempts at this model.** An established starter's first match
  back after missing 2+ club matches: the model said P(start) 0.15, reality is
  0.41, gap +0.262/+0.264 in the two seasons — because every trailing feature
  describes the absence, not the player. `xpts/return_features.py`
  (`FPL_MINUTES_EXTRA=ret`, 10 features off ended `tm_injury` spells with a
  24h pre-deadline guard on `until_date`) closes it: overall minutes log-loss
  **−1.66%/−1.71%** (prior best of eight sharpening attempts: ≤0.25%), segment
  −9..−13%, E[min] MAE −1.0/−1.2%. Decision level, 74 paired gameweeks:
  `spearman` +0.0031***, every squad metric null — the E11 signature, 1–2
  returners a gameweek cannot move eleven slots. **Not shipped on**; the live
  availability-overlay path is where it would plausibly pay, and that is the
  one channel a replay cannot price.
* **The congestion features count PL matches only, and that understates 13% of
  rows by 2+ matches.** `tm_club_match` (all-competition club calendars, one
  request per club-season) fixes the observation layer. The one miscalibrated
  pocket it exposes: likely starters with cup fixtures on BOTH sides of the PL
  match are overpredicted by −0.05/−0.06 — double-cup rotation.
  `FPL_MINUTES_EXTRA=cong` repairs half of it, −0.3% overall log-loss,
  bounded at zero decision value by the ret result above. Extras only.
* **No historical predicted-lineup source passes the as-of test.** Wayback
  coverage of RotoWire / FFScout / SportsGambler at the 76 deadlines of the
  two backtest seasons: at best 5/38 within 24h, mostly 0. The forward
  collector remains the only honest route to the one lever that matters.
* **Manager-change selection reset: nothing to fix** — the shipped model is
  calibrated within +0.01 across the first eight fixtures under a new manager
  (squad-share and consecutive-start features adapt fast enough). Screened,
  no feature built. New-signing prior-club minutes are unreachable
  (Transfermarkt's performance pages are client-rendered; Understat covers
  22/350 from-abroad debutants before arrival).
* Two lessons re-earned: the **datetime64 unit trap** (a `datetime64[D]`
  array stored into a DataFrame column silently reverts to `[ns]`; divide by
  a `Timedelta`, never `astype(int)` — this bit the first congestion screen),
  and **screen calibration before building features** — both real pockets
  this round were found by the screen, and both rejections cost nothing.

### Round 18: the actionable minutes-error atlas — the minutes programme is closed

E16 in `RESEARCH_LOG.md`: masked minutes oracles (realised minutes substituted
for ONE pocket of rows at a time) over 74 paired gameweeks, plus per-row error
classification and calibration screens. What survives for future rounds:

* **The error is P(start), not minutes|start.** ~30 false starters + ~29
  missed starters a gameweek carry 55% of start log-loss; duration errors
  (start right, 60' class wrong) carry 3% and +0.005 top11. Half of false
  starters still come off the bench — "benched", not "injured".
* **False starters are the single most valuable error class**: resolving them
  is +0.40 top11 points per pick, twice any other class — a phantom starter
  with high xP sits in the XI and returns 0-2, while a surprise starter
  rarely projects high enough to be picked (missed starters: +0.03 top11).
* **Value density peaks at baseline xP ranks 5-15**: resolving ELEVEN
  players' minutes is worth +0.51 top11 — more than the whole 95-row
  ambiguous band. Ranks 5-30 (~26 players/gw) carry most of the pick value;
  rank 61+ carries nearly all the rank value and none of the pick value.
  Perfect minutes changes 3.6-4.1 metric-XI slots EVERY gameweek (+7/gw at
  the unconstrained rank layer); the captain channel is +0.1/gw — nothing.
* **Judge a lineup feed on its false-starter hit rate among players ranked
  5-30** (~26 rows/gw, each worth ~10× the average row), sharpening E14's
  band-accuracy rule.
* **Every deadline-visible candidate context screens calibrated** (hooked
  early last match, bench-promotion trajectory, sub last match,
  international-break adjacency) — the only two miscalibrated contexts are
  the two E15 already built (`ret`, `cong`). The rest of the A/B error is
  the manager's unannounced decision. The minutes model is FROZEN pending a
  genuinely new source; downstream needs no round (E13/E14: no estimator
  headroom beyond exposure — the ceiling is match luck).
* Verification pass fixed a real defect: the TM squad crawl silently dropped
  City/Utd 2024-25 (`except: continue`) and the per-season club mapping
  blanked their calendars; the mapping now travels on stable `team.code`
  and all 100 PL club-seasons verify complete. Twice this round an EMPTY
  source produced a plausible zero (the `fixture` table is empty for
  historical seasons — use `team_match`); check the n before believing a
  null arm.
