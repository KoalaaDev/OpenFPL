# FPLabs — free and paid tiers

What is already built, what a paid tier would gate, and how to switch it on
without a refactor. Nothing here is live: `FPLABS_ENFORCE_PLANS` is off and
every visitor currently gets the Pro entitlements.

## What exists today (the seam)

* Every Google account has a `plan` column (`free` / `pro`) in `data/app.sqlite`.
  Anonymous visitors are always `free`.
* `app/plans.py` is the single table of entitlements per plan and the one
  function (`check`) that clamps a solve request to a plan. It reports which
  knobs it clamped, so the UI can say "horizon capped at 3 on Free" rather
  than silently returning a smaller plan.
* `/api/auth/me` returns `plan` + `entitlements`; the account menu shows a
  PRO / FREE badge and an "Upgrade" button once enforcement is on.
* Turning it on: `FPLABS_ENFORCE_PLANS=1`. Flipping a user:
  `userdata.set_plan(uid, "pro")` — which is exactly what a billing webhook
  calls.

## Suggested split

Keep the *reading* surface free: it is the funnel, it costs almost nothing per
visitor (projections are built once a day for everyone), and it is what gets
shared on Reddit/Twitter. Charge for the *decision* surface, which is the part
that costs CPU per user and is genuinely differentiated by the research in
CLAUDE.md.

| feature | Free | Pro | why |
|---|---|---|---|
| Projections table (next 6 GWs), fixtures, price movers | yes | yes | the funnel |
| Planner drafts | 2 | 50 | storage is per user |
| Solver horizon | 3 GWs | 8 GWs | a longer horizon is the measured mechanism of the multi-period solver (Round 15) |
| Playstyles per solve | 1 | 3 | 3× the CPU |
| Chip planning (WC/FH/BB/TC option value) | no | yes | the chip-reserve curve is the most original piece of the research |
| Mini-league analysis | no | yes | fans out 20-50 FPL API calls per view |
| Projection history (model drift per player) | no | yes | cheap to serve, feels premium |
| Simulator: floors/ceilings/P(haul) | no | yes | not in the web UI yet — a natural Pro-only addition |
| Solve time limit | 30 s | 120 s | CPU |

Everything in the Pro column is already implemented and gated by a single
entitlement key; the Free column is the clamp.

## Pricing shape

* **Monthly with a season pass.** FPL is seasonal: churn in June is total,
  so a "season pass" (Aug–May) at ~8× the monthly price converts the people
  who would otherwise cancel in the international breaks.
* Anchor low. Comparable tools (FPL Review, Fantasy Football Fix, Fantasy
  Football Hub) sit at roughly £2–£5 a month; the edge measured in this
  repo is ~80–90 points a season over a competent manual manager, which is
  real but is not a promise anyone should pay £20 a month for.
* Free trial by gameweek, not by days: "your first 3 solves are Pro" is a
  cleaner hook than a 7-day clock during an international break.

## Billing options

| | fit | notes |
|---|---|---|
| **Stripe Checkout + Customer Portal** | best control, lowest fees | you are the merchant of record: VAT/MOSS for UK+EU buyers is on you (Stripe Tax handles the calculation, not the filing) |
| **Lemon Squeezy / Paddle** | easiest | merchant of record — they handle VAT and invoicing; ~5% + fees; one webhook (`subscription_created/updated/cancelled`) → `set_plan` |
| Ko-fi / BuyMeACoffee memberships | simplest, weakest | no API worth relying on for entitlements; fine for a "support the lab" button |

Recommendation: **Lemon Squeezy first** (MoR removes the tax question for a
one-person shop), Stripe when volume justifies the ~2% saving.

Integration is ~150 lines: a `POST /api/billing/webhook` route that verifies
the provider's signature, maps the customer email to `user.email`, calls
`userdata.set_plan`, and a `GET /api/billing/portal` redirect. Keep the
webhook idempotent (store the event id).

## Things to settle before charging

1. **Terms with the data sources.** The FPL API has no published terms for
   third-party commercial use; every commercial FPL tool uses it anyway.
   Understat's `robots.txt` disallows everything (already flagged in
   CLAUDE.md) and Transfermarkt's terms prohibit automated extraction — a
   paid product built on them is a materially different position from a
   personal tool. RotoWire lineups likewise. Decide which feeds a paid tier
   may lean on; the model degrades gracefully without Understat
   (`np.nan_to_num`) and Transfermarkt is display-only.
2. **Never ship cookie import on a paid tier.** It moves a visitor's FPL
   session cookie through your server; the bookmarklet path never does.
   `FPLABS_ALLOW_COOKIE_IMPORT` stays off.
3. **Privacy page + data deletion.** Google sign-in stores email, name and
   avatar URL. A "delete my account" button (`DELETE /api/auth/me` →
   remove `user` + `user_doc` rows) is a day's work and a legal must.
4. **Capacity.** One process, one CBC solver at a time per visitor, two
   globally (`FPLABS_MAX_SOLVES`). A paid tier implies a queue and a bigger
   box before it implies a price.
5. **Fairness of the numbers.** The projections are one shared cache — every
   tier sees the same model. That is the right design (a "better model for
   Pro" is a trap: it splits your evaluation data) and worth saying on the
   pricing page.
