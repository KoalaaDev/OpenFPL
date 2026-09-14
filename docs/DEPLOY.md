# Deploying FPLabs on the public internet

`python -m app` serves the API and the built site on one port. Put a
TLS-terminating reverse proxy in front of it; do not expose uvicorn's port
directly.

## 1. Build and run

```
cd web && npm install && npm run build      # -> app/static
cd .. && python -m app                       # 0.0.0.0:9999 by default
```

Environment (all optional except the Google trio if you want sign-in):

| variable | default | meaning |
|---|---|---|
| `FPLABS_HOST` / `FPLABS_PORT` | `0.0.0.0` / `9999` | bind address |
| `FPLABS_BASE_URL` | — | public origin, e.g. `https://fplabs.example.com`. Enables `Secure` cookies + HSTS, and is the OAuth redirect base |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | — | from Google Cloud Console (below). Sign-in is hidden until both are set |
| `FPLABS_ADMIN_EMAILS` | — | comma-separated Google emails allowed to press Refresh / trigger a pull by hand |
| `FPLABS_SECRET_KEY` | generated into `data/web_cache/secret.key` | cookie-signing secret; set it explicitly so a redeploy does not sign everyone out |
| `FPLABS_TRUSTED_PROXY` | off | `1` when behind your own proxy: rate limits then key on `X-Forwarded-For` |
| `FPLABS_AUTO_REFRESH` | on | `0` disables the scheduler |
| `FPLABS_REFRESH_UTC` | `04:30` | daily pull + reprojection time (UTC) |
| `FPLABS_PRE_DEADLINE_HOURS` | `2` | an extra refresh this long before each deadline |
| `FPLABS_HORIZON` | `6` | gameweeks projected on each refresh (max 8) |
| `FPLABS_UNDERSTAT` | on | `0` skips Understat on scheduled pulls |
| `FPLABS_MAX_SOLVES` | `2` | concurrent solver jobs across all visitors |
| `FPLABS_ALLOW_COOKIE_IMPORT` | off | `1` re-enables the FPL-cookie squad import (needs TLS; the bookmarklet is the safe path) |
| `FPLABS_ENFORCE_PLANS` | off | `1` turns on the Free/Pro gating (docs/MONETISATION.md) |
| `FPLABS_DEBUG` | off | `1` exposes `/api/docs` |
| `FPLABS_CONTACT_EMAIL` | `hello@koalaa.dev` | contact address on `/privacy` and `/terms` |
| `FPLABS_JURISDICTION` | England and Wales | governing law named in `/terms` |
| `FPLABS_APP_DB` | `data/app.sqlite` | accounts + per-user documents. **Back this file up** — the pipeline DB is disposable, this one is not |

The simplest way to set them is a `.env` file in the repository root
(`python -m app` reads it at startup; it is gitignored, and anything already
in the environment wins over the file). Copy `.env.example`, fill in the
Google client id and secret, your admin email and `FPLABS_BASE_URL`, then
restart the app. Never commit `.env`.

A systemd unit or a Windows service that runs `python -m app` with those
variables is all the process management needed; the scheduler is a thread
inside the process.

## 2. Reverse proxy (Caddy — automatic HTTPS)

```
fplabs.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:9999
}
```

Set `FPLABS_HOST=127.0.0.1` so only Caddy can reach uvicorn, and
`FPLABS_TRUSTED_PROXY=1` so the rate limiter sees real client addresses.
nginx works the same way (`proxy_set_header X-Forwarded-For $remote_addr`).

## 3. Google sign-in

Values for **fpl.koalaa.dev** (paste these into the Google Cloud Console):

| field | value |
|---|---|
| App name | FPLabs by KoalaaDev |
| Application home page | `https://fpl.koalaa.dev/` |
| Application privacy policy link | `https://fpl.koalaa.dev/privacy` |
| Application terms of service link | `https://fpl.koalaa.dev/terms` |
| Authorised domain | `koalaa.dev` |
| Authorised JavaScript origin | `https://fpl.koalaa.dev` |
| Authorised redirect URI | `https://fpl.koalaa.dev/api/auth/google/callback` |
| Scopes | `openid`, `.../auth/userinfo.email`, `.../auth/userinfo.profile` (non-sensitive, no verification review needed) |

and in the server environment: `FPLABS_BASE_URL=https://fpl.koalaa.dev`,
`FPLABS_CONTACT_EMAIL=<your contact address>` (printed on both legal
pages; default `hello@koalaa.dev`), optionally `FPLABS_JURISDICTION`
(default "England and Wales") and `FPLABS_LEGAL_DATE`. The pages are served
by `app/legal.py` from the templates in `app/legal/`; edit the HTML there if
the wording needs to change. Signed-in users can delete their account (and
every document under it) from the account menu — `DELETE /api/auth/me` —
which is what the privacy policy promises.


1. Google Cloud Console → APIs & Services → Credentials → *Create
   credentials* → *OAuth client ID* → Web application.
2. Authorised redirect URI: `https://fplabs.example.com/api/auth/google/callback`
   (exactly `FPLABS_BASE_URL` + `/api/auth/google/callback`).
3. OAuth consent screen: external, scopes `openid email profile`. While the
   app is in "Testing" only listed test users can sign in; publish it to
   let anyone.
4. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `FPLABS_BASE_URL`, restart.

The flow is server-side authorization-code; the `id_token` is verified through
Google's `tokeninfo` endpoint and then checked for audience, nonce and a
verified email. Nothing but the opaque user id lives in the cookie.

Signing in as an admin for the first time adopts the old single-user
`data/web_cache/{drafts,my_team,transfer_watch}.json` files into that
account and renames them `*.adopted`.

## 4. What the hardening covers

Found during the review of the app as it stood (everything below is now in
place, and `tests/test_app_security.py` pins each one):

| issue | fix |
|---|---|
| `--reload` + `0.0.0.0` in `__main__` | reload removed; host/port from env; no `Server` header |
| One shared `my_team.json` / `drafts.json` / `transfer_watch.json` for every visitor — anyone could read or overwrite anyone's squad | per-session documents in `app.sqlite`, keyed by a signed HttpOnly cookie; sign-in migrates them to the account |
| `/api/pull` and `/api/projections/build` (minutes of CPU, outbound crawl) open to the world | admin-only; a scheduler does the work instead |
| `/api/solve` accepted unbounded `time_limit`, `horizon`, `keep_per_position` | clamped (`services.SOLVE_BOUNDS`), per-visitor one solve at a time, global cap, token-bucket rate limit |
| Job results (containing squads) readable by anyone who had the id | jobs carry an owner; 404 for anyone else |
| Tracebacks returned to the client on job failure | message only; traceback to the server log |
| `/api/img/{kind}/{code}` fetched and stored any integer `code` — disk fill + open proxy to the PL CDN | only codes present in the live bootstrap |
| No request body cap | 1 MB (`FPLABS_MAX_BODY`) |
| No CSRF protection on cookie-authenticated mutations | `SameSite=Lax` cookie + mutations must be `application/json` with `X-Requested-With: fetch` |
| No security headers | CSP, `X-Frame-Options: DENY`, nosniff, referrer policy, permissions policy, HSTS when HTTPS, `no-store` on `/api` |
| OpenAPI/Swagger exposed | off unless `FPLABS_DEBUG=1` |
| Visitors' FPL session cookies transiting the server | route disabled by default |
| Squad / drafts / prefs bodies written to disk untyped | validated and bounded |
| Entry and league ids unbounded (FPL API amplification) | range-checked; league fan-out rate limited |

Still your responsibility: TLS at the proxy, backups of `data/app.sqlite`,
and keeping `pip` dependencies current (`pip list --outdated`).

## 5. Operations

* `python -m fpl_engine verify` after any manual data work — it exits
  non-zero on invariant breaks.
* `python -m fpl_engine transfermarkt` is **not** part of the scheduled
  refresh (it is a rate-limited crawl); run it by hand every few weeks or the
  Transfermarkt panels stay empty after a database rebuild.
* `python -m fpl_engine postmortem --gw N` and `lineup-feed --gw N` after
  each gameweek; the JSONs under `data/` are committed research state.
* `python -m acquire pull --source bbc` archives BBC lineups and live text for
  the last eight days, and `--source bbc_pressers` the Friday press-conference
  pages; it is not yet on the scheduler (add it once the first
  hypothesis on that data is worth keeping current).
* The scheduler writes its state to `data/web_cache/scheduler.json` and it is
  visible in `/api/status` under `auto_refresh`.
