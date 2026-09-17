"""FastAPI app: JSON API for the FPLabs planner + static frontend serving.

Run with:  python -m app   (see app/__main__.py for host/port environment)

Everything a visitor saves is scoped to their session (see app/userdata.py);
signing in with Google (app/auth.py) makes it follow them between devices.
Expensive operations are rate limited (app/security.py), data refreshes run
on a schedule (app/scheduler.py) and are otherwise admin-only.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import warnings

warnings.filterwarnings("ignore")  # sklearn pickle version chatter

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from fpl_engine import config, db

from . import (auth, deadline, images, jobs, legal, live, modelhistory, plans,
               scheduler, security, services, userdata)

log = logging.getLogger("fplabs")
DEBUG = os.environ.get("FPLABS_DEBUG") == "1"
MAX_ENTRY_ID = 20_000_000

app = FastAPI(title="FPLabs by KoalaaDev",
              docs_url="/api/docs" if DEBUG else None,
              redoc_url=None,
              openapi_url="/api/openapi.json" if DEBUG else None)

# outermost first: headers wrap everything, then the body cap, then the
# session (so every handler sees a principal), then the CSRF guard
app.add_middleware(security.MutationGuard)
app.add_middleware(security.SessionMiddleware)
app.add_middleware(security.BodyLimit)
app.add_middleware(security.SecurityHeaders)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db(config.DB_PATH)
    userdata.connect().close()
    # Pull every club's shirt and badge into the local cache in the background,
    # so the first pitch view is not 15 cross-origin requests deep.
    try:
        codes = [t["code"] for t in services.bootstrap()["teams"] if t.get("code")]
        images.prewarm(codes)
    except Exception:
        pass          # art is cosmetic; never let it block startup
    if scheduler.start():
        log.info("auto-refresh scheduler started")
    # warm the minutes model's start probabilities before the first visitor;
    # the Players page never waits on this either way
    def _warm():
        for fn in (services.bootstrap, services._history_rates,
                   services._team_form_strengths):
            try:
                fn()
            except Exception:  # noqa: BLE001 - offline is fine
                pass
        services._compute_start_probs(config.CURRENT_SEASON)

    threading.Thread(target=_warm, daemon=True, name="warm-caches").start()


@app.on_event("shutdown")
def _shutdown() -> None:
    scheduler.stop()


# --- request context ---------------------------------------------------------

def principal(request: Request) -> str:
    return auth.principal_of(request.state.session)


def current_user(request: Request) -> dict | None:
    uid = request.state.session.get("uid")
    if not uid:
        return None
    cached = getattr(request.state, "_user", None)
    if cached is None:
        cached = userdata.get_user(uid) or {}
        request.state._user = cached
    return cached or None


def require_admin(request: Request) -> dict:
    user = current_user(request)
    if not auth.is_admin(user):
        raise HTTPException(403, "admin only")
    return user


def rate(cls: str):
    def _dep(request: Request) -> None:
        if not security.limiter.allow(request.state.client_ip, cls):
            raise HTTPException(429, "too many requests — slow down a little")
    return _dep


def _entry_id(entry_id: int) -> int:
    if not (0 < entry_id <= MAX_ENTRY_ID):
        raise HTTPException(400, "not a valid FPL team id")
    return entry_id


def _entitlements(request: Request) -> dict:
    return plans.entitlements(plans.plan_for(current_user(request)))


# --- auth ------------------------------------------------------------------

@app.get("/api/auth/me")
def auth_me(request: Request, _=Depends(rate("cheap"))):
    user = current_user(request)
    plan = plans.plan_for(user)
    return {"user": auth.public_user(user),
            "google_login": auth.google_enabled(),
            "plan": plan, "entitlements": plans.entitlements(plan),
            "plans_enforced": plans.enforced(),
            "prefs": services.load_prefs(principal(request))}


@app.get("/api/auth/google/start")
def auth_google_start(request: Request, next: str = "/", _=Depends(rate("auth"))):
    if not auth.google_enabled():
        raise HTTPException(503, "Google sign-in is not configured on this server")
    url, state_cookie = auth.start_login(next)
    resp = RedirectResponse(url, status_code=302)
    resp.headers.append("set-cookie", auth.cookie_header(
        auth.OAUTH_COOKIE, state_cookie, auth.OAUTH_STATE_SECONDS,
        path="/api/auth/google"))
    return resp


@app.get("/api/auth/google/callback")
def auth_google_callback(request: Request, code: str | None = None,
                         state: str | None = None, error: str | None = None,
                         _=Depends(rate("auth"))):
    if error or not code:
        return RedirectResponse("/?login=cancelled", status_code=302)
    try:
        claims, next_path = auth.finish_login(
            code, request.cookies.get(auth.OAUTH_COOKIE), state)
    except Exception as exc:  # noqa: BLE001 - shown as a login failure
        log.warning("google login failed: %s", exc)
        return RedirectResponse("/?login=failed", status_code=302)
    user = userdata.upsert_user(claims["sub"], claims["email"],
                                claims.get("name"), claims.get("picture"))
    old = principal(request)
    new_principal = f"user:{user['id']}"
    if old != new_principal:
        userdata.migrate(old, new_principal)
    if auth.is_admin(user):
        adopted = services.adopt_legacy(new_principal)
        if adopted:
            log.info("legacy planner files adopted by %s: %s", user["email"], adopted)
    request.state.session = auth.new_session(user["id"])
    request.state.session_dirty = True
    resp = RedirectResponse(next_path, status_code=302)
    resp.headers.append("set-cookie", auth.cookie_header(
        auth.OAUTH_COOKIE, "", 0, path="/api/auth/google"))
    return resp


@app.delete("/api/auth/me")
def auth_delete(request: Request, _=Depends(rate("auth"))):
    """Delete the signed-in account and every document saved under it —
    the deletion the privacy policy promises, immediate and permanent."""
    user = current_user(request)
    if not user:
        raise HTTPException(401, "not signed in")
    userdata.delete_user(user["id"])
    request.state.session = auth.new_session()
    request.state.session_dirty = True
    return {"deleted": True, "user": None}


@app.post("/api/auth/logout")
def auth_logout(request: Request, _=Depends(rate("auth"))):
    request.state.session = auth.new_session()
    request.state.session_dirty = True
    return {"user": None}


@app.get("/api/prefs")
def prefs_get(request: Request, _=Depends(rate("cheap"))):
    return services.load_prefs(principal(request))


@app.put("/api/prefs")
def prefs_put(request: Request, body: dict, _=Depends(rate("write"))):
    return services.save_prefs(body or {}, principal(request))


# --- meta / data ----------------------------------------------------------

@app.get("/api/status")
def status(_=Depends(rate("cheap"))):
    return services.status_payload()


@app.get("/api/players")
def players(_=Depends(rate("cheap"))):
    return services.players_payload()


_known_codes: tuple[float, set[int]] = (0.0, set())


def _code_allowed(kind: str, code: int) -> bool:
    """Only art for players and clubs that exist may be fetched through the
    cache: an unbounded `code` would let anyone fill the disk one PNG at a
    time and use the server as a proxy against the Premier League CDN."""
    global _known_codes
    if time.monotonic() - _known_codes[0] > 900:
        try:
            bs = services.bootstrap()
            codes = {int(t["code"]) for t in bs["teams"] if t.get("code")}
            codes |= {int(e["code"]) for e in bs["elements"] if e.get("code")}
            _known_codes = (time.monotonic(), codes)
        except Exception:  # noqa: BLE001 - offline: keep what we had
            _known_codes = (time.monotonic(), _known_codes[1])
    return code in _known_codes[1]


@app.get("/api/img/{kind}/{code}")
def api_img(kind: str, code: str):
    """Club shirt or badge, cached on disk after the first fetch."""
    # `code` is a string on purpose: a declared int makes FastAPI answer a
    # stray "undefined" with a 422 validation error, which is noise in the
    # console for what is simply a missing image.
    try:
        code_i = int(code)
    except (TypeError, ValueError):
        raise HTTPException(404, "no such image")
    if kind not in images.SOURCES or not _code_allowed(kind, code_i):
        raise HTTPException(404, "no such image")
    path = images.path_for(kind, code_i)
    if not path:
        raise HTTPException(404, "image unavailable")
    return FileResponse(path, media_type=images.media_type(kind),
                        headers={"Cache-Control": "public, max-age=604800, immutable"})


@app.get("/api/context")
def api_context(_=Depends(rate("cheap"))):
    """Transfermarkt context: injuries, age, contract, value, manager."""
    return services.transfermarkt_context()


@app.get("/api/prices")
def api_prices(limit: int = 30, _=Depends(rate("cheap"))):
    return services.prices_payload(limit=max(5, min(60, limit)))


@app.get("/api/fixtures")
def fixtures(_=Depends(rate("cheap"))):
    return services.fixtures_payload()


@app.get("/api/projections")
def projections(_=Depends(rate("cheap"))):
    return services.projections_payload()


@app.get("/api/player/{player_id}/breakdown")
def player_breakdown(player_id: int, _=Depends(rate("cheap"))):
    """Where a player's projection comes from, per gameweek."""
    if player_id <= 0:
        raise HTTPException(400, "bad player id")
    out = services.player_breakdown(player_id)
    if out is None:
        raise HTTPException(404, "no projection for that player")
    return out


@app.get("/api/projections/history")
def projections_history(request: Request, _=Depends(rate("cheap"))):
    if not _entitlements(request)["history"]:
        return {"snapshots": [], "locked": True}
    return services.projection_history_payload()


@app.post("/api/projections/build")
def projections_build(body: dict, request: Request, admin=Depends(require_admin)):
    gws = body.get("gws")
    if not gws:
        raise HTTPException(400, "gws required")
    if jobs.running("projections") or jobs.running("solve") or jobs.running("refresh"):
        raise HTTPException(409, "a projection/solve job is already running")
    job_id = jobs.start("projections", services.build_projections,
                        [int(g) for g in gws][:8], force=bool(body.get("force")),
                        blend=body.get("blend"), owner="system")
    return {"job_id": job_id}


@app.post("/api/pull")
def pull(request: Request, body: dict | None = None, admin=Depends(require_admin)):
    if jobs.running():
        raise HTTPException(409, "another job is already running")
    job_id = jobs.start("pull", services.run_pull,
                        bool((body or {}).get("understat", True)), owner="system")
    return {"job_id": job_id}


@app.get("/api/live")
def api_live(request: Request, force: int = 0, _=Depends(rate("cheap"))):
    """The live desk. Cached for a minute server-side: it is the one page
    everyone opens at the same time, in the hour before a deadline.

    Outside its window (24 h before a deadline to 6 h after) the desk is an
    admin preview, so everyone else gets only the schedule — the tab is hidden
    for them too, and the endpoint must not hand the content out early to
    anyone who calls it directly. `force` is honoured for admins only; it
    bypasses the cache and is a CPU knob on a public server."""
    admin = auth.is_admin(current_user(request))
    win = live.window((services.status_payload().get("deadlines") or {}))
    if win["phase"] == "idle" and not admin:
        return {"window": win, "preview_only": True}
    return live.payload(force=bool(force) and admin)


@app.get("/api/admin/model")
def admin_model(request: Request, force: int = 0, admin=Depends(require_admin)):
    """How the model has actually been doing, from the scorecards the
    scheduled refresh writes after every gameweek."""
    return modelhistory.payload(force=bool(force))


@app.get("/api/admin/deadline")
def admin_deadline(request: Request, force: int = 0, admin=Depends(require_admin)):
    """What the model is thinking before the deadline — the operator's desk."""
    return deadline.payload(force=bool(force))


@app.post("/api/refresh")
def refresh_now(request: Request, admin=Depends(require_admin)):
    """Pull + reproject, the same thing the scheduler does overnight."""
    job_id = scheduler.run_now("manual")
    if not job_id:
        raise HTTPException(409, "another job is already running")
    return {"job_id": job_id}


@app.get("/api/entry/{entry_id}")
def entry(entry_id: int, request: Request, _=Depends(rate("cheap"))):
    return services.entry_payload(_entry_id(entry_id), principal(request))


@app.get("/api/league/{league_id}")
def league(league_id: int, request: Request, gw: int | None = None,
           limit: int = 20, _=Depends(rate("league"))):
    if not (0 < league_id <= 50_000_000):
        raise HTTPException(400, "not a valid league id")
    if not _entitlements(request)["league"]:
        raise HTTPException(402, "mini-league analysis is a Pro feature")
    try:
        return services.league_payload(league_id, gw=gw,
                                       limit=max(2, min(50, limit)))
    except Exception as exc:
        raise HTTPException(400, f"league fetch failed: {exc}")


# --- my team (pre-deadline squads are private to the public API) ----------

@app.get("/api/myteam")
def myteam_get(request: Request, _=Depends(rate("cheap"))):
    doc = services.load_my_team(principal(request))
    return doc or {"squad": None}


@app.put("/api/myteam")
def myteam_put(body: dict, request: Request, _=Depends(rate("write"))):
    try:
        return services.save_my_team(body, principal(request))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(400, str(exc) or "bad squad")


@app.delete("/api/myteam")
def myteam_delete(request: Request, _=Depends(rate("write"))):
    services.save_my_team(None, principal(request))
    return {"squad": None}


@app.post("/api/myteam/paste")
def myteam_paste(body: dict, request: Request, _=Depends(rate("write"))):
    """Squad pasted from the FPLabs bookmarklet (no cookie involved)."""
    payload = body.get("payload")
    if not payload:
        raise HTTPException(400, "payload required")
    try:
        return services.import_my_team_from_payload(
            payload, entry_id=body.get("entry"), principal=principal(request))
    except Exception as exc:
        raise HTTPException(400, f"import failed: {exc}")


@app.post("/api/myteam/import")
def myteam_import(body: dict, request: Request, _=Depends(rate("write"))):
    # A visitor's FPL session cookie is a password-equivalent. It is never
    # stored, but it does transit this server, so the route is off unless the
    # operator has TLS and turns it on deliberately.
    if os.environ.get("FPLABS_ALLOW_COOKIE_IMPORT") != "1":
        raise HTTPException(403, "cookie import is disabled on this server — "
                                 "use the bookmarklet")
    cookie = (body.get("cookie") or "").strip()
    try:
        entry_id = int(body.get("entry") or 0)
    except (TypeError, ValueError):
        entry_id = 0
    if not cookie or not entry_id or len(cookie) > 8000:
        raise HTTPException(400, "cookie and entry required")
    try:
        return services.import_my_team_with_cookie(
            _entry_id(entry_id), cookie, principal(request))
    except Exception as exc:  # surface the reason (401, no picks, …)
        raise HTTPException(400, f"import failed: {exc}")


# --- solver ---------------------------------------------------------------

@app.post("/api/solve")
def solve(body: dict, request: Request, _=Depends(rate("heavy"))):
    who = principal(request)
    if jobs.running_for(who, "solve"):
        raise HTTPException(409, "your previous solve is still running")
    if len(jobs.running("solve")) >= int(os.environ.get("FPLABS_MAX_SOLVES", 2)):
        raise HTTPException(503, "the solver is busy — try again in a minute")
    params, clamped = plans.check(_entitlements(request), body or {})
    if params.get("entry") is not None:
        try:
            params["entry"] = _entry_id(int(params["entry"]))
        except (TypeError, ValueError):
            raise HTTPException(400, "not a valid FPL team id")
    job_id = jobs.start("solve", services.run_solve, params, who, owner=who)
    return {"job_id": job_id, "clamped": clamped}


@app.get("/api/jobs/{job_id}")
def job(job_id: str, request: Request, _=Depends(rate("cheap"))):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "unknown job")
    if j.get("owner") not in (principal(request), "system") and \
            not auth.is_admin(current_user(request)):
        raise HTTPException(404, "unknown job")
    return j


# --- drafts / watch ---------------------------------------------------------

@app.get("/api/transferwatch")
def api_transfer_watch(request: Request, _=Depends(rate("cheap"))):
    return services.transfer_watch_payload(principal(request))


@app.put("/api/transferwatch")
def api_save_transfer_watch(body: dict, request: Request, _=Depends(rate("write"))):
    services.save_transfer_watch(body or {}, principal(request))
    return services.transfer_watch_payload(principal(request))


@app.get("/api/drafts")
def drafts_get(request: Request, _=Depends(rate("cheap"))):
    return services.load_drafts(principal(request))


@app.put("/api/drafts")
def drafts_put(body: dict, request: Request, _=Depends(rate("write"))):
    try:
        return services.save_drafts(body or {}, principal(request),
                                    max_drafts=_entitlements(request)["drafts"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# --- legal pages (before the static mount so they win the route) ----------
app.include_router(legal.router)


# --- static frontend (vite build output) ----------------------------------

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
