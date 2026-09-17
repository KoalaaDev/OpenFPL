"""Automatic refresh: the model runs itself, nobody presses a button.

A public planner cannot depend on someone clicking "Data" — the numbers must
already be current when a visitor arrives. A daemon thread therefore runs
the pipeline on a schedule:

* once a day at `FPLABS_REFRESH_UTC` (HH:MM, default 04:30 — after the
  night's odds and FPL price changes, before Europe wakes up),
* `FPLABS_PRE_DEADLINE_HOURS` (default 2) before every gameweek deadline,
  when the last team news has landed and the availability overlay is worth
  the most,
* at startup, when the projection cache is missing or older than a day.

A refresh is: pull (FPL live + backfill + odds, Understat when enabled)
then project the next `FPLABS_HORIZON` gameweeks, so the Planner, Solver
and Projections tabs are all warm. It goes through `jobs`, so the UI sees it
as a running job exactly as it would a manual one, and it never overlaps a
manual pull or solve. Failures are recorded, not raised — the next tick
tries again.

Set `FPLABS_AUTO_REFRESH=0` to disable (tests, laptops).
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from . import jobs

STATE_PATH = None      # resolved lazily: services owns the cache directory
_thread: threading.Thread | None = None
_stop = threading.Event()
_state: dict = {"last_run": None, "last_ok": None, "last_error": None,
                "next_run": None, "runs": 0}
_lock = threading.Lock()


def enabled() -> bool:
    return os.environ.get("FPLABS_AUTO_REFRESH", "1") != "0"


def horizon() -> int:
    try:
        return max(1, min(8, int(os.environ.get("FPLABS_HORIZON", 6))))
    except ValueError:
        return 6


def _daily_hhmm() -> tuple[int, int]:
    raw = os.environ.get("FPLABS_REFRESH_UTC", "04:30")
    try:
        h, m = raw.split(":")
        return max(0, min(23, int(h))), max(0, min(59, int(m)))
    except ValueError:
        return 4, 30


def _pre_deadline_hours() -> float:
    try:
        return max(0.25, float(os.environ.get("FPLABS_PRE_DEADLINE_HOURS", 2)))
    except ValueError:
        return 2.0


def next_run(now: float, deadlines: list[float], daily: tuple[int, int],
             pre_hours: float) -> tuple[float, str]:
    """The next moment to refresh and why — pure, so it is testable.

    `deadlines` are unix timestamps; only those still ahead count, and each
    contributes one refresh `pre_hours` before it.
    """
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    daily_dt = dt.replace(hour=daily[0], minute=daily[1], second=0, microsecond=0)
    if daily_dt.timestamp() <= now:
        daily_dt += timedelta(days=1)
    best, why = daily_dt.timestamp(), "daily"
    for d in sorted(deadlines):
        t = d - pre_hours * 3600.0
        if t > now and t < best:
            best, why = t, "pre-deadline"
            break
    return best, why


def state() -> dict:
    with _lock:
        return dict(_state)


def _save_state() -> None:
    from . import services
    try:
        os.makedirs(services.WEB_CACHE, exist_ok=True)
        path = os.path.join(services.WEB_CACHE, "scheduler.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state(), f)
    except OSError:
        pass


def _refresh(job_id: str, *, understat: bool = True) -> dict:
    """One full refresh, run inside a job thread."""
    from . import services
    from fpl_engine import config, db
    from fpl_engine.pipeline import next_gw

    out = services.run_pull(job_id, understat)
    jobs.progress(job_id, "Recording FPL team news…", pct=0.5)
    pull_team_news()
    # Friday's press conferences (BBC) -> player-level availability
    # observations for the coming gameweek; the engine's live path applies
    # them on top of FPL's own status (Round 18). Never fatal.
    try:
        from acquire import storage as _st
        from acquire.sources import bbc_pressers as _bp
        from fpl_engine import pressers as _pr
        from acquire.sources import bbc as _bbc
        jobs.progress(job_id, "Archiving BBC lineups and press conferences…", pct=0.55)
        with _st.connect(config.DB_PATH) as _c:
            _st.init(_c)
            _bbc.pull(_c)          # last eight days of lineups: the shipped role block reads them
            _bp.pull(_c)
            _pr.extract_pressers(_c, seasons=[config.CURRENT_SEASON])
    except Exception as exc:  # noqa: BLE001
        jobs.progress(job_id, f"press conferences skipped: {exc}")
    conn = db.connect(config.DB_PATH)
    try:
        start = next_gw(conn, config.CURRENT_SEASON)
        scheduled = [r["gw"] for r in conn.execute(
            "SELECT DISTINCT gw FROM fixture WHERE season=? AND gw>=? AND gw "
            "IS NOT NULL ORDER BY gw", (config.CURRENT_SEASON, start))]
    finally:
        conn.close()
    gws = scheduled[:horizon()] or [start]
    jobs.progress(job_id, f"Projecting GW{gws[0]}–GW{gws[-1]}…", pct=0.6)
    services.build_projections(job_id, gws, force=True)
    with _lock:
        _state["reproject_pending"] = []
    # The two after-the-fact scorecards used to be things an admin was told to
    # run in a terminal, which meant they were usually empty and the desk read
    # as though nothing was automatic. They are library calls; they run here,
    # once per finished gameweek, and never fail a refresh.
    score_finished_gameweeks(job_id)
    # keep the anonymous-doc table from growing without bound
    try:
        from . import userdata
        userdata.purge_anonymous()
    except Exception:  # noqa: BLE001
        pass
    return {"pulled": out.get("summary"), "projected": gws}


def score_finished_gameweeks(job_id: str | None = None, *, limit: int = 6) -> dict:
    """Post-mortem and lineup-feed scorecards for every finished gameweek that
    has not been scored yet.

    Both write one JSON under ``data/`` and both are cheap to skip (the file
    already being there is the "done" marker), so a refresh re-scores nothing
    and a season backfills itself a gameweek at a time. Errors are reported
    into the job and swallowed: a scorecard must never cost a data pull.
    """
    from fpl_engine import config, db
    done = {"postmortem": [], "lineup_feed": [], "model_record": [], "errors": []}
    season = config.CURRENT_SEASON
    conn = db.connect(config.DB_PATH)
    try:
        finished = [int(r[0]) for r in conn.execute(
            "SELECT DISTINCT gw FROM fixture WHERE season=? AND finished=1 "
            "AND gw IS NOT NULL ORDER BY gw DESC", (season,))][:limit]
        for gw in sorted(finished):
            path = os.path.join(config.DATA_DIR, f"postmortem_{season}_gw{gw}.json")
            if os.path.exists(path):
                continue
            try:
                from fpl_engine import postmortem
                if job_id:
                    jobs.progress(job_id, f"Scoring GW{gw} against what happened…")
                postmortem.run(conn, season=season, gw=gw)
                done["postmortem"].append(gw)
            except Exception as exc:  # noqa: BLE001
                done["errors"].append(f"postmortem gw{gw}: {exc}")
        try:
            # the model's picks before each deadline, scored against what
            # happened, the average manager and perfect hindsight
            from . import modelrecord
            r = modelrecord.refresh(progress=(lambda m: jobs.progress(job_id, m)) if job_id else None)
            done["model_record"] = r.get("built", [])
        except Exception as exc:  # noqa: BLE001
            done["errors"].append(f"model record: {exc}")
        try:
            # the model's own season-long team: carried week to week with
            # real transfers, banked free transfers and -4 hits
            from . import modelteam
            r = modelteam.build(progress=(lambda m: jobs.progress(job_id, m)) if job_id else None)
            done["model_team"] = r.get("built", [])
        except Exception as exc:  # noqa: BLE001
            done["errors"].append(f"model team: {exc}")
        try:
            from fpl_engine import lineup_feed as lf
            archive = lf.load_archive(season)
            scored = _scored_feed_gws(season)
            for gw in sorted(finished):
                if gw in scored or archive.empty:
                    continue
                if not len(archive[(archive["gw"] == gw)
                                   & (archive["status"] == "predicted")]):
                    continue        # nothing was forecast for that gameweek
                if job_id:
                    jobs.progress(job_id, f"Scoring the lineup feed for GW{gw}…")
                lf.save(season, lf.score_gw(conn, season, gw, archive=archive))
                done["lineup_feed"].append(gw)
        except Exception as exc:  # noqa: BLE001
            done["errors"].append(f"lineup feed: {exc}")
    finally:
        conn.close()
    return done


def _scored_feed_gws(season: str) -> set[int]:
    from fpl_engine import config
    path = os.path.join(config.DATA_DIR, f"lineup_feed_{season}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return {int(g) for g in (json.load(fh).get("gws") or {})}
    except (OSError, ValueError):
        return set()


def run_now(reason: str = "manual") -> str | None:
    """Start a refresh job unless one (or a manual pull/solve) is running."""
    if jobs.running():
        return None
    job_id = jobs.start("refresh", _refresh, owner="system",
                        understat=os.environ.get("FPLABS_UNDERSTAT", "1") != "0")
    with _lock:
        _state["last_run"] = time.time()
        _state["last_reason"] = reason
        _state["runs"] += 1
        _state["job_id"] = job_id
    return job_id


def _await(job_id: str) -> None:
    while True:
        j = jobs.get(job_id)
        if not j or j["status"] != "running":
            with _lock:
                _state["last_ok"] = time.time() if j and j["status"] == "done" else _state["last_ok"]
                _state["last_error"] = (j or {}).get("error") if j and j["status"] == "error" else None
            _save_state()
            return
        if _stop.wait(5):
            return


def _cache_age_hours() -> float | None:
    from . import services
    try:
        with open(services.PROJ_PATH, encoding="utf-8") as f:
            doc = json.load(f)
        t = doc.get("updated_at")
        return (time.time() - float(t)) / 3600.0 if t else None
    except (OSError, ValueError):
        return None


def _deadlines() -> list[float]:
    from . import services
    try:
        return list(services._deadlines().values())
    except Exception:  # noqa: BLE001 - offline: daily cadence only
        return []


def _loop(startup_delay: float) -> None:
    if _stop.wait(startup_delay):
        return
    age = _cache_age_hours()
    if age is None or age > 24:
        jid = run_now("startup")
        if jid:
            _await(jid)
    while not _stop.is_set():
        when, why = next_run(time.time(), _deadlines(), _daily_hhmm(),
                             _pre_deadline_hours())
        with _lock:
            _state["next_run"] = when
            _state["next_reason"] = why
        _save_state()
        # wake early and re-plan: deadlines can move (postponements) and the
        # daily time may be edited while the process runs
        wait = max(1.0, min(when - time.time(), 1800.0))
        if _stop.wait(wait):
            return
        if time.time() + 1.0 < when:
            continue
        jid = run_now(why)
        if jid:
            _await(jid)
        else:                 # a manual job holds the lock; retry shortly
            if _stop.wait(120):
                return


# ------------------------------------------------------------- team news --
#
# FPL's team news — status, chance of playing, the text and FPL's own
# `news_added` — is recorded as a change log (`acq_player_availability`) that
# the Live desk's feed and the model record's point-in-time replays read.
# NOTHING in the app wrote it: the full refresh overwrites each player's
# current status, but the log only grew when someone ran
# `python -m acquire pull --source fpl` by hand, which last happened on
# 2026-09-10. So the desk showed "6 d ago" while FPL had news from that
# morning.
#
# Polling it is one bootstrap request and writes only the players whose state
# changed, so it runs far more often than the model does.

def news_minutes() -> float:
    try:
        return max(5.0, float(os.environ.get("FPLABS_NEWS_MINUTES", 15)))
    except ValueError:
        return 15.0


def pull_team_news() -> dict:
    """Record any FPL availability changes since the last pull. Never raises."""
    from fpl_engine import config
    try:
        from acquire import storage as _st
        from acquire.sources import fpl_availability as _fa
        with _st.connect(config.DB_PATH) as conn:
            _st.init(conn)
            out = _fa.pull(conn, season=config.CURRENT_SEASON)
        with _lock:
            _state["news_last_run"] = time.time()
            _state["news_last_changes"] = out.get("changes")
            _state["news_last_error"] = out.get("error")
        if out.get("changes") and out.get("raw_id"):
            try:
                with _st.connect(config.DB_PATH) as conn:
                    triggers = significant_changes(conn, config.CURRENT_SEASON, out["raw_id"])
                out["significant"] = [t["name"] for t in triggers]
                if enabled():
                    out["reproject_job"] = maybe_reproject(triggers)
            except Exception as exc:  # noqa: BLE001
                out["significant_error"] = str(exc)
        elif enabled() and (_state.get("reproject_pending")):
            out["reproject_job"] = maybe_reproject([])   # news waiting out a cooldown
        if out.get("changes"):
            try:
                from . import live            # the desk caches for a minute
                live._cache["v"] = None
            except Exception:  # noqa: BLE001
                pass
        return out
    except Exception as exc:  # noqa: BLE001 - a news poll must never take the app down
        with _lock:
            _state["news_last_error"] = str(exc)
        return {"error": str(exc)}


# ------------------------------------------------ reproject on team news --
#
# News used to reach the Live desk within 15 minutes but the PROJECTIONS only
# at the next scheduled rebuild (daily, and 2 h before the deadline) — so a
# first-choice striker ruled out on Thursday morning kept his full projection,
# and every Solver run and model pick built on it, until Friday afternoon.
#
# A rebuild costs ~5 minutes of server time, so not every change earns one.
# A change triggers a rebuild only when it is MATERIAL (the player's status
# class changes, or his chance of playing moves by 25 points or more) and the
# player MATTERS: widely owned, or someone the model itself relies on — a
# real projection or a likely starter. At most one rebuild per
# FPLABS_REPROJECT_MINUTES (default 60); news that lands inside the cooldown
# waits for the next poll after it rather than being dropped.

REPROJECT_OWNERSHIP = 10.0      # % selected: the crowd will act on it
REPROJECT_EP = 3.0              # next-gameweek projection the model relies on
REPROJECT_P_START = 0.6         # a likely starter by the model's own P(start)
REPROJECT_CHANCE_MOVE = 25      # points of chance_of_playing


def reproject_minutes() -> float:
    try:
        return max(20.0, float(os.environ.get("FPLABS_REPROJECT_MINUTES", 60)))
    except ValueError:
        return 60.0


def _availability_class(status, chance) -> tuple[str, int]:
    """(class, chance 0-100): 'a' fit, 'd' doubt, 'o' out of this gameweek."""
    c = None if chance is None else float(chance)
    if c is not None and c <= 1:
        c *= 100
    if status in (None, "a"):
        return "a", int(c if c is not None else 100)
    if status == "d":
        return "d", int(c if c is not None else 50)
    return "o", int(c if c is not None else 0)


def material(prev, cur) -> bool:
    """Did this change move the player enough to move his projection?"""
    if prev is None:
        return cur[0] != "a"                  # first sighting: only if not fit
    return prev[0] != cur[0] or abs(prev[1] - cur[1]) >= REPROJECT_CHANCE_MOVE


def significant_changes(conn, season: str, raw_id: int) -> list[dict]:
    """The changes written by one poll that should trigger a rebuild."""
    from . import services
    rows = conn.execute(
        "SELECT player_id, status, chance_next, observed_utc FROM acq_player_availability "
        "WHERE season = ? AND raw_id = ?", (season, raw_id)).fetchall()
    if not rows:
        return []
    try:
        own = {p["id"]: p.get("own") or 0.0 for p in services.players_payload().get("players", [])}
    except Exception:  # noqa: BLE001
        own = {}
    cache = services._load_proj_cache()
    gw = str(services.editable_gw() or "")
    start_p = services._model_start_probs(season)
    names = {int(r[0]): r[1] for r in conn.execute(
        "SELECT player_id, web_name FROM player WHERE season = ?", (season,))}
    out = []
    for pid, status, chance, observed in rows:
        prev = conn.execute(
            "SELECT status, chance_next FROM acq_player_availability "
            "WHERE season = ? AND player_id = ? AND observed_utc < ? "
            "ORDER BY observed_utc DESC LIMIT 1", (season, pid, observed)).fetchone()
        before = _availability_class(*prev) if prev else None
        after = _availability_class(status, chance)
        if not material(before, after):
            continue
        ep = ((cache.get("players", {}).get(str(pid)) or {}).get("ep") or {}).get(gw) or 0.0
        why = []
        if own.get(pid, 0.0) >= REPROJECT_OWNERSHIP:
            why.append(f"{own[pid]:.0f}% owned")
        if ep >= REPROJECT_EP:
            why.append(f"projected {ep:.1f}")
        if (start_p.get(pid) or 0.0) >= REPROJECT_P_START:
            why.append(f"P(start) {start_p[pid]:.2f}")
        if why:
            out.append({"player_id": int(pid), "name": names.get(int(pid), str(pid)),
                        "from": before[0] if before else None, "to": after[0],
                        "chance": after[1], "why": why})
    return out


def _reproject(job_id: str) -> dict:
    """Statuses from the bootstrap, then the horizon re-projected. No pull,
    no press conferences, no scorecards: only what a status change moves."""
    from . import services
    from fpl_engine import config, db
    from fpl_engine.ingest import fpl_api
    jobs.progress(job_id, "Updating player statuses…", pct=0.05)
    conn = db.connect(config.DB_PATH)
    try:
        fpl_api.ingest_bootstrap(conn, config.CURRENT_SEASON)
        conn.commit()
        start = services.editable_gw() or 1
        gws = [r[0] for r in conn.execute(
            "SELECT DISTINCT gw FROM fixture WHERE season = ? AND gw >= ? AND gw IS NOT NULL "
            "ORDER BY gw", (config.CURRENT_SEASON, start))][:horizon()]
    finally:
        conn.close()
    services.build_projections(job_id, gws, force=True)
    try:
        from . import live
        live._cache["v"] = None
    except Exception:  # noqa: BLE001
        pass
    return {"projected": gws}


def maybe_reproject(triggers: list[dict]) -> str | None:
    """Queue triggers; start a rebuild when the cooldown allows and nothing
    else is running. Returns the job id when one starts."""
    now = time.time()
    with _lock:
        pending = {t["player_id"]: t for t in _state.get("reproject_pending") or []}
        for t in triggers:
            pending[t["player_id"]] = t
        _state["reproject_pending"] = list(pending.values())
        last = _state.get("reproject_last_run") or 0.0
    if not pending or now - last < reproject_minutes() * 60.0 or jobs.running():
        return None
    job_id = jobs.start("reproject", _reproject, owner="system")
    with _lock:
        _state["reproject_last_run"] = now
        _state["reproject_last_reason"] = [
            f"{t['name']} ({', '.join(t['why'])})" for t in pending.values()][:12]
        _state["reproject_pending"] = []
        _state["job_id"] = job_id
    _save_state()
    return job_id


def _news_loop(startup_delay: float) -> None:
    if _stop.wait(startup_delay):
        return
    while not _stop.is_set():
        pull_team_news()
        if _stop.wait(news_minutes() * 60.0):
            return


_news_thread: threading.Thread | None = None


def start(startup_delay: float = 20.0) -> bool:
    global _thread, _news_thread
    if not enabled() or (_thread and _thread.is_alive()):
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(startup_delay,),
                               daemon=True, name="fplabs-refresh")
    _thread.start()
    _news_thread = threading.Thread(target=_news_loop, args=(startup_delay,),
                                    daemon=True, name="fplabs-news")
    _news_thread.start()
    return True


def stop() -> None:
    _stop.set()
