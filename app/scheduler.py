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


def start(startup_delay: float = 20.0) -> bool:
    global _thread
    if not enabled() or (_thread and _thread.is_alive()):
        return False
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(startup_delay,),
                               daemon=True, name="fplabs-refresh")
    _thread.start()
    return True


def stop() -> None:
    _stop.set()
