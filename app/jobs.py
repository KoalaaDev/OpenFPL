"""Tiny in-process background-job manager for long-running work (solves,
projection builds, data pulls, scheduled refreshes). One thread per job;
polled via the API.

Every job has an *owner* — the principal that started it, or "system" for
the scheduler — and the API only serves a job to its owner (or an admin), so
one visitor's solve, which contains their squad, is not readable by another
visitor who guesses the id.

Failures are reported to the client as the exception message only; the
traceback goes to the server log. A stack trace is a map of the codebase and
the file system, and it belongs in the log, not in an HTTP response.
"""
from __future__ import annotations

import logging
import threading
import traceback
import uuid
from datetime import datetime, timezone

log = logging.getLogger("fplabs.jobs")

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
MAX_KEPT = 500           # finished jobs remembered before the oldest go


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def start(kind: str, target, *args, owner: str | None = None, **kwargs) -> str:
    """Run ``target(job_id, *args, **kwargs)`` in a thread; return job id."""
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"id": job_id, "kind": kind, "status": "running",
                         "progress": [], "pct": 0.0, "result": None,
                         "error": None, "started_at": _now(),
                         "owner": owner or "system"}
        if len(_jobs) > MAX_KEPT:
            for k in [k for k, j in _jobs.items() if j["status"] != "running"][:100]:
                _jobs.pop(k, None)

    def _run():
        try:
            result = target(job_id, *args, **kwargs)
            with _lock:
                _jobs[job_id].update(status="done", result=result, pct=1.0)
        except Exception as exc:  # noqa: BLE001 - reported to the client
            log.error("job %s (%s) failed: %s\n%s", job_id, kind, exc,
                      traceback.format_exc())
            with _lock:
                _jobs[job_id].update(status="error", error=str(exc)[:500])

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def progress(job_id: str, msg: str, pct: float | None = None) -> None:
    with _lock:
        j = _jobs.get(job_id)
        if j is None:
            return
        j["progress"].append({"t": _now(), "msg": msg})
        if len(j["progress"]) > 200:
            del j["progress"][:-200]
        if pct is not None:
            j["pct"] = max(j["pct"], min(1.0, pct))


def get(job_id: str) -> dict | None:
    with _lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None


def running(kind: str | None = None) -> list[dict]:
    with _lock:
        return [dict(j) for j in _jobs.values()
                if j["status"] == "running" and (kind is None or j["kind"] == kind)]


def running_for(owner: str, kind: str | None = None) -> list[dict]:
    return [j for j in running(kind) if j.get("owner") == owner]
