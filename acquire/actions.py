"""File-based collector for scheduled GitHub Actions runs.

The SQLite database is a local artefact — it is gitignored and rebuilt from
free sources, so nothing collected into it survives a fresh clone. This
module makes **the repository itself the point-in-time archive**: a scheduled
Actions run (see .github/workflows/collect.yml) fetches the official FPL
bootstrap, appends what changed to plain files under ``data/collected/`` and
commits them. A year of runs becomes the deadline-honest history that the
modelling rounds established is the one unexploited lever (team news /
minutes), and that no third party archives.

Design rules, in priority order:

* **Point-in-time or nothing.** Every record carries ``observed_utc`` (this
  run's clock). FPL's own ``news_added`` is kept verbatim as the source's
  claim, never substituted for ours.
* **Append-only and idempotent.** The availability log gains a line only
  when a player's (status, chance, news) actually changed; re-running on an
  unchanged payload writes nothing, so the git history stays honest and
  small. Files are never rewritten retroactively.
* **Stdlib only.** The Actions job installs nothing; ``acquire.core.http``
  is urllib. Model code is never imported (the acquirer test enforces it).
* **Small.** One run appends a few KB of changes plus one ~15 KB gzipped
  market snapshot for the deadline-decay research. Ownership is one CSV per
  gameweek, overwritten until its deadline passes and frozen after —
  the file that remains is the last pre-deadline state.

Layout under ``data/collected/``:

    availability.jsonl        append-only change log (the team-news archive)
    availability_state.json   last known state per player code (diff base)
    snapshots/<utc>.csv.gz    compact per-run market snapshot (decay research)
    ownership/gw<n>.csv       last pre-deadline ownership for each gameweek
    picks/<season>/gw<n>.csv  elite-panel squads, once per finished deadline
    panel.json                the fixed panel (entry ids; graded on PAST
                              seasons only — see fpl_managers.build_panel)
    meta.json                 last run summary

``actions-import`` replays the archive into the local SQLite change-log
tables so research code keeps one query surface.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os

from .core import http
from .sources import fpl_availability, predicted_lineups

BOOTSTRAP_URL = fpl_availability.URL
API = "https://fantasy.premierleague.com/api"
OUT_DIR = os.path.join("data", "collected")
SNAPSHOT_FIELDS = ("id", "code", "status", "chance_next", "news_added",
                   "now_cost", "selected_by_percent", "transfers_in_event",
                   "transfers_out_event", "ep_next", "form")
PICK_FIELDS = ("gw", "entry_id", "element", "slot", "multiplier",
               "is_captain", "is_vice", "active_chip", "observed_utc")
LINEUP_FIELDS = ("observed_utc", "gw", "team_abbr", "side", "status",
                 "position", "slot", "player", "rotowire_id")
MAX_PICK_GWS_PER_RUN = 3      # backfill politely, a few deadlines at a time


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def _season_label(boot: dict) -> str:
    for ev in boot.get("events", []):
        dl = ev.get("deadline_time") or ""
        if len(dl) >= 7:
            y = int(dl[:4])
            m = int(dl[5:7])
            start = y if m >= 7 else y - 1
            return f"{start}-{str(start + 1)[-4:][-2:]}"
    return "unknown"


def _availability_rows(boot: dict, observed: str, season: str) -> list[dict]:
    rows = []
    for e in boot.get("elements", []):
        rows.append({
            "observed_utc": observed, "season": season,
            "player_id": int(e["id"]), "code": int(e["code"]),
            "web_name": e.get("web_name"),
            "status": e.get("status"),
            "chance_next": fpl_availability._chance(e),
            "news": (e.get("news") or "").strip() or None,
            "news_added": e.get("news_added"),
        })
    return rows


def _update_availability(out_dir: str, rows: list[dict]) -> int:
    """Append changed states to the JSONL log; return how many changed."""
    state_path = os.path.join(out_dir, "availability_state.json")
    log_path = os.path.join(out_dir, "availability.jsonl")
    state = _read_json(state_path, {})
    changed = []
    for r in rows:
        key = str(r["code"])
        sig = [r["status"], r["chance_next"], r["news"], r["news_added"]]
        if state.get(key) != sig:
            state[key] = sig
            changed.append(r)
    if changed:
        os.makedirs(out_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            for r in changed:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        _write_json(state_path, state)
    return len(changed)


def _write_snapshot(out_dir: str, boot: dict, observed: str,
                    next_gw: int | None) -> str:
    """One compact gzipped market snapshot per run, for decay research."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(SNAPSHOT_FIELDS)
    for e in boot.get("elements", []):
        row = dict(e)
        row["chance_next"] = fpl_availability._chance(e)
        w.writerow([row.get(f) for f in SNAPSHOT_FIELDS])
    stamp = observed.replace(":", "").replace("-", "")[:13]  # YYYYMMDDTHHMM
    name = f"{stamp}_gw{next_gw or 0:02d}.csv.gz"
    path = os.path.join(out_dir, "snapshots", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    return name


def _write_ownership(out_dir: str, boot: dict, next_gw: int | None) -> str | None:
    """Ownership snapshot for the NEXT gameweek, overwritten until its
    deadline passes. FPL flips ``is_next`` at the deadline, so the last file
    written under a gameweek's name is the last pre-deadline state — the
    exact lagged-EO input the rank work uses, collected forward."""
    if not next_gw:
        return None
    path = os.path.join(out_dir, "ownership", f"gw{next_gw:02d}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "code", "selected_by_percent",
                    "transfers_in_event", "transfers_out_event", "now_cost"])
        for e in boot.get("elements", []):
            w.writerow([e.get("id"), e.get("code"),
                        e.get("selected_by_percent"),
                        e.get("transfers_in_event"),
                        e.get("transfers_out_event"), e.get("now_cost")])
    return os.path.basename(path)


def _collect_picks(out_dir: str, boot: dict, season: str,
                   observed: str) -> list[str]:
    """Panel squads for finished deadlines that are not yet archived.

    Picks lock at the deadline, so any collection after it is point-in-time
    valid. Only the current season is served by the endpoint — which is the
    whole reason this archive exists.
    """
    panel = _read_json(os.path.join(out_dir, "panel.json"), None)
    if not panel or not panel.get("entries"):
        return []
    # any gw whose deadline passed qualifies; is_current covers the live one
    passed = sorted({int(ev["id"]) for ev in boot.get("events", [])
                     if ev.get("finished") or ev.get("is_current")})
    written = []
    for gw in passed:
        path = os.path.join(out_dir, "picks", season, f"gw{gw:02d}.csv")
        if os.path.exists(path):
            continue
        rows = []
        for eid in panel["entries"]:
            d = None
            resp = http.get(f"{API}/entry/{eid}/event/{gw}/picks/", delay=0.3)
            if resp.ok:
                try:
                    d = json.loads(resp.text)
                except ValueError:
                    d = None
            if not d or "picks" not in d:
                continue
            for p in d["picks"]:
                rows.append([gw, eid, int(p["element"]), p.get("position"),
                             p.get("multiplier"),
                             int(bool(p.get("is_captain"))),
                             int(bool(p.get("is_vice_captain"))),
                             d.get("active_chip"), observed])
        if not rows:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(PICK_FIELDS)
            w.writerows(rows)
        written.append(os.path.basename(path))
        if len(written) >= MAX_PICK_GWS_PER_RUN:
            break
    return written



def _collect_lineups(out_dir: str, season: str, next_gw: int | None,
                     observed: str) -> int:
    """Append this run's predicted XIs, append-only and timestamped.

    THIS IS THE POINT OF THE COLLECTOR. The oracle decomposition closed every
    other channel: a perfect RATE estimate is worth nothing (the attacking
    ceiling is variance, not estimator error) and a perfect LOCAL rate is
    significantly worse, so there is no form signal to chase either. What is
    left is whether a player starts and lasts an hour, and the model is
    already correctly calibrated in the band where it is unsure — because the
    manager has not decided yet. Only an outside forecast resolves it, and
    E8b prices that at ~+89 points a season, LINEAR in the fraction resolved.

    Nobody archives past predictions, so this cannot be backtested from
    history; forward collection is the only route and starting it is the
    whole value. Every row carries this run's clock, so the last observation
    STRICTLY BEFORE a deadline is the one a decision could have used — which
    is what makes the file honest later. Confirmed XIs land after the
    deadline and are kept as the ground truth to score the predictions
    against, never as an input.

    A failure is swallowed: a third-party page must not be able to break the
    scheduled run that also archives FPL's own team news.
    """
    try:
        resp = http.get(predicted_lineups.URL, delay=1.0)
        if not resp.ok:
            return 0
        rows = predicted_lineups.parse(resp.text)
    except Exception:                                # noqa: BLE001
        return 0
    rows = [r for r in rows if r.get("in_xi")]
    if not rows:
        return 0
    path = os.path.join(out_dir, "lineups", f"{season}.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    # a run that saw exactly the same XI for a team writes nothing: the
    # archive records CHANGES of forecast, which is what a decay study needs
    seen = _lineup_state(path)
    wrote = 0
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(LINEUP_FIELDS)
        for key, group in _by_side(rows):
            fingerprint = "|".join(f"{r['position']}:{r['player']}"
                                   for r in group)
            if seen.get(key) == fingerprint:
                continue
            for r in group:
                w.writerow([observed, next_gw, r["team_abbr"], r["side"],
                            r["status"], r["position"], r["slot"],
                            r["player"], r["rotowire_id"]])
                wrote += 1
    return wrote


def _by_side(rows: list[dict]):
    order, groups = [], {}
    for r in rows:
        key = (r["team_abbr"], r["side"], r["status"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    return [(k, groups[k]) for k in order]


def _lineup_state(path: str) -> dict:
    """The most recent XI already recorded for each (team, side, status)."""
    if not os.path.exists(path):
        return {}
    out: dict = {}
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return {}
    for r in rows:
        key = (r["team_abbr"], r["side"], r["status"])
        out.setdefault(key, {}).setdefault(r["observed_utc"], []).append(
            f"{r['position']}:{r['player']}")
    return {k: "|".join(v[max(v)]) for k, v in out.items() if v}


def collect(out_dir: str = OUT_DIR, *, payload: str | None = None) -> dict:
    """One scheduled run: fetch, diff, append, snapshot. Returns a summary.

    ``payload`` injects a bootstrap JSON string for tests; normally the
    official endpoint is fetched.
    """
    observed = http.utcnow()
    if payload is None:
        resp = http.get(BOOTSTRAP_URL, delay=1.0)
        if not resp.ok:
            raise RuntimeError(f"bootstrap fetch failed: HTTP {resp.status}")
        payload = resp.text
    boot = json.loads(payload)
    season = _season_label(boot)
    next_gw = next((int(ev["id"]) for ev in boot.get("events", [])
                    if ev.get("is_next")), None)

    changed = _update_availability(
        out_dir, _availability_rows(boot, observed, season))
    snap = _write_snapshot(out_dir, boot, observed, next_gw)
    own = _write_ownership(out_dir, boot, next_gw)
    picks = _collect_picks(out_dir, boot, season, observed)
    lineups = _collect_lineups(out_dir, season, next_gw, observed)

    summary = {"observed_utc": observed, "season": season,
               "next_gw": next_gw, "availability_changes": changed,
               "snapshot": snap, "ownership": own, "picks": picks,
               "lineups": lineups}
    _write_json(os.path.join(out_dir, "meta.json"), summary)
    return summary


# ------------------------------------------------------------------ import
def import_collected(conn, out_dir: str = OUT_DIR) -> dict:
    """Replay the file archive into the SQLite change-log tables, so research
    code keeps one query surface (``availability_as_of`` etc.)."""
    from . import storage
    from .sources import fpl_managers
    storage.init(conn)
    fpl_availability.register(conn)
    fpl_managers.init(conn)

    n_avail = 0
    log_path = os.path.join(out_dir, "availability.jsonl")
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cur = conn.execute(
                    "INSERT OR IGNORE INTO acq_player_availability "
                    "(season, player_id, observed_utc, source_id, "
                    "source_published_utc, status, chance_next, news, raw_id) "
                    "VALUES (?,?,?,?,?,?,?,?,NULL)",
                    (r["season"], r["player_id"], r["observed_utc"],
                     fpl_availability.SOURCE_ID, r.get("news_added"),
                     r.get("status"), r.get("chance_next"), r.get("news")))
                n_avail += cur.rowcount
    n_picks = 0
    picks_root = os.path.join(out_dir, "picks")
    if os.path.isdir(picks_root):
        for season in sorted(os.listdir(picks_root)):
            sdir = os.path.join(picks_root, season)
            for name in sorted(os.listdir(sdir)):
                with open(os.path.join(sdir, name), encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        cur = conn.execute(
                            "INSERT OR IGNORE INTO acq_manager_pick (season, "
                            "gw, entry_id, player_id, slot, multiplier, "
                            "is_captain, is_vice, active_chip, observed_utc, "
                            "source_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (season, int(r["gw"]), int(r["entry_id"]),
                             int(r["element"]), r["slot"], r["multiplier"],
                             int(r["is_captain"]), int(r["is_vice"]),
                             r["active_chip"] or None, r["observed_utc"],
                             fpl_managers.SOURCE_ID))
                        n_picks += cur.rowcount
    conn.commit()
    return {"availability_rows": n_avail, "pick_rows": n_picks}
