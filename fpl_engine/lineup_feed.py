"""Price the predicted-lineup feed, gameweek by gameweek, on its band accuracy.

E8b (RESEARCH_LOG) established how a lineup source is judged: NOT by a
decision backtest — the Round 13 wall means a season of replays cannot resolve
it — but by its hit rate in the band where the model is genuinely unsure,
P(start) in [0.30, 0.70]. Resolving that band is worth ~89 points a season and
the value is LINEAR in the fraction resolved, so a feed is priced by

    points/season ~= (accuracy_in_band - 0.55) / 0.45 x 89

against the model's own ~55% there. ~90 ambiguous rows a gameweek means about
five gameweeks give a usable estimate, against the season-plus a decision
backtest would need.

Point-in-time discipline, per row:

* the FORECAST is the last RotoWire *predicted* XI observed strictly before the
  deadline (first kickoff minus 90 minutes). Confirmed XIs land after the
  deadline and are never an input.
* the MODEL is re-run as of the gameweek's first kickoff with the availability
  overlay taken from the acquirer's change log AS IT STOOD AT THE DEADLINE —
  not today's status. Both sides see exactly what a manager could have seen.
* the TRUTH is ``player_gw.starts`` once the fixture is finished, or, until
  the results are pulled, the club's confirmed XI from the archive.

Run after each gameweek::

    python -m fpl_engine lineup-feed --gw 3

Results accumulate in data/lineup_feed_<season>.json with the row-level
records, so the pooled estimate is recomputed exactly every time.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import unicodedata
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import config
from .xpts import engine, minutes_model

try:                                   # the single definition of "a player"
    from .verify import PLAYER_POSITIONS
except ImportError:                    # pragma: no cover
    PLAYER_POSITIONS = ("GK", "DEF", "MID", "FWD")

ARCHIVE_DIR = os.path.join("data", "collected", "lineups")
BAND = (0.30, 0.70)
DEADLINE_LEAD = timedelta(minutes=90)      # FPL: 90 minutes before first kickoff
ABBR = {"NOT": "NFO"}                       # RotoWire -> FPL short_name
# E8b: the model's own accuracy in the band, and the value of resolving it
ACC_BAR, POINTS_AT_CEILING, TARGET_ROWS = 0.55, 89.0, 450

_TRANS = str.maketrans({
    "ø": "o", "Ø": "O", "ß": "ss", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
    "æ": "ae", "Æ": "AE", "œ": "oe", "ı": "i", "þ": "th", "ð": "d"})


def norm(s) -> str:
    """Accent-, case- and punctuation-insensitive name key."""
    s = str(s or "").translate(_TRANS)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())


# ------------------------------------------------------------ resolution --
def resolve(names_by_team: dict, players: pd.DataFrame):
    """RotoWire (team_abbr, name) -> FPL player_id.

    Order of trust: exact full name within the club, exact web name, unique
    surname, unique surname token, and finally a unique full-name match
    anywhere in the league (the local player table can lag a completed
    transfer). Anything else is UNRESOLVED and reported — never dropped
    silently, per the entity-resolution rule.
    """
    p = players.assign(n_full=players["full_name"].map(norm),
                       n_web=players["web_name"].map(norm))
    p["n_last"] = p["n_full"].str.split().str[-1]
    by_team = {t: d for t, d in p.groupby("short_name")}
    resolved, unresolved, mismatched = {}, [], []
    for team, names in names_by_team.items():
        tp = by_team.get(ABBR.get(team, team), p.iloc[0:0])
        for name in names:
            n = norm(name)
            last = n.split()[-1] if n else ""
            pid = None
            for cand in (tp[tp.n_full == n], tp[tp.n_web == n],
                         tp[tp.n_last == last] if last else tp.iloc[0:0],
                         tp[tp.n_full.str.split().map(lambda t: last in t)]
                         if last else tp.iloc[0:0]):
                if len(cand) == 1:
                    pid = int(cand.player_id.iloc[0])
                    break
            if pid is None:
                anywhere = p[p.n_full == n]
                if len(anywhere) == 1:
                    pid = int(anywhere.player_id.iloc[0])
                    mismatched.append((team, name,
                                       str(anywhere.short_name.iloc[0])))
            if pid is None:
                unresolved.append((team, name))
            else:
                resolved[(team, name)] = pid
    return resolved, unresolved, mismatched


# --------------------------------------------------------------- archive --
def load_archive(season: str, path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(ARCHIVE_DIR, f"{season}.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["observed_utc", "gw", "team_abbr", "side",
                                     "status", "position", "slot", "player",
                                     "rotowire_id", "kickoff_utc"])
    with open(path, encoding="utf-8", newline="") as fh:
        df = pd.DataFrame(list(csv.DictReader(fh)))
    df["gw"] = pd.to_numeric(df["gw"], errors="coerce")
    df["observed"] = pd.to_datetime(df["observed_utc"], utc=True,
                                    format="ISO8601")
    return df


def pre_deadline_forecasts(archive: pd.DataFrame, gw: int, cutoff) -> dict:
    """team_abbr -> (observed_utc, set of names): the LAST predicted XI seen
    strictly before ``cutoff``. Confirmed rows are excluded whatever their
    time — they are the answer, not the forecast."""
    cutoff = pd.Timestamp(cutoff).tz_convert("UTC") if pd.Timestamp(cutoff).tzinfo \
        else pd.Timestamp(cutoff, tz="UTC")
    d = archive[(archive["gw"] == gw) & (archive["status"] == "predicted")
                & (archive["observed"] < cutoff)]
    out = {}
    for team, g in d.groupby("team_abbr"):
        last = g["observed"].max()
        out[team] = (last.isoformat(),
                     set(g.loc[g["observed"] == last, "player"]))
    return out


def confirmed_xis(archive: pd.DataFrame, gw: int) -> dict:
    """team_abbr -> set of names from the latest CONFIRMED XI (ground truth)."""
    d = archive[(archive["gw"] == gw) & (archive["status"] == "confirmed")]
    out = {}
    for team, g in d.groupby("team_abbr"):
        last = g["observed"].max()
        out[team] = set(g.loc[g["observed"] == last, "player"])
    return out


# ---------------------------------------------------------------- model --
def availability_at(conn, season: str, when_utc: str) -> dict:
    """player_id -> availability factor as the change log stood at ``when``.

    Mirrors the live overlay: chance_next when published, else 1 for
    status 'a' and 0 for anything else. Empty when the acquirer's table is
    absent (then no overlay is applied and the report says so).
    """
    try:
        rows = conn.execute(
            "SELECT player_id, status, chance_next FROM ("
            "  SELECT *, ROW_NUMBER() OVER (PARTITION BY player_id "
            "    ORDER BY observed_utc DESC) rn "
            "  FROM acq_player_availability WHERE season=? AND observed_utc<=?"
            ") WHERE rn=1", (season, when_utc)).fetchall()
    except Exception:                                    # noqa: BLE001
        return {}
    out = {}
    for pid, status, chance in rows:
        if chance is not None:
            c = float(chance)
            out[int(pid)] = min(1.0, max(0.0, c / 100.0 if c > 1 else c))
        else:
            out[int(pid)] = 1.0 if status in (None, "a") else 0.0
    return out


def model_p_start(conn, season: str, gw: int, as_of: str, cutoff_utc: str,
                  minutes_bundle=None) -> pd.DataFrame:
    clf, meta = minutes_bundle or minutes_model.ensure(conn)
    mins = minutes_model.predict_gw(conn, season, as_of, clf, meta, gw=gw,
                                    use_availability=False)
    avail = availability_at(conn, season, cutoff_utc)
    mins["p_start_raw"] = mins["p_start"]
    mins["avail"] = mins["player_id"].map(avail).fillna(1.0)
    mins["p_start"] = mins["p_start_raw"] * mins["avail"]
    mins.attrs["availability_rows"] = len(avail)
    return mins[["player_id", "p_start_raw", "avail", "p_start"]]


# -------------------------------------------------------------- metrics --
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def implied_points(acc: float) -> float:
    """E8b: each point of band accuracy above the model's own 55% is worth
    89/45 points a season; below it the feed would cost points."""
    if acc is None or (isinstance(acc, float) and math.isnan(acc)):
        return float("nan")
    return (acc - ACC_BAR) / (1.0 - ACC_BAR) * POINTS_AT_CEILING


def block(d: pd.DataFrame) -> dict:
    n = int(len(d))
    if n == 0:
        return {"n": 0}
    started = d["started"].astype(int)
    feed_hit = (d["feed_xi"].astype(int) == started)
    model_hit = ((d["p_start"] >= 0.5).astype(int) == started)
    k = int(feed_hit.sum())
    lo, hi = wilson(k, n)
    disagree = d["feed_xi"].astype(int) != (d["p_start"] >= 0.5).astype(int)
    out = {
        "n": n,
        "base_rate_started": round(float(started.mean()), 3),
        "feed_accuracy": round(k / n, 3),
        "feed_accuracy_ci95": [round(lo, 3), round(hi, 3)],
        "model_accuracy": round(float(model_hit.mean()), 3),
        "model_brier": round(float(((d["p_start"] - started) ** 2).mean()), 4),
        "feed_brier": round(float(((d["feed_xi"].astype(float) - started) ** 2)
                                  .mean()), 4),
        "disagreements": int(disagree.sum()),
        "feed_right_when_disagree": (round(float(feed_hit[disagree].mean()), 3)
                                     if disagree.any() else None),
        "implied_points_per_season": round(implied_points(k / n), 1),
        "implied_points_ci95": [round(implied_points(lo), 1),
                                round(implied_points(hi), 1)],
    }
    return out


def xi_precision(d: pd.DataFrame) -> dict:
    """Share of the feed's eleven who started, against the model's own top
    eleven per club — the like-for-like XI question."""
    if d.empty:
        return {}
    feed = d[d["feed_xi"]]
    model_xi = d.sort_values("p_start", ascending=False).groupby("team").head(11)
    return {"feed_xi_started": round(float(feed["started"].mean()), 3)
            if len(feed) else None,
            "model_xi_started": round(float(model_xi["started"].mean()), 3),
            "n_teams": int(d["team"].nunique())}


# ------------------------------------------------------------------ run --
def _iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def score_gw(conn, season: str, gw: int, *, archive: pd.DataFrame | None = None,
             band=BAND, minutes_bundle=None) -> dict:
    archive = load_archive(season) if archive is None else archive
    as_of = engine.first_kickoff(conn, season, gw)
    if as_of is None:
        raise SystemExit(f"no fixtures for {season} GW{gw}")
    cutoff = pd.Timestamp(as_of, tz="UTC") - DEADLINE_LEAD
    cutoff_utc = _iso(cutoff)

    forecasts = pre_deadline_forecasts(archive, gw, cutoff)
    confirmed = confirmed_xis(archive, gw)

    players = pd.read_sql_query(
        "SELECT p.player_id, p.web_name, p.full_name, p.team_id, t.short_name "
        "FROM player p JOIN team t ON t.season=p.season AND t.team_id=p.team_id "
        "WHERE p.season=? AND p.position IN (%s)" % ",".join(
            "?" * len(PLAYER_POSITIONS)), conn,
        params=(season, *PLAYER_POSITIONS))
    names = {t: set(v[1]) | confirmed.get(t, set())
             for t, v in forecasts.items()}
    for t, v in confirmed.items():
        names.setdefault(t, set()).update(v)
    resolved, unresolved, mismatched = resolve(names, players)

    # truth: finished fixtures from player_gw, else the confirmed XI
    finished = pd.read_sql_query(
        "SELECT pg.player_id, MAX(COALESCE(pg.starts, 0)) starts "
        "FROM player_gw pg JOIN fixture f ON f.season=pg.season "
        "AND f.fixture_id=pg.fixture_id WHERE pg.season=? AND pg.gw=? "
        "AND f.finished=1 GROUP BY pg.player_id", conn, params=(season, gw))
    finished_teams = {int(r[0]) for r in conn.execute(
        "SELECT team_h FROM fixture WHERE season=? AND gw=? AND finished=1 "
        "UNION SELECT team_a FROM fixture WHERE season=? AND gw=? AND finished=1",
        (season, gw, season, gw))}
    started_db = dict(zip(finished["player_id"].astype(int),
                          finished["starts"].astype(int) > 0))

    mp = model_p_start(conn, season, gw, as_of, cutoff_utc, minutes_bundle)
    df = players.merge(mp, on="player_id", how="left")
    df["p_start"] = df["p_start"].fillna(0.0)
    abbr_of = {v: k for k, v in ABBR.items()}
    df["team"] = df["short_name"].map(lambda s: abbr_of.get(s, s))

    feed_ids = {t: {resolved[(t, n)] for n in v[1] if (t, n) in resolved}
                for t, v in forecasts.items()}
    conf_ids = {t: {resolved[(t, n)] for n in v if (t, n) in resolved}
                for t, v in confirmed.items()}
    team_id_of = dict(zip(df["team"], df["team_id"]))

    rows = []
    for team, g in df.groupby("team"):
        if team not in feed_ids:
            continue                              # no pre-deadline forecast
        tid = int(team_id_of[team])
        if tid in finished_teams:
            src = "player_gw"
        elif team in conf_ids and len(conf_ids[team]) >= 10:
            src = "confirmed_xi"
        else:
            continue                              # no truth yet
        for r in g.itertuples():
            started = (started_db.get(int(r.player_id), False) if src == "player_gw"
                       else int(r.player_id) in conf_ids[team])
            rows.append({"player_id": int(r.player_id), "team": team,
                         "web_name": r.web_name,
                         "p_start": round(float(r.p_start), 4),
                         "p_start_raw": round(float(r.p_start_raw), 4)
                         if pd.notna(r.p_start_raw) else None,
                         "feed_xi": int(r.player_id) in feed_ids[team],
                         "started": bool(started), "truth": src})
    rec = pd.DataFrame(rows)
    in_band = rec[(rec["p_start"] >= band[0]) & (rec["p_start"] <= band[1])] \
        if len(rec) else rec

    rep = {
        "season": season, "gw": gw, "as_of": as_of, "deadline_cutoff": cutoff_utc,
        "teams_forecast_pre_deadline": sorted(forecasts),
        "teams_scored": sorted(rec["team"].unique()) if len(rec) else [],
        "truth_sources": rec.groupby("truth").size().to_dict() if len(rec) else {},
        "availability_rows_at_deadline": int(mp.attrs.get("availability_rows", 0)),
        "unresolved_names": [f"{t}: {n}" for t, n in unresolved],
        "club_mismatches": [f"{t}: {n} (FPL has {c})" for t, n, c in mismatched],
        "band": list(band),
        "band_metrics": block(in_band),
        "all_metrics": block(rec),
        "xi_precision": xi_precision(rec),
        "records": rows,
    }
    return rep


def pooled(reports: dict, band=BAND) -> dict:
    recs = [r for rep in reports.values() for r in rep.get("records", [])]
    if not recs:
        return {"band_metrics": {"n": 0}, "all_metrics": {"n": 0}}
    rec = pd.DataFrame(recs)
    in_band = rec[(rec["p_start"] >= band[0]) & (rec["p_start"] <= band[1])]
    out = {"gameweeks": sorted(int(g) for g in reports),
           "band_metrics": block(in_band), "all_metrics": block(rec),
           "xi_precision": xi_precision(rec),
           "rows_needed_for_estimate": TARGET_ROWS,
           "priceable": bool(len(in_band) >= TARGET_ROWS)}
    return out


def save(season: str, rep: dict) -> str:
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"lineup_feed_{season}.json")
    doc = {"season": season, "gws": {}}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    doc["gws"][str(rep["gw"])] = rep
    doc["pooled"] = pooled(doc["gws"])
    doc["updated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=float)
    return path


def latest_scorable_gw(conn, season: str, archive: pd.DataFrame) -> int | None:
    gws = set()
    r = conn.execute("SELECT MAX(gw) FROM fixture WHERE season=? AND finished=1",
                     (season,)).fetchone()
    if r and r[0] is not None:
        gws.add(int(r[0]))
    if len(archive):
        conf = archive.loc[archive["status"] == "confirmed", "gw"].dropna()
        if len(conf):
            gws.add(int(conf.max()))
    return max(gws) if gws else None


def print_report(rep: dict, pool: dict | None = None) -> None:
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    b, a = rep["band_metrics"], rep["all_metrics"]
    print(f"\nLineup feed vs model — {rep['season']} GW{rep['gw']} "
          f"(deadline cutoff {rep['deadline_cutoff']})")
    print(f"  forecasts before the deadline: {len(rep['teams_forecast_pre_deadline'])} clubs; "
          f"scored: {len(rep['teams_scored'])} clubs, truth {rep['truth_sources']}; "
          f"availability rows at deadline: {rep['availability_rows_at_deadline']}")
    if rep["unresolved_names"]:
        print(f"  UNRESOLVED ({len(rep['unresolved_names'])}): "
              + "; ".join(rep["unresolved_names"]))
    if rep["club_mismatches"]:
        print(f"  club mismatches (resolved league-wide): "
              + "; ".join(rep["club_mismatches"]))

    def line(title, m):
        if not m.get("n"):
            print(f"  {title}: no rows")
            return
        print(f"  {title}: n={m['n']}  started={m['base_rate_started']:.2f}  "
              f"feed acc {m['feed_accuracy']:.3f} "
              f"[{m['feed_accuracy_ci95'][0]:.2f}, {m['feed_accuracy_ci95'][1]:.2f}]  "
              f"model acc {m['model_accuracy']:.3f}  "
              f"Brier feed {m['feed_brier']:.3f} / model {m['model_brier']:.3f}  "
              f"disagree {m['disagreements']} (feed right "
              f"{m['feed_right_when_disagree']})")
    line(f"ambiguous band {rep['band']}", b)
    line("all rows", a)
    x = rep.get("xi_precision") or {}
    if x:
        print(f"  XI precision: feed {x.get('feed_xi_started')}  "
              f"model top-11 {x.get('model_xi_started')}  ({x.get('n_teams')} clubs)")
    if b.get("n"):
        print(f"  implied value (E8b): {b['implied_points_per_season']:+.1f} pts/season "
              f"[{b['implied_points_ci95'][0]:+.1f}, {b['implied_points_ci95'][1]:+.1f}]")
    if pool:
        pb = pool["band_metrics"]
        print(f"\nPooled over GW{pool['gameweeks']}: band rows {pb.get('n', 0)} "
              f"of {pool['rows_needed_for_estimate']} needed"
              + (f"; feed acc {pb['feed_accuracy']:.3f} "
                 f"[{pb['feed_accuracy_ci95'][0]:.2f}, {pb['feed_accuracy_ci95'][1]:.2f}] "
                 f"vs model {pb['model_accuracy']:.3f}; implied "
                 f"{pb['implied_points_per_season']:+.1f} pts/season "
                 f"[{pb['implied_points_ci95'][0]:+.1f}, {pb['implied_points_ci95'][1]:+.1f}]"
                 if pb.get("n") else ""))
        print("  verdict:", "priceable — enough band rows to judge the feed"
              if pool["priceable"] else
              "NOT yet priceable — keep collecting; the CI above is the honest width")
