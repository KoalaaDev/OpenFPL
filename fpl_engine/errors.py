"""Model-error database: where the champion is wrong, persistently.

Phase 2.5 changes the question from "can another objective beat max-xP?"
(answered: no, at n=111) to "where is max-xP wrong, why, and could new
information have corrected it?" That needs errors to be *kept*, not
recomputed and forgotten: every gameweek's per-player prediction, the
realised outcome, and a deterministic classification of the miss, in one
table that error studies and the info-value framework can join against
(availability change log on player/time, decay snapshots on gameweek).

The classification is deliberately coarse and causal-shaped:

    did_not_play         expected on the pitch, played 0' — the minutes
                         model (or late news) is the cause, not the rates
    unexpected_appearance the reverse miss
    under_minutes        played, but far less than expected (hooked/benched
                         late) — partial minutes error
    haul_missed          played 60+, outscored the projection by 6+ — rate
                         underestimate or plain variance
    blank_despite_minutes played 60+, undershot by 4+ — rate overestimate
                         or variance
    ok                   within 3 points either way
    other                everything else

Points are extremely noisy per player-match, so the *counts and biases* by
position/team/price band are the signal; a single row never is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import scoring
from .xpts import engine as xpts_engine, minutes_model

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_error (
    season       TEXT NOT NULL,
    gw           INTEGER NOT NULL,
    player_id    INTEGER NOT NULL,
    prediction   REAL,
    actual       REAL,
    err          REAL,               -- actual - prediction
    e_min        REAL,
    minutes      REAL,
    min_err      REAL,               -- minutes - e_min
    p_play       REAL,
    p_60         REAL,
    position     TEXT,
    team_id      INTEGER,
    price        REAL,               -- tenths, at that gw where stored
    class        TEXT,
    observed_utc TEXT,
    PRIMARY KEY (season, gw, player_id)
);
CREATE INDEX IF NOT EXISTS idx_model_error_class
    ON model_error (season, class);
"""


def ensure_schema(conn) -> None:
    conn.executescript(SCHEMA)


def classify(row) -> str:
    e_min = row["e_min"] or 0.0
    minutes = row["minutes"] or 0.0
    err = row["err"] or 0.0
    if e_min >= 30 and minutes == 0:
        return "did_not_play"
    if e_min < 20 and minutes >= 60:
        return "unexpected_appearance"
    if minutes > 0 and (e_min - minutes) >= 45:
        return "under_minutes"
    if minutes >= 60 and err >= 6:
        return "haul_missed"
    if minutes >= 60 and err <= -4:
        return "blank_despite_minutes"
    if abs(err) < 3:
        return "ok"
    return "other"


def record_gw(conn, season: str, gw: int, preds: pd.DataFrame) -> int:
    """Store one gameweek's per-player errors. Idempotent on re-run.

    ``preds`` is the engine's frame (player_id, prediction, e_min, p_play,
    p_60, position, team_id). Actuals come from player_gw, summed over
    fixtures so double gameweeks are one row.
    """
    ensure_schema(conn)
    act = pd.read_sql_query(
        "SELECT pg.player_id, SUM(pg.total_points) actual, "
        "SUM(pg.minutes) minutes, MAX(pg.price) price FROM player_gw pg "
        "JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        "WHERE pg.season=? AND pg.gw=? AND p.position IN "
        "('GK','DEF','MID','FWD') GROUP BY pg.player_id",
        conn, params=(season, gw))
    if act.empty:
        return 0
    j = act.merge(preds, on="player_id", how="left")
    j["prediction"] = j["prediction"].fillna(0.0)
    j["e_min"] = j["e_min"].fillna(0.0)
    j["err"] = j["actual"] - j["prediction"]
    j["min_err"] = j["minutes"] - j["e_min"]
    j["cls"] = j.apply(classify, axis=1)   # "class" is renamed by itertuples
    from .http import utcnow_iso
    now = utcnow_iso()
    rows = [(season, gw, int(r.player_id), float(r.prediction),
             float(r.actual), float(r.err), float(r.e_min), float(r.minutes),
             float(r.min_err), float(r.p_play) if pd.notna(r.p_play) else None,
             float(r.p_60) if pd.notna(r.p_60) else None,
             r.position if pd.notna(r.position) else None,
             int(r.team_id) if pd.notna(r.team_id) else None,
             float(r.price) if pd.notna(r.price) else None,
             r.cls, now)
            for r in j.itertuples()]
    conn.executemany(
        "INSERT OR REPLACE INTO model_error VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def replay_season(conn, season: str, *, gws: list[int] | None = None,
                  progress=None) -> int:
    """Point-in-time replay: predict each gw with the bt-tagged minutes
    model (trained only on prior seasons) and record the errors.

    Availability stays off, as in every replay — the stored status is
    today's. For the live season, record after each gw with the live model
    instead (``errors --gw N``).
    """
    tag = f"bt{season}"
    if minutes_model.load(tag)[0] is None:
        from . import config
        train = [s for s in config.BACKFILL_SEASONS if s < season]
        minutes_model.train(conn, seasons=train, tag=tag)
    bundle = minutes_model.load(tag)
    rules = scoring.load_rules()
    all_gws = [int(r[0]) for r in conn.execute(
        "SELECT DISTINCT gw FROM player_gw WHERE season=? AND gw IS NOT NULL "
        "ORDER BY gw", (season,))]
    gws = [int(g) for g in gws] if gws else [g for g in all_gws if g >= 2]
    n = 0
    for g in gws:
        if progress:
            progress(f"GW{g}…")
        preds = xpts_engine.xpts_predict_gw(conn, season, g,
                                            use_availability=False,
                                            minutes_bundle=bundle, rules=rules)
        if not preds.empty:
            n += record_gw(conn, season, g, preds)
    return n


def analyse(conn, season: str | None = None) -> dict:
    """Systematic failure modes: bias and MAE cut the ways that could name
    a cause. Individual rows are noise; persistent group biases are not."""
    where = "WHERE season=?" if season else ""
    params = (season,) if season else ()
    df = pd.read_sql_query(f"SELECT * FROM model_error {where}", conn,
                           params=params)
    if df.empty:
        return {"rows": 0}
    df = df.rename(columns={"class": "cls"})
    df["price_band"] = pd.cut(df["price"].fillna(0),
                              [0, 45, 55, 65, 80, 100, 200],
                              labels=["<4.5", "4.5-5.5", "5.5-6.5",
                                      "6.5-8", "8-10", "10+"])
    played = df[df["minutes"] > 0]

    def _cut(frame, key):
        g = frame.groupby(key).agg(
            n=("err", "size"), bias=("err", "mean"),
            mae=("err", lambda s: s.abs().mean()),
            min_bias=("min_err", "mean"))
        return {str(k): {c: round(float(v), 3) for c, v in r.items()}
                for k, r in g.iterrows()}

    worst = df.reindex(df["err"].abs().sort_values(ascending=False).index)
    return {
        "rows": len(df), "gws": int(df["gw"].nunique()),
        "bias": round(float(df["err"].mean()), 4),
        "bias_played": round(float(played["err"].mean()), 4),
        "mae_played": round(float(played["err"].abs().mean()), 3),
        "minutes_mae": round(float((df["min_err"]).abs().mean()), 2),
        "classes": {k: int(v) for k, v in
                    df["cls"].value_counts().items()},
        "by_position": _cut(played, "position"),
        "by_price": _cut(played, "price_band"),
        "team_bias_extremes": dict(sorted(
            _cut(played, "team_id").items(),
            key=lambda kv: kv[1]["bias"])[:3] + sorted(
            _cut(played, "team_id").items(),
            key=lambda kv: -kv[1]["bias"])[:3]),
        "worst_misses": [
            {"gw": int(r.gw), "player_id": int(r.player_id),
             "pred": round(float(r.prediction), 2),
             "actual": float(r.actual), "class": r.cls}
            for r in worst.head(10).itertuples()],
    }
