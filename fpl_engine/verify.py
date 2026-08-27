"""Data invariants that fail loudly.

This project has now shipped four separate identity defects — the Understat
resolution never run for backfilled seasons, `understat_id` blanked by every
bootstrap pull, one Understat id handed to two different players in all four
seasons (Gabriel / Gabriel Jesus), and price columns mixing £m with tenths.
Every one of them was silent: the pipeline kept running and produced NaNs or,
worse, plausible numbers. A model built on that is not worth improving.

So the invariants are checked explicitly and the check is a command. Each one
is either an ERROR (the data is wrong and downstream numbers cannot be
trusted) or a WARNING (a known, documented condition worth seeing).

    python -m fpl_engine verify          # exits non-zero on any error
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, db

# The four positions a *player* can hold. FPL also lists Assistant Managers
# (element_type 5) as selectable entries: they score points, have prices, and
# never play a minute. They are not players and must never enter a player
# model or a player-model metric — in 2024-25 they occupied up to 8 of the 20
# highest-scoring slots in a gameweek, which no player model can reach.
PLAYER_POSITIONS = ("GK", "DEF", "MID", "FWD")

PRICE_TENTHS_RANGE = (30.0, 200.0)     # player_gw.price is in tenths of £m
PRICE_M_RANGE = (3.0, 20.0)            # player.now_cost is in £m


@dataclass
class Finding:
    level: str          # "error" | "warning"
    check: str
    detail: str
    count: int = 0


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self):
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self):
        return [f for f in self.findings if f.level == "warning"]

    def ok(self) -> bool:
        return not self.errors


def _one(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else 0


def run(conn) -> Report:
    r = Report()

    def err(check, detail, count=0):
        r.findings.append(Finding("error", check, detail, count))

    def warn(check, detail, count=0):
        r.findings.append(Finding("warning", check, detail, count))

    # ---- identity: the mapping must be one-to-one, both ways ---------------
    n = _one(conn, "SELECT COUNT(*) FROM (SELECT season, understat_id FROM player "
                   "WHERE understat_id IS NOT NULL GROUP BY season, understat_id "
                   "HAVING COUNT(*) > 1)")
    if n:
        err("identity.one_understat_per_player",
            "an Understat id is shared by two players in the same season — one "
            "of them is getting the other's history", n)
    n = _one(conn, "SELECT COUNT(*) FROM (SELECT code FROM player "
                   "WHERE understat_id IS NOT NULL AND code IS NOT NULL "
                   "GROUP BY code HAVING COUNT(DISTINCT understat_id) > 1)")
    if n:
        err("identity.stable_across_seasons",
            "one FPL player maps to different Understat ids in different "
            "seasons; player.code is stable, so this is a resolution error", n)
    n = _one(conn, "SELECT COUNT(*) FROM (SELECT understat_id FROM player "
                   "WHERE understat_id IS NOT NULL AND code IS NOT NULL "
                   "GROUP BY understat_id HAVING COUNT(DISTINCT code) > 1)")
    if n:
        err("identity.one_player_per_understat",
            "one Understat id is claimed by two different FPL players", n)
    n = _one(conn, "SELECT COUNT(*) FROM player WHERE code IS NULL")
    if n:
        err("identity.code_present",
            "player rows without the stable cross-season code", n)

    # ---- referential integrity --------------------------------------------
    n = _one(conn, "SELECT COUNT(*) FROM player_gw pg LEFT JOIN player p "
                   "ON p.season=pg.season AND p.player_id=pg.player_id "
                   "WHERE p.player_id IS NULL")
    if n:
        err("refs.player_gw_has_player",
            "player_gw rows with no matching player row", n)
    n = _one(conn, "SELECT COUNT(*) FROM player_gw WHERE player_code IS NULL")
    if n:
        warn("refs.player_gw_code",
             "player_gw rows without player_code (cross-season joins skip them)", n)
    n = _one(conn, "SELECT COUNT(*) FROM understat_shot s LEFT JOIN player p "
                   "ON p.understat_id = s.understat_id WHERE p.understat_id IS NULL")
    if n:
        warn("refs.shots_resolve",
             "shots whose shooter is not resolved to any FPL player", n)

    # ---- non-players -------------------------------------------------------
    placeholders = ",".join("?" * len(PLAYER_POSITIONS))
    n = _one(conn, f"SELECT COUNT(*) FROM player WHERE position IS NULL "
                   f"OR position NOT IN ({placeholders})", PLAYER_POSITIONS)
    if n:
        warn("nonplayers.present",
             "entries that are not players (FPL Assistant Managers). They score "
             "points and never play; every player model and player metric must "
             "exclude them — see verify.PLAYER_POSITIONS", n)

    # ---- units -------------------------------------------------------------
    lo, hi = PRICE_TENTHS_RANGE
    n = _one(conn, "SELECT COUNT(*) FROM player_gw pg JOIN player p "
                   "ON p.season=pg.season AND p.player_id=pg.player_id "
                   f"WHERE pg.price IS NOT NULL AND p.position IN ({placeholders}) "
                   "AND (pg.price < ? OR pg.price > ?)",
             tuple(PLAYER_POSITIONS) + (lo, hi))
    if n:
        err("units.player_gw_price_is_tenths",
            f"player_gw.price outside [{lo}, {hi}] tenths of £m — the column "
            "mixes units when something writes £m into it", n)
    lo, hi = PRICE_M_RANGE
    n = _one(conn, f"SELECT COUNT(*) FROM player WHERE now_cost IS NOT NULL "
                   f"AND position IN ({placeholders}) "
                   "AND (now_cost < ? OR now_cost > ?)",
             tuple(PLAYER_POSITIONS) + (lo, hi))
    if n:
        err("units.now_cost_is_millions",
            f"player.now_cost outside [{lo}, {hi}] £m", n)

    # ---- value sanity ------------------------------------------------------
    n = _one(conn, "SELECT COUNT(*) FROM player_gw WHERE minutes < 0 OR minutes > 120")
    if n:
        err("values.minutes", "minutes outside [0, 120]", n)
    n = _one(conn, "SELECT COUNT(*) FROM team_match "
                   "WHERE goals_for < 0 OR goals_for > 15")
    if n:
        err("values.goals", "team goals outside [0, 15]", n)

    # ---- point-in-time -----------------------------------------------------
    n = _one(conn, "SELECT COUNT(*) FROM player_gw "
                   "WHERE kickoff_utc IS NULL AND minutes > 0")
    if n:
        err("pit.kickoff_present",
            "played matches with no kickoff time — the point-in-time filter "
            "silently drops or admits them", n)
    n = _one(conn, "SELECT COUNT(*) FROM player_gw "
                   "WHERE kickoff_utc > strftime('%Y-%m-%dT%H:%M:%SZ','now') "
                   "AND minutes > 0")
    if n:
        err("pit.no_future_outcomes",
            "matches in the future that already carry minutes — a result has "
            "leaked backwards", n)
    n = _one(conn, "SELECT COUNT(*) FROM understat_shot WHERE substr(match_date,1,4) "
                   "NOT IN (substr(season,1,4), "
                   "CAST(CAST(substr(season,1,4) AS INT)+1 AS TEXT))")
    if n:
        err("pit.shot_season_window",
            "shots dated outside the season they are filed under", n)

    # ---- coverage (informational, but a collapse means something broke) ----
    for season in list(config.BACKFILL_SEASONS) + [config.CURRENT_SEASON]:
        tot = _one(conn, "SELECT COUNT(*) FROM player WHERE season=?", (season,))
        if not tot:
            continue
        res = _one(conn, "SELECT COUNT(*) FROM player WHERE season=? "
                         "AND understat_id IS NOT NULL", (season,))
        if res / tot < 0.5:
            warn("coverage.understat",
                 f"{season}: only {res}/{tot} players resolved to Understat "
                 "(features fall back to FPL's own expected stats)", tot - res)
    return r


def format_report(r: Report) -> str:
    lines = []
    for f in r.findings:
        tag = "ERROR  " if f.level == "error" else "warning"
        n = f" [{f.count}]" if f.count else ""
        lines.append(f"  {tag} {f.check}{n}\n           {f.detail}")
    if not lines:
        return "  all invariants hold"
    return "\n".join(lines)
