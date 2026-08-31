"""High-level pipeline orchestration used by the CLI.

One place that wires ingest -> resolve -> build -> predict so both the CLI and
tests share the same flow.
"""
from __future__ import annotations

import pandas as pd

from . import config, db, features, predict as predict_mod
from .ingest import fpl_api, understat, vaastav
from . import progress


def pull(conn, *, season: str | None = None, use_cache: bool = False,
         history: bool = True, backfill: bool = True,
         with_understat: bool = False) -> dict:
    """Pull all free data into SQLite: FPL live (+history), vaastav backfill.

    ``use_cache`` applies ONLY to the static historical backfill (safe to cache
    across runs). Live FPL data (bootstrap, fixtures, current-season history) is
    ALWAYS fetched fresh so a scheduled run genuinely updates the data.
    """
    season = season or config.CURRENT_SEASON
    summary = {"season": season}
    summary["fpl"] = fpl_api.ingest_all(conn, season, use_cache=False,
                                        history=history)
    if backfill:
        summary["backfill"] = vaastav.ingest_seasons(conn, use_cache=use_cache)
    summary["birth_dates"] = backfill_birth_dates(conn)
    if with_understat and understat.available():
        summary["understat"] = _pull_understat(conn, season, use_cache=use_cache)
    else:
        summary["understat"] = "skipped/unavailable (FPL-only degradation)"
    summary["odds"] = _pull_odds(conn, season)
    return summary


def _pull_odds(conn, season: str) -> dict | str:
    """Best-effort odds pull (an enhancement, never a hard dependency).

    football-data.co.uk needs no key and covers played matches of the current
    season; The Odds API (``$ODDS_API_KEY``) adds the upcoming fixtures the
    engine actually predicts. Failures degrade to the pure team model."""
    import os
    from .ingest import odds as odds_ingest
    out = {}
    try:      # fresh fetch: the season CSV grows every week
        out["football_data"] = odds_ingest.ingest_football_data(
            conn, [season], use_cache=False)
    except Exception as e:  # noqa: BLE001 - odds must never break a pull
        out["football_data"] = f"skipped ({e})"
    if os.environ.get("ODDS_API_KEY"):
        try:
            out["odds_api"] = odds_ingest.ingest_odds_api(conn, season)
        except Exception as e:  # noqa: BLE001
            out["odds_api"] = f"skipped ({e})"
    else:
        out["odds_api"] = "skipped (ODDS_API_KEY not set)"
    # Prediction-market prices. Free and keyless, and they land in their own
    # table — they are shown next to the bookmaker's view, never fed to the
    # model, because every Polymarket EPL market is team level.
    try:
        from .ingest import polymarket
        out["polymarket"] = polymarket.ingest(conn, season)
    except Exception as e:  # noqa: BLE001 - a market feed must never break a pull
        out["polymarket"] = f"skipped ({e})"
    # Transfer rumours. Surfaced as suggestions only: FPL reclassifies a player
    # after a move completes, so between the deal and that update the engine
    # projects him onto a club he has left.
    try:
        from .ingest import transfermarkt
        out["rumours"] = transfermarkt.ingest(conn, season)
    except Exception as e:  # noqa: BLE001
        out["rumours"] = f"skipped ({e})"
    return out


def _needs_refresh(us_latest: str | None, fpl_latest: str | None) -> bool:
    """Re-fetch a player's Understat log only if FPL shows he has played a
    match newer than the latest Understat match we hold (or we hold none).
    Pre-season that is nobody; in-season it is the players who featured."""
    if not us_latest:
        return True
    if not fpl_latest:
        return False
    return str(fpl_latest)[:10] > str(us_latest)[:10]


def _resolve_backfill_seasons(conn, season: str, *, use_cache: bool) -> None:
    """Resolve Understat identities for the BACKFILLED seasons too.

    Resolving only the current season leaves ``player.understat_id`` NULL for
    every historical season, so no Understat feature is testable and the
    OpenFPL feature builder silently emits NaN for all of them — which is the
    state this repo was in. Each season costs one cached league-player
    request; the cross-season fill afterwards is free, because ``player.code``
    is stable, so an id found in any season is that player's id in all of them.
    """
    from .resolve import resolve_players, resolve_teams
    for s_ in config.BACKFILL_SEASONS:
        if s_ == season:
            continue
        try:
            resolve_teams(conn, s_)
            names, clubs = {}, {}
            for pl in understat.fetch_league_players(s_, use_cache=use_cache):
                names[str(pl.get("id"))] = pl.get("player_name")
                if pl.get("team_title"):
                    clubs[str(pl.get("id"))] = pl["team_title"]
            if names:
                r = resolve_players(conn, s_, names, understat_teams=clubs)
                progress.log(f"    {s_}: {len(r['resolved'])} resolved, "
                             f"{len(r['unresolved'])} unresolved")
        except Exception as e:  # noqa: BLE001 - a bad season must not abort
            progress.log(f"    {s_}: resolution skipped ({e})")
    conn.execute(
        "UPDATE player SET understat_id = ("
        "  SELECT p2.understat_id FROM player p2 "
        "  WHERE p2.code = player.code AND p2.understat_id IS NOT NULL LIMIT 1) "
        "WHERE understat_id IS NULL AND code IS NOT NULL")
    reconcile_understat_ids(conn)
    conn.commit()


def reconcile_understat_ids(conn) -> dict:
    """One player, one Understat id — and refuse rather than pick.

    The cross-season fill above only touches rows that are NULL, so when two
    seasons resolve the SAME man to DIFFERENT ids nothing reconciles them, and
    `player.code` ends up pointing at two footballers. That is not theoretical:
    FPL's Amad Diallo resolved to Understat 8127 ("Amad Diallo Traore",
    Manchester United) in two seasons and to 12200 ("Amadou Diallo", Newcastle
    United) in three — a different player entirely, whose shots and match roles
    were then attached to him for most of the database.

    There is no safe automatic tiebreak. Neither the number of seasons nor the
    volume of match data picks the right id here: the wrong one wins on both.
    So a code claiming more than one id is unset in every season and reported,
    the same rule the resolver already applies to two players claiming one id —
    handing a player another man's history is worse than having none. An
    `entity_override` row pins a case a human has actually checked.
    """
    bad = [int(r[0]) for r in conn.execute(
        "SELECT code FROM player WHERE understat_id IS NOT NULL "
        "AND code IS NOT NULL GROUP BY code "
        "HAVING COUNT(DISTINCT understat_id) > 1")]
    if bad:
        conn.executemany(
            "UPDATE player SET understat_id = NULL WHERE code = ?",
            [(c,) for c in bad])
        progress.log(f"    {len(bad)} player(s) claimed more than one Understat "
                     f"id across seasons; unset (ambiguous): {bad[:10]}")
    return {"ambiguous_codes": len(bad)}


def backfill_birth_dates(conn) -> dict:
    """Carry a player's date of birth back across seasons on the stable `code`.

    FPL began publishing `birth_date` in 2024-25 and it is absent before, so
    the training seasons would otherwise have no age at all. `code` survives
    the summer renumbering, so one row fills every other row for the same man.
    Only a player who left the league before 2024-25 stays blank — which is
    the gap Transfermarkt's squad pages cover.
    """
    conn.execute(
        "UPDATE player SET birth_date = ("
        "  SELECT q.birth_date FROM player q WHERE q.code = player.code "
        "  AND q.birth_date IS NOT NULL LIMIT 1) "
        "WHERE birth_date IS NULL AND code IS NOT NULL")
    conn.commit()
    have = conn.execute(
        "SELECT COUNT(*) FROM player WHERE birth_date IS NOT NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM player").fetchone()[0]
    return {"with_birth_date": int(have), "players": int(total)}


def _pull_understat(conn, season: str, *, use_cache: bool,
                    history_seasons: int = 1, player_limit: int | None = None,
                    refresh_all: bool = False, workers: int = 2) -> dict:
    """Understat pull: club stats for the current + ``history_seasons`` previous
    seasons (one request each), FPL<->Understat player resolution from the
    season player lists, then per-player match logs (all seasons in one call).

    Player logs are pulled *incrementally*: only players whose FPL match log
    has a newer match than their latest Understat row (see ``_needs_refresh``)
    unless ``refresh_all``. Fetches run on a small thread pool (the per-host
    throttle in ``http`` still spaces request starts); rows are written and
    committed from this thread every 25 players so progress persists.
    """
    from .resolve import resolve_players, resolve_teams
    resolve_teams(conn, season)
    year = understat.season_to_year(season)
    seasons = [understat.year_to_season(y)
               for y in range(year - history_seasons, year + 1)]
    team_rows = 0
    names: dict[str, str] = {}
    clubs: dict[str, str] = {}
    titles: set[str] = set()
    current_ids: set[str] = set()
    for s_ in seasons:
        live = s_ == season
        progress.step(f"Understat: club match stats {s_}…")
        data = understat.league_data(s_, use_cache=use_cache and not live)
        if data:
            titles |= {t.get("title") for t in (data.get("teams") or {}).values()
                       if t.get("title")}
            team_rows += understat.ingest_league_teams(conn, s_,
                                                       use_cache=use_cache and not live)
        for pl in understat.fetch_league_players(s_, use_cache=use_cache and not live):
            names[str(pl.get("id"))] = pl.get("player_name")
            if pl.get("team_title"):
                clubs[str(pl.get("id"))] = pl["team_title"]   # latest season wins
            if live:
                current_ids.add(str(pl.get("id")))
    res = resolve_players(conn, season, names, understat_teams=clubs)
    progress.step(f"Understat: {len(res['resolved'])} players resolved, "
                  f"{len(res['unresolved'])} unresolved, "
                  f"{len(res['ambiguous'])} ambiguous (features stay NaN for those).")
    _resolve_backfill_seasons(conn, season, use_cache=use_cache)
    rows_ = conn.execute(
        "SELECT p.understat_id AS uid, "
        "       (SELECT MAX(g.kickoff_utc) FROM player_gw g "
        "         WHERE g.player_code = p.code AND g.minutes > 0) AS fpl_latest, "
        "       (SELECT MAX(u.match_date) FROM understat_player_match u "
        "         WHERE u.understat_id = p.understat_id) AS us_latest "
        "FROM player p WHERE p.season=? AND p.understat_id IS NOT NULL "
        "ORDER BY p.player_id", (season,)).fetchall()
    all_uids = [r["uid"] for r in rows_]
    uids = [r["uid"] for r in rows_
            if refresh_all or _needs_refresh(r["us_latest"], r["fpl_latest"])]
    if player_limit is not None:
        uids = uids[:player_limit]
    total = len(uids)
    progress.step(f"Understat: {total}/{len(all_uids)} player logs need a refresh "
                  f"(~{max(1, round(total * 3 / 60))} min)…")
    n = 0
    if total:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch(uid):
            live = uid in current_ids
            return uid, understat.fetch_player_matches(
                uid, use_cache=use_cache and not live)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for i, (uid, matches) in enumerate(ex.map(_fetch, uids), 1):
                if matches:
                    n += db.upsert(conn, "understat_player_match",
                                   understat.player_rows_from_matches(
                                       uid, matches, epl_titles=titles or None))
                if i % 25 == 0 or i == total:
                    conn.commit()   # keep partial progress if interrupted
                    progress.log(f"    …{i}/{total} players ({n} match rows)")
    return {"seasons": seasons, "team_match_rows": team_rows,
            "player_logs_refreshed": total, "player_match_rows": n,
            "players_resolved": len(res["resolved"]),
            "players_unresolved": len(res["unresolved"]),
            "players_ambiguous": len(res["ambiguous"])}


def build(conn, gw: int, *, season: str | None = None, store: bool = True) -> pd.DataFrame:
    season = season or config.CURRENT_SEASON
    from .resolve import resolve_teams
    resolve_teams(conn, season)
    df = features.build_samples(conn, season, gw)
    if store:
        features.store_samples(conn, df, season, gw)
    return df


def xpts_weight() -> float | None:
    """Backtest-fitted xPts blend weight, or None when never fitted.

    Written by ``python -m fpl_engine backtest`` to models/xpts/blend.json;
    a weight of 0 means the backtest preferred pure OpenFPL.
    """
    import json
    import os
    path = os.path.join(config.MODELS_DIR, "xpts", "blend.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            w = float(json.load(fh).get("weight", 0))
        return w if w > 0 else None
    except Exception:
        return None


def resolve_blend(conn, season: str, blend):
    """Resolve a --blend argument into (retrained_models_or_None, weight).

    ``blend`` may be None/0 (pure OpenFPL), 'auto' (weight from season progress),
    or a float in [0,1]. Returns (None, 0.0) if no retrained models exist.
    """
    if blend in (None, 0, 0.0, "0"):
        return None, 0.0
    from . import train
    retrained = train.load_retrained()
    if retrained is None:
        return None, 0.0
    weight = (train.season_blend_weight(conn, season) if blend == "auto"
              else float(blend))
    return retrained, weight


def predict_gw(conn, gw: int, *, season: str | None = None, bundle=None,
               blend=None) -> pd.DataFrame:
    """End-to-end: build point-in-time samples for the gw and run OpenFPL."""
    season = season or config.CURRENT_SEASON
    df = build(conn, gw, season=season, store=True)
    retrained, weight = resolve_blend(conn, season, blend)
    preds = predict_mod.predict(df, bundle=bundle, retrained=retrained, blend=weight)
    return preds.sort_values("prediction", ascending=False).reset_index(drop=True)


def _require_data(conn, season: str) -> None:
    """Fail with a helpful message if the database has not been populated yet."""
    n_players = conn.execute(
        "SELECT COUNT(*) FROM player WHERE season=?", (season,)).fetchone()[0]
    n_fixtures = conn.execute(
        "SELECT COUNT(*) FROM fixture WHERE season=?", (season,)).fetchone()[0]
    if not n_players or not n_fixtures:
        raise SystemExit(
            f"No {season} data in the database yet.\n"
            f"Pull the free data first:\n"
            f"    python -m fpl_engine pull\n"
            f"then re-run the optimiser.")
    n_priced = conn.execute(
        "SELECT COUNT(*) FROM player WHERE season=? AND now_cost IS NOT NULL",
        (season,)).fetchone()[0]
    if not n_priced:
        raise SystemExit(
            f"{season} players have no prices (database predates a schema "
            f"update). Refresh it:\n    python -m fpl_engine pull")


def next_gw(conn, season: str) -> int:
    """The next unfinished gameweek that has scheduled fixtures."""
    row = conn.execute(
        "SELECT MIN(gw) g FROM fixture WHERE season=? AND finished=0 AND gw IS NOT NULL",
        (season,)).fetchone()
    if row and row["g"]:
        return int(row["g"])
    row = conn.execute(
        "SELECT MIN(gw) g FROM fixture WHERE season=? AND gw IS NOT NULL",
        (season,)).fetchone()
    return int(row["g"]) if row and row["g"] else 1


def optimise_squad(conn, *, entry_id: int, season: str | None = None,
                   horizon: int = 5, budget: float = 100.0, bundle=None,
                   decay: float = 0.85, max_transfers_per_gw: int = 3,
                   keep_per_position: int = 30, time_limit: int = 40,
                   use_cache: bool = False, blend=None) -> dict:
    """End-to-end squad optimisation for an FPL entry id.

    Fetches the manager's current squad (or None pre-season), projects points
    across the horizon, and runs the MILP — suggesting transfers/hits, or
    building a fresh squad from budget when no squad exists yet.
    """
    from . import manager
    from .optimise import milp, project

    from . import progress
    season = season or config.CURRENT_SEASON
    _require_data(conn, season)
    from .resolve import resolve_teams
    resolve_teams(conn, season)

    start = next_gw(conn, season)
    scheduled = [r["gw"] for r in conn.execute(
        "SELECT DISTINCT gw FROM fixture WHERE season=? AND gw>=? AND gw IS NOT NULL "
        "ORDER BY gw", (season, start))]
    gws = scheduled[:horizon] or [start]
    progress.step(f"Planning horizon GW{gws[0]}–GW{gws[-1]}")

    bundle = bundle or predict_mod.load_models()
    retrained, weight = resolve_blend(conn, season, blend)
    if weight > 0:
        progress.step(f"Blending retrained model (weight {weight:.2f})")
    progress.step(f"Projecting points for {len(gws)} gameweeks…")
    xw = xpts_weight()
    if xw:
        progress.step(f"Blending xPts component engine (weight {xw:.2f})")
    proj = project.horizon_projections(conn, season, gws, bundle=bundle,
                                       decay=decay, retrained=retrained,
                                       blend=weight, xpts_w=xw)

    progress.step(f"Fetching entry {entry_id}…")
    squad_state = manager.current_squad(entry_id, use_cache=use_cache)
    if squad_state is None:
        progress.step("No existing squad found — building a fresh squad from "
                      f"£{budget:.0f}m. Solving optimiser…")
        proj_p = project.prune(proj, keep_per_position=keep_per_position)
        plan = milp.build_from_scratch(
            proj_p, gws, budget=budget, decay=decay,
            max_transfers_per_gw=max_transfers_per_gw, time_limit=time_limit)
        mode = "build-from-scratch"
        state = {"bank": budget, "free_transfers": None}
    else:
        progress.step(f"Squad found ({squad_state['free_transfers']} FT, "
                      f"£{squad_state['bank']:.1f}m bank). Solving optimiser…")
        owned = {p["element"]: p["selling_price"] for p in squad_state["squad"]}
        proj = _ensure_players(conn, season, proj, owned, gws)
        proj_p = project.prune(proj, keep_per_position=keep_per_position,
                               must_keep=set(owned))
        plan = milp.optimise(
            proj_p, gws, initial=owned, bank=squad_state["bank"],
            free_transfers=squad_state["free_transfers"], budget=budget,
            decay=decay, max_transfers_per_gw=max_transfers_per_gw,
            time_limit=time_limit)
        mode = "optimise-transfers"
        state = {"bank": squad_state["bank"],
                 "free_transfers": squad_state["free_transfers"],
                 "manager": squad_state.get("name")}

    return {"mode": mode, "entry_id": entry_id, "gws": gws, "state": state,
            "plan": plan}


def _ensure_players(conn, season, proj, owned, gws):
    """Guarantee every owned player has a projection row (ep 0 if unprojectable)."""
    have = set(proj["player_id"])
    missing = [pid for pid in owned if pid not in have]
    if not missing:
        return proj
    rows = []
    for pid in missing:
        r = conn.execute(
            "SELECT p.player_id, p.full_name, p.position, p.team_id, p.now_cost, "
            "t.name team FROM player p LEFT JOIN team t "
            "ON p.season=t.season AND p.team_id=t.team_id "
            "WHERE p.season=? AND p.player_id=?", (season, pid)).fetchone()
        if not r:
            continue
        row = {"player_id": pid, "player": r["full_name"], "position": r["position"],
               "team_id": r["team_id"], "team": r["team"],
               "price": r["now_cost"] or 0.0, "available": 0.0, "ep_total": 0.0}
        for g in gws:
            row[f"ep_gw{g}"] = 0.0
        rows.append(row)
    return pd.concat([proj, pd.DataFrame(rows)], ignore_index=True) if rows else proj
