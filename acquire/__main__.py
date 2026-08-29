"""Command line for the acquisition engine.

    python -m acquire pull --source fpl --season 2026-27
    python -m acquire backfill --source fpl
    python -m acquire validate
    python -m acquire status
    python -m acquire as-of --season 2026-27 --at 2026-08-28T17:00:00Z

It never touches the modelling engine. Its failures are its own.
"""
from __future__ import annotations

import argparse
import sys

from . import actions, storage, validate
from .sources import fpl_availability, fpl_managers

SOURCES = {"fpl": fpl_availability}


def _conn(args):
    conn = storage.connect(getattr(args, "db", None))
    storage.init(conn)
    return conn


def cmd_pull(args):
    names = list(SOURCES) if args.source == "all" else [args.source]
    with storage.connect(args.db) as conn:
        storage.init(conn)
        for name in names:
            mod = SOURCES.get(name)
            if mod is None:
                print(f"  unknown source: {name}")
                continue
            res = mod.pull(conn, season=args.season, dry_run=args.dry_run)
            print(f"  {name}: {res}")
        conn.commit()
    return 0


def cmd_backfill(args):
    with storage.connect(args.db) as conn:
        storage.init(conn)
        fpl_availability.register(conn)
        res = fpl_availability.backfill_from_snapshots(conn, season=args.season)
        conn.commit()
    print(f"  replayed archived bootstrap payloads: {res}")
    return 0


def cmd_actions_collect(args):
    res = actions.collect(args.out)
    print(f"  collected: {res}")
    return 0


def cmd_actions_import(args):
    with storage.connect(args.db) as conn:
        res = actions.import_collected(conn, args.out)
    print(f"  imported: {res}")
    return 0


def cmd_panel(args):
    """Grade managers on their PAST seasons and keep the proven ones."""
    with storage.connect(args.db) as conn:
        storage.init(conn)
        res = fpl_managers.build_panel(conn, pages=args.pages,
                                       progress=lambda m: print(m, flush=True))
        conn.commit()
        rows = conn.execute(
            "SELECT COUNT(*) n, SUM(is_panel) p, MIN(median_rank) b "
            "FROM acq_manager").fetchone()
    print(f"  scanned {res['scanned']} entries, graded {res['graded']}, "
          f"panel = {res['panel']}")
    print(f"  (panel = finished inside the top "
          f"{fpl_managers.ELITE_RANK:,} in at least "
          f"{fpl_managers.ELITE_SEASONS} past seasons)")
    print("  Selection uses PAST seasons only, so nothing from the season in")
    print("  progress enters it — which is what keeps the picks out of sample.")
    return 0


def cmd_picks(args):
    with storage.connect(args.db) as conn:
        storage.init(conn)
        if not fpl_managers.panel_entries(conn):
            print("  no panel yet - run `python -m acquire panel` first")
            return 1
        res = fpl_managers.pull_picks(conn, season=args.season, gw=args.gw,
                                      progress=lambda m: print(m, flush=True))
        conn.commit()
        own = fpl_managers.panel_ownership(conn, args.season, args.gw)
        names = {r["player_id"]: r["web_name"] for r in conn.execute(
            "SELECT player_id, web_name FROM player WHERE season=?",
            (args.season,))}
    print(f"  {res}")
    if own:
        size = own[0]["panel_size"] or 1
        print(f"\n  most-owned by the proven panel (n={size}) in GW{args.gw}:")
        for r in own[:args.top]:
            print(f"    {names.get(r['player_id'], r['player_id']):<16}"
                  f" owned {100*r['owned']/size:5.1f}%"
                  f"  started {100*r['started']/size:5.1f}%"
                  f"  captained {100*r['captained']/size:5.1f}%")
    return 0


def cmd_validate(args):
    with storage.connect(args.db) as conn:
        storage.init(conn)
        rep = validate.run(conn)
    print("Acquisition invariants:")
    print(validate.format_report(rep))
    print()
    if rep.ok():
        print(f"OK — {len(rep.warnings)} warning(s), no errors.")
        return 0
    print(f"FAILED — {len(rep.errors)} error(s); the affected dataset must not "
          "be published to the model.")
    return 1


def cmd_status(args):
    with storage.connect(args.db) as conn:
        storage.init(conn)
        print("Sources:")
        for r in conn.execute("SELECT source_id, name, enabled, request_delay, "
                              "last_success, last_failure FROM acq_source"):
            print(f"  {r['source_id']:<10} {r['name'][:44]:<46} "
                  f"enabled={r['enabled']} delay={r['request_delay']}s")
            print(f"  {'':<10} last success: {r['last_success']}")
        print("\nRaw documents:")
        for r in conn.execute("SELECT source_id, COUNT(*) n, MIN(retrieved_utc) lo,"
                              " MAX(retrieved_utc) hi FROM acq_raw_document "
                              "GROUP BY source_id"):
            print(f"  {r['source_id']:<10} {r['n']:>5} docs   {r['lo']} .. {r['hi']}")
        print("\nAvailability change log:")
        for r in conn.execute("SELECT season, COUNT(*) n, "
                              "COUNT(DISTINCT player_id) players, "
                              "MIN(observed_utc) lo, MAX(observed_utc) hi "
                              "FROM acq_player_availability GROUP BY season"):
            print(f"  {r['season']:<10} {r['n']:>6} observations over "
                  f"{r['players']:>4} players   {r['lo']} .. {r['hi']}")
    return 0


def cmd_as_of(args):
    """What the availability picture looked like at a past instant."""
    with storage.connect(args.db) as conn:
        storage.init(conn)
        rows = storage.availability_as_of(conn, args.season, args.at)
        names = {r["player_id"]: r["web_name"] for r in conn.execute(
            "SELECT player_id, web_name FROM player WHERE season=?",
            (args.season,))}
    flagged = [r for r in rows if r["status"] and r["status"] != "a"]
    print(f"As known at {args.at}: {len(rows)} players on record, "
          f"{len(flagged)} not fully available")
    for r in sorted(flagged, key=lambda x: (x["chance_next"] is None,
                                            x["chance_next"] or 0))[:args.top]:
        ch = "-" if r["chance_next"] is None else f"{r['chance_next']:.2f}"
        print(f"  {names.get(r['player_id'], r['player_id']):<16} "
              f"status={r['status']} chance={ch:<5} "
              f"since={str(r['source_published_utc'])[:19]}")
        if r["news"]:
            print(f"      {r['news'][:88]}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="acquire",
                                description="independent data acquisition")
    p.add_argument("--db", help="SQLite path (defaults to the engine's)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pull", help="fetch and record new observations")
    sp.add_argument("--source", default="all",
                    help="source id, or 'all' (" + ", ".join(SOURCES) + ")")
    sp.add_argument("--season", default="2026-27")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("backfill",
                        help="replay bootstrap payloads already archived")
    sp.add_argument("--source", default="fpl")
    sp.add_argument("--season", default="2026-27")
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("panel",
                        help="build the proven-manager panel (past seasons only)")
    sp.add_argument("--pages", type=int, default=20,
                    help="pages of the Overall table to enumerate (50 each)")
    sp.set_defaults(func=cmd_panel)

    sp = sub.add_parser("picks", help="snapshot the panel's squads for a gameweek")
    sp.add_argument("--season", default="2026-27")
    sp.add_argument("--gw", type=int, required=True)
    sp.add_argument("--top", type=int, default=12)
    sp.set_defaults(func=cmd_picks)

    sp = sub.add_parser("actions-collect",
                        help="scheduled file-based collection into "
                             "data/collected (used by GitHub Actions)")
    sp.add_argument("--out", default=actions.OUT_DIR)
    sp.set_defaults(func=cmd_actions_collect)

    sp = sub.add_parser("actions-import",
                        help="replay data/collected into the SQLite "
                             "change-log tables")
    sp.add_argument("--out", default=actions.OUT_DIR)
    sp.set_defaults(func=cmd_actions_import)

    sp = sub.add_parser("validate", help="run the invariants; non-zero on error")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("status", help="coverage and provenance summary")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("as-of", help="availability as it stood at an instant")
    sp.add_argument("--season", default="2026-27")
    sp.add_argument("--at", required=True, help="ISO instant, e.g. 2026-08-28T17:00:00Z")
    sp.add_argument("--top", type=int, default=15)
    sp.set_defaults(func=cmd_as_of)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
