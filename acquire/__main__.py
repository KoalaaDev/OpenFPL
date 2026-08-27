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

from . import storage, validate
from .sources import fpl_availability

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
