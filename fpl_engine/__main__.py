"""Command-line interface for the fpl_engine data pipeline.

Examples
--------
    python -m fpl_engine init-db
    python -m fpl_engine pull                 # FPL live + vaastav backfill -> SQLite
    python -m fpl_engine backfill             # historical seasons only
    python -m fpl_engine build --gw 1         # build point-in-time samples
    python -m fpl_engine predict --gw 1       # end-to-end OpenFPL predictions
    python -m fpl_engine run --gw 1           # pull + build + predict in one go
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")  # silence sklearn version-mismatch warnings

from . import config, db
from .ingest import vaastav


def _print_df(df, n=40):
    import pandas as pd
    with pd.option_context("display.max_rows", n, "display.width", 200):
        print(df.head(n).to_string(index=False))


def cmd_init_db(args):
    db.init_db(args.db)
    print(f"Initialised SQLite database at {args.db or config.DB_PATH}")


def cmd_pull(args):
    from .pipeline import pull
    with db.session(args.db) as conn:
        db.init_db(args.db)
        summary = pull(conn, season=args.season, use_cache=args.cache,
                       history=not args.no_history, backfill=not args.no_backfill,
                       with_understat=args.understat)
    print("Pull complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_backfill(args):
    with db.session(args.db) as conn:
        db.init_db(args.db)
        out = vaastav.ingest_seasons(conn, args.seasons or None, use_cache=args.cache)
    for row in out:
        print(" ", row)


def cmd_build(args):
    from .pipeline import build
    db.init_db(args.db)
    with db.session(args.db) as conn:
        df = build(conn, args.gw, season=args.season, store=not args.no_store)
    print(f"Built {len(df)} samples for {args.season or config.CURRENT_SEASON} "
          f"GW{args.gw}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


def cmd_train(args):
    from . import train
    db.init_db(args.db)
    with db.session(args.db) as conn:
        meta = train.train(conn, seasons=args.seasons or None,
                           valid_season=args.valid_season, gw_step=args.gw_step,
                           device=args.device)
    print("Retraining complete. Forward-in-time validation "
          f"(held-out {meta['valid_season']}), device={meta['device']}:")
    for pos, m in meta["metrics"].items():
        print(f"  {pos}: {m}")
    print(f"\nSaved to {train.RETRAINED_DIR}. Use it via --blend, e.g. "
          f"`predict --gw 1 --blend auto`.")


def cmd_verify(args):
    from . import verify
    db.init_db(args.db)
    with db.session(args.db) as conn:
        rep = verify.run(conn)
    print("Data invariants:")
    print(verify.format_report(rep))
    print()
    if rep.ok():
        print(f"OK — {len(rep.warnings)} warning(s), no errors.")
        return 0
    print(f"FAILED — {len(rep.errors)} error(s). Downstream numbers cannot be "
          "trusted until these are fixed.")
    return 1


def cmd_prices(args):
    from . import price_model
    db.init_db(args.db)
    with db.session(args.db) as conn:
        bundle = price_model.ensure(conn)
        preds = price_model.predict(conn, args.season or config.CURRENT_SEASON,
                                    gw=args.gw, bundle=bundle)
        names = {r["player_id"]: r["web_name"] for r in conn.execute(
            "SELECT player_id, web_name FROM player WHERE season=?",
            (args.season or config.CURRENT_SEASON,))}
    if preds.empty:
        print("No price history yet for this season — run `pull` first.")
        return 0
    hold = bundle[1].get("holdout") or {}
    if hold:
        print(f"(held out on {hold['season']}: {hold['p_rise_given_top10']:.0%} "
              f"of the top-10 ranked risers actually rose, against a "
              f"{hold['base_rate']:.1%} base rate)")
    preds = preds.assign(player=preds["player_id"].map(names))
    gw = int(preds["gw"].iloc[0])
    left = max(0, 38 - gw)
    preds["pts_value"] = [price_model.points_value(v, left)
                          for v in preds["e_delta"]]
    print(f"\nPrice moves out of GW{gw} — most likely RISERS:")
    _print_df(preds.head(args.top)[["player", "price_m", "p_rise", "p_fall",
                                    "e_delta", "pts_value"]], args.top)
    print("\nMost likely FALLERS:")
    _print_df(preds.tail(args.top).iloc[::-1][["player", "price_m", "p_rise",
                                               "p_fall", "e_delta",
                                               "pts_value"]], args.top)
    print()
    print("pts_value converts the expected move into points at the measured rate:")
    print(f"  £1m is worth {price_model.POINTS_PER_MILLION_PER_GW} pts per "
          f"gameweek, halved by FPL's sell-on rule, over {left} gameweeks left.")
    print("It is a tie-breaker between transfers you already rate equally, not a "
          "term\nthat should overturn an expected-points ranking.")
    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")
    return 0


def cmd_simulate(args):
    from .xpts import simulate
    from .pipeline import next_gw
    db.init_db(args.db)
    season = args.season or config.CURRENT_SEASON
    with db.session(args.db) as conn:
        gw = args.gw if args.gw is not None else next_gw(conn, season)
        out = simulate.simulate_gw(conn, season, gw, n_sims=args.sims)
        s = simulate.summarise(out)
        names = {r["player_id"]: r["web_name"] for r in conn.execute(
            "SELECT player_id, web_name FROM player WHERE season=?", (season,))}
        if not s.empty:
            xi = list(s.nlargest(11, "mean")["player_id"])
            port = simulate.portfolio(out, xi)
    if s.empty:
        print(f"No fixtures to simulate for {season} GW{gw}.")
        return 0
    s = s[s["mean"] > 0].assign(player=s["player_id"].map(names))
    print(f"{season} GW{gw}: {args.sims} simulated matches")
    print()
    print("Ranking here is NOT better than `predict` — the distribution is "
          "for risk, not for choosing.")
    print()
    _print_df(s.nlargest(args.top, "mean")[
        ["player", "mean", "sd", "floor", "median", "ceiling", "p_blank",
         "p_haul"]].round(3), args.top)
    if port:
        print(f"\nTop-11 by mean, as a portfolio: mean {port['mean']:.1f}, "
              f"floor {port['floor']:.0f}, ceiling {port['ceiling']:.0f}")
        print(f"  sd {port['sd']:.2f} — treating the players as independent "
              f"would say {port['independent_sd']:.2f} "
              f"({100 * (port['sd'] / port['independent_sd'] - 1):+.0f}%)")
    return 0


def cmd_compare_backtests(args):
    from .backtest import compare
    res = compare(args.before, args.after, model=args.model)
    print(f"{res['model']}: {res['gws']} paired gameweeks")
    print()
    print(f"{'metric':<18}{'before':>10}{'after':>10}{'delta':>10}{'t':>8}{'p':>9}")
    for k, m in res["metrics"].items():
        print(f"{k:<18}{m['before']:>10.4f}{m['after']:>10.4f}"
              f"{m['delta']:>+10.4f}{m['t']:>8.2f}{m['p']:>9.4f}")
    print()
    print("p is a paired t-test over gameweeks; treat p > 0.05 as unproven.")


def cmd_predict(args):
    from .pipeline import next_gw, predict_gw
    db.init_db(args.db)
    with db.session(args.db) as conn:
        gw = args.gw if args.gw is not None else next_gw(
            conn, args.season or config.CURRENT_SEASON)
        preds = predict_gw(conn, gw, season=args.season, blend=args.blend)
    _print_df(preds, args.top)
    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


def cmd_optimise(args):
    from .pipeline import optimise_squad
    from .manager import DEFAULT_ENTRY
    entry = args.entry if args.entry is not None else DEFAULT_ENTRY
    db.init_db(args.db)
    with db.session(args.db) as conn:
        result = optimise_squad(
            conn, entry_id=entry, season=args.season, horizon=args.horizon,
            budget=args.budget, decay=args.decay,
            max_transfers_per_gw=args.max_transfers, time_limit=args.time_limit,
            use_cache=args.cache, blend=args.blend)
    plan = result["plan"]
    print(f"Entry {result['entry_id']} | mode: {result['mode']} | "
          f"horizon GW{result['gws'][0]}–{result['gws'][-1]}")
    print(f"State: {result['state']}")
    print("=" * 64)
    print(plan.summary())
    print("=" * 64)
    first = plan.per_gw[0]
    print(f"\nRecommended squad for GW{first['gw']} "
          f"(captain: {first['captain']}):")
    for pos in ("GK", "DEF", "MID", "FWD"):
        line = [f"{n}{'*' if n in first['xi'] else ''} ({e})"
                for n, pp, e in first["squad"] if pp == pos]
        print(f"  {pos}: " + ", ".join(line))
    print("  (* = starting XI)")


def cmd_backtest(args):
    from . import backtest
    db.init_db(args.db)
    with db.session(args.db) as conn:
        report = backtest.run(conn, args.backtest_season, gws=args.gws or None,
                              openfpl_every=args.openfpl_every,
                              retrain_minutes=args.retrain_minutes,
                              with_openfpl=not args.no_openfpl)
    print(f"\nBacktest {report['season']} "
          f"(minutes-model holdout acc {report['minutes_holdout_accuracy']}):")
    cols = ["spearman", "spearman_played", "p_at_20", "top11", "top30",
            "captain", "captain_best", "rmse", "gws"]
    print(f"{'model':<10}" + "".join(f"{c:>16}" for c in cols))
    for name, m in sorted(report["summary"].items()):
        print(f"{name:<10}" + "".join(f"{m.get(c, float('nan')):>16}" for c in cols))
    if report.get("blend"):
        b = report["blend"]
        print(f"\nBlend fit: w(xpts)={b['weight']}  (judged on the held-out "
              "second half of the season)")
        for key in ("eval_spearman_played", "eval_top30"):
            if key in b:
                vals = "  ".join(f"{k}={v:.4f}" for k, v in b[key].items())
                print(f"  {key[len('eval_'):]:<16} {vals}")
        print("Saved to models/xpts/blend.json (used automatically by predict/web).")


def cmd_decay(args):
    from . import decay
    db.init_db(args.db)
    with db.session(args.db) as conn:
        res = decay.analyse(conn, args.season)
    if not res["buckets"]:
        print(res["note"])
        return 0
    print(f"deadline decay, {res['season']} (availability field vs realised "
          "appearances):")
    hdr = (f"{'bucket':<14}{'n':>7}{'gws':>5}{'brier':>8}{'flagged brier':>15}"
           f"{'flagged %':>11}")
    print(hdr)
    for b in res["buckets"]:
        fb = b["brier_play_flagged"]
        print(f"{b['bucket']:<14}{b['n']:>7}{b['gws']:>5}"
              f"{b['brier_play']:>8.4f}"
              f"{(f'{fb:.4f}' if fb is not None else '—'):>15}"
              f"{b['flagged_share']:>11.3f}")
    return 0


def cmd_sensitivity(args):
    import pandas as pd

    from . import config as cfg
    from .rank_backtest import template_squad
    from .xpts import sensitivity, simulate
    db.init_db(args.db)
    with db.session(args.db) as conn:
        season = args.season or cfg.CURRENT_SEASON
        sim = simulate.simulate_gw(conn, season, args.gw, n_sims=args.sims)
        if not len(sim["players"]):
            print("no fixtures/simulation for that gameweek")
            return 1
        players = pd.read_sql_query(
            "SELECT player_id, position, team_id FROM player WHERE season=?",
            conn, params=(season,))
        players = pd.DataFrame({"player_id": sim["players"]}).merge(
            players, on="player_id", how="left")
        if args.entry:
            from .manager import fetch_picks
            gw_prev = args.gw - 1
            picks = fetch_picks(args.entry, gw_prev)
            if not picks:
                print(f"no picks for entry {args.entry} GW{gw_prev}")
                return 1
            squad = [int(p["element"]) for p in picks["picks"]]
        else:
            own = pd.read_sql_query(
                "SELECT player_id, MAX(selected) selected FROM player_gw "
                "WHERE season=? GROUP BY player_id", conn, params=(season,))
            meta = players.merge(own, on="player_id", how="left").fillna(
                {"selected": 0.0})
            squad = template_squad(meta[meta["player_id"].isin(
                set(sim["players"]))])
        res = sensitivity.analyse_squad(sim["points"], players, squad)
        names = dict(conn.execute(
            "SELECT player_id, web_name FROM player WHERE season=?",
            (season,)))
    nm = lambda p: names.get(p, p)  # noqa: E731
    print(f"GW{args.gw} decision sensitivity "
          f"({'entry ' + str(args.entry) if args.entry else 'template squad'}):")
    print(f"  captain: {nm(res['captain'])}  margin {res['captain_margin']} "
          f"xP over {nm(res['vice'])}  | stable in "
          f"{res['captain_stability']:.0%} of bootstraps")
    print(f"  XI stable in {res['xi_stability']:.0%} of bootstraps")
    print("  tightest XI calls (starter vs best bench alternative):")
    for s in res["tightest_swaps"]:
        print(f"    {nm(s['out'])} over {nm(s['in'])}: margin {s['margin']}")
    print(f"  verdict: {'FRAGILE — worth watching team news' if res['fragile'] else 'robust'}")


def cmd_errors(args):
    from . import errors
    db.init_db(args.db)
    with db.session(args.db) as conn:
        if args.replay:
            n = errors.replay_season(conn, args.season, gws=args.gws or None,
                                     progress=lambda m: print(m, flush=True))
            print(f"recorded {n} error rows for {args.season}")
        elif args.gw is not None:
            from . import scoring as sc
            from .xpts import engine as xe
            preds = xe.xpts_predict_gw(conn, args.season, args.gw,
                                       rules=sc.load_rules())
            n = errors.record_gw(conn, args.season, args.gw, preds)
            print(f"recorded {n} error rows for GW{args.gw}")
        res = errors.analyse(conn, args.season)
    if res.get("rows"):
        print(f"\n{res['rows']} rows over {res['gws']} gws | bias "
              f"{res['bias']:+.3f} (played {res['bias_played']:+.3f}) | "
              f"MAE played {res['mae_played']} | minutes MAE "
              f"{res['minutes_mae']}")
        print(f"classes: {res['classes']}")
        print("by position:", res["by_position"])
        print("by price:", res["by_price"])
        print("team bias extremes:", res["team_bias_extremes"])
        print("worst misses:", res["worst_misses"][:5])
    else:
        print("no error rows recorded yet")


def cmd_rank_backtest(args):
    from . import rank_backtest
    db.init_db(args.db)
    with db.session(args.db) as conn:
        report = rank_backtest.run(conn, args.backtest_season,
                                   gws=args.gws or None, n_sims=args.sims)
    print(f"\nRank backtest {report['season']} (template squad held fixed; "
          "realised points, real autosubs):")
    print(f"{'arm':<12}{'pts/gw':>10}{'delta vs field':>16}{'gws':>6}")
    for arm, m in report["summary"].items():
        print(f"{arm:<12}{m['pts']:>10.2f}{m['delta']:>16.2f}{m['gws']:>6}")
    print(f"\nSaved to data/rank_backtest_{report['season']}.json — pool "
          "seasons with `compare-rank-backtests` before believing anything.")


def cmd_compare_rank_backtests(args):
    from .rank_backtest import compare
    res = compare(args.paths, baseline=args.baseline)
    print(f"baseline: {res['baseline']}   "
          f"({res['n_comparisons']} comparisons; Bonferroni alpha "
          f"{res['bonferroni_alpha']})")
    hdr = (f"{'arm':<14}{'gws':>5}{'same cap':>9}{'beat fld':>9}"
           f"{'pts arm':>9}{'d pts':>8}{'ci95':>17}{'p':>8}  per-season")
    print(hdr)
    for arm, m in res["arms"].items():
        seasons = "  ".join(f"{s[2:]}:{v:+.2f}"
                            for s, v in m["pts"]["per_season"].items())
        ci = f"[{m['pts']['ci95'][0]:+.2f},{m['pts']['ci95'][1]:+.2f}]"
        print(f"{arm:<14}{m['gws']:>5}{m['same_captain']:>9.2f}"
              f"{m['beat_field']:>9.2f}"
              f"{m['pts']['arm']:>9.2f}{m['pts']['delta']:>+8.2f}"
              f"{ci:>17}{m['pts']['p']:>8.4f}  {seasons}")
    print(f"\nbaseline beat-the-field share: "
          f"{next(iter(res['arms'].values()))['beat_field_baseline']:.2f}"
          f"   |   p is a paired t over gameweeks; with "
          f"{res['n_comparisons']} arms, only p < "
          f"{res['bonferroni_alpha']} survives multiple testing.")


def cmd_run(args):
    from .pipeline import pull, predict_gw
    with db.session(args.db) as conn:
        db.init_db(args.db)
        pull(conn, season=args.season, use_cache=args.cache,
             history=not args.no_history, backfill=not args.no_backfill,
             with_understat=args.understat)
        preds = predict_gw(conn, args.gw, season=args.season)
    _print_df(preds, args.top)
    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fpl_engine",
                                description="Free, automatic FPL data pipeline -> SQLite -> OpenFPL")
    p.add_argument("--db", help="SQLite path (default data/fpl.sqlite or $FPL_DB_PATH)")
    p.add_argument("--season", help=f"season (default {config.CURRENT_SEASON})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init-db", help="create the SQLite schema")
    sp.set_defaults(func=cmd_init_db)

    sp = sub.add_parser("pull", help="pull FPL live + historical into SQLite")
    sp.add_argument("--cache", action="store_true",
                    help="cache the static historical backfill (live data stays fresh)")
    sp.add_argument("--no-history", action="store_true", help="skip per-player history")
    sp.add_argument("--no-backfill", action="store_true", help="skip vaastav backfill")
    sp.add_argument("--understat", action="store_true", help="also pull Understat if available")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("backfill", help="historical seasons only (vaastav)")
    sp.add_argument("--cache", action="store_true")
    sp.add_argument("--seasons", nargs="*", help="e.g. 2023-24 2024-25")
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("build", help="build point-in-time samples for a gw")
    sp.add_argument("--gw", type=int, required=True)
    sp.add_argument("--no-store", action="store_true")
    sp.add_argument("--out", help="write samples CSV")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("train", help="retrain per-position models on the feature store (GPU-aware)")
    sp.add_argument("--seasons", nargs="*", help="seasons to train on (default backfill set)")
    sp.add_argument("--valid-season", help="held-out season for forward validation")
    sp.add_argument("--gw-step", type=int, default=1,
                    help="subsample gameweeks for a faster run (e.g. 2)")
    sp.add_argument("--device", help="cuda | cpu (default: auto-detect)")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="end-to-end predictions for a gw")
    sp.add_argument("--gw", type=int, default=None,
                    help="gameweek (default: next scheduled)")
    sp.add_argument("--top", type=int, default=40, help="rows to print")
    sp.add_argument("--out", help="write predictions CSV")
    sp.add_argument("--blend", default=None,
                    help="blend retrained model: 'auto', or a weight 0..1 (needs `train` first)")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("optimise", help="suggest transfers / build a squad for an FPL entry")
    sp.add_argument("--entry", type=int, default=None,
                    help="FPL entry (squad) id; defaults to 883566")
    sp.add_argument("--horizon", type=int, default=5, help="gameweeks to plan over")
    sp.add_argument("--budget", type=float, default=100.0)
    sp.add_argument("--decay", type=float, default=0.85)
    sp.add_argument("--max-transfers", type=int, default=3,
                    help="cap transfers per gameweek (bounds the search)")
    sp.add_argument("--time-limit", type=int, default=40, help="solver seconds")
    sp.add_argument("--cache", action="store_true")
    sp.add_argument("--blend", default=None,
                    help="blend retrained model: 'auto', or a weight 0..1 (needs `train` first)")
    sp.set_defaults(func=cmd_optimise)

    sp = sub.add_parser("backtest", help="replay past gameweeks: xpts vs OpenFPL vs baselines")
    sp.add_argument("--backtest-season", default="2025-26",
                    help="season to replay (default 2025-26)")
    sp.add_argument("--gws", nargs="*", type=int, help="specific gameweeks only")
    sp.add_argument("--openfpl-every", type=int, default=4,
                    help="run the (slow) OpenFPL ensemble every Nth gw")
    sp.add_argument("--no-openfpl", action="store_true",
                    help="skip the OpenFPL comparison entirely")
    sp.add_argument("--retrain-minutes", action="store_true",
                    help="force retraining the minutes classifier")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("simulate",
                        help="simulate a gameweek: floors, ceilings, P(haul), joint risk")
    sp.add_argument("--gw", type=int, default=None)
    sp.add_argument("--sims", type=int, default=4000)
    sp.add_argument("--top", type=int, default=20)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("verify",
                        help="check the data invariants; exits non-zero on error")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("prices", help="who is about to rise or fall in price")
    sp.add_argument("--gw", type=int, default=None,
                    help="predict the move out of this gw (default: latest with data)")
    sp.add_argument("--top", type=int, default=15, help="rows each way")
    sp.add_argument("--out", help="write predictions CSV")
    sp.set_defaults(func=cmd_prices)

    sp = sub.add_parser("decay",
                        help="grade archived pre-deadline snapshots by "
                             "hours-to-deadline (fills in as data accrues)")
    sp.add_argument("--season", default=None)
    sp.set_defaults(func=cmd_decay)

    sp = sub.add_parser("sensitivity",
                        help="which XI/captain calls are fragile vs robust "
                             "(margins + bootstrap stability)")
    sp.add_argument("--gw", type=int, required=True)
    sp.add_argument("--season", default=None)
    sp.add_argument("--entry", type=int, default=None,
                    help="analyse this FPL entry's 15 (default: template)")
    sp.add_argument("--sims", type=int, default=3000)
    sp.set_defaults(func=cmd_sensitivity)

    sp = sub.add_parser("errors",
                        help="model-error database: record and analyse "
                             "where the champion is systematically wrong")
    sp.add_argument("--season", default=config.CURRENT_SEASON)
    sp.add_argument("--replay", action="store_true",
                    help="point-in-time replay of a past season")
    sp.add_argument("--gw", type=int, default=None,
                    help="record one live gameweek after it finishes")
    sp.add_argument("--gws", type=int, nargs="*", default=None)
    sp.set_defaults(func=cmd_errors)

    sp = sub.add_parser("rank-backtest",
                        help="A/B the decision layer (MFRU vs max-xP) on a "
                             "fixed template squad, realised points")
    sp.add_argument("--backtest-season", default="2025-26")
    sp.add_argument("--gws", type=int, nargs="*", default=None)
    sp.add_argument("--sims", type=int, default=3000)
    sp.set_defaults(func=cmd_rank_backtest)

    sp = sub.add_parser("compare-rank-backtests",
                        help="pool rank-backtest JSONs, paired t vs baseline")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--baseline", default="xp")
    sp.set_defaults(func=cmd_compare_rank_backtests)

    sp = sub.add_parser("compare-backtests",
                        help="paired per-gameweek A/B of two saved backtest reports")
    sp.add_argument("before", help="backtest JSON (or a directory of them) from before the change")
    sp.add_argument("after", help="backtest JSON (or a directory of them) from after it")
    sp.add_argument("--model", default="xpts", help="which model's series to compare")
    sp.set_defaults(func=cmd_compare_backtests)

    sp = sub.add_parser("run", help="pull + build + predict")
    sp.add_argument("--gw", type=int, required=True)
    sp.add_argument("--cache", action="store_true")
    sp.add_argument("--no-history", action="store_true")
    sp.add_argument("--no-backfill", action="store_true")
    sp.add_argument("--understat", action="store_true")
    sp.add_argument("--top", type=int, default=40)
    sp.add_argument("--out", help="write predictions CSV")
    sp.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
