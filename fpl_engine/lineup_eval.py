"""Score the forward-collected predicted-lineup feed against the frozen model.

E16 closed the minutes programme: the remaining actionable error is P(start)
among the ~26 players the model ranks 5-30, false starters are the most
valuable class to remove, and no deadline-visible context resolves any of it.
The one instrument left is the RotoWire predicted-XI archive the scheduled
Action collects (`data/collected/lineups/<season>.csv`). This module is the
pre-registered evaluator for that feed (RESEARCH_LOG E17): it establishes the
feed's value BEFORE anything is integrated into the engine, one gameweek at a
time, as coverage accrues.

    python -m fpl_engine lineup-eval --gw 3

Design rules, in order of importance:

* **Deadline honesty.** A club's snapshot is the LAST predicted forecast
  observed STRICTLY before the FPL deadline (first kickoff minus 90 minutes).
  Confirmed XIs land after the deadline and are never an input; realised
  starts from `player_gw` are the ground truth.
* **Never trust the stored `gw` column.** The collector stamps FPL's
  "next_gw" at run time, which mislabels forecasts captured while the
  previous gameweek is still being played. Each (team, observation) is
  re-assigned to the gameweek of that club's next kickoff after the
  observation; disagreements with the stored column are reported.
* **Fail loudly on empty sources.** Every stage prints its row count and
  raises rather than returning a plausible zero — the E16 process rule,
  earned twice in one round.
* **No optimisation on the test gameweeks.** The soft arm's likelihood
  ratios are fitted only on gameweeks scored BEFORE the target one (none at
  first, so the first scored gameweek uses the pre-registered default), and
  the hard arm's exposure constants come from seasons before the target
  season.

Arms (the E17 comparison):
  baseline — the frozen production minutes model, availability overlay off
             (consistent with every backtested number; the live model is
             strictly stronger, so feed value vs this baseline is an upper
             bound on its value over the live path).
  hard     — feed treated as truth for covered clubs: XI members get the
             historical starter exposure profile, everyone else the
             non-starter profile.
  soft     — Bayesian update of the model's P(start) by the feed's
             likelihood ratio, exposure classes rescaled accordingly.
"""
from __future__ import annotations

import argparse
import os
import unicodedata

import numpy as np
import pandas as pd

from . import config, db, scoring, verify
from .xpts import engine as xe, minutes_model

# RotoWire abbreviation -> FPL short_name. Identity except where noted; an
# abbreviation not in FPL's short names for the season fails loudly.
ABBR_FIX = {"NOT": "NFO"}

# Pre-registered soft-arm default likelihood ratio for the FIRST scored
# gameweek (no feed history to fit on): a forecast in/out of the XI multiplies
# the start odds by 4 / divides by 4. From the first scored gameweek onward
# the ratios are refitted on previously scored gameweeks only.
DEFAULT_LR = 4.0

FEED_REL = os.path.join("data", "collected", "lineups")


# ---------------------------------------------------------------- loading

def feed_path(season: str, root: str | None = None) -> str:
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, FEED_REL, f"{season}.csv")


def load_feed(season: str, path: str | None = None) -> pd.DataFrame:
    path = path or feed_path(season)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"lineup archive not found: {path} — the scheduled collector "
            f"commits it to main; merge main or pass --feed")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"lineup archive is EMPTY: {path} — refusing to "
                         f"score an empty source as zero signal")
    df["observed_utc"] = pd.to_datetime(df["observed_utc"], utc=True)
    print(f"[lineup-eval] archive rows: {len(df)}  "
          f"({df.observed_utc.min():%Y-%m-%d} .. {df.observed_utc.max():%Y-%m-%d}, "
          f"{df.team_abbr.nunique()} clubs, statuses {dict(df.status.value_counts())})")
    return df


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().replace("-", " ").replace("'", "").split())


def team_map(conn, season: str) -> dict[str, int]:
    t = pd.read_sql_query(
        "SELECT team_id, short_name FROM team WHERE season=?",
        conn, params=(season,))
    if t.empty:
        raise ValueError(f"no teams stored for {season} — run pull first")
    by_short = dict(zip(t.short_name, t.team_id))
    return by_short


def resolve_abbr(abbr: str, by_short: dict[str, int]) -> int:
    short = ABBR_FIX.get(abbr, abbr)
    if short not in by_short:
        raise ValueError(f"RotoWire club abbreviation {abbr!r} does not map "
                         f"to an FPL club — extend ABBR_FIX")
    return by_short[short]


def match_players(conn, season: str, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach player_id to feed rows by name, within the mapped club.

    Exact normalized full name, then FPL web_name, then a last-name token
    unique within the club. Unmatched rows are returned with NaN and counted
    loudly by the caller.
    """
    pl = pd.read_sql_query(
        "SELECT player_id, team_id, full_name, web_name "
        "FROM player WHERE season=?", conn, params=(season,))
    by_team: dict[int, list[tuple]] = {}
    for r in pl.itertuples():
        full = _norm(r.full_name)
        by_team.setdefault(r.team_id, []).append(
            (r.player_id, full, _norm(r.web_name),
             full.split()[-1] if full else ""))
    out = []
    for r in rows.itertuples():
        cands = by_team.get(r.fpl_team_id, [])
        name = _norm(r.player)
        last = name.split()[-1] if name else ""
        pid = None
        for p, full, web, _ in cands:
            if name == full or name == web:
                pid = p
                break
        if pid is None:
            hits = [p for p, full, web, lst in cands
                    if last and (last == lst or last == web)]
            if len(hits) == 1:
                pid = hits[0]
        out.append(pid)
    rows = rows.copy()
    rows["player_id"] = out
    return rows


# ------------------------------------------------------- gw / deadlines

def gw_kickoffs(conn, season: str) -> pd.DataFrame:
    """(gw, team_id, kickoff) for every known fixture of the season."""
    fx = pd.read_sql_query(
        "SELECT gw, kickoff_utc, team_h, team_a FROM fixture "
        "WHERE season=? AND kickoff_utc IS NOT NULL", conn, params=(season,))
    tm = pd.read_sql_query(
        "SELECT gw, kickoff_utc, team_id FROM team_match "
        "WHERE season=? AND kickoff_utc IS NOT NULL",
        conn, params=(season,))
    # union of both sources: `fixture` carries the live season's future
    # fixtures, `team_match` the played ones — and for historical seasons
    # `fixture` is EMPTY, which produced two plausible zeros in E16. Neither
    # alone can be trusted to cover every gameweek.
    parts = [tm]
    if not fx.empty:
        parts += [
            fx.rename(columns={"team_h": "team_id"})[["gw", "kickoff_utc", "team_id"]],
            fx.rename(columns={"team_a": "team_id"})[["gw", "kickoff_utc", "team_id"]]]
    rows = pd.concat(parts, ignore_index=True)
    if rows.empty:
        raise ValueError(f"no fixtures with kickoffs stored for {season} — "
                         f"run pull first")
    rows["gw"] = pd.to_numeric(rows["gw"], errors="coerce")
    rows = rows.dropna(subset=["gw"])
    rows["kickoff"] = pd.to_datetime(rows["kickoff_utc"], utc=True)
    return (rows[["gw", "team_id", "kickoff"]]
            .drop_duplicates(["gw", "team_id", "kickoff"]))


def deadline_for(kicks: pd.DataFrame, gw: int) -> pd.Timestamp:
    g = kicks[kicks.gw == gw]
    if g.empty:
        raise ValueError(f"gameweek {gw} has no stored kickoffs")
    return g.kickoff.min() - pd.Timedelta(minutes=90)


def snapshot(feed: pd.DataFrame, conn, season: str, gw: int) -> pd.DataFrame:
    """The last pre-deadline predicted XI per club, player-matched.

    Rows are assigned to a gameweek by the club's NEXT kickoff after the
    observation, never by the stored column.
    """
    kicks = gw_kickoffs(conn, season)
    dl = deadline_for(kicks, gw)
    by_short = team_map(conn, season)
    f = feed[feed.status == "predicted"].copy()
    if f.empty:
        raise ValueError("archive holds no predicted rows at all")
    f["fpl_team_id"] = [resolve_abbr(a, by_short) for a in f.team_abbr]

    # each club's next kickoff after the observation decides the gw
    kt = {t: g.sort_values("kickoff") for t, g in kicks.groupby("team_id")}
    def gw_of(row):
        g = kt.get(row.fpl_team_id)
        if g is None:
            return np.nan
        nxt = g[g.kickoff > row.observed_utc]
        return int(nxt.gw.iloc[0]) if len(nxt) else np.nan
    f["gw_true"] = [gw_of(r) for r in f.itertuples()]
    mislabel = int((f["gw_true"] != f["gw"]).sum())
    if mislabel:
        print(f"[lineup-eval] {mislabel} rows re-assigned from the stored gw "
              f"column by next-kickoff (collector stamps next_gw at run time)")

    f = f[(f.gw_true == gw) & (f.observed_utc < dl)]
    print(f"[lineup-eval] GW{gw} deadline {dl:%Y-%m-%d %H:%M} UTC — "
          f"pre-deadline predicted rows: {len(f)} over "
          f"{f.team_abbr.nunique()} clubs")
    if f.empty:
        raise ValueError(
            f"no pre-deadline predicted rows for GW{gw} — the feed does not "
            f"cover this deadline (collection began 2026-08-31); this is a "
            f"coverage gap, not a zero-signal result")

    # last forecast per club
    last = f.groupby("fpl_team_id").observed_utc.transform("max")
    f = f[f.observed_utc == last].copy()
    f = match_players(conn, season, f)
    n_un = int(f.player_id.isna().sum())
    print(f"[lineup-eval] snapshot: {f.fpl_team_id.nunique()} clubs x XI = "
          f"{len(f)} rows, matched {len(f) - n_un} "
          f"({(len(f) - n_un) / len(f) * 100:.1f}%)")
    if n_un:
        for r in f[f.player_id.isna()].itertuples():
            print(f"    unmatched: {r.team_abbr}  {r.player!r}")
    bad = f.groupby("fpl_team_id").size()
    short = bad[bad != 11]
    if len(short):
        print(f"[lineup-eval] WARNING: {len(short)} clubs without exactly 11 "
              f"rows: {dict(short)}")
    return f


# ------------------------------------------------------------- the arms

def _exposure_profiles(conn, before_season: str) -> dict:
    """Historical exposure of starters / fit non-starters, from seasons
    strictly before the target season — the hard arm's constants."""
    ph = ",".join("?" * len(verify.PLAYER_POSITIONS))
    pg = pd.read_sql_query(
        f"SELECT pg.season, pg.minutes, pg.starts FROM player_gw pg "
        f"JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        f"WHERE pg.season < ? AND p.position IN ({ph}) "
        f"AND pg.kickoff_utc IS NOT NULL",
        conn, params=[before_season, *verify.PLAYER_POSITIONS])
    if pg.empty:
        raise ValueError("no historical player_gw rows to measure exposure "
                         "profiles from")
    for c in ("minutes", "starts"):
        pg[c] = pd.to_numeric(pg[c], errors="coerce").fillna(0)
    st = pg[pg.starts > 0]
    ns = pg[pg.starts == 0]
    prof = {
        "start": {"p_none": 0.0,
                  "p_full": float((st.minutes >= 60).mean()),
                  "p_sub": float((st.minutes < 60).mean()),
                  "p_start": 1.0,
                  "e_min": float(st.minutes.mean())},
        # non-starters include the injured; the feed cannot tell a rested
        # player from a crocked one, so the pooled profile is the honest one
        "bench": {"p_none": float((ns.minutes == 0).mean()),
                  "p_full": float((ns.minutes >= 60).mean()),
                  "p_sub": float(((ns.minutes > 0) & (ns.minutes < 60)).mean()),
                  "p_start": 0.0,
                  "e_min": float(ns.minutes.mean())},
    }
    print(f"[lineup-eval] exposure profiles (< {before_season}, "
          f"n={len(pg)}): starter P(60+)={prof['start']['p_full']:.3f} "
          f"E[min]={prof['start']['e_min']:.1f}; non-starter "
          f"P(0)={prof['bench']['p_none']:.3f}")
    return prof


def hard_frame(base: pd.DataFrame, snap: pd.DataFrame, prof: dict) -> pd.DataFrame:
    """Feed-as-truth override for covered clubs; uncovered clubs untouched."""
    out = base.copy()
    xi = set(snap.player_id.dropna().astype(int))
    covered = set(snap.fpl_team_id)
    in_xi = out.player_id.isin(xi).to_numpy()
    cov = out.team_id.isin(covered).to_numpy()
    for k, v in prof["start"].items():
        out.loc[in_xi, k] = v
    benched = cov & ~in_xi
    for k, v in prof["bench"].items():
        out.loc[benched, k] = v
    out.loc[in_xi | benched, "m_played"] = np.maximum(
        out.loc[in_xi | benched, "m_played"], 1.0)
    return out


def soft_frame(base: pd.DataFrame, snap: pd.DataFrame,
               lr_in: float, lr_out: float) -> pd.DataFrame:
    """Bayesian update of P(start) by the feed's likelihood ratio; the
    exposure classes are rescaled with the start probability."""
    out = base.copy()
    xi = set(snap.player_id.dropna().astype(int))
    covered = set(snap.fpl_team_id)
    s = out.p_start.clip(0.001, 0.999)
    odds = s / (1 - s)
    in_xi = out.player_id.isin(xi).to_numpy()
    benched = out.team_id.isin(covered).to_numpy() & ~in_xi
    odds = np.where(in_xi, odds * lr_in,
                    np.where(benched, odds / lr_out, odds))
    s_new = pd.Series(odds / (1 + odds), index=out.index)
    r = (s_new / s).clip(0.05, 20.0)
    touched = in_xi | benched
    p_full = (out.p_full * r).clip(0, 0.97)
    p_sub = out.p_sub * (0.5 + 0.5 * r).clip(0.25, 2.0)
    tot = (p_full + p_sub).clip(upper=0.99)
    scale = np.where(tot > 0.99, 0.99 / tot, 1.0)
    p_full, p_sub = p_full * scale, p_sub * scale
    e_old = (out.p_full + 0.4 * out.p_sub).clip(lower=0.01)
    e_new = p_full + 0.4 * p_sub
    for col, val in (("p_start", s_new), ("p_full", p_full), ("p_sub", p_sub),
                     ("p_none", 1.0 - p_full - p_sub),
                     ("e_min", out.e_min * (e_new / e_old))):
        out.loc[touched, col] = val[touched] if hasattr(val, "__getitem__") \
            else val
    return out


def fitted_lrs(scored: pd.DataFrame | None) -> tuple[float, float]:
    """LRs from previously scored gameweeks; the pre-registered default when
    none exist yet."""
    if scored is None or len(scored) < 60:
        print(f"[lineup-eval] soft arm: no prior scored gameweeks — using the "
              f"pre-registered default LR {DEFAULT_LR}")
        return DEFAULT_LR, DEFAULT_LR
    started = scored.started.astype(bool)
    inxi = scored.feed_start.astype(bool)
    tpr = float((inxi & started).sum()) / max(int(started.sum()), 1)
    fpr = float((inxi & ~started).sum()) / max(int((~started).sum()), 1)
    fnr, tnr = 1 - tpr, 1 - fpr
    lr_in = max(tpr, 1e-3) / max(fpr, 1e-3)
    lr_out = max(tnr, 1e-3) / max(fnr, 1e-3)
    print(f"[lineup-eval] soft arm: LRs fitted on {len(scored)} prior rows: "
          f"in-XI x{lr_in:.2f}, out /{lr_out:.2f}")
    return lr_in, lr_out


# ------------------------------------------------------------- scoring

def _metrics(pred: pd.DataFrame, act: pd.DataFrame) -> dict:
    j = pred.merge(act, on="player_id", how="inner")
    top = j.sort_values("prediction", ascending=False)
    return {"top11": float(top.head(11)["total_points"].mean()),
            "top30": float(top.head(30)["total_points"].mean()),
            "captain": float(top.head(1)["total_points"].iloc[0]),
            "n": len(j)}


def evaluate(conn, season: str, gw: int, feed: pd.DataFrame) -> dict:
    snap = snapshot(feed, conn, season, gw)
    bundle = minutes_model.ensure(conn)
    kicks = gw_kickoffs(conn, season)
    dl = deadline_for(kicks, gw)
    as_of = xe.first_kickoff(conn, season, gw)
    base = minutes_model.predict_gw(conn, season, as_of, *bundle, gw=gw,
                                    use_availability=False)
    team_of = pd.read_sql_query(
        "SELECT player_id, team_id FROM player WHERE season=?",
        conn, params=(season,))
    base = base.merge(team_of, on="player_id", how="left")
    print(f"[lineup-eval] model frame: {len(base)} players")
    if base.empty:
        raise ValueError("minutes model returned an empty frame")

    rules = scoring.load_rules()
    pred0 = xe.xpts_predict_gw(conn, season, gw, as_of=as_of,
                               use_availability=False, minutes_bundle=bundle,
                               rules=rules, minutes_override=base)
    if pred0.empty:
        raise ValueError("engine returned an empty prediction frame")
    rank = pred0.sort_values("prediction", ascending=False)
    rank_of = {p: i + 1 for i, p in enumerate(rank.player_id)}
    base["xp_rank"] = base.player_id.map(rank_of)
    base["feed_covered"] = base.team_id.isin(set(snap.fpl_team_id))
    xi = set(snap.player_id.dropna().astype(int))
    base["feed_start"] = base.player_id.isin(xi)

    ph = ",".join("?" * len(verify.PLAYER_POSITIONS))
    act = pd.read_sql_query(
        f"SELECT pg.player_id, SUM(pg.minutes) minutes, MAX(pg.starts) starts,"
        f" SUM(pg.total_points) total_points FROM player_gw pg "
        f"JOIN player p ON p.season=pg.season AND p.player_id=pg.player_id "
        f"WHERE pg.season=? AND pg.gw=? AND p.position IN ({ph}) "
        f"GROUP BY pg.player_id", conn,
        params=[season, gw, *verify.PLAYER_POSITIONS])
    for c in ("minutes", "starts", "total_points"):
        act[c] = pd.to_numeric(act[c], errors="coerce").fillna(0)
    played = act[act.minutes > 0]
    complete = len(played) > 150  # a full round has ~220+ players with minutes

    res = {"season": season, "gw": gw, "deadline": str(dl),
           "clubs_covered": int(snap.fpl_team_id.nunique()),
           "rows_matched": int(snap.player_id.notna().sum()),
           "complete": bool(complete)}

    b = base.merge(act[["player_id", "starts", "minutes"]],
                   on="player_id", how="left")
    b["started"] = b.starts.fillna(0) > 0
    focus = b[b.feed_covered & b.xp_rank.between(5, 30)]
    narrow = b[b.feed_covered & b.xp_rank.between(5, 15)]
    print(f"[lineup-eval] rank 5-30 rows in covered clubs: {len(focus)} "
          f"(5-15: {len(narrow)})")

    # the disagreement set the whole hypothesis lives on
    dis = focus[(focus.p_start > 0.6) & ~focus.feed_start]
    res["removals_5_30"] = int(len(dis))
    print(f"[lineup-eval] feed REMOVES {len(dis)} rank-5-30 players the model "
          f"has P(start)>0.6:")
    for r in dis.itertuples():
        print(f"    rank {int(r.xp_rank):3d}  p_start={r.p_start:.2f}  "
              f"player_id={int(r.player_id)}")
    add = focus[(focus.p_start < 0.4) & focus.feed_start]
    res["discoveries_5_30"] = int(len(add))

    if not complete:
        print(f"[lineup-eval] GW{gw} not complete ({len(played)} players with "
              f"minutes) — DRY RUN, no outcome scoring")
        return res

    # ---- outcome scoring ----
    cov = b[b.feed_covered]
    res["feed_tpr"] = float((cov.feed_start & cov.started).sum()
                            / max(int(cov.started.sum()), 1))
    res["feed_fpr"] = float((cov.feed_start & ~cov.started).sum()
                            / max(int((~cov.started).sum()), 1))
    res["removal_precision_5_30"] = float((~dis.started).mean()) if len(dis) \
        else np.nan
    res["discovery_precision_5_30"] = float(add.started.mean()) if len(add) \
        else np.nan
    for name, g in (("all_covered", cov), ("rank_5_30", focus),
                    ("rank_5_15", narrow)):
        fs = g[g.p_start > 0.6]
        res[f"false_starter_rate_model_{name}"] = \
            float((~fs.started).mean()) if len(fs) else np.nan
        ff = g[g.feed_start]
        res[f"false_starter_rate_feed_{name}"] = \
            float((~ff.started).mean()) if len(ff) else np.nan
        res[f"n_{name}"] = int(len(g))

    prof = _exposure_profiles(conn, season)
    lr_in, lr_out = fitted_lrs(None)   # prior scored gws wired in later
    # the full-minutes oracle normalises every arm: (arm - baseline) /
    # (oracle - baseline) is the share of the reachable gain the feed gets
    oracle = base.copy()
    om = oracle.merge(act[["player_id", "minutes", "starts"]],
                      on="player_id", how="left")
    mins = om["minutes"].fillna(0.0).to_numpy()
    pl_, fu_ = (mins > 0).astype(float), (mins >= 60).astype(float)
    oracle["p_none"], oracle["p_sub"], oracle["p_full"] = \
        1.0 - pl_, pl_ - fu_, fu_
    oracle["p_start"] = om["starts"].fillna(0.0).clip(0, 1).to_numpy()
    oracle["e_min"] = mins
    oracle["m_played"] = np.where(mins > 0, mins, oracle["m_played"])
    arms = {"baseline": base,
            "hard": hard_frame(base, snap, prof),
            "soft": soft_frame(base, snap, lr_in, lr_out),
            "oracle": oracle}
    eps = 1e-6
    for name, mo in arms.items():
        pred = pred0 if name == "baseline" else xe.xpts_predict_gw(
            conn, season, gw, as_of=as_of, use_availability=False,
            minutes_bundle=bundle, rules=rules,
            minutes_override=mo.drop(columns=[c for c in mo.columns
                                              if c not in base.columns]))
        m = _metrics(pred, act)
        j = mo.merge(act[["player_id", "starts"]], on="player_id", how="left")
        j["started"] = j.starts.fillna(0) > 0
        jc = j[j.player_id.isin(cov.player_id)]
        p = jc.p_start.clip(eps, 1 - eps)
        m["ll_start_covered"] = float(-np.mean(
            np.where(jc.started, np.log(p), np.log(1 - p))))
        m["cal_start_covered"] = float(jc.started.mean() - jc.p_start.mean())
        res[name] = m
        print(f"[lineup-eval] arm {name:9s} top11={m['top11']:.2f} "
              f"top30={m['top30']:.2f} captain={m['captain']:.0f} "
              f"ll_start={m['ll_start_covered']:.4f}")
    # oracle: how much of the start-side error does the best arm remove?
    j = base.merge(act[["player_id", "starts"]], on="player_id", how="left")
    j["started"] = j.starts.fillna(0) > 0
    jc = j[j.feed_covered]
    p0 = jc.p_start.clip(eps, 1 - eps)
    ll0 = float(-np.mean(np.where(jc.started, np.log(p0), np.log(1 - p0))))
    # perfect start knowledge drives start log-loss to zero, so the share of
    # baseline log-loss removed IS the share of residual P(start) error the
    # feed resolves (the E17 oracle question)
    arm_ll = {n: a["ll_start_covered"] for n, a in res.items()
              if isinstance(a, dict) and "ll_start_covered" in a
              and n not in ("baseline", "oracle")}
    res["oracle_share_ll"] = float((ll0 - min(arm_ll.values())) / ll0) \
        if ll0 > 0 and arm_ll else np.nan
    o_gain = res["oracle"]["top11"] - res["baseline"]["top11"]
    if abs(o_gain) > 1e-9:
        for n in ("hard", "soft"):
            res[f"oracle_share_top11_{n}"] = float(
                (res[n]["top11"] - res["baseline"]["top11"]) / o_gain)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default=config.CURRENT_SEASON)
    ap.add_argument("--gw", type=int, required=True)
    ap.add_argument("--feed", default=None,
                    help="path to the lineup archive CSV (default: "
                         "data/collected/lineups/<season>.csv)")
    a = ap.parse_args(argv)
    feed = load_feed(a.season, a.feed)
    with db.session() as conn:
        res = evaluate(conn, a.season, a.gw, feed)
    print("\n[lineup-eval] result:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    return res


if __name__ == "__main__":
    main()
