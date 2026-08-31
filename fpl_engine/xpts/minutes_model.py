"""Minutes model: P(plays 0 / 1-59 / 60+) and E[minutes] per player per gameweek.

Most week-to-week prediction error in FPL is minutes, not form. A replay of
2024-25 and 2025-26 with *perfect* minutes knowledge (everything else
unchanged) gains +0.21 Spearman and +0.7-0.9 actual points per player across
the top 30 — several times what any rate-model tuning is worth. This module is
therefore the highest-leverage component in the engine.

Design notes
------------
**One feature builder for training and prediction.** ``_frame`` appends
synthetic rows for the gameweek being predicted to the history frame, so the
same shifted rolling transforms produce the serving features. (An earlier
version re-implemented the rolling features by hand inside ``predict_gw``,
which is the classic source of train/serve skew.)

**Every prior season trains the model.** The holdout accuracy is measured by a
probe fitted on all-but-the-last season, and the shipped model is then refitted
on *all* of them — the most recent season is the most relevant one and must not
be thrown away.

**Features** are point-in-time by construction (all shift(1) within a player):

* history      trailing minutes/starts/appearances, recency of last outing
* role/depth   average minutes *when he starts*, consecutive starts, share of
               his team+position's minutes, minutes volatility, price rank
               inside the team+position (raw price is deliberately excluded —
               it adds a "fame" bias that overrides recent-minutes signal for
               benched stars; the rank alone carries the depth chart)
* context      days of rest, matches in the previous 14 days, home/away, how
               many fixtures his team has this gameweek
* crowd        the previous gameweek's ownership and net transfers — the
               market's read on who is starting, free and known at the deadline

Availability (FPL's status/chance_next) is applied *on top* at prediction time:
the played-probability mass is scaled by the availability and the remainder
moved to the 0-minutes class. Backtests disable it (the stored status is
today's, not that gameweek's).

The trained classifier is cached in models/xpts/. ``ensure`` trains it on
demand; a cache built from a different feature set is detected and retrained.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .. import config

MODEL_DIR = os.path.join(config.MODELS_DIR, "xpts")
MODEL_PATH = os.path.join(MODEL_DIR, "minutes_xgb.json")
REG_PATH = os.path.join(MODEL_DIR, "minutes_reg.json")
START_PATH = os.path.join(MODEL_DIR, "minutes_start.json")
META_PATH = os.path.join(MODEL_DIR, "minutes_meta.json")


def _paths(tag: str | None):
    """Cache file names. ``tag`` keeps a backtest's model (trained only on the
    seasons before the replayed one) out of the live cache — otherwise running
    a backtest silently downgrades every subsequent live prediction."""
    if not tag:
        return MODEL_PATH, REG_PATH, START_PATH, META_PATH
    return (MODEL_PATH.replace(".json", f".{tag}.json"),
            REG_PATH.replace(".json", f".{tag}.json"),
            START_PATH.replace(".json", f".{tag}.json"),
            META_PATH.replace(".json", f".{tag}.json"))

HISTORY_FEATURES = [
    "starts_l5", "mins_l1", "mins_l3", "mins_l5", "avg_mins_when_played",
    "apps_l10", "since_last_app", "season_min_share",
    "new_club", "career_apps", "reserve_flag", "season_idx",
    "is_gk", "is_def", "is_mid", "is_fwd",
]
ROLE_FEATURES = ["price_rank", "avg_mins_when_started", "start_rate_l10",
                 "consec_starts", "pos_mins_share_l5", "mins_std_l5", "started_l1"]
CONTEXT_FEATURES = ["days_rest", "team_matches_14d", "is_home", "team_gw_fixtures"]
CROWD_FEATURES = ["sel_share_lag", "sel_rank_pos", "net_transfer_frac"]

FEATURES = HISTORY_FEATURES + ROLE_FEATURES + CONTEXT_FEATURES + CROWD_FEATURES
LABELS = {0: "none", 1: "sub", 2: "full"}   # 0 min / 1-59 / 60+

# --- optional exogenous blocks (Transfermarkt) ------------------------------
# Empty by default, so the shipped model is bit-identical to the one every
# number in CLAUDE.md was measured on. A block is switched on for one process
# with $FPL_MINUTES_EXTRA (``inj``, ``tm``, ``tm_player``, …), which is how a
# backtest runs the challenger arm without a second copy of the module.
EXTRA_BLOCKS = {"age": "fpl birth date", "inj": "injury",
                "tm": "transfermarkt"}
EXTRA_FEATURES: list[str] = []


def _resolve_extras(names: list[str]) -> list[str]:
    from . import injury_features as _inj, tm_features as _tm
    out: list[str] = []
    for n in names:
        n = n.strip().lower()
        if not n:
            continue
        if n == "age":
            out += ["fpl_age", "age_known"]
        elif n == "inj":
            out += _inj.FEATURES
        elif n == "inj_history":
            out += _inj.HISTORY_ONLY
        elif n == "inj_out":
            out += ["inj_currently_out"]
        elif n == "tm":
            out += _tm.ALL
        elif n in _tm.FAMILIES:
            out += _tm.FAMILIES[n]
        else:
            raise ValueError(f"unknown minutes-model extra block: {n!r}")
    return list(dict.fromkeys(out))


def set_extras(names: list[str] | str | None) -> list[str]:
    """Switch optional feature blocks on for this process. Returns the list."""
    global EXTRA_FEATURES
    if not names:
        EXTRA_FEATURES = []
    else:
        if isinstance(names, str):
            names = names.split(",")
        EXTRA_FEATURES = _resolve_extras(list(names))
    return EXTRA_FEATURES


def active_features() -> list[str]:
    return FEATURES + EXTRA_FEATURES


def _attach_extras(conn, df: pd.DataFrame) -> pd.DataFrame:
    """Join whichever optional blocks the active feature set asks for.

    Both attach on ``player_code`` and filter strictly on ``kick``, so this is
    the same point-in-time contract the rolling features honour.
    """
    if not EXTRA_FEATURES or df.empty:
        return df
    from . import injury_features as _inj, tm_features as _tm
    if any(f in EXTRA_FEATURES for f in _inj.FEATURES):
        df = _inj.add_features(df, _inj.spells(conn))
    if any(f in EXTRA_FEATURES for f in _tm.ALL):
        df = _tm.add_features(df, _tm.load(conn),
                              pl_clubs=_tm.pl_club_ids(conn))
    return df


# ---------------------------------------------------------------- frame -----
def _history(conn, seasons: list[str], before: str | None) -> pd.DataFrame:
    q = ("SELECT pg.season, pg.gw, pg.player_id, pg.player_code, pg.fixture_id, "
         "pg.kickoff_utc, pg.minutes, pg.starts, pg.team_id, pg.was_home, "
         "pg.price price_gw, pg.selected, pg.transfers_in, pg.transfers_out, "
         "p.position position, p.birth_date birth_date, p.now_cost * 10.0 price "
         "FROM player_gw pg JOIN player p "
         "ON p.season=pg.season AND p.player_id=pg.player_id "
         # Assistant Managers are selectable entries that score points and
         # never play; they are not training rows for a minutes model
         "WHERE p.position IN ('GK','DEF','MID','FWD') "
         "AND pg.season IN (%s)" % ",".join("?" * len(seasons)))
    args = list(seasons)
    if before:
        q += " AND pg.kickoff_utc < ?"
        args.append(before)
    df = pd.read_sql_query(q, conn, params=args)
    # ``player.now_cost`` is in £m and ``player_gw.price`` in tenths of £m —
    # the query rescales the former so the fallback below never mixes units
    # inside a team+position group and corrupts the depth-chart rank.
    # Coerce up front too: an all-NULL numeric column comes back as object,
    # and concatenating it with the target rows would change dtypes mid-build
    for c in ("minutes", "starts", "team_id", "was_home", "price_gw", "price",
              "selected", "transfers_in", "transfers_out"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _target_rows(conn, season: str, gw: int, as_of: str) -> pd.DataFrame:
    """Synthetic rows for the gameweek being predicted (no outcome columns)."""
    fx = pd.read_sql_query(
        "SELECT fixture_id, gw, kickoff_utc, team_h, team_a FROM fixture "
        "WHERE season=? AND gw=?", conn, params=(season, gw))
    if fx.empty:      # historical seasons: fixtures live in team_match only
        fx = pd.read_sql_query(
            "SELECT fixture_id, gw, kickoff_utc, team_id team_h, opponent_id team_a "
            "FROM team_match WHERE season=? AND gw=? AND was_home=1",
            conn, params=(season, gw))
    if fx.empty:
        return pd.DataFrame()
    tf = pd.concat([
        fx.rename(columns={"team_h": "team_id", "team_a": "opponent_id"}).assign(was_home=1),
        fx.rename(columns={"team_a": "team_id", "team_h": "opponent_id"}).assign(was_home=0),
    ])[["fixture_id", "gw", "kickoff_utc", "team_id", "was_home"]]

    players = pd.read_sql_query(          # now_cost £m -> tenths, see _history
        "SELECT player_id, code player_code, position, team_id, birth_date, "
        "now_cost * 10.0 price FROM player WHERE season=? "
        "AND position IN ('GK','DEF','MID','FWD')",
        conn, params=(season,))
    # point-in-time club: the team of his most recent match THIS season. Only a
    # player with no rows yet (a summer signing) falls back to the squad list,
    # which is what makes ``new_club`` mean "arrived since his last match".
    last_team = pd.read_sql_query(
        "SELECT player_id, team_id FROM ("
        "  SELECT player_id, team_id, ROW_NUMBER() OVER "
        "    (PARTITION BY player_id ORDER BY kickoff_utc DESC) rn "
        "  FROM player_gw WHERE season=? AND kickoff_utc < ?) WHERE rn=1",
        conn, params=(season, as_of))
    players["team_id"] = pd.to_numeric(players["player_id"].map(
        dict(zip(last_team["player_id"], last_team["team_id"]))
    ).fillna(players["team_id"]), errors="coerce")
    # price: the squad-list price, else the last per-gameweek price seen. An
    # unpriced deep-squad player must not become a NaN branch the training
    # data never contains.
    last_price = pd.read_sql_query(
        "SELECT player_id, price FROM ("
        "  SELECT player_id, price, ROW_NUMBER() OVER "
        "    (PARTITION BY player_id ORDER BY kickoff_utc DESC) rn "
        "  FROM player_gw WHERE season=? AND kickoff_utc < ? AND price IS NOT NULL"
        ") WHERE rn=1", conn, params=(season, as_of))
    players["price"] = pd.to_numeric(players["price"], errors="coerce").fillna(
        players["player_id"].map(dict(zip(last_price["player_id"],
                                          last_price["price"]))))

    out = players.merge(tf, on="team_id", how="inner")
    out["season"] = season
    for c in ("minutes", "starts", "price_gw", "selected", "transfers_in",
              "transfers_out"):
        out[c] = np.nan
    out["_target"] = 1
    return out


EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DAY = pd.Timedelta(days=1)


def _team_congestion(tm: pd.DataFrame) -> pd.DataFrame:
    """Days of rest and matches in the previous 14 days, per team-match.

    Days come from dividing by a Timedelta, not from rescaling the integer
    view. pandas keeps whatever resolution a timestamp was parsed at, and
    since pandas 2.0 an ISO8601 string parses to MICROseconds — so
    ``astype("int64") / 86_400e9`` silently returned days/1000. ``days_rest``
    survived that (a tree only reads the order, and nothing then hit the
    30-day clip) but ``team_matches_14d`` did not: a 14 that means 14,000 days
    counts every previous match of the season, so the congestion feature was
    really a gameweek counter. The model has documented itself as carrying a
    fixture-congestion signal that it did not have.
    """
    tm = tm.sort_values(["season", "team_id", "kick"]).copy()
    rest, cong = [], []
    for _, d in tm.groupby(["season", "team_id"], sort=False):
        ks = ((d["kick"] - EPOCH) / DAY).to_numpy()               # days
        for i, k in enumerate(ks):
            prev = ks[:i]
            rest.append(k - prev[-1] if i else np.nan)
            cong.append(float((k - prev <= 14).sum()))
    tm["days_rest"] = rest
    tm["team_matches_14d"] = cong
    return tm


def _frame(conn, seasons: list[str], before: str | None = None,
           target: tuple | None = None) -> pd.DataFrame:
    """One row per player-fixture with point-in-time features + label.

    Rows are ordered per player by kickoff; every feature uses shifted
    (strictly prior) values only. ``target=(season, gw, as_of)`` appends the
    rows of the gameweek being predicted, whose label is NaN.
    """
    hist = _history(conn, seasons, before)
    hist["_target"] = 0
    df = hist
    if target is not None:
        tgt = _target_rows(conn, *target)
        if len(tgt):
            df = pd.concat([hist, tgt], ignore_index=True, sort=False)
    if df.empty:
        return df

    # price at that gameweek (vaastav/FPL `value`), falling back to the season
    # row's now_cost; point-in-time, so a summer price change never leaks back
    df["price"] = pd.to_numeric(df["price_gw"], errors="coerce").fillna(
        pd.to_numeric(df["price"], errors="coerce"))
    df["kick"] = pd.to_datetime(df["kickoff_utc"], utc=True, format="ISO8601")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["played"] = np.where(df["minutes"].notna(),
                            (df["minutes"] > 0).astype(float), np.nan)
    df["started"] = pd.to_numeric(df["starts"], errors="coerce")
    df = df.sort_values(["player_code", "kick"]).reset_index(drop=True)

    # depth chart: price rank within that gameweek's team+position (1 = priciest)
    df["price_rank"] = df.groupby(["season", "gw", "team_id", "position"])[
        "price"].rank(ascending=False, method="min")

    g = df.groupby("player_code", sort=False)
    df["starts_l5"] = g["started"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    df["start_rate_l10"] = g["started"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())
    df["started_l1"] = g["started"].transform(lambda s: s.shift(1))
    for n in (1, 3, 5):
        df[f"mins_l{n}"] = g["minutes"].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean())
    df["mins_std_l5"] = g["minutes"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std())
    df["_pm"] = df["minutes"].where(df["played"] > 0)
    df["avg_mins_when_played"] = g["_pm"].transform(
        lambda s: s.shift(1).expanding().mean())
    # a 90-minute man and one habitually hooked at 65 are different assets
    df["_ms"] = df["minutes"].where(df["started"] > 0)
    df["avg_mins_when_started"] = g["_ms"].transform(
        lambda s: s.shift(1).expanding().mean())
    df["apps_l10"] = g["played"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).sum())

    def _since(s):          # matches since his last appearance
        out, count = [], 99.0
        for v in s:
            out.append(count)
            count = 0.0 if (v is not None and v > 0) else min(99.0, count + 1)
        return pd.Series(out, index=s.index)
    df["since_last_app"] = g["played"].transform(_since)

    def _consec(s):         # length of his current run of starts
        out, run = [], 0.0
        for v in s:
            out.append(run)
            run = run + 1 if v == 1 else 0.0
        return pd.Series(out, index=s.index)
    df["consec_starts"] = g["started"].transform(_consec)

    cum_min = g["minutes"].transform(lambda s: s.shift(1).expanding().sum())
    cum_n = g["minutes"].transform(lambda s: s.shift(1).expanding().count())
    df["season_min_share"] = cum_min / (cum_n * 90.0)
    df["new_club"] = (g["team_id"].shift(1) != df["team_id"]).astype(float)
    df["career_apps"] = g.cumcount().astype(float)
    # perennial reserve: long squad tenure without a single appearance. This
    # separates the deep-squad phantom (never plays) from a debut row — both
    # share since_last_app=99, and debuts DO often play.
    _played_before = g["played"].transform(
        lambda s: s.shift(1).fillna(0.0).cumsum())
    df["reserve_flag"] = ((_played_before == 0) & (df["career_apps"] >= 10)
                          ).astype(float)
    df["season_idx"] = df.groupby(["player_code", "season"],
                                  sort=False).cumcount().astype(float)
    for pos, col in (("GK", "is_gk"), ("DEF", "is_def"),
                     ("MID", "is_mid"), ("FWD", "is_fwd")):
        df[col] = (df["position"] == pos).astype(float)

    # fixture context: rotation is driven by the calendar, not only by form
    tm = df[["season", "team_id", "fixture_id", "kick"]].drop_duplicates(
        ["season", "team_id", "fixture_id"])
    df = df.merge(_team_congestion(tm)[["season", "team_id", "fixture_id",
                                        "days_rest", "team_matches_14d"]],
                  on=["season", "team_id", "fixture_id"], how="left")
    df["days_rest"] = df["days_rest"].clip(upper=30).fillna(30.0)
    df["is_home"] = pd.to_numeric(df["was_home"], errors="coerce").fillna(0.5)
    df["team_gw_fixtures"] = df.groupby(["season", "gw", "team_id"])[
        "fixture_id"].transform("nunique")

    # his share of the team+position minutes over the last 5 matches
    df["_m5"] = g["minutes"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["pos_mins_share_l5"] = df["_m5"] / df.groupby(
        ["season", "gw", "team_id", "position"])["_m5"].transform("sum").replace(0, np.nan)

    # crowd signals, lagged one gameweek so only settled snapshots are used
    df["sel_lag"] = g["selected"].transform(lambda s: s.shift(1))
    df["sel_share_lag"] = df["sel_lag"] / df.groupby(["season", "gw"])[
        "sel_lag"].transform("sum").replace(0, np.nan)
    df["sel_rank_pos"] = df.groupby(["season", "gw", "team_id", "position"])[
        "sel_lag"].rank(ascending=False, method="min")
    ti = g["transfers_in"].transform(lambda s: s.shift(1))
    to = g["transfers_out"].transform(lambda s: s.shift(1))
    df["net_transfer_frac"] = ((ti - to) / df["sel_lag"].replace(0, np.nan)
                               ).clip(-2, 2)

    # FPL publishes `birth_date` from 2024-25 and the pull carries it back
    # across seasons on the stable code, so age is free wherever FPL has ever
    # listed the player. It is the one thing that separates the two kinds of
    # player with no Premier League history at all: among men with under five
    # career appearances every trailing feature is identical, and P(60+) still
    # runs from 0.00 at 15-18 to 0.40 at 24-27.
    #
    # The remainder is IMPUTED rather than left missing, and that is not a
    # convenience. FPL began publishing the field in 2024-25, so coverage runs
    # 56% / 60% / 88% / 99% across seasons: a tree left to learn the missing
    # branch learns it on a training population that has all but vanished by
    # serve time. Held out on 2025-26 the raw column was WORSE than no age at
    # all (+0.34% log-loss on the cold-start segment) while the imputed one is
    # better (-1.57%), and the difference is entirely this shift.
    if "birth_date" in df.columns:
        bd = pd.to_datetime(df["birth_date"], errors="coerce", utc=True)
        bd = bd.groupby(df["player_code"]).transform(
            lambda s: s.ffill().bfill())
        age = (df["kick"] - bd).dt.days / 365.25
        med = age.groupby(df["position"]).transform("median")
        df["fpl_age"] = age.fillna(med).fillna(age.median())
        df["age_known"] = age.notna().astype(float)
    else:
        df["fpl_age"] = np.nan
        df["age_known"] = 0.0

    df["label"] = np.select([df["minutes"] >= 60, df["minutes"] > 0],
                            [2, 1], 0).astype(float)
    df.loc[df["minutes"].isna(), "label"] = np.nan
    df = df.drop(columns=["_pm", "_ms", "_m5"])
    return _attach_extras(conn, df)


# ---------------------------------------------------------------- train -----
def _fit(frame: pd.DataFrame, seasons: list[str], device: str | None):
    import xgboost as xgb
    tr = frame[frame["label"].notna() & frame["season"].isin(seasons)]
    clf = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, n_estimators=400, max_depth=5,
        learning_rate=0.06, subsample=0.9, colsample_bytree=0.8,
        eval_metric="mlogloss", device=device or "cpu")
    clf.fit(tr[active_features()], tr["label"].astype(int))
    return clf, tr


def _fit_minutes(tr: pd.DataFrame, device: str | None):
    """Regressor for E[minutes | the player appears]."""
    import xgboost as xgb
    played = tr[tr["minutes"] > 0]
    m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.06,
                         subsample=0.9, colsample_bytree=0.8,
                         objective="reg:squarederror", device=device or "cpu")
    m.fit(played[active_features()], played["minutes"])
    return m


def _fit_start(tr: pd.DataFrame, device: str | None):
    """P(he is in the starting XI).

    Not more accurate than the three-class model's 60+ class — measured at AUC
    0.944/0.953 against 0.945/0.954, and six attempts at sharpening it have all
    returned nothing. It is trained because it answers the question people
    actually ask, on a scale they can read, and because the alternative on
    display was a trailing average of the last ten starts, which carries more
    than twice the log-loss (0.57/0.54 against 0.28/0.25).
    """
    import xgboost as xgb
    y = (tr["starts"].fillna(0) > 0).astype(int)
    m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=400,
                          max_depth=5, learning_rate=0.06, subsample=0.9,
                          colsample_bytree=0.8, eval_metric="logloss",
                          device=device or "cpu")
    m.fit(tr[active_features()], y)
    return m


def train(conn, *, seasons: list[str] | None = None,
          device: str | None = None, tag: str | None = None) -> dict:
    """Train and cache the classifier; returns metadata with holdout accuracy.

    The holdout number comes from a probe fitted on all-but-the-last season;
    the cached model is then refitted on every season given, because the most
    recent one is the most relevant training data there is.
    """
    seasons = seasons or config.BACKFILL_SEASONS
    frame = _frame(conn, seasons)
    acc = None
    valid_season = seasons[-1] if len(seasons) > 1 else None
    if valid_season:
        probe, _ = _fit(frame, seasons[:-1], device)
        va = frame[frame["label"].notna() & (frame["season"] == valid_season)]
        if len(va):
            acc = float((probe.predict(va[active_features()])
                         == va["label"]).mean())
    clf, tr = _fit(frame, seasons, device)
    # E[minutes | he plays], fitted on appearances only. Rebuilding expected
    # minutes as P(plays) x this beats the class-mean reconstruction it
    # replaced by 0.43 / 0.47 minutes of MAE across two replayed seasons, and
    # exposure multiplies every rate in the engine.
    reg = _fit_minutes(tr, device)
    start = _fit_start(tr, device)
    m_sub = float(tr.loc[tr.label == 1, "minutes"].mean() or 30.0)
    m_full = float(tr.loc[tr.label == 2, "minutes"].mean() or 84.0)
    model_path, reg_path, start_path, meta_path = _paths(tag)
    os.makedirs(MODEL_DIR, exist_ok=True)
    clf.save_model(model_path)
    reg.save_model(reg_path)
    start.save_model(start_path)
    meta = {"features": active_features(), "train_seasons": seasons,
            "valid_season": valid_season, "holdout_accuracy": acc,
            "mean_minutes": {"sub": m_sub, "full": m_full}}
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def load(tag: str | None = None):
    """Return (clf, meta), or (None, None) when absent or built on old features.

    ``meta["_reg"]`` and ``meta["_start"]`` carry loaded models. They are live
    objects, not part of the JSON the trainer writes, so never serialise a
    loaded meta.
    """
    import xgboost as xgb
    model_path, reg_path, start_path, meta_path = _paths(tag)
    if not all(os.path.exists(p)
               for p in (model_path, reg_path, start_path, meta_path)):
        return None, None
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("features") != active_features():
        return None, None      # stale cache: the feature set has changed
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    reg = xgb.XGBRegressor()
    reg.load_model(reg_path)
    start = xgb.XGBClassifier()
    start.load_model(start_path)
    meta = {**meta, "_reg": reg, "_start": start}
    return clf, meta


def ensure(conn, *, seasons: list[str] | None = None,
           device: str | None = None, tag: str | None = None):
    """Load the cached model, training it first if it is missing or stale."""
    clf, meta = load(tag)
    if clf is None:
        train(conn, seasons=seasons, device=device, tag=tag)
        clf, meta = load(tag)
    return clf, meta


# -------------------------------------------------------------- predict -----
def _gw_for(conn, season: str, as_of: str) -> int | None:
    for table in ("fixture", "team_match"):
        r = conn.execute(
            f"SELECT MIN(gw) g FROM {table} WHERE season=? AND kickoff_utc>=?",
            (season, as_of)).fetchone()
        if r and r["g"] is not None:
            return int(r["g"])
    return None


def predict_gw(conn, season: str, as_of: str, clf, meta, *,
               gw: int | None = None, use_availability: bool = True) -> pd.DataFrame:
    """P(none/sub/full) and E[minutes] for every player of ``season``.

    Features come from history strictly before ``as_of`` (cross-season via
    player_code, so early-season windows reach back into last year).
    ``use_availability`` applies the *current* FPL status/chance_next on top —
    right for live predictions, wrong for historical backtests (the stored
    status is today's, not that gameweek's), so backtests disable it.
    """
    gw = gw if gw is not None else _gw_for(conn, season, as_of)
    seasons = list(dict.fromkeys(config.BACKFILL_SEASONS + [season]))
    players = pd.read_sql_query(
        "SELECT player_id, position, status, chance_next FROM player "
        "WHERE season=? AND position IN ('GK','DEF','MID','FWD')",
        conn, params=(season,))

    t = feat_rows = pd.DataFrame()
    if gw is not None:
        df = _frame(conn, seasons, before=as_of, target=(season, gw, as_of))
        if len(df):
            t = feat_rows = df[df["_target"] == 1]
    if len(t):
        proba = clf.predict_proba(t[meta["features"]].astype(float))
        t = t.assign(p_none=proba[:, 0], p_sub=proba[:, 1], p_full=proba[:, 2])
        # a double gameweek gives one row per fixture; the engine scales
        # exposure per fixture itself, so hand it the per-fixture average
        sm = meta.get("_start")
        if sm is not None:
            t = t.assign(p_start=sm.predict_proba(
                t[meta["features"]].astype(float))[:, 1])
        else:
            t = t.assign(p_start=t["p_full"])
        t = t.groupby("player_id", as_index=False).agg(
            p_none=("p_none", "mean"), p_sub=("p_sub", "mean"),
            p_full=("p_full", "mean"), p_start=("p_start", "mean"),
            m_started=("avg_mins_when_started", "mean"))
    else:                      # blank gameweek, or no fixture list yet
        t = pd.DataFrame(columns=["player_id", "p_none", "p_sub", "p_full",
                                  "p_start", "m_started"])

    # players whose club has no fixture this gameweek still need a row: the
    # engine finds no fixtures for them and scores 0, but they must not vanish
    out = players.merge(t, on="player_id", how="left")
    out["p_sub"] = out["p_sub"].fillna(0.0)
    out["p_full"] = out["p_full"].fillna(0.0)
    out["p_none"] = out["p_none"].fillna(1.0)
    out["p_start"] = out["p_start"].fillna(0.0)

    if use_availability:
        # availability overlay: scale played mass, dump remainder on p_none
        chance = pd.to_numeric(out["chance_next"], errors="coerce")
        avail = chance.where(
            chance.notna(),
            out["status"].isin([None, "a"]).astype(float)).clip(0, 1).fillna(1.0)
        out["p_sub"] *= avail
        out["p_full"] *= avail
        out["p_start"] *= avail
        out["p_none"] = 1.0 - out["p_sub"] - out["p_full"]

    ms, mf = meta["mean_minutes"]["sub"], meta["mean_minutes"]["full"]
    reg = meta.get("_reg")
    if reg is not None and len(feat_rows):
        # hybrid: P(plays) x E[minutes | plays]
        cond = pd.Series(reg.predict(feat_rows[meta["features"]].astype(float)),
                         index=feat_rows.index).clip(1, 95)
        cond = cond.groupby(feat_rows["player_id"]).mean()
        cm = out["player_id"].map(cond)
        out["m_played"] = cm.fillna(out["m_started"].fillna(mf).clip(60, 90))
        out["e_min"] = (out["p_sub"] + out["p_full"]) * out["m_played"]
    else:                       # no regressor cached: class-mean fallback
        out["m_played"] = out["m_started"].fillna(mf).clip(60, 90)
        out["e_min"] = out["p_sub"] * ms + out["p_full"] * out["m_played"]
    # ``m_played`` is E[minutes | he appears]. It is published because the
    # simulator needs it directly: E[minutes] can no longer be inverted back
    # into a per-class minutes figure now that it is P(plays) x this.
    return out[["player_id", "position", "p_none", "p_sub", "p_full",
                "p_start", "e_min", "m_played"]]
