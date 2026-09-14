"""Round 20 fixture context from the BBC archive: the referee's card rate and
each club's set-play / open-play split, both point-in-time.

* `referee_factors` — the appointed referee's decayed yellow cards per match
  over his PRIOR matches, shrunk to the league rate and expressed relative to
  it. Officials come from the lineups payload (`acq_bbc_official`), the cards
  from `player_gw`. The appointment is public days before the deadline, so
  for a replay the match's own referee is a legal input.
* `setplay_factors` — for every club-fixture, how the OPPONENT's expected
  goals conceded split between set play and open play over its prior
  matches (`acq_bbc_match_stats`, Opta xG split, 2024-25 on), relative to
  the league split and shrunk toward it. A club that leaks from corners is a
  better fixture for a defender than its total xGA says. The player's own
  set-play share is his position's league share (`POS_SET_SHARE`) until a
  per-player shot log is loaded, so the factor is mean-preserving for a
  league-average shot mix and only reshapes it.

Every factor is 1.0 wherever the archive is silent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..pressers import BBC_TO_FPL

REF_HALF_LIFE_DAYS = 365.0
REF_K0 = 15.0            # matches of shrinkage toward the league rate
SET_HALF_LIFE_DAYS = 180.0
SET_K0 = 8.0
# share of a position's goals that come from set plays (corners, free kicks,
# throw-ins; penalties excluded), league-wide — the shape prior until a
# per-player shot log is available
POS_SET_SHARE = {"GK": 0.0, "DEF": 0.55, "MID": 0.20, "FWD": 0.28}


def fixture_map(conn) -> pd.DataFrame:
    """event_urn -> (season, fixture_id, home_id, away_id, kick)."""
    bm = pd.read_sql_query(
        "SELECT event_urn, season, home, away, kickoff_utc FROM acq_bbc_match "
        "WHERE kickoff_utc IS NOT NULL", conn)
    if bm.empty:
        return pd.DataFrame(columns=["event_urn", "season", "fixture_id", "home_id", "away_id", "kick"])
    teams = pd.read_sql_query("SELECT season, team_id, name FROM team", conn)
    key = {(r.season, r.name): int(r.team_id) for r in teams.itertuples()}
    bm["home_id"] = [key.get((s, BBC_TO_FPL.get(t, t))) for s, t in zip(bm["season"], bm["home"])]
    bm["away_id"] = [key.get((s, BBC_TO_FPL.get(t, t))) for s, t in zip(bm["season"], bm["away"])]
    bm["kick"] = pd.to_datetime(bm["kickoff_utc"], utc=True, errors="coerce")
    bm = bm.dropna(subset=["home_id", "away_id", "kick"])
    bm["day"] = bm["kick"].dt.floor("D")
    tm = pd.read_sql_query(
        "SELECT season, team_id AS home_id, fixture_id, kickoff_utc FROM team_match "
        "WHERE was_home=1 AND kickoff_utc IS NOT NULL", conn)
    fx = pd.read_sql_query(
        "SELECT season, team_h AS home_id, fixture_id, kickoff_utc FROM fixture "
        "WHERE kickoff_utc IS NOT NULL", conn)
    f = pd.concat([tm, fx], ignore_index=True).drop_duplicates(["season", "fixture_id"])
    f["day"] = pd.to_datetime(f["kickoff_utc"], utc=True, errors="coerce").dt.floor("D")
    f["home_id"] = pd.to_numeric(f["home_id"], errors="coerce")
    bm["home_id"] = bm["home_id"].astype(float)
    out = bm.merge(f[["season", "home_id", "day", "fixture_id"]], on=["season", "home_id", "day"], how="inner")
    return out[["event_urn", "season", "fixture_id", "home_id", "away_id", "kick"]].astype(
        {"fixture_id": int, "home_id": int, "away_id": int})


def _decayed_prior(df: pd.DataFrame, group: str, value: str, half_life: float,
                   k0: float, league: pd.Series | None = None) -> pd.Series:
    """Per row: decayed mean of ``value`` over the group's STRICTLY prior rows,
    shrunk with k0 pseudo-matches toward the league prior; NaN -> 1.0 ratio
    is left to the caller. Rows must be sorted by kick."""
    out = np.full(len(df), np.nan)
    w_sum = np.zeros(len(df)); v_sum = np.zeros(len(df))
    lam = np.log(2) / half_life
    # days as floats: divide by a Timedelta, never an integer view (E11c)
    days_all = ((df["kick"] - pd.Timestamp("2000-01-01", tz="UTC")) / pd.Timedelta(days=1)).to_numpy(float)
    for _, idx in df.groupby(group, sort=False).indices.items():
        idx = np.asarray(idx)
        ks = days_all[idx]; vs = df[value].to_numpy(float)[idx]
        for j in range(1, len(idx)):
            dt = ks[j] - ks[:j]
            w = np.exp(-lam * dt)
            ok = ~np.isnan(vs[:j])
            w_sum[idx[j]] = w[ok].sum(); v_sum[idx[j]] = (w[ok] * vs[:j][ok]).sum()
    return pd.Series(w_sum, index=df.index), pd.Series(v_sum, index=df.index)


def referee_factors(conn) -> pd.DataFrame:
    """(season, fixture_id, ref_factor): the referee's prior yellow rate
    relative to the league's, shrunk; only fixtures with a known referee."""
    off = pd.read_sql_query(
        "SELECT event_urn, official_urn FROM acq_bbc_official WHERE role='Referee'", conn)
    fm = fixture_map(conn)
    if off.empty or fm.empty:
        return pd.DataFrame(columns=["season", "fixture_id", "ref_factor"])
    yc = pd.read_sql_query(
        "SELECT season, fixture_id, SUM(COALESCE(yellow_cards,0)) AS yellows FROM player_gw "
        "GROUP BY season, fixture_id", conn)
    d = fm.merge(off, on="event_urn").merge(yc, on=["season", "fixture_id"], how="left")
    d = d.sort_values("kick").reset_index(drop=True)
    # league prior: expanding mean of yellows per match over all prior matches
    lg = d["yellows"].shift(1).expanding().mean().fillna(d["yellows"].mean())
    w, v = _decayed_prior(d, "official_urn", "yellows", REF_HALF_LIFE_DAYS, REF_K0)
    rate = (v + REF_K0 * lg) / (w + REF_K0)
    d["ref_factor"] = (rate / lg.replace(0, np.nan)).fillna(1.0).clip(0.6, 1.6)
    return d[["season", "fixture_id", "ref_factor"]]


def setplay_factors(conn) -> pd.DataFrame:
    """(season, fixture_id, team_id, set_def_rel, open_def_rel) — for the club
    ATTACKING in that fixture, the opponent's prior set-play / open-play
    share of xG conceded relative to the league's, shrunk."""
    st = pd.read_sql_query(
        "SELECT event_urn, team, side, xg_open, xg_set FROM acq_bbc_match_stats "
        "WHERE xg_open IS NOT NULL AND xg_set IS NOT NULL", conn)
    fm = fixture_map(conn)
    if st.empty or fm.empty:
        return pd.DataFrame(columns=["season", "fixture_id", "team_id", "set_def_rel", "open_def_rel"])
    d = st.merge(fm, on="event_urn")
    # the xG a club CONCEDES in the match is the other side's xG
    d["conceder"] = np.where(d["side"] == "home", d["away_id"], d["home_id"])
    d = d.sort_values("kick").reset_index(drop=True)
    d["set_share"] = d["xg_set"] / (d["xg_set"] + d["xg_open"]).replace(0, np.nan)
    lg = d["set_share"].shift(1).expanding().mean().fillna(d["set_share"].mean())
    w, v = _decayed_prior(d, "conceder", "set_share", SET_HALF_LIFE_DAYS, SET_K0)
    share = (v + SET_K0 * lg) / (w + SET_K0)
    d["set_def_rel"] = (share / lg).fillna(1.0)
    d["open_def_rel"] = ((1 - share) / (1 - lg)).fillna(1.0)
    # the row belongs to the ATTACKING club (the one whose xG this was)
    d["team_id"] = np.where(d["side"] == "home", d["home_id"], d["away_id"])
    return d[["season", "fixture_id", "team_id", "set_def_rel", "open_def_rel"]].astype(
        {"fixture_id": int, "team_id": int})


def factor_maps(conn, season: str, fixture_ids: list[int]) -> tuple[dict, dict]:
    """(fixture_id -> ref_factor, (fixture_id, team_id) -> (set_rel, open_rel))
    for one gameweek; empty dicts when the archive has nothing."""
    ref, sp = {}, {}
    try:
        r = referee_factors(conn)
        r = r[(r["season"] == season) & r["fixture_id"].isin(fixture_ids)]
        ref = dict(zip(r["fixture_id"].astype(int), r["ref_factor"].astype(float)))
    except Exception:      # noqa: BLE001
        ref = {}
    try:
        s = setplay_factors(conn)
        s = s[(s["season"] == season) & s["fixture_id"].isin(fixture_ids)]
        sp = {(int(a), int(b)): (float(c), float(d))
              for a, b, c, d in zip(s["fixture_id"], s["team_id"], s["set_def_rel"], s["open_def_rel"])}
    except Exception:      # noqa: BLE001
        sp = {}
    return ref, sp


# ---------------------------------------------------------------------------
# Round 22: the opponent's possession scales a player's DefCon crossing rate
# ---------------------------------------------------------------------------
POSS_HALF_LIFE_DAYS = 240.0
POSS_K0 = 5.0          # matches of shrinkage toward 50% possession
BETA_N0 = 500.0        # rows of shrinkage on the fitted slope toward 0


def possession_factors(conn, as_of: str) -> tuple[dict, dict]:
    """({(season, team_id): the club's prior possession, decayed and shrunk
    to 50, from matches strictly before ``as_of``}, {position: relative
    change in a player's DefCon crossing rate per point of OPPONENT
    possession, fitted within player on rows before ``as_of`` and shrunk
    toward zero}). Both point-in-time; keyed by club so an unplayed fixture
    (the live horizon) is priced from the `fixture` table's team ids."""
    st = pd.read_sql_query(
        "SELECT event_urn, side, possession FROM acq_bbc_match_stats WHERE possession IS NOT NULL", conn)
    fm = fixture_map(conn)
    if st.empty or fm.empty:
        return {}, {}
    d = st.merge(fm, on="event_urn")
    d["team_id"] = np.where(d["side"] == "home", d["home_id"], d["away_id"])
    cut = pd.Timestamp(as_of.replace("Z", "+00:00")) if as_of else None
    if cut is not None:
        d = d[d["kick"] < cut]
    if d.empty:
        return {}, {}
    lam = np.log(2) / POSS_HALF_LIFE_DAYS
    ref = cut if cut is not None else d["kick"].max()
    d = d.assign(age=((ref - d["kick"]) / pd.Timedelta(days=1)).astype(float))
    d["w"] = np.exp(-lam * d["age"].clip(lower=0))
    g = d.groupby(["season", "team_id"])
    prior = ((g.apply(lambda x: (x["w"] * x["possession"]).sum()) + POSS_K0 * 50.0)
             / (g["w"].sum() + POSS_K0))
    poss = {(str(k[0]), int(k[1])): float(v) for k, v in prior.items()}
    # the slope, within player-season, on rows before as_of (DefCon era only)
    pg = pd.read_sql_query(
        "SELECT pg.season, pg.fixture_id, pg.player_id, pg.opponent_id, pg.minutes, pg.defcon, "
        "pg.kickoff_utc, p.position FROM player_gw pg JOIN player p ON p.season=pg.season AND "
        "p.player_id=pg.player_id WHERE pg.defcon IS NOT NULL AND pg.minutes >= 60 "
        "AND p.position IN ('DEF','MID')", conn)
    betas: dict = {}
    if len(pg):
        pg["kick"] = pd.to_datetime(pg["kickoff_utc"], utc=True, errors="coerce")
        if cut is not None:
            pg = pg[pg["kick"] < cut]
        real = d[["season", "fixture_id", "team_id", "possession"]].rename(
            columns={"team_id": "opponent_id", "possession": "opp_poss"})
        pg = pg.merge(real, on=["season", "fixture_id", "opponent_id"])
        thr = {"DEF": 10, "MID": 12}
        pg["cross"] = (pg["defcon"] >= pg["position"].map(thr)).astype(float)
        for pos, gg in pg.groupby("position"):
            if len(gg) < 50:
                continue
            y = gg["cross"] - gg.groupby(["season", "player_id"])["cross"].transform("mean")
            x = gg["opp_poss"] - gg.groupby(["season", "player_id"])["opp_poss"].transform("mean")
            if x.std() == 0:
                continue
            slope = float((x * y).sum() / (x * x).sum())
            base = float(gg["cross"].mean())
            if base <= 0:
                continue
            betas[pos] = (slope / base) * len(gg) / (len(gg) + BETA_N0)
    return poss, betas


def defcon_factor_map(conn, season: str, fixtures: list, as_of: str) -> dict:
    """(fixture_id, team_id, position) -> multiplier on the DefCon crossing
    rate, 1 + beta_pos * (opponent prior possession - 50), clipped; built
    from the gameweek's fixtures (dicts with fixture_id, team_h, team_a)."""
    try:
        poss, betas = possession_factors(conn, as_of)
    except Exception:      # noqa: BLE001
        return {}
    if not poss or not betas:
        return {}
    out = {}
    for f in fixtures:
        for team, opp in ((f["team_h"], f["team_a"]), (f["team_a"], f["team_h"])):
            op = poss.get((season, int(opp)))
            if op is None:
                continue
            for pos, b in betas.items():
                out[(int(f["fixture_id"]), int(team), pos)] = float(np.clip(1.0 + b * (op - 50.0), 0.5, 1.8))
    return out


# ---------------------------------------------------------------------------
# Round 22b: the opponent's style reaches keepers (shot volume -> saves) and
# midfielders (box touches allowed -> xG); forwards failed the gate
# ---------------------------------------------------------------------------
STYLE_STATS = {"saves": "shots_on", "att": "box_allowed"}     # kind -> club prior stat
STYLE_POS = {"saves": ("GK",), "att": ("MID",)}


def _club_priors(conn, as_of: str, cols: tuple) -> pd.DataFrame:
    """Per (season, team_id): decayed prior of each stat (own, and what the
    club ALLOWED - the other side's), strictly before as_of, shrunk (k0=5)
    toward the club's own mean."""
    st = pd.read_sql_query(
        "SELECT event_urn, side, possession, shots, shots_on, touches_box, crosses FROM acq_bbc_match_stats", conn)
    fm = fixture_map(conn)
    if st.empty or fm.empty:
        return pd.DataFrame()
    d = st.merge(fm, on="event_urn")
    d["team_id"] = np.where(d["side"] == "home", d["home_id"], d["away_id"])
    other = d[["event_urn", "side", "shots", "shots_on", "touches_box", "crosses"]].copy()
    other["side"] = np.where(other["side"] == "home", "away", "home")
    other = other.rename(columns={"shots": "shots_allowed", "shots_on": "sot_allowed",
                                  "touches_box": "box_allowed", "crosses": "crosses_allowed"})
    d = d.merge(other, on=["event_urn", "side"])
    cut = pd.Timestamp(as_of.replace("Z", "+00:00"))
    d = d[d["kick"] < cut]
    if d.empty:
        return pd.DataFrame()
    lam = np.log(2) / POSS_HALF_LIFE_DAYS
    d = d.assign(w=np.exp(-lam * ((cut - d["kick"]) / pd.Timedelta(days=1)).clip(lower=0).astype(float)))
    rows = []
    for (season, tid), g in d.groupby(["season", "team_id"]):
        r = {"season": str(season), "team_id": int(tid)}
        for c in cols:
            v = g[c].astype(float)
            ok = v.notna()
            if ok.sum() == 0:
                r[c] = np.nan
                continue
            r[c] = float(((g.loc[ok, "w"] * v[ok]).sum() + POSS_K0 * v[ok].mean())
                         / (g.loc[ok, "w"].sum() + POSS_K0))
        rows.append(r)
    return pd.DataFrame(rows)


def style_factor_map(conn, season: str, fixtures: list, as_of: str) -> dict:
    """(fixture_id, team_id, kind) -> multiplier. kind "saves": the OPPONENT's
    prior shots on target relative to the league, times a slope fitted
    within keeper on rows before as_of; kind "att": the opponent's prior box
    touches allowed, likewise for midfielders' xG. Mean-preserving at the
    league average; 1.0 wherever the archive is silent."""
    try:
        pri = _club_priors(conn, as_of, ("shots_on", "box_allowed"))
    except Exception:      # noqa: BLE001
        return {}
    if pri.empty:
        return {}
    cur = pri[pri["season"] == season].set_index("team_id")
    if cur.empty:
        return {}
    league = {c: float(cur[c].mean()) for c in ("shots_on", "box_allowed")}
    cut = pd.Timestamp(as_of.replace("Z", "+00:00"))
    pg = pd.read_sql_query(
        "SELECT pg.season, pg.fixture_id, pg.player_id, pg.opponent_id, pg.minutes, pg.saves, pg.xg, "
        "pg.kickoff_utc, p.position FROM player_gw pg JOIN player p ON p.season=pg.season AND "
        "p.player_id=pg.player_id WHERE pg.minutes >= 60 AND p.position IN ('GK','MID')", conn)
    pg["kick"] = pd.to_datetime(pg["kickoff_utc"], utc=True, errors="coerce")
    pg = pg[pg["kick"] < cut]
    # the regressor is the opponent's season-level club prior as of as_of: what the
    # map applies at serve time, and cheap enough to refit every gameweek
    pri_all = pri.rename(columns={"team_id": "opponent_id"})
    pg = pg.merge(pri_all, on=["season", "opponent_id"], how="inner")
    slopes = {}
    for kind, stat in STYLE_STATS.items():
        pos = STYLE_POS[kind][0]
        g = pg[pg["position"] == pos].copy()
        g["y"] = (g["saves"] if kind == "saves" else pd.to_numeric(g["xg"], errors="coerce")) / g["minutes"] * 90.0
        g["x"] = np.log(g[stat].astype(float) / g.groupby("season")[stat].transform("mean"))
        g = g.dropna(subset=["x", "y"])
        if len(g) < 100:
            continue
        y = g["y"] - g.groupby(["season", "player_id"])["y"].transform("mean")
        x = g["x"] - g.groupby(["season", "player_id"])["x"].transform("mean")
        if x.std() == 0 or g["y"].mean() <= 0:
            continue
        slope = float((x * y).sum() / (x * x).sum()) / float(g["y"].mean())   # relative per log-unit
        slopes[kind] = slope * len(g) / (len(g) + BETA_N0)
    if not slopes:
        return {}
    out = {}
    for f in fixtures:
        for team, opp in ((f["team_h"], f["team_a"]), (f["team_a"], f["team_h"])):
            if int(opp) not in cur.index:
                continue
            for kind, stat in STYLE_STATS.items():
                if kind not in slopes or pd.isna(cur.loc[int(opp), stat]) or league[stat] <= 0:
                    continue
                rel = np.log(float(cur.loc[int(opp), stat]) / league[stat])
                out[(int(f["fixture_id"]), int(team), kind)] = float(np.clip(1.0 + slopes[kind] * rel, 0.5, 1.8))
    return out
