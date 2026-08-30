"""Point-in-time Transfermarkt feature families for the minutes model.

Transfermarkt carries context the FPL feed does not: what a player is worth on
an open market, when he last changed clubs and for how much, how old he is, and
who else at his club plays his position. The question this module exists to
answer is whether any of that survives the repo's own bar, which log-loss alone
has never been.

WHY THESE AND NOT THE OBVIOUS ONES. The standing rule from four independent
confirmations (Understat rates, the set-piece decomposition, game state,
substitute productivity) is that *any effect measured in realised outcomes is
already absorbed by estimates fitted to realised outcomes*. Transfermarkt's
appearance and minutes tables are exactly that — a second copy of `player_gw` —
so they are deliberately not ingested. What is kept is the part that is
**exogenous**: valuations, transfers, and dates of birth are not derived from
the trailing minutes the model already has.

FAMILIES

  TM_PLAYER    age, height, foot, and Transfermarkt's detailed position
               (Defensive Midfield vs Attacking Midfield, where FPL says MID)
  TM_TRANSFER  days since his last completed move, the fee, whether he arrived
               from outside the league, loan status, moves in the last 3 years
  TM_MV        market value as of the deadline, its 180/365-day growth, its
               distance from his own peak, and his rank inside the club's
               depth chart at his position
  TM_SQUAD     competition for the shirt: how many team-mates at his position
               are valued near him, and whether one of them has just arrived

POINT-IN-TIME DISCIPLINE. `tm_transfer` and `tm_market_value` date every row,
so both are filtered strictly before the fixture's kickoff exactly like the
injury spells — `merge_asof(..., allow_exact_matches=False)` is the whole
mechanism. The one family that cannot be filtered is TM_PLAYER's detailed
position: Transfermarkt stores one current role per player and serves it on
every historical page, so a player who converted from winger to full-back in
2025 reads as a full-back in 2023. Date of birth, height and foot are immutable
and carry no such risk. The role feature is therefore kept in its own family so
it can be dropped on its own, and it is not claimed as backtest-clean.

IDENTITY. Everything joins on `player.code`, never on `player_id`. FPL
reassigns element ids every season — measured on this database, **99.7% of ids
point to a different footballer one season later** — so a Transfermarkt player
resolved against the current squad and joined to an old season on `player_id`
lands on somebody else, silently and with a full history attached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TM_PLAYER = ["tm_age", "tm_height_cm", "tm_foot_left", "tm_role_line",
             "tm_role_wide", "tm_role_vs_fpl"]
TM_TRANSFER = ["tm_days_since_move", "tm_new_signing", "tm_fee_log",
               "tm_fee_known", "tm_moves_3y", "tm_from_abroad", "tm_on_loan"]
TM_MV = ["tm_mv_log", "tm_mv_growth_365", "tm_mv_growth_180",
         "tm_mv_from_peak", "tm_mv_known"]
TM_SQUAD = ["tm_mv_rank_pos", "tm_mv_share_pos", "tm_rivals_near",
            "tm_new_rival_90d"]

FAMILIES = {"tm_player": TM_PLAYER, "tm_transfer": TM_TRANSFER,
            "tm_mv": TM_MV, "tm_squad": TM_SQUAD}
ALL = TM_PLAYER + TM_TRANSFER + TM_MV + TM_SQUAD

# Transfermarkt's detailed position mapped onto a single ordinal "how far up
# the pitch does he start", plus whether he starts wide. Two numbers rather
# than a dozen indicators: the tree only needs the ordering, and a sparse
# one-hot over sixteen roles is mostly empty columns.
ROLE_LINE = {
    "goalkeeper": 0,
    "centre-back": 1, "left-back": 2, "right-back": 2,
    "defensive midfield": 3, "central midfield": 4,
    "attacking midfield": 5, "left midfield": 5, "right midfield": 5,
    "left winger": 6, "right winger": 6,
    "second striker": 7, "centre-forward": 7,
}
ROLE_WIDE = {"left-back", "right-back", "left winger", "right winger",
             "left midfield", "right midfield"}
# where FPL's four-way label would put each line, so the two can be differenced
FPL_LINE = {"GK": 0.0, "DEF": 1.5, "MID": 4.5, "FWD": 7.0}


# ------------------------------------------------------------------ load --
def load(conn) -> dict[str, pd.DataFrame]:
    """The three Transfermarkt tables, joined onto the stable `player.code`."""
    ident = pd.read_sql_query(
        "SELECT tm_player_id, player_code FROM tm_player "
        "WHERE player_code IS NOT NULL", conn)
    out: dict[str, pd.DataFrame] = {"ident": ident}
    if ident.empty:
        return {"ident": ident, "attr": pd.DataFrame(), "mv": pd.DataFrame(),
                "tr": pd.DataFrame()}

    attr = pd.read_sql_query(
        "SELECT tm_player_id, dob, height_cm, foot, detail_position "
        "FROM tm_squad WHERE dob IS NOT NULL", conn)
    # one row per player: the attributes are immutable, so the most complete
    # observation wins rather than the most recent season's
    attr = (attr.sort_values("detail_position")
                .groupby("tm_player_id", as_index=False).first()
                .merge(ident, on="tm_player_id"))
    attr["dob_dt"] = pd.to_datetime(attr["dob"], errors="coerce", utc=True)
    role = attr["detail_position"].fillna("").str.strip().str.lower()
    attr["tm_role_line"] = role.map(ROLE_LINE)
    attr["tm_role_wide"] = role.isin(ROLE_WIDE).astype(float)
    attr["tm_foot_left"] = (attr["foot"].fillna("") == "left").astype(float)

    mv = pd.read_sql_query(
        "SELECT tm_player_id, value_date, value_eur FROM tm_market_value "
        "WHERE value_eur IS NOT NULL", conn).merge(ident, on="tm_player_id")
    if not mv.empty:
        mv["mv_dt"] = pd.to_datetime(mv["value_date"], errors="coerce", utc=True)
        mv = mv.dropna(subset=["mv_dt"]).sort_values(["player_code", "mv_dt"])
        mv["mv_peak"] = mv.groupby("player_code")["value_eur"].cummax()

    tr = pd.read_sql_query(
        "SELECT tm_player_id, transfer_date, from_club_id, fee_eur, fee_text "
        "FROM tm_transfer", conn).merge(ident, on="tm_player_id")
    if not tr.empty:
        tr["tr_dt"] = pd.to_datetime(tr["transfer_date"], errors="coerce",
                                     utc=True)
        tr = tr.dropna(subset=["tr_dt"]).sort_values(["player_code", "tr_dt"])
        txt = tr["fee_text"].fillna("").str.lower()
        tr["is_loan"] = txt.str.contains("loan").astype(float)
    return {"ident": ident, "attr": attr, "mv": mv, "tr": tr}


def pl_club_ids(conn) -> set[int]:
    """Transfermarkt club ids that have played a Premier League season here."""
    return {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT tm_club_id FROM tm_squad WHERE tm_club_id IS NOT NULL")}


# -------------------------------------------------------------- features --
def _asof(frame: pd.DataFrame, right: pd.DataFrame, left_time: pd.Series,
          right_time: str, cols: list[str], suffix: str = "") -> pd.DataFrame:
    """Last `right` row strictly before `left_time`, per player_code."""
    left = pd.DataFrame({"player_code": frame["player_code"].to_numpy(),
                         "_t": left_time.to_numpy()})
    left["_i"] = np.arange(len(left))
    left = left.dropna(subset=["_t"]).sort_values("_t")
    r = right.dropna(subset=[right_time]).sort_values(right_time)
    if left.empty or r.empty:
        return pd.DataFrame(index=frame.index,
                            columns=[c + suffix for c in cols], dtype=float)
    m = pd.merge_asof(left, r[["player_code", right_time] + cols],
                      left_on="_t", right_on=right_time, by="player_code",
                      allow_exact_matches=False)
    out = pd.DataFrame(index=frame.index, columns=[c + suffix for c in cols],
                       dtype=float)
    for c in cols:
        out.loc[frame.index[m["_i"].to_numpy()], c + suffix] = \
            pd.to_numeric(m[c], errors="coerce").to_numpy()
    return out


def add_features(frame: pd.DataFrame, data: dict, *,
                 pl_clubs: set[int] | None = None) -> pd.DataFrame:
    """Attach every Transfermarkt family to a minutes-model frame.

    `frame` needs `player_code`, `kick`, and — for the squad family — `season`,
    `gw`, `team_id` and `position`. Missing Transfermarkt coverage leaves NaN,
    which is what XGBoost wants: a player nobody has valued is a different case
    from one valued at zero.
    """
    out = frame.copy()
    for f in ALL:
        out[f] = np.nan
    if not data or data.get("ident") is None or data["ident"].empty:
        return out
    kick = pd.to_datetime(out["kick"], errors="coerce", utc=True)

    # ---- TM_PLAYER: immutable attributes, plus the current detailed role
    attr = data.get("attr")
    if attr is not None and not attr.empty:
        a = attr.set_index("player_code")
        idx = out["player_code"]
        dob = pd.to_datetime(idx.map(a["dob_dt"]), utc=True, errors="coerce")
        out["tm_age"] = (kick - dob).dt.days / 365.25
        out["tm_height_cm"] = pd.to_numeric(idx.map(a["height_cm"]),
                                            errors="coerce")
        out["tm_foot_left"] = pd.to_numeric(idx.map(a["tm_foot_left"]),
                                            errors="coerce")
        out["tm_role_line"] = pd.to_numeric(idx.map(a["tm_role_line"]),
                                            errors="coerce")
        out["tm_role_wide"] = pd.to_numeric(idx.map(a["tm_role_wide"]),
                                            errors="coerce")
        # a full-back FPL calls a midfielder, or a winger it calls a forward
        out["tm_role_vs_fpl"] = out["tm_role_line"] - out["position"].map(FPL_LINE)

    # ---- TM_MV: valuations strictly before kickoff, and their trajectory
    mv = data.get("mv")
    if mv is not None and not mv.empty:
        now = _asof(out, mv, kick, "mv_dt", ["value_eur", "mv_peak"])
        v = now["value_eur"]
        out["tm_mv_log"] = np.log1p(v)
        out["tm_mv_known"] = v.notna().astype(float)
        out["tm_mv_from_peak"] = v / now["mv_peak"].replace(0, np.nan)
        for days, col in ((365, "tm_mv_growth_365"), (180, "tm_mv_growth_180")):
            past = _asof(out, mv, kick - pd.Timedelta(days=days), "mv_dt",
                         ["value_eur"], suffix=f"_{days}")
            out[col] = np.log1p(v) - np.log1p(past[f"value_eur_{days}"])

    # ---- TM_TRANSFER: his last completed move before kickoff
    tr = data.get("tr")
    if tr is not None and not tr.empty:
        last = _asof(out, tr, kick, "tr_dt",
                     ["fee_eur", "from_club_id", "is_loan"])
        # merge_asof cannot carry the matched timestamp through the numeric
        # path above, so the date comes back on its own pass
        t2 = tr.assign(_ts=tr["tr_dt"].astype("int64") / 86_400e9)
        when = _asof(out, t2, kick, "tr_dt", ["_ts"])
        out["tm_days_since_move"] = (
            kick.astype("int64") / 86_400e9 - when["_ts"]).clip(0, 3650)
        out["tm_new_signing"] = (out["tm_days_since_move"] < 120).astype(float)
        out.loc[out["tm_days_since_move"].isna(), "tm_new_signing"] = np.nan
        out["tm_fee_log"] = np.log1p(last["fee_eur"])
        out["tm_fee_known"] = last["fee_eur"].notna().astype(float)
        out["tm_on_loan"] = last["is_loan"]
        if pl_clubs:
            src = last["from_club_id"]
            out["tm_from_abroad"] = np.where(
                src.isna(), np.nan,
                (~src.fillna(-1).astype(int).isin(pl_clubs)).astype(float))
        counts = _moves_in_window(out, tr, kick, days=1095)
        out["tm_moves_3y"] = counts

    # ---- TM_SQUAD: the depth chart, priced by the market rather than by FPL
    if "team_id" in out.columns and out["tm_mv_log"].notna().any():
        grp = ["season", "gw", "team_id", "position"]
        mvv = np.expm1(out["tm_mv_log"])
        out["tm_mv_rank_pos"] = mvv.groupby(
            [out[c] for c in grp]).rank(ascending=False, method="min")
        tot = mvv.groupby([out[c] for c in grp]).transform("sum")
        out["tm_mv_share_pos"] = mvv / tot.replace(0, np.nan)
        best = mvv.groupby([out[c] for c in grp]).transform("max")
        # team-mates at his position valued within 25% of him: the men who
        # actually threaten the shirt, as opposed to squad filler
        near = (mvv >= 0.75 * best).astype(float)
        out["tm_rivals_near"] = near.groupby(
            [out[c] for c in grp]).transform("sum") - near
        # a rival who has just walked in the door: the signal exists BEFORE
        # the incumbent starts losing minutes, which is the whole point of it
        fresh = ((out["tm_days_since_move"] < 90) & (mvv >= 0.75 * best)
                 ).astype(float)
        out["tm_new_rival_90d"] = fresh.groupby(
            [out[c] for c in grp]).transform("sum") - fresh
    return out


def _moves_in_window(frame: pd.DataFrame, tr: pd.DataFrame, kick: pd.Series,
                     days: int) -> pd.Series:
    """How many completed moves fall in the `days` before each kickoff."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    by = {c: g["tr_dt"].to_numpy() for c, g in tr.groupby("player_code")}
    codes = frame["player_code"].to_numpy()
    ks = kick.to_numpy()
    window = np.timedelta64(days, "D")
    vals = np.full(len(frame), np.nan)
    for i in range(len(frame)):
        arr = by.get(codes[i])
        if arr is None or ks[i] != ks[i]:
            continue
        vals[i] = float(((arr < ks[i]) & (arr >= ks[i] - window)).sum())
    out[:] = vals
    return out
