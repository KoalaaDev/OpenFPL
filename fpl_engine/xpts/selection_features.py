"""Round 21 block `sel`: the manager's SELECTION process, from what he did last time.

Every feature is built from strictly prior rows and answers a question the
trailing-form features do not ask:

  prev_unused        he was an unused substitute in his club's last match
                     (on the BBC bench, no minutes) — distinct from not being
                     in the squad at all (prev_absent)
  prev_sub_mins      minutes he played as a substitute last match (0 if not)
  prev_hooked        he started last match and came off before the hour
  self_returning     this is his first match after a KNOWN absence (ban or
                     dated injury); pos/team_regular_returning count the
                     regulars (>=3 starts in the 5 before their absence)
                     coming back at his position / club, excluding himself
  home_start_bias    his trailing start rate at home minus away (last 10 of
                     each), NaN with under 3 of either
  yellows_todate     league yellows this season before this match;
                     ban_threat = one booking from a ban (4 by matchday 19,
                     9 by matchday 32)
  pts_l1, gi_l1      his points and goal involvements in his last match, and
  pts_l1_vs_avg      that against his mean over the ten before
  team_ga_l1 etc.    the club's last result: goals against / for, a loss, a
                     clean sheet, and how much of that XI it had kept from
                     the match before (xi_kept_l1)
  consec_x_congest   consecutive starts x matches in the last 14 days

Gate: `research/selection_gate.py`. `CORE` (the gate's survivors) is SHIPPED
into `minutes_model.FEATURES` as `SELECTION_FEATURES` (Round 21); the rest of
`FEATURES` stays a research block (`$FPL_MINUTES_EXTRA=sel`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["prev_unused", "prev_absent", "prev_sub_mins", "prev_hooked",
            "self_returning", "pos_regular_returning", "team_regular_returning",
            "home_start_bias", "yellows_todate", "ban_threat",
            "pts_l1", "gi_l1", "pts_l1_vs_avg",
            "team_ga_l1", "team_gf_l1", "team_lost_l1", "team_cs_l1", "xi_kept_l1",
            "consec_x_congest",
            # Round 21b refinements (owner): the SPECIFIC stand-in who held the
            # returning regular's formation slot, and the club's last margin
            "replacement_for_returning", "gd_l1"]
# the gate's survivors (research/selection_gate.py): a returning player and the
# regulars returning around him, last match's bench status, last match's
# returns as a selection signal, and bookings to date
CORE = ["self_returning", "pos_regular_returning", "team_regular_returning",
        "prev_unused", "prev_absent", "pts_l1", "gi_l1", "pts_l1_vs_avg", "yellows_todate"]


def load(conn) -> dict:
    """Everything the block needs, keyed for point-in-time joins."""
    out: dict = {}
    out["pg"] = pd.read_sql_query(
        "SELECT season, player_code, fixture_id, total_points, goals_scored, assists, "
        "yellow_cards, starts, minutes FROM player_gw", conn)
    out["tm"] = pd.read_sql_query(
        "SELECT season, team_id, fixture_id, kickoff_utc, goals_for, goals_against FROM team_match "
        "WHERE kickoff_utc IS NOT NULL", conn)
    out["fx"] = pd.read_sql_query(
        "SELECT season, fixture_id, kickoff_utc, team_h, team_a FROM fixture "
        "WHERE kickoff_utc IS NOT NULL", conn)
    # BBC squad membership (bench included) resolved to player_code
    try:
        from . import bbc_context as _bc, bbc_role_features as _br
        fm = _bc.fixture_map(conn)
        lu = pd.read_sql_query(
            "SELECT event_urn, player_urn, is_starter, mins, position, pitch_row FROM acq_bbc_lineup", conn)
        pm = _br.player_map(conn)
        lu["player_code"] = lu["player_urn"].map(pm)
        lu = lu.dropna(subset=["player_code"]).merge(fm[["event_urn", "season", "fixture_id"]], on="event_urn")
        lu["player_code"] = lu["player_code"].astype(int)
        lu["slot"] = lu["position"].fillna("?") + "|" + lu["pitch_row"].astype(str)
        out["squad"] = lu[["season", "player_code", "fixture_id", "is_starter", "mins", "slot"]].drop_duplicates(
            ["season", "player_code", "fixture_id"])
    except Exception:      # noqa: BLE001 - archive absent
        out["squad"] = pd.DataFrame(columns=["season", "player_code", "fixture_id", "is_starter", "mins", "slot"])
    try:
        from . import absence as _ab
        # replayed seasons: bans + dated injuries; the live season adds FPL's
        # own flag from the change log, which is what the deadline knows
        out["absent"] = _ab.known_absences(conn, ("sus", "inj", "fpl"))
    except Exception:      # noqa: BLE001
        out["absent"] = pd.DataFrame(columns=["season", "team_id", "player_code", "fixture_id", "kind"])
    return out


def _club_sequence(data: dict) -> pd.DataFrame:
    """(season, team_id, fixture_id, kick, seq) — every club match, played
    (team_match) or scheduled (fixture), ordered."""
    tm = data["tm"][["season", "team_id", "fixture_id", "kickoff_utc"]]
    fx = data["fx"]
    live = pd.concat([fx.rename(columns={"team_h": "team_id"})[["season", "team_id", "fixture_id", "kickoff_utc"]],
                      fx.rename(columns={"team_a": "team_id"})[["season", "team_id", "fixture_id", "kickoff_utc"]]])
    seq = pd.concat([tm, live], ignore_index=True).drop_duplicates(["season", "team_id", "fixture_id"])
    seq["kick"] = pd.to_datetime(seq["kickoff_utc"], utc=True, errors="coerce")
    seq = seq.dropna(subset=["kick"]).sort_values(["season", "team_id", "kick"])
    seq["seq"] = seq.groupby(["season", "team_id"]).cumcount()
    return seq[["season", "team_id", "fixture_id", "kick", "seq"]]


def _team_last(data: dict, seq: pd.DataFrame) -> pd.DataFrame:
    """Per club-fixture: the club's PREVIOUS match's result and XI continuity."""
    tm = data["tm"].merge(seq[["season", "team_id", "fixture_id", "seq"]], on=["season", "team_id", "fixture_id"])
    pg = data["pg"][["season", "fixture_id", "player_code", "starts"]]
    # XI per club-fixture from starts; team via the club sequence
    xi = pg[pg["starts"].fillna(0) > 0].merge(seq[["season", "team_id", "fixture_id", "seq"]],
                                              on=["season", "fixture_id"])
    xi_sets = xi.groupby(["season", "team_id", "seq"])["player_code"].apply(set)
    rows = []
    for (season, team), g in tm.sort_values("seq").groupby(["season", "team_id"]):
        prev_set = None
        for r in g.itertuples():
            s = xi_sets.get((season, team, r.seq))
            kept = (len(s & prev_set) / 11.0) if (s and prev_set) else np.nan
            rows.append((season, team, r.seq, r.goals_for, r.goals_against, kept))
            prev_set = s if s else prev_set
    res = pd.DataFrame(rows, columns=["season", "team_id", "seq", "gf", "ga", "xi_kept"])
    res["seq"] = res["seq"] + 1            # these describe the NEXT club match's "last"
    return res


def add_features(frame: pd.DataFrame, data: dict) -> pd.DataFrame:
    frame = frame.copy()
    for f in FEATURES:
        frame[f] = np.nan
    if frame.empty or data is None:
        return frame
    key = ["season", "player_code", "fixture_id"]
    f = frame[["season", "player_code", "fixture_id", "team_id", "kick", "started", "minutes",
               "was_home", "position", "consec_starts", "team_matches_14d"]].copy()
    f["_i"] = np.arange(len(f))
    for c in ("player_code", "fixture_id", "team_id"):
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.sort_values(["player_code", "kick"])
    g = f.groupby("player_code", sort=False)

    # ---- his previous fixture: status, minutes, points -------------------
    pg = data["pg"].copy()
    for c in ("player_code", "fixture_id"):
        pg[c] = pd.to_numeric(pg[c], errors="coerce")
    sq = data["squad"].copy()
    for c in ("player_code", "fixture_id"):
        sq[c] = pd.to_numeric(sq[c], errors="coerce")
    f["prev_season"] = g["season"].shift(1)
    f["prev_fid"] = g["fixture_id"].shift(1)
    prev = f[["prev_season", "prev_fid", "player_code"]].rename(
        columns={"prev_season": "season", "prev_fid": "fixture_id"})
    prev = prev.merge(pg[key + ["total_points", "goals_scored", "assists", "starts", "minutes"]],
                      on=key, how="left").merge(sq, on=key, how="left")
    prev.index = f.index
    started_prev = prev["starts"].fillna(0) > 0
    mins_prev = prev["minutes"].fillna(0)
    in_squad = prev["is_starter"].notna()
    has_bbc = prev["fixture_id"].notna() & in_squad.groupby(f["season"]).transform("any")
    f["prev_unused"] = np.where(has_bbc, (in_squad & (mins_prev == 0)).astype(float), np.nan)
    f["prev_absent"] = np.where(has_bbc, ((~in_squad) & prev["fixture_id"].notna()).astype(float), np.nan)
    f["prev_sub_mins"] = np.where(started_prev, 0.0, mins_prev)
    f["prev_hooked"] = (started_prev & (mins_prev < 60) & (mins_prev > 0)).astype(float)
    f["pts_l1"] = prev["total_points"]
    f["gi_l1"] = prev["goals_scored"].fillna(0) + prev["assists"].fillna(0)
    own = f[["player_code", "fixture_id", "season"]].merge(pg[key + ["total_points"]], on=key, how="left")
    own.index = f.index
    f["_pts"] = own["total_points"]
    avg10 = f.groupby("player_code", sort=False)["_pts"].transform(
        lambda s: s.shift(2).rolling(10, min_periods=3).mean())
    f["pts_l1_vs_avg"] = f["pts_l1"] - avg10

    # ---- home / away start bias ------------------------------------------
    home = pd.to_numeric(f["was_home"], errors="coerce")
    st = pd.to_numeric(f["started"], errors="coerce")
    f["_sh"] = st.where(home == 1)
    f["_sa"] = st.where(home == 0)
    g = f.groupby("player_code", sort=False)
    rh = g["_sh"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).apply(
        lambda x: np.nan if np.isnan(x).sum() > len(x) - 3 else np.nanmean(x), raw=True))
    ra = g["_sa"].transform(lambda s: s.shift(1).rolling(20, min_periods=1).apply(
        lambda x: np.nan if np.isnan(x).sum() > len(x) - 3 else np.nanmean(x), raw=True))
    f["home_start_bias"] = (rh - ra) * np.where(home == 1, 1.0, np.where(home == 0, -1.0, np.nan))

    # ---- yellows to date and the ban threat -------------------------------
    yc = f[["season", "player_code", "fixture_id"]].merge(pg[key + ["yellow_cards"]], on=key, how="left")
    yc.index = f.index
    f["_yc"] = yc["yellow_cards"].fillna(0)
    f["yellows_todate"] = f.groupby(["player_code", "season"], sort=False)["_yc"].transform(
        lambda s: s.shift(1).cumsum()).fillna(0)
    seq = _club_sequence(data)
    f = f.merge(seq[["season", "team_id", "fixture_id", "seq"]], on=["season", "team_id", "fixture_id"], how="left")
    md = f["seq"].fillna(0) + 1
    f["ban_threat"] = (((f["yellows_todate"] == 4) & (md <= 19)) | ((f["yellows_todate"] == 9) & (md <= 32))).astype(float)

    # ---- team shocks from the club's previous match -----------------------
    tl = _team_last(data, seq)
    f = f.merge(tl, on=["season", "team_id", "seq"], how="left")
    f["team_ga_l1"] = f["ga"]
    f["team_gf_l1"] = f["gf"]
    f["gd_l1"] = f["gf"] - f["ga"]
    f["team_lost_l1"] = (f["gf"] < f["ga"]).astype(float).where(f["gf"].notna())
    f["team_cs_l1"] = (f["ga"] == 0).astype(float).where(f["ga"].notna())
    f["xi_kept_l1"] = f["xi_kept"]

    # ---- returning regulars -----------------------------------------------
    ab = data["absent"]
    f["self_returning"] = 0.0
    f["pos_regular_returning"] = 0.0
    f["team_regular_returning"] = 0.0
    f["replacement_for_returning"] = 0.0
    if ab is not None and len(ab):
        a = ab.merge(seq[["season", "team_id", "fixture_id", "seq"]], on=["season", "team_id", "fixture_id"])
        a = a.sort_values(["season", "team_id", "player_code", "seq"])
        # a spell = consecutive club fixtures absent; the return fixture is seq+1 after its last one
        a["_gap"] = a.groupby(["season", "team_id", "player_code"])["seq"].diff().fillna(99)
        a["_spell"] = (a["_gap"] > 1).cumsum()
        spells = a.groupby(["season", "team_id", "player_code", "_spell"]).agg(first=("seq", "min"), last=("seq", "max")).reset_index()
        # regular before the spell: >=3 starts in the 5 club fixtures before it began
        starts = pg[["season", "player_code", "fixture_id", "starts"]].merge(
            seq[["season", "team_id", "fixture_id", "seq"]], on=["season", "fixture_id"])
        sidx = starts.set_index(["season", "team_id", "player_code", "seq"])["starts"].fillna(0)
        reg = []
        for r in spells.itertuples():
            n = sum(sidx.get((r.season, r.team_id, r.player_code, s), 0) for s in range(r.first - 5, r.first))
            reg.append(n >= 3)
        spells["regular"] = reg
        spells["ret_seq"] = spells["last"] + 1
        ret = spells[["season", "team_id", "player_code", "ret_seq", "regular"]].rename(columns={"ret_seq": "seq"})
        f = f.merge(ret.assign(self_returning=1.0)[["season", "team_id", "player_code", "seq", "self_returning"]],
                    on=["season", "team_id", "player_code", "seq"], how="left", suffixes=("_x", ""))
        f["self_returning"] = f["self_returning"].fillna(0.0)
        # positions of the returning regulars from the frame itself
        posmap = f[["season", "player_code", "position"]].drop_duplicates(["season", "player_code"])
        rr = ret[ret["regular"]].merge(posmap, on=["season", "player_code"], how="left")
        by_pos = rr.groupby(["season", "team_id", "seq", "position"]).size().rename("_npos").reset_index()
        by_team = rr.groupby(["season", "team_id", "seq"]).size().rename("_nteam").reset_index()
        f = f.merge(by_pos, on=["season", "team_id", "seq", "position"], how="left").merge(
            by_team, on=["season", "team_id", "seq"], how="left")
        self_reg = f[["season", "team_id", "player_code", "seq"]].merge(
            rr.assign(_self=1.0)[["season", "team_id", "player_code", "seq", "_self"]],
            on=["season", "team_id", "player_code", "seq"], how="left")["_self"].fillna(0).to_numpy()
        f["pos_regular_returning"] = (f["_npos"].fillna(0) - self_reg).clip(lower=0)
        f["team_regular_returning"] = (f["_nteam"].fillna(0) - self_reg).clip(lower=0)
        # the specific stand-in: whoever started in the returning regular's
        # modal pre-absence formation slot in the LAST match of the absence
        sq = data["squad"]
        if len(sq) and "slot" in sq.columns:
            st = sq[sq["is_starter"].fillna(0) > 0].merge(
                seq[["season", "team_id", "fixture_id", "seq"]], on=["season", "fixture_id"])
            slot_of = st.set_index(["season", "team_id", "player_code", "seq"])["slot"].sort_index()
            holders = st.set_index(["season", "team_id", "seq", "slot"])["player_code"].sort_index()
            rep_rows = []
            for r in rr.itertuples():
                first = spells.loc[(spells.season == r.season) & (spells.team_id == r.team_id)
                                   & (spells.player_code == r.player_code) & (spells.ret_seq == r.seq), "first"]
                if not len(first):
                    continue
                first = int(first.iloc[0])
                before = [slot_of.get((r.season, r.team_id, r.player_code, q)) for q in range(first - 5, first)]
                before = [b for b in before if b is not None]
                if not before:
                    continue
                modal = max(set(before), key=before.count)
                h = holders.get((r.season, r.team_id, r.seq - 1, modal))
                if h is None:
                    continue
                for code in (h if hasattr(h, "__iter__") and not isinstance(h, (str, bytes)) else [h]):
                    if int(code) != int(r.player_code):
                        rep_rows.append((r.season, r.team_id, int(code), r.seq))
            if rep_rows:
                rep = pd.DataFrame(rep_rows, columns=["season", "team_id", "player_code", "seq"]).drop_duplicates()
                rep["_rep"] = 1.0
                f = f.merge(rep, on=["season", "team_id", "player_code", "seq"], how="left")
                f["replacement_for_returning"] = f["_rep"].fillna(0.0)

    f["consec_x_congest"] = (pd.to_numeric(f["consec_starts"], errors="coerce").fillna(0)
                             * pd.to_numeric(f["team_matches_14d"], errors="coerce").fillna(0))
    f = f.set_index("_i").sort_index()
    for c in FEATURES:
        frame[c] = f[c].to_numpy()
    return frame
