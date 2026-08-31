"""The Tactics/Manager expert: an isolated feature family, built to be rejected.

The hypothesis is that the chain

    manager -> tactical system -> player role -> opponent -> FPL output

carries information the engine does not already hold. It is not assumed. This
module exists so the claim can be measured against the rest of the model rather
than argued, and every family is separable so the ablation can say *which* part
(if any) survives.

WHAT IT IS BUILT FROM, and why these two sources

  `tm_manager_spell`  Transfermarkt's staff history: every managerial spell at
      every club that has played a season in this database, with an appointment
      date and a departure date. The DATES are the point: a manager's name is a
      categorical with no history, but his appointment date makes "who was in
      charge on the day of this fixture, and for how long" a point-in-time fact.

  `understat_team_match`  PPDA (pressing intensity), deep completions (final-
      third entry), and xG for and against, per team per match, dated.

  `understat_player_match`  the ROLE a player actually occupied in a match —
      AMR, DMC, FWL — which is far richer than FPL's four-way label, and is the
      only free per-match role feed there is. A formation is reconstructed from
      the roles of the eleven who started.

WHAT IS NOT AVAILABLE, checked rather than assumed. StatsBomb's open data holds
only Premier League 2003/04 and 2015/16, so no event-level tactical source
overlaps a season this repo replays; there is no free feed of possession, field
tilt, crossing frequency or build-up style with per-match history. PPDA and deep
completions are the two style axes that actually exist here, and the rest of the
"playing style" list in the brief is not reachable — saying so is cheaper than
approximating it badly.

TWO LEAKS THAT WOULD BE EASY TO SHIP
  * `tm_manager_spell.matches` and `.ppg` are TODAY's career totals for the
    spell, not the figures as they stood at any past date. They are never read
    here.
  * A manager's style has to be aggregated from his matches STRICTLY BEFORE the
    fixture, not from the spell as a whole. Every aggregate below is built from
    a shifted expanding or rolling window for exactly that reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DAY = pd.Timedelta(days=1)

# Understat's per-match role -> the line he actually played on. FPL says MID
# for all of DMC, MC, AMC and AML; only 15% of "MID" minutes are really MC.
ROLE_BUCKET = {
    "GK": "GK",
    "DL": "DEF", "DC": "DEF", "DR": "DEF",
    "DML": "DM", "DMC": "DM", "DMR": "DM",
    "ML": "MID", "MC": "MID", "MR": "MID",
    "AML": "AM", "AMC": "AM", "AMR": "AM",
    "FWL": "FWD", "FW": "FWD", "FWR": "FWD",
}
BUCKETS = ["GK", "DEF", "DM", "MID", "AM", "FWD"]

TAC_MANAGER = ["mgr_days_in_post", "mgr_new_60d", "mgr_matches_in_post",
               "mgr_career_matches", "mgr_known"]
TAC_STYLE = ["tac_ppda_l5", "tac_deep_l5", "tac_deep_allowed_l5",
             "tac_ppda_allowed_l5", "tac_opp_ppda_l5", "tac_opp_deep_l5",
             "tac_ppda_gap"]
TAC_FORMATION = [f"form_{b.lower()}_l5" for b in BUCKETS[1:]] + [
    "form_entropy_l5", "form_changed_l1", "form_seen_l5"]
TAC_ROLE = ["role_slots_l5", "role_share_l5", "role_is_am", "role_is_dm",
            "role_vs_fpl_line"]
TAC_MGR_ROLE = ["mgr_role_share", "mgr_role_share_vs_league"]
TAC_MGR_OPP = ["mgr_ppda_vs_strong", "mgr_deep_vs_strong"]

FAMILIES = {"tac_manager": TAC_MANAGER, "tac_style": TAC_STYLE,
            "tac_formation": TAC_FORMATION, "tac_role": TAC_ROLE,
            "tac_mgr_role": TAC_MGR_ROLE, "tac_mgr_opp": TAC_MGR_OPP}
ALL = [f for fam in FAMILIES.values() for f in fam]

# where FPL's four-way label would put each line, so a role can be differenced
# against it (an FPL "MID" who is really a DM is the case that matters)
BUCKET_LINE = {"GK": 0.0, "DEF": 1.0, "DM": 2.0, "MID": 3.0, "AM": 4.0,
               "FWD": 5.0}
FPL_LINE = {"GK": 0.0, "DEF": 1.0, "MID": 3.0, "FWD": 5.0}


# ------------------------------------------------------------------ load --
def load(conn) -> dict:
    """Manager spells, team style series and per-match roles, on FPL keys."""
    out: dict[str, pd.DataFrame] = {}

    # --- Transfermarkt club id -> FPL team id, per season. Derived from the
    # squads rather than from club names: a name map is one alias away from
    # silently attaching a club's manager to a different club.
    club = pd.read_sql_query(
        "SELECT s.season, s.tm_club_id, p.team_id, COUNT(*) n "
        "FROM tm_squad s "
        "JOIN tm_player m ON m.tm_player_id = s.tm_player_id "
        "JOIN player p ON p.code = m.player_code AND p.season = s.season "
        "WHERE m.player_code IS NOT NULL "
        "GROUP BY s.season, s.tm_club_id, p.team_id", conn)
    if not club.empty:
        club = (club.sort_values("n", ascending=False)
                    .drop_duplicates(["season", "tm_club_id"]))
    out["club"] = club[["season", "tm_club_id", "team_id"]] if not club.empty \
        else pd.DataFrame(columns=["season", "tm_club_id", "team_id"])

    mgr = pd.read_sql_query(
        "SELECT tm_club_id, tm_manager_id, manager, appointed, left_date "
        "FROM tm_manager_spell", conn)
    if not mgr.empty:
        mgr["appointed_dt"] = pd.to_datetime(mgr["appointed"], errors="coerce",
                                             utc=True)
        mgr["left_dt"] = pd.to_datetime(mgr["left_date"], errors="coerce",
                                        utc=True).fillna(
            pd.Timestamp("2100-01-01", tz="UTC"))
        mgr = mgr.dropna(subset=["appointed_dt"])
    out["mgr"] = mgr

    # --- team style, on FPL team ids and real kickoffs
    style = pd.read_sql_query(
        "SELECT u.season, u.match_date, u.understat_match_id, "
        "       u.xg ux_xg, u.xga ux_xga, u.deep, "
        "       u.deep_allowed, u.ppda_att, u.ppda_def, "
        "       u.ppda_allowed_att, u.ppda_allowed_def, t.team_id "
        "FROM understat_team_match u JOIN team t "
        "ON t.season = u.season AND t.understat_name = u.understat_team", conn)
    if not style.empty:
        style["date"] = pd.to_datetime(style["match_date"], errors="coerce",
                                       utc=True)
        # PPDA is passes allowed per defensive action: LOW means heavy pressing
        style["ppda"] = (pd.to_numeric(style["ppda_att"], errors="coerce")
                         / pd.to_numeric(style["ppda_def"],
                                         errors="coerce").replace(0, np.nan))
        style["ppda_allowed"] = (
            pd.to_numeric(style["ppda_allowed_att"], errors="coerce")
            / pd.to_numeric(style["ppda_allowed_def"],
                            errors="coerce").replace(0, np.nan))
        style = style.dropna(subset=["date", "team_id"]).sort_values(
            ["team_id", "date"])
    out["style"] = style

    # --- per-match roles, on player_code and FPL team ids
    roles = pd.read_sql_query(
        "SELECT u.season, u.match_date, u.position, u.minutes, "
        "       p.code player_code, p.team_id, p.position fpl_position "
        "FROM understat_player_match u JOIN player p "
        "ON p.season = u.season AND p.understat_id = u.understat_id", conn)
    if not roles.empty:
        roles["date"] = pd.to_datetime(roles["match_date"], errors="coerce",
                                       utc=True)
        roles["bucket"] = roles["position"].map(ROLE_BUCKET)
        roles["minutes"] = pd.to_numeric(roles["minutes"], errors="coerce")
        # a bucketed role means he was on the pitch from the start; "Sub"
        # maps to nothing, which is what separates a starting XI from a squad
        roles["started"] = roles["bucket"].notna().astype(float)
        roles = roles.dropna(subset=["date", "player_code"])
    out["roles"] = roles
    return out


# -------------------------------------------------------------- builders --
def _team_match_style(style: pd.DataFrame) -> pd.DataFrame:
    """Trailing style per team-match, SHIFTED so a match never sees itself."""
    if style.empty:
        return style
    s = style.sort_values(["team_id", "date"]).copy()
    g = s.groupby("team_id", sort=False)
    for col, out in (("ppda", "tac_ppda_l5"), ("deep", "tac_deep_l5"),
                     ("deep_allowed", "tac_deep_allowed_l5"),
                     ("ppda_allowed", "tac_ppda_allowed_l5")):
        s[out] = g[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    return s[["team_id", "date", "tac_ppda_l5", "tac_deep_l5",
              "tac_deep_allowed_l5", "tac_ppda_allowed_l5"]]


def _team_match_shape(roles: pd.DataFrame) -> pd.DataFrame:
    """Formation per team-match, then its trailing mean and its stability.

    The shape is the count of STARTERS on each line — Understat names the role
    a player actually occupied, so this is the formation the manager picked,
    not the one a team sheet was labelled with.
    """
    if roles.empty:
        return pd.DataFrame()
    st = roles[roles["started"] > 0]
    if st.empty:
        return pd.DataFrame()
    shape = (st.groupby(["team_id", "date", "bucket"]).size()
               .unstack("bucket").reindex(columns=BUCKETS).fillna(0.0)
               .reset_index().sort_values(["team_id", "date"]))
    # Understat resolves ~65% of players, so a team-match shows about seven of
    # the eleven who started. Counts would therefore carry the RESOLUTION rate
    # as much as the formation, and the resolution rate varies by club and
    # season. Shares divide it out; `form_seen` keeps how much of the XI was
    # actually observed so the model can discount a thin sample.
    seen = shape[BUCKETS].sum(axis=1).replace(0, np.nan)
    for b in BUCKETS:
        shape[b] = shape[b] / seen
    shape["form_seen"] = seen
    # rolling cannot aggregate strings, so the shape is factorised to a code
    shape["shape_key"] = pd.factorize(
        (shape[BUCKETS[1:]].fillna(0) * seen.fillna(0).values[:, None])
        .round().astype(int).astype(str).agg("-".join, axis=1))[0]
    g = shape.groupby("team_id", sort=False)
    for b in BUCKETS[1:]:
        shape[f"form_{b.lower()}_l5"] = g[b].transform(
            lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    # how many DIFFERENT shapes in the last five: 1 is a manager who never
    # changes, 5 is one who never repeats
    shape["form_entropy_l5"] = g["shape_key"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).apply(
            lambda w: float(len(np.unique(w))), raw=True))
    shape["form_changed_l1"] = g["shape_key"].transform(
        lambda x: (x.shift(1) != x.shift(2)).astype(float))
    shape["form_seen_l5"] = g["form_seen"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    keep = ["team_id", "date"] + [f"form_{b.lower()}_l5" for b in BUCKETS[1:]] \
        + ["form_entropy_l5", "form_changed_l1", "form_seen_l5"]
    return shape[keep]


def _role_slots(roles: pd.DataFrame) -> pd.DataFrame:
    """Per player-match: how contested his own line is, and his share of it."""
    if roles.empty:
        return pd.DataFrame()
    st = roles[roles["started"] > 0]
    if st.empty:
        return pd.DataFrame()
    slots = (st.groupby(["team_id", "date", "bucket"]).size()
               .rename("slots").reset_index())
    mine = st[["player_code", "team_id", "date", "bucket"]].merge(
        slots, on=["team_id", "date", "bucket"], how="left")
    mine = mine.sort_values(["player_code", "date"])
    g = mine.groupby("player_code", sort=False)
    mine["role_slots_l5"] = g["slots"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    mine["_one"] = 1.0
    mine["role_share_l5"] = g["_one"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).sum()) / \
        g["slots"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    return mine[["player_code", "date", "bucket", "role_slots_l5",
                 "role_share_l5"]]


def _manager_at(frame: pd.DataFrame, kick: pd.Series, data: dict) -> pd.DataFrame:
    """Which manager was in charge, and for how long, at each kickoff."""
    out = pd.DataFrame(index=frame.index)
    for f in TAC_MANAGER:
        out[f] = np.nan
    mgr, club = data.get("mgr"), data.get("club")
    if mgr is None or mgr.empty or club is None or club.empty:
        return out
    key = frame[["season", "team_id"]].copy()
    key["_i"] = np.arange(len(frame))
    key["_t"] = kick.to_numpy()
    spells = club.merge(mgr, on="tm_club_id", how="inner")
    j = key.merge(spells, on=["season", "team_id"], how="left")
    j = j[(j["appointed_dt"] <= j["_t"]) & (j["left_dt"] > j["_t"])]
    # a caretaker overlapping a permanent appointment: the later start wins
    j = j.sort_values("appointed_dt").drop_duplicates("_i", keep="last")
    if j.empty:
        return out
    idx = frame.index[j["_i"].to_numpy()]
    days = (j["_t"] - j["appointed_dt"]) / DAY
    out.loc[idx, "mgr_days_in_post"] = np.clip(days.to_numpy(), 0, 5000)
    out.loc[idx, "mgr_new_60d"] = (days.to_numpy() < 60).astype(float)
    out.loc[idx, "mgr_known"] = 1.0
    out["mgr_known"] = out["mgr_known"].fillna(0.0)
    out.loc[idx, "_mid"] = j["tm_manager_id"].to_numpy()

    # matches under him, counted point-in-time from the team's own fixtures
    style = data.get("style")
    if style is not None and not style.empty:
        played = style[["team_id", "date"]].sort_values("date").copy()
        played["date"] = played["date"].dt.tz_localize(None)
        arr = {t: g["date"].to_numpy().astype("datetime64[ns]")
               for t, g in played.groupby("team_id", sort=False)}
        tids = frame["team_id"].to_numpy()
        ts = kick.to_numpy()
        appv = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
        pos = j["_i"].to_numpy()
        appv[pos] = pd.to_datetime(j["appointed_dt"]).dt.tz_localize(
            None).to_numpy().astype("datetime64[ns]")
        n_post = np.full(len(frame), np.nan)
        tsv = pd.to_datetime(kick).dt.tz_localize(None).to_numpy().astype(
            "datetime64[ns]")
        for i in range(len(frame)):
            a = arr.get(tids[i])
            if a is None or np.isnat(appv[i]) or np.isnat(tsv[i]):
                continue
            n_post[i] = float(((a >= appv[i]) & (a < tsv[i])).sum())
        out["mgr_matches_in_post"] = n_post
        # his whole career in this database, across clubs
        mid = out.get("_mid")
        if mid is not None:
            spell_by_mgr = spells.groupby("tm_manager_id")
            career = {}
            for m, g in spell_by_mgr:
                seg = []
                for _, r in g.iterrows():
                    a = arr.get(r["team_id"])
                    if a is None:
                        continue
                    ap = np.datetime64(r["appointed_dt"].tz_localize(None))
                    lv = np.datetime64(r["left_dt"].tz_localize(None))
                    seg.append(a[(a >= ap) & (a < lv)])
                career[m] = np.sort(np.concatenate(seg)) if seg else np.array(
                    [], dtype="datetime64[ns]")
            mv = mid.to_numpy()
            cm = np.full(len(frame), np.nan)
            for i in range(len(frame)):
                if mv[i] != mv[i] or np.isnat(tsv[i]):
                    continue
                a = career.get(mv[i])
                if a is None or not len(a):
                    cm[i] = 0.0
                    continue
                cm[i] = float(np.searchsorted(a, tsv[i], "left"))
            out["mgr_career_matches"] = cm
    return out


def _manager_style(frame: pd.DataFrame, kick: pd.Series, data: dict,
                   mid: pd.Series) -> pd.DataFrame:
    """Manager x opponent: does he press differently against good teams?

    Built from HIS OWN prior matches only, and expressed as a difference from
    his own overall level, so a manager at a strong club does not read as
    "presses more" simply because his club does.
    """
    out = pd.DataFrame(index=frame.index)
    for f in TAC_MGR_OPP:
        out[f] = np.nan
    style, mgrspell, club = data.get("style"), data.get("mgr"), data.get("club")
    if style is None or style.empty or mgrspell is None or mgrspell.empty:
        return out
    spells = club.merge(mgrspell, on="tm_club_id", how="inner")
    s = style.copy()
    # Trailing attacking strength of each club, point-in-time.
    s["own_strength"] = s.groupby("team_id")["ux_xg"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    # The OPPONENT's strength is the question — "does he change his approach
    # against good teams" is meaningless if `strength` is his own club's, which
    # would only rediscover that good clubs press. The two sides of a match
    # share an Understat match id, so the opponent is one self-join away.
    other = s[["season", "understat_match_id", "team_id", "own_strength"]].rename(
        columns={"team_id": "opp_id", "own_strength": "strength"})
    s = s.merge(other, on=["season", "understat_match_id"], how="left")
    s = s[s["team_id"] != s["opp_id"]]
    # attach the manager in charge of each historical team-match
    sp = spells[["team_id", "tm_manager_id", "appointed_dt", "left_dt"]]
    m = s.merge(sp, on="team_id", how="left")
    m = m[(m["appointed_dt"] <= m["date"]) & (m["left_dt"] > m["date"])]
    if m.empty:
        return out
    strong = m["strength"] >= m.groupby("season")["strength"].transform(
        lambda x: x.quantile(0.75))
    m = m.assign(_strong=strong.astype(float)).sort_values(
        ["tm_manager_id", "date"])
    g = m.groupby("tm_manager_id", sort=False)
    for col, name in (("ppda", "mgr_ppda_vs_strong"),
                      ("deep", "mgr_deep_vs_strong")):
        overall = g[col].transform(lambda x: x.shift(1).expanding().mean())
        vs = m[col].where(m["_strong"] > 0)
        vs_mean = vs.groupby(m["tm_manager_id"]).transform(
            lambda x: x.shift(1).expanding().mean())
        m[name] = vs_mean - overall
    latest = m.sort_values("date").drop_duplicates(
        ["tm_manager_id"], keep="last")[
        ["tm_manager_id"] + TAC_MGR_OPP]
    # NOTE: `latest` is his standing estimate as of his most recent match. The
    # per-row point-in-time version is the merge_asof below.
    m = m.sort_values("date")
    left = pd.DataFrame({"tm_manager_id": mid.to_numpy(),
                         "_t": kick.to_numpy()})
    left["_i"] = np.arange(len(left))
    left = left.dropna(subset=["tm_manager_id", "_t"]).sort_values("_t")
    if left.empty:
        return out
    left["tm_manager_id"] = left["tm_manager_id"].astype("int64")
    r = m[["tm_manager_id", "date"] + TAC_MGR_OPP].dropna(subset=["date"])
    j = pd.merge_asof(left, r.sort_values("date"), left_on="_t",
                      right_on="date", by="tm_manager_id",
                      allow_exact_matches=False)
    idx = frame.index[j["_i"].to_numpy()]
    for f in TAC_MGR_OPP:
        out.loc[idx, f] = pd.to_numeric(j[f], errors="coerce").to_numpy()
    return out


def _manager_role(frame: pd.DataFrame, kick: pd.Series, data: dict,
                  mid: pd.Series, bucket: pd.Series) -> pd.DataFrame:
    """Manager x player role: how much of his XI does he spend on this line?

    The share is taken over his own prior matches and then differenced against
    the league's share for the same line, so "Guardiola fields two number tens"
    reads as a manager effect rather than as a fact about number tens.
    """
    out = pd.DataFrame(index=frame.index)
    for f in TAC_MGR_ROLE:
        out[f] = np.nan
    roles, mgrspell, club = data.get("roles"), data.get("mgr"), data.get("club")
    if roles is None or roles.empty or mgrspell is None or mgrspell.empty:
        return out
    st = roles[roles["started"] > 0]
    if st.empty:
        return out
    spells = club.merge(mgrspell, on="tm_club_id", how="inner")
    shape = (st.groupby(["team_id", "date", "bucket"]).size()
               .rename("n").reset_index())
    tot = shape.groupby(["team_id", "date"])["n"].transform("sum")
    shape["share"] = shape["n"] / tot.replace(0, np.nan)
    sp = spells[["team_id", "tm_manager_id", "appointed_dt", "left_dt"]]
    m = shape.merge(sp, on="team_id", how="left")
    m = m[(m["appointed_dt"] <= m["date"]) & (m["left_dt"] > m["date"])]
    if m.empty:
        return out
    m = m.sort_values(["tm_manager_id", "bucket", "date"])
    m["mgr_role_share"] = m.groupby(["tm_manager_id", "bucket"],
                                    sort=False)["share"].transform(
        lambda x: x.shift(1).expanding().mean())
    league = m.sort_values("date").groupby("bucket")["share"].transform(
        lambda x: x.shift(1).expanding().mean())
    m["mgr_role_share_vs_league"] = m["mgr_role_share"] - league

    left = pd.DataFrame({"tm_manager_id": mid.to_numpy(),
                         "bucket": bucket.to_numpy(), "_t": kick.to_numpy()})
    left["_i"] = np.arange(len(left))
    left = left.dropna(subset=["tm_manager_id", "bucket", "_t"]).sort_values("_t")
    if left.empty:
        return out
    left["tm_manager_id"] = left["tm_manager_id"].astype("int64")
    r = m[["tm_manager_id", "bucket", "date"] + TAC_MGR_ROLE].sort_values("date")
    j = pd.merge_asof(left, r, left_on="_t", right_on="date",
                      by=["tm_manager_id", "bucket"], allow_exact_matches=False)
    idx = frame.index[j["_i"].to_numpy()]
    for f in TAC_MGR_ROLE:
        out.loc[idx, f] = pd.to_numeric(j[f], errors="coerce").to_numpy()
    return out


def _asof_by_team(frame: pd.DataFrame, kick: pd.Series, right: pd.DataFrame,
                  cols: list[str], by: str = "team_id",
                  left_key: str | None = None) -> pd.DataFrame:
    """Last row of `right` strictly before each kickoff, matched on one key."""
    out = pd.DataFrame(index=frame.index, columns=cols, dtype=float)
    if right is None or right.empty:
        return out
    left = pd.DataFrame({by: frame[left_key or by].to_numpy(),
                         "_t": kick.to_numpy()})
    left["_i"] = np.arange(len(left))
    left = left.dropna(subset=[by, "_t"]).sort_values("_t")
    if left.empty:
        return out
    left[by] = left[by].astype("int64")
    r = right.dropna(subset=["date"]).sort_values("date").copy()
    r[by] = pd.to_numeric(r[by], errors="coerce")
    r = r.dropna(subset=[by])
    r[by] = r[by].astype("int64")
    j = pd.merge_asof(left, r[[by, "date"] + cols], left_on="_t",
                      right_on="date", by=by, allow_exact_matches=False)
    idx = frame.index[j["_i"].to_numpy()]
    for c in cols:
        out.loc[idx, c] = pd.to_numeric(j[c], errors="coerce").to_numpy()
    return out


def add_features(frame: pd.DataFrame, data: dict, *,
                 opponents: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach every tactical family to a minutes-model frame.

    `frame` needs `season`, `player_code`, `team_id`, `position` and `kick`.
    `opponents` (season, fixture_id, team_id, opponent_id) supplies the other
    side, which the minutes frame does not carry.
    """
    out = frame.copy()
    for f in ALL:
        out[f] = np.nan
    if not data:
        return out
    kick = pd.to_datetime(out["kick"], errors="coerce", utc=True)

    # ---- manager identity and tenure
    mg = _manager_at(out, kick, data)
    mid = mg.pop("_mid") if "_mid" in mg.columns else pd.Series(
        np.nan, index=out.index)
    for f in TAC_MANAGER:
        out[f] = mg[f]

    # ---- team style, own and opponent
    st = _team_match_style(data.get("style", pd.DataFrame()))
    own = _asof_by_team(out, kick, st, ["tac_ppda_l5", "tac_deep_l5",
                                        "tac_deep_allowed_l5",
                                        "tac_ppda_allowed_l5"])
    for c in own.columns:
        out[c] = own[c]
    if opponents is not None and not opponents.empty:
        o = out.merge(opponents, on=["season", "fixture_id", "team_id"],
                      how="left")["opponent_id"]
        out["_opp"] = o.to_numpy()
        opp = _asof_by_team(out, kick, st, ["tac_ppda_l5", "tac_deep_l5"],
                            left_key="_opp")
        out["tac_opp_ppda_l5"] = opp["tac_ppda_l5"].to_numpy()
        out["tac_opp_deep_l5"] = opp["tac_deep_l5"].to_numpy()
        out = out.drop(columns=["_opp"])
    # a team that presses harder than its opponent is the one inviting
    # transitions; the level alone cannot say that
    out["tac_ppda_gap"] = out["tac_ppda_l5"] - out["tac_opp_ppda_l5"]

    # ---- formation
    shape = _team_match_shape(data.get("roles", pd.DataFrame()))
    cols = [f"form_{b.lower()}_l5" for b in BUCKETS[1:]] + [
        "form_entropy_l5", "form_changed_l1", "form_seen_l5"]
    sh = _asof_by_team(out, kick, shape, cols)
    for c in cols:
        out[c] = sh[c]

    # ---- the player's own line
    roles = data.get("roles", pd.DataFrame())
    bucket = pd.Series(pd.NA, index=out.index, dtype="object")
    if roles is not None and not roles.empty:
        slots = _role_slots(roles)
        left = pd.DataFrame({"player_code": out["player_code"].to_numpy(),
                             "_t": kick.to_numpy()})
        left["_i"] = np.arange(len(left))
        left = left.dropna(subset=["player_code", "_t"]).sort_values("_t")
        r = slots.dropna(subset=["date"]).sort_values("date")
        j = pd.merge_asof(left, r, left_on="_t", right_on="date",
                          by="player_code", allow_exact_matches=False)
        idx = out.index[j["_i"].to_numpy()]
        out.loc[idx, "role_slots_l5"] = pd.to_numeric(
            j["role_slots_l5"], errors="coerce").to_numpy()
        out.loc[idx, "role_share_l5"] = pd.to_numeric(
            j["role_share_l5"], errors="coerce").to_numpy()
        bucket.loc[idx] = j["bucket"].to_numpy()
        out["role_is_am"] = (bucket == "AM").astype(float)
        out["role_is_dm"] = (bucket == "DM").astype(float)
        line = bucket.map(BUCKET_LINE)
        out["role_vs_fpl_line"] = pd.to_numeric(line, errors="coerce") - \
            out["position"].map(FPL_LINE)

    # ---- the two interactions
    for f, v in _manager_role(out, kick, data, mid, bucket).items():
        out[f] = v
    for f, v in _manager_style(out, kick, data, mid).items():
        out[f] = v
    return out


def opponent_map(conn, seasons: list[str]) -> pd.DataFrame:
    """(season, fixture_id, team_id, opponent_id) from played and future rows.

    `team_match` covers what has happened and `fixture` what has not, and the
    tactics frame needs both — the gameweek being predicted has no team_match
    row at all.
    """
    q = ",".join("?" * len(seasons))
    played = pd.read_sql_query(
        f"SELECT season, fixture_id, team_id, opponent_id FROM team_match "
        f"WHERE season IN ({q})", conn, params=seasons)
    fx = pd.read_sql_query(
        f"SELECT season, fixture_id, team_h, team_a FROM fixture "
        f"WHERE season IN ({q})", conn, params=seasons)
    if not fx.empty:
        fx = pd.concat([
            fx.rename(columns={"team_h": "team_id", "team_a": "opponent_id"}),
            fx.rename(columns={"team_a": "team_id", "team_h": "opponent_id"}),
        ])[["season", "fixture_id", "team_id", "opponent_id"]]
    out = pd.concat([played, fx], ignore_index=True) if not fx.empty else played
    return out.dropna().drop_duplicates(["season", "fixture_id", "team_id"])
