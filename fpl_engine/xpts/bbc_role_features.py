"""Round 19 block `bbcrole`: per-match role from BBC lineups, no Understat.

The shipped LINE_FEATURES (`role_is_am`, `role_is_dm`, `role_vs_fpl_line`)
carry the largest rank gain in this file and depend on Understat's per-match
role, a site whose robots.txt disallows everything. BBC's `match-lineups`
container gives, for every match since 2022-23, each starter's played
position (Goalkeeper/Defender/Midfielder/Forward) and his ROW in the
formation graphic — GK row 0 up to the most attacking row — which is a
finer, keyless statement of the same thing.

Identity: BBC player URNs are resolved to FPL `player.code` once per
club-season by name (full name, then a surname unique within that squad);
anything ambiguous stays unmapped and its features NaN. Features are built
from strictly prior matches (shifted), so they are point-in-time.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..pressers import BBC_TO_FPL

FEATURES = ["bbc_row_l1", "bbc_row_l5", "bbc_pos_l1", "bbc_pos_vs_fpl",
            "bbc_is_am", "bbc_is_dm"]
# Round 20 research family `bbcx`: the club captaincy (captains are rarely
# dropped) and slot competition — his share of the club's recent starts in
# his own formation slot (position label + row) and how many different
# players have held that slot lately. Env-gated until it clears a replay.
EXTRA = ["bbc_cap_l1", "bbc_slot_share_l5", "bbc_slot_rivals_l5"]
# BBC's played-position vocabulary, on one attacking axis so that "how far
# forward did he play last time" is a number; FPL's four labels sit on the
# same axis for the difference feature
POS_ORD = {"Goalkeeper": 0.0, "Defender": 1.0, "Wing Back": 1.5,
           "Defensive Midfielder": 2.0, "Midfielder": 2.5, "Attacking Midfielder": 3.0,
           "Forward": 4.0, "Striker": 4.0,
           "GK": 0.0, "DEF": 1.0, "MID": 2.5, "FWD": 4.0}


def _norm(s: str) -> str:
    """Lower-case ASCII letters only, accents transliterated (Magalhães ->
    magalhaes), so BBC's and FPL's spellings meet in the middle."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def player_map(conn) -> dict[str, int]:
    """BBC player URN -> FPL player.code, resolved per club-season by name."""
    try:
        lu = pd.read_sql_query(
            "SELECT l.player_urn, l.team, l.name_first, l.name_last, m.season "
            "FROM acq_bbc_lineup l JOIN acq_bbc_match m ON m.event_urn=l.event_urn "
            "GROUP BY l.player_urn, l.team, m.season", conn)
    except Exception:      # noqa: BLE001 - no BBC archive in this database
        return {}
    if lu.empty:
        return {}
    pl = pd.read_sql_query(
        "SELECT p.season, p.code, p.full_name, p.web_name, t.name team FROM player p "
        "JOIN team t ON t.season=p.season AND t.team_id=p.team_id "
        "WHERE p.position IN ('GK','DEF','MID','FWD')", conn)
    out: dict[str, int] = {}
    by_key: dict[tuple, pd.DataFrame] = {k: g for k, g in pl.groupby(["season", "team"])}
    for r in lu.itertuples():
        fpl_team = BBC_TO_FPL.get(r.team, r.team)
        g = by_key.get((r.season, fpl_team))
        if g is None or r.player_urn in out:
            continue
        full = _norm(f"{r.name_first or ''} {r.name_last or ''}")
        last = _norm(r.name_last or "")
        cand = g[g["full_name"].map(_norm) == full]
        if len(cand) != 1 and last:
            sur = g["full_name"].map(lambda x: _norm(x).split(" ")[-1] if x else "")
            cand = g[(sur == last) | (g["web_name"].map(_norm) == last)]
        if len(cand) != 1:
            # long legal names ("Caicedo Corozo", "Salah Hamed Mahrous Ghaly")
            # against FPL's short ones: the squad member sharing the most
            # distinctive name tokens, if that member is unique
            toks = {t for t in full.split() if len(t) >= 4}
            if toks:
                def overlap(row):
                    ft = set(_norm(row["full_name"]).split()) | set(_norm(row["web_name"]).split())
                    return len(toks & {t for t in ft if len(t) >= 4})
                sc = g.apply(overlap, axis=1)
                best = sc.max() if len(sc) else 0
                if best >= 1 and (sc == best).sum() == 1:
                    cand = g[sc == best]
        if len(cand) == 1:
            out[r.player_urn] = int(cand["code"].iloc[0])
    return out


def load(conn) -> pd.DataFrame:
    """One row per (player_code, kick): played position ordinal and normalised row."""
    empty = pd.DataFrame(columns=["player_code", "kick", "pos", "row", "is_am", "is_dm",
                                  "is_cap", "slot", "team_id", "season"])
    try:
        lu = pd.read_sql_query(
            "SELECT l.player_urn, l.event_urn, l.position, l.pitch_row, l.is_starter, "
            "l.is_captain, l.team, m.kickoff_utc, m.season FROM acq_bbc_lineup l "
            "JOIN acq_bbc_match m ON m.event_urn=l.event_urn WHERE l.is_starter=1", conn)
    except Exception:      # noqa: BLE001 - no BBC archive: every feature stays NaN
        return empty
    if lu.empty:
        return empty
    pm = player_map(conn)
    lu["player_code"] = lu["player_urn"].map(pm)
    lu = lu.dropna(subset=["player_code"])
    nrows = lu.groupby("event_urn")["pitch_row"].transform("max").replace(0, np.nan)
    lu["row"] = (lu["pitch_row"] / nrows).astype(float)
    lu["pos"] = lu["position"].map(POS_ORD).astype(float)
    lu["is_am"] = (lu["position"] == "Attacking Midfielder").astype(float)
    lu["is_dm"] = (lu["position"] == "Defensive Midfielder").astype(float)
    lu["is_cap"] = pd.to_numeric(lu["is_captain"], errors="coerce").fillna(0).astype(float)
    lu["slot"] = lu["position"].fillna("?") + "|" + lu["pitch_row"].astype(str)
    teams = pd.read_sql_query("SELECT season, team_id, name FROM team", conn)
    tkey = {(r.season, r.name): int(r.team_id) for r in teams.itertuples()}
    lu["team_id"] = [tkey.get((se, BBC_TO_FPL.get(t, t))) for se, t in zip(lu["season"], lu["team"])]
    lu["kick"] = pd.to_datetime(lu["kickoff_utc"], utc=True, errors="coerce")
    out = lu.dropna(subset=["kick"])[["player_code", "kick", "pos", "row", "is_am", "is_dm",
                                      "is_cap", "slot", "team_id", "season"]]
    out["player_code"] = out["player_code"].astype(int)
    return out.sort_values(["player_code", "kick"]).reset_index(drop=True)


def add_features(frame: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """Latest strictly-prior start per player, joined as-of (vectorised: the
    first version looped the 100k-row frame with a filter per row, which
    turned every projection build into minutes)."""
    frame = frame.copy()
    for f in FEATURES:
        frame[f] = np.nan
    if roles is None or roles.empty or frame.empty:
        return frame
    r = roles.sort_values("kick").copy()
    r["player_code"] = r["player_code"].astype(int)
    # rolling mean of the row over this and the previous four starts, so the
    # matched (latest prior) row carries "mean of the last five prior starts"
    r["row_l5"] = r.groupby("player_code")["row"].transform(
        lambda x: x.rolling(5, min_periods=1).mean())
    left = pd.DataFrame({
        "_i": np.arange(len(frame)),
        "player_code": pd.to_numeric(frame["player_code"], errors="coerce"),
        "_t": pd.to_datetime(frame["kick"], utc=True, errors="coerce") - pd.Timedelta(hours=6),
    }).dropna(subset=["player_code", "_t"])
    left["player_code"] = left["player_code"].astype(int)
    left = left.sort_values("_t")
    m = pd.merge_asof(left, r[["player_code", "kick", "row", "row_l5", "pos", "is_am", "is_dm"]],
                      left_on="_t", right_on="kick", by="player_code",
                      direction="backward", allow_exact_matches=False)
    m = m.set_index("_i")
    fpl_ord = frame["position"].map(POS_ORD).astype(float)
    frame.loc[m.index, "bbc_row_l1"] = m["row"].to_numpy()
    frame.loc[m.index, "bbc_row_l5"] = m["row_l5"].to_numpy()
    frame.loc[m.index, "bbc_pos_l1"] = m["pos"].to_numpy()
    frame.loc[m.index, "bbc_is_am"] = m["is_am"].to_numpy()
    frame.loc[m.index, "bbc_is_dm"] = m["is_dm"].to_numpy()
    frame["bbc_pos_vs_fpl"] = frame["bbc_pos_l1"] - fpl_ord
    return frame


def add_extra_features(frame: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    """The `bbcx` family: captaincy and slot competition, from prior matches."""
    frame = frame.copy()
    for f in EXTRA:
        frame[f] = np.nan
    if roles is None or roles.empty or frame.empty or "slot" not in roles.columns:
        return frame
    kick = pd.to_datetime(frame["kick"], utc=True, errors="coerce")
    codes = pd.to_numeric(frame["player_code"], errors="coerce")
    by_player = {k: g for k, g in roles.groupby("player_code")}
    # club-match -> {slot: [player codes]} for slot competition
    club = {}
    for (season, tid), g in roles.dropna(subset=["team_id"]).groupby(["season", "team_id"]):
        matches = []
        for k, gm in g.groupby("kick"):
            matches.append((k, dict(zip(gm["slot"], gm["player_code"]))))
        club[(season, int(tid))] = matches       # ascending by kick
    cap = np.full(len(frame), np.nan)
    share = np.full(len(frame), np.nan)
    rivals = np.full(len(frame), np.nan)
    seasons = frame["season"].to_numpy()
    teams = pd.to_numeric(frame["team_id"], errors="coerce").to_numpy()
    ks = kick.to_numpy(); cs = codes.to_numpy()
    for i in range(len(frame)):
        if cs[i] != cs[i] or pd.isna(ks[i]):
            continue
        t = pd.Timestamp(ks[i]) - pd.Timedelta(hours=6)
        g = by_player.get(int(cs[i]))
        if g is None:
            continue
        prior = g[g["kick"] < t]
        if not len(prior):
            continue
        cap[i] = prior["is_cap"].iloc[-1]
        last5 = prior.tail(5)
        modal = last5["slot"].mode().iloc[0]
        cm = club.get((seasons[i], int(teams[i]) if teams[i] == teams[i] else -1))
        if not cm:
            continue
        recent = [m for (k, m) in cm if k < t][-5:]
        if not recent:
            continue
        held = [m.get(modal) for m in recent]
        share[i] = sum(1 for h in held if h == int(cs[i])) / len(recent)
        rivals[i] = len({h for h in held if h is not None and h != int(cs[i])})
    frame["bbc_cap_l1"] = cap
    frame["bbc_slot_share_l5"] = share
    frame["bbc_slot_rivals_l5"] = rivals
    return frame
