"""Return-from-injury dynamics: the first mispriced pocket in the minutes model.

WHY THIS EXISTS. Eight attempts to sharpen expected minutes failed because the
model was already calibrated: where it said 55%, reality delivered 55%, and the
residual was the manager's own undecided choice (Round 8). This block targets
the one segment where that is measurably NOT true. On a player's first
appearance-opportunity after an ended injury spell, the shipped model says
**0.15 P(start) for a previously established starter and reality says 0.41** —
a +0.26 calibration gap replicated to the third decimal across two independent
seasons (+0.262 / +0.264), decaying along the ramp (+0.15 at the second
opportunity, +0.09, +0.05) and largest for short absences.

THE MECHANISM, which is why this is information the model cannot derive: its
trailing features see the injury as a string of zeros — indistinguishable from
being dropped — and nothing in the feature set says the absence was exogenous
and HAS ENDED. Transfermarkt dates the end of every spell; the player's
pre-injury role says what he returns to; his appearances since the return are
realised past. None of that is a re-arrangement of trailing minutes: trailing
minutes are exactly what the injury destroyed.

LEAKAGE RULES, each load-bearing:
  * ended spells only; an ongoing spell's until_date is Transfermarkt's
    forecast and is never read;
  * the spell end must predate the row's GAMEWEEK DEADLINE by a clear day —
    4.5% of spells carry an until_date on the return match day itself
    (back-filled), and those rows are excluded rather than trusted;
  * `ret_apps_since` counts appearances strictly before this kickoff;
  * pre-injury form uses fixtures strictly before the spell began;
  * the replacement's record uses only the absence window, which ends before
    the return window begins.

The k>=2 rows are additionally immune to any Transfermarkt dating quirk: "he
appeared last gameweek after a real absence" is realised history whatever the
spell's dates say, and the gap there (+0.14-0.16 for established starters) is
the part no dating artefact can manufacture.

Everything is NaN outside a return window. An absent observation is not a
negative one — coding the quiet weeks 0 would assert "not returning" about
every healthy player (the tac_line lesson, which cost 0.14 points per pick
before it was caught).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DAY = pd.Timedelta(days=1)

SOFT = ("hamstring", "muscle", "muscular", "calf", "groin", "thigh",
        "adductor", "strain")

FEATURES = ["ret_opp_index", "ret_apps_since", "ret_days_since_end",
            "ret_missed", "ret_spell_days", "ret_soft",
            "ret_pre_start_rate", "ret_pre_consec",
            "ret_repl_starts", "ret_repl_started_last"]

WINDOW_OPPS = 4          # the ramp is closed by the fourth opportunity
MIN_MISSED = 2           # one skipped match is rotation, not an absence


def spells(conn) -> pd.DataFrame:
    """Ended spells on the stable player code. Ongoing spells are not read."""
    df = pd.read_sql_query(
        "SELECT i.from_date, i.until_date, i.days, i.injury, m.player_code "
        "FROM tm_injury i JOIN tm_player m ON m.tm_player_id = i.tm_player_id "
        "WHERE m.player_code IS NOT NULL AND i.until_date IS NOT NULL", conn)
    if df.empty:
        return df
    df["from_dt"] = pd.to_datetime(df["from_date"], errors="coerce", utc=True)
    df["until_dt"] = pd.to_datetime(df["until_date"], errors="coerce", utc=True)
    df["spell_days"] = pd.to_numeric(df["days"], errors="coerce")
    df["soft"] = df["injury"].fillna("").str.lower().apply(
        lambda s: float(any(k in s for k in SOFT)))
    return df.dropna(subset=["from_dt", "until_dt", "player_code"])


def add_features(frame: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Attach the return block. Loops over SPELLS (~3k), never over rows."""
    out = frame.copy()
    for f in FEATURES:
        out[f] = np.nan
    if sp is None or sp.empty or not len(out):
        return out

    kick = pd.to_datetime(out["kick"], errors="coerce", utc=True)
    # the decision moment: the gameweek's first kickoff minus 90 minutes
    deadline = kick.groupby([out["season"], out["gw"]]).transform("min") \
        - pd.Timedelta(minutes=90)
    started = pd.to_numeric(out.get("started"), errors="coerce")
    played = pd.to_numeric(out.get("played"), errors="coerce")
    if played is None or played.isna().all():
        played = (pd.to_numeric(out.get("minutes"), errors="coerce") > 0
                  ).astype(float)
    played = played.fillna(0.0)
    started = started.fillna(0.0)

    # per-player sorted views, on tz-NAIVE datetime64 — pandas keeps tz-aware
    # stamps as object arrays, where numpy comparisons die
    kick_n = kick.dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    dl_n = deadline.dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    fill = np.datetime64("1970-01-01")
    kick_n = np.where(np.isnat(kick_n), fill, kick_n)
    order = np.lexsort([kick_n.astype("int64"), out["player_code"].to_numpy()])
    codes_sorted = out["player_code"].to_numpy()[order]
    kick_sorted = kick_n[order]
    started_sorted = started.to_numpy()[order]
    played_sorted = played.to_numpy()[order]
    dl_sorted = dl_n[order]
    idx_sorted = out.index.to_numpy()[order]

    bounds: dict = {}
    uniq, starts_at = np.unique(codes_sorted, return_index=True)
    for c, s0 in zip(uniq, starts_at):
        bounds[c] = s0
    ends = dict(zip(uniq, list(starts_at[1:]) + [len(codes_sorted)]))

    # team+position groups for the replacement question
    grp_key = list(zip(out["season"].to_numpy()[order],
                       out["team_id"].to_numpy()[order],
                       out["position"].to_numpy()[order]))
    team_pos: dict = {}
    for i, k in enumerate(grp_key):
        team_pos.setdefault(k, []).append(i)

    cols = {f: np.full(len(out), np.nan) for f in FEATURES}

    # ---- everything the spell loop needs, converted ONCE ------------------
    sp_codes = sp["player_code"].to_numpy()
    sp_from = sp["from_dt"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    sp_until = sp["until_dt"].dt.tz_localize(None).to_numpy().astype("datetime64[ns]")
    sp_days = pd.to_numeric(sp["spell_days"], errors="coerce").to_numpy()
    sp_soft = sp["soft"].to_numpy()

    seasons_sorted = out["season"].to_numpy()[order]
    teams_sorted = out["team_id"].to_numpy()[order]
    pos_sorted = out["position"].to_numpy()[order]

    # per (season, team, position, player): that player's sorted row positions
    by_group_player: dict = {}
    for i in range(len(codes_sorted)):
        by_group_player.setdefault(
            (seasons_sorted[i], teams_sorted[i], pos_sorted[i],
             codes_sorted[i]), []).append(i)
    group_members: dict = {}
    for (se, te, po, cc) in by_group_player:
        group_members.setdefault((se, te, po), set()).add(cc)
    played_prefix = {}      # per player-in-group: prefix sums for window maths
    for k, rows_ in by_group_player.items():
        arr = np.asarray(rows_)
        played_prefix[k] = (kick_sorted[arr], started_sorted[arr], arr)

    day24 = np.timedelta64(24, "h")
    one_day = np.timedelta64(1, "D")

    for si in range(len(sp_codes)):
        c = sp_codes[si]
        if c not in bounds:
            continue
        lo, hi = bounds[c], ends[c]
        pk = kick_sorted[lo:hi]
        frm, unt = sp_from[si], sp_until[si]
        f0 = int(np.searchsorted(pk, frm, "left"))
        u0 = int(np.searchsorted(pk, unt, "right"))
        missed = int((played_sorted[lo + f0:lo + u0] == 0).sum())
        if missed < MIN_MISSED:
            continue
        pre_s = started_sorted[max(lo, lo + f0 - 10):lo + f0]
        pre_rate = float(pre_s.mean()) if len(pre_s) >= 3 else np.nan
        consec = 0
        for v in started_sorted[lo:lo + f0][::-1]:
            if v == 1:
                consec += 1
            else:
                break

        # the stand-in, from precomputed per-player views of the position group
        anchor = lo + min(f0, hi - lo - 1)
        gk = (seasons_sorted[anchor], teams_sorted[anchor], pos_sorted[anchor])
        repl, repl_last = 0.0, 0.0
        for cc in group_members.get(gk, ()):  # ~5-9 teammates
            if cc == c:
                continue
            mk, ms, _ = played_prefix[(gk[0], gk[1], gk[2], cc)]
            a = int(np.searchsorted(mk, frm, "left"))
            b = int(np.searchsorted(mk, unt, "right"))
            if b <= a:
                continue
            pre_mate = ms[max(0, a - 8):a]
            if len(pre_mate) and float(pre_mate.mean()) >= 0.5:
                continue                     # a regular is not a stand-in
            n_started = float(ms[a:b].sum())
            if n_started > repl:
                repl = n_started
                repl_last = float(ms[b - 1])

        for k in range(WINDOW_OPPS):
            row = lo + u0 + k
            if row >= hi:
                break
            if not (unt < dl_sorted[row] - day24):
                continue
            apps = float(played_sorted[lo + u0:row].sum())
            pos_i = int(idx_sorted[row])
            vals = (
                float(k + 1), apps,
                float(np.clip((kick_sorted[row] - unt) / one_day, 0, 45)),
                float(min(missed, 15)),
                float(np.clip(sp_days[si], 0, 200)) if sp_days[si] == sp_days[si] else np.nan,
                float(sp_soft[si]), pre_rate, float(min(consec, 6)),
                float(np.clip(repl, 0, 8)), repl_last,
            )
            for f, v in zip(FEATURES, vals):
                cols[f][pos_i] = v

    for f in FEATURES:
        out[f] = cols[f]
    return out
