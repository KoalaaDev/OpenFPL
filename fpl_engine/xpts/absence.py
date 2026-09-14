"""Known absences, point-in-time, from every dated source this database holds.

Three kinds, each knowable before the deadline of the fixture it covers:

  sus      a ban derived from the card log (`absence_features.suspensions`)
  inj      a Transfermarkt injury spell that began at least a day before
           kickoff and had not ended by it (`injury_features.spells`) — the
           same boundary `inj_currently_out` uses
  presser  a manager's "out" statement on BBC's Friday page for that
           gameweek, published before the fixture (`presser_obs`)
  fpl      FPL's own flag, from the availability change log
           (`acq_player_availability`, live season only): a spell of status
           injured/suspended/unavailable, or chance of playing <= 25%, from
           the change that opened it to the change that closed it — the
           serve-time source for what Transfermarkt supplies in a replay

`known_absences` is the long table for the minutes-model block (one row per
player-fixture absent); `known_out_ids` is the per-gameweek set the engine's
`spill` arm consumes. Which kinds count is `$FPL_ABSENCE_KINDS`
(comma-separated; default "sus", so Round 20's first arm stays reproducible).
"""
from __future__ import annotations

import os

import pandas as pd

KINDS = ("sus", "inj", "presser", "fpl")
COLS = ["season", "team_id", "player_code", "fixture_id", "kind"]


def kinds_from_env() -> tuple[str, ...]:
    raw = os.environ.get("FPL_ABSENCE_KINDS", "sus")
    ks = tuple(k.strip() for k in raw.split(",") if k.strip())
    bad = [k for k in ks if k not in KINDS]
    if bad:
        raise ValueError(f"unknown absence kind(s): {bad}")
    return ks


def _club_fixtures(conn) -> pd.DataFrame:
    """(season, team_id, fixture_id, kick, gw) for every club-match, replayed
    seasons from `team_match`, the live season from `fixture`."""
    tm = pd.read_sql_query(
        "SELECT season, team_id, fixture_id, kickoff_utc, gw FROM team_match "
        "WHERE kickoff_utc IS NOT NULL", conn)
    fx = pd.read_sql_query(
        "SELECT season, fixture_id, kickoff_utc, gw, team_h, team_a FROM fixture "
        "WHERE kickoff_utc IS NOT NULL", conn)
    live = pd.concat([
        fx.rename(columns={"team_h": "team_id"})[["season", "team_id", "fixture_id", "kickoff_utc", "gw"]],
        fx.rename(columns={"team_a": "team_id"})[["season", "team_id", "fixture_id", "kickoff_utc", "gw"]]])
    out = pd.concat([tm, live], ignore_index=True).drop_duplicates(["season", "team_id", "fixture_id"])
    out["kick"] = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
    return out.dropna(subset=["kick"])


def _players(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT season, player_id, code AS player_code, team_id FROM player "
                             "WHERE code IS NOT NULL", conn)


def known_absences(conn, kinds: tuple[str, ...] | None = None) -> pd.DataFrame:
    kinds = kinds or kinds_from_env()
    parts = []
    if "sus" in kinds:
        from . import absence_features as _af
        s = _af.load_suspensions(conn)
        if len(s):
            parts.append(s.assign(kind="sus")[COLS])
    if "inj" in kinds or "presser" in kinds:
        cf = _club_fixtures(conn)
        pl = _players(conn)
    if "inj" in kinds:
        from . import injury_features as _inj
        try:
            sp = _inj.spells(conn)
        except Exception:      # noqa: BLE001 - table absent
            sp = pd.DataFrame()
        if len(sp):
            sp = sp[["player_code", "from_dt", "until_dt"]].dropna(subset=["from_dt"])
            m = sp.merge(pl, on="player_code").merge(cf, on=["season", "team_id"])
            known = m["from_dt"] <= m["kick"] - pd.Timedelta(days=1)
            still = m["until_dt"].isna() | (m["until_dt"] >= m["kick"])
            m = m[known & still]
            parts.append(m.assign(kind="inj")[COLS].drop_duplicates())
    if "presser" in kinds:
        try:
            po = pd.read_sql_query(
                "SELECT season, gw, player_id, published_utc FROM presser_obs "
                "WHERE cls='out' AND source='presser'", conn)
        except Exception:      # noqa: BLE001
            po = pd.DataFrame()
        if len(po):
            po["pub"] = pd.to_datetime(po["published_utc"], utc=True, errors="coerce")
            m = po.merge(pl, on=["season", "player_id"]).merge(cf, on=["season", "team_id", "gw"])
            m = m[m["pub"] < m["kick"]]
            parts.append(m.assign(kind="presser")[COLS].drop_duplicates())
    if "fpl" in kinds:
        try:
            av = pd.read_sql_query(
                "SELECT season, player_id, observed_utc, status, chance_next "
                "FROM acq_player_availability ORDER BY player_id, observed_utc", conn)
        except Exception:      # noqa: BLE001
            av = pd.DataFrame()
        if len(av):
            if "inj" not in kinds and "presser" not in kinds:
                cf = _club_fixtures(conn)
                pl = _players(conn)
            av["t"] = pd.to_datetime(av["observed_utc"], utc=True, errors="coerce")
            ch = pd.to_numeric(av["chance_next"], errors="coerce")
            av["out"] = av["status"].isin(["i", "s", "u"]) | (ch <= 25)
            av = av.dropna(subset=["t"]).sort_values(["season", "player_id", "t"])
            av["t_next"] = av.groupby(["season", "player_id"])["t"].shift(-1)
            spells = av[av["out"]][["season", "player_id", "t", "t_next"]]
            m = spells.merge(pl, on=["season", "player_id"]).merge(cf, on=["season", "team_id"])
            known = m["t"] <= m["kick"] - pd.Timedelta(hours=12)
            still = m["t_next"].isna() | (m["t_next"] >= m["kick"])
            m = m[known & still]
            parts.append(m.assign(kind="fpl")[COLS].drop_duplicates())
    if not parts:
        return pd.DataFrame(columns=COLS)
    out = pd.concat(parts, ignore_index=True)
    for c in ("team_id", "player_code", "fixture_id"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["player_code", "fixture_id"]).astype(
        {"team_id": int, "player_code": int, "fixture_id": int})


def known_out_ids(conn, season: str, gw: int, as_of: str | None = None,
                  kinds: tuple[str, ...] | None = None) -> dict[int, str]:
    """player_id -> kind for players known absent from a fixture of ``gw``."""
    ab = known_absences(conn, kinds)
    if ab.empty:
        return {}
    fids = {int(r[0]) for r in conn.execute(
        "SELECT fixture_id FROM fixture WHERE season=? AND gw=?", (season, gw))}
    fids |= {int(r[0]) for r in conn.execute(
        "SELECT DISTINCT fixture_id FROM team_match WHERE season=? AND gw=?", (season, gw))}
    ab = ab[(ab["season"] == season) & ab["fixture_id"].isin(fids)]
    if ab.empty:
        return {}
    pl = _players(conn)
    pl = pl[pl["season"] == season]
    code_to_id = dict(zip(pl["player_code"].astype(int), pl["player_id"].astype(int)))
    out: dict[int, str] = {}
    for r in ab.itertuples():
        pid = code_to_id.get(int(r.player_code))
        if pid is not None:
            out.setdefault(pid, r.kind)
    return out
