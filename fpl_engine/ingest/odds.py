"""Match-odds ingestion into ``match_odds``.

Two free sources, one table:

* **football-data.co.uk** — historical (and in-season) EPL odds CSVs, no key
  needed. Used for backtests and to keep past gameweeks populated. The
  early-snapshot market-average columns (``AvgH``…) are preferred over the
  closing ones (``AvgCH``…): they are collected days before kickoff, which is
  the honest point-in-time signal for a prediction made before the
  gameweek's first kickoff.
* **The Odds API** (the-odds-api.com) — upcoming-fixture odds for the live
  season, key from ``$ODDS_API_KEY``. One request covers every EPL fixture
  (h2h + totals ⇒ 2 credits of the free 500/month).

Rows are keyed (season, fixture_id) and upserted idempotently. Team-name
resolution fails loud: an unmapped name raises instead of silently dropping
a match (CLAUDE.md principle #5).
"""
from __future__ import annotations

import csv
import io
import re
import json
import os
import urllib.request

from .. import db, progress
from ..http import get_text
from ..xpts import odds_model

# Club names differ per source and the promoted clubs change every summer, so
# names are resolved against the season's *actual* FPL teams rather than a
# frozen list. Only genuinely irregular renderings need an explicit override —
# the ones a token match cannot reach (abbreviations, nicknames).
NAME_OVERRIDES = {
    # football-data.co.uk
    "man united": "Man Utd",
    "tottenham": "Spurs",
    "nott'm forest": "Nott'm Forest",
    # The Odds API
    "manchester united": "Man Utd",
    "tottenham hotspur": "Spurs",
    "nottingham forest": "Nott'm Forest",
    "wolverhampton wanderers": "Wolves",
    "brighton and hove albion": "Brighton",
}
_STOPWORDS = {"fc", "afc", "the"}


def _tokens(name: str) -> list[str]:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower().replace("&", " and "))
    return [t for t in s.split() if t not in _STOPWORDS]


def _token_match(a: list[str], b: list[str]) -> bool:
    """True when the shorter name is a positional prefix of the longer.

    Matches "Newcastle" to "Newcastle United" and "Man City" to "Manchester
    City", while keeping "Man Utd"/"Manchester City" apart (utd vs city).
    """
    if not a or not b:
        return False
    if len(a) > len(b):
        a, b = b, a
    return all(b[i].startswith(a[i]) or a[i].startswith(b[i])
               for i in range(len(a)))


def resolve_team(name: str, fpl_names: list[str]) -> str:
    """Map a source's club name onto this season's FPL team name.

    Fails loud on a miss or an ambiguous match — silently dropping a fixture
    would leave that match without odds and nobody would notice
    (CLAUDE.md principle #5).
    """
    over = NAME_OVERRIDES.get((name or "").strip().lower())
    if over and over in fpl_names:
        return over
    exact = [f for f in fpl_names if f.lower() == (name or "").strip().lower()]
    if exact:
        return exact[0]
    toks = _tokens(name)
    hits = [f for f in fpl_names if _token_match(toks, _tokens(f))]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"odds ingest: cannot resolve club {name!r} to an FPL team "
        f"({'ambiguous: ' + ', '.join(hits) if hits else 'no match'}) — "
        "add it to NAME_OVERRIDES")


FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def _season_code(season: str) -> str:                 # "2025-26" -> "2526"
    return season[2:4] + season[5:7]


def _fd_date_iso(d: str) -> str | None:               # "16/08/2025" -> ISO date
    try:
        dd, mm, yy = d.split("/")
        if len(yy) == 2:
            yy = "20" + yy
        return f"{yy}-{mm}-{dd}"
    except (ValueError, AttributeError):
        return None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _team_ids(conn, season: str) -> dict[str, int]:
    return {r["name"]: int(r["team_id"]) for r in conn.execute(
        "SELECT name, team_id FROM team WHERE season=?", (season,))}


def _fixture_by_date_home(conn, season: str) -> dict[tuple[str, int], int]:
    """(kickoff date, home team_id) -> fixture_id, from team_match."""
    return {(r["kickoff_utc"][:10], int(r["team_id"])): int(r["fixture_id"])
            for r in conn.execute(
                "SELECT kickoff_utc, team_id, fixture_id FROM team_match "
                "WHERE season=? AND was_home=1 AND kickoff_utc IS NOT NULL",
                (season,))}


def _row(season, fixture_id, source, date, home_id, away_id,
         oh, od, oa, o_over, o_under) -> dict | None:
    p = odds_model.demargin(oh, od, oa)
    if not p:
        return None
    tot = odds_model.demargin(o_over, o_under)
    p_over = tot[0] if tot else None
    lam_h, lam_a = odds_model.implied_rates(p[0], p[1], p[2], p_over)
    return {"season": season, "fixture_id": fixture_id, "source": source,
            "kickoff_date": date, "home_id": home_id, "away_id": away_id,
            "p_home": round(p[0], 4), "p_draw": round(p[1], 4),
            "p_away": round(p[2], 4),
            "p_over25": round(p_over, 4) if p_over is not None else None,
            "lam_home": lam_h, "lam_away": lam_a}


def ingest_football_data(conn, seasons: list[str], *,
                         use_cache: bool = True) -> dict:
    """Load football-data.co.uk odds for ``seasons`` into match_odds."""
    total, unmatched = 0, []
    for season in seasons:
        txt = get_text(FD_URL.format(code=_season_code(season)),
                       use_cache=use_cache)
        rows = list(csv.DictReader(io.StringIO(txt.lstrip("﻿"))))
        ids = _team_ids(conn, season)
        fpl_names = list(ids)
        fx = _fixture_by_date_home(conn, season)
        out = []
        for m in rows:
            if not m.get("HomeTeam") or not m.get("AwayTeam"):
                continue
            hname = resolve_team(m["HomeTeam"], fpl_names)
            aname = resolve_team(m["AwayTeam"], fpl_names)
            date = _fd_date_iso(m.get("Date"))
            fid = fx.get((date, ids[hname]))
            if fid is None:              # future match: no team_match row yet
                unmatched.append((season, date, hname))
                continue
            # early-snapshot market average, then Bet365, then Pinnacle
            oh = _num(m.get("AvgH")) or _num(m.get("B365H")) or _num(m.get("PSH"))
            od = _num(m.get("AvgD")) or _num(m.get("B365D")) or _num(m.get("PSD"))
            oa = _num(m.get("AvgA")) or _num(m.get("B365A")) or _num(m.get("PSA"))
            o_over = (_num(m.get("Avg>2.5")) or _num(m.get("B365>2.5"))
                      or _num(m.get("P>2.5")))
            o_under = (_num(m.get("Avg<2.5")) or _num(m.get("B365<2.5"))
                       or _num(m.get("P<2.5")))
            r = _row(season, fid, "football-data", date,
                     ids[hname], ids[aname], oh, od, oa, o_over, o_under)
            if r:
                out.append(r)
        db.upsert(conn, "match_odds", out)
        total += len(out)
        progress.log(f"    odds {season}: {len(out)}/{len(rows)} matches")
    return {"rows": total, "unmatched": len(unmatched)}


def ingest_odds_api(conn, season: str, *, api_key: str | None = None,
                    timeout: int = 30) -> dict:
    """Pull upcoming-fixture odds from The Odds API into match_odds.

    Median odds across bookmakers; totals restricted to the 2.5 line. Needs
    ``$ODDS_API_KEY`` (free tier: one call = 2 credits of 500/month).
    """
    key = api_key or os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set — get a free key at "
                           "the-odds-api.com and export it")
    url = ("https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
           f"?apiKey={key}&regions=eu&markets=h2h,totals&oddsFormat=decimal")
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "fpl-engine/0.1"}),
            timeout=timeout) as r:
        events = json.loads(r.read().decode("utf-8"))

    ids = _team_ids(conn, season)
    fpl_names = list(ids)
    fixtures = {}                        # (date, home_id) -> fixture_id
    for r in conn.execute(
            "SELECT kickoff_utc, team_h, fixture_id FROM fixture "
            "WHERE season=? AND kickoff_utc IS NOT NULL", (season,)):
        fixtures[(r["kickoff_utc"][:10], int(r["team_h"]))] = int(r["fixture_id"])

    def median(xs):
        xs = sorted(x for x in xs if x)
        return xs[len(xs) // 2] if xs else None

    out, skipped = [], []
    for ev in events:
        hn, an = ev.get("home_team"), ev.get("away_team")
        try:
            h = resolve_team(hn, fpl_names)
            a = resolve_team(an, fpl_names)
        except ValueError:
            skipped.append(hn)           # club not in this FPL season at all
            continue
        oh, od, oa, o_over, o_under = [], [], [], [], []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] == "h2h":
                    prices = {o["name"]: o["price"] for o in mk["outcomes"]}
                    oh.append(prices.get(hn)); od.append(prices.get("Draw"))
                    oa.append(prices.get(an))
                elif mk["key"] == "totals":
                    for o in mk["outcomes"]:
                        if abs(float(o.get("point") or 0) - 2.5) < 1e-9:
                            (o_over if o["name"] == "Over" else o_under).append(
                                o["price"])
        date = (ev.get("commence_time") or "")[:10]
        fid = fixtures.get((date, ids[h]))
        if fid is None:
            skipped.append(f"{hn} {date}")
            continue
        r = _row(season, fid, "odds-api", date, ids[h], ids[a],
                 median(oh), median(od), median(oa),
                 median(o_over), median(o_under))
        if r:
            out.append(r)
    db.upsert(conn, "match_odds", out)
    progress.log(f"    odds-api: {len(out)} fixtures ({len(skipped)} skipped)")
    return {"rows": len(out), "skipped": skipped}
