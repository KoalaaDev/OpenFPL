"""Polymarket prediction-market prices for Premier League fixtures.

Free, keyless, and `robots.txt` carries no disallow rules. Coverage is real:
18 fixtures a gameweek with 1-cent spreads and seven-figure liquidity on the
match markets.

**It is deliberately NOT wired into the model, and the reason is measured.**
Every Polymarket EPL market is TEAM level — match result, exact score, halftime,
first team to score, corners. There is no anytime-goalscorer or assist market,
so nothing here reaches a player except through the fixture attack scaler, and
that channel has been measured to be second order: `ODDS_WEIGHT` at 0 / 0.5 /
0.85 / 1.0 is indistinguishable (<=0.001 spearman_played, <=0.02 top-30). We
already carry bookmaker prices for the same fixtures, including Pinnacle, so a
second team-level source is redundant with something already worth ~nothing at
the player level.

What it IS worth is showing. A prediction market disagreeing with a sportsbook
is genuinely interesting to a human making a captaincy call, even when it moves
no model output — the same treatment the price model gets: reported next to the
recommendation, never inside the objective.

Quotes therefore live in their own table. `match_odds` is read by
`odds_model.fixture_odds_map`, which takes every row for a fixture and lets the
last one win, so writing here would silently change lambda depending on row
order — a change with no backtest behind it.
"""
from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone

GAMMA = "https://gamma-api.polymarket.com"
UA = "OpenFPL (personal FPL research; contact via github)"
TIMEOUT = 30
SOURCE = "polymarket"

# Polymarket writes full legal club names; FPL writes short ones.
NAME_FIX = {
    "manchester city": "Man City",
    "manchester united": "Man Utd",
    "tottenham hotspur": "Spurs",
    "newcastle united": "Newcastle",
    "wolverhampton wanderers": "Wolves",
    "brighton & hove albion": "Brighton",
    "nottingham forest": "Nott'm Forest",
    "west ham united": "West Ham",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "ipswich town": "Ipswich",
    "sheffield united": "Sheffield Utd",
    "coventry city": "Coventry",
    "hull city": "Hull",
    "afc bournemouth": "Bournemouth",
    "sunderland": "Sunderland",
}


def _clean(name: str) -> str:
    """'Crystal Palace FC' -> 'crystal palace'."""
    s = (name or "").strip()
    s = re.sub(r"\s+(FC|AFC|CF)\b", "", s, flags=re.I).strip()
    return s.lower()


def resolve_team(name: str, fpl_names: dict[str, int]) -> int | None:
    """Map a Polymarket club name onto an FPL team_id, or None.

    Returns None rather than guessing: a wrong fixture mapping would attach one
    club's market price to another's game, which is worse than no price at all.
    """
    c = _clean(name)
    want = NAME_FIX.get(c, c)
    for fpl, tid in fpl_names.items():
        f = fpl.lower()
        if f == want.lower() or f == c:
            return tid
        # 'Man City' vs 'manchester city': match on the distinctive token
        if want.lower() in f or f in want.lower():
            return tid
    return None


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def fetch_events(limit: int = 200) -> list[dict]:
    """Open EPL events, newest first. One call, no key."""
    return _get(f"{GAMMA}/events?limit={limit}&closed=false&tag_slug=epl")


def _outcome_prices(m: dict) -> dict[str, float]:
    """{outcome_label: price} for one market, prices already probabilities."""
    try:
        labels = json.loads(m.get("outcomes") or "[]")
        prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except (ValueError, TypeError):
        return {}
    return dict(zip(labels, prices)) if len(labels) == len(prices) else {}


def parse_match_event(ev: dict) -> dict | None:
    """Home/draw/away probabilities from one 'A vs. B' event.

    Polymarket splits a three-way match into separate Yes/No markets, one per
    outcome, so the three prices are collected by their group label and then
    normalised — they sum near 1 but not exactly, because each carries its own
    bid/ask spread.
    """
    title = ev.get("title") or ""
    if " vs. " not in title or " - " in title:
        return None
    home_name, away_name = [s.strip() for s in title.split(" vs. ", 1)]
    legs: dict[str, float] = {}
    for m in ev.get("markets") or []:
        px = _outcome_prices(m)
        if "Yes" not in px:
            continue
        label = (m.get("groupItemTitle") or m.get("question") or "").strip()
        if label:
            legs[label] = px["Yes"]
    if not legs:
        return None

    # Classify each leg EXACTLY. The draw's label is
    # "Draw (Crystal Palace FC vs. Manchester City FC)" — it contains BOTH club
    # names, so any substring match on a team name also matches the draw and
    # silently returns the draw price for the away side.
    p_home = p_draw = p_away = None
    h_key, a_key = _clean(home_name), _clean(away_name)
    for label, v in legs.items():
        low = label.lower()
        if low.startswith("draw") or "end in a draw" in low:
            p_draw = v
            continue
        key = _clean(label)
        if key == h_key:
            p_home = v
        elif key == a_key:
            p_away = v
    if p_home is None or p_away is None:
        return None
    if p_draw is None:
        p_draw = max(0.0, 1.0 - p_home - p_away)
    tot = p_home + p_draw + p_away
    if tot <= 0:
        return None
    return {
        "home_name": home_name, "away_name": away_name,
        "p_home": p_home / tot, "p_draw": p_draw / tot, "p_away": p_away / tot,
        "volume": float(ev.get("volume") or 0.0),
        "liquidity": float(ev.get("liquidity") or 0.0),
        "end": ev.get("endDate"),
        "slug": ev.get("slug"),
    }


def ingest(conn, season: str, *, limit: int = 200) -> dict:
    """Pull open EPL match markets into `market_quote`. Idempotent."""
    from .. import db
    db.ensure_market_quote(conn)
    teams = {r["name"]: int(r["team_id"]) for r in conn.execute(
        "SELECT team_id, name FROM team WHERE season=?", (season,))}
    fixtures = {(int(r["team_h"]), int(r["team_a"])): int(r["fixture_id"])
                for r in conn.execute(
                    "SELECT fixture_id, team_h, team_a FROM fixture "
                    "WHERE season=? AND finished=0", (season,))}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    rows, unmatched = [], []
    for ev in fetch_events(limit=limit):
        q = parse_match_event(ev)
        if not q:
            continue
        h = resolve_team(q["home_name"], teams)
        a = resolve_team(q["away_name"], teams)
        if h is None or a is None:
            unmatched.append(f"{q['home_name']} vs {q['away_name']}")
            continue
        fid = fixtures.get((h, a))
        if fid is None:
            continue                       # already played, or not this season
        rows.append((season, fid, SOURCE, now, h, a,
                     round(q["p_home"], 4), round(q["p_draw"], 4),
                     round(q["p_away"], 4), q["volume"], q["liquidity"],
                     q["slug"]))
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO market_quote (season, fixture_id, source, "
            "observed_utc, home_id, away_id, p_home, p_draw, p_away, volume, "
            "liquidity, ref) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return {"quotes": len(rows), "unmatched": unmatched[:8]}
