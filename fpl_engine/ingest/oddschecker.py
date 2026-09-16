"""Oddschecker — bookmaker odds for upcoming Premier League fixtures, keyless.

Found by the owner (2026-09-16) after the Odds API key died. Each match page
(`/football/english/premier-league/<home>-v-<away>/winner`) embeds a
hypernova JSON blob (`data-hypernova-key="subeventmarkets"`) holding, for
every populated market, each selection's decimal odds per bookmaker (~24 of
them) and Oddschecker's own implied probability. Three markets are populated
server-side and are all this module needs:

  * Win Market      -> 1X2, de-margined from the median bookmaker price
  * Correct Score   -> the scoreline distribution, from which P(over 2.5),
                       P(clean sheet) for each side and P(both score) follow
                       directly — the exact-score market CLAUDE.md flagged
                       as strictly more information than 1X2 + O/U
  * Anytime Goalscorer -> a per-player price, the first market signal in
                       this repo that reaches a player without going
                       through the fixture channel

Access: Cloudflare challenges Python's TLS fingerprint, so pages are fetched
through the system curl with a cookie jar and a Referer (the index visit
sets the cookie the match pages need), one request a second. Written into `match_odds` (source
"oddschecker", one row per fixture, which is what the engine blends at
`ODDS_WEIGHT`), into `market_prop` (clean sheets, both-to-score, anytime
scorer per player) and appended to `data/collected/oddschecker/<season>.csv`
so the archive survives the database. Never cached: odds move.
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import time
from datetime import datetime, timezone

from .. import config, db
from ..xpts import odds_model
from .odds import resolve_team

BASE = "https://www.oddschecker.com"
INDEX_URL = f"{BASE}/football/english/premier-league"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
DELAY = 1.0
SOURCE = "oddschecker"
NAME_OVERRIDES = {
    "tottenham": "Spurs", "man city": "Man City", "man utd": "Man Utd", "man united": "Man Utd",
    "nottm forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest", "wolves": "Wolves",
    "brighton": "Brighton", "newcastle": "Newcastle", "west ham": "West Ham", "sheffield utd": "Sheffield Utd",
    "leeds": "Leeds", "hull": "Hull City", "ipswich": "Ipswich Town", "coventry": "Coventry City",
    "leicester": "Leicester",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_prop (
    season        TEXT NOT NULL,
    fixture_id    INTEGER NOT NULL,
    kind          TEXT NOT NULL,        -- cs_home | cs_away | btts | anytime
    player        TEXT NOT NULL DEFAULT '',
    prob          REAL,                 -- implied probability
    best_odds     REAL,
    median_odds   REAL,
    n_bookmakers  INTEGER,
    observed_utc  TEXT NOT NULL,
    PRIMARY KEY (season, fixture_id, kind, player)
);
"""


# ------------------------------------------------------------------ fetch --
class Client:
    """A cookie-keeping session through the system ``curl``.

    Cloudflare's bot check challenges Python's TLS fingerprint (requests gets
    a 403 with ``cf-mitigated: challenge`` on every path) while a plain curl
    with a cookie jar and a Referer is served — the index visit sets the
    ``__cf_bm`` cookie the match pages need. One request a second."""

    def __init__(self, jar: str | None = None):
        import tempfile
        self.jar = jar or os.path.join(tempfile.gettempdir(), "oddschecker.cookies")
        self.last = 0.0
        self.referer = INDEX_URL

    def get(self, url: str) -> str | None:
        import subprocess
        wait = DELAY - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        cmd = ["curl", "-s", "-L", "--max-time", "40", "-c", self.jar, "-b", self.jar, "-A", UA,
               "-H", "Accept: text/html,application/xhtml+xml", "-H", "Accept-Language: en-GB,en;q=0.9",
               "-H", f"Referer: {self.referer}", "-w", "\n%{http_code}", url]
        body = ""
        for attempt in range(2):          # the first hit without a cookie is challenged
            try:
                res = subprocess.run(cmd, capture_output=True, timeout=60)
            except Exception:      # noqa: BLE001
                return None
            self.last = time.time()
            body = res.stdout.decode("utf-8", errors="ignore")
            if body.rstrip().endswith("200"):
                break
            time.sleep(DELAY)
        text, _, code = body.rpartition("\n")
        if code.strip() != "200":
            return None
        self.referer = url
        return text


# ------------------------------------------------------------------ parse --
def parse_index(html: str) -> list[str]:
    """Match-page URLs on the league index, in page order, de-duplicated."""
    out, seen = [], set()
    for m in re.finditer(r'href="/?(football/english/premier-league/[a-z0-9-]+-v-[a-z0-9-]+/winner)"', html):
        u = m.group(1)
        if u not in seen:
            seen.add(u)
            out.append(f"{BASE}/{u}")
    return out


def _hypernova(html: str, key: str) -> dict | None:
    m = re.search(r'<script type="application/json" data-hypernova-key="' + key
                  + r'"[^>]*>\s*<!--(.*?)-->\s*</script>', html, flags=re.S)
    if not m:
        return None
    raw = m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _prices(odds: dict, bet_id) -> tuple[float | None, float | None, int]:
    o = odds.get(str(bet_id)) or {}
    vals = [float(x["oddsDecimal"]) for x in o.values()
            if x.get("status") == "ACTIVE" and x.get("oddsDecimal")]
    if not vals:
        return None, None, 0
    return max(vals), statistics.median(vals), len(vals)


def parse_match(html: str) -> dict | None:
    """The three populated markets, with median/best decimal odds per bet."""
    hdr = _hypernova(html, "subeventheader") or {}
    mk = _hypernova(html, "subeventmarkets")
    if not mk:
        return None
    bo = mk.get("bestOdds") or {}
    markets = {str(v["ocMarketId"]): v.get("marketTypeName") for v in (bo.get("markets") or {}).get("entities", {}).values()}
    bets = list((bo.get("bets") or {}).get("entities", {}).values())
    odds = bo.get("odds") or {}
    name = hdr.get("subeventName") or ""
    home, away = (name.split(" vs ", 1) + [None])[:2] if " vs " in name else (None, None)
    out = {"home": home, "away": away, "kickoff_utc": hdr.get("subeventStartTime"),
           "win": [], "correct_score": [], "anytime": [],
           "siblings": [s.get("subeventUrl") for s in hdr.get("siblingSubevents") or []]}
    for b in bets:
        kind = markets.get(str(b.get("marketId")))
        best, med, n = _prices(odds, b.get("ocBetId"))
        row = {"name": b.get("betName"), "line": b.get("line"), "best": best, "median": med,
               "n": n, "prob": b.get("probability")}
        if kind == "Win Market":
            out["win"].append(row)
        elif kind == "Correct Score":
            out["correct_score"].append(row)
        elif kind == "Anytime Goalscorer":
            out["anytime"].append(row)
    return out


# ----------------------------------------------------------------- derive --
def derive(match: dict) -> dict | None:
    """De-margined 1X2, the scoreline distribution's P(over 2.5), clean
    sheets and both-to-score, and the implied Poisson rates."""
    home, away = match.get("home"), match.get("away")
    win = {r["name"]: r for r in match.get("win") or [] if r.get("median")}
    if not (home in win and away in win and "Draw" in win):
        return None
    inv = {k: 1.0 / win[k]["median"] for k in (home, "Draw", away)}
    tot = sum(inv.values())
    p_home, p_draw, p_away = inv[home] / tot, inv["Draw"] / tot, inv[away] / tot
    p_over = p_cs_home = p_cs_away = p_btts = None
    cs = [r for r in match.get("correct_score") or [] if r.get("median") and r.get("line")]
    if len(cs) >= 10:
        dist = {}
        for r in cs:
            try:
                a, b = (int(x) for x in str(r["line"]).split("-"))
            except ValueError:
                continue
            # the line reads winner's goals first; a draw is symmetric
            if r["name"] == home:
                h, aw = a, b
            elif r["name"] == away:
                h, aw = b, a
            else:
                h, aw = a, b
            dist[(h, aw)] = dist.get((h, aw), 0.0) + 1.0 / r["median"]
        z = sum(dist.values())
        if z > 0:
            dist = {k: v / z for k, v in dist.items()}
            p_over = sum(v for (h, aw), v in dist.items() if h + aw >= 3)
            p_cs_home = sum(v for (h, aw), v in dist.items() if aw == 0)
            p_cs_away = sum(v for (h, aw), v in dist.items() if h == 0)
            p_btts = sum(v for (h, aw), v in dist.items() if h > 0 and aw > 0)
    lam_h, lam_a = odds_model.implied_rates(p_home, p_draw, p_away, p_over)
    anytime = [{"player": r["name"], "prob": (1.0 / r["median"]) if r.get("median") else None,
                "best": r.get("best"), "median": r.get("median"), "n": r.get("n"),
                "site_prob": r.get("prob")} for r in match.get("anytime") or []]
    return {"p_home": p_home, "p_draw": p_draw, "p_away": p_away, "p_over25": p_over,
            "p_cs_home": p_cs_home, "p_cs_away": p_cs_away, "p_btts": p_btts,
            "lam_home": lam_h, "lam_away": lam_a, "anytime": anytime}


def _resolve(name: str, fpl_names: list[str]) -> str:
    """An override that names a real FPL club wins; otherwise the shared
    resolver (exact, then token overlap; fails loud on ambiguity)."""
    over = NAME_OVERRIDES.get((name or "").strip().lower())
    if over and over in fpl_names:
        return over
    return resolve_team(name, fpl_names)


# ----------------------------------------------------------------- ingest --
def _fixture_lookup(conn, season: str) -> dict:
    return {(r["kickoff_utc"][:10], int(r["team_h"])): int(r["fixture_id"]) for r in conn.execute(
        "SELECT kickoff_utc, team_h, fixture_id FROM fixture WHERE season=? AND kickoff_utc IS NOT NULL",
        (season,))}


def ingest(conn, season: str, *, client=None, max_fixtures: int = 20,
           pages: dict | None = None) -> dict:
    """Fetch the index and each upcoming fixture's page, derive, store.
    ``pages`` ({url: html}) injects fetched pages for tests."""
    conn.executescript(SCHEMA)
    ids = {r["name"]: int(r["team_id"]) for r in conn.execute(
        "SELECT name, team_id FROM team WHERE season=?", (season,))}
    fpl_names = list(ids)
    lookup = _fixture_lookup(conn, season)
    cl = client
    if pages is None:
        cl = cl or Client()
        index = cl.get(INDEX_URL)
        urls = parse_index(index or "")
    else:
        urls = [u for u in pages if u != INDEX_URL]
    out = {"fixtures": 0, "props": 0, "unresolved": [], "errors": 0}
    now = datetime.now(timezone.utc).isoformat()
    archive = os.path.join(config.DATA_DIR, "collected", "oddschecker", f"{season}.csv")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    new_file = not os.path.exists(archive)
    with open(archive, "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["observed_utc", "fixture_id", "home", "away", "kickoff_utc", "kind", "player",
                        "prob", "best_odds", "median_odds", "n_bookmakers", "site_prob"])
        for url in urls[:max_fixtures]:
            html = pages.get(url) if pages is not None else cl.get(url)
            if not html:
                out["errors"] += 1
                continue
            m = parse_match(html)
            if not m or not m.get("home"):
                out["errors"] += 1
                continue
            try:
                hn = _resolve(m["home"], fpl_names)
                an = _resolve(m["away"], fpl_names)
            except Exception:      # noqa: BLE001 - fail loud per fixture, keep going
                out["unresolved"].append(f"{m['home']} v {m['away']}")
                continue
            if hn not in ids or an not in ids:
                out["unresolved"].append(f"{m['home']} v {m['away']}")
                continue
            d = derive(m)
            if not d:
                out["errors"] += 1
                continue
            kick = (m.get("kickoff_utc") or "")[:10]
            fid = lookup.get((kick, ids[hn]))
            if fid is None:
                out["unresolved"].append(f"{m['home']} v {m['away']} @ {kick} (no fixture)")
                continue
            db.upsert(conn, "match_odds", [{
                "season": season, "fixture_id": fid, "source": SOURCE, "kickoff_date": kick,
                "home_id": ids[hn], "away_id": ids[an], "p_home": d["p_home"], "p_draw": d["p_draw"],
                "p_away": d["p_away"], "p_over25": d["p_over25"], "lam_home": d["lam_home"],
                "lam_away": d["lam_away"]}])
            out["fixtures"] += 1
            props = []
            for kind in ("cs_home", "cs_away", "btts"):
                if d.get(f"p_{kind}") is not None:
                    props.append({"season": season, "fixture_id": fid, "kind": kind, "player": "",
                                  "prob": d[f"p_{kind}"], "best_odds": None, "median_odds": None,
                                  "n_bookmakers": None, "observed_utc": now})
            for a in d["anytime"]:
                if a["prob"] is None:
                    continue
                props.append({"season": season, "fixture_id": fid, "kind": "anytime", "player": a["player"],
                              "prob": a["prob"], "best_odds": a["best"], "median_odds": a["median"],
                              "n_bookmakers": a["n"], "observed_utc": now})
            if props:
                db.upsert(conn, "market_prop", props)
                out["props"] += len(props)
            w.writerow([now, fid, hn, an, m.get("kickoff_utc"), "1x2", "", f"{d['p_home']:.4f}/{d['p_draw']:.4f}/{d['p_away']:.4f}",
                        "", "", "", ""])
            for p in props:
                w.writerow([now, fid, hn, an, m.get("kickoff_utc"), p["kind"], p["player"],
                            round(p["prob"], 4), p["best_odds"], p["median_odds"], p["n_bookmakers"], ""])
    conn.commit()
    return out
