"""SportsGambler's predicted / confirmed Premier League lineups — a third
predicted-XI feed, keyless.

`https://www.sportsgambler.com/lineups/football/england-premier-league/`
lists the upcoming fixtures, each with a lineup id; the XIs themselves come
from `GET /lineups/lineups-load2.php?id=<id>` (the page's own AJAX call),
which returns both sides' XIs as `lineups-home` / `lineups-away` containers
of `player-name` spans grouped into `players-line` rows (the goalkeeper's
line is `goalie`), with the formation in a `lineups-toggle-formation` span.
The fixture row's toggle button says "Predicted Lineups" or "Confirmed
Lineups", which is the status. Checked 2026-09-16. Pure stdlib.
"""
from __future__ import annotations

import html as _html
import re

INDEX_URL = "https://www.sportsgambler.com/lineups/football/england-premier-league/"
LOAD_URL = "https://www.sportsgambler.com/lineups/lineups-load2.php?id={id}"
SOURCE_ID = "sportsgambler_lineups"
PARSER_VERSION = "1"

CLUBS = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Brighton & Hove Albion": "BHA", "Burnley": "BUR", "Chelsea": "CHE",
    "Coventry": "COV", "Coventry City": "COV", "Crystal Palace": "CRY", "Everton": "EVE",
    "Fulham": "FUL", "Hull": "HUL", "Hull City": "HUL", "Ipswich": "IPS", "Ipswich Town": "IPS",
    "Leeds": "LEE", "Leeds United": "LEE", "Leicester": "LEI", "Liverpool": "LIV",
    "Manchester City": "MCI", "Man City": "MCI", "Manchester United": "MUN", "Man Utd": "MUN",
    "Newcastle": "NEW", "Newcastle United": "NEW", "Nottingham Forest": "NFO", "Nottm Forest": "NFO",
    "Nott'm Forest": "NFO", "Nott’m Forest": "NFO", "Man United": "MUN",
    "Sunderland": "SUN", "Tottenham": "TOT", "Tottenham Hotspur": "TOT", "West Ham": "WHU",
    "Wolves": "WOL", "Wolverhampton": "WOL",
}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def parse_index(page: str) -> list[dict]:
    """Fixtures on the index: id, date headline, time, home, away, status."""
    out = []
    date = None
    for chunk in re.split(r'(?=<h3 class="date-headline">)', page):
        dm = re.match(r'<h3 class="date-headline">\s*([^<]+?)\s*</h3>', chunk)
        if dm:
            date = _clean(dm.group(1))
        for row in re.split(r'<div class="lineup-row">', chunk)[1:]:
            lid = re.search(r'rel="#lineup(\d+)"', row)
            if not lid:
                continue
            teams = re.findall(r'<span class="fxs-team[^"]*">\s*([^<]+?)\s*</span>', row)
            league = _clean((re.search(r'fxs-league[^>]*>([^<]*)<', row) or [None, ""])[1])
            if len(teams) < 2 or "Premier League" not in league:
                continue
            btn = _clean((re.search(r'<span class="h-sm">([^<]*)</span>', row) or [None, ""])[1]).lower()
            out.append({"id": lid.group(1), "date": date,
                        "time": _clean((re.search(r'fxs-time">([^<]*)<', row) or [None, ""])[1]) or None,
                        "home": _clean(teams[0]), "away": _clean(teams[1]),
                        "home_abbr": CLUBS.get(_clean(teams[0])), "away_abbr": CLUBS.get(_clean(teams[1])),
                        "status": "confirmed" if "confirmed" in btn else "predicted"})
    return out


def parse_lineup(payload: str) -> dict:
    """{'home': [players...], 'away': [...], 'formation': {'home': '4-2-3-1', 'away': ...}}.
    Each player: row (1 = goalkeeper), slot, shirt, player."""
    out = {"home": [], "away": [], "formation": {}}
    forms = re.findall(r'<h3><span>([^<]+?) Predicted Lineup</span>\s*<span class="lineups-toggle-formation">([^<]*)</span>', payload)
    if len(forms) >= 2:
        out["formation"] = {"home": forms[0][1].strip(), "away": forms[1][1].strip()}
    for side in ("home", "away"):
        m = re.search(r'<div class="lineups-' + side + r'[^"]*">(.*?)<!--lineups ' + side + r'-->', payload, flags=re.S)
        if not m:
            m = re.search(r'<div class="lineups-' + side + r'[^"]*">(.*?)(?=<div class="lineups-(?:home|away)|<!--lineups)', payload, flags=re.S)
        if not m:
            continue
        slot = 0
        for row_no, line in enumerate(re.split(r'<div class="players-line', m.group(1))[1:], start=1):
            for pm in re.finditer(r'<span class="lineups-player">(.*?)</span>\s*</span>', line, flags=re.S):
                shirt = re.search(r'player-profile">\s*(\d+)', pm.group(1))
                name = _clean((re.search(r'player-name">([^<]*)', pm.group(1)) or [None, ""])[1])
                if not name:
                    continue
                slot += 1
                out[side].append({"row": row_no, "slot": slot, "shirt": int(shirt.group(1)) if shirt else None,
                                  "player": name})
    return out
