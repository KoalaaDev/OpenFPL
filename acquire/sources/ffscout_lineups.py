"""Fantasy Football Scout's free predicted line-ups page — a second predicted-XI
feed, plus every club's Out / Doubts (with a percentage) / Banned lists.

`https://www.fantasyfootballscout.co.uk/predicted-line-ups/` renders, for
each of the 20 clubs, an `<h2>` with the club name, a "Next Match" line, a
`scout-picks-pitch formation formation-4-2-3-1` block whose `<ul class="row-N">`
lists hold the predicted XI (`<li title="Havertz (Kai)">` with a short
`player-name` span), and a `story-parts` list with Out / Doubts (each doubt
carrying a `doubt-percent`) / Banned and a "Latest News" paragraph. All of it
is served to anonymous visitors (checked 2026-09-16); the lineup tester's
2020-21 season found this provider the most often-best. Pure stdlib parsing:
the scheduled collector runs with no third-party packages.
"""
from __future__ import annotations

import html as _html
import re

URL = "https://www.fantasyfootballscout.co.uk/predicted-line-ups/"
SOURCE_ID = "ffscout_lineups"
PARSER_VERSION = "1"

# the page's club names -> FPL short names (bootstrap `short_name`)
CLUBS = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "AFC Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton and Hove Albion": "BHA", "Brighton": "BHA", "Burnley": "BUR",
    "Chelsea": "CHE", "Coventry City": "COV", "Coventry": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull City": "HUL", "Hull": "HUL", "Ipswich Town": "IPS",
    "Ipswich": "IPS", "Leeds United": "LEE", "Leeds": "LEE", "Leicester City": "LEI",
    "Liverpool": "LIV", "Manchester City": "MCI", "Man City": "MCI", "Manchester United": "MUN",
    "Man Utd": "MUN", "Newcastle United": "NEW", "Newcastle": "NEW", "Nottingham Forest": "NFO",
    "Nott'm Forest": "NFO", "Sheffield United": "SHU", "Southampton": "SOU", "Sunderland": "SUN",
    "Tottenham Hotspur": "TOT", "Tottenham": "TOT", "Spurs": "TOT", "West Ham United": "WHU",
    "West Ham": "WHU", "Wolverhampton Wanderers": "WOL", "Wolves": "WOL", "Luton Town": "LUT",
}


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", _html.unescape(s)).strip()


def parse(page: str) -> tuple[list[dict], list[dict]]:
    """(lineup rows, team-news rows).

    lineup rows: team_abbr, team, formation, row, slot, player (full, from the
    title), short (the pitch label), next_match.
    team-news rows: team_abbr, team, kind (out/doubt/banned), player, pct
    (doubts only), news (the club's "Latest News" paragraph, once per club).
    """
    lineups: list[dict] = []
    news: list[dict] = []
    heads = list(re.finditer(r'<h2[^>]*>\s*([^<]+?)\s*</h2>', page))
    for i, m in enumerate(heads):
        name = _clean(m.group(1))
        abbr = CLUBS.get(name)
        if not abbr:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(page)
        blk = page[m.end():end]
        nxt = _clean((re.search(r'Next Match:</strong>\s*([^<]+)', blk) or [None, ""])[1]) or None
        fm = re.search(r'formation-([\d-]+)', blk)
        formation = fm.group(1) if fm else None
        pitch = re.search(r'scout-picks-pitch.*?</div>', blk, flags=re.S)
        slot = 0
        if pitch:
            for rm in re.finditer(r'<ul class="row-(\d+)">(.*?)</ul>', pitch.group(0), flags=re.S):
                row = int(rm.group(1))
                for pm in re.finditer(r'<li[^>]*title="([^"]*)"[^>]*>(.*?)</li>', rm.group(2), flags=re.S):
                    slot += 1
                    short = _clean((re.search(r'player-name[^>]*>([^<]*)<', pm.group(2)) or [None, ""])[1])
                    lineups.append({"team_abbr": abbr, "team": name, "formation": formation, "row": row,
                                    "slot": slot, "player": _html.unescape(pm.group(1)).strip(),
                                    "short": short, "next_match": nxt})
        latest = re.search(r'Latest News:\s*</strong>(.*?)</p>', blk, flags=re.S)
        latest_txt = _clean(latest.group(1)) if latest else None
        for kind, label in (("out", "Out"), ("doubt", "Doubts"), ("banned", "Banned")):
            sec = re.search(r'<strong>' + label + r':</strong>\s*<ul class="players">(.*?)</ul>', blk, flags=re.S | re.I)
            if not sec:
                continue
            for li in re.finditer(r'<li>(.*?)</li>', sec.group(1), flags=re.S):
                pct = re.search(r'doubt-percent">\s*(\d+)%', li.group(1))
                player = _clean(re.sub(r'<span class="doubt-percent">.*?</span>', "", li.group(1)))
                if not player:
                    continue
                news.append({"team_abbr": abbr, "team": name, "kind": kind, "player": player,
                             "pct": int(pct.group(1)) if pct else None, "news": None})
        if latest_txt:
            news.append({"team_abbr": abbr, "team": name, "kind": "news", "player": None,
                         "pct": None, "news": latest_txt[:1500]})
    return lineups, news
