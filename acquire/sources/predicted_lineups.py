"""Predicted starting XIs — the one channel the decomposition leaves open.

WHY THIS SOURCE AND NOT ANOTHER PIECE OF MODELLING. The oracle decomposition
(RESEARCH_LOG E13/E14) replaced every component of the engine with the truth
and then, separately, with a perfect RATE estimate. A perfect rate is worth
nothing — the attacking ceiling is irreducible match-to-match variance, not
estimator error — and a perfect LOCAL rate is significantly worse, so there is
no exploitable form signal either. The only component with reachable headroom
left is whether a player starts and lasts an hour, and no amount of modelling
reaches it: the residual sits in the band where P(start) is 0.30-0.70 and the
model is already CORRECTLY calibrated, because the manager has not decided.

E8b priced that band: resolving it is worth ~+89 points a season, and the value
is LINEAR in the fraction resolved, so a feed that is half right is worth half.

THE ARCHIVE WALL, and why this is a collector rather than a feature. Nobody
publishes a timestamped archive of past predicted lineups, so this cannot be
backtested from history — the same wall that stopped availability, expected
lineups and manager picks. Forward collection is the only route, which makes
starting it the whole point. Per E8b it becomes priceable after roughly five
gameweeks (~90 ambiguous rows each, ~450 observations), long before a decision
backtest would have the power to say anything.

SOURCE. RotoWire publishes confirmed and predicted XIs for the Premier League,
server-rendered, with a per-player position using the same vocabulary Understat
uses (GK/DL/DC/DMC/AMC/FW). Its robots.txt blocks a list of named crawlers and
allows a normal agent. Nothing here is scraped faster than once per scheduled
run, and only the eleven names per side are kept.

WHAT IS STORED, and the distinction that decides its worth: `status` records
whether the XI was CONFIRMED (published by the club, ~1h before kickoff and
therefore AFTER the deadline) or PREDICTED (a forecast, available before it).
Only the predicted rows are usable for a decision; the confirmed rows are the
ground truth to score them against. Conflating the two would manufacture an
oracle out of a feed.
"""
from __future__ import annotations

import re

from ..core import http

URL = "https://www.rotowire.com/soccer/lineups.php"
SOURCE_ID = "rotowire_lineups"
PARSER_VERSION = "1"

# RotoWire marks each side's list with one of these; only "expected" is a
# forecast, and only a forecast is worth anything before a deadline
STATUS = {"is-expected": "predicted", "is-confirmed": "confirmed"}


def fetch(url: str = URL) -> str:
    return http.get_text(url)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def parse(html: str) -> list[dict]:
    """One row per named player in a published or predicted XI.

    Each match is a `lineup is-soccer` box holding two `lineup__list` blocks,
    home then visitor, and each block carries its own status — a club can have
    confirmed its XI while the opponent has not, and the two must not be
    labelled alike.
    """
    out: list[dict] = []
    boxes = re.split(r'<div class="lineup is-soccer"', html)[1:]
    for box in boxes:
        teams = re.findall(
            r'<div class="lineup__abbr">\s*([A-Za-z0-9]{2,4})\s*</div>', box)
        lists = re.split(r'<ul class="lineup__list', box)[1:]
        for i, lst in enumerate(lists):
            side = ("home" if "is-home" in lst[:40]
                    else "away" if "is-visit" in lst[:40] else None)
            if side is None:
                continue
            m = re.search(r'class="lineup__status ([^"]*)"', lst)
            status = None
            if m:
                for k, v in STATUS.items():
                    if k in m.group(1):
                        status = v
                        break
                if status is None:
                    txt = _clean(lst[m.start():m.start() + 300]).lower()
                    status = "confirmed" if "confirmed" in txt else "predicted"
            team = teams[0] if side == "home" and teams else (
                teams[1] if side == "away" and len(teams) > 1 else None)
            # Everything after the "Injuries" heading is the doubt list, not
            # the XI, and it reuses `lineup__player` with a one-letter
            # position. Counting to eleven instead would work today and break
            # the first time a side is listed with a short bench.
            xi_part = re.split(r'<li class="lineup__title[^"]*">\s*Injur',
                               lst)[0]
            order = 0
            for pl in re.split(r'<li class="lineup__player"', xi_part)[1:]:
                pos = _one(r'<div class="lineup__pos[^"]*">\s*([A-Za-z]+)', pl)
                name = _one(r'<a title="([^"]+)"', pl) or _one(
                    r'/soccer/player/[^"]+">\s*([^<]+?)\s*</a>', pl)
                if not name:
                    continue
                order += 1
                out.append({
                    "team_abbr": team, "side": side, "status": status,
                    "position": pos, "player": name.strip(),
                    "slot": order,
                "in_xi": order <= 11,
                    "rotowire_id": _one(r"/soccer/player/[a-z0-9\-]+-(\d+)", pl),
                })
    return out


def _one(pat: str, s: str, d=None):
    m = re.search(pat, s, re.S)
    return m.group(1).strip() if m else d
