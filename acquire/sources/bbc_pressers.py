"""BBC Sport's Friday "Premier League news conferences" live pages — the
first keyless, archived, PRE-DEADLINE team-news source this repo has found.

Checked against the site on 2026-09-14 (page c64gdwl1yrqwt): a live blog
that runs from ~08:00 to ~14:30 UK time on the Friday before a gameweek,
one post per manager quote, each post carrying the fixture it concerns as
its header ("Bournemouth v Brentford (Sat, 15:00 BST)") and a publication
timestamp. Of 71 posts that day, 20 carried concrete availability news
("Collins ... picked up a calf injury, won't be involved", "Maddison is
available", "Caicedo will not be available"). All of it precedes Saturday's
deadline, which is what makes it usable — the same page's confirmed XIs
never are.

DISCOVERY. Pages have opaque ids, so past weeks are found through BBC's
search container (`search-results`, paginated with `defaultPageNumber`),
filtered to live pages whose headline mentions news conferences. Results are
relevance-ordered and go back to at least early 2024. The stream itself is
the same `stream` container the match pages use, keyed by the page's
`liveTextStreamId`, which is read from the page HTML.

WHAT IS STORED: every post verbatim with its timestamp and fixture label
(`acq_bbc_presser`), and one row per page (`acq_bbc_presser_page`). No
interpretation happens here — turning "won't be involved" into a player-level
availability observation is the modelling engine's job and the hypothesis
this data exists to test (RESEARCH_LOG: press-conference extraction).
"""
from __future__ import annotations

import json
import re
import urllib.parse
from datetime import date, timedelta

from ..core import http
from .. import storage

SOURCE_ID = "bbc_pressers"
PARSER_VERSION = "1"
BASE = "https://www.bbc.co.uk/wc-data/container"
SEARCH_TERM = "Premier League news conferences"
# relevance-ordered search misses weeks whose headline is phrased differently;
# each extra term is a few dozen cheap requests and the union is what counts
SEARCH_TERMS = (SEARCH_TERM, "Premier League press conferences",
                "Premier League news conferences recap", "Premier League build-up news conferences",
                "Friday's Premier League news conferences", "Premier League managers news conferences")
TITLE_RE_WIDE = re.compile(r"news conference|press conference|build-up", re.I)
DELAY = 1.0
TITLE_RE = re.compile(r"news conference", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS acq_bbc_presser_page (
    page_id       TEXT PRIMARY KEY,
    headline      TEXT,
    date_published TEXT,
    start_utc     TEXT,
    stream_id     TEXT,
    pages_done    INTEGER DEFAULT 0,
    pages_total   INTEGER,
    posts         INTEGER DEFAULT 0,
    observed_utc  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS acq_bbc_presser (
    page_id       TEXT NOT NULL,
    post_urn      TEXT NOT NULL,
    published_utc TEXT,
    fixture_label TEXT,
    text          TEXT,
    raw_id        INTEGER,
    PRIMARY KEY (page_id, post_urn)
);
CREATE INDEX IF NOT EXISTS idx_acq_bbc_presser_time ON acq_bbc_presser (published_utc);
"""


def register(conn) -> None:
    conn.executescript(SCHEMA)
    storage.register_source(
        conn, SOURCE_ID, name="BBC Sport 'Premier League news conferences' live pages",
        base_url=BASE, source_type="api", enabled=1,
        robots_policy="/sport allowed for generic agents",
        terms_note="personal, non-commercial use per BBC terms; owner's decision",
        request_delay=DELAY, parser_version=PARSER_VERSION)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def search_url(page: int, term: str = SEARCH_TERM) -> str:
    q = urllib.parse.quote(term)
    return (f"{BASE}/search-results?country=gb&defaultPageNumber={page}"
            f"&getDestination=SPORT_GNL&getFacets=%22%22&getSearchService=search"
            f"&getSearchTerm={q}&isUk=true&language=en-GB")


def parse_search(payload: dict) -> list[dict]:
    """Live pages about news conferences: (page_id, headline, date)."""
    d = payload.get("data", payload)
    res = d.get("initialResults") or {}
    out = []
    for it in res.get("items") or []:
        head = it.get("headline") or ""
        cats = it.get("categories") or []
        uri = it.get("uri") or ""
        if "urn:bbc:tipo:type:live" not in cats or not TITLE_RE.search(head):
            continue
        pid = uri.rsplit(":", 1)[-1] if uri.startswith("urn:bbc:tipo:topic:") else None
        if not pid:
            url = it.get("url") or ""
            pid = url.rstrip("/").rsplit("/", 1)[-1] if "/live/" in url else None
        if pid:
            out.append({"page_id": pid, "headline": head,
                        "date_published": it.get("datePublished")})
    return out


def enumerate_pages(*, max_pages: int = 200, stop_after_empty: int = 8,
                    progress=print, terms=SEARCH_TERMS) -> list[dict]:
    """Walk the search results for every term until several consecutive
    pages add nothing; the union over terms is returned, oldest first."""
    seen: dict[str, dict] = {}
    for term in terms:
        _enumerate_term(term, seen, max_pages, stop_after_empty, progress)
    return sorted(seen.values(), key=lambda i: i["date_published"] or "")


def _enumerate_term(term, seen, max_pages, stop_after_empty, progress):
    empty = 0
    for p in range(1, max_pages + 1):
        resp = http.get(search_url(p, term), delay=DELAY)
        if not resp.ok:
            break
        try:
            items = parse_search(json.loads(resp.text))
        except ValueError:
            break
        new = [i for i in items if i["page_id"] not in seen]
        for i in new:
            seen[i["page_id"]] = i
        empty = empty + 1 if not new else 0
        if p % 10 == 0:
            progress(f"  '{term}' page {p}: {len(seen)} press-conference pages so far")
        if empty >= stop_after_empty:
            break


# --------------------------------------------------------------------------
# one page
# --------------------------------------------------------------------------

def page_url(page_id: str) -> str:
    return f"https://www.bbc.co.uk/sport/football/live/{page_id}"


def stream_key_from_html(html: str) -> str | None:
    """The page's own `stream?...` container query, straight from the HTML
    (the ids inside it are what the stream endpoint needs)."""
    m = re.search(r'stream\?assetId=[^"\\]*?liveTextStreamId=([A-Za-z0-9-]+)[^"\\]*', html)
    return m.group(1) if m else None


def stream_url(page_id: str, stream_id: str, page: int) -> str:
    # exactly the query the page itself issues; unlike the match streams
    # this one has no `type=football`, and adding it returns an empty body
    return (f"{BASE}/stream?assetId={page_id}&enableDotcomAds=true"
            f"&globalContainerPolling=true&home=sport&language=en-GB"
            f"&liveTextStreamId={stream_id}&pageNumber={page}&pageSize=20"
            f"&pageUrl=%2Fsport%2Ffootball%2Flive%2F{page_id}")


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _ptext(o) -> str:
    return " ".join(b["model"].get("text", "") for b in _walk(o)
                    if isinstance(b, dict) and b.get("type") == "paragraph"
                    and isinstance(b.get("model"), dict)).strip()


def parse_stream(payload: dict) -> tuple[list[dict], int]:
    d = payload.get("data", payload)
    total = int((d.get("page") or {}).get("total") or 1)
    posts = []
    for r in d.get("results") or []:
        dates = r.get("dates") or {}
        label = _ptext(r.get("header"))
        if not label:                       # `titles` holds dicts: {"title": …}
            t0 = (r.get("titles") or [None])[0]
            label = (t0.get("title") if isinstance(t0, dict) else t0) or ""
        posts.append({
            "post_urn": r.get("urn") or f"{dates.get('firstPublished')}|{_ptext(r.get('content'))[:40]}",
            "published_utc": dates.get("firstPublished") or r.get("unformattedTime"),
            "fixture_label": (label or "")[:120],
            "text": _ptext(r.get("content")),
        })
    return posts, total


def collect_page(conn, page: dict, *, progress=print, refresh: bool = False) -> dict:
    """Archive a page's live text.

    Incremental by default: a page whose every stream page has been read is
    skipped. `refresh` re-reads the LAST stream page instead, which is what a
    live blog needs — it keeps appending posts to the page it is on all day,
    so "we have read all N pages" is only true until the next post. Posts are
    keyed by their own urn, so re-reading writes nothing new twice.
    """
    now = http.utcnow()
    pid = page["page_id"]
    row = conn.execute("SELECT stream_id, pages_done, pages_total FROM "
                       "acq_bbc_presser_page WHERE page_id=?", (pid,)).fetchone()
    done = bool(row and row[2] is not None and row[1] >= row[2])
    if done and not refresh:
        return {"page_id": pid, "skipped": True}
    stream_id = row[0] if row and row[0] else None
    if not stream_id:
        resp = http.get(page_url(pid), delay=DELAY)
        if not resp.ok:
            return {"page_id": pid, "error": resp.error or resp.status}
        storage.store_raw(conn, SOURCE_ID, resp, parser_version=PARSER_VERSION,
                          keep_payload=False)
        stream_id = stream_key_from_html(resp.text)
        if not stream_id:
            return {"page_id": pid, "error": "no liveTextStreamId in page"}
    conn.execute(
        "INSERT INTO acq_bbc_presser_page (page_id, headline, date_published, stream_id, "
        "observed_utc) VALUES (?,?,?,?,?) ON CONFLICT(page_id) DO UPDATE SET "
        "headline=excluded.headline, date_published=excluded.date_published, "
        "stream_id=excluded.stream_id, observed_utc=excluded.observed_utc",
        (pid, page.get("headline"), page.get("date_published"), stream_id, now))
    before = conn.execute("SELECT COUNT(*) FROM acq_bbc_presser WHERE page_id=?",
                          (pid,)).fetchone()[0]
    n_posts = 0
    p = (row[1] if row else 0) + 1
    if done and refresh:
        p = max(1, p - 1)                   # the page the blog is still writing
        total = None                        # and re-ask how many there are now
    else:
        total = row[2] if row and row[2] else None
    while total is None or p <= total:
        resp = http.get(stream_url(pid, stream_id, p), delay=DELAY)
        if not resp.ok or not resp.text.strip():
            break
        try:
            payload = json.loads(resp.text)
        except ValueError:
            break
        rid = storage.store_raw(conn, SOURCE_ID, resp, parser_version=PARSER_VERSION)
        posts, total = parse_stream(payload)
        for q in posts:
            conn.execute(
                "INSERT OR REPLACE INTO acq_bbc_presser (page_id, post_urn, published_utc, "
                "fixture_label, text, raw_id) VALUES (?,?,?,?,?,?)",
                (pid, q["post_urn"], q["published_utc"], q["fixture_label"], q["text"], rid))
        n_posts += len(posts)
        # count the rows rather than adding to a running total: a refresh
        # re-reads a page it has already seen
        conn.execute("UPDATE acq_bbc_presser_page SET pages_done=?, pages_total=?, "
                     "posts=(SELECT COUNT(*) FROM acq_bbc_presser WHERE page_id=?) "
                     "WHERE page_id=?", (p, total, pid, pid))
        p += 1
    conn.commit()
    after = conn.execute("SELECT COUNT(*) FROM acq_bbc_presser WHERE page_id=?",
                         (pid,)).fetchone()[0]
    return {"page_id": pid, "posts": n_posts, "new": after - before, "pages": total}


def refresh_live(conn, *, within_days: int = 3, progress=print) -> dict:
    """Re-read the newest archived page, for a blog that is still running.

    The Friday page is published in the morning and written all day — the
    managers speak in blocks through to the evening — so a collector that
    stops at "every page read" freezes at whatever had been posted when it
    first ran.
    """
    register(conn)
    since = (date.today() - timedelta(days=within_days)).isoformat()
    row = conn.execute(
        "SELECT page_id, headline, date_published FROM acq_bbc_presser_page "
        "WHERE date_published >= ? ORDER BY date_published DESC LIMIT 1",
        (since,)).fetchone()
    if not row:
        return {"pages": 0, "posts": 0, "new": 0}
    out = collect_page(conn, {"page_id": row[0], "headline": row[1],
                              "date_published": row[2]}, progress=progress, refresh=True)
    return {"pages": 1, "posts": out.get("posts", 0), "new": out.get("new", 0),
            "page_id": row[0]}


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def backfill(conn, *, since: str = "2023-07-01", progress=print) -> dict:
    register(conn)
    pages = [p for p in enumerate_pages(progress=progress, stop_after_empty=25)
             if (p["date_published"] or "") >= since]
    progress(f"  {len(pages)} press-conference pages since {since}")
    out = {"pages": len(pages), "collected": 0, "posts": 0, "skipped": 0, "errors": 0}
    for pg in pages:
        r = collect_page(conn, pg, progress=progress)
        if r.get("skipped"):
            out["skipped"] += 1
        elif r.get("error"):
            out["errors"] += 1
            progress(f"  {pg['page_id']} {pg.get('date_published', '')[:10]}: {r['error']}")
        else:
            out["collected"] += 1
            out["posts"] += r.get("posts", 0)
            progress(f"  {pg.get('date_published', '')[:10]} {pg['headline'][:60]}: "
                     f"{r.get('posts')} posts")
    storage.mark(conn, SOURCE_ID, True, http.utcnow())
    conn.commit()
    return out


def pull(conn, *, season: str | None = None, dry_run: bool = False) -> dict:
    """Scheduled: the most recent pages only (first few search pages)."""
    register(conn)
    since = (date.today() - timedelta(days=21)).isoformat()
    pages = [p for p in enumerate_pages(max_pages=5, stop_after_empty=5, progress=lambda m: None)
             if (p["date_published"] or "") >= since]
    if dry_run:
        return {"pages": len(pages)}
    out = {"pages": len(pages), "posts": 0}
    for pg in pages:
        r = collect_page(conn, pg, progress=lambda m: None)
        out["posts"] += r.get("posts", 0) or 0
    conn.commit()
    return out
