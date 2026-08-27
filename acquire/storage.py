"""Storage for the acquisition engine, inside the existing database.

Three strict layers, and the boundary between them is the point:

    RAW         exactly what the source returned, hashed and timestamped
    NORMALISED  canonical entities and standardised fields (these tables)
    FEATURES    model-ready variables — built by the modelling engine, never here

Every normalised observation carries ``observed_utc`` (when we saw it) and,
where the source provides one, ``source_published_utc`` (when the source says
it became true). Both are needed to answer the only question a backtest may
ask: *what could have been known at time T?*

The schema is additive and lives in the same SQLite file as the modelling
tables, so there is one database rather than two that drift apart.
"""
from __future__ import annotations

from fpl_engine import db as _db   # connection handling only; no model logic

SCHEMA = """
-- Which sources exist, what they permit, and how to treat them.
CREATE TABLE IF NOT EXISTS acq_source (
    source_id     TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    base_url      TEXT,
    source_type   TEXT,              -- api | html | archive
    enabled       INTEGER DEFAULT 1,
    robots_policy TEXT,              -- what robots.txt actually says
    terms_note    TEXT,              -- ToS findings, in plain words
    request_delay REAL DEFAULT 2.0,
    parser_version TEXT,
    last_success  TEXT,
    last_failure  TEXT
);

-- RAW. Never destroyed, deduplicated by content hash, so a better parser can
-- be re-run over history instead of re-downloading it.
CREATE TABLE IF NOT EXISTS acq_raw_document (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL,
    url           TEXT NOT NULL,
    retrieved_utc TEXT NOT NULL,
    http_status   INTEGER,
    content_hash  TEXT NOT NULL,
    content_type  TEXT,
    parser_version TEXT,
    payload       TEXT,
    UNIQUE (source_id, url, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_acq_raw_time
    ON acq_raw_document (source_id, retrieved_utc);

-- NORMALISED: availability. Stored as a CHANGE LOG — a row is written only
-- when a player's state differs from the last one recorded, so "state at time
-- T" is the latest row at or before T, and the table stays small.
--
-- ``news`` is kept verbatim. "75% chance of playing" must not be flattened
-- into "available" on the way in; what the source actually said is the record.
CREATE TABLE IF NOT EXISTS acq_player_availability (
    season        TEXT NOT NULL,
    player_id     INTEGER NOT NULL,     -- FPL element id for that season
    observed_utc  TEXT NOT NULL,        -- when this state was observed by us
    source_id     TEXT NOT NULL,
    source_published_utc TEXT,          -- when the source says it began
    status        TEXT,                 -- a | d | i | s | u
    chance_next   REAL,                 -- 0..1, NULL when the source is silent
    news          TEXT,                 -- verbatim
    raw_id        INTEGER,              -- provenance -> acq_raw_document.id
    PRIMARY KEY (season, player_id, observed_utc)
);
CREATE INDEX IF NOT EXISTS idx_acq_avail_time
    ON acq_player_availability (season, observed_utc);

-- Incremental state, so a pull does not refetch what it already has.
CREATE TABLE IF NOT EXISTS acq_sync_state (
    source_id     TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    season        TEXT,
    last_cursor   TEXT,
    last_hash     TEXT,
    last_success_utc TEXT,
    PRIMARY KEY (source_id, dataset, season)
);
"""


def connect(db_path: str | None = None):
    return _db.connect(db_path)


def init(conn) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def register_source(conn, source_id: str, **fields) -> None:
    cols = ["source_id"] + list(fields)
    vals = [source_id] + [fields[k] for k in fields]
    conn.execute(
        f"INSERT OR REPLACE INTO acq_source ({','.join(cols)}) "
        f"VALUES ({','.join('?' * len(cols))})", vals)


def mark(conn, source_id: str, ok: bool, when: str) -> None:
    col = "last_success" if ok else "last_failure"
    conn.execute(f"UPDATE acq_source SET {col}=? WHERE source_id=?",
                 (when, source_id))


def store_raw(conn, source_id: str, resp, *, parser_version: str,
              keep_payload: bool = True) -> int | None:
    """Record a response. Returns the raw_document id, or None if unchanged.

    Identical content fetched again is not stored twice — the hash is the
    identity — but the caller still learns the id of the existing row.
    """
    row = conn.execute(
        "SELECT id FROM acq_raw_document WHERE source_id=? AND url=? "
        "AND content_hash=?", (source_id, resp.url, resp.hash)).fetchone()
    if row:
        return int(row[0])
    cur = conn.execute(
        "INSERT INTO acq_raw_document (source_id, url, retrieved_utc, "
        "http_status, content_hash, content_type, parser_version, payload) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (source_id, resp.url, resp.retrieved_utc, resp.status, resp.hash,
         resp.content_type, parser_version,
         resp.text if keep_payload else None))
    return int(cur.lastrowid)


def last_availability(conn, season: str) -> dict[int, tuple]:
    """Most recent recorded state per player, for change detection."""
    rows = conn.execute(
        "SELECT player_id, status, chance_next, news FROM ("
        "  SELECT player_id, status, chance_next, news, ROW_NUMBER() OVER "
        "    (PARTITION BY player_id ORDER BY observed_utc DESC) rn "
        "  FROM acq_player_availability WHERE season=?) WHERE rn=1",
        (season,)).fetchall()
    return {int(r[0]): (r[1], r[2], r[3]) for r in rows}


def availability_as_of(conn, season: str, when_utc: str):
    """Availability as it stood at ``when_utc`` — the temporal-integrity API.

    Nothing observed after that instant is returned, which is what lets a
    backtest use this table without leaking.
    """
    return conn.execute(
        "SELECT player_id, status, chance_next, news, observed_utc, "
        "source_published_utc FROM ("
        "  SELECT *, ROW_NUMBER() OVER (PARTITION BY player_id "
        "    ORDER BY observed_utc DESC) rn "
        "  FROM acq_player_availability WHERE season=? AND observed_utc<=?"
        ") WHERE rn=1", (season, when_utc)).fetchall()
