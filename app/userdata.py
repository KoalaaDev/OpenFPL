"""Per-user (per-principal) documents: squad, drafts, transfer watch, prefs.

The planner used to keep ONE `my_team.json`, ONE `drafts.json` and ONE
`transfer_watch.json` on disk. That is fine for a single manager on a laptop
and a data leak the moment the app is on the public internet: every visitor
saw — and overwrote — the same squad. Everything a visitor saves now lives in
a row keyed by their *principal*:

    anon:<session id>   a visitor who has not signed in (cookie-scoped)
    user:<user id>      a Google account

Documents follow the account: when an anonymous visitor signs in, the
documents saved under their anonymous session are moved onto the account,
unless the account already has one of that kind (an older, signed-in save
wins over a fresh anonymous one only if the anonymous one is empty).

Storage is a second SQLite file (`data/app.sqlite`, override with
`$FPLABS_APP_DB`) kept apart from the pipeline database on purpose: the
pipeline file is disposable and rebuilt from free sources; this one holds
people's accounts and must survive a data reset.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

from fpl_engine import config

APP_DB_PATH = os.environ.get("FPLABS_APP_DB") or os.path.join(
    config.DATA_DIR, "app.sqlite")

KINDS = ("my_team", "drafts", "transfer_watch", "prefs")

_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user (
    id          TEXT PRIMARY KEY,
    google_sub  TEXT UNIQUE,
    email       TEXT,
    name        TEXT,
    picture     TEXT,
    plan        TEXT NOT NULL DEFAULT 'free',
    created_at  REAL NOT NULL,
    last_login  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS user_doc (
    principal   TEXT NOT NULL,
    kind        TEXT NOT NULL,
    doc         TEXT NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (principal, kind)
);
CREATE INDEX IF NOT EXISTS user_doc_updated ON user_doc(updated_at);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    path = path or APP_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

def get_doc(principal: str, kind: str, path: str | None = None):
    if kind not in KINDS:
        raise ValueError(f"unknown doc kind {kind!r}")
    with _lock, connect(path) as conn:
        row = conn.execute(
            "SELECT doc FROM user_doc WHERE principal=? AND kind=?",
            (principal, kind)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["doc"])
    except ValueError:
        return None


def set_doc(principal: str, kind: str, doc, path: str | None = None) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown doc kind {kind!r}")
    if doc is None:
        return delete_doc(principal, kind, path)
    blob = json.dumps(doc)
    with _lock, connect(path) as conn:
        conn.execute(
            "INSERT INTO user_doc(principal, kind, doc, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(principal, kind) DO UPDATE SET "
            "doc=excluded.doc, updated_at=excluded.updated_at",
            (principal, kind, blob, time.time()))


def delete_doc(principal: str, kind: str, path: str | None = None) -> None:
    with _lock, connect(path) as conn:
        conn.execute("DELETE FROM user_doc WHERE principal=? AND kind=?",
                     (principal, kind))


def migrate(src: str, dst: str, path: str | None = None) -> list[str]:
    """Move every document from `src` to `dst`; return the kinds moved.

    A kind the destination already holds is kept as it is (the account's own
    save) and the source copy is dropped, so signing in never clobbers what
    the account saved earlier.
    """
    moved: list[str] = []
    with _lock, connect(path) as conn:
        rows = conn.execute(
            "SELECT kind, doc, updated_at FROM user_doc WHERE principal=?",
            (src,)).fetchall()
        for r in rows:
            have = conn.execute(
                "SELECT 1 FROM user_doc WHERE principal=? AND kind=?",
                (dst, r["kind"])).fetchone()
            if have is None:
                conn.execute(
                    "INSERT INTO user_doc(principal, kind, doc, updated_at) "
                    "VALUES (?,?,?,?)", (dst, r["kind"], r["doc"], r["updated_at"]))
                moved.append(r["kind"])
        conn.execute("DELETE FROM user_doc WHERE principal=?", (src,))
    return moved


def purge_anonymous(older_than_days: float = 45.0,
                    path: str | None = None) -> int:
    """Drop anonymous documents nobody has touched in a while."""
    cutoff = time.time() - older_than_days * 86400.0
    with _lock, connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM user_doc WHERE principal LIKE 'anon:%' AND updated_at<?",
            (cutoff,))
        return int(cur.rowcount or 0)


# --------------------------------------------------------------------------
# users
# --------------------------------------------------------------------------

def upsert_user(google_sub: str, email: str, name: str | None,
                picture: str | None, path: str | None = None) -> dict:
    """Create or refresh the account behind a verified Google identity."""
    import uuid
    now = time.time()
    with _lock, connect(path) as conn:
        row = conn.execute("SELECT * FROM user WHERE google_sub=?",
                           (google_sub,)).fetchone()
        if row is None:
            uid = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO user(id, google_sub, email, name, picture, plan, "
                "created_at, last_login) VALUES (?,?,?,?,?,?,?,?)",
                (uid, google_sub, email, name, picture, "free", now, now))
        else:
            uid = row["id"]
            conn.execute(
                "UPDATE user SET email=?, name=?, picture=?, last_login=? "
                "WHERE id=?", (email, name, picture, now, uid))
        row = conn.execute("SELECT * FROM user WHERE id=?", (uid,)).fetchone()
    return dict(row)


def get_user(uid: str, path: str | None = None) -> dict | None:
    with _lock, connect(path) as conn:
        row = conn.execute("SELECT * FROM user WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def set_plan(uid: str, plan: str, path: str | None = None) -> None:
    with _lock, connect(path) as conn:
        conn.execute("UPDATE user SET plan=? WHERE id=?", (plan, uid))


def delete_user(uid: str, path: str | None = None) -> None:
    """Remove the account row and every document it owns."""
    with _lock, connect(path) as conn:
        conn.execute("DELETE FROM user_doc WHERE principal=?", (f"user:{uid}",))
        conn.execute("DELETE FROM user WHERE id=?", (uid,))


def count_users(path: str | None = None) -> int:
    with _lock, connect(path) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM user").fetchone()[0])
