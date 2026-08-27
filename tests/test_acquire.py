"""Acquisition engine: change-log semantics, provenance, temporal integrity.

The property the whole design exists for is the last one — a backtest must be
able to ask what was knowable at an instant and get an answer that contains
nothing from after it.
"""
import json
import os
import tempfile

import pytest

from acquire import storage, validate
from acquire.core import http
from acquire.sources import fpl_availability as fa


def _boot(players, deadline="2026-08-14T17:30:00Z"):
    return json.dumps({
        "events": [{"deadline_time": deadline}],
        "elements": [
            {"id": p["id"], "status": p.get("status", "a"),
             "chance_of_playing_next_round": p.get("chance"),
             "news": p.get("news", ""), "news_added": p.get("added")}
            for p in players
        ],
    })


class _Resp:
    def __init__(self, text, url="http://x/y", status=200, when="2026-08-10T12:00:00Z"):
        self.text, self.url, self.status = text, url, status
        self.retrieved_utc, self.content_type, self.error = when, "application/json", ""

    @property
    def ok(self):
        return self.status == 200

    @property
    def hash(self):
        return http.content_hash(self.text)


@pytest.fixture()
def conn():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    from fpl_engine import db
    db.init_db(path)
    c = storage.connect(path)
    storage.init(c)
    fa.register(c)
    c.commit()
    yield c
    c.close()
    try:
        os.remove(path)
    except PermissionError:
        pass


# ------------------------------------------------------------------ parse ---
def test_missing_chance_stays_none_not_zero(conn):
    """A source that is silent is not a source saying zero."""
    rows = {r["player_id"]: r for r in fa.parse(_boot([
        {"id": 1},                       # no chance field at all
        {"id": 2, "chance": 0},          # explicitly zero
        {"id": 3, "chance": 75},
    ]))}
    assert rows[1]["chance_next"] is None
    assert rows[2]["chance_next"] == 0.0
    assert rows[3]["chance_next"] == pytest.approx(0.75)


def test_news_is_kept_verbatim(conn):
    """'75% chance of playing' must not be flattened into 'available'."""
    txt = "Thigh injury - 75% chance of playing"
    got = fa.parse(_boot([{"id": 1, "status": "d", "chance": 75, "news": txt}]))
    assert got[0]["news"] == txt


# -------------------------------------------------------------- change log ---
def test_only_changes_are_recorded(conn):
    p = [{"id": 1, "status": "a"}, {"id": 2, "status": "i", "news": "Knee"}]
    n1 = fa.ingest_payload(conn, _boot(p), season="2026-27",
                           observed_utc="2026-08-10T12:00:00Z", raw_id=None)
    assert n1 == 2
    n2 = fa.ingest_payload(conn, _boot(p), season="2026-27",
                           observed_utc="2026-08-11T12:00:00Z", raw_id=None)
    assert n2 == 0, "an unchanged payload must not write rows"
    p[1]["news"] = "Knee injury - Expected back 21 Aug"
    n3 = fa.ingest_payload(conn, _boot(p), season="2026-27",
                           observed_utc="2026-08-12T12:00:00Z", raw_id=None)
    assert n3 == 1, "only the player whose state moved"


def test_reingesting_is_idempotent(conn):
    p = [{"id": 1, "status": "d", "chance": 50}]
    for _ in range(3):
        fa.ingest_payload(conn, _boot(p), season="2026-27",
                          observed_utc="2026-08-10T12:00:00Z", raw_id=None)
    n = conn.execute("SELECT COUNT(*) FROM acq_player_availability").fetchone()[0]
    assert n == 1


# ------------------------------------------------------- temporal integrity ---
def test_as_of_never_returns_the_future(conn):
    """The single most important property: no leakage past the decision time."""
    fa.ingest_payload(conn, _boot([{"id": 1, "status": "a"}]), season="2026-27",
                      observed_utc="2026-08-10T12:00:00Z", raw_id=None)
    fa.ingest_payload(conn, _boot([{"id": 1, "status": "i", "news": "Torn"}]),
                      season="2026-27", observed_utc="2026-08-20T12:00:00Z",
                      raw_id=None)
    before = {r["player_id"]: r for r in
              storage.availability_as_of(conn, "2026-27", "2026-08-15T00:00:00Z")}
    assert before[1]["status"] == "a", "the later injury must be invisible"
    after = {r["player_id"]: r for r in
             storage.availability_as_of(conn, "2026-27", "2026-08-25T00:00:00Z")}
    assert after[1]["status"] == "i"
    assert after[1]["news"] == "Torn"


def test_as_of_returns_the_latest_state_at_or_before_the_instant(conn):
    for day, st in (("10", "a"), ("12", "d"), ("14", "i")):
        fa.ingest_payload(conn, _boot([{"id": 1, "status": st, "news": st}]),
                          season="2026-27",
                          observed_utc=f"2026-08-{day}T12:00:00Z", raw_id=None)
    got = storage.availability_as_of(conn, "2026-27", "2026-08-13T00:00:00Z")
    assert got[0]["status"] == "d"


def test_source_published_time_is_preserved(conn):
    """FPL's own news_added lets a later snapshot still date a standing item."""
    fa.ingest_payload(conn, _boot([{"id": 1, "status": "i", "news": "Knee",
                                    "added": "2026-07-23T12:01:23Z"}]),
                      season="2026-27", observed_utc="2026-08-20T12:00:00Z",
                      raw_id=None)
    r = storage.availability_as_of(conn, "2026-27", "2026-08-21T00:00:00Z")[0]
    assert r["source_published_utc"] == "2026-07-23T12:01:23Z"
    assert r["observed_utc"] > r["source_published_utc"]


# ------------------------------------------------------------- provenance ---
def test_identical_content_is_stored_once(conn):
    r = _Resp(_boot([{"id": 1}]))
    a = storage.store_raw(conn, "fpl_api", r, parser_version="1")
    b = storage.store_raw(conn, "fpl_api", r, parser_version="1")
    assert a == b
    n = conn.execute("SELECT COUNT(*) FROM acq_raw_document").fetchone()[0]
    assert n == 1


def test_changed_content_is_stored_separately(conn):
    a = storage.store_raw(conn, "fpl_api", _Resp(_boot([{"id": 1}])),
                          parser_version="1")
    b = storage.store_raw(conn, "fpl_api",
                          _Resp(_boot([{"id": 1, "status": "i"}])),
                          parser_version="1")
    assert a != b


def test_an_observation_can_be_traced_to_its_response(conn):
    r = _Resp(_boot([{"id": 1, "status": "i", "news": "Knee"}]))
    raw_id = storage.store_raw(conn, "fpl_api", r, parser_version="1")
    fa.ingest_payload(conn, r.text, season="2026-27",
                      observed_utc=r.retrieved_utc, raw_id=raw_id)
    row = conn.execute("SELECT raw_id FROM acq_player_availability").fetchone()
    doc = conn.execute("SELECT url, content_hash FROM acq_raw_document "
                       "WHERE id=?", (row["raw_id"],)).fetchone()
    assert doc["content_hash"] == r.hash


# -------------------------------------------------------------- validation ---
def _checks(conn):
    return {f.check for f in validate.run(conn).errors}


def test_clean_data_validates(conn):
    fa.ingest_payload(conn, _boot([{"id": 1, "status": "a"}]), season="2026-27",
                      observed_utc="2026-08-10T12:00:00Z", raw_id=None)
    assert validate.run(conn).ok()


def test_a_percentage_left_unscaled_is_caught(conn):
    conn.execute("INSERT INTO acq_player_availability (season, player_id, "
                 "observed_utc, source_id, status, chance_next) "
                 "VALUES ('2026-27',1,'2026-08-10T12:00:00Z','fpl_api','d',75)")
    assert "values.chance_is_a_probability" in _checks(conn)


def test_published_after_observed_is_caught(conn):
    conn.execute("INSERT INTO acq_player_availability (season, player_id, "
                 "observed_utc, source_id, source_published_utc, status) "
                 "VALUES ('2026-27',1,'2026-08-10T12:00:00Z','fpl_api',"
                 "'2026-08-20T12:00:00Z','i')")
    assert "temporal.published_before_observed" in _checks(conn)


def test_an_unregistered_source_is_caught(conn):
    conn.execute("INSERT INTO acq_player_availability (season, player_id, "
                 "observed_utc, source_id, status) "
                 "VALUES ('2026-27',1,'2026-08-10T12:00:00Z','mystery','a')")
    assert "provenance.source_registered" in _checks(conn)


def test_an_unknown_status_code_is_caught(conn):
    conn.execute("INSERT INTO acq_player_availability (season, player_id, "
                 "observed_utc, source_id, status) "
                 "VALUES ('2026-27',1,'2026-08-10T12:00:00Z','fpl_api','?')")
    assert "values.status_vocabulary" in _checks(conn)


# ------------------------------------------------------------ independence ---
def test_the_acquirer_does_not_import_the_model():
    """Architecture rule: data in, no modelling. Only db access is shared."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "acquire"
    banned = ("xpts", "predict", "optimise", "backtest", "features",
              "price_model", "scoring")
    for path in root.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for mod in banned:
            assert f"fpl_engine.{mod}" not in src and f"from ..{mod}" not in src, \
                f"{path.name} reaches into the modelling engine ({mod})"
