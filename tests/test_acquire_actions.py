"""The file-based Actions collector: append-only, idempotent, point-in-time.

No network: bootstrap payloads are injected. The properties tested are the
ones a year of scheduled runs depends on — an unchanged payload appends
nothing, a changed state appends exactly the change, ownership freezes to
the last pre-deadline write, and the importer replays the files into the
SQLite change log idempotently.
"""
import gzip
import json
import os

import pytest

from acquire import actions


def _boot(status="a", chance=None, news="", next_gw=3, code=1001):
    return json.dumps({
        "events": [
            {"id": 1, "deadline_time": "2026-08-15T17:30:00Z", "finished": True},
            {"id": 2, "deadline_time": "2026-08-22T17:30:00Z",
             "is_current": True},
            {"id": next_gw, "deadline_time": "2026-08-29T17:30:00Z",
             "is_next": True},
        ],
        "elements": [{
            "id": 7, "code": code, "web_name": "Saka", "status": status,
            "chance_of_playing_next_round": chance, "news": news,
            "news_added": "2026-08-20T10:00:00Z" if news else None,
            "now_cost": 105, "selected_by_percent": "45.3",
            "transfers_in_event": 1000, "transfers_out_event": 50,
            "ep_next": "6.1", "form": "5.5",
        }],
    })


def _lines(out):
    path = os.path.join(out, "availability.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def test_first_run_records_everyone_second_run_records_nothing(tmp_path):
    out = str(tmp_path)
    actions.collect(out, payload=_boot())
    assert len(_lines(out)) == 1
    actions.collect(out, payload=_boot())
    assert len(_lines(out)) == 1          # unchanged payload appends nothing


def test_a_state_change_appends_exactly_one_line(tmp_path):
    out = str(tmp_path)
    actions.collect(out, payload=_boot())
    actions.collect(out, payload=_boot(status="d", chance=75,
                                       news="Knock - 75% chance of playing"))
    lines = _lines(out)
    assert len(lines) == 2
    assert lines[-1]["status"] == "d"
    assert lines[-1]["chance_next"] == 0.75
    # verbatim, never flattened to "available"/"doubtful"
    assert lines[-1]["news"] == "Knock - 75% chance of playing"
    assert lines[-1]["news_added"] == "2026-08-20T10:00:00Z"


def test_ownership_named_by_next_gw_and_snapshot_written(tmp_path):
    out = str(tmp_path)
    res = actions.collect(out, payload=_boot(next_gw=3))
    assert res["ownership"] == "gw03.csv"
    own = open(os.path.join(out, "ownership", "gw03.csv")).read()
    assert "45.3" in own
    snap = os.path.join(out, "snapshots", res["snapshot"])
    with gzip.open(snap, "rt") as fh:
        assert "selected_by_percent" in fh.readline()
    # the deadline passing moves is_next on; the old file is left frozen
    res2 = actions.collect(out, payload=_boot(next_gw=4))
    assert res2["ownership"] == "gw04.csv"
    assert open(os.path.join(out, "ownership", "gw03.csv")).read() == own


def test_picks_are_skipped_without_a_panel(tmp_path):
    res = actions.collect(str(tmp_path), payload=_boot())
    assert res["picks"] == []


def test_import_replays_the_archive_idempotently(tmp_path):
    out = str(tmp_path)
    actions.collect(out, payload=_boot())
    actions.collect(out, payload=_boot(status="i", news="Ankle injury"))
    # a picks file, as the collector would write it
    pdir = os.path.join(out, "picks", "2026-27")
    os.makedirs(pdir)
    with open(os.path.join(pdir, "gw02.csv"), "w", newline="") as fh:
        fh.write("gw,entry_id,element,slot,multiplier,is_captain,is_vice,"
                 "active_chip,observed_utc\n"
                 "2,42,7,1,2,1,0,,2026-08-23T00:00:00Z\n")
    import sqlite3
    conn = sqlite3.connect(os.path.join(str(tmp_path), "t.sqlite"))
    res = actions.import_collected(conn, out)
    assert res == {"availability_rows": 2, "pick_rows": 1}
    res2 = actions.import_collected(conn, out)
    assert res2 == {"availability_rows": 0, "pick_rows": 0}
    got = conn.execute(
        "SELECT status, news, source_published_utc FROM "
        "acq_player_availability ORDER BY observed_utc").fetchall()
    assert got[0][0] == "a" and got[1][:2] == ("i", "Ankle injury")
    # the no-model-import architecture rule is enforced for actions.py by
    # test_acquire.test_the_acquirer_does_not_import_the_model (static scan
    # of every file under acquire/), so it is not duplicated here
