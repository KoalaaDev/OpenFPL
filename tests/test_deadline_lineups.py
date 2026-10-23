"""The predicted-XI panel's contract with the resolver.

Both of these were live, silent and wrong: the desk handed `lineup_feed.
resolve` a frame with no `full_name`/`short_name`, and then treated its
(resolved, unresolved, mismatched) TRIPLE as a dict. A bare `except` turned
both into "0 of 11 resolved" for every club — which the panel then rendered
as 175 players the feed disagreed with the model about. A resolution failure
must never be able to look like a finding.
"""
import pandas as pd

from app import deadline
from fpl_engine import lineup_feed as lf


def players():
    return pd.DataFrame([
        {"player_id": 1, "full_name": "David Raya", "web_name": "Raya",
         "team_id": 1, "short_name": "ARS"},
        {"player_id": 2, "full_name": "Bukayo Saka", "web_name": "Saka",
         "team_id": 1, "short_name": "ARS"},
        {"player_id": 3, "full_name": "Ollie Watkins", "web_name": "Watkins",
         "team_id": 2, "short_name": "AVL"},
    ])


def test_resolve_takes_the_frame_the_desk_builds():
    """The columns are the contract; losing one used to cost every club."""
    resolved, unresolved, _ = lf.resolve({"ARS": {"Raya", "Saka"}}, players())
    assert resolved == {("ARS", "Raya"): 1, ("ARS", "Saka"): 2}
    assert unresolved == []


def test_resolve_returns_a_triple_not_a_mapping():
    out = lf.resolve({"ARS": {"Raya"}}, players())
    assert isinstance(out, tuple) and len(out) == 3
    assert not hasattr(out, "values")


def test_an_unmatched_name_is_reported_not_dropped():
    _, unresolved, _ = lf.resolve({"ARS": {"Nobody At All"}}, players())
    assert unresolved == [("ARS", "Nobody At All")]


def test_the_desk_frame_carries_every_column_resolve_reads():
    """Pins the exact SELECT the desk runs against what the resolver touches,
    so a future trim of either side fails here instead of in production."""
    import inspect
    src = inspect.getsource(deadline._lineups)
    for col in ("full_name", "web_name", "short_name", "team_id"):
        assert col in src, col


def _write_feed(tmp_path, src, rows):
    d = tmp_path / "collected" / f"lineups_{src}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "2026-27.csv", index=False)


def _rows(team, formation, players, observed="2026-09-16T05:00:00+00:00", status="predicted"):
    out, slot = [], 0
    for row, names in enumerate(players, start=1):
        for n in names:
            slot += 1
            out.append({"observed_utc": observed, "gw": 5, "source": "x", "team_abbr": team,
                        "opponent_abbr": "", "status": status, "formation": formation,
                        "row": row, "slot": slot, "player": n, "short": "", "kickoff_text": ""})
    return out


LEEDS = [["Trafford"], ["Justin", "Elvedi", "Muharemovic"],
         ["Bogle", "Stach", "Ampadu", "Tanaka", "Gudmundsson"], ["Calvert-Lewin", "Okafor"]]


def test_the_formation_is_the_one_the_feed_states(tmp_path, monkeypatch):
    """RotoWire's positions were a template (five patterns across 239 XIs), so
    the shape now comes from a feed that STATES one, drawn by its own rows."""
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    _write_feed(tmp_path, "sportsgambler", _rows("LEE", "3-5-2", LEEDS))
    out = deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"))
    assert out["LEE"]["formation"] == "3-5-2"
    assert [len(r) for r in out["LEE"]["rows"]] == [1, 3, 5, 2]
    assert out["LEE"]["source"] == "sportsgambler"


def test_sportsgambler_wins_and_ffscout_only_fills_gaps(tmp_path, monkeypatch):
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    _write_feed(tmp_path, "sportsgambler", _rows("LEE", "3-5-2", LEEDS))
    _write_feed(tmp_path, "ffscout", _rows("LEE", "3-4-3", LEEDS[:2] + [LEEDS[2][:4], LEEDS[3] + ["X"]])
                + _rows("HUL", "5-4-1", [["K"], list("abcde"), list("fghi"), ["j"]]))
    out = deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"))
    assert out["LEE"]["source"] == "sportsgambler" and out["LEE"]["formation"] == "3-5-2"
    assert out["HUL"]["source"] == "ffscout"


def test_a_snapshot_taken_after_now_is_not_used(tmp_path, monkeypatch):
    """The desk shows what was known; a later observation must not leak in."""
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    _write_feed(tmp_path, "sportsgambler",
                _rows("LEE", "3-5-2", LEEDS, observed="2026-09-16T05:00:00+00:00")
                + _rows("LEE", "4-4-2", [["T"], list("abcd"), list("efgh"), ["i", "j"]],
                        observed="2026-09-18T05:00:00+00:00"))
    out = deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"))
    assert out["LEE"]["formation"] == "3-5-2"


def test_a_short_eleven_is_not_drawn(tmp_path, monkeypatch):
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    _write_feed(tmp_path, "sportsgambler", _rows("LEE", "3-5-2", [["Trafford"], ["Justin"]]))
    assert deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC")) == {}


def test_rotowire_does_not_draw_the_shape():
    """It remains the scored feed; it must not come back as a formation source."""
    assert "rotowire" not in deadline.FORMATION_FEEDS
    assert deadline.FORMATION_FEEDS[0] == "sportsgambler"


def test_the_lineup_classes_do_not_reuse_the_mini_league_names():
    """.xi-card (68px wide) belongs to the Mini League; reusing it squeezed
    every club on the Live desk into a strip."""
    import pathlib
    live = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "tabs" / "Live.jsx").read_text(encoding="utf-8")
    assert 'className="xi-' not in live and "`xi-" not in live


# ---- GW8 (23 Oct 2026) ------------------------------------------------------

def test_german_style_transliterations_resolve():
    """SportsGambler writes Oedegaard / Lindeloef / Roehl for Ødegaard /
    Lindelöf / Röhl; the accent-stripped key differs by an 'e'."""
    p = pd.DataFrame([
        {"player_id": 7, "full_name": "Martin Ødegaard", "web_name": "Ødegaard",
         "team_id": 1, "short_name": "ARS"},
        {"player_id": 8, "full_name": "Victor Nilsson Lindelöf", "web_name": "Lindelöf",
         "team_id": 2, "short_name": "AVL"},
        {"player_id": 9, "full_name": "Merlin Röhl", "web_name": "Röhl",
         "team_id": 3, "short_name": "EVE"}])
    resolved, unresolved, _ = lf.resolve(
        {"ARS": {"Martin Oedegaard"}, "AVL": {"V. Nilsson Lindeloef"}, "EVE": {"Merlin Roehl"}}, p)
    assert unresolved == []
    assert resolved == {("ARS", "Martin Oedegaard"): 7, ("AVL", "V. Nilsson Lindeloef"): 8,
                        ("EVE", "Merlin Roehl"): 9}


def test_last_weeks_team_sheet_filed_under_this_week_is_not_drawn(tmp_path, monkeypatch):
    """The collectors stamp rows with FPL's NEXT gameweek at run time, so a
    Saturday team sheet captured after the deadline is filed under the
    following gameweek. Its stated kickoff gives it away."""
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    stale = [dict(r, kickoff_text="Saturday 12 September 15:00", status="confirmed")
             for r in _rows("LEE", "3-4-3", LEEDS, observed="2026-09-13T14:00:00+00:00")]
    fresh = [dict(r, kickoff_text="Saturday 19 September 15:00")
             for r in _rows("LEE", "3-5-2", LEEDS, observed="2026-09-16T05:00:00+00:00")]
    _write_feed(tmp_path, "sportsgambler", stale + fresh)
    deadline_gw5 = pd.Timestamp("2026-09-18T17:30", tz="UTC")
    out = deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"), deadline_gw5)
    assert out["LEE"]["formation"] == "3-5-2" and out["LEE"]["status"] == "predicted"
    # and with only the stale sheet, nothing is drawn rather than last week's
    _write_feed(tmp_path, "sportsgambler", stale)
    assert deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"), deadline_gw5) == {}


def test_the_servers_own_archive_is_read_and_the_newest_wins(tmp_path, monkeypatch):
    """The repository archive reaches a running server only on redeploy; the
    news loop's refresh lands in `live_lineups_dir`, and a fresher snapshot
    there must win."""
    from fpl_engine import config
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    _write_feed(tmp_path, "sportsgambler", _rows("LEE", "3-4-3", LEEDS, observed="2026-09-14T05:00:00+00:00"))
    live = tmp_path / "web_cache" / "live_feeds" / "lineups_sportsgambler"
    live.mkdir(parents=True)
    pd.DataFrame(_rows("LEE", "3-5-2", LEEDS, observed="2026-09-16T05:00:00+00:00")).to_csv(
        live / "2026-27.csv", index=False)
    out = deadline._feed_xis("2026-27", 5, pd.Timestamp("2026-09-17", tz="UTC"))
    assert out["LEE"]["formation"] == "3-5-2"
    assert out["LEE"]["observed"].startswith("2026-09-16")


def test_a_failing_refresh_never_raises(monkeypatch):
    from app import services
    monkeypatch.setattr(services, "bootstrap", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert "error" in deadline.refresh_feeds()
