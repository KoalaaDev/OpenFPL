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
