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


def test_forecasts_names_strips_the_timestamp():
    f = {"ARS": ("2026-09-18T09:00:00+00:00", {"Raya", "Saka"})}
    assert deadline.forecasts_names(f) == {"ARS": {"Raya", "Saka"}}


def test_the_desk_frame_carries_every_column_resolve_reads():
    """Pins the exact SELECT the desk runs against what the resolver touches,
    so a future trim of either side fails here instead of in production."""
    import inspect
    src = inspect.getsource(deadline._lineups)
    for col in ("full_name", "web_name", "short_name", "team_id"):
        assert col in src, col


def test_formation_uses_the_feeds_positions_not_fpls_labels():
    """FPL calls a winger a MID, so a 4-2-3-1 drawn from FPL labels reads as
    4-5-1. The feed's own pitch positions give the real shape."""
    archive = pd.DataFrame([
        {"gw": 5, "team_abbr": "ARS", "status": "predicted", "observed": pd.Timestamp("2026-09-16T05:00Z"),
         "player": n, "position": pos, "slot": str(i + 1)}
        for i, (n, pos) in enumerate([
            ("David Raya", "GK"), ("Left Back", "DL"), ("Centre One", "DC"), ("Centre Two", "DC"),
            ("Right Back", "DR"), ("Holder One", "DMC"), ("Holder Two", "DMC"),
            ("Left Wing", "AML"), ("Ten", "AMC"), ("Bukayo Saka", "AMR"), ("Striker", "FW")])])
    names = set(archive["player"])
    xi, formation = deadline._formation(
        archive, 5, "ARS", "2026-09-16T05:00:00+00:00", names,
        {("ARS", "David Raya"): 1}, {1: {"p_start": 0.91, "disagree": False}},
        {1: {"name": "Raya"}})
    assert formation == "4-2-3-1"
    assert len(xi) == 11
    raya = next(x for x in xi if x["full_name"] == "David Raya")
    assert raya["name"] == "Raya" and raya["p_start"] == 0.91 and raya["resolved"]
    # an unmatched name still stands in the XI, with no model number
    saka = next(x for x in xi if x["full_name"] == "Bukayo Saka")
    assert saka["resolved"] is False and saka["p_start"] is None and saka["name"] == "Saka"
    # left to right across a band
    band4 = [x["position"] for x in xi if x["band"] == 4]
    assert band4 == ["AML", "AMC", "AMR"]


def test_the_lineup_classes_do_not_reuse_the_mini_league_names():
    """.xi-card (68px wide) belongs to the Mini League; reusing it squeezed
    every club on the Live desk into a strip."""
    import pathlib
    live = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "tabs" / "Live.jsx").read_text(encoding="utf-8")
    assert 'className="xi-' not in live and "`xi-" not in live
