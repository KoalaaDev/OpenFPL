import pandas as pd

from fpl_engine.xpts import absence_features as ab


def _fixtures(n=6, team=1):
    return pd.DataFrame({
        "season": ["2025-26"] * n, "fixture_id": list(range(100, 100 + n)),
        "kickoff_utc": [(pd.Timestamp("2025-09-10T14:00:00Z") + pd.Timedelta(days=7 * i)).isoformat()
                        for i in range(n)],
        "team_h": [team] * n, "team_a": [99] * n,
    })


def test_a_red_card_is_one_match_whatever_its_kind():
    cards = pd.DataFrame({
        "season": ["2025-26"] * 2, "team_id": [1, 1], "player_code": [7, 9],
        "fixture_id": [100, 100], "yellow_cards": [1, 0], "red_cards": [1, 1]})
    s = ab.suspensions(cards, _fixtures())
    by = s.groupby("player_code")["fixture_id"].apply(sorted).to_dict()
    assert by[7] == [101]
    assert by[9] == [101]


def test_five_yellows_before_matchday_19_is_one_match():
    cards = pd.DataFrame({
        "season": ["2025-26"] * 5, "team_id": [1] * 5, "player_code": [7] * 5,
        "fixture_id": [100, 101, 102, 103, 104], "yellow_cards": [1] * 5, "red_cards": [0] * 5})
    s = ab.suspensions(cards, _fixtures())
    assert sorted(s["fixture_id"]) == [105]


def test_spillover_counts_regular_starters_at_the_same_position_only():
    sus = pd.DataFrame({"season": ["2025-26"], "team_id": [1], "player_code": [7], "fixture_id": [101]})
    frame = pd.DataFrame({
        "season": ["2025-26"] * 4, "team_id": [1] * 4, "fixture_id": [101] * 4,
        "player_code": [7, 8, 9, 10], "position": ["MID", "MID", "DEF", "MID"],
        "starts_l5": [5, 0, 5, 4],
    })
    out = ab.add_features(frame, sus)
    assert out["sus_self"].tolist() == [1.0, 0.0, 0.0, 0.0]
    # the banned man is a regular: his midfield team-mates see one regular out,
    # the defender sees none at his position but one at the club
    assert out["pos_regulars_out"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert out["team_regulars_out"].tolist() == [0.0, 1.0, 1.0, 1.0]


def test_no_suspensions_means_zero_features_not_nan():
    frame = pd.DataFrame({"season": ["2025-26"], "team_id": [1], "fixture_id": [101],
                          "player_code": [7], "position": ["MID"], "starts_l5": [5]})
    out = ab.add_features(frame, pd.DataFrame())
    assert out[ab.FEATURES].sum().sum() == 0
