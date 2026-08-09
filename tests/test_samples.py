from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfpl.samples import generate_samples


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _build_bootstrap() -> dict:
    return {
        "events": [
            {"id": 1, "finished": True, "deadline_time": "2024-08-10T18:00:00Z", "data_checked": True},
            {"id": 2, "finished": True, "deadline_time": "2024-08-17T18:00:00Z", "data_checked": True},
            {"id": 3, "is_next": True, "deadline_time": "2024-08-24T18:00:00Z"},
        ],
        "teams": [
            {"id": 1, "name": "Alpha FC"},
            {"id": 2, "name": "Beta FC"},
        ],
        "elements": [
            {"id": 101, "first_name": "John", "second_name": "Doe", "team": 1, "element_type": 4, "status": "a"},
            {"id": 102, "first_name": "Jane", "second_name": "Smith", "team": 2, "element_type": 1, "status": "d"},
        ],
    }


def _build_fixtures() -> list[dict]:
    return [
        {"event": 1, "team_h": 1, "team_a": 2, "team_h_score": 2, "team_a_score": 1, "finished": True},
        {"event": 2, "team_h": 2, "team_a": 1, "team_h_score": 1, "team_a_score": 0, "finished": True},
        {"event": 3, "team_h": 1, "team_a": 2, "team_h_score": None, "team_a_score": None, "finished": False},
    ]


def _event_payload(event_id: int) -> dict:
    if event_id == 1:
        return {
            "elements": [
                {
                    "id": 101,
                    "stats": {
                        "minutes": 90,
                        "total_points": 8,
                        "goals_scored": 1,
                        "assists": 1,
                        "goals_conceded": 1,
                        "own_goals": 0,
                        "penalties_saved": 0,
                        "penalties_missed": 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "saves": 0,
                        "bps": 32,
                        "bonus": 3,
                        "total_shots": 3,
                        "key_passes": 2,
                        "influence": "45.0",
                        "creativity": "20.0",
                        "threat": "50.0",
                        "expected_goals": "0.6",
                        "expected_assists": "0.4",
                        "xg_chain": "0.9",
                        "xg_buildup": "0.3",
                    },
                },
                {
                    "id": 102,
                    "stats": {
                        "minutes": 90,
                        "total_points": 2,
                        "goals_scored": 0,
                        "assists": 0,
                        "goals_conceded": 2,
                        "own_goals": 0,
                        "penalties_saved": 0,
                        "penalties_missed": 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "saves": 3,
                        "bps": 20,
                        "bonus": 0,
                        "total_shots": 0,
                        "key_passes": 0,
                        "influence": "10.0",
                        "creativity": "1.0",
                        "threat": "0.5",
                        "expected_goals": "0.0",
                        "expected_assists": "0.0",
                        "xg_chain": "0.0",
                        "xg_buildup": "0.0",
                    },
                },
            ]
        }
    return {
        "elements": [
            {
                "id": 101,
                "stats": {
                    "minutes": 85,
                    "total_points": 2,
                    "goals_scored": 0,
                    "assists": 0,
                    "goals_conceded": 1,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 1,
                    "red_cards": 0,
                    "saves": 0,
                    "bps": 14,
                    "bonus": 0,
                    "total_shots": 1,
                    "key_passes": 1,
                    "influence": "18.0",
                    "creativity": "15.0",
                    "threat": "22.0",
                    "expected_goals": "0.1",
                    "expected_assists": "0.05",
                    "xg_chain": "0.2",
                    "xg_buildup": "0.1",
                },
            },
            {
                "id": 102,
                "stats": {
                    "minutes": 90,
                    "total_points": 7,
                    "goals_scored": 0,
                    "assists": 0,
                    "goals_conceded": 0,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "saves": 4,
                    "bps": 28,
                    "bonus": 2,
                    "total_shots": 0,
                    "key_passes": 0,
                    "influence": "25.0",
                    "creativity": "2.0",
                    "threat": "0.0",
                    "expected_goals": "0.0",
                    "expected_assists": "0.0",
                    "xg_chain": "0.0",
                    "xg_buildup": "0.0",
                },
            },
        ]
    }


def test_generate_samples_basic(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_json(raw_dir / "bootstrap-static.json", _build_bootstrap())
    _write_json(raw_dir / "fixtures.json", _build_fixtures())
    events_dir = raw_dir / "events"
    _write_json(events_dir / "1.json", _event_payload(1))
    _write_json(events_dir / "2.json", _event_payload(2))

    output_path = tmp_path / "samples.csv"
    df = generate_samples(raw_dir, output_path)

    assert output_path.exists()
    assert not df.empty

    john_row = df[df["player"] == "John Doe"].iloc[0]
    assert john_row["gw"] == 3
    assert john_row["team"] == "Alpha FC"
    assert bool(john_row["home"]) is True
    assert np.isclose(john_row["player fpl points 1"], 2.0)
    assert np.isclose(john_row["player fpl points 3"], 5.0)
    assert np.isclose(john_row["player relevant fpl points 3"], 5.0)
    assert np.isclose(john_row["team goals scored 1"], 0.0)
    assert np.isclose(john_row["team goals scored 3"], 1.0)
    assert np.isclose(john_row["opponent goals conceded 1"], 0.0)
    assert john_row["status player availability"] == 1.0
    assert john_row["status opponent league rank"] == 2.0
    assert np.isnan(john_row["team deep allowed 1"])

    jane_row = df[df["player"] == "Jane Smith"].iloc[0]
    assert bool(jane_row["home"]) is False
    assert jane_row["status player availability"] == 0.5
    assert jane_row["status opponent league rank"] == 1.0
    assert np.isclose(jane_row["opponent goals scored 1"], 0.0)

    # The CSV should contain the same columns as the DataFrame
    loaded = pd.read_csv(output_path)
    assert list(loaded.columns) == list(df.columns)
