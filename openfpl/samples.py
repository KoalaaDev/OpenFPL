"""Generation of OpenFPL input samples from raw Fantasy Premier League data."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from .data_store import load_json

ROLLING_WINDOWS: Sequence[int] = (1, 3, 5, 10, 38)
RELEVANT_MINUTES_THRESHOLD = 45
POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

PLAYER_STAT_MAP: Mapping[str, str] = {
    "player fpl points": "total_points",
    "player minutes played": "minutes",
    "player influence": "influence",
    "player creativity": "creativity",
    "player threat": "threat",
    "player goals scored": "goals_scored",
    "player penalties missed": "penalties_missed",
    "player assists": "assists",
    "player goals conceded": "goals_conceded",
    "player own goals": "own_goals",
    "player saves": "saves",
    "player penalties saved": "penalties_saved",
    "player yellow cards": "yellow_cards",
    "player red cards": "red_cards",
    "player bps": "bps",
    "player fpl bonus points": "bonus",
    "player shots": "total_shots",
    "player key passes": "key_passes",
    "player xa": "expected_assists",
    "player xg": "expected_goals",
    "player xgchain": "xg_chain",
    "player xgbuildup": "xg_buildup",
}

TEAM_STAT_MAP: Mapping[str, str] = {
    "team goals scored": "goals_for",
    "team goals conceded": "goals_against",
    "team league rank": "league_rank_post",
    "team opponent league rank": "opponent_rank_pre",
    "team xg": "xg",
    "team xga": "xga",
    "team deep allowed": "deep_allowed",
    "team ppda allowed att": "ppda_allowed_att",
    "team ppda allowed def": "ppda_allowed_def",
    "team deep": "deep",
    "team ppda att": "ppda_att",
    "team ppda def": "ppda_def",
}

OPPONENT_PREFIX = "opponent "
# Only a subset of team metrics are mirrored for opponents in the input file.
OPPONENT_FEATURE_BASES = (
    "team goals scored",
    "team goals conceded",
    "team xg",
    "team deep allowed",
    "team ppda allowed att",
    "team ppda allowed def",
    "team xga",
    "team deep",
    "team ppda att",
    "team ppda def",
)
STATUS_COLUMNS = ["status player availability", "status team league rank", "status opponent league rank"]
METADATA_COLUMNS = ["season", "gw", "position", "player", "team", "opponent", "home"]


@dataclass
class PlayerInfo:
    id: int
    first_name: str
    second_name: str
    team_id: int
    element_type: int
    status: str

    @property
    def full_name(self) -> str:
        if self.first_name and self.second_name:
            return f"{self.first_name} {self.second_name}".strip()
        return self.first_name or self.second_name


@dataclass
class FixtureInfo:
    event: int
    team_h: int
    team_a: int
    team_h_score: Optional[int]
    team_a_score: Optional[int]
    finished: bool


@dataclass
class TeamEventRecord:
    event: int
    team_id: int
    opponent_id: int
    was_home: bool
    goals_for: float
    goals_against: float
    league_rank_pre: Optional[float]
    league_rank_post: Optional[float]
    opponent_rank_pre: Optional[float]
    xg: float
    xga: float


STATUS_MAP = {
    "a": 1.0,
    "d": 0.5,
    "i": 0.0,
    "s": 0.0,
    "u": 0.0,
    "n": 0.0,
}


def _infer_season(events: Iterable[dict]) -> str:
    deadlines = [event.get("deadline_time") for event in events if event.get("deadline_time")]
    if not deadlines:
        current_year = datetime.utcnow().year
    else:
        first_deadline = min(deadlines)
        current_year = datetime.fromisoformat(first_deadline.replace("Z", "+00:00")).year
    return f"{current_year}-{(current_year + 1) % 100:02d}"


def _build_player_lookup(bootstrap: dict) -> Dict[int, PlayerInfo]:
    players = {}
    for element in bootstrap.get("elements", []):
        players[element["id"]] = PlayerInfo(
            id=element["id"],
            first_name=element.get("first_name", ""),
            second_name=element.get("second_name", ""),
            team_id=element.get("team"),
            element_type=element.get("element_type"),
            status=element.get("status", ""),
        )
    return players


def _build_team_lookup(bootstrap: dict) -> Dict[int, str]:
    return {team["id"]: team.get("name", str(team["id"])) for team in bootstrap.get("teams", [])}


def _build_fixtures(fixtures_payload: Iterable[dict]) -> Dict[int, List[FixtureInfo]]:
    fixtures_by_event: Dict[int, List[FixtureInfo]] = defaultdict(list)
    for fixture in fixtures_payload:
        event = fixture.get("event")
        if event is None:
            continue
        fixtures_by_event[event].append(
            FixtureInfo(
                event=event,
                team_h=fixture["team_h"],
                team_a=fixture["team_a"],
                team_h_score=fixture.get("team_h_score"),
                team_a_score=fixture.get("team_a_score"),
                finished=fixture.get("finished", False),
            )
        )
    return fixtures_by_event


def _build_fixture_lookup(fixtures_by_event: Mapping[int, Sequence[FixtureInfo]]) -> Dict[int, Dict[int, Dict[str, object]]]:
    lookup: Dict[int, Dict[int, Dict[str, object]]] = defaultdict(dict)
    for event, fixtures in fixtures_by_event.items():
        for fixture in fixtures:
            lookup[event][fixture.team_h] = {"opponent": fixture.team_a, "was_home": True}
            lookup[event][fixture.team_a] = {"opponent": fixture.team_h, "was_home": False}
    return lookup


def _collect_player_event_stats(
    event_payloads: Mapping[int, dict],
    player_lookup: Mapping[int, PlayerInfo],
    fixture_lookup: Mapping[int, Mapping[int, Mapping[str, object]]],
) -> pd.DataFrame:
    rows: List[dict] = []
    for event_id, payload in event_payloads.items():
        team_mapping = fixture_lookup.get(event_id, {})
        for element in payload.get("elements", []):
            player_id = element["id"]
            player = player_lookup.get(player_id)
            if not player:
                continue
            team_info = team_mapping.get(player.team_id)
            if not team_info:
                continue
            stats = element.get("stats", {})
            record = {
                "event": event_id,
                "player_id": player_id,
                "team_id": player.team_id,
                "opponent_team_id": team_info["opponent"],
                "was_home": bool(team_info["was_home"]),
            }
            for alias, key in PLAYER_STAT_MAP.items():
                value = stats.get(key)
                if value in ("", None):
                    record[key] = np.nan
                else:
                    try:
                        record[key] = float(value)
                    except (TypeError, ValueError):
                        record[key] = np.nan
            record["minutes"] = float(stats.get("minutes", 0))
            record["total_points"] = float(stats.get("total_points", 0))
            for key in (
                "goals_scored",
                "assists",
                "goals_conceded",
                "own_goals",
                "penalties_saved",
                "penalties_missed",
                "yellow_cards",
                "red_cards",
                "saves",
                "bps",
                "bonus",
                "total_shots",
                "key_passes",
            ):
                record[key] = float(stats.get(key, 0))
            rows.append(record)
    if not rows:
        return pd.DataFrame(columns=["event", "player_id", "team_id"])
    df = pd.DataFrame(rows)
    df.sort_values(["player_id", "event"], inplace=True)
    return df


def _calculate_table_ranks(table: Mapping[int, MutableMapping[str, float]]) -> Dict[int, float]:
    sorted_teams = sorted(
        table.items(),
        key=lambda item: (
            -item[1]["points"],
            -item[1]["goal_difference"],
            -item[1]["goals_for"],
        ),
    )
    return {team_id: float(index + 1) for index, (team_id, _) in enumerate(sorted_teams)}


def _update_table(table: Dict[int, Dict[str, float]], fixture: FixtureInfo) -> None:
    if fixture.team_h_score is None or fixture.team_a_score is None:
        return
    home, away = fixture.team_h, fixture.team_a
    home_goals = float(fixture.team_h_score)
    away_goals = float(fixture.team_a_score)
    table.setdefault(home, {"points": 0.0, "goal_difference": 0.0, "goals_for": 0.0, "goals_against": 0.0})
    table.setdefault(away, {"points": 0.0, "goal_difference": 0.0, "goals_for": 0.0, "goals_against": 0.0})

    if home_goals > away_goals:
        table[home]["points"] += 3
    elif home_goals < away_goals:
        table[away]["points"] += 3
    else:
        table[home]["points"] += 1
        table[away]["points"] += 1

    table[home]["goals_for"] += home_goals
    table[home]["goals_against"] += away_goals
    table[home]["goal_difference"] = table[home]["goals_for"] - table[home]["goals_against"]

    table[away]["goals_for"] += away_goals
    table[away]["goals_against"] += home_goals
    table[away]["goal_difference"] = table[away]["goals_for"] - table[away]["goals_against"]


def _build_team_event_records(
    fixtures_by_event: Mapping[int, Sequence[FixtureInfo]],
    player_event_df: pd.DataFrame,
    next_event_id: int,
) -> tuple[Dict[int, List[TeamEventRecord]], Dict[int, float]]:
    table: Dict[int, Dict[str, float]] = {}
    pre_ranks: Dict[int, Dict[int, float]] = {}
    post_ranks: Dict[int, Dict[int, float]] = {}

    events = [event for event in sorted(fixtures_by_event) if event < next_event_id]

    for event in events:
        pre_ranks[event] = _calculate_table_ranks(table)
        for fixture in fixtures_by_event[event]:
            _update_table(table, fixture)
        post_ranks[event] = _calculate_table_ranks(table)

    team_records: Dict[int, List[TeamEventRecord]] = defaultdict(list)

    if player_event_df.empty:
        grouped_xg = pd.DataFrame()
    else:
        grouped_xg = (
            player_event_df[player_event_df["event"] < next_event_id]
            .groupby(["event", "team_id"])
            .agg({"expected_goals": "sum"})
            .rename(columns={"expected_goals": "xg"})
        )

    for event in events:
        for fixture in fixtures_by_event[event]:
            if fixture.team_h_score is None or fixture.team_a_score is None:
                continue
            def _xg(event_id: int, team_id: int) -> float:
                if grouped_xg.empty:
                    return float("nan")
                try:
                    return float(grouped_xg.loc[(event_id, team_id), "xg"])
                except KeyError:
                    return float("nan")

            home_xg = _xg(event, fixture.team_h)
            away_xg = _xg(event, fixture.team_a)

            pre_rank = pre_ranks.get(event, {})
            post_rank = post_ranks.get(event, {})

            team_records[fixture.team_h].append(
                TeamEventRecord(
                    event=event,
                    team_id=fixture.team_h,
                    opponent_id=fixture.team_a,
                    was_home=True,
                    goals_for=float(fixture.team_h_score),
                    goals_against=float(fixture.team_a_score),
                    league_rank_pre=pre_rank.get(fixture.team_h),
                    league_rank_post=post_rank.get(fixture.team_h),
                    opponent_rank_pre=pre_rank.get(fixture.team_a),
                    xg=home_xg,
                    xga=away_xg,
                )
            )
            team_records[fixture.team_a].append(
                TeamEventRecord(
                    event=event,
                    team_id=fixture.team_a,
                    opponent_id=fixture.team_h,
                    was_home=False,
                    goals_for=float(fixture.team_a_score),
                    goals_against=float(fixture.team_h_score),
                    league_rank_pre=pre_rank.get(fixture.team_a),
                    league_rank_post=post_rank.get(fixture.team_a),
                    opponent_rank_pre=pre_rank.get(fixture.team_h),
                    xg=away_xg,
                    xga=home_xg,
                )
            )
    final_ranks = _calculate_table_ranks(table)
    return team_records, final_ranks


def _compute_player_features(history: pd.DataFrame) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if history.empty:
        for stat in PLAYER_STAT_MAP:
            for window in ROLLING_WINDOWS:
                features[f"{stat} {window}"] = np.nan
        return features
    for window in ROLLING_WINDOWS:
        window_df = history.tail(window)
        features[f"player fpl points {window}"] = window_df["total_points"].mean()
        relevant = window_df[window_df["minutes"] >= RELEVANT_MINUTES_THRESHOLD]
        features[f"player relevant fpl points {window}"] = relevant["total_points"].mean() if not relevant.empty else np.nan
        for alias, column in PLAYER_STAT_MAP.items():
            if alias in {"player fpl points", "player relevant fpl points"}:
                continue
            col_values = window_df[column] if column in window_df else pd.Series(dtype=float)
            features[f"{alias} {window}"] = col_values.mean() if not col_values.empty else np.nan
    return features


def _compute_team_features(records: Sequence[TeamEventRecord]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if not records:
        for alias in TEAM_STAT_MAP:
            for window in ROLLING_WINDOWS:
                features[f"{alias} {window}"] = np.nan
        return features

    df = pd.DataFrame([r.__dict__ for r in records])
    for window in ROLLING_WINDOWS:
        window_df = df.tail(window)
        for alias, column in TEAM_STAT_MAP.items():
            if column not in window_df:
                features[f"{alias} {window}"] = np.nan
            else:
                features[f"{alias} {window}"] = window_df[column].mean()
    return features


def _build_expected_columns(sample_template_path: Optional[Path]) -> List[str]:
    if sample_template_path and sample_template_path.exists():
        with sample_template_path.open("r", encoding="utf-8") as fh:
            header_line = fh.readline().strip()
        if header_line:
            return header_line.split(",")
    columns: List[str] = list(METADATA_COLUMNS)
    for window in ROLLING_WINDOWS:
        columns.append(f"player fpl points {window}")
        columns.append(f"player relevant fpl points {window}")
    for alias in PLAYER_STAT_MAP:
        if alias == "player fpl points":
            continue
        for window in ROLLING_WINDOWS:
            columns.append(f"{alias} {window}")
    for alias in TEAM_STAT_MAP:
        for window in ROLLING_WINDOWS:
            columns.append(f"{alias} {window}")
    for alias in OPPONENT_FEATURE_BASES:
        suffix = alias[len("team ") :]
        for window in ROLLING_WINDOWS:
            columns.append(f"{OPPONENT_PREFIX}{suffix} {window}")
    columns.extend(STATUS_COLUMNS)
    return columns


def _build_upcoming_fixtures(fixtures_by_event: Mapping[int, Sequence[FixtureInfo]], event_id: int) -> Dict[int, List[Dict[str, object]]]:
    upcoming: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for fixture in fixtures_by_event.get(event_id, []):
        upcoming[fixture.team_h].append({"opponent": fixture.team_a, "home": True})
        upcoming[fixture.team_a].append({"opponent": fixture.team_h, "home": False})
    return upcoming


def generate_samples(
    raw_data_dir: Path,
    output_path: Path,
    *,
    sample_template: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate the OpenFPL input samples for the upcoming gameweek."""

    raw_data_dir = Path(raw_data_dir)
    output_path = Path(output_path)

    bootstrap = load_json(raw_data_dir / "bootstrap-static.json")
    fixtures_payload = load_json(raw_data_dir / "fixtures.json")

    event_payloads: Dict[int, dict] = {}
    events_dir = raw_data_dir / "events"
    if events_dir.exists():
        for path in events_dir.glob("*.json"):
            event_payloads[int(path.stem)] = load_json(path)

    events = bootstrap.get("events", [])
    next_event = next((event for event in events if event.get("is_next")), None)
    if not next_event:
        raise ValueError("Unable to determine upcoming gameweek from bootstrap data")
    next_event_id = next_event["id"]

    season = _infer_season(events)

    fixtures_by_event = _build_fixtures(fixtures_payload)
    fixture_lookup = _build_fixture_lookup(fixtures_by_event)

    player_lookup = _build_player_lookup(bootstrap)
    team_lookup = _build_team_lookup(bootstrap)

    player_event_df = _collect_player_event_stats(event_payloads, player_lookup, fixture_lookup)

    team_records, final_ranks = _build_team_event_records(fixtures_by_event, player_event_df, next_event_id)
    team_features: Dict[int, Dict[str, float]] = {}
    for team_id, records in team_records.items():
        team_features[team_id] = _compute_team_features(records)

    upcoming_fixtures = _build_upcoming_fixtures(fixtures_by_event, next_event_id)

    columns = _build_expected_columns(sample_template)

    rows: List[Dict[str, object]] = []
    current_team_ranks: Dict[int, Optional[float]] = {}
    if final_ranks:
        current_team_ranks.update(final_ranks)
    if team_records:
        for team_id, records in team_records.items():
            if records and records[-1].league_rank_post is not None:
                current_team_ranks.setdefault(team_id, records[-1].league_rank_post)

    for player_id, info in player_lookup.items():
        team_id = info.team_id
        upcoming = upcoming_fixtures.get(team_id)
        if not upcoming:
            continue
        history = player_event_df[
            (player_event_df["player_id"] == player_id)
            & (player_event_df["event"] < next_event_id)
        ]
        player_features = _compute_player_features(history)
        team_feature_values = team_features.get(team_id, {})
        for fixture in upcoming:
            opponent_id = fixture["opponent"]
            opponent_features = team_features.get(opponent_id, {})
            row: Dict[str, object] = {
                "season": season,
                "gw": next_event_id,
                "position": POSITION_MAP.get(info.element_type, ""),
                "player": info.full_name,
                "team": team_lookup.get(team_id, str(team_id)),
                "opponent": team_lookup.get(opponent_id, str(opponent_id)),
                "home": bool(fixture["home"]),
            }
            row.update(player_features)
            row.update(team_feature_values)
            for alias, value in opponent_features.items():
                if alias.startswith("team league rank") or alias.startswith("team opponent league rank"):
                    continue
                if not any(alias.startswith(base) for base in OPPONENT_FEATURE_BASES):
                    continue
                suffix = alias[len("team ") :]
                row[f"{OPPONENT_PREFIX}{suffix}"] = value
            row["status player availability"] = STATUS_MAP.get(info.status, np.nan)
            row["status team league rank"] = current_team_ranks.get(team_id)
            row["status opponent league rank"] = current_team_ranks.get(opponent_id)
            rows.append(row)

    samples_df = pd.DataFrame(rows)

    for column in columns:
        if column not in samples_df:
            samples_df[column] = np.nan
    samples_df = samples_df[columns]
    samples_df.sort_values(["team", "player"], inplace=True)
    samples_df.to_csv(output_path, index=False)
    return samples_df


__all__ = ["generate_samples"]
