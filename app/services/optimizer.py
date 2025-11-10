from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pulp

from app.utils.naming import build_name_keys, normalize_name

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_REQUIREMENTS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

POSITION_ALIASES = {
    "GK": "GKP",
    "GKP": "GKP",
    "GOALKEEPER": "GKP",
    "DEF": "DEF",
    "DF": "DEF",
    "DEFENDER": "DEF",
    "MID": "MID",
    "MF": "MID",
    "MIDFIELDER": "MID",
    "FWD": "FWD",
    "FW": "FWD",
    "FORWARD": "FWD",
}


@dataclass
class PlayerProjection:
    element_id: int
    name: str
    position: str
    team_id: int
    team_name: str
    price: float
    prediction: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "element_id": self.element_id,
            "name": self.name,
            "position": self.position,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "price": round(self.price, 1),
            "prediction": round(self.prediction, 2),
        }


class OptimizationError(Exception):
    pass


def load_predictions(data_dir: Path, season: str, gameweek: int) -> Tuple[pd.DataFrame, int]:
    """Load the latest available prediction file at or before the requested gameweek."""

    candidate = gameweek
    while candidate >= 1:
        filename = data_dir / f"predictions_{season}GW{candidate}.csv"
        if filename.exists():
            df = pd.read_csv(filename)
            if df.empty:
                raise OptimizationError("Prediction file is empty")
            return df, candidate
        candidate -= 1

    raise OptimizationError(
        f"Prediction file for season {season} up to GW{gameweek} not found"
    )


def build_player_catalog(bootstrap: Dict[str, object]) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[int, str]]:
    elements = bootstrap.get("elements", [])
    teams = bootstrap.get("teams", [])

    name_index: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for element in elements:
        keys = build_name_keys(
            element.get("web_name"),
            f"{element.get('first_name')} {element.get('second_name')}",
            element.get("first_name"),
            element.get("second_name"),
        )
        for key in keys:
            name_index[key].append(element)

    team_lookup = {team["id"]: team["name"] for team in teams}
    return name_index, team_lookup


def match_projection(row: pd.Series, name_index: Dict[str, List[Dict[str, object]]], team_lookup: Dict[int, str]) -> Optional[PlayerProjection]:
    normalized_player = normalize_name(row.player)
    candidates = name_index.get(normalized_player, [])

    if not candidates:
        # Attempt to split combined names or handle alternative forms
        parts = row.player.split()
        if len(parts) > 1:
            normalized_alt = normalize_name(parts[-1])
            candidates = name_index.get(normalized_alt, [])

    if not candidates:
        return None

    normalized_team = normalize_name(row.team)
    if normalized_team:
        filtered = [element for element in candidates if normalize_name(team_lookup.get(element["team"], "")) == normalized_team]
        if filtered:
            candidates = filtered

    predicted_position_raw = str(row.position).upper()
    predicted_position = POSITION_ALIASES.get(predicted_position_raw, predicted_position_raw)
    if predicted_position:
        filtered = [
            element
            for element in candidates
            if POSITION_MAP.get(element["element_type"], "").upper() == predicted_position
        ]
        if filtered:
            candidates = filtered

    element = candidates[0]
    position = POSITION_MAP.get(element["element_type"])
    if not position:
        return None
    return PlayerProjection(
        element_id=element["id"],
        name=element["web_name"],
        position=position,
        team_id=element["team"],
        team_name=team_lookup.get(element["team"], ""),
        price=element["now_cost"] / 10.0,
        prediction=float(row.prediction),
    )


def build_projection_pool(predictions: pd.DataFrame, name_index: Dict[str, List[Dict[str, object]]], team_lookup: Dict[int, str]) -> List[PlayerProjection]:
    projections: Dict[int, PlayerProjection] = {}
    for _, row in predictions.iterrows():
        projection = match_projection(row, name_index, team_lookup)
        if projection:
            existing = projections.get(projection.element_id)
            if not existing or projection.prediction > existing.prediction:
                projections[projection.element_id] = projection
    if not projections:
        raise OptimizationError("No projections could be matched to FPL players")
    return list(projections.values())


def optimize_squad(
    projections: Iterable[PlayerProjection],
    budget: float,
    current_squad_ids: Optional[Iterable[int]] = None,
    free_transfers: int = 1,
    hit_penalty: float = 4.0,
) -> List[PlayerProjection]:
    projections = list(projections)
    problem = pulp.LpProblem("fpl_squad_optimization", pulp.LpMaximize)

    decision_vars = {proj.element_id: pulp.LpVariable(f"player_{proj.element_id}", cat="Binary") for proj in projections}
    objective = pulp.lpSum(proj.prediction * decision_vars[proj.element_id] for proj in projections)

    current_ids_set = set(current_squad_ids or [])
    if current_ids_set:
        new_player_ids = [proj.element_id for proj in projections if proj.element_id not in current_ids_set]
        if new_player_ids:
            extra_transfers = pulp.LpVariable("extra_transfers", lowBound=0)
            new_player_count = pulp.lpSum(decision_vars[player_id] for player_id in new_player_ids)
            problem += extra_transfers >= new_player_count - float(free_transfers)
            problem += extra_transfers >= 0
            objective -= hit_penalty * extra_transfers

    problem += objective

    # Total squad size
    problem += pulp.lpSum(decision_vars.values()) == 15

    # Positional constraints
    for position, required in POSITION_REQUIREMENTS.items():
        problem += (
            pulp.lpSum(decision_vars[proj.element_id] for proj in projections if proj.position == position)
            == required
        )

    # Budget constraint
    problem += pulp.lpSum(proj.price * decision_vars[proj.element_id] for proj in projections) <= budget

    # Team constraint
    team_groups: Dict[int, List[PlayerProjection]] = defaultdict(list)
    for proj in projections:
        team_groups[proj.team_id].append(proj)
    for team_id, players in team_groups.items():
        problem += pulp.lpSum(decision_vars[player.element_id] for player in players) <= 3

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError("Unable to compute an optimal squad with the provided constraints")

    selected = [proj for proj in projections if decision_vars[proj.element_id].value() >= 0.99]
    return selected


def optimize_starting_eleven(squad: Iterable[PlayerProjection]) -> Tuple[List[PlayerProjection], List[PlayerProjection]]:
    squad = list(squad)
    problem = pulp.LpProblem("fpl_starting_eleven", pulp.LpMaximize)

    decision_vars = {proj.element_id: pulp.LpVariable(f"start_{proj.element_id}", cat="Binary") for proj in squad}
    problem += pulp.lpSum(proj.prediction * decision_vars[proj.element_id] for proj in squad)

    problem += pulp.lpSum(decision_vars.values()) == 11

    # Positional constraints for starting XI
    problem += pulp.lpSum(decision_vars[proj.element_id] for proj in squad if proj.position == "GKP") == 1
    problem += pulp.lpSum(decision_vars[proj.element_id] for proj in squad if proj.position == "DEF") >= 3
    problem += pulp.lpSum(decision_vars[proj.element_id] for proj in squad if proj.position == "MID") >= 2
    problem += pulp.lpSum(decision_vars[proj.element_id] for proj in squad if proj.position == "FWD") >= 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationError("Unable to determine an optimal starting XI")

    starting = [proj for proj in squad if decision_vars[proj.element_id].value() >= 0.99]
    bench = [proj for proj in squad if proj not in starting]
    return starting, bench


def summarize_predictions(players: Iterable[PlayerProjection]) -> float:
    return round(sum(player.prediction for player in players), 2)
