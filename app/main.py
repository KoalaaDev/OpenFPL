from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator

from app.services.fpl_client import gather_with_client, get_fpl_client
from app.services.optimizer import (
    OptimizationError,
    PlayerProjection,
    build_player_catalog,
    build_projection_pool,
    load_predictions,
    optimize_squad,
    optimize_starting_eleven,
    summarize_predictions,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAVE_DIR = Path(__file__).resolve().parent / "saved_teams"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = FastAPI(title="OpenFPL Optimizer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")


class OptimizeRequest(BaseModel):
    team_id: int
    season: str
    gameweek: int

    @validator("season")
    def validate_season(cls, value: str) -> str:
        if "-" not in value:
            raise ValueError("Season must be in YYYY-YY format, e.g., 2025-26")
        return value

    @validator("gameweek")
    def validate_gameweek(cls, value: int) -> int:
        if value < 1 or value > 38:
            raise ValueError("Gameweek must be between 1 and 38")
        return value


class OptimizeResponse(BaseModel):
    team_name: str
    season: str
    gameweek: int
    budget: float
    bank: float
    value: float
    optimized_starting: List[Dict[str, Any]]
    optimized_bench: List[Dict[str, Any]]
    captain: Optional[Dict[str, Any]]
    vice_captain: Optional[Dict[str, Any]]
    total_predicted_points: float
    bench_predicted_points: float
    chip_recommendations: Dict[str, str]
    current_team_projection: Optional[Dict[str, Any]]
    saved_at: str


async def render_template(name: str) -> str:
    template_path = TEMPLATES_DIR / name
    with open(template_path, "r", encoding="utf-8") as file:
        return file.read()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    html = await render_template("index.html")
    return HTMLResponse(html)


@app.post("/api/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    prediction_df = load_predictions(DATA_DIR, request.season, request.gameweek)

    event_id = request.gameweek
    try:
        data = await gather_with_client(request.team_id, event_id)
    except Exception as exc:  # pragma: no cover - upstream error handling
        raise HTTPException(status_code=502, detail=f"Failed to reach FPL API: {exc}") from exc

    bootstrap = data["bootstrap"]
    entry = data["entry"]
    team_name = entry.get("name", f"Team {request.team_id}")
    bank = entry.get("bank", 0) / 10.0
    value = entry.get("value", 0) / 10.0
    budget = bank + value

    name_index, team_lookup = build_player_catalog(bootstrap)
    projection_pool = build_projection_pool(prediction_df, name_index, team_lookup)
    projection_lookup = {proj.element_id: proj for proj in projection_pool}

    try:
        optimized_squad = optimize_squad(projection_pool, budget)
        starting, bench = optimize_starting_eleven(optimized_squad)
    except OptimizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    starting_sorted = sorted(starting, key=lambda player: player.prediction, reverse=True)
    bench_sorted = sorted(bench, key=lambda player: player.prediction, reverse=True)

    captain = starting_sorted[0] if starting_sorted else None
    vice_captain = starting_sorted[1] if len(starting_sorted) > 1 else None

    total_points = summarize_predictions(starting_sorted)
    bench_points = summarize_predictions(bench_sorted)

    chip_recommendations = generate_chip_recommendations(
        captain=captain,
        bench=bench_sorted,
        starting=starting_sorted,
        current_team=data.get("picks"),
        projection_lookup=projection_lookup,
        optimized_points=total_points,
    )

    saved_payload = {
        "team_name": team_name,
        "team_id": request.team_id,
        "season": request.season,
        "gameweek": request.gameweek,
        "optimized_starting": [player.to_dict() for player in starting_sorted],
        "optimized_bench": [player.to_dict() for player in bench_sorted],
        "captain": captain.to_dict() if captain else None,
        "vice_captain": vice_captain.to_dict() if vice_captain else None,
        "total_predicted_points": total_points,
        "bench_predicted_points": bench_points,
        "chip_recommendations": chip_recommendations,
        "budget": budget,
        "bank": bank,
        "value": value,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    save_team(request.team_id, saved_payload)

    current_projection = None
    if data.get("picks"):
        current_projection = build_current_team_projection(
            data["picks"],
            projection_lookup,
        )

    return OptimizeResponse(
        team_name=team_name,
        season=request.season,
        gameweek=request.gameweek,
        budget=round(budget, 1),
        bank=round(bank, 1),
        value=round(value, 1),
        optimized_starting=[player.to_dict() for player in starting_sorted],
        optimized_bench=[player.to_dict() for player in bench_sorted],
        captain=captain.to_dict() if captain else None,
        vice_captain=vice_captain.to_dict() if vice_captain else None,
        total_predicted_points=total_points,
        bench_predicted_points=bench_points,
        chip_recommendations=chip_recommendations,
        current_team_projection=current_projection,
        saved_at=saved_payload["saved_at"],
    )


@app.get("/api/saved/{team_id}")
async def get_saved(team_id: int) -> Dict[str, Any]:
    saved_file = SAVE_DIR / f"{team_id}.json"
    if not saved_file.exists():
        raise HTTPException(status_code=404, detail="No saved optimization found for this team")
    with open(saved_file, "r", encoding="utf-8") as file:
        return json.load(file)


def save_team(team_id: int, payload: Dict[str, Any]) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    saved_file = SAVE_DIR / f"{team_id}.json"
    with open(saved_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def generate_chip_recommendations(
    captain: Optional[PlayerProjection],
    bench: List[PlayerProjection],
    starting: List[PlayerProjection],
    current_team: Optional[Dict[str, Any]],
    projection_lookup: Dict[int, PlayerProjection],
    optimized_points: float,
) -> Dict[str, str]:
    recommendations: Dict[str, str] = {}

    if captain and captain.prediction >= 10:
        diff = captain.prediction - (starting[1].prediction if len(starting) > 1 else 0)
        if diff >= 2:
            recommendations["triple_captain"] = (
                f"Consider Triple Captain on {captain.name} ({captain.team_name}) with an expected {captain.prediction:.1f} points."
            )
        else:
            recommendations["triple_captain"] = "Hold the Triple Captain chip for a better differential."
    else:
        recommendations["triple_captain"] = "No standout Triple Captain option this week."

    bench_total = summarize_predictions(bench)
    if bench_total >= 16:
        recommendations["bench_boost"] = (
            f"Bench Boost could yield {bench_total:.1f} points from your bench."
        )
    else:
        recommendations["bench_boost"] = "Bench Boost not recommended with the current bench projections."

    recommendations["free_hit"] = evaluate_free_hit(
        current_team,
        projection_lookup,
        optimized_points,
    )
    return recommendations


def evaluate_free_hit(
    current_team: Optional[Dict[str, Any]],
    projection_lookup: Dict[int, PlayerProjection],
    optimized_points: float,
) -> str:
    if not current_team:
        return "Free Hit data unavailable; unable to evaluate."

    picks = current_team.get("picks", [])
    if not picks:
        return "Free Hit not necessary; no picks available for comparison."

    total = 0.0
    for pick in picks:
        element = pick.get("element")
        position = pick.get("position")
        # Starting XI positions are 1-11
        if position and position <= 11:
            projection = projection_lookup.get(element)
            if projection:
                total += projection.prediction

    if optimized_points - total >= 20:
        return "Strong case for Free Hit with a projected 20+ point gain over current XI."
    if optimized_points - total >= 12:
        return "Free Hit could be worthwhile; projected gain exceeds 12 points."
    return "Free Hit not recommended based on current projections."


def build_current_team_projection(
    picks: Dict[str, Any],
    projection_lookup: Dict[int, PlayerProjection],
) -> Dict[str, Any]:
    elements = picks.get("picks", [])
    if not elements:
        return {}

    starting: List[Dict[str, Any]] = []
    bench: List[Dict[str, Any]] = []
    for pick in elements:
        element_id = pick.get("element")
        projection = projection_lookup.get(element_id)
        if not projection:
            continue
        payload = projection.to_dict()
        payload["multiplier"] = pick.get("multiplier")
        payload["is_captain"] = pick.get("is_captain")
        payload["is_vice_captain"] = pick.get("is_vice_captain")
        if pick.get("position", 0) <= 11:
            starting.append(payload)
        else:
            bench.append(payload)

    return {
        "starting": sorted(starting, key=lambda player: player.get("prediction", 0), reverse=True),
        "bench": sorted(bench, key=lambda player: player.get("prediction", 0), reverse=True),
    }


@app.on_event("shutdown")
async def shutdown_event() -> None:
    client = get_fpl_client()
    if hasattr(client, "close"):
        await client.close()
