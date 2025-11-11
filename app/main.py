from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Dict, Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator

from app.services.fpl_client import gather_with_client, get_fpl_client
from app.services.optimizer import (
    OptimizationError,
    PlayerProjection,
    POSITION_MAP,
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
    free_transfers: Optional[int] = None

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

    @validator("free_transfers")
    def validate_free_transfers(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        if value < 0:
            raise ValueError("Free transfers cannot be negative")
        if value > 15:
            raise ValueError("Free transfers cannot exceed squad size")
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
    net_predicted_points: float
    bench_predicted_points: float
    chip_recommendations: Dict[str, str]
    current_team_projection: Optional[Dict[str, Any]]
    transfer_summary: Dict[str, Any]
    finance_snapshot: Dict[str, str]
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
    prediction_gameweek = prediction_df.attrs.get("gameweek", request.gameweek)

    try:
        data = await gather_with_client(request.team_id, prediction_gameweek)
    except Exception as exc:  # pragma: no cover - upstream error handling
        raise HTTPException(status_code=502, detail=f"Failed to reach FPL API: {exc}") from exc

    bootstrap = data["bootstrap"]
    entry = data["entry"]
    picks_payload = data.get("picks") or {}
    picks_event = data.get("picks_event") or prediction_gameweek

    resolved_gameweek = min(prediction_gameweek, picks_event)
    if resolved_gameweek != prediction_gameweek:
        prediction_df = load_predictions(DATA_DIR, request.season, resolved_gameweek)
        prediction_gameweek = prediction_df.attrs.get("gameweek", resolved_gameweek)
        resolved_gameweek = prediction_gameweek

    entry_history = picks_payload.get("entry_history") if isinstance(picks_payload, dict) else None
    entry_history_event = entry_history.get("event") if entry_history else None
    current_event = entry.get("current_event")

    raw_bank: Optional[int] = None
    raw_value: Optional[int] = None
    finance_snapshot = {"bank": "default", "value": "default"}

    # Prefer the latest finance snapshot from the entry payload when we are
    # looking at a past gameweek compared to the manager's current event.
    if current_event and (entry_history_event is None or entry_history_event < current_event):
        latest_bank = entry.get("last_deadline_bank")
        latest_value = entry.get("last_deadline_value")
        if latest_bank is not None:
            raw_bank = latest_bank
            finance_snapshot["bank"] = "entry_last_deadline"
        if latest_value is not None:
            raw_value = latest_value
            finance_snapshot["value"] = "entry_last_deadline"

    if raw_bank is None and entry_history:
        event_bank = entry_history.get("bank")
        if event_bank is not None:
            raw_bank = event_bank
            finance_snapshot["bank"] = "entry_history"
    if raw_value is None and entry_history:
        event_value = entry_history.get("value")
        if event_value is not None:
            raw_value = event_value
            finance_snapshot["value"] = "entry_history"

    if raw_bank is None:
        entry_bank = entry.get("bank")
        if entry_bank is not None:
            raw_bank = entry_bank
            finance_snapshot["bank"] = "entry_fallback"
    if raw_value is None:
        entry_value = entry.get("value")
        if entry_value is not None:
            raw_value = entry_value
            finance_snapshot["value"] = "entry_fallback"

    bank = max((raw_bank or 0) / 10.0, 0.0)
    total_budget = max((raw_value or 0) / 10.0, 0.0)

    if total_budget <= 0:
        if bank > 0:
            total_budget = bank
        else:
            total_budget = 100.0
            bank = 0.0

    if bank > total_budget:
        bank = total_budget

    squad_value = max(total_budget - bank, 0.0)

    budget = total_budget
    team_name = entry.get("name", f"Team {request.team_id}")

    name_index, team_lookup = build_player_catalog(bootstrap)
    projection_pool = build_projection_pool(
        prediction_df,
        name_index,
        team_lookup,
        bootstrap.get("elements", []),
    )
    projection_lookup = {proj.element_id: proj for proj in projection_pool}

    current_picks = picks_payload.get("picks") if isinstance(picks_payload, dict) else None
    current_squad_ids = {pick["element"] for pick in current_picks} if current_picks else set()

    free_transfers = request.free_transfers
    if free_transfers is None:
        free_transfers = estimate_free_transfers(entry, picks_payload)
    free_transfers = int(free_transfers)
    free_transfer_source = "user" if request.free_transfers is not None else "estimated"

    element_lookup = {element["id"]: element for element in bootstrap.get("elements", [])}

    try:
        optimized_squad = optimize_squad(
            projection_pool,
            budget,
            current_squad_ids=current_squad_ids if current_squad_ids else None,
            free_transfers=free_transfers,
        )
        starting, bench = optimize_starting_eleven(optimized_squad)
    except OptimizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    starting_sorted = sorted(starting, key=lambda player: player.prediction, reverse=True)
    bench_sorted = sorted(bench, key=lambda player: player.prediction, reverse=True)

    captain = starting_sorted[0] if starting_sorted else None
    vice_captain = starting_sorted[1] if len(starting_sorted) > 1 else None

    total_points = summarize_predictions(starting_sorted)
    bench_points = summarize_predictions(bench_sorted)

    transfer_summary = summarize_transfers(
        optimized_squad,
        current_picks,
        free_transfers,
        projection_lookup,
        element_lookup,
        team_lookup,
        free_transfer_source,
    )
    hit_points = transfer_summary.get("hit_points", 0.0)
    net_points = round(total_points - hit_points, 2)
    transfer_summary["net_points"] = net_points

    chip_recommendations = generate_chip_recommendations(
        captain=captain,
        bench=bench_sorted,
        starting=starting_sorted,
        current_team=picks_payload,
        projection_lookup=projection_lookup,
        optimized_points=net_points,
    )

    saved_payload = {
        "team_name": team_name,
        "team_id": request.team_id,
        "season": request.season,
        "gameweek": resolved_gameweek,
        "optimized_starting": [player.to_dict() for player in starting_sorted],
        "optimized_bench": [player.to_dict() for player in bench_sorted],
        "captain": captain.to_dict() if captain else None,
        "vice_captain": vice_captain.to_dict() if vice_captain else None,
        "total_predicted_points": total_points,
        "net_predicted_points": net_points,
        "bench_predicted_points": bench_points,
        "chip_recommendations": chip_recommendations,
        "transfer_summary": transfer_summary,
        "budget": budget,
        "bank": bank,
        "value": squad_value,
        "finance_snapshot": finance_snapshot,
        "saved_at": datetime.utcnow().isoformat() + "Z",
    }
    save_team(request.team_id, saved_payload)

    current_projection = None
    if picks_payload:
        current_projection = build_current_team_projection(
            picks_payload,
            projection_lookup,
        )

    return OptimizeResponse(
        team_name=team_name,
        season=request.season,
        gameweek=resolved_gameweek,
        budget=round(budget, 1),
        bank=round(bank, 1),
        value=round(squad_value, 1),
        optimized_starting=[player.to_dict() for player in starting_sorted],
        optimized_bench=[player.to_dict() for player in bench_sorted],
        captain=captain.to_dict() if captain else None,
        vice_captain=vice_captain.to_dict() if vice_captain else None,
        total_predicted_points=total_points,
        net_predicted_points=net_points,
        bench_predicted_points=bench_points,
        chip_recommendations=chip_recommendations,
        current_team_projection=current_projection,
        transfer_summary=transfer_summary,
        finance_snapshot=finance_snapshot,
        saved_at=saved_payload["saved_at"],
    )


@app.get("/api/saved/{team_id}")
async def get_saved(team_id: int) -> Dict[str, Any]:
    saved_file = SAVE_DIR / f"{team_id}.json"
    if not saved_file.exists():
        raise HTTPException(status_code=404, detail="No saved optimization found for this team")
    with open(saved_file, "r", encoding="utf-8") as file:
        return json.load(file)


def list_saved_teams() -> List[Dict[str, Any]]:
    if not SAVE_DIR.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for saved_file in SAVE_DIR.glob("*.json"):
        try:
            with open(saved_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        team_id_raw = payload.get("team_id") or saved_file.stem
        try:
            team_id = int(team_id_raw)
        except (TypeError, ValueError):
            team_id = str(team_id_raw)

        entries.append(
            {
                "team_id": team_id,
                "team_name": payload.get("team_name") or f"Team {team_id}",
                "season": payload.get("season"),
                "gameweek": payload.get("gameweek"),
                "net_predicted_points": payload.get("net_predicted_points"),
                "saved_at": payload.get("saved_at"),
            }
        )

    entries.sort(key=lambda item: item.get("saved_at") or "", reverse=True)
    return entries


@app.get("/api/saved")
async def get_saved_index() -> List[Dict[str, Any]]:
    return list_saved_teams()


def save_team(team_id: int, payload: Dict[str, Any]) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    saved_file = SAVE_DIR / f"{team_id}.json"
    with open(saved_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def estimate_free_transfers(entry: Dict[str, Any], picks_payload: Optional[Dict[str, Any]]) -> int:
    history = picks_payload.get("entry_history") if isinstance(picks_payload, dict) else None
    if history:
        free_transfers = history.get("free_transfers")
        if isinstance(free_transfers, int) and 0 <= free_transfers <= 15:
            return free_transfers
        transfers_made = history.get("event_transfers")
        hits_cost = history.get("event_transfers_cost") or 0
        if isinstance(transfers_made, int):
            if transfers_made == 0:
                return 1
            if hits_cost:
                return 0
    return 1


def summarize_transfers(
    optimized_squad: Iterable[PlayerProjection],
    current_picks: Optional[Iterable[Dict[str, Any]]],
    free_transfers: int,
    projection_lookup: Dict[int, PlayerProjection],
    element_lookup: Dict[int, Dict[str, Any]],
    team_lookup: Dict[int, str],
    free_transfer_source: str,
) -> Dict[str, Any]:
    free_transfers = max(0, min(int(free_transfers), 15))

    if not current_picks:
        return {
            "free_transfers": free_transfers,
            "free_transfers_source": free_transfer_source,
            "transfers_needed": 0,
            "transfer_hits": 0,
            "hit_points": 0.0,
            "transfers_in": [],
            "transfers_out": [],
        }

    current_ids = {pick.get("element") for pick in current_picks if pick.get("element") is not None}
    optimized_ids = {player.element_id for player in optimized_squad}

    transfers_in = [player for player in optimized_squad if player.element_id not in current_ids]
    transfers_out_ids = [player_id for player_id in current_ids if player_id not in optimized_ids]

    transfers_in_payload = [player.to_dict() for player in transfers_in]

    transfers_out_payload: List[Dict[str, Any]] = []
    for player_id in transfers_out_ids:
        projection = projection_lookup.get(player_id)
        if projection:
            transfers_out_payload.append(projection.to_dict())
            continue
        element = element_lookup.get(player_id)
        if not element:
            continue
        transfers_out_payload.append(
            {
                "element_id": player_id,
                "name": element.get("web_name", ""),
                "position": POSITION_MAP.get(element.get("element_type"), ""),
                "team_id": element.get("team"),
                "team_name": team_lookup.get(element.get("team"), ""),
                "price": round((element.get("now_cost") or 0) / 10.0, 1),
                "prediction": 0.0,
            }
        )

    transfers_needed = len(transfers_in_payload)
    transfer_hits = max(transfers_needed - free_transfers, 0)
    hit_points = float(transfer_hits * 4)

    return {
        "free_transfers": free_transfers,
        "free_transfers_source": free_transfer_source,
        "transfers_needed": transfers_needed,
        "transfer_hits": transfer_hits,
        "hit_points": hit_points,
        "transfers_in": transfers_in_payload,
        "transfers_out": transfers_out_payload,
    }


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
