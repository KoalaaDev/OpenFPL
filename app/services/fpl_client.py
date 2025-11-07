from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx

BOOTSTRAP_STATIC = "https://fantasy.premierleague.com/api/bootstrap-static/"
ENTRY_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/"
PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{event_id}/picks/"


class FPLClient:
    def __init__(self, timeout: Optional[float] = 10.0) -> None:
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_bootstrap(self) -> Dict[str, Any]:
        response = await self._client.get(BOOTSTRAP_STATIC)
        response.raise_for_status()
        return response.json()

    async def fetch_entry(self, team_id: int) -> Dict[str, Any]:
        response = await self._client.get(ENTRY_URL.format(team_id=team_id))
        response.raise_for_status()
        return response.json()

    async def fetch_picks(self, team_id: int, event_id: int) -> Dict[str, Any]:
        response = await self._client.get(PICKS_URL.format(team_id=team_id, event_id=event_id))
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()


@lru_cache(maxsize=1)
def get_fpl_client() -> FPLClient:
    return FPLClient()


async def gather_with_client(team_id: int, event_id: int) -> Dict[str, Any]:
    client = get_fpl_client()
    bootstrap_task = asyncio.create_task(client.fetch_bootstrap())
    entry_task = asyncio.create_task(client.fetch_entry(team_id))
    picks_task = asyncio.create_task(client.fetch_picks(team_id, event_id))

    bootstrap, entry, picks = await asyncio.gather(bootstrap_task, entry_task, picks_task)
    return {"bootstrap": bootstrap, "entry": entry, "picks": picks or None}
