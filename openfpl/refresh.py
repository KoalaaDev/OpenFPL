"""Refresh helpers for downloading Fantasy Premier League data."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .data_store import save_json
from .fpl_client import FPLClient


def _determine_event_ids_to_download(bootstrap: dict, next_event_id: Optional[int]) -> Iterable[int]:
    for event in bootstrap.get("events", []):
        event_id = event["id"]
        if event.get("finished") or event.get("data_checked"):
            if next_event_id is None or event_id < next_event_id:
                yield event_id


def refresh_fpl_data(output_dir: Path, *, client: Optional[FPLClient] = None) -> None:
    """Download the latest public data from the Fantasy Premier League API.

    The files are stored under *output_dir* using the following structure::

        bootstrap-static.json
        fixtures.json
        events/<event_id>.json

    Parameters
    ----------
    output_dir:
        Directory in which the files should be stored.
    client:
        Optional :class:`~openfpl.fpl_client.FPLClient` instance. When omitted a
        default client is created.
    """

    output_dir = Path(output_dir)
    client = client or FPLClient()

    bootstrap = client.get_bootstrap_static()
    save_json(bootstrap, output_dir / "bootstrap-static.json")

    fixtures = client.get_fixtures()
    save_json(fixtures, output_dir / "fixtures.json")

    next_event_id = next((event["id"] for event in bootstrap.get("events", []) if event.get("is_next")), None)

    for event_id in _determine_event_ids_to_download(bootstrap, next_event_id):
        event_payload = client.get_event_live(event_id)
        save_json(event_payload, output_dir / "events" / f"{event_id}.json")


__all__ = ["refresh_fpl_data"]
