"""Utilities for persisting raw data fetched from the FPL API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(payload: Dict[str, Any], path: Path) -> None:
    """Serialize *payload* as JSON to *path*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON data from *path*."""

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


__all__ = ["save_json", "load_json"]
