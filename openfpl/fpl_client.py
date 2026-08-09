"""Client for interacting with the public Fantasy Premier League API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


DEFAULT_BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_USER_AGENT = "OpenFPL/0.1 (+https://github.com/)"


@dataclass
class FPLClient:
    """Simple wrapper around the Fantasy Premier League HTTP API.

    Parameters
    ----------
    base_url:
        Base URL for the API. This is exposed for testability.
    user_agent:
        User agent header sent with every request.
    proxy_url:
        Optional explicit proxy URL. When omitted the values from the
        ``HTTP_PROXY``/``HTTPS_PROXY`` environment variables (if any) are used.
    timeout:
        Request timeout in seconds.
    """

    base_url: str = DEFAULT_BASE_URL
    user_agent: str = DEFAULT_USER_AGENT
    proxy_url: Optional[str] = None
    timeout: int = 30
    trust_env: bool = True

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        self._session = requests.Session()
        self._session.trust_env = self.trust_env
        self._session.headers.update({"User-Agent": self.user_agent})
        self._proxies: Optional[Dict[str, str]]
        if self.proxy_url:
            self._proxies = {"http": self.proxy_url, "https": self.proxy_url}
        else:
            self._proxies = None

    def _get(self, path: str) -> Dict[str, Any]:
        response = self._session.get(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            timeout=self.timeout,
            proxies=self._proxies,
        )
        response.raise_for_status()
        return response.json()

    def get_bootstrap_static(self) -> Dict[str, Any]:
        """Return the payload from ``/bootstrap-static/``."""

        return self._get("bootstrap-static/")

    def get_fixtures(self) -> Dict[str, Any]:
        """Return the payload from ``/fixtures/``."""

        return self._get("fixtures/")

    def get_event_live(self, event_id: int) -> Dict[str, Any]:
        """Return the live data for a given gameweek."""

        return self._get(f"event/{event_id}/live/")


__all__ = ["FPLClient"]
