"""Privacy policy and terms of service, served at /privacy and /terms.

Both are static HTML templates under app/legal/ with three substitutions,
so the published pages carry the real origin, contact and effective date:

    {{BASE_URL}}      $FPLABS_BASE_URL (falls back to the request origin)
    {{CONTACT}}       $FPLABS_CONTACT_EMAIL
    {{JURISDICTION}}  $FPLABS_JURISDICTION (default "England and Wales")
    {{DATE}}          $FPLABS_LEGAL_DATE   (default: the file's last change)

Google's OAuth consent screen requires public URLs for both documents; these
are the ones to enter (see docs/DEPLOY.md).
"""
from __future__ import annotations

import datetime as _dt
import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

LEGAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legal")
router = APIRouter()


def _date(path: str) -> str:
    override = os.environ.get("FPLABS_LEGAL_DATE")
    if override:
        return override
    ts = os.path.getmtime(path)
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%d %B %Y")


def render(name: str, request: Request | None = None) -> str:
    path = os.path.join(LEGAL_DIR, f"{name}.html")
    with open(path, encoding="utf-8") as f:
        html = f.read()
    base = os.environ.get("FPLABS_BASE_URL", "").rstrip("/")
    if not base and request is not None:
        base = str(request.base_url).rstrip("/")
    subs = {
        "{{BASE_URL}}": base or "this site",
        "{{CONTACT}}": os.environ.get("FPLABS_CONTACT_EMAIL", "hello@koalaa.dev"),
        "{{JURISDICTION}}": os.environ.get("FPLABS_JURISDICTION", "England and Wales"),
        "{{DATE}}": _date(path),
    }
    for k, v in subs.items():
        html = html.replace(k, v)
    return html


@router.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
def privacy(request: Request):
    return HTMLResponse(render("privacy", request),
                        headers={"Cache-Control": "no-cache"})


@router.get("/terms", response_class=HTMLResponse, include_in_schema=False)
def terms(request: Request):
    return HTMLResponse(render("terms", request),
                        headers={"Cache-Control": "no-cache"})
