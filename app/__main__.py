"""``python -m app`` — serve FPLabs.

Environment:
    FPLABS_HOST   bind address (default 0.0.0.0 — put a TLS-terminating
                  reverse proxy such as Caddy or nginx in front of it)
    FPLABS_PORT   port (default 9999)
    FPLABS_WORKERS  uvicorn workers (default 1 — the job manager, rate
                  limiter and scheduler are in-process, so keep it at 1)
    FPLABS_TRUSTED_PROXY=1  trust X-Forwarded-* from the proxy in front

Auto-reload is deliberately not offered: it watches the whole tree and
respawns the process, which on a public server is both a resource sink and
a way to run twice everything the scheduler does once.
"""
import os

import uvicorn


def load_dotenv(path: str = ".env") -> list[str]:
    """Read KEY=VALUE lines from a .env file into the environment.

    Existing variables win, so a value exported in the shell or a service
    unit overrides the file. Quotes around values are stripped; blank lines
    and # comments are ignored. The file is gitignored: it is where the
    Google client secret and the cookie-signing key live on the server.
    """
    loaded: list[str] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    loaded.append(k)
    except OSError:
        pass
    return loaded

if __name__ == "__main__":
    keys = load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
    if keys:
        print(f"[fplabs] loaded {len(keys)} settings from .env: {', '.join(sorted(keys))}")
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("FPLABS_HOST", "0.0.0.0"),
        port=int(os.environ.get("FPLABS_PORT", 9999)),
        workers=1,
        log_level=os.environ.get("FPLABS_LOG_LEVEL", "info"),
        proxy_headers=os.environ.get("FPLABS_TRUSTED_PROXY") == "1",
        forwarded_allow_ips="*" if os.environ.get("FPLABS_TRUSTED_PROXY") == "1" else None,
        server_header=False,
        date_header=True,
    )
