#!/usr/bin/env python3
"""Download the latest Fantasy Premier League data into ``data/raw/fpl``."""
from __future__ import annotations

import argparse
from pathlib import Path

from openfpl.fpl_client import FPLClient
from openfpl.refresh import refresh_fpl_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/fpl"),
        help="Directory where the raw JSON responses should be stored.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional proxy URL (e.g. http://proxy:8080). If omitted the environment configuration is used.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable environment proxy settings when making HTTP requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = FPLClient(proxy_url=args.proxy, timeout=args.timeout, trust_env=not args.no_proxy)
    refresh_fpl_data(args.output_dir, client=client)


if __name__ == "__main__":
    main()
