#!/usr/bin/env python3
"""Generate ``data/samples.csv`` for the upcoming Fantasy Premier League gameweek."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from openfpl.fpl_client import FPLClient
from openfpl.refresh import refresh_fpl_data
from openfpl.samples import generate_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/fpl"),
        help="Directory containing raw FPL API responses (defaults to data/raw/fpl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples.csv"),
        help="Location where the generated samples should be written.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="Optional CSV file whose header defines the expected column order (defaults to the current output file if it exists).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh the raw API data before generating samples.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional proxy URL to use when refreshing data.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable environment proxy settings for HTTP requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds when refreshing (default: 30).",
    )
    return parser.parse_args()


def resolve_template_path(output: Path, explicit_template: Optional[Path]) -> Optional[Path]:
    if explicit_template is not None:
        return explicit_template
    if output.exists():
        return output
    return None


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir
    if args.refresh:
        client = FPLClient(proxy_url=args.proxy, timeout=args.timeout, trust_env=not args.no_proxy)
        refresh_fpl_data(raw_dir, client=client)
    template = resolve_template_path(args.output, args.template)
    generate_samples(raw_dir, args.output, sample_template=template)


if __name__ == "__main__":
    main()
