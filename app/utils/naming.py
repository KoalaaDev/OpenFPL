import re
import unicodedata
from typing import Iterable, List


def normalize_name(name: str) -> str:
    """Normalize player names by removing accents and non-alphanumeric characters."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    alphanumeric = re.sub(r"[^A-Za-z0-9]", "", without_accents)
    return alphanumeric.lower()


def build_name_keys(*parts: Iterable[str]) -> List[str]:
    keys = []
    for raw_part in parts:
        if not raw_part:
            continue
        if isinstance(raw_part, str):
            values = [raw_part]
        else:
            values = [value for value in raw_part if value]
        for value in values:
            normalized = normalize_name(value)
            if normalized and normalized not in keys:
                keys.append(normalized)
    return keys
