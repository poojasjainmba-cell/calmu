from __future__ import annotations

import re
from typing import Any

from .source_mapping import clean_text


SEASON_LABELS = {
    "SP": "Spring",
    "SU": "Summer",
    "FA": "Fall",
}

SEASON_CODES = {label.upper(): code for code, label in SEASON_LABELS.items()}


def parse_budget_term_header(value: Any) -> dict[str, str] | None:
    """Parse budget headers such as SP2-G, SP2G, SP2-A, or SU2 -A."""

    text = clean_text(value)
    if not text:
        return None
    compact = re.sub(r"[\s_\u00a0]+", "", text.upper())
    compact = compact.replace("–", "-").replace("—", "-")
    match = re.match(r"^(SP|SU|FA)-?([12])(?:-?([AG]))?$", compact)
    if not match:
        return None
    season, number, suffix = match.groups()
    metric = "goal" if suffix == "G" else ("actual" if suffix == "A" else "term")
    return {
        "raw_term": text,
        "term": f"{season}{number}",
        "term_label": f"{SEASON_LABELS[season]} {number}",
        "term_metric": metric,
    }


def normalize_term_token(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    parsed = parse_budget_term_header(text)
    if parsed:
        return parsed["term"]

    compact = re.sub(r"[\s_\u00a0-]+", "", text.upper())
    compact = compact.replace("–", "").replace("—", "")
    direct = re.match(r"^(SP|SU|FA)([12])$", compact)
    if direct:
        return f"{direct.group(1)}{direct.group(2)}"

    words = re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()
    named = re.match(r"^(SPRING|SUMMER|FALL)\s*([12])$", words)
    if named:
        return f"{SEASON_CODES[named.group(1)]}{named.group(2)}"

    return text


def display_term_label(value: Any) -> str:
    token = normalize_term_token(value)
    match = re.match(r"^(SP|SU|FA)([12])$", token)
    if match:
        return f"{SEASON_LABELS[match.group(1)]} {match.group(2)}"
    return clean_text(value)
