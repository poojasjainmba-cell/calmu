from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import requests

from tuition_loader import TUITION_SOURCE_URL, load_degree_tuition_config, save_degree_tuition_config


PROGRAM_PATTERNS = {
    "Certificate": r"Certificate Programs\s+\$(?P<tuition>[\d,]+)\s+(?P<credits>\d+)\s+\$(?P<total>[\d,]+)\s+\$(?P<annual>[\d,]+)",
    "Associate": r"Associate Degree\s+\$(?P<tuition>[\d,]+)\s+(?P<credits>\d+)\s+\$(?P<total>[\d,]+)\s+\$(?P<annual>[\d,]+)",
    "Bachelor": r"Bachelor'?s Degree\s+\$(?P<tuition>[\d,]+)\s+(?P<credits>\d+)\s+\$(?P<total>[\d,]+)\s+\$(?P<annual>[\d,]+)",
    "Master": r"Master'?s Degree Programs\s+\$(?P<tuition>[\d,]+)\s+(?P<credits>\d+)\s+\$(?P<total>[\d,]+)\s+\$(?P<annual>[\d,]+)",
    "Doctoral": r"Doctoral Degree Programs\s+\$(?P<tuition>[\d,]+)\s+(?P<credits>\d+)\s+\$(?P<total>[\d,]+)\s+\$(?P<annual>[\d,]+)",
}


def _number(value: str) -> int:
    return int(value.replace(",", ""))


def parse_tuition_page(text: str) -> dict[str, dict[str, int]]:
    normalized = re.sub(r"\s+", " ", text)
    parsed: dict[str, dict[str, int]] = {}
    for degree_level, pattern in PROGRAM_PATTERNS.items():
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        parsed[degree_level] = {
            "tuition_per_credit": _number(match.group("tuition")),
            "credits_required": _number(match.group("credits")),
            "estimated_total_tuition": _number(match.group("total")),
            "estimated_annual_tuition": _number(match.group("annual")),
        }
    return parsed


def refresh_tuition_config(url: str = TUITION_SOURCE_URL, timeout: int = 30) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    parsed = parse_tuition_page(response.text)
    if not parsed:
        raise RuntimeError("Could not parse tuition values from CalMU tuition page.")

    config = load_degree_tuition_config()
    for degree_level, values in parsed.items():
        config.setdefault(degree_level, {}).update(values)
    config["_metadata"] = {
        "source_url": url,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "note": "Tuition and fees are subject to change. Review CalMU policy before relying on forecasts.",
    }
    save_degree_tuition_config(config)
    return config


def main() -> int:
    try:
        config = refresh_tuition_config()
    except Exception as exc:
        print(f"Tuition refresh failed: {exc}")
        return 1
    print("Updated config/degree_tuition.json from CalMU tuition page.")
    for degree_level, values in config.items():
        if degree_level.startswith("_"):
            continue
        print(
            f"{degree_level}: ${values.get('tuition_per_credit')}/credit, "
            f"{values.get('credits_required')} credits, "
            f"${values.get('estimated_total_tuition')} total"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
