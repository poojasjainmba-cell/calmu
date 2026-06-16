from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd

from config import METRIC_DEFINITIONS_PATH, load_json_safe


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


@lru_cache(maxsize=4)
def _load_metric_definitions_cached(mtime_ns: int) -> dict[str, Any]:
    return load_json_safe(METRIC_DEFINITIONS_PATH, default={"metrics": {}, "sections": {}}) or {
        "metrics": {},
        "sections": {},
    }


def load_metric_definitions() -> dict[str, Any]:
    try:
        mtime_ns = METRIC_DEFINITIONS_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        mtime_ns = 0
    return _load_metric_definitions_cached(mtime_ns)


def _metric_lookup() -> dict[str, dict[str, Any]]:
    config = load_metric_definitions()
    lookup: dict[str, dict[str, Any]] = {}
    for term, payload in (config.get("metrics") or {}).items():
        if isinstance(payload, str):
            payload = {"definition": payload}
        aliases = [term, *(payload.get("aliases") or [])]
        for alias in aliases:
            lookup[_normalize(alias)] = {"term": term, **payload}
    return lookup


def metric_definition(label: Any) -> str:
    payload = _metric_lookup().get(_normalize(label), {})
    return str(payload.get("definition") or "")


def section_help_text(section_name: str) -> str:
    payload = (load_metric_definitions().get("sections") or {}).get(section_name) or {}
    if not payload:
        return ""
    parts = [
        f"What: {payload.get('what')}",
        f"Why: {payload.get('why')}",
        f"Action: {payload.get('action')}",
    ]
    return "  ".join(part for part in parts if not part.endswith("None"))


def definitions_frame() -> pd.DataFrame:
    rows = []
    metrics = load_metric_definitions().get("metrics") or {}
    for term, payload in metrics.items():
        if isinstance(payload, str):
            definition = payload
            category = "Metric"
        else:
            definition = payload.get("definition", "")
            category = payload.get("category", "Metric")
        rows.append({"Term": term, "Category": category, "Definition": definition})
    return pd.DataFrame(rows, columns=["Term", "Category", "Definition"]).sort_values(["Category", "Term"])
