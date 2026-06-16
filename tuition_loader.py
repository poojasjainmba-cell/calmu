from __future__ import annotations

from typing import Any

import pandas as pd

from config import DEGREE_TUITION_PATH, load_json_safe, save_json


TUITION_SOURCE_URL = "https://www.calmu.edu/tuition"
TUITION_SOURCE_NOTE = "Tuition estimates are based on CalMU published tuition rates and are subject to change."

DEFAULT_DEGREE_TUITION: dict[str, dict[str, Any]] = {
    "Certificate": {
        "tuition_per_credit": 585,
        "credits_required": 18,
        "estimated_total_tuition": 10530,
        "estimated_annual_tuition": 10530,
        "duration_years": 0.5,
        "duration_months": 6,
    },
    "Associate": {
        "tuition_per_credit": 585,
        "credits_required": 60,
        "estimated_total_tuition": 35100,
        "estimated_annual_tuition": 14040,
        "duration_years": 2,
        "duration_months": 24,
    },
    "Bachelor": {
        "tuition_per_credit": 585,
        "credits_required": 120,
        "estimated_total_tuition": 70200,
        "estimated_annual_tuition": 14040,
        "duration_years": 4,
        "duration_months": 48,
    },
    "Master": {
        "tuition_per_credit": 841,
        "credits_required": 39,
        "estimated_total_tuition": 32799,
        "estimated_annual_tuition": 15138,
        "duration_years": 2,
        "duration_months": 24,
    },
    "Doctoral": {
        "tuition_per_credit": 884,
        "credits_required": 61,
        "estimated_total_tuition": 53924,
        "estimated_annual_tuition": 10608,
        "duration_years": 4,
        "duration_months": 48,
        "note": "confirm actual expected duration with CalMU policy",
    },
}


def classify_degree_level(program: Any) -> str | None:
    if program is None:
        return None
    try:
        if pd.isna(program):
            return None
    except (TypeError, ValueError):
        pass
    text = str(program).strip().lower()
    if not text or text in {"nan", "nat", "<na>", "none"}:
        return None
    if "certificate" in text:
        return "Certificate"
    if "associate" in text:
        return "Associate"
    if "bachelor" in text or "bsba" in text or "bsbt" in text:
        return "Bachelor"
    if (
        "master" in text
        or "mba" in text
        or "mscis" in text
        or "msai" in text
        or "ms " in text
        or text.startswith("ms")
    ):
        return "Master"
    if "doctoral" in text or "doctorate" in text or "doctor" in text or "dba" in text:
        return "Doctoral"
    return None


def normalize_degree_level(value: Any) -> str | None:
    return classify_degree_level(value)


def load_degree_tuition_config() -> dict[str, dict[str, Any]]:
    loaded = load_json_safe(DEGREE_TUITION_PATH, default={}) or {}
    merged = {key: value.copy() for key, value in DEFAULT_DEGREE_TUITION.items()}
    for degree_level, values in loaded.items():
        if str(degree_level).startswith("_"):
            continue
        if isinstance(values, dict):
            merged.setdefault(degree_level, {}).update(values)
    if not DEGREE_TUITION_PATH.exists():
        save_json(DEGREE_TUITION_PATH, merged)
    return merged


def save_degree_tuition_config(config: dict[str, dict[str, Any]]) -> None:
    save_json(DEGREE_TUITION_PATH, config)


def get_degree_tuition(
    program: Any,
    config: dict[str, dict[str, Any]] | None = None,
    explicit_degree_level: Any = None,
) -> dict[str, Any]:
    config = config or load_degree_tuition_config()
    explicit_level = normalize_degree_level(explicit_degree_level)
    inferred_level = classify_degree_level(program)
    degree_level = explicit_level or inferred_level
    confidence = "High" if explicit_level else "Medium" if inferred_level else "Low"
    if not degree_level:
        return {
            "degree_level": None,
            "program_degree_level": None,
            "tuition_per_credit": pd.NA,
            "credits_required": pd.NA,
            "estimated_total_tuition": pd.NA,
            "estimated_annual_tuition": pd.NA,
            "program_duration_years": pd.NA,
            "program_duration_months": pd.NA,
            "duration_note": "Program missing or not matched to a configured degree level.",
            "tuition_estimate_source": "unmatched_program",
            "revenue_confidence": "Low",
        }

    values = config.get(degree_level, {})
    return {
        "degree_level": degree_level,
        "program_degree_level": degree_level,
        "tuition_per_credit": values.get("tuition_per_credit", pd.NA),
        "credits_required": values.get("credits_required", pd.NA),
        "estimated_total_tuition": values.get("estimated_total_tuition", pd.NA),
        "estimated_annual_tuition": values.get("estimated_annual_tuition", pd.NA),
        "program_duration_years": values.get("duration_years", pd.NA),
        "program_duration_months": values.get("duration_months", pd.NA),
        "duration_note": values.get("note", ""),
        "tuition_estimate_source": "calmu_published_tuition_config",
        "revenue_confidence": confidence,
    }


def tuition_config_frame(config: dict[str, dict[str, Any]] | None = None) -> pd.DataFrame:
    config = config or load_degree_tuition_config()
    rows = []
    for degree_level, values in config.items():
        if str(degree_level).startswith("_"):
            continue
        row = {"program_degree_level": degree_level}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)
