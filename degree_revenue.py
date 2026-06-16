from __future__ import annotations

from typing import Any

from revenue_estimator import add_program_duration_fields, add_program_revenue_fields
from tuition_loader import (
    DEFAULT_DEGREE_TUITION,
    classify_degree_level,
    get_degree_tuition,
    load_degree_tuition_config,
)


def get_program_duration(program: Any, config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    return get_degree_tuition(program, config)
