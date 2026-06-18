from __future__ import annotations

import pandas as pd

from app import roundup_term_stats, udr_budget_summary
from modules.budget_loader import _allocation_rows
from modules.term_utils import normalize_term_token, parse_budget_term_header


def test_budget_term_headers_parse_actuals_and_goals() -> None:
    assert parse_budget_term_header("SP2-A") == {
        "raw_term": "SP2-A",
        "term": "SP2",
        "term_label": "Spring 2",
        "term_metric": "actual",
    }
    assert parse_budget_term_header("SP2G")["term_metric"] == "goal"
    assert parse_budget_term_header("SU2 -A")["term_label"] == "Summer 2"
    assert normalize_term_token("Fall 1") == "FA1"


def test_udr_budget_summary_uses_term_goal_not_actual() -> None:
    raw = pd.DataFrame(
        [
            ["", "UDR", "Budget", "SP2-A", "SP2-G", "SU2-A", "SU2-G"],
            ["", "UDR 01", 42, 7, 10, 11, 14],
            ["", "Total", 42, 7, 10, 11, 14],
            ["", "Starts", "", 6, "", 8, ""],
            ["", "Start%", "", 0.86, "", 0.73, ""],
        ]
    )

    allocations, term_allocations = _allocation_rows(raw, "Budget")

    spring = udr_budget_summary(term_allocations, allocations, ["Spring 2"])
    udr_01 = spring.set_index("budget_udr").loc["UDR 01"]
    assert udr_01["budget_goal"] == 10
    assert udr_01["budget_sheet_actual"] == 7
    assert round(udr_01["budget_pct_goal"], 2) == 0.7

    annual = udr_budget_summary(term_allocations, allocations)
    assert annual.set_index("budget_udr").loc["UDR 01", "budget_goal"] == 42

    stats = roundup_term_stats(term_allocations, ["Summer 2"])
    assert stats["goal"] == 14
    assert stats["actual"] == 11
    assert stats["starts"] == 8
    assert round(stats["start_rate"], 2) == 0.73
