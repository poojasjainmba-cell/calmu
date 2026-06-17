import pandas as pd

from modules.budget_loader import _allocation_rows
from modules.enrollment_ops import canonicalize_udr_columns, canonicalize_udr_goals, goals_by_udr
from modules.term_utils import parse_budget_term_header, normalize_term_token


def test_budget_term_header_parses_actual_vs_goal():
    assert parse_budget_term_header("SP2-A")["term_metric"] == "actual"
    assert parse_budget_term_header("SP2G")["term_metric"] == "goal"
    assert parse_budget_term_header("SU2- G")["term"] == "SU2"
    assert normalize_term_token("Summer 2") == "SU2"


def test_budget_loader_uses_goal_columns_and_keeps_blank_total_term_goals():
    raw = pd.DataFrame(
        [
            ["", "", "", "", ""],
            ["", "", "Budget", "SP2-A", "SP2-G"],
            ["", "Jordan King", 107, 14, 15],
            ["", "Crystal Cassidy", None, 5, 6],
            ["", "TOTAL", 107, 19, 21],
            ["", "Vendor Section", "", "", ""],
        ]
    )

    allocations, terms = _allocation_rows(raw, "Sheet1")

    assert goals_by_udr(allocations) == {"Jordan King": 107.0}
    assert goals_by_udr(terms, ["Spring 2"]) == {"Crystal Cassidy": 6.0, "Jordan King": 15.0}
    assert set(terms["term_metric"]) == {"goal"}
    assert "TOTAL" not in set(allocations["budget_name"])


def test_udr_names_align_short_tracker_names_to_budget_full_names():
    leads = pd.DataFrame({"assigned_udr": ["Tasha Berger"]})
    enrollments = pd.DataFrame({"enrollment_udr": ["Tasha"]})
    goals = {"Tasha Berger": 20}

    aligned_leads, aligned_enrollments = canonicalize_udr_columns(leads, enrollments, list(goals))
    aligned_goals = canonicalize_udr_goals(goals, aligned_leads["assigned_udr"].tolist())

    assert aligned_enrollments.loc[0, "enrollment_udr"] == "Tasha Berger"
    assert aligned_goals == {"Tasha Berger": 20.0}
