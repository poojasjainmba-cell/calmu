from __future__ import annotations

import pandas as pd

from pipeline_health import cleanup_needed, enrollment_path, lead_status_summary, stage_bottlenecks


def _sample_contacts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "lead_status": "New",
                "paid_lead_flag": True,
                "lead_temperature": "Hot",
                "days_since_last_activity": 20,
                "lead_age_days": 35,
                "has_open_deal": True,
                "has_won_deal": False,
                "has_lost_deal": False,
                "potential_revenue": 10000,
                "last_activity_date": "2026-01-02T00:00:00Z",
                "next_activity_date": pd.NA,
                "salesman_name": "Owner One",
            },
            {
                "contact_id": "c2",
                "final_lead_status": "Dead",
                "paid_lead_flag": False,
                "lead_temperature": "Dead",
                "days_since_last_activity": 45,
                "lead_age_days": 90,
                "has_open_deal": False,
                "has_won_deal": False,
                "has_lost_deal": True,
                "potential_revenue": 20000,
                "salesman_name": "Owner Two",
            },
            {
                "contact_id": "c3",
                "enrollment_status": "EA Signed",
                "paid_lead_flag": True,
                "lead_temperature": "Warm",
                "days_since_last_activity": 3,
                "lead_age_days": 12,
                "has_open_deal": False,
                "has_won_deal": True,
                "has_lost_deal": False,
                "potential_revenue": 30000,
                "enrolled_revenue": 30000,
                "last_activity_date": "2026-01-08T00:00:00Z",
                "num_notes": 4,
                "salesman_name": "Owner Three",
            },
            {
                "contact_id": "c4",
                "paid_lead_flag": False,
                "lead_temperature": "Cold",
                "days_since_last_activity": pd.NA,
                "lead_age_days": 50,
                "has_open_deal": True,
                "has_won_deal": False,
                "has_lost_deal": False,
                "potential_revenue": 40000,
                "salesman_name": "Unassigned",
            },
        ]
    )


def _sample_fact() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "deal_id": "d1",
                "is_won": False,
                "is_lost": False,
                "revenue_attributed": 0,
            },
            {
                "contact_id": "c3",
                "deal_id": "d3",
                "is_won": True,
                "is_lost": False,
                "revenue_attributed": 30000,
            },
        ]
    )


def test_lead_status_summary_answers_status_counts_and_revenue() -> None:
    summary = lead_status_summary(_sample_contacts(), _sample_fact()).set_index("status")

    assert summary.loc["New", "total leads"] == 1
    assert summary.loc["New", "paid leads"] == 1
    assert summary.loc["New", "hot leads"] == 1
    assert summary.loc["New", "stale leads"] == 1
    assert summary.loc["Enrolled / Closed Won", "enrolled"] == 1
    assert summary.loc["Enrolled / Closed Won", "actual revenue"] == 30000
    assert summary.loc["Closed Lost / Dead", "dead leads"] == 1


def test_stage_bottlenecks_and_cleanup_find_stuck_records() -> None:
    bottlenecks = stage_bottlenecks(_sample_contacts(), _sample_fact()).set_index("stage/status")
    cleanup = cleanup_needed(_sample_contacts(), _sample_fact()).set_index("cleanup issue")

    assert bottlenecks.loc["New", "leads stuck 14+ days"] == 1
    assert bottlenecks.loc["Cold Lead", "leads stuck 30+ days"] == 1
    assert cleanup.loc["old open leads", "leads"] == 2
    assert cleanup.loc["inactive owner with open leads", "leads"] == 1
    assert cleanup.loc["no next activity", "leads"] == 2


def test_enrollment_path_uses_calculated_milestones() -> None:
    path = enrollment_path(_sample_contacts(), _sample_fact()).set_index("milestone")

    assert path.loc["Lead created", "leads"] == 4
    assert path.loc["Contacted", "leads"] == 2
    assert path.loc["Application / deal created", "leads"] == 4
    assert path.loc["Enrolled / closed won", "leads"] == 1
    assert path.loc["Closed lost / dead", "leads"] == 1
