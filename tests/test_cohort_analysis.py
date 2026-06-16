from __future__ import annotations

import pandas as pd

from cohort_analysis import cohort_calculated_fields, cohort_summary, prepare_cohort_contacts


def _contacts() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "contact_created_at": "2026-01-01T00:00:00Z",
                "cohort": "Official 2026A",
                "start_term": "2026 Fall I",
                "paid_lead_flag": True,
                "has_won_deal": True,
                "enrollment_status": "Enrolled",
                "enrolled_revenue": 10000,
                "potential_revenue": 10000,
                "time_to_first_engagement": 86400000,
                "num_notes": 4,
                "paid_vendor": "Atra",
                "program": "MBA",
                "salesman_name": "Owner A",
                "source_group": "Paid Search",
                "utm_campaign": "Campaign A",
                "program_degree_level": "Master",
                "lead_temperature": "Hot",
                "next_activity_date": "2026-01-03T00:00:00Z",
                "days_since_last_activity": 1,
                "has_open_deal": False,
            },
            {
                "contact_id": "c2",
                "contact_created_at": "2026-02-01T00:00:00Z",
                "cohort": "",
                "start_term": "2026 Fall II",
                "paid_lead_flag": False,
                "has_won_deal": False,
                "enrollment_status": "",
                "enrolled_revenue": 0,
                "potential_revenue": 20000,
                "num_notes": 0,
                "paid_vendor": "Unknown",
                "program": "Bachelor",
                "salesman_name": "Owner B",
                "lead_temperature": "Cold",
                "has_open_deal": True,
            },
            {
                "contact_id": "c3",
                "contact_created_at": "2026-03-01T00:00:00Z",
                "cohort": "",
                "start_term": "",
                "paid_lead_flag": True,
                "has_won_deal": True,
                "enrollment_status": "",
                "enrolled_revenue": 30000,
                "potential_revenue": 30000,
                "num_notes": 6,
                "paid_vendor": "Google",
                "program": "DBA",
                "salesman_name": "Owner C",
                "lead_temperature": "Warm",
                "has_open_deal": False,
            },
            {
                "contact_id": "c4",
                "contact_created_at": "2026-04-15T00:00:00Z",
                "cohort": "",
                "start_term": "",
                "paid_lead_flag": False,
                "has_won_deal": False,
                "enrollment_status": "",
                "enrolled_revenue": 0,
                "potential_revenue": 40000,
                "num_notes": 0,
                "paid_vendor": "Unknown",
                "program": "Certificate",
                "salesman_name": "Owner D",
                "lead_temperature": "Dead",
                "has_open_deal": False,
            },
        ]
    )


def _fact() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "deal_id": "d1",
                "is_won": True,
                "has_won_deal": True,
                "closed_won_date": "2026-01-11T00:00:00Z",
                "close_date": "2026-01-11T00:00:00Z",
                "revenue_attributed": 9000,
            },
            {
                "contact_id": "c3",
                "deal_id": "d3",
                "is_won": True,
                "has_won_deal": True,
                "closed_won_date": "2026-05-20T00:00:00Z",
                "close_date": "2026-05-20T00:00:00Z",
                "revenue_attributed": 25000,
            },
        ]
    )


def test_cohort_priority_uses_official_start_enrollment_then_created_month() -> None:
    prepared = prepare_cohort_contacts(_contacts(), _fact()).set_index("contact_id")

    assert prepared.loc["c1", "cohort"] == "Official 2026A"
    assert prepared.loc["c1", "cohort_source"] == "official cohort"
    assert prepared.loc["c2", "cohort"] == "2026 Fall II"
    assert prepared.loc["c2", "cohort_source"] == "start_term"
    assert prepared.loc["c3", "cohort"] == "2026-05"
    assert prepared.loc["c3", "cohort_source"] == "enrollment date"
    assert prepared.loc["c4", "cohort"] == "2026-04"
    assert prepared.loc["c4", "cohort_source"] == "contact created month"


def test_cohort_summary_and_calculated_fields() -> None:
    prepared = prepare_cohort_contacts(_contacts(), _fact())
    summary = cohort_summary(prepared).set_index("cohort")
    fields = cohort_calculated_fields(prepared).set_index("cohort")

    assert summary.loc["Official 2026A", "leads"] == 1
    assert summary.loc["Official 2026A", "paid leads"] == 1
    assert summary.loc["Official 2026A", "enrolled students"] == 1
    assert summary.loc["Official 2026A", "actual revenue"] == 9000
    assert summary.loc["Official 2026A", "estimated revenue"] == 10000
    assert summary.loc["Official 2026A", "avg days to first contact"] == 1
    assert summary.loc["Official 2026A", "avg days to enroll"] == 10
    assert summary.loc["Official 2026A", "avg touches to enroll"] == 4
    assert summary.loc["Official 2026A", "top vendor"] == "Atra"

    assert fields.loc["Official 2026A", "cohort_size"] == 1
    assert pd.notna(fields.loc["Official 2026A", "cohort_start_date"])
    assert fields.loc["Official 2026A", "cohort_paid_share"] == 1
    assert fields.loc["Official 2026A", "cohort_hot_lead_share"] == 1
    assert fields.loc["2026-04", "cohort_dead_lead_share"] == 1
