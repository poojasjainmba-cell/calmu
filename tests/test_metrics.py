from __future__ import annotations

import pandas as pd

from metrics import (
    calculate_data_quality_metrics,
    calculate_executive_metrics,
    calculate_paid_lead_metrics,
    calculate_salesman_metrics,
    monthly_leads,
    monthly_revenue,
)


def test_metrics_revenue_not_double_counted() -> None:
    contacts = pd.DataFrame(
        [
            {"contact_id": "c1", "paid_lead_flag": True, "lead_temperature": "Hot"},
            {"contact_id": "c2", "paid_lead_flag": True, "lead_temperature": "Warm"},
        ]
    )
    fact = pd.DataFrame(
        [
            {"deal_id": "d1", "deal_countable": True, "is_won": True, "revenue_attributed": 1000, "paid_lead_flag": True, "days_to_close": 5},
            {"deal_id": "d1", "deal_countable": False, "is_won": True, "revenue_attributed": 0, "paid_lead_flag": True, "days_to_close": 5},
        ]
    )

    metrics = calculate_executive_metrics(contacts, fact)

    assert metrics["won_deals"] == 1
    assert metrics["revenue"] == 1000


def test_monthly_metrics_exclude_missing_dates() -> None:
    contacts = pd.DataFrame(
        [
            {"contact_id": "c1", "contact_created_at": "2026-05-01T00:00:00Z"},
            {"contact_id": "c2", "contact_created_at": pd.NaT},
        ]
    )
    fact = pd.DataFrame(
        [
            {"deal_id": "d1", "deal_countable": True, "is_won": True, "close_date": "2026-05-10T00:00:00Z", "revenue_attributed": 1000},
            {"deal_id": "d2", "deal_countable": True, "is_won": True, "close_date": pd.NaT, "revenue_attributed": 500},
        ]
    )

    assert monthly_leads(contacts)["month"].tolist() == ["2026-05"]
    assert monthly_revenue(fact)["month"].tolist() == ["2026-05"]


def test_data_quality_metrics_tolerate_missing_columns() -> None:
    contacts = pd.DataFrame([{"contact_id": "c1"}])
    deals = pd.DataFrame([{"deal_id": "d1"}])
    fact = pd.DataFrame([{"contact_id": "c1"}])

    quality = calculate_data_quality_metrics(contacts, deals, fact)

    assert quality["records_missing_owner"] == 1
    assert quality["records_missing_source"] == 1


def test_paid_vendor_potential_uses_all_paid_leads_not_only_won_deals() -> None:
    contacts = pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "paid_lead_flag": True,
                "vendor": "Atra",
                "source_group": "Paid Search",
                "utm_campaign": "Spring",
                "lead_temperature": "Hot",
                "last_activity_date": "2026-01-02T00:00:00Z",
                "num_notes": 2,
                "potential_program_revenue": 10000,
                "open_pipeline_potential_revenue": 10000,
                "potential_six_month_revenue": 5000,
                "potential_twelve_month_revenue": 10000,
                "potential_twenty_four_month_revenue": 10000,
                "potential_annualized_program_revenue": 10000,
                "enrolled_revenue": 10000,
            },
            {
                "contact_id": "c2",
                "paid_lead_flag": True,
                "vendor": "Atra",
                "source_group": "Paid Search",
                "utm_campaign": "Spring",
                "lead_temperature": "Dead",
                "num_notes": 0,
                "potential_program_revenue": 20000,
                "open_pipeline_potential_revenue": 20000,
                "potential_six_month_revenue": 10000,
                "potential_twelve_month_revenue": 20000,
                "potential_twenty_four_month_revenue": 20000,
                "potential_annualized_program_revenue": 20000,
                "enrolled_revenue": 0,
            },
        ]
    )
    fact = pd.DataFrame(
        [
            {
                "contact_id": "c1",
                "deal_id": "d1",
                "deal_countable": True,
                "is_won": True,
                "paid_lead_flag": True,
                "vendor": "Atra",
                "source_group": "Paid Search",
                "utm_campaign": "Spring",
                "revenue_attributed": 1000,
                "total_program_revenue": 10000,
                "six_month_revenue": 5000,
                "twelve_month_revenue": 10000,
                "twenty_four_month_revenue": 10000,
                "annualized_program_revenue": 10000,
                "days_to_close": 10,
            }
        ]
    )

    paid = calculate_paid_lead_metrics(contacts, fact)
    atra = paid["paid_vendor_performance"].set_index("vendor").loc["Atra"]

    assert atra["paid_leads"] == 2
    assert atra["contacted_leads"] == 1
    assert atra["uncontacted_leads"] == 1
    assert atra["hot_leads"] == 1
    assert atra["dead_leads"] == 1
    assert atra["deals_created"] == 1
    assert atra["enrolled_students"] == 1
    assert atra["actual_revenue"] == 1000
    assert atra["estimated_enrolled_revenue"] == 10000
    assert atra["open_potential_revenue"] == 30000
    assert atra["close_rate"] == 0.5
    assert atra["enrollment_rate"] == 0.5


def test_salesman_scorecard_revenue_columns_and_sort() -> None:
    contacts = pd.DataFrame(
        [
            {
                "contact_id": "a1",
                "salesman_name": "Owner A",
                "paid_lead_flag": True,
                "lead_temperature": "Cold",
                "has_open_deal": False,
                "days_since_last_activity": 10,
                "potential_program_revenue": 10000,
                "open_pipeline_potential_revenue": 10000,
                "enrolled_revenue": 0,
                "revenue_confidence": "Low",
            },
            {
                "contact_id": "b1",
                "salesman_name": "Owner B",
                "paid_lead_flag": True,
                "lead_temperature": "Hot",
                "has_open_deal": True,
                "days_since_last_activity": 1,
                "potential_program_revenue": 32799,
                "open_pipeline_potential_revenue": 32799,
                "enrolled_revenue": 32799,
                "revenue_confidence": "Medium",
            },
            {
                "contact_id": "b2",
                "salesman_name": "Owner B",
                "paid_lead_flag": True,
                "lead_temperature": "Cold",
                "has_open_deal": False,
                "days_since_last_activity": 10,
                "potential_program_revenue": 20000,
                "open_pipeline_potential_revenue": 20000,
                "enrolled_revenue": 0,
                "revenue_confidence": "Low",
            },
        ]
    )
    fact = pd.DataFrame(
        [
            {
                "contact_id": "b1",
                "deal_id": "d1",
                "salesman_name": "Owner B",
                "deal_countable": True,
                "is_won": True,
                "paid_lead_flag": True,
                "revenue_attributed": 5000,
                "total_program_revenue": 32799,
                "days_to_close": 12,
            }
        ]
    )

    table = calculate_salesman_metrics(contacts, fact)
    leader = table.iloc[0]

    assert {
        "actual_enrolled_revenue",
        "estimated_enrolled_revenue",
        "open_pipeline_potential_revenue",
        "hot_lead_potential_revenue",
        "paid_lead_potential_revenue",
        "average_estimated_revenue_per_lead",
        "average_estimated_revenue_per_enrolled_student",
        "revenue_confidence_mix",
    }.issubset(table.columns)
    assert leader["salesman_name"] == "Owner B"
    assert leader["action_load"] == 3
    assert leader["paid_lead_leakage"] == 1
    assert leader["actual_enrolled_revenue"] == 5000
    assert leader["estimated_enrolled_revenue"] == 32799
    assert leader["open_pipeline_potential_revenue"] == 52799
    assert leader["hot_lead_potential_revenue"] == 32799
    assert leader["paid_lead_potential_revenue"] == 52799
    assert leader["average_estimated_revenue_per_enrolled_student"] == 32799
    assert leader["revenue_confidence_mix"] == "High: 0 / Medium: 1 / Low: 1"
