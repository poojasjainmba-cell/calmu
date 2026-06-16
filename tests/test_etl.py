from __future__ import annotations

import pandas as pd

from etl import process_hubspot_data


MAPPING = {
    "contact": {
        "contact_id": "hs_object_id",
        "contact_created_at": "createdate",
        "contact_owner_id": "hubspot_owner_id",
        "first_name": "firstname",
        "last_name": "lastname",
        "full_name": "hs_full_name_or_email",
        "email": "email",
        "phone": "phone",
        "lifecycle_stage": "lifecyclestage",
        "lead_status": "hs_lead_status",
        "original_source": "hs_analytics_source",
        "latest_source": None,
        "source": "hs_analytics_source",
        "source_detail": None,
        "utm_source": "utm_source",
        "utm_medium": "utm_medium",
        "utm_campaign": None,
        "campaign": "utm_campaign",
        "vendor": None,
        "program": "program",
        "degree_level": "degree_level",
        "degree_program": "degree_program",
        "intended_program": "intended_program",
        "student_type": "student_type",
        "enrollment_status": "enrollment_status",
        "enrollment_date": "enrollment_date",
        "application_date": "application_date",
        "start_term": "session_start",
        "cohort": "cohort",
        "campus": "campus_location",
        "modality": "modality",
        "program_total_tuition": "standard_tuition_total",
        "owner_assigned_date": "hubspot_owner_assigneddate",
        "first_activity_date": "hs_first_outreach_date",
        "last_activity_date": "notes_last_updated",
        "next_activity_date": None,
        "number_of_sales_activities": "num_notes",
        "time_to_first_engagement": "hs_time_to_first_engagement",
    },
    "deal": {
        "deal_id": "hs_object_id",
        "deal_created_at": "createdate",
        "deal_owner_id": "hubspot_owner_id",
        "deal_stage": "dealstage",
        "pipeline": "pipeline",
        "close_date": "closedate",
        "deal_amount": "amount",
        "revenue": "amount",
    },
}


def test_contact_owner_used_before_deal_owner_and_fallback_only_when_missing() -> None:
    contacts = pd.DataFrame(
        [
            {
                "hs_object_id": "c1",
                "createdate": "2026-05-01T00:00:00Z",
                "hubspot_owner_id": "owner_contact",
                "email": "one@example.com",
                "phone": "",
                "hs_analytics_source": "organic search",
                "utm_source": "",
                "utm_medium": "",
                "notes_last_updated": "2026-05-05T00:00:00Z",
            },
            {
                "hs_object_id": "c2",
                "createdate": "2026-05-02T00:00:00Z",
                "hubspot_owner_id": "",
                "email": "two@example.com",
                "phone": "",
                "hs_analytics_source": "google ads",
                "utm_source": "google ads",
                "utm_medium": "cpc",
                "notes_last_updated": "2026-05-05T00:00:00Z",
            },
        ]
    )
    deals = pd.DataFrame(
        [
            {
                "hs_object_id": "d1",
                "createdate": "2026-05-03T00:00:00Z",
                "hubspot_owner_id": "owner_deal",
                "dealstage": "Open",
                "pipeline": "default",
                "closedate": pd.NaT,
                "amount": "500",
            },
            {
                "hs_object_id": "d2",
                "createdate": "2026-05-04T00:00:00Z",
                "hubspot_owner_id": "owner_deal",
                "dealstage": "Open",
                "pipeline": "default",
                "closedate": pd.NaT,
                "amount": "700",
            },
        ]
    )
    owners = pd.DataFrame(
        [
            {"owner_id": "owner_contact", "salesman_name": "Contact Owner"},
            {"owner_id": "owner_deal", "salesman_name": "Deal Owner"},
        ]
    )
    associations = pd.DataFrame(
        [
            {"contact_id": "c1", "deal_id": "d1"},
            {"contact_id": "c2", "deal_id": "d2"},
        ]
    )

    processed = process_hubspot_data(contacts, deals, owners, associations, MAPPING)
    fact = processed["lead_deal_fact"]

    c1 = fact[fact["contact_id"] == "c1"].iloc[0]
    c2 = fact[fact["contact_id"] == "c2"].iloc[0]
    assert c1["salesman_id"] == "owner_contact"
    assert c1["attribution_type"] == "contact_owner"
    assert c2["salesman_id"] == "owner_deal"
    assert c2["attribution_type"] == "fallback_to_deal_owner"


def test_revenue_is_not_double_counted() -> None:
    contacts = pd.DataFrame(
        [
            {"hs_object_id": "c1", "createdate": "2026-05-01T00:00:00Z", "hubspot_owner_id": "owner_contact", "email": "one@example.com"},
            {"hs_object_id": "c2", "createdate": "2026-05-02T00:00:00Z", "hubspot_owner_id": "owner_contact", "email": "two@example.com"},
        ]
    )
    deals = pd.DataFrame(
        [
            {
                "hs_object_id": "d1",
                "createdate": "2026-05-03T00:00:00Z",
                "hubspot_owner_id": "owner_deal",
                "dealstage": "Closed Won",
                "closedate": "2026-05-10T00:00:00Z",
                "amount": "1000",
            }
        ]
    )
    owners = pd.DataFrame([{"owner_id": "owner_contact", "salesman_name": "Contact Owner"}])
    associations = pd.DataFrame(
        [
            {"contact_id": "c1", "deal_id": "d1"},
            {"contact_id": "c2", "deal_id": "d1"},
        ]
    )

    processed = process_hubspot_data(contacts, deals, owners, associations, MAPPING)
    fact = processed["lead_deal_fact"]

    assert fact["revenue_attributed"].sum() == 1000
    assert fact["revenue_countable"].sum() == 1
    assert fact["unclear_revenue_attribution"].any()


def test_program_tuition_drives_duration_revenue_when_available() -> None:
    contacts = pd.DataFrame(
        [
            {
                "hs_object_id": "c1",
                "createdate": "2026-05-01T00:00:00Z",
                "hubspot_owner_id": "owner_contact",
                "email": "one@example.com",
                "program": "Bachelor of Science",
                "standard_tuition_total": "48000",
            }
        ]
    )
    deals = pd.DataFrame(
        [
            {
                "hs_object_id": "d1",
                "createdate": "2026-05-03T00:00:00Z",
                "hubspot_owner_id": "owner_deal",
                "dealstage": "Closed Won",
                "closedate": "2026-05-10T00:00:00Z",
                "amount": "0",
            }
        ]
    )
    owners = pd.DataFrame([{"owner_id": "owner_contact", "salesman_name": "Contact Owner"}])
    associations = pd.DataFrame([{"contact_id": "c1", "deal_id": "d1"}])

    processed = process_hubspot_data(contacts, deals, owners, associations, MAPPING)
    fact = processed["lead_deal_fact"]
    row = fact.iloc[0]

    assert row["program_revenue_source"] == "contact_program_total_tuition"
    assert row["total_program_revenue"] == 48000
    assert row["program_duration_months"] == 48
    assert row["degree_level"] == "Bachelor"
    assert row["revenue_confidence"] == "Medium"
    assert row["monthly_program_revenue"] == 1000
    assert row["twelve_month_revenue"] == 12000


def test_degree_program_fallback_and_enrolled_revenue_fields() -> None:
    contacts = pd.DataFrame(
        [
            {
                "hs_object_id": "c1",
                "createdate": "2026-05-01T00:00:00Z",
                "hubspot_owner_id": "owner_contact",
                "email": "one@example.com",
                "degree_program": "MBA",
                "enrollment_status": "EA Signed",
                "student_type": "Domestic",
                "session_start": "2026 Fall I",
                "campus_location": "San Diego",
            }
        ]
    )
    deals = pd.DataFrame()
    owners = pd.DataFrame([{"owner_id": "owner_contact", "salesman_name": "Contact Owner"}])
    associations = pd.DataFrame()

    processed = process_hubspot_data(contacts, deals, owners, associations, MAPPING)
    contact = processed["contacts_clean"].iloc[0]

    assert contact["program"] == "MBA"
    assert contact["degree_level"] == "Master"
    assert contact["enrolled_revenue"] == 32799
    assert contact["student_type"] == "Domestic"
    assert contact["start_term"] == "2026 Fall I"
    assert contact["campus"] == "San Diego"


def test_new_processed_tables_and_canonical_dashboard_fields() -> None:
    contacts = pd.DataFrame(
        [
            {
                "hs_object_id": "c1",
                "createdate": "2026-05-01T00:00:00Z",
                "hubspot_owner_id": "owner",
                "firstname": "Avery",
                "lastname": "Student",
                "hs_full_name_or_email": "Avery Student",
                "email": "avery@example.com",
                "phone": "555-0100",
                "hs_analytics_source": "atra paid",
                "utm_source": "atra",
                "utm_medium": "cpc",
                "utm_campaign": "fall-campaign",
                "program": "MBA",
                "degree_level": "Master",
                "enrollment_status": "EA Signed",
                "enrollment_date": "2026-06-01T00:00:00Z",
                "application_date": "2026-05-15T00:00:00Z",
                "session_start": "2026 Fall I",
                "cohort": "Official Fall 2026",
                "campus_location": "San Diego",
                "modality": "Online",
                "hubspot_owner_assigneddate": "2026-05-02T00:00:00Z",
                "hs_first_outreach_date": "2026-05-03T00:00:00Z",
                "notes_last_updated": "2026-05-04T00:00:00Z",
                "num_notes": "5",
                "hs_time_to_first_engagement": "86400000",
            }
        ]
    )
    deals = pd.DataFrame(
        [
            {
                "hs_object_id": "d1",
                "createdate": "2026-05-20T00:00:00Z",
                "hubspot_owner_id": "owner",
                "dealstage": "Closed Won",
                "closedate": "2026-06-10T00:00:00Z",
                "amount": "12000",
            }
        ]
    )
    owners = pd.DataFrame([{"owner_id": "owner", "salesman_name": "Owner"}])
    associations = pd.DataFrame([{"contact_id": "c1", "deal_id": "d1"}])

    processed = process_hubspot_data(contacts, deals, owners, associations, MAPPING)

    for table_name in [
        "contacts_clean",
        "deals_clean",
        "lead_deal_fact",
        "student_journey_fact",
        "cohort_fact",
        "vendor_fact",
        "salesman_revenue_fact",
    ]:
        assert table_name in processed
        assert not processed[table_name].empty

    contact = processed["contacts_clean"].iloc[0]
    deal = processed["deals_clean"].iloc[0]
    journey = processed["student_journey_fact"].iloc[0]
    cohort = processed["cohort_fact"].iloc[0]

    assert contact["first_name"] == "Avery"
    assert contact["last_name"] == "Student"
    assert contact["full_name"] == "Avery Student"
    assert contact["campaign"] == "fall-campaign"
    assert contact["number_of_sales_activities"] == 5
    assert pd.notna(contact["first_activity_date"])
    assert pd.notna(contact["application_date"])
    assert pd.notna(contact["enrollment_date"])
    assert deal["deal_amount"] == 12000
    assert journey["actual_revenue"] == 12000
    assert cohort["cohort"] == "Official Fall 2026"
    assert cohort["cohort_enrolled_students"] == 1


def test_standard_hubspot_closed_flags_drive_won_lost_when_stage_is_internal_id() -> None:
    mapping = {
        **MAPPING,
        "deal": {
            **MAPPING["deal"],
            "is_closed_won": "hs_is_closed_won",
            "is_closed": "hs_is_closed",
        },
    }
    contacts = pd.DataFrame(
        [{"hs_object_id": "c1", "createdate": "2026-05-01T00:00:00Z", "hubspot_owner_id": "owner"}]
    )
    deals = pd.DataFrame(
        [
            {
                "hs_object_id": "d1",
                "createdate": "2026-05-02T00:00:00Z",
                "hubspot_owner_id": "owner",
                "dealstage": "123456789",
                "closedate": "2026-05-10T00:00:00Z",
                "amount": "1000",
                "hs_is_closed_won": "true",
                "hs_is_closed": "true",
            }
        ]
    )
    owners = pd.DataFrame([{"owner_id": "owner", "salesman_name": "Owner"}])
    associations = pd.DataFrame([{"contact_id": "c1", "deal_id": "d1"}])

    processed = process_hubspot_data(contacts, deals, owners, associations, mapping)

    assert bool(processed["deals_clean"].loc[0, "is_won"]) is True
    assert processed["lead_deal_fact"]["revenue_attributed"].sum() == 1000
