from __future__ import annotations

import pandas as pd

from etl import build_activity_events
from student_journey import build_journey_events, journey_metrics, search_students


def test_student_search_uses_name_email_phone_and_contact_id() -> None:
    contacts = pd.DataFrame(
        [
            {
                "contact_id": "101",
                "student_name": "Avery Student",
                "email": "avery@example.com",
                "phone": "555-1111",
            },
            {
                "contact_id": "202",
                "student_name": "Blake Learner",
                "email": "blake@example.com",
                "phone": "555-2222",
            },
        ]
    )

    assert search_students(contacts, "avery")["contact_id"].tolist() == ["101"]
    assert search_students(contacts, "blake@example")["contact_id"].tolist() == ["202"]
    assert search_students(contacts, "555-1111")["contact_id"].tolist() == ["101"]
    assert search_students(contacts, "202")["contact_id"].tolist() == ["202"]


def test_activity_objects_become_contact_journey_events() -> None:
    raw_activities = pd.DataFrame(
        [
            {
                "activity_type": "calls",
                "hs_object_id": "call-1",
                "contact_id": "101",
                "hs_timestamp": "2026-01-02T12:00:00Z",
                "hs_call_title": "Intro call",
                "hs_call_status": "COMPLETED",
                "hubspot_owner_id": "owner-1",
            }
        ]
    )
    owners = pd.DataFrame([{"owner_id": "owner-1", "salesman_name": "Owner One"}])

    events = build_activity_events(raw_activities, owners)

    assert events.loc[0, "contact_id"] == "101"
    assert events.loc[0, "event_type"] == "Call"
    assert events.loc[0, "owner"] == "Owner One"
    assert events.loc[0, "description"] == "Intro call"
    assert events.loc[0, "outcome"] == "COMPLETED"


def test_journey_falls_back_to_summary_activity_fields() -> None:
    contact = pd.Series(
        {
            "contact_id": "101",
            "contact_created_at": "2026-01-01T00:00:00Z",
            "student_name": "Avery Student",
            "salesman_name": "Owner One",
            "last_activity_date": "2026-01-04T00:00:00Z",
            "num_notes": 3,
            "lead_temperature": "Warm",
            "has_open_deal": False,
        }
    )
    fact = pd.DataFrame(
        [
            {
                "contact_id": "101",
                "deal_id": "deal-1",
                "deal_created_at": "2026-01-03T00:00:00Z",
                "salesman_name": "Owner One",
                "deal_stage": "Open",
                "is_won": False,
                "is_lost": False,
            }
        ]
    )

    events, detailed_missing = build_journey_events(contact, fact, pd.DataFrame())
    metrics = journey_metrics(contact, fact, events, detailed_missing)

    assert detailed_missing is True
    assert "Last activity" in set(events["event_type"])
    assert metrics["number_of_sales_touches"] == 3
    assert metrics["days_from_lead_creation_to_deal_creation"] == 2
