from __future__ import annotations

import pandas as pd

from field_mapping import field_available, get_mapped_field, required_fields_available
from field_profiler import build_dashboard_field_mapping


def test_field_mapping_picks_populated_preferred_alternative() -> None:
    inventory = pd.DataFrame(
        [
            {"object_type": "contact", "property_name": "phone", "label": "Phone", "usable": False, "percent_filled": 0},
            {"object_type": "contact", "property_name": "mobilephone", "label": "Mobile Phone", "usable": True, "percent_filled": 80},
            {"object_type": "contact", "property_name": "firstname", "label": "First Name", "usable": True, "percent_filled": 80},
            {"object_type": "contact", "property_name": "lastname", "label": "Last Name", "usable": True, "percent_filled": 80},
            {"object_type": "contact", "property_name": "hs_full_name_or_email", "label": "Full Name or Email", "usable": True, "percent_filled": 80},
            {"object_type": "contact", "property_name": "standard_tuition_total", "label": "Standard Tuition Total", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "student_type", "label": "Student Type", "usable": True, "percent_filled": 80},
            {"object_type": "contact", "property_name": "session_start", "label": "Session Start", "usable": True, "percent_filled": 20},
            {"object_type": "contact", "property_name": "enrollment_date", "label": "Enrollment Date", "usable": True, "percent_filled": 20},
            {"object_type": "contact", "property_name": "application_date", "label": "Application Date", "usable": True, "percent_filled": 20},
            {"object_type": "contact", "property_name": "campus_location", "label": "Campus Location", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "modality", "label": "Modality", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "partner_vendor", "label": "Partner Vendor", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "utm_campaign", "label": "UTM Campaign", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "hubspot_owner_assigneddate", "label": "Owner Assigned Date", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "num_notes", "label": "Number of Sales Activities", "usable": True, "percent_filled": 10},
            {"object_type": "contact", "property_name": "hs_object_id", "label": "Record ID", "usable": True, "percent_filled": 100},
            {"object_type": "deal", "property_name": "amount", "label": "Amount", "usable": True, "percent_filled": 95},
            {"object_type": "deal", "property_name": "dealstage", "label": "Deal Stage", "usable": True, "percent_filled": 95},
            {"object_type": "deal", "property_name": "closedate", "label": "Close Date", "usable": True, "percent_filled": 95},
            {"object_type": "deal", "property_name": "hs_object_id", "label": "Record ID", "usable": True, "percent_filled": 100},
        ]
    )

    mapping, removed = build_dashboard_field_mapping(inventory)

    assert mapping["contact"]["phone"] == "mobilephone"
    assert mapping["contact"]["first_name"] == "firstname"
    assert mapping["contact"]["last_name"] == "lastname"
    assert mapping["contact"]["full_name"] == "hs_full_name_or_email"
    assert mapping["contact"]["program_total_tuition"] == "standard_tuition_total"
    assert mapping["contact"]["student_type"] == "student_type"
    assert mapping["contact"]["start_term"] == "session_start"
    assert mapping["contact"]["enrollment_date"] == "enrollment_date"
    assert mapping["contact"]["application_date"] == "application_date"
    assert mapping["contact"]["campus"] == "campus_location"
    assert mapping["contact"]["modality"] == "modality"
    assert mapping["contact"]["vendor"] == "partner_vendor"
    assert mapping["contact"]["campaign"] == "utm_campaign"
    assert mapping["contact"]["owner_assigned_date"] == "hubspot_owner_assigneddate"
    assert mapping["contact"]["number_of_sales_activities"] == "num_notes"
    assert mapping["deal"]["deal_amount"] == "amount"
    assert mapping["deal"]["deal_stage"] == "dealstage"
    assert mapping["deal"]["close_date"] == "closedate"
    assert mapping["deal"]["revenue"] == "amount"
    assert "phone" not in set(removed.get("dashboard_field", []))


def test_missing_fields_do_not_crash() -> None:
    mapping = {"contact": {"email": None}, "deal": {}}

    assert get_mapped_field("email", "contact", mapping) is None
    assert field_available("email", "contact", mapping) is False
    assert required_fields_available(["email"], "contact", mapping) is False
