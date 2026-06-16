from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import (
    FIELD_INVENTORY_PATH,
    FIELD_MAPPING_PATH,
    REMOVED_FIELDS_PATH,
    ensure_directories,
    friendly_missing_token_message,
    save_json,
    token_is_set,
)
from hubspot_client import HubSpotAPIError, HubSpotClient


DASHBOARD_FIELDS: dict[str, dict[str, list[str]]] = {
    "contact": {
        "contact_id": ["hs_object_id"],
        "contact_created_at": ["createdate"],
        "contact_owner_id": ["hubspot_owner_id"],
        "first_name": ["firstname", "first_name"],
        "last_name": ["lastname", "last_name"],
        "full_name": ["hs_full_name_or_email", "fullname", "full_name", "name"],
        "email": ["email"],
        "phone": ["phone", "mobilephone"],
        "lifecycle_stage": ["lifecyclestage"],
        "lead_status": ["hs_lead_status"],
        "final_lead_status": ["final_lead_status"],
        "original_source": ["hs_analytics_source"],
        "latest_source": ["hs_latest_source"],
        "source": ["source", "hs_analytics_source", "hs_latest_source"],
        "source_detail": [
            "hs_analytics_source_data_1",
            "hs_analytics_source_data_2",
            "hs_latest_source_data_1",
            "hs_latest_source_data_2",
        ],
        "utm_source": ["utm_source"],
        "utm_medium": ["utm_medium"],
        "utm_campaign": ["utm_campaign"],
        "campaign": ["utm_campaign", "campaign", "campaigns", "hs_analytics_source_data_2"],
        "vendor": ["vendor", "partner", "agency", "referral_partner", "paid_lead_vendor", "paid_lead_list"],
        "program": ["program", "program_of_interest", "degree_program", "academic_program"],
        "degree_level": ["degree_level", "desired_degree_level"],
        "degree_program": ["degree_program", "degree", "program"],
        "intended_program": ["intended_program", "program_interest", "degreeprogram_interest"],
        "student_type": ["student_type", "secondary_student_type", "transfer_student"],
        "enrollment_status": ["enrollment_status", "hs_prospecting_agent_enrollment_status"],
        "enrollment_date": ["enrollment_date", "enrolled_date", "student_enrollment_date", "start_date"],
        "application_date": ["application_date", "application_submitted_date", "applied_date"],
        "start_term": ["start_term", "session_start", "term_start", "start_date"],
        "cohort": ["cohort", "cohort_term", "cohort_start"],
        "campus": ["campus", "campus_location", "work_campus"],
        "modality": ["modality", "learning_modality", "delivery_modality"],
        "program_total_tuition": [
            "standard_tuition_total",
            "total_tuition_and_fees",
            "total_tuition",
            "program_tuition_total",
            "tuition_total",
            "estimated_total_tuition",
        ],
        "last_activity_date": [
            "notes_last_updated",
            "hs_lastmodifieddate",
            "hs_sales_email_last_replied",
            "hs_last_sales_activity_timestamp",
        ],
        "owner_assigned_date": ["hubspot_owner_assigneddate"],
        "first_activity_date": [
            "hs_first_outreach_date",
            "hs_sa_first_engagement_date",
            "hs_time_to_first_engagement",
        ],
        "next_activity_date": ["hs_next_activity_date"],
        "number_of_sales_activities": ["num_notes", "hs_sales_email_clicks", "hs_sales_email_opens"],
        "time_to_first_engagement": ["hs_time_to_first_engagement"],
    },
    "deal": {
        "deal_id": ["hs_object_id"],
        "deal_created_at": ["createdate"],
        "deal_owner_id": ["hubspot_owner_id"],
        "deal_owner_assigned_date": ["hubspot_owner_assigneddate"],
        "deal_stage": ["dealstage"],
        "pipeline": ["pipeline"],
        "close_date": ["closedate"],
        "closed_won_date": ["hs_closed_won_date"],
        "deal_amount": ["amount", "revenue", "tuition", "net_revenue", "program_revenue"],
        "revenue": ["amount", "revenue", "tuition", "net_revenue", "program_revenue"],
        "is_closed_won": ["hs_is_closed_won"],
        "is_closed": ["hs_is_closed"],
    },
}

SEARCH_HINTS: dict[str, list[str]] = {
    "phone": ["phone", "mobile"],
    "first_name": ["first", "name"],
    "last_name": ["last", "name"],
    "full_name": ["full", "name"],
    "source_detail": ["source", "detail"],
    "source": ["source"],
    "campaign": ["campaign", "utm"],
    "vendor": ["vendor", "partner", "agency", "referral", "source", "campaign"],
    "program": ["program", "degree", "academic", "interest"],
    "degree_level": ["degree", "level"],
    "degree_program": ["degree", "program", "major", "academic"],
    "intended_program": ["program", "degree", "major", "concentration", "academic", "interest"],
    "student_type": ["student", "type"],
    "enrollment_status": ["enrolled", "enrollment", "status", "student"],
    "enrollment_date": ["enrolled", "enrollment", "date"],
    "application_date": ["application", "applied", "date"],
    "start_term": ["term", "start", "session"],
    "cohort": ["cohort"],
    "campus": ["campus"],
    "modality": ["modality"],
    "program_total_tuition": ["tuition", "total"],
    "last_activity_date": ["last", "activity", "modified", "reply"],
    "owner_assigned_date": ["owner", "assigned", "date"],
    "first_activity_date": ["first", "activity", "engagement", "outreach"],
    "next_activity_date": ["next", "activity"],
    "number_of_sales_activities": ["sales", "activity", "notes"],
    "time_to_first_engagement": ["time", "first", "engagement"],
    "deal_amount": ["amount", "revenue", "tuition"],
    "revenue": ["amount", "revenue", "tuition"],
    "lead_status": ["lead", "status"],
    "lifecycle_stage": ["lifecycle", "stage"],
}


def _normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else None


def _sample_values(series: pd.Series, limit: int = 5) -> str:
    values = []
    for value in series.dropna().astype(str):
        cleaned = value.strip()
        if cleaned and cleaned not in values:
            values.append(cleaned)
        if len(values) >= limit:
            break
    return " | ".join(values)


def _profile_object_properties(
    object_type: str,
    properties_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_records = len(sample_df)
    for prop in properties_df.to_dict("records"):
        prop_name = prop.get("name")
        if not prop_name:
            continue
        series = sample_df[prop_name] if prop_name in sample_df.columns else pd.Series(dtype="object")
        non_null = int(series.map(_normalize_value).dropna().shape[0]) if total_records else 0
        percent_filled = round((non_null / total_records) * 100, 2) if total_records else 0.0
        rows.append(
            {
                "object_type": object_type,
                "property_name": prop_name,
                "label": prop.get("label") or prop_name,
                "data_type": prop.get("type"),
                "field_type": prop.get("fieldType"),
                "non_null_count": non_null,
                "percent_filled": percent_filled,
                "sample_values": _sample_values(series),
                "usable": bool(non_null > 0),
            }
        )
    return pd.DataFrame(rows)


def _score_alternative(logical_name: str, property_name: str, label: str) -> int:
    haystack = f"{property_name} {label}".lower().replace("_", " ")
    tokens = SEARCH_HINTS.get(logical_name) or logical_name.split("_")
    return sum(1 for token in tokens if token.lower() in haystack)


def _find_usable_property(
    inventory: pd.DataFrame,
    object_type: str,
    logical_name: str,
    preferred: list[str],
) -> str | None:
    scoped = inventory[(inventory["object_type"] == object_type) & (inventory["usable"] == True)]
    if scoped.empty:
        return None

    by_name = {row["property_name"]: row for row in scoped.to_dict("records")}
    for candidate in preferred:
        if candidate in by_name:
            return candidate

    alternatives: list[tuple[int, float, str]] = []
    for row in scoped.to_dict("records"):
        prop_name = row.get("property_name") or ""
        label = row.get("label") or ""
        score = _score_alternative(logical_name, prop_name, label)
        threshold = min(2, len(SEARCH_HINTS.get(logical_name) or logical_name.split("_")))
        if score >= threshold:
            alternatives.append((score, float(row.get("percent_filled") or 0), prop_name))
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return alternatives[0][2]


def build_dashboard_field_mapping(inventory: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    mapping: dict[str, Any] = {
        "contact": {},
        "deal": {},
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rule": "Preferred populated fields first; populated label/name alternatives only.",
        },
    }
    removed_rows: list[dict[str, Any]] = []

    for object_type, fields in DASHBOARD_FIELDS.items():
        for logical_name, preferred in fields.items():
            selected = _find_usable_property(inventory, object_type, logical_name, preferred)
            mapping[object_type][logical_name] = selected
            if not selected:
                removed_rows.append(
                    {
                        "object_type": object_type,
                        "dashboard_field": logical_name,
                        "reason": "No populated HubSpot property found.",
                        "preferred_candidates": ", ".join(preferred),
                    }
                )

    return mapping, pd.DataFrame(removed_rows)


def profile_fields(sample_size: int = 500) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    ensure_directories()
    if not token_is_set():
        raise RuntimeError(friendly_missing_token_message())

    client = HubSpotClient()
    contact_props = client.get_properties("contacts")
    deal_props = client.get_properties("deals")

    contact_property_names = contact_props["name"].dropna().tolist() if "name" in contact_props else []
    deal_property_names = deal_props["name"].dropna().tolist() if "name" in deal_props else []
    contacts_sample = client.search_objects("contacts", contact_property_names, max_records=sample_size)
    deals_sample = client.search_objects("deals", deal_property_names, max_records=sample_size)

    inventory = pd.concat(
        [
            _profile_object_properties("contact", contact_props, contacts_sample),
            _profile_object_properties("deal", deal_props, deals_sample),
        ],
        ignore_index=True,
    )
    mapping, removed = build_dashboard_field_mapping(inventory)

    inventory.to_csv(FIELD_INVENTORY_PATH, index=False)
    save_json(FIELD_MAPPING_PATH, mapping)
    removed.to_csv(REMOVED_FIELDS_PATH, index=False)
    return inventory, mapping, removed


def main() -> int:
    try:
        inventory, _mapping, removed = profile_fields()
    except RuntimeError as exc:
        print(str(exc))
        return 1
    except HubSpotAPIError as exc:
        print(f"HubSpot field profiling failed: {exc}")
        return 1

    print(f"Saved field inventory: {FIELD_INVENTORY_PATH}")
    print(f"Profiled fields: {len(inventory)}")
    print(f"Dashboard fields removed: {len(removed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
