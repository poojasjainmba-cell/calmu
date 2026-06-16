from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from config import (
    FIELD_MAPPING_PATH,
    FIELD_INVENTORY_PATH,
    LAST_REFRESH_PATH,
    REFRESH_STATUS_PATH,
    ensure_directories,
    friendly_missing_token_message,
    load_json_safe,
    save_json,
    token_is_set,
)
from etl import process_hubspot_data
from field_mapping import mapped_properties_for_object
from field_profiler import profile_fields
from hubspot_client import HubSpotAPIError, HubSpotClient, HubSpotPermissionError
from storage import save_processed_tables, save_raw_tables


STUDENT_JOURNEY_CONTACT_PROPERTIES = [
    "firstname",
    "lastname",
    "hs_full_name_or_email",
    "email",
    "phone",
    "mobilephone",
    "final_lead_status",
    "application_status",
    "application_date",
    "application_submitted_date",
    "enrollment_date",
    "enrolled_date",
    "hubspot_owner_assigneddate",
    "notes_last_updated",
    "notes_next_activity_date",
    "num_notes",
    "hs_last_sales_activity_timestamp",
    "hs_time_to_first_engagement",
    "hs_first_outreach_date",
    "hs_sa_first_engagement_date",
    "hs_sa_first_engagement_descr",
    "hs_sa_first_engagement_object_type",
    "program",
    "program_of_interest",
    "degree_program",
    "intended_program",
    "degree_level",
    "enrollment_status",
    "session_start",
    "start_term",
    "cohort",
    "campus",
    "campus_location",
    "modality",
    "utm_campaign",
]

STUDENT_JOURNEY_DEAL_PROPERTIES = [
    "hs_closed_won_date",
    "hubspot_owner_assigneddate",
    "amount",
    "dealstage",
    "closedate",
]

ACTIVITY_PROPERTIES = {
    "calls": [
        "hs_timestamp",
        "hs_call_title",
        "hs_call_body",
        "hs_call_status",
        "hs_call_disposition",
        "hs_call_duration",
        "hubspot_owner_id",
    ],
    "emails": [
        "hs_timestamp",
        "hs_email_subject",
        "hs_email_text",
        "hs_email_html",
        "hs_email_status",
        "hs_email_direction",
        "hubspot_owner_id",
    ],
    "meetings": [
        "hs_timestamp",
        "hs_meeting_title",
        "hs_meeting_body",
        "hs_meeting_outcome",
        "hubspot_owner_id",
    ],
    "notes": [
        "hs_timestamp",
        "hs_note_body",
        "hubspot_owner_id",
    ],
    "tasks": [
        "hs_timestamp",
        "hs_task_subject",
        "hs_task_body",
        "hs_task_status",
        "hs_task_priority",
        "hubspot_owner_id",
    ],
}

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_MAX_CONTACTS = 10000
HUBSPOT_SEARCH_RESULT_LIMIT = 10000
VENDOR_SOURCE_PROPERTIES = [
    "source",
    "source_group",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "campaigns",
    "hs_analytics_source",
    "hs_analytics_source_data_1",
    "hs_analytics_source_data_2",
    "hs_latest_source",
    "hs_latest_source_data_1",
    "hs_latest_source_data_2",
    "paid_lead_list",
]


def _status_payload(status: str, message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_status(status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    save_json(REFRESH_STATUS_PATH, _status_payload(status, message, extra))


def _load_or_create_mapping() -> dict[str, Any]:
    mapping = load_json_safe(FIELD_MAPPING_PATH, default=None)
    if mapping:
        return mapping
    print("Dashboard field mapping not found. Running HubSpot field profiler first...")
    _inventory, mapping, _removed = profile_fields()
    return mapping


def _vendor_candidate_properties() -> list[str]:
    candidates = set(VENDOR_SOURCE_PROPERTIES)
    if FIELD_INVENTORY_PATH.exists():
        inventory = pd.read_csv(FIELD_INVENTORY_PATH)
        terms = ("vendor", "partner", "agency", "referral", "source", "campaign")
        for row in inventory.to_dict("records"):
            if str(row.get("object_type", "")).lower() != "contact":
                continue
            name = str(row.get("property_name", "") or "")
            label = str(row.get("label", "") or "")
            usable = str(row.get("usable", "")).lower() in {"true", "1", "yes"}
            if usable and any(term in f"{name} {label}".lower() for term in terms):
                candidates.add(name)
    return sorted(candidate for candidate in candidates if candidate)


def _fetch_optional_activities(client: HubSpotClient) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for activity_type, properties in ACTIVITY_PROPERTIES.items():
        try:
            print(f"Fetching {activity_type}...", flush=True)
            activities = client.fetch_activity_objects(activity_type, properties=properties)
            print(f"Fetched {activity_type}: {len(activities):,}", flush=True)
            if activities.empty:
                continue
            activity_ids = activities["hs_object_id"] if "hs_object_id" in activities.columns else activities.get("id")
            associations = client.fetch_activity_contact_associations(
                activity_type,
                activity_ids,
                progress_label=activity_type,
            )
            activities = activities.copy()
            activities["activity_type"] = activity_type
            if associations.empty:
                activities["contact_id"] = pd.NA
                frames.append(activities)
            else:
                frames.append(
                    activities.merge(
                        associations[["activity_type", "activity_id", "contact_id", "association_type"]],
                        left_on=["activity_type", "hs_object_id"],
                        right_on=["activity_type", "activity_id"],
                        how="left",
                    )
                )
        except HubSpotPermissionError as exc:
            print(f"Skipping {activity_type}: token lacks activity read permission ({exc})", flush=True)
        except HubSpotAPIError as exc:
            print(f"Skipping {activity_type}: activity endpoint unavailable ({exc})", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _sync_detailed_activities_enabled() -> bool:
    return os.getenv("HUBSPOT_SYNC_ACTIVITIES", "").strip().lower() in TRUTHY_ENV_VALUES


def _max_records_from_env(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def sync_hubspot() -> dict[str, pd.DataFrame]:
    ensure_directories()
    if not token_is_set():
        raise RuntimeError(friendly_missing_token_message())

    print("Loading dashboard field mapping...", flush=True)
    mapping = _load_or_create_mapping()
    contact_properties = sorted(
        set(mapped_properties_for_object("contact", mapping))
        | set(STUDENT_JOURNEY_CONTACT_PROPERTIES)
        | set(_vendor_candidate_properties())
    )
    deal_properties = sorted(set(mapped_properties_for_object("deal", mapping)) | set(STUDENT_JOURNEY_DEAL_PROPERTIES))
    print(
        f"Requesting {len(contact_properties)} contact properties and {len(deal_properties)} deal properties.",
        flush=True,
    )

    client = HubSpotClient()
    max_contacts = _max_records_from_env("HUBSPOT_MAX_CONTACTS", DEFAULT_MAX_CONTACTS)
    max_deals = _max_records_from_env("HUBSPOT_MAX_DEALS", None)
    if max_contacts:
        if max_contacts > HUBSPOT_SEARCH_RESULT_LIMIT:
            print(
                f"HubSpot search supports up to {HUBSPOT_SEARCH_RESULT_LIMIT:,} recent contacts per hosted sync. "
                f"Using {HUBSPOT_SEARCH_RESULT_LIMIT:,}. Set HUBSPOT_MAX_CONTACTS=0 for an all-contact sync.",
                flush=True,
            )
            max_contacts = HUBSPOT_SEARCH_RESULT_LIMIT
        print(f"Fetching most recent contacts, limited to {max_contacts:,}. Set HUBSPOT_MAX_CONTACTS=0 for all contacts.", flush=True)
        contacts = client.search_objects(
            "contacts",
            contact_properties,
            max_records=max_contacts,
            progress_label="contacts",
            sorts=[{"propertyName": "createdate", "direction": "DESCENDING"}],
        )
    else:
        print("Fetching all contacts...", flush=True)
        contacts = client.fetch_objects("contacts", contact_properties, progress_label="contacts")
    print(f"Fetched contacts: {len(contacts):,}", flush=True)
    if max_deals:
        print(f"Fetching most recent deals, limited to {max_deals:,}.", flush=True)
        deals = client.search_objects(
            "deals",
            deal_properties,
            max_records=max_deals,
            progress_label="deals",
            sorts=[{"propertyName": "createdate", "direction": "DESCENDING"}],
        )
    else:
        print("Fetching all deals...", flush=True)
        deals = client.fetch_objects("deals", deal_properties, progress_label="deals")
    print(f"Fetched deals: {len(deals):,}", flush=True)
    print("Fetching owners...", flush=True)
    owners = client.fetch_owners(progress_label="owners")
    print(f"Fetched owners: {len(owners):,}", flush=True)
    if _sync_detailed_activities_enabled():
        activities = _fetch_optional_activities(client)
    else:
        activities = pd.DataFrame()
        print(
            "Skipping detailed activity history by default. Set HUBSPOT_SYNC_ACTIVITIES=true to enable it.",
            flush=True,
        )
    if activities.empty:
        print("Detailed activity history is unavailable; dashboard will use contact summary activity fields.", flush=True)
    else:
        print(f"Fetched detailed activity rows: {len(activities):,}", flush=True)

    contact_ids = contacts["hs_object_id"] if "hs_object_id" in contacts.columns else contacts.get("id", pd.Series(dtype=str))
    print("Fetching contact-to-deal associations...", flush=True)
    associations = client.fetch_contact_deal_associations(contact_ids, progress_label="contacts")
    print(f"Fetched contact-to-deal association links: {len(associations):,}", flush=True)

    print("Saving raw HubSpot cache...", flush=True)
    save_raw_tables(
        {
            "contacts": contacts,
            "deals": deals,
            "owners": owners,
            "associations": associations,
            "activities": activities,
        }
    )
    print("Building processed dashboard tables...", flush=True)
    processed = process_hubspot_data(contacts, deals, owners, associations, mapping, activities)
    print("Saving processed dashboard tables...", flush=True)
    save_processed_tables(processed)
    refreshed_at = datetime.now(timezone.utc).isoformat()
    LAST_REFRESH_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_REFRESH_PATH.write_text(refreshed_at, encoding="utf-8")
    _write_status(
        "success",
        "HubSpot data refreshed successfully.",
        {
            "contacts": int(len(contacts)),
            "deals": int(len(deals)),
            "owners": int(len(owners)),
            "associations": int(len(associations)),
            "activities": int(len(activities)),
            "activity_history_available": bool(not activities.empty),
            "contact_limit": int(max_contacts) if max_contacts else None,
            "deal_limit": int(max_deals) if max_deals else None,
        },
    )
    return processed


def main() -> int:
    try:
        processed = sync_hubspot()
    except RuntimeError as exc:
        message = str(exc)
        print(message)
        _write_status("failed", message)
        return 1
    except HubSpotAPIError as exc:
        message = f"HubSpot sync failed: {exc}"
        print(message)
        _write_status("failed", message)
        return 1
    except Exception as exc:
        message = f"Sync failed unexpectedly: {exc}"
        print(message)
        _write_status("failed", message)
        return 1

    print("HubSpot sync complete.")
    for name, table in processed.items():
        print(f"{name}: {len(table)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
