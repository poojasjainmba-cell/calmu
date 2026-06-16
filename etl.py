from __future__ import annotations

from typing import Any

import pandas as pd

from lead_scoring import parse_datetime_series, score_leads
from revenue_estimator import add_program_revenue_fields
from source_classification import add_source_classification, vendor_source_columns


CONTACT_COLUMNS = [
    "contact_id",
    "contact_created_at",
    "contact_owner_id",
    "owner_assigned_date",
    "salesman_id",
    "salesman_name",
    "first_name",
    "last_name",
    "full_name",
    "student_name",
    "email",
    "phone",
    "lifecycle_stage",
    "lead_status",
    "final_lead_status",
    "original_source",
    "latest_source",
    "source_detail",
    "source",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "campaign",
    "program",
    "degree_level",
    "degree_program",
    "intended_program",
    "student_type",
    "enrollment_status",
    "enrollment_date",
    "application_date",
    "start_term",
    "cohort",
    "campus",
    "modality",
    "program_degree_level",
    "program_duration_years",
    "program_duration_months",
    "program_total_tuition",
    "tuition_per_credit",
    "credits_required",
    "estimated_total_tuition",
    "estimated_annual_tuition",
    "potential_revenue",
    "enrolled_revenue",
    "open_pipeline_potential_revenue",
    "revenue_confidence",
    "tuition_estimate_source",
    "last_activity_date",
    "first_activity_date",
    "next_activity_date",
    "num_notes",
    "number_of_sales_activities",
    "last_sales_activity_timestamp",
    "time_to_first_engagement",
    "first_outreach_date",
    "first_engagement_date",
    "first_engagement_description",
    "first_engagement_type",
    "application_status",
    "paid_lead_flag",
    "vendor",
    "vendor_confidence",
    "vendor_source_field",
    "paid_vendor",
    "source_group",
    "source_text",
]

DEAL_COLUMNS = [
    "deal_id",
    "deal_created_at",
    "deal_owner_id",
    "deal_owner_assigned_date",
    "deal_stage",
    "pipeline",
    "close_date",
    "closed_won_date",
    "deal_amount",
    "revenue",
    "is_won",
    "is_lost",
]


def _mapping_section(mapping: dict[str, Any], object_type: str) -> dict[str, Any]:
    return mapping.get(object_type) or mapping.get(f"{object_type}s") or {}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _logical_series(
    raw_df: pd.DataFrame,
    mapping: dict[str, Any],
    object_type: str,
    logical_name: str,
    fallbacks: list[str] | None = None,
) -> pd.Series:
    fallbacks = fallbacks or []
    mapped = _mapping_section(mapping, object_type).get(logical_name)
    candidates = [mapped] if mapped else []
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate and candidate in raw_df.columns:
            return raw_df[candidate]
    return pd.Series([pd.NA] * len(raw_df), index=raw_df.index)


def _as_clean_id(series: pd.Series) -> pd.Series:
    return series.map(_clean_text).replace("", pd.NA)


def _first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = _clean_text(row.get(key))
        if value:
            return value
    return ""


def _owner_lookup(owners_df: pd.DataFrame) -> dict[str, str]:
    if owners_df.empty or "owner_id" not in owners_df.columns:
        return {}
    lookup: dict[str, str] = {}
    for row in owners_df.to_dict("records"):
        owner_id = _clean_text(row.get("owner_id"))
        name = _clean_text(row.get("salesman_name")) or _clean_text(row.get("email")) or owner_id
        if owner_id:
            lookup[owner_id] = name
    return lookup


def _map_owner_name(owner_ids: pd.Series, lookup: dict[str, str]) -> pd.Series:
    return owner_ids.map(lambda owner_id: lookup.get(_clean_text(owner_id), "Unassigned"))


def _first_nonempty_series(*series: pd.Series) -> pd.Series:
    if not series:
        return pd.Series(dtype=object)
    output = series[0].copy()
    output = output.where(output.map(lambda value: bool(_clean_text(value))), pd.NA)
    for candidate in series[1:]:
        candidate = candidate.where(candidate.map(lambda value: bool(_clean_text(value))), pd.NA)
        output = output.combine_first(candidate)
    return output


def build_contacts_clean(
    raw_contacts: pd.DataFrame,
    owners_df: pd.DataFrame,
    mapping: dict[str, Any],
) -> pd.DataFrame:
    owners = _owner_lookup(owners_df)
    contacts = pd.DataFrame(index=raw_contacts.index)
    contacts["contact_id"] = _as_clean_id(
        _logical_series(raw_contacts, mapping, "contact", "contact_id", ["hs_object_id", "id"])
    )
    contacts["contact_created_at"] = parse_datetime_series(
        _logical_series(raw_contacts, mapping, "contact", "contact_created_at", ["createdate", "createdAt"])
    )
    contacts["contact_owner_id"] = _as_clean_id(
        _logical_series(raw_contacts, mapping, "contact", "contact_owner_id", ["hubspot_owner_id"])
    )
    contacts["owner_assigned_date"] = parse_datetime_series(
        _logical_series(raw_contacts, mapping, "contact", "owner_assigned_date", ["hubspot_owner_assigneddate"])
    )
    contacts["first_name"] = _logical_series(raw_contacts, mapping, "contact", "first_name", ["firstname"])
    contacts["last_name"] = _logical_series(raw_contacts, mapping, "contact", "last_name", ["lastname"])
    generated_full_name = (
        contacts["first_name"].fillna("").astype(str).str.strip()
        + " "
        + contacts["last_name"].fillna("").astype(str).str.strip()
    ).str.strip()
    contacts["full_name"] = _first_nonempty_series(
        _logical_series(raw_contacts, mapping, "contact", "full_name", ["hs_full_name_or_email", "fullname"]),
        generated_full_name,
    )
    contacts["student_name"] = _first_nonempty_series(
        contacts["full_name"],
        _logical_series(raw_contacts, mapping, "contact", "student_name", ["hs_full_name_or_email"]),
    )

    for logical_name in (
        "email",
        "phone",
        "lifecycle_stage",
        "lead_status",
        "final_lead_status",
        "original_source",
        "latest_source",
        "source",
        "source_detail",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "campaign",
        "vendor",
        "program",
        "degree_level",
        "degree_program",
        "intended_program",
        "student_type",
        "enrollment_status",
        "enrollment_date",
        "application_date",
        "start_term",
        "cohort",
        "campus",
        "modality",
        "last_activity_date",
        "first_activity_date",
        "next_activity_date",
        "num_notes",
        "number_of_sales_activities",
        "last_sales_activity_timestamp",
        "time_to_first_engagement",
        "first_outreach_date",
        "first_engagement_date",
        "first_engagement_description",
        "first_engagement_type",
        "application_status",
    ):
        contacts[logical_name] = _logical_series(
            raw_contacts,
            mapping,
            "contact",
            logical_name,
            {
                "last_activity_date": ["notes_last_updated"],
                "next_activity_date": ["notes_next_activity_date"],
                "last_sales_activity_timestamp": ["hs_last_sales_activity_timestamp"],
                "time_to_first_engagement": ["hs_time_to_first_engagement"],
                "first_outreach_date": ["hs_first_outreach_date"],
                "first_engagement_date": ["hs_sa_first_engagement_date"],
                "first_activity_date": ["hs_first_outreach_date", "hs_sa_first_engagement_date"],
                "first_engagement_description": ["hs_sa_first_engagement_descr"],
                "first_engagement_type": ["hs_sa_first_engagement_object_type"],
                "final_lead_status": ["final_lead_status"],
                "application_status": ["application_status"],
                "application_date": ["application_date", "application_submitted_date"],
                "enrollment_date": ["enrollment_date", "enrolled_date"],
                "source": ["source", "hs_analytics_source", "hs_latest_source"],
                "campaign": ["utm_campaign", "campaign", "campaigns"],
                "number_of_sales_activities": ["num_notes"],
            }.get(logical_name, []),
        )
    for column in (
        "enrollment_date",
        "application_date",
        "last_activity_date",
        "first_activity_date",
        "next_activity_date",
        "last_sales_activity_timestamp",
        "first_outreach_date",
        "first_engagement_date",
    ):
        contacts[column] = parse_datetime_series(contacts[column])
    contacts["first_activity_date"] = contacts["first_activity_date"].combine_first(
        parse_datetime_series(contacts["first_outreach_date"])
    ).combine_first(parse_datetime_series(contacts["first_engagement_date"]))
    contacts["num_notes"] = pd.to_numeric(contacts["num_notes"], errors="coerce").fillna(0)
    contacts["number_of_sales_activities"] = pd.to_numeric(
        contacts["number_of_sales_activities"],
        errors="coerce",
    ).fillna(contacts["num_notes"])
    contacts["time_to_first_engagement"] = pd.to_numeric(
        contacts["time_to_first_engagement"], errors="coerce"
    )
    contacts["program"] = contacts["program"].replace("", pd.NA)
    for fallback_column in ("degree_program", "intended_program"):
        contacts[fallback_column] = contacts[fallback_column].replace("", pd.NA)
        contacts["program"] = contacts["program"].combine_first(contacts[fallback_column])
    contacts["program_total_tuition"] = pd.to_numeric(
        _logical_series(
            raw_contacts,
            mapping,
            "contact",
            "program_total_tuition",
            ["standard_tuition_total", "total_tuition_and_fees"],
        ),
        errors="coerce",
    ).fillna(0)

    contacts["salesman_id"] = contacts["contact_owner_id"]
    contacts["salesman_name"] = _map_owner_name(contacts["salesman_id"], owners)

    for column in vendor_source_columns(raw_contacts):
        if column not in contacts.columns:
            contacts[column] = raw_contacts[column]

    contacts = add_source_classification(
        contacts,
        vendor_source_columns(contacts),
    )
    contacts["has_open_deal"] = False
    contacts["has_won_deal"] = False
    contacts["has_lost_deal"] = False
    contacts["latest_deal_stage"] = pd.NA
    contacts["contact_attribution_type"] = contacts["contact_owner_id"].map(
        lambda owner_id: "contact_owner" if _clean_text(owner_id) else "missing_owner"
    )
    contacts = add_program_revenue_fields(contacts, total_revenue_column="contact_realized_revenue")
    return contacts


def build_deals_clean(raw_deals: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    deals = pd.DataFrame(index=raw_deals.index)
    deals["deal_id"] = _as_clean_id(
        _logical_series(raw_deals, mapping, "deal", "deal_id", ["hs_object_id", "id"])
    )
    deals["deal_created_at"] = parse_datetime_series(
        _logical_series(raw_deals, mapping, "deal", "deal_created_at", ["createdate", "createdAt"])
    )
    deals["deal_owner_id"] = _as_clean_id(
        _logical_series(raw_deals, mapping, "deal", "deal_owner_id", ["hubspot_owner_id"])
    )
    deals["deal_owner_assigned_date"] = parse_datetime_series(
        _logical_series(raw_deals, mapping, "deal", "deal_owner_assigned_date", ["hubspot_owner_assigneddate"])
    )
    for logical_name in ("deal_stage", "pipeline", "close_date"):
        deals[logical_name] = _logical_series(raw_deals, mapping, "deal", logical_name)
    deals["close_date"] = parse_datetime_series(deals["close_date"])
    deals["closed_won_date"] = parse_datetime_series(
        _logical_series(raw_deals, mapping, "deal", "closed_won_date", ["hs_closed_won_date"])
    )
    deals["deal_amount"] = pd.to_numeric(
        _logical_series(raw_deals, mapping, "deal", "deal_amount", ["amount", "revenue"]),
        errors="coerce",
    ).fillna(0)
    mapped_revenue = pd.to_numeric(
        _logical_series(raw_deals, mapping, "deal", "revenue", ["amount"]),
        errors="coerce",
    )
    deals["revenue"] = mapped_revenue.fillna(deals["deal_amount"]).fillna(0)

    stage = deals["deal_stage"].fillna("").astype(str).str.lower()
    closed_won = _logical_series(raw_deals, mapping, "deal", "is_closed_won", ["hs_is_closed_won"])
    closed = _logical_series(raw_deals, mapping, "deal", "is_closed", ["hs_is_closed"])
    has_closed_won = closed_won.map(_clean_text).ne("").any()
    has_closed = closed.map(_clean_text).ne("").any()
    deals["is_won"] = _as_bool(closed_won) if has_closed_won else stage.str.contains("won", na=False)
    deals["is_lost"] = (
        _as_bool(closed) & ~deals["is_won"] if has_closed else stage.str.contains("lost", na=False)
    )
    return deals


def build_activity_events(raw_activities: pd.DataFrame, owners_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "contact_id",
        "activity_id",
        "event_date",
        "event_type",
        "owner_id",
        "owner",
        "description",
        "outcome",
        "source",
    ]
    if raw_activities.empty:
        return pd.DataFrame(columns=columns)

    owners = _owner_lookup(owners_df)
    rows: list[dict[str, Any]] = []
    for raw in raw_activities.to_dict("records"):
        activity_type = _clean_text(raw.get("activity_type")).lower()
        event_type = {
            "calls": "Call",
            "emails": "Email",
            "meetings": "Meeting",
            "notes": "Note",
            "tasks": "Task",
        }.get(activity_type, activity_type.title() or "Activity")
        description = _first_text(
            raw,
            {
                "calls": ["hs_call_title", "hs_call_body"],
                "emails": ["hs_email_subject", "hs_email_text", "hs_email_html"],
                "meetings": ["hs_meeting_title", "hs_meeting_body"],
                "notes": ["hs_note_body"],
                "tasks": ["hs_task_subject", "hs_task_body"],
            }.get(activity_type, []),
        )
        outcome = _first_text(
            raw,
            {
                "calls": ["hs_call_disposition", "hs_call_status"],
                "emails": ["hs_email_status", "hs_email_direction"],
                "meetings": ["hs_meeting_outcome"],
                "tasks": ["hs_task_status", "hs_task_priority"],
            }.get(activity_type, []),
        )
        owner_id = _clean_text(raw.get("hubspot_owner_id"))
        rows.append(
            {
                "contact_id": _clean_text(raw.get("contact_id")),
                "activity_id": _clean_text(raw.get("hs_object_id") or raw.get("id") or raw.get("activity_id")),
                "event_date": raw.get("hs_timestamp") or raw.get("createdAt"),
                "event_type": event_type,
                "owner_id": owner_id or pd.NA,
                "owner": owners.get(owner_id, "Unassigned") if owner_id else "Unassigned",
                "description": description,
                "outcome": outcome,
                "source": "activity_object",
            }
        )

    events = pd.DataFrame(rows)
    events["event_date"] = parse_datetime_series(events["event_date"])
    events = events[events["contact_id"].astype(str).str.strip().ne("")].copy()
    events = events.dropna(subset=["event_date"])
    if events.empty:
        return pd.DataFrame(columns=columns)
    return events[columns].sort_values(["contact_id", "event_date", "event_type"]).drop_duplicates()


def _association_frame(associations_df: pd.DataFrame) -> pd.DataFrame:
    if associations_df.empty:
        return pd.DataFrame(columns=["contact_id", "deal_id", "association_type"])
    assoc = associations_df.copy()
    for column in ("contact_id", "deal_id", "association_type"):
        if column not in assoc.columns:
            assoc[column] = pd.NA
    assoc["contact_id"] = _as_clean_id(assoc["contact_id"])
    assoc["deal_id"] = _as_clean_id(assoc["deal_id"])
    assoc = assoc.dropna(subset=["contact_id", "deal_id"])
    return assoc[["contact_id", "deal_id", "association_type"]].drop_duplicates()


def summarize_contact_deals(fact_df: pd.DataFrame) -> pd.DataFrame:
    if fact_df.empty or "contact_id" not in fact_df.columns:
        return pd.DataFrame(
            columns=[
                "contact_id",
                "has_open_deal",
                "has_won_deal",
                "has_lost_deal",
                "fallback_deal_owner_id",
                "latest_deal_stage",
            ]
        )

    associated = fact_df[
        fact_df["contact_id"].notna()
        & fact_df["deal_id"].notna()
        & (fact_df.get("fact_record_type", "") == "associated")
    ].copy()
    if associated.empty:
        return pd.DataFrame(columns=["contact_id"])

    rows: list[dict[str, Any]] = []
    for contact_id, group in associated.groupby("contact_id", dropna=True):
        sorted_group = group.sort_values("deal_created_at", na_position="first")
        deal_owner = next((_clean_text(value) for value in group["deal_owner_id"] if _clean_text(value)), "")
        latest_stage = next(
            (
                _clean_text(value)
                for value in reversed(sorted_group["deal_stage"].tolist())
                if _clean_text(value)
            ),
            "",
        )
        rows.append(
            {
                "contact_id": contact_id,
                "has_open_deal": bool((~group["is_won"].fillna(False) & ~group["is_lost"].fillna(False)).any()),
                "has_won_deal": bool(group["is_won"].fillna(False).any()),
                "has_lost_deal": bool(group["is_lost"].fillna(False).any()),
                "fallback_deal_owner_id": deal_owner or pd.NA,
                "latest_deal_stage": latest_stage or pd.NA,
            }
        )
    return pd.DataFrame(rows)


def enrich_contacts_with_deal_summary(
    contacts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    owners_df: pd.DataFrame,
) -> pd.DataFrame:
    contacts = contacts_df.copy()
    owners = _owner_lookup(owners_df)
    if not summary_df.empty and "contact_id" in summary_df.columns:
        contacts = contacts.merge(summary_df, on="contact_id", how="left", suffixes=("", "_summary"))

    for column in ("has_open_deal", "has_won_deal", "has_lost_deal"):
        summary_col = f"{column}_summary"
        if summary_col in contacts.columns:
            contacts[column] = contacts[summary_col].combine_first(contacts[column])
            contacts = contacts.drop(columns=[summary_col])
        contacts[column] = contacts[column].fillna(False).astype(bool)

    if "latest_deal_stage_summary" in contacts.columns:
        contacts["latest_deal_stage"] = contacts["latest_deal_stage_summary"].combine_first(
            contacts["latest_deal_stage"]
        )
        contacts = contacts.drop(columns=["latest_deal_stage_summary"])

    if "fallback_deal_owner_id" not in contacts.columns:
        contacts["fallback_deal_owner_id"] = pd.NA

    missing_contact_owner = contacts["contact_owner_id"].map(lambda value: not bool(_clean_text(value)))
    has_deal_owner = contacts["fallback_deal_owner_id"].map(lambda value: bool(_clean_text(value)))
    fallback_mask = missing_contact_owner & has_deal_owner
    contacts.loc[fallback_mask, "salesman_id"] = contacts.loc[fallback_mask, "fallback_deal_owner_id"]
    contacts["salesman_name"] = _map_owner_name(contacts["salesman_id"], owners)
    contacts["contact_attribution_type"] = "contact_owner"
    contacts.loc[fallback_mask, "contact_attribution_type"] = "fallback_to_deal_owner"
    contacts.loc[
        contacts["salesman_id"].map(lambda value: not bool(_clean_text(value))),
        "contact_attribution_type",
    ] = "missing_owner"
    return contacts


def _dict_by_id(df: pd.DataFrame, id_column: str) -> dict[str, dict[str, Any]]:
    if df.empty or id_column not in df.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        record_id = _clean_text(row.get(id_column))
        if record_id:
            output[record_id] = row
    return output


def build_lead_deal_fact(
    contacts_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    associations_df: pd.DataFrame,
    owners_df: pd.DataFrame,
) -> pd.DataFrame:
    contacts_by_id = _dict_by_id(contacts_df, "contact_id")
    deals_by_id = _dict_by_id(deals_df, "deal_id")
    assoc = _association_frame(associations_df)
    owner_names = _owner_lookup(owners_df)
    rows: list[dict[str, Any]] = []

    for assoc_row in assoc.to_dict("records"):
        contact_id = _clean_text(assoc_row.get("contact_id"))
        deal_id = _clean_text(assoc_row.get("deal_id"))
        row: dict[str, Any] = {"fact_record_type": "associated"}
        row.update(contacts_by_id.get(contact_id, {"contact_id": contact_id}))
        row.update(deals_by_id.get(deal_id, {"deal_id": deal_id}))
        row["association_type"] = assoc_row.get("association_type")
        rows.append(row)

    associated_contact_ids = set(assoc["contact_id"].dropna().astype(str))
    associated_deal_ids = set(assoc["deal_id"].dropna().astype(str))

    for contact_id, contact in contacts_by_id.items():
        if contact_id not in associated_contact_ids:
            row = {"fact_record_type": "contact_without_deal"}
            row.update(contact)
            rows.append(row)

    for deal_id, deal in deals_by_id.items():
        if deal_id not in associated_deal_ids:
            row = {"fact_record_type": "deal_without_contact"}
            row.update(deal)
            rows.append(row)

    fact = pd.DataFrame(rows)
    for column in CONTACT_COLUMNS + DEAL_COLUMNS:
        if column not in fact.columns:
            fact[column] = pd.NA

    if fact.empty:
        return fact

    for column in ("contact_id", "deal_id", "contact_owner_id", "deal_owner_id"):
        fact[column] = _as_clean_id(fact[column])

    def attribution(row: pd.Series) -> tuple[str | None, str]:
        contact_owner = _clean_text(row.get("contact_owner_id"))
        deal_owner = _clean_text(row.get("deal_owner_id"))
        if contact_owner:
            return contact_owner, "contact_owner"
        if deal_owner:
            return deal_owner, "fallback_to_deal_owner"
        return None, "missing_owner"

    attribution_values = fact.apply(attribution, axis=1)
    fact["salesman_id"] = attribution_values.map(lambda item: item[0])
    fact["attribution_type"] = attribution_values.map(lambda item: item[1])
    fact["salesman_name"] = fact["salesman_id"].map(
        lambda owner_id: owner_names.get(_clean_text(owner_id), "Unassigned")
    )

    deal_contact_counts = assoc.groupby("deal_id")["contact_id"].nunique().to_dict() if not assoc.empty else {}
    fact["deal_contact_count"] = fact["deal_id"].map(lambda deal_id: deal_contact_counts.get(_clean_text(deal_id), 0))

    seen_deals: set[str] = set()
    revenue_countable: list[bool] = []
    for deal_id in fact["deal_id"]:
        cleaned = _clean_text(deal_id)
        if not cleaned:
            revenue_countable.append(False)
        elif cleaned not in seen_deals:
            revenue_countable.append(True)
            seen_deals.add(cleaned)
        else:
            revenue_countable.append(False)
    fact["revenue_countable"] = revenue_countable
    fact["deal_countable"] = revenue_countable
    fact["revenue"] = pd.to_numeric(fact["revenue"], errors="coerce").fillna(0)
    fact["is_won"] = fact["is_won"].fillna(False).astype(bool)
    fact["is_lost"] = fact["is_lost"].fillna(False).astype(bool)
    fact["revenue_attributed"] = fact.apply(
        lambda row: row["revenue"] if bool(row["revenue_countable"]) and bool(row["is_won"]) else 0,
        axis=1,
    )
    fact["won_deal_count"] = fact.apply(
        lambda row: 1 if bool(row["deal_countable"]) and bool(row["is_won"]) else 0,
        axis=1,
    )
    countable_won = fact["revenue_countable"].fillna(False).astype(bool) & fact["is_won"].fillna(False).astype(bool)
    contact_tuition = pd.to_numeric(fact["program_total_tuition"], errors="coerce").fillna(0).clip(lower=0)
    deal_revenue = pd.to_numeric(fact["revenue"], errors="coerce").fillna(0).clip(lower=0)
    published_tuition = pd.to_numeric(fact["estimated_total_tuition"], errors="coerce").fillna(0).clip(lower=0)
    program_revenue_base = contact_tuition.where(
        contact_tuition > 0,
        deal_revenue.where(deal_revenue > 0, published_tuition),
    )
    fact["program_revenue_input"] = program_revenue_base.where(countable_won, 0)
    fact["program_revenue_source"] = "not_countable_or_not_won"
    fact.loc[countable_won & (contact_tuition > 0), "program_revenue_source"] = "contact_program_total_tuition"
    fact.loc[
        countable_won & (contact_tuition <= 0) & (deal_revenue > 0),
        "program_revenue_source",
    ] = "deal_revenue"
    fact.loc[
        countable_won & (contact_tuition <= 0) & (deal_revenue <= 0) & (published_tuition > 0),
        "program_revenue_source",
    ] = "calmu_published_tuition_config"
    fact.loc[
        countable_won & (contact_tuition <= 0) & (deal_revenue <= 0) & (published_tuition <= 0),
        "program_revenue_source",
    ] = "missing_revenue"
    fact = add_program_revenue_fields(fact, total_revenue_column="program_revenue_input")
    fact["unclear_revenue_attribution"] = (
        (fact["deal_contact_count"].fillna(0) > 1) | (fact["attribution_type"] == "missing_owner")
    )

    close_date = parse_datetime_series(fact["close_date"])
    contact_created = parse_datetime_series(fact["contact_created_at"])
    deal_created = parse_datetime_series(fact["deal_created_at"])
    start_date = contact_created.combine_first(deal_created)
    fact["days_to_close"] = (close_date - start_date).dt.days
    fact.loc[fact["days_to_close"] < 0, "days_to_close"] = pd.NA
    return fact


def build_student_journey_fact(
    contacts_df: pd.DataFrame,
    fact_df: pd.DataFrame,
    activity_events: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "contact_id",
        "student_name",
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone",
        "salesman_name",
        "owner_assigned_date",
        "contact_created_at",
        "first_activity_date",
        "last_activity_date",
        "next_activity_date",
        "number_of_sales_activities",
        "time_to_first_engagement",
        "application_date",
        "enrollment_date",
        "program",
        "degree_level",
        "cohort",
        "start_term",
        "campus",
        "modality",
        "vendor",
        "campaign",
        "source",
        "source_group",
        "paid_lead_flag",
        "lead_status",
        "final_lead_status",
        "enrollment_status",
        "lead_temperature",
        "deal_created_at",
        "deal_stage",
        "close_date",
        "actual_revenue",
        "estimated_enrolled_revenue",
        "potential_revenue",
    ]
    if contacts_df.empty:
        return pd.DataFrame(columns=columns)

    output = contacts_df.drop_duplicates(subset=["contact_id"], keep="first").copy()
    output["actual_revenue"] = 0.0
    if not fact_df.empty and "contact_id" in fact_df.columns:
        fact = fact_df.copy()
        fact["contact_id"] = fact["contact_id"].fillna("").astype(str)
        revenue = pd.to_numeric(fact.get("revenue_attributed", pd.Series(0, index=fact.index)), errors="coerce").fillna(0)
        output["actual_revenue"] = output["contact_id"].astype(str).map(revenue.groupby(fact["contact_id"]).sum()).fillna(0)
        for source_column, output_column in (
            ("deal_created_at", "deal_created_at"),
            ("deal_stage", "deal_stage"),
            ("close_date", "close_date"),
        ):
            if source_column in fact.columns:
                values = fact[fact["contact_id"].ne("")].groupby("contact_id")[source_column].first()
                output[output_column] = output["contact_id"].astype(str).map(values)

    if not activity_events.empty and "contact_id" in activity_events.columns:
        events = activity_events.copy()
        events["contact_id"] = events["contact_id"].fillna("").astype(str)
        dates = parse_datetime_series(events.get("event_date", pd.Series(index=events.index, dtype=object)))
        activity = events.assign(_event_date=dates).dropna(subset=["_event_date"])
        if not activity.empty:
            grouped = activity.groupby("contact_id")["_event_date"]
            output["first_activity_date"] = output.get("first_activity_date", pd.Series(pd.NaT, index=output.index)).combine_first(
                output["contact_id"].astype(str).map(grouped.min())
            )
            output["last_activity_date"] = output.get("last_activity_date", pd.Series(pd.NaT, index=output.index)).combine_first(
                output["contact_id"].astype(str).map(grouped.max())
            )
            output["detailed_activity_count"] = output["contact_id"].astype(str).map(grouped.size()).fillna(0).astype(int)
        else:
            output["detailed_activity_count"] = 0
    else:
        output["detailed_activity_count"] = 0

    output["estimated_enrolled_revenue"] = pd.to_numeric(
        output.get("enrolled_revenue", pd.Series(0, index=output.index)),
        errors="coerce",
    ).fillna(0)
    if "campaign" not in output.columns:
        output["campaign"] = output.get("utm_campaign", pd.Series(pd.NA, index=output.index))
    if "source" not in output.columns:
        output["source"] = output.get("source_text", pd.Series(pd.NA, index=output.index))
    for column in columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output[columns]


def build_cohort_fact(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from cohort_analysis import cohort_calculated_fields, prepare_cohort_contacts
    except Exception:
        return pd.DataFrame()
    prepared = prepare_cohort_contacts(contacts_df, fact_df)
    return cohort_calculated_fields(prepared)


def build_vendor_fact(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> pd.DataFrame:
    from metrics import calculate_paid_lead_metrics

    return calculate_paid_lead_metrics(contacts_df, fact_df).get("paid_vendor_performance", pd.DataFrame())


def build_salesman_revenue_fact(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> pd.DataFrame:
    from metrics import calculate_salesman_metrics

    return calculate_salesman_metrics(contacts_df, fact_df)


def process_hubspot_data(
    raw_contacts: pd.DataFrame,
    raw_deals: pd.DataFrame,
    owners_df: pd.DataFrame,
    associations_df: pd.DataFrame,
    mapping: dict[str, Any],
    raw_activities: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    contacts_base = build_contacts_clean(raw_contacts, owners_df, mapping)
    deals_clean = build_deals_clean(raw_deals, mapping)
    activity_events = build_activity_events(raw_activities if raw_activities is not None else pd.DataFrame(), owners_df)
    initial_fact = build_lead_deal_fact(contacts_base, deals_clean, associations_df, owners_df)
    deal_summary = summarize_contact_deals(initial_fact)
    contacts_enriched = enrich_contacts_with_deal_summary(contacts_base, deal_summary, owners_df)
    contacts_clean = score_leads(contacts_enriched)
    lead_deal_fact = build_lead_deal_fact(contacts_clean, deals_clean, associations_df, owners_df)
    student_journey_fact = build_student_journey_fact(contacts_clean, lead_deal_fact, activity_events)
    cohort_fact = build_cohort_fact(contacts_clean, lead_deal_fact)
    vendor_fact = build_vendor_fact(contacts_clean, lead_deal_fact)
    salesman_revenue_fact = build_salesman_revenue_fact(contacts_clean, lead_deal_fact)
    return {
        "contacts_clean": contacts_clean,
        "deals_clean": deals_clean,
        "lead_deal_fact": lead_deal_fact,
        "student_journey_fact": student_journey_fact,
        "cohort_fact": cohort_fact,
        "vendor_fact": vendor_fact,
        "salesman_revenue_fact": salesman_revenue_fact,
        "activity_events": activity_events,
    }
