from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .source_mapping import UNKNOWN_SOURCE, source_set


def _add(rows: list[dict[str, Any]], check: str, count: int, severity: str, details: str = "") -> None:
    status = "Pass" if count == 0 and severity != "Info" else ("Info" if severity == "Info" else "Review")
    rows.append(
        {
            "check": check,
            "status": status,
            "severity": severity,
            "count": int(count) if pd.notna(count) else 0,
            "details": details,
        }
    )


def _missing_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].fillna("").astype(str).str.strip().eq("").sum())


def _invalid_datetime_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    original = df[column]
    parsed = pd.to_datetime(original, errors="coerce")
    present = original.fillna("").astype(str).str.strip().ne("")
    return int((present & parsed.isna()).sum())


def run_qa_checks(
    leads: pd.DataFrame,
    paid_leads: pd.DataFrame,
    udr_leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    budget_allocations: pd.DataFrame,
    pivot_diffs: pd.DataFrame,
    hubspot_state: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not leads.empty:
        duplicate_records = 0
        if "record_id" in leads.columns:
            ids = leads["record_id"].fillna("").astype(str).str.strip()
            duplicate_records = int(ids[ids.ne("")].duplicated().sum())
        duplicate_emails = 0
        if "email" in leads.columns:
            emails = leads["email"].fillna("").astype(str).str.strip().str.lower()
            duplicate_emails = int(emails[emails.ne("")].duplicated().sum())
        _add(rows, "Duplicate Record IDs", duplicate_records, "High")
        _add(rows, "Duplicate emails", duplicate_emails, "Medium")
        _add(rows, "Missing email", _missing_count(leads, "email"), "Medium")
        _add(rows, "Missing contact owner/UDR", _missing_count(leads, "contact_owner"), "High")
        _add(rows, "Missing lead status", _missing_count(leads, "lead_status"), "Medium")
        _add(rows, "Missing lifecycle stage", _missing_count(leads, "lifecycle_stage"), "Medium")
        _add(rows, "Missing create date", _missing_count(leads, "create_date"), "Medium")
        _add(rows, "Missing Degree/Program", _missing_count(leads, "degree"), "Low")
        _add(
            rows,
            "Missing source",
            int(leads.get("normalized_source", pd.Series(dtype=str)).fillna(UNKNOWN_SOURCE).eq(UNKNOWN_SOURCE).sum()),
            "High",
        )

    if not enrollments.empty:
        _add(rows, "Missing enrollment term", _missing_count(enrollments, "term"), "High")
        _add(
            rows,
            "Missing enrolled date",
            _missing_count(enrollments, "enrolled_date"),
            "High",
        )
        _add(rows, "Missing source", _missing_count(enrollments, "source"), "Medium", "Enrollment tracker source field.")
        _add(rows, "Missing revenue", int(pd.to_numeric(enrollments.get("revenue"), errors="coerce").isna().sum()), "Medium")
        _add(rows, "Missing payment/funding", _missing_count(enrollments, "payment_funding"), "Low")
        _add(rows, "Invalid dates", _invalid_datetime_count(enrollments, "enrolled_date"), "Medium")
        days = pd.to_numeric(enrollments.get("days_to_enroll"), errors="coerce")
        _add(rows, "Negative days to enroll", int((days < 0).sum()), "High")

    if not leads.empty and not enrollments.empty:
        crm_enrolled = int(leads.get("lifecycle_stage", pd.Series(dtype=str)).fillna("").astype(str).str.lower().eq("enrolled").sum())
        actual = int(len(enrollments))
        _add(
            rows,
            "HubSpot CRM enrolled vs tracker actual enrolled mismatch",
            abs(crm_enrolled - actual),
            "Medium",
            f"CRM enrolled={crm_enrolled:,}; tracker actual={actual:,}.",
        )

    budget_sources = source_set(budget_allocations, "source") if not budget_allocations.empty else set()
    hubspot_sources = source_set(leads, "normalized_source")
    tracker_sources = source_set(enrollments, "normalized_source")
    _add(
        rows,
        "Budget source not found in HubSpot",
        len(budget_sources - hubspot_sources),
        "Medium",
        ", ".join(sorted(list(budget_sources - hubspot_sources))[:12]),
    )
    _add(
        rows,
        "HubSpot source not found in budget",
        len(hubspot_sources - budget_sources),
        "Medium",
        ", ".join(sorted(list(hubspot_sources - budget_sources))[:12]),
    )
    _add(
        rows,
        "Tracker source not found in budget",
        len(tracker_sources - budget_sources),
        "Medium",
        ", ".join(sorted(list(tracker_sources - budget_sources))[:12]),
    )

    if not pivot_diffs.empty:
        mismatch_count = int((pd.to_numeric(pivot_diffs["difference"], errors="coerce").fillna(0) != 0).sum())
        _add(rows, "Uploaded pivot totals vs recalculated raw totals", mismatch_count, "Medium")

    if hubspot_state.get("error"):
        _add(rows, "HubSpot API fetch failure", 1, "High", str(hubspot_state["error"])[:220])
    elif not hubspot_state.get("token_present"):
        _add(rows, "HubSpot API fetch failure", 1, "Info", "No token configured; static uploaded baseline is in use.")
    else:
        _add(rows, "HubSpot API fetch failure", 0, "High")

    fetched_at = hubspot_state.get("fetched_at")
    if fetched_at:
        try:
            age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        except Exception:
            age_hours = 0
        _add(rows, "Stale data warning", int(age_hours > 24), "Medium", f"Last live refresh age: {age_hours:.1f} hours.")
    else:
        _add(rows, "Stale data warning", 1, "Info", "Live HubSpot data has not been refreshed in this session.")

    return pd.DataFrame(rows)
