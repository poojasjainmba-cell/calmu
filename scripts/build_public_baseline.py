from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.budget_loader import load_budget
from modules.data_loader import load_uploaded_lead_data
from modules.enrollment_tracker import load_enrollment_tracker


OUTPUT_DIR = Path("public_data/baseline")
PII_COLUMNS = {
    "first_name",
    "last_name",
    "email",
    "phone_number",
    "record_id",
    "contact_owner_id",
    "id",
    "hs_object_id",
    "student",
    "notes",
    "sender",
}
OWNER_COLUMNS = {"contact_owner", "udr"}
SOURCE_COLUMNS = {
    "source",
    "normalized_source",
    "paid_lead_list",
    "organic_lead_list",
    "event_attended",
    "campaign",
    "utm_source",
    "utm_campaign",
    "utm_medium",
}
PUBLIC_KEEP_VALUES = {"", "unmapped", "unknown", "missing", "nan", "none", "(blank)"}


def _clean_label(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _label_map(columns: set[str], prefix: str, *frames: pd.DataFrame) -> dict[str, str]:
    values: set[str] = set()
    for frame in frames:
        for column in columns:
            if column in frame.columns:
                values.update(_clean_label(value) for value in frame[column].dropna())
    values = {value for value in values if value.lower() not in PUBLIC_KEEP_VALUES}
    return {value: f"{prefix} {idx:02d}" for idx, value in enumerate(sorted(values), start=1)}


def _pseudonymize(value: object, *mappings: dict[str, str]) -> object:
    text = _clean_label(value)
    for mapping in mappings:
        if text in mapping:
            return mapping[text]
    return value


def _sanitize_frame(
    df: pd.DataFrame,
    owner_mapping: dict[str, str],
    source_mapping: dict[str, str],
    add_student_placeholder: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if add_student_placeholder:
        out["student"] = [f"Student {idx:04d}" for idx in range(1, len(out) + 1)]
    out = out.drop(columns=[column for column in PII_COLUMNS if column in out.columns and column != "student"], errors="ignore")
    for column in OWNER_COLUMNS | {"budget_name", "normalized_budget_name"}:
        if column in out.columns:
            out[column] = out[column].map(lambda value: _pseudonymize(value, owner_mapping, source_mapping))
    for column in SOURCE_COLUMNS | {"row_label"}:
        if column in out.columns:
            out[column] = out[column].map(lambda value: _pseudonymize(value, owner_mapping, source_mapping))
    return out


def _write_csv(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / name, index=False)


def main() -> None:
    uploaded = load_uploaded_lead_data()
    tracker = load_enrollment_tracker()
    budget = load_budget()

    owner_mapping = _label_map(
        OWNER_COLUMNS | {"budget_name", "normalized_budget_name"},
        "UDR",
        uploaded.paid_leads,
        uploaded.udr_leads,
        tracker.enrollments,
        tracker.roundup_allocations,
        budget.allocations,
        budget.term_allocations,
    )
    source_mapping = _label_map(
        SOURCE_COLUMNS | {"row_label"},
        "Source",
        uploaded.paid_leads,
        uploaded.udr_leads,
        tracker.enrollments,
        tracker.roundup_allocations,
        budget.allocations,
        budget.term_allocations,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(_sanitize_frame(uploaded.paid_leads, owner_mapping, source_mapping), "paid_leads.csv")
    _write_csv(_sanitize_frame(uploaded.udr_leads, owner_mapping, source_mapping), "udr_leads.csv")
    _write_csv(
        _sanitize_frame(tracker.enrollments, owner_mapping, source_mapping, add_student_placeholder=True),
        "enrollments.csv",
    )
    _write_csv(_sanitize_frame(tracker.roundup_summary, owner_mapping, source_mapping), "tracker_roundup_summary.csv")
    _write_csv(_sanitize_frame(tracker.roundup_allocations, owner_mapping, source_mapping), "tracker_roundup_allocations.csv")
    _write_csv(_sanitize_frame(budget.summary, owner_mapping, source_mapping), "budget_summary.csv")
    _write_csv(_sanitize_frame(budget.allocations, owner_mapping, source_mapping), "budget_allocations.csv")
    _write_csv(_sanitize_frame(budget.term_allocations, owner_mapping, source_mapping), "budget_term_allocations.csv")

    for prefix, pivots in [("paid", uploaded.paid_pivots), ("udr", uploaded.udr_pivots)]:
        for name, frame in pivots.items():
            _write_csv(_sanitize_frame(frame, owner_mapping, source_mapping), f"{prefix}_pivot_{name}.csv")

    metadata = {
        "privacy": "Names, emails, phone numbers, raw Record IDs, notes, UDR/contact-owner labels, and source/list labels are excluded or pseudonymized.",
        "paid_rows": len(uploaded.paid_leads),
        "udr_rows": len(uploaded.udr_leads),
        "enrollment_rows": len(tracker.enrollments),
        "tracker_revenue": float(pd.to_numeric(tracker.enrollments.get("revenue"), errors="coerce").fillna(0).sum()),
        "budget_allocation_rows": len(budget.allocations),
        "workbook_term": tracker.workbook_term,
        "email_context": {
            "subject": uploaded.email_context.subject,
            "sent_at": uploaded.email_context.sent_at,
            "image_count": uploaded.email_context.image_count,
            "data_lines": uploaded.email_context.data_lines,
            "notes": uploaded.email_context.notes,
        },
        "load_notes": [
            "Sanitized public baseline generated from local private uploaded files.",
            *uploaded.load_notes,
            *tracker.notes,
            *budget.notes,
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ["paid_rows", "udr_rows", "enrollment_rows", "tracker_revenue"]}))


if __name__ == "__main__":
    main()
