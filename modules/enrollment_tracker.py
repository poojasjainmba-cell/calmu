from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .source_mapping import add_source_columns, clean_text, standardize_columns


TRACKER_FILE = Path("user_files/03-Summer2tracker-1-.xlsx")
PUBLIC_BASELINE_DIR = Path("public_data/baseline")


@dataclass
class EnrollmentTrackerData:
    enrollments: pd.DataFrame
    roundup_summary: pd.DataFrame
    roundup_allocations: pd.DataFrame
    workbook_term: str
    notes: list[str]


def _term_label(value: object) -> str:
    if pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.notna(timestamp):
        return timestamp.strftime("%Y-%m-%d")
    return clean_text(value)


def _load_enrollments(path: Path) -> tuple[pd.DataFrame, str]:
    banner = pd.read_excel(path, sheet_name="Enrollments", header=None, nrows=1)
    workbook_term = clean_text(banner.iloc[0, 1]) if banner.shape[1] > 1 else ""
    raw = pd.read_excel(path, sheet_name="Enrollments", header=1)
    out = standardize_columns(raw)
    rename = {
        "unnamed_0": "row_number",
        "student": "student",
        "udr": "udr",
        "program": "program",
        "modality": "modality",
        "enrolled_date": "enrolled_date",
        "student_type": "student_type",
        "payment": "payment_funding",
        "new_roll": "new_roll",
        "term": "term",
        "source": "source",
        "days_to_enroll": "days_to_enroll",
        "rev": "revenue",
        "revenue": "revenue",
        "notes": "notes",
    }
    out = out.rename(columns={column: rename.get(column, column) for column in out.columns})
    for column in [
        "student",
        "udr",
        "program",
        "modality",
        "student_type",
        "payment_funding",
        "new_roll",
        "source",
        "notes",
    ]:
        if column not in out.columns:
            out[column] = pd.NA
    for column in ["enrolled_date", "term"]:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    out["days_to_enroll"] = pd.to_numeric(out.get("days_to_enroll", pd.NA), errors="coerce")
    out["revenue"] = pd.to_numeric(out.get("revenue", pd.NA), errors="coerce")
    out["student"] = out["student"].fillna("").astype(str).str.strip()
    out = out[out["student"].ne("")].copy()
    out["term_label"] = out["term"].map(_term_label)
    out["workbook_term"] = workbook_term
    out["data_origin"] = "Uploaded enrollment tracker"
    out = add_source_columns(out)
    return out, workbook_term


def _summary_from_roundup(raw: pd.DataFrame, sheet: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        metric = clean_text(row.get(1, ""))
        if metric.lower() in {"budget", "enrolled", "starts", "start%"}:
            rows.append({"sheet": sheet, "metric": metric, "value": row.get(2)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["numeric_value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def _allocations_from_roundup(raw: pd.DataFrame, sheet: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    header_idx: int | None = None
    total_col: int | None = None
    for idx, row in raw.iterrows():
        values = [clean_text(value).lower() for value in row.tolist()]
        budget_cols = [i for i, value in enumerate(values) if value == "budget"]
        if budget_cols:
            header_idx = idx
            total_col = budget_cols[-1]
            break
    if header_idx is None or total_col is None or total_col == 0:
        return pd.DataFrame()
    name_col = total_col - 1
    headers = raw.iloc[header_idx].to_dict()
    term_cols = [
        column
        for column, value in headers.items()
        if column > total_col and clean_text(value)
    ]
    for idx in range(header_idx + 1, len(raw)):
        label = clean_text(raw.iat[idx, name_col])
        total = pd.to_numeric(raw.iat[idx, total_col], errors="coerce")
        if not label or pd.isna(total):
            continue
        rows.append(
            {
                "sheet": sheet,
                "budget_dimension": "UDR",
                "budget_name": label,
                "source": label,
                "planned_budget": float(total),
                "term": "Total",
                "term_budget": float(total),
            }
        )
        for column in term_cols:
            term_value = pd.to_numeric(raw.iat[idx, column], errors="coerce")
            if pd.notna(term_value):
                rows.append(
                    {
                        "sheet": sheet,
                        "budget_dimension": "UDR",
                        "budget_name": label,
                        "source": label,
                        "planned_budget": float(total),
                        "term": clean_text(headers[column]),
                        "term_budget": float(term_value),
                    }
                )
    return pd.DataFrame(rows)


def load_enrollment_tracker(path: Path = TRACKER_FILE) -> EnrollmentTrackerData:
    if not path.exists():
        enrollments_path = PUBLIC_BASELINE_DIR / "enrollments.csv"
        if enrollments_path.exists():
            enrollments = pd.read_csv(enrollments_path)
            for column in ["enrolled_date", "term"]:
                if column in enrollments.columns:
                    enrollments[column] = pd.to_datetime(enrollments[column], errors="coerce")
            for column in ["days_to_enroll", "revenue"]:
                if column in enrollments.columns:
                    enrollments[column] = pd.to_numeric(enrollments[column], errors="coerce")
            summary_path = PUBLIC_BASELINE_DIR / "tracker_roundup_summary.csv"
            allocations_path = PUBLIC_BASELINE_DIR / "tracker_roundup_allocations.csv"
            summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
            allocations = pd.read_csv(allocations_path) if allocations_path.exists() else pd.DataFrame()
            workbook_term = ""
            metadata_path = PUBLIC_BASELINE_DIR / "metadata.json"
            if metadata_path.exists():
                try:
                    import json

                    workbook_term = json.loads(metadata_path.read_text(encoding="utf-8")).get("workbook_term", "")
                except json.JSONDecodeError:
                    workbook_term = ""
            return EnrollmentTrackerData(
                enrollments,
                summary,
                allocations,
                workbook_term,
                [f"Private enrollment tracker not found; using sanitized public baseline from {PUBLIC_BASELINE_DIR}."],
            )
        return EnrollmentTrackerData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "", [f"Missing tracker: {path}"])

    xl = pd.ExcelFile(path)
    notes = [f"Enrollment tracker sheets inspected: {', '.join(xl.sheet_names)}"]
    enrollments, workbook_term = _load_enrollments(path) if "Enrollments" in xl.sheet_names else (pd.DataFrame(), "")
    if "2026Roundup" in xl.sheet_names:
        raw_roundup = pd.read_excel(path, sheet_name="2026Roundup", header=None)
        summary = _summary_from_roundup(raw_roundup, "2026Roundup")
        allocations = _allocations_from_roundup(raw_roundup, "2026Roundup")
    else:
        summary = pd.DataFrame()
        allocations = pd.DataFrame()
        notes.append("2026Roundup sheet was not found.")
    return EnrollmentTrackerData(enrollments, summary, allocations, workbook_term, notes)
