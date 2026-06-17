from __future__ import annotations

import html
import io
import json
import re
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import pandas as pd

from .source_mapping import add_source_columns, clean_text, standardize_columns


USER_FILES_DIR = Path("user_files")
PUBLIC_BASELINE_DIR = Path("public_data/baseline")
PAID_LEADS_FILE = USER_FILES_DIR / "01-PaidleadsJune11-1-.xlsx"
EMAIL_FILE = USER_FILES_DIR / "02-Summer-2-Enrollment-Update-Week-6-June-8-12-1-.eml"
TRACKER_FILE = USER_FILES_DIR / "03-Summer2tracker-1-.xlsx"
UDR_FILE = USER_FILES_DIR / "04-UDRConversionsJune11-1-.xlsx"
BUDGET_FILE = USER_FILES_DIR / "01-2026Budget.xlsx"


LEAD_RENAME = {
    "record_id": "record_id",
    "hs_object_id": "record_id",
    "first_name": "first_name",
    "firstname": "first_name",
    "last_name": "last_name",
    "lastname": "last_name",
    "email": "email",
    "lead_status": "lead_status",
    "hs_lead_status": "lead_status",
    "lifecycle_stage": "lifecycle_stage",
    "lifecyclestage": "lifecycle_stage",
    "phone_number": "phone_number",
    "phone": "phone_number",
    "contact_owner": "contact_owner",
    "hubspot_owner_id": "contact_owner_id",
    "last_activity_date": "last_activity_date",
    "hs_last_sales_activity_timestamp": "last_activity_date",
    "notes_last_updated": "last_activity_date",
    "marketing_contact_status": "marketing_contact_status",
    "create_date": "create_date",
    "createdate": "create_date",
    "student_type": "student_type",
    "athletics": "athletics",
    "event_attended": "event_attended",
    "campus_location": "campus_location",
    "paid_lead_list": "paid_lead_list",
    "organic_lead_list": "organic_lead_list",
    "degree": "degree",
    "program": "degree",
    "utm_source": "utm_source",
    "utm_campaign": "utm_campaign",
    "utm_medium": "utm_medium",
    "campaign": "campaign",
    "hs_latest_source": "hs_latest_source",
    "hs_latest_source_data_1": "hs_latest_source_data_1",
    "hs_latest_source_data_2": "hs_latest_source_data_2",
    "hs_analytics_source": "hs_analytics_source",
    "hs_analytics_source_data_1": "hs_analytics_source_data_1",
    "hs_analytics_source_data_2": "hs_analytics_source_data_2",
}


@dataclass
class EmailContext:
    subject: str
    sent_at: str
    sender: str
    text: str
    data_lines: list[str]
    image_count: int
    ocr_text: str
    notes: list[str]


@dataclass
class UploadedLeadData:
    paid_leads: pd.DataFrame
    udr_leads: pd.DataFrame
    paid_pivots: dict[str, pd.DataFrame]
    udr_pivots: dict[str, pd.DataFrame]
    email_context: EmailContext
    load_notes: list[str]


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    data: dict[str, pd.Series] = {}
    ordered_columns = list(dict.fromkeys(df.columns))
    for column in ordered_columns:
        subset = df.loc[:, df.columns == column]
        if subset.shape[1] == 1:
            data[column] = subset.iloc[:, 0]
            continue
        normalized = subset.replace("", pd.NA)
        data[column] = normalized.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(data, index=df.index)


def standardize_leads(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = standardize_columns(df)
    rename = {column: LEAD_RENAME[column] for column in out.columns if column in LEAD_RENAME}
    out = out.rename(columns=rename)
    out = _coalesce_duplicate_columns(out)
    out = _ensure_columns(
        out,
        [
            "record_id",
            "first_name",
            "last_name",
            "email",
            "lead_status",
            "lifecycle_stage",
            "phone_number",
            "contact_owner",
            "last_activity_date",
            "create_date",
            "student_type",
            "event_attended",
            "campus_location",
            "paid_lead_list",
            "organic_lead_list",
            "degree",
        ],
    )
    out["record_id"] = out["record_id"].fillna("").astype(str).str.strip()
    out["email"] = out["email"].fillna("").astype(str).str.strip().str.lower()
    out["lead_status"] = out["lead_status"].fillna("").astype(str).str.strip()
    out["lifecycle_stage"] = out["lifecycle_stage"].fillna("").astype(str).str.strip()
    out["contact_owner"] = out["contact_owner"].fillna("").astype(str).str.strip()
    out["create_date"] = pd.to_datetime(out["create_date"], errors="coerce")
    out["last_activity_date"] = pd.to_datetime(out["last_activity_date"], errors="coerce")
    out["data_origin"] = origin
    out = add_source_columns(out)
    return out


def parse_pivot_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_idx: int | None = None
    for idx, row in raw.iterrows():
        values = [clean_text(value).lower() for value in row.tolist()]
        if "row labels" in values and "grand total" in values:
            header_idx = idx
            break
    if header_idx is None:
        return pd.DataFrame()

    headers = [clean_text(value) or f"column_{i}" for i, value in enumerate(raw.iloc[header_idx].tolist())]
    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = headers
    first_col = headers[0]
    data[first_col] = data[first_col].map(clean_text)
    data = data[data[first_col].ne("")]
    data = data[~data[first_col].str.lower().isin(["grand total", "(blank)"])]
    data = standardize_columns(data)
    if "row_labels" in data.columns:
        data = data.rename(columns={"row_labels": "row_label"})
    return data


def _baseline_csv(name: str) -> Path:
    return PUBLIC_BASELINE_DIR / name


def _read_baseline_csv(name: str) -> pd.DataFrame:
    path = _baseline_csv(name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_baseline_metadata() -> dict[str, Any]:
    path = PUBLIC_BASELINE_DIR / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_baseline_pivots(prefix: str) -> dict[str, pd.DataFrame]:
    pivots: dict[str, pd.DataFrame] = {}
    for path in PUBLIC_BASELINE_DIR.glob(f"{prefix}_pivot_*.csv"):
        name = path.stem.replace(f"{prefix}_pivot_", "")
        pivots[name] = pd.read_csv(path)
    return pivots


def load_paid_leads(path: Path = PAID_LEADS_FILE) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    notes: list[str] = []
    if not path.exists():
        baseline = _read_baseline_csv("paid_leads.csv")
        if not baseline.empty:
            return (
                standardize_leads(baseline, "Sanitized public paid leads baseline"),
                _load_baseline_pivots("paid"),
                [f"Private paid leads workbook not found; using sanitized public baseline from {PUBLIC_BASELINE_DIR}."],
            )
        return pd.DataFrame(), {}, [f"Missing paid leads workbook: {path}"]
    xl = pd.ExcelFile(path)
    notes.append(f"Paid leads workbook sheets inspected: {', '.join(xl.sheet_names)}")
    raw = pd.read_excel(path, sheet_name="Paid Leads") if "Paid Leads" in xl.sheet_names else pd.DataFrame()
    pivots = {
        name: parse_pivot_sheet(path, name)
        for name in ["PLS", "PLC", "OLS", "OLC"]
        if name in xl.sheet_names
    }
    return standardize_leads(raw, "Uploaded paid leads workbook"), pivots, notes


def load_udr_conversions(path: Path = UDR_FILE) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    notes: list[str] = []
    if not path.exists():
        baseline = _read_baseline_csv("udr_leads.csv")
        if not baseline.empty:
            return (
                standardize_leads(baseline, "Sanitized public UDR conversions baseline"),
                _load_baseline_pivots("udr"),
                [f"Private UDR conversions workbook not found; using sanitized public baseline from {PUBLIC_BASELINE_DIR}."],
            )
        return pd.DataFrame(), {}, [f"Missing UDR conversions workbook: {path}"]
    xl = pd.ExcelFile(path)
    notes.append(f"UDR conversions workbook sheets inspected: {', '.join(xl.sheet_names)}")
    sheet = "UDR leads conversions"
    raw = pd.read_excel(path, sheet_name=sheet) if sheet in xl.sheet_names else pd.DataFrame()
    pivots = {
        name: parse_pivot_sheet(path, name)
        for name in ["L2C", "L2A", "VLC", "VLA"]
        if name in xl.sheet_names
    }
    return standardize_leads(raw, "Uploaded UDR conversions workbook"), pivots, notes


def _html_to_text(markup: str) -> str:
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(markup, "html.parser").get_text("\n")
    except Exception:
        text = re.sub(r"<(br|p|div|li|tr|h[1-6])[^>]*>", "\n", markup, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return html.unescape(text)


def _extract_text_from_message(path: Path) -> tuple[Any, list[str], int, list[bytes]]:
    with path.open("rb") as handle:
        message = BytesParser(policy=policy.default).parse(handle)

    text_parts: list[str] = []
    image_parts: list[bytes] = []
    notes: list[str] = []

    for part in message.walk():
        content_type = part.get_content_type()
        if content_type == "text/plain":
            payload = part.get_content()
            if payload:
                text_parts.append(str(payload))
        elif content_type == "text/html":
            payload = part.get_content()
            if payload:
                text_parts.append(_html_to_text(str(payload)))
        elif part.get_content_maintype() == "image":
            try:
                image_parts.append(part.get_content())
            except Exception as exc:
                notes.append(f"Could not extract embedded image: {exc}")

    image_count = len(image_parts)
    return message, text_parts, image_count, image_parts


def _ocr_images(image_parts: list[bytes]) -> tuple[str, list[str]]:
    if not image_parts:
        return "", []
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return "", ["Embedded image OCR was skipped because PIL/pytesseract is not available."]

    text_chunks: list[str] = []
    notes: list[str] = []
    for idx, payload in enumerate(image_parts, start=1):
        try:
            image = Image.open(io.BytesIO(payload))
            text = pytesseract.image_to_string(image)
            if text.strip():
                text_chunks.append(text.strip())
        except Exception as exc:
            notes.append(f"OCR failed for embedded image {idx}: {exc}")
    return "\n\n".join(text_chunks), notes


def parse_email_context(path: Path = EMAIL_FILE) -> EmailContext:
    if not path.exists():
        context = (_read_baseline_metadata().get("email_context") or {})
        if context:
            return EmailContext(
                subject=clean_text(context.get("subject", "")),
                sent_at=clean_text(context.get("sent_at", "")),
                sender="",
                text="",
                data_lines=list(context.get("data_lines") or []),
                image_count=int(context.get("image_count") or 0),
                ocr_text="",
                notes=list(context.get("notes") or ["Private email source not found; using sanitized public baseline metadata."]),
            )
        return EmailContext("", "", "", "", [], 0, "", [f"Missing email source: {path}"])

    message, text_parts, image_count, image_parts = _extract_text_from_message(path)
    combined = "\n".join(text_parts)
    combined = re.sub(r"\n{3,}", "\n\n", combined).strip()
    ocr_text, ocr_notes = _ocr_images(image_parts)
    combined_for_scan = "\n".join([combined, ocr_text]).strip()
    keywords = re.compile(
        r"(enroll|lead|applicant|start|budget|revenue|goal|source|udr|summer|week|pace|conversion)",
        re.I,
    )
    data_lines = [
        re.sub(r"\s+", " ", line).strip()
        for line in combined_for_scan.splitlines()
        if any(char.isdigit() for char in line) and keywords.search(line)
    ]
    data_lines = [line for line in data_lines if line][:25]
    notes = list(ocr_notes)
    if not data_lines:
        notes.append("No usable weekly summary metrics were found in the parsed email text.")
    return EmailContext(
        subject=clean_text(message.get("subject", "")),
        sent_at=clean_text(message.get("date", "")),
        sender=clean_text(message.get("from", "")),
        text=combined,
        data_lines=data_lines,
        image_count=image_count,
        ocr_text=ocr_text,
        notes=notes,
    )


def load_uploaded_lead_data() -> UploadedLeadData:
    paid_leads, paid_pivots, paid_notes = load_paid_leads()
    udr_leads, udr_pivots, udr_notes = load_udr_conversions()
    email_context = parse_email_context()
    return UploadedLeadData(
        paid_leads=paid_leads,
        udr_leads=udr_leads,
        paid_pivots=paid_pivots,
        udr_pivots=udr_pivots,
        email_context=email_context,
        load_notes=[*paid_notes, *udr_notes, *email_context.notes],
    )


def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    pii_columns = {
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "student",
        "notes",
        "sender",
    }
    return df.drop(columns=[column for column in pii_columns if column in df.columns], errors="ignore")
