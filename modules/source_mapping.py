from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


UNKNOWN_SOURCE = "Unmapped"

SOURCE_PRIORITY = [
    "paid_lead_list",
    "organic_lead_list",
    "campaign",
    "utm_campaign",
    "utm_source",
    "utm_medium",
    "event_attended",
    "source",
    "enrollment_source",
    "hs_latest_source_data_1",
    "hs_latest_source_data_2",
    "hs_analytics_source_data_1",
    "hs_analytics_source_data_2",
    "hs_latest_source",
    "hs_analytics_source",
]

PAID_HINTS = {
    "ad",
    "ads",
    "agent",
    "atra",
    "clearance jobs",
    "eddie",
    "eddie leads",
    "ebg",
    "falcon",
    "hiregi",
    "kara",
    "karyl",
    "leads",
    "nathen",
    "oby",
    "paid",
    "project falcon",
    "vendor",
}

ORGANIC_HINTS = {
    "alumni",
    "direct",
    "event",
    "organic",
    "referral",
    "website",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\xa0", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_column_name(column: object) -> str:
    text = clean_text(column).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unnamed"


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in out.columns:
        base = clean_column_name(column)
        count = seen.get(base, 0)
        seen[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count + 1}")
    out.columns = columns
    return out


def normalize_source_value(value: object) -> str:
    text = clean_text(value)
    if not text:
        return UNKNOWN_SOURCE
    lowered = text.lower()
    aliases = {
        "(blank)": UNKNOWN_SOURCE,
        "nan": UNKNOWN_SOURCE,
        "none": UNKNOWN_SOURCE,
        "website": "Website",
        "referral": "Referral",
        "alumni": "Alumni",
        "falcon": "Project Falcon",
        "project falcon": "Project Falcon",
        "eddie leads": "Eddie Leads",
        "kara leads": "Kara Leads",
        "karyl leads": "Karyl Leads",
        "hiregi": "HireGI",
        "ebg": "EBG",
        "atra": "ATRA",
    }
    if lowered in aliases:
        return aliases[lowered]
    return text


def first_nonempty(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        if column in row.index:
            value = clean_text(row[column])
            if value:
                return value
    return ""


def normalized_source(row: pd.Series, columns: Iterable[str] | None = None) -> str:
    fields = list(columns or SOURCE_PRIORITY)
    return normalize_source_value(first_nonempty(row, fields))


def classify_paid_organic(row: pd.Series) -> str:
    paid_value = clean_text(row.get("paid_lead_list", ""))
    organic_value = clean_text(row.get("organic_lead_list", ""))
    if paid_value:
        return "Paid"
    if organic_value:
        return "Organic"

    source = normalized_source(row).lower()
    if source == UNKNOWN_SOURCE.lower():
        return "Unknown"
    if any(hint in source for hint in PAID_HINTS):
        return "Paid"
    if any(hint in source for hint in ORGANIC_HINTS):
        return "Organic"
    return "Unknown"


def add_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["normalized_source"] = pd.Series(dtype=str)
        out["lead_type"] = pd.Series(dtype=str)
        return out
    out = df.copy()
    out["normalized_source"] = out.apply(normalized_source, axis=1)
    out["lead_type"] = out.apply(classify_paid_organic, axis=1)
    return out


def source_set(df: pd.DataFrame, column: str = "normalized_source") -> set[str]:
    if df.empty or column not in df.columns:
        return set()
    return {
        normalize_source_value(value)
        for value in df[column].dropna().astype(str)
        if normalize_source_value(value) != UNKNOWN_SOURCE
    }

