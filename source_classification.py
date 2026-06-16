from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from config import PAID_SOURCE_RULES_PATH, VENDOR_RULES_PATH, load_json_safe, save_json


DEFAULT_RULES = {
    "paid_terms": [
        "paid",
        "cpc",
        "ppc",
        "google ads",
        "adwords",
        "facebook",
        "instagram",
        "meta",
        "linkedin",
        "bing",
        "microsoft ads",
        "tiktok",
        "display",
        "paid search",
        "paid social",
        "atra",
        "klient boost",
        "klientboost",
    ],
    "paid_search_terms": ["paid search", "google ads", "adwords", "cpc", "ppc", "bing", "microsoft ads"],
    "paid_social_terms": ["paid social", "facebook", "instagram", "meta", "linkedin", "tiktok"],
    "vendor_aliases": {
        "Atra": ["atra"],
        "Google Ads": ["google ads", "adwords"],
        "Meta": ["facebook", "instagram", "meta"],
        "LinkedIn": ["linkedin"],
        "Bing": ["bing"],
        "TikTok": ["tiktok"],
    },
}
DEFAULT_VENDOR_RULES = {
    "vendor_aliases": {
        "Atra": ["atra", "atra analytics"],
        "KlientBoost": ["klient boost", "klientboost"],
        "Google": ["google", "google ads", "adwords"],
        "Meta": ["facebook", "meta", "instagram"],
        "LinkedIn": ["linkedin"],
        "Microsoft/Bing": ["bing", "microsoft ads"],
    },
    "vendor_field_terms": ["vendor", "partner", "agency", "referral", "source", "campaign"],
    "preferred_source_fields": [
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
    ],
}


def load_source_rules() -> dict[str, list[str]]:
    rules = load_json_safe(PAID_SOURCE_RULES_PATH, default=None)
    if not rules:
        save_json(PAID_SOURCE_RULES_PATH, DEFAULT_RULES)
        return DEFAULT_RULES
    return rules


def load_vendor_rules() -> dict[str, Any]:
    rules = load_json_safe(VENDOR_RULES_PATH, default=None)
    if not rules:
        save_json(VENDOR_RULES_PATH, DEFAULT_VENDOR_RULES)
        return DEFAULT_VENDOR_RULES
    return rules


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def is_paid_source(source_text: str, rules: dict[str, list[str]] | None = None) -> bool:
    rules = rules or load_source_rules()
    return _contains_any(source_text or "", rules.get("paid_terms", []))


def classify_source_group(source_text: str, rules: dict[str, list[str]] | None = None) -> str:
    rules = rules or load_source_rules()
    text = (source_text or "").strip().lower()
    if not text:
        return "Unknown"

    if _contains_any(text, rules.get("paid_social_terms", [])):
        return "Paid Social"
    if _contains_any(text, rules.get("paid_search_terms", [])):
        return "Paid Search"
    if "organic" in text or "seo" in text:
        return "Organic Search"
    if "referral" in text or "refer" in text:
        return "Referral"
    if "email" in text or "newsletter" in text:
        return "Email"
    if "direct" in text:
        return "Direct"
    return "Other"


def classify_paid_vendor(source_text: str, rules: dict[str, Any] | None = None) -> str:
    rules = rules or load_vendor_rules()
    text = (source_text or "").strip()
    if not text:
        return "Unknown"
    aliases = rules.get("vendor_aliases", {})
    if isinstance(aliases, dict):
        for vendor, terms in aliases.items():
            if _contains_any(text, terms):
                return vendor
    parts = [part.strip() for part in text.split("|") if part.strip()]
    return parts[0] if parts else "Unknown"


def vendor_source_columns(df: pd.DataFrame, rules: dict[str, Any] | None = None) -> list[str]:
    rules = rules or load_vendor_rules()
    preferred = [column for column in rules.get("preferred_source_fields", []) if column in df.columns]
    terms = [str(term).lower() for term in rules.get("vendor_field_terms", [])]
    dynamic = [
        column
        for column in df.columns
        if any(term in column.lower() for term in terms)
    ]
    output: list[str] = []
    for column in [*preferred, *dynamic]:
        if column not in output:
            output.append(column)
    return output


def detect_vendor_from_row(row: pd.Series, columns: Iterable[str], rules: dict[str, Any] | None = None) -> tuple[str, str, str]:
    rules = rules or load_vendor_rules()
    aliases = rules.get("vendor_aliases", {})
    for column in columns:
        value = row.get(column)
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            continue
        if isinstance(aliases, dict):
            for vendor, terms in aliases.items():
                if _contains_any(text, terms):
                    return str(vendor), "High", column

    for column in columns:
        value = row.get(column)
        text = "" if pd.isna(value) else str(value).strip()
        if not text:
            continue
        vendor = classify_paid_vendor(text, rules)
        if vendor != "Unknown":
            return vendor, "Medium", column
    return "Unknown", "Low", ""


def add_source_classification(
    df: pd.DataFrame,
    source_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    output = df.copy()
    if source_columns is None:
        source_columns = vendor_source_columns(output)
    available = [col for col in source_columns if col in output.columns]
    if available:
        source_text = (
            output[available]
            .fillna("")
            .astype(str)
            .agg(" | ".join, axis=1)
            .str.strip()
            .str.strip("|")
        )
    else:
        source_text = pd.Series([""] * len(output), index=output.index)

    rules = load_source_rules()
    vendor_rules = load_vendor_rules()
    output["source_text"] = source_text
    output["paid_lead_flag"] = source_text.map(lambda text: is_paid_source(text, rules))
    output["source_group"] = source_text.map(lambda text: classify_source_group(text, rules))
    vendor_results = output.apply(lambda row: detect_vendor_from_row(row, available, vendor_rules), axis=1)
    output["vendor"] = vendor_results.map(lambda result: result[0])
    output["vendor_confidence"] = vendor_results.map(lambda result: result[1])
    output["vendor_source_field"] = vendor_results.map(lambda result: result[2])
    output["paid_vendor"] = output["vendor"]
    return output
