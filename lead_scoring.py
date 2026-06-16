from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


QUALIFIED_TERMS = ("mql", "sql", "qualified", "opportunity")
DEAD_TERMS = (
    "closed lost",
    "closedlost",
    "disqualified",
    "unqualified",
    "not interested",
    "bad fit",
    "invalid",
    "junk",
    "duplicate",
)
ARCHIVE_TERMS = (
    "duplicate",
    "junk",
    "invalid",
    "disqualified",
    "not interested",
    "closed lost",
    "closedlost",
)


def parse_datetime_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.to_datetime(series, errors="coerce", utc=True)
    if is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean() if len(numeric) else 0
    if numeric_ratio > 0.5 and numeric.dropna().median() > 10_000_000_000:
        return pd.to_datetime(numeric, errors="coerce", unit="ms", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def _contains_any(text: Any, terms: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def _has_text(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "nat", "<na>", "none"}


def _status_text(row: pd.Series) -> str:
    values = [
        row.get("lifecycle_stage"),
        row.get("lead_status"),
        row.get("deal_stage"),
        row.get("latest_deal_stage"),
    ]
    return " | ".join(str(value) for value in values if pd.notna(value) and str(value).strip())


def _dead_reason(row: pd.Series) -> str:
    status = _status_text(row).lower()
    for term in DEAD_TERMS:
        if term in status:
            return term.replace(" ", "_")

    lead_age = row.get("lead_age_days")
    inactive_days = row.get("days_since_last_activity")
    has_open_deal = bool(row.get("has_open_deal", False))
    never_contacted = bool(row.get("never_contacted", False))

    if pd.notna(lead_age) and lead_age > 60 and not has_open_deal:
        if pd.isna(inactive_days) or inactive_days >= 30:
            return "inactive_60_days_no_open_deal"
    if pd.notna(lead_age) and lead_age > 30 and never_contacted:
        return "never_contacted_30_days"
    if row.get("lead_score", 0) < 15:
        return "low_score"
    return ""


def score_leads(df: pd.DataFrame, today: datetime | None = None) -> pd.DataFrame:
    output = df.copy()
    today = today or datetime.now(timezone.utc)

    for column in ("contact_created_at", "last_activity_date", "next_activity_date"):
        if column not in output.columns:
            output[column] = pd.NaT
        output[column] = parse_datetime_series(output[column])

    output["lead_age_days"] = (today - output["contact_created_at"]).dt.days
    output["days_since_last_activity"] = (today - output["last_activity_date"]).dt.days
    output["has_open_deal"] = output.get("has_open_deal", False)
    output["has_open_deal"] = output["has_open_deal"].fillna(False).astype(bool)
    output["never_contacted"] = output["last_activity_date"].isna()

    scores: list[int] = []
    for _, row in output.iterrows():
        score = 0
        lead_age = row.get("lead_age_days")
        inactive_days = row.get("days_since_last_activity")
        status = _status_text(row)

        if pd.notna(lead_age) and lead_age <= 7:
            score += 25
        if bool(row.get("paid_lead_flag", False)):
            score += 15
        if _contains_any(status, QUALIFIED_TERMS):
            score += 20
        if bool(row.get("has_open_deal", False)):
            score += 25
        if pd.notna(inactive_days) and inactive_days <= 7:
            score += 20
        if pd.notna(row.get("next_activity_date")):
            score += 10
        if _has_text(row.get("email")) or _has_text(row.get("phone")):
            score += 5
        if pd.notna(inactive_days) and 15 <= inactive_days <= 30:
            score -= 20
        if pd.notna(inactive_days) and inactive_days > 30:
            score -= 40
        if pd.notna(lead_age) and lead_age > 45 and not bool(row.get("has_open_deal", False)):
            score -= 25
        scores.append(score)

    output["lead_score"] = scores
    output["dead_reason"] = output.apply(_dead_reason, axis=1)

    temperatures: list[str] = []
    for _, row in output.iterrows():
        status = _status_text(row)
        dead_override = bool(row.get("dead_reason"))
        score = int(row.get("lead_score") or 0)
        if dead_override or score < 15:
            temperatures.append("Dead")
        elif score >= 60 and not _contains_any(status, ("lost", "disqualified")):
            temperatures.append("Hot")
        elif 35 <= score <= 59:
            temperatures.append("Warm")
        else:
            temperatures.append("Cold")
    output["lead_temperature"] = temperatures

    def reviveable(row: pd.Series) -> bool:
        if row.get("lead_temperature") != "Dead":
            return False
        if row.get("dead_reason") not in {
            "inactive_60_days_no_open_deal",
            "never_contacted_30_days",
            "low_score",
        }:
            return False
        has_value = any(_has_text(row.get(column)) for column in ("program", "email", "phone"))
        has_previous_engagement = pd.notna(row.get("last_activity_date"))
        has_value = has_value or bool(has_previous_engagement)
        return bool(row.get("paid_lead_flag", False)) or has_value

    output["reviveable_flag"] = output.apply(reviveable, axis=1)
    output["archive_remove_flag"] = output["dead_reason"].map(
        lambda reason: any(term.replace(" ", "_") in str(reason) for term in ARCHIVE_TERMS)
    )
    return output
