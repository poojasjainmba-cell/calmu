from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


BAD_LEAD_STATUSES = {
    "dead lead",
    "do not contact",
    "duplicate lead",
    "app submitted - unqualified",
}

CONTACTED_STATUSES = {
    "outreach attempt 1",
    "outreach attempt 2",
    "outreach attempt 3",
    "outreach attempt 4",
    "responded",
    "warm lead",
    "hot lead",
    "cold lead",
    "qualified meeting set",
    "interviewed - app ready",
    "interviewed - not ready",
    "app submitted - qualified",
    "app submitted - unqualified",
    "future applicant",
    "enrolled",
}


@dataclass
class LeadSummary:
    total_leads: int
    paid_leads: int
    organic_leads: int
    contacted_leads: int
    applicants: int
    crm_enrolled: int
    bad_leads: int
    uncontacted_leads: int
    lead_to_contact_rate: float
    lead_to_applicant_rate: float
    lead_to_crm_enrolled_rate: float
    bad_lead_rate: float


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _identity_col(df: pd.DataFrame) -> str | None:
    for column in ["record_id", "hs_object_id", "id", "email"]:
        if column in df.columns and df[column].fillna("").astype(str).str.strip().ne("").any():
            return column
    return None


def _unique_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    identity = _identity_col(df)
    if identity:
        return int(df[identity].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    return int(len(df))


def _lower_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].fillna("").astype(str).str.strip().str.lower()


def add_lead_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for column in ["is_contacted", "is_applicant", "is_crm_enrolled", "is_bad_lead"]:
            out[column] = pd.Series(dtype=bool)
        return out

    lead_status = _lower_series(out, "lead_status")
    lifecycle = _lower_series(out, "lifecycle_stage")
    explicit_contact_cols = [
        column
        for column in ["contacted", "contacted_progressed", "is_contacted"]
        if column in out.columns
    ]
    if explicit_contact_cols:
        explicit = out[explicit_contact_cols[0]].fillna("").astype(str).str.lower()
        contacted = explicit.isin(["true", "1", "yes", "y", "contacted"])
    else:
        last_activity = (
            out["last_activity_date"].notna()
            if "last_activity_date" in out.columns
            else pd.Series([False] * len(out), index=out.index)
        )
        contacted = last_activity | lead_status.isin(CONTACTED_STATUSES)

    out["is_contacted"] = contacted.fillna(False)
    out["is_applicant"] = lifecycle.eq("applicant")
    out["is_crm_enrolled"] = lifecycle.eq("enrolled")
    out["is_bad_lead"] = lifecycle.eq("not a lead") | lead_status.isin(BAD_LEAD_STATUSES)
    return out


def summarize_leads(df: pd.DataFrame) -> LeadSummary:
    flagged = add_lead_flags(df)
    total = _unique_count(flagged)
    lead_type = (
        flagged["lead_type"].fillna("").astype(str)
        if "lead_type" in flagged.columns
        else pd.Series([""] * len(flagged), index=flagged.index)
    )
    paid = _unique_count(flagged[lead_type.eq("Paid")])
    organic = _unique_count(flagged[lead_type.eq("Organic")])
    contacted = _unique_count(flagged[flagged["is_contacted"]])
    applicants = _unique_count(flagged[flagged["is_applicant"]])
    enrolled = _unique_count(flagged[flagged["is_crm_enrolled"]])
    bad = _unique_count(flagged[flagged["is_bad_lead"]])
    uncontacted = max(total - contacted, 0)
    return LeadSummary(
        total_leads=total,
        paid_leads=paid,
        organic_leads=organic,
        contacted_leads=contacted,
        applicants=applicants,
        crm_enrolled=enrolled,
        bad_leads=bad,
        uncontacted_leads=uncontacted,
        lead_to_contact_rate=safe_divide(contacted, total),
        lead_to_applicant_rate=safe_divide(applicants, total),
        lead_to_crm_enrolled_rate=safe_divide(enrolled, total),
        bad_lead_rate=safe_divide(bad, total),
    )


def _metric_dict_for_group(group: pd.DataFrame) -> dict[str, Any]:
    summary = summarize_leads(group)
    return {
        "leads": summary.total_leads,
        "paid_leads": summary.paid_leads,
        "organic_leads": summary.organic_leads,
        "contacted_progressed": summary.contacted_leads,
        "applicants": summary.applicants,
        "crm_enrolled": summary.crm_enrolled,
        "bad_leads": summary.bad_leads,
        "uncontacted_leads": summary.uncontacted_leads,
        "l2c": summary.lead_to_contact_rate,
        "l2a": summary.lead_to_applicant_rate,
        "l2e": summary.lead_to_crm_enrolled_rate,
        "bad_lead_rate": summary.bad_lead_rate,
    }


def lead_performance_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for label, group in df.groupby(group_col, dropna=False):
        label_text = str(label) if pd.notna(label) and str(label).strip() else "Unmapped"
        rows.append({group_col: label_text, **_metric_dict_for_group(group)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["leads", group_col], ascending=[False, True])
    return out


def enrollment_summary(enrollments: pd.DataFrame, goal: float | None = None, starts: float | None = None) -> dict[str, Any]:
    actual = int(len(enrollments)) if not enrollments.empty else 0
    revenue = float(pd.to_numeric(enrollments.get("revenue", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    avg_days = float(
        pd.to_numeric(enrollments.get("days_to_enroll", pd.Series(dtype=float)), errors="coerce").dropna().mean()
    ) if not enrollments.empty else 0.0
    goal_value = float(goal or 0)
    starts_value = float(starts or 0)
    remaining = max(goal_value - actual, 0)
    return {
        "actual_enrollments": actual,
        "enrollment_goal": goal_value,
        "percent_of_goal": safe_divide(actual, goal_value),
        "remaining_enrollments": remaining,
        "starts": starts_value,
        "start_rate": safe_divide(starts_value, actual),
        "revenue": revenue,
        "revenue_per_enrollment": safe_divide(revenue, actual),
        "average_days_to_enroll": avg_days if pd.notna(avg_days) else 0.0,
    }


def roundup_value(summary: pd.DataFrame, metric: str) -> float | None:
    if summary.empty:
        return None
    match = summary[summary["metric"].fillna("").astype(str).str.lower().eq(metric.lower())]
    if match.empty:
        return None
    value = pd.to_numeric(match.iloc[0].get("numeric_value"), errors="coerce")
    return float(value) if pd.notna(value) else None


def enrollment_by(enrollments: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if enrollments.empty or group_col not in enrollments.columns:
        return pd.DataFrame()
    grouped = enrollments.groupby(group_col, dropna=False)
    out = grouped.agg(
        actual_enrollments=("student", "count"),
        revenue=("revenue", "sum"),
        average_days_to_enroll=("days_to_enroll", "mean"),
    ).reset_index()
    out[group_col] = out[group_col].fillna("Missing").astype(str).replace("", "Missing")
    return out.sort_values("actual_enrollments", ascending=False)


def source_performance(leads: pd.DataFrame, enrollments: pd.DataFrame, budget_allocations: pd.DataFrame) -> pd.DataFrame:
    lead_perf = lead_performance_by(leads, "normalized_source")
    enroll_perf = enrollment_by(enrollments, "normalized_source")
    if not enroll_perf.empty:
        enroll_perf = enroll_perf.rename(columns={"normalized_source": "normalized_source"})
    frames = []
    if not lead_perf.empty:
        frames.append(lead_perf.set_index("normalized_source"))
    if not enroll_perf.empty:
        frames.append(enroll_perf.set_index("normalized_source"))
    if frames:
        out = pd.concat(frames, axis=1).reset_index()
    else:
        out = pd.DataFrame(columns=["normalized_source"])

    if not budget_allocations.empty and "source" in budget_allocations.columns:
        budget = budget_allocations.groupby("source", dropna=False)["planned_budget"].sum().reset_index()
        budget = budget.rename(columns={"source": "normalized_source"})
        out = out.merge(budget, on="normalized_source", how="outer") if not out.empty else budget
    for column in ["leads", "applicants", "actual_enrollments", "planned_budget", "revenue"]:
        if column not in out.columns:
            out[column] = 0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)
    out["cost_per_lead"] = out.apply(lambda row: safe_divide(row["planned_budget"], row["leads"]), axis=1)
    out["cost_per_applicant"] = out.apply(lambda row: safe_divide(row["planned_budget"], row["applicants"]), axis=1)
    out["cost_per_enrollment"] = out.apply(
        lambda row: safe_divide(row["planned_budget"], row["actual_enrollments"]), axis=1
    )
    return out.sort_values(["leads", "actual_enrollments"], ascending=False)


def funnel_counts(leads: pd.DataFrame, actual_enrollments: int) -> pd.DataFrame:
    summary = summarize_leads(leads)
    return pd.DataFrame(
        [
            {"stage": "Leads", "count": summary.total_leads},
            {"stage": "Contacted", "count": summary.contacted_leads},
            {"stage": "Applicants", "count": summary.applicants},
            {"stage": "CRM enrolled", "count": summary.crm_enrolled},
            {"stage": "Actual enrollments", "count": actual_enrollments},
        ]
    )


def weekly_counts(df: pd.DataFrame, date_col: str, value_name: str) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return pd.DataFrame(columns=["week", value_name])
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_convert(None)
    work = df[dates.notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["week", value_name])
    work["week"] = dates[dates.notna()].dt.to_period("W").dt.start_time
    return work.groupby("week").size().reset_index(name=value_name).sort_values("week")


def filter_date_range(df: pd.DataFrame, column: str, start: date | None, end: date | None) -> pd.DataFrame:
    if df.empty or column not in df.columns or (start is None and end is None):
        return df
    dates = pd.to_datetime(df[column], errors="coerce", utc=True)
    mask = pd.Series([True] * len(df), index=df.index)
    if start is not None:
        mask &= dates.dt.date >= start
    if end is not None:
        mask &= dates.dt.date <= end
    return df[mask].copy()


def compare_pivot_totals(raw: pd.DataFrame, pivot: pd.DataFrame, group_col: str, pivot_name: str) -> pd.DataFrame:
    if raw.empty or pivot.empty or group_col not in raw.columns or "row_label" not in pivot.columns:
        return pd.DataFrame()
    raw_counts = raw.groupby(group_col, dropna=False).size().rename("raw_total").reset_index()
    raw_counts[group_col] = raw_counts[group_col].fillna("").astype(str).replace("", "Unmapped")
    grand_col = "grand_total" if "grand_total" in pivot.columns else None
    if not grand_col:
        return pd.DataFrame()
    pivot_counts = pivot[["row_label", grand_col]].copy()
    pivot_counts[grand_col] = pd.to_numeric(pivot_counts[grand_col], errors="coerce").fillna(0)
    out = raw_counts.merge(pivot_counts, left_on=group_col, right_on="row_label", how="outer")
    out["raw_total"] = pd.to_numeric(out["raw_total"], errors="coerce").fillna(0)
    out["pivot_total"] = pd.to_numeric(out[grand_col], errors="coerce").fillna(0)
    out["difference"] = out["raw_total"] - out["pivot_total"]
    out["pivot_name"] = pivot_name
    out["group"] = out[group_col].fillna(out["row_label"]).fillna("Unmapped")
    return out[["pivot_name", "group", "raw_total", "pivot_total", "difference"]]
