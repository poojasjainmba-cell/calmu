from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from definitions import metric_definition, section_help_text


STATUS_COLUMNS = [
    "enrollment_status",
    "final_lead_status",
    "lead_status",
    "lifecycle_stage",
    "latest_deal_stage",
    "deal_stage",
]
DEAD_TERMS = ("dead", "lost", "disqualified", "unqualified", "not interested", "invalid", "junk", "duplicate")
ENROLLED_TERMS = ("enrolled", "ea signed", "active student", "registered", "accepted", "closed won", "won")


def _section_intro(name: str) -> None:
    help_text = section_help_text(name)
    if help_text:
        st.caption(help_text)


def _table_column_config(df: pd.DataFrame) -> dict[str, Any]:
    config = {}
    for column in df.columns:
        help_text = metric_definition(column)
        if help_text:
            config[column] = st.column_config.Column(help=help_text)
    return config


def _data_table(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch", hide_index=True, column_config=_table_column_config(df))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "<na>", "none"} else text


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[column].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _num_series(df: pd.DataFrame, column: str, default: float = 0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    text = df[column].fillna("").astype(str).str.strip()
    return text.mask(text.str.lower().isin({"nan", "nat", "<na>", "none"}), "")


def _has_text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return _text_series(df, column).ne("")


def _contact_ids(df: pd.DataFrame) -> pd.Series:
    if "contact_id" not in df.columns:
        return pd.Series(dtype=str)
    return df["contact_id"].dropna().astype(str)


def _enrolled_mask(df: pd.DataFrame) -> pd.Series:
    mask = _bool_series(df, "has_won_deal") | _bool_series(df, "is_won")
    pattern = "|".join(re.escape(term) for term in ENROLLED_TERMS)
    for column in ("enrollment_status", "final_lead_status", "lead_status", "lifecycle_stage", "latest_deal_stage", "deal_stage"):
        if column in df.columns:
            mask = mask | _text_series(df, column).str.lower().str.contains(pattern, regex=True, na=False)
    if "enrolled_revenue" in df.columns:
        mask = mask | (_num_series(df, "enrolled_revenue").fillna(0) > 0)
    return mask


def _dead_mask(df: pd.DataFrame) -> pd.Series:
    mask = _bool_series(df, "has_lost_deal")
    if "lead_temperature" in df.columns:
        mask = mask | _text_series(df, "lead_temperature").str.lower().eq("dead")
    pattern = "|".join(re.escape(term) for term in DEAD_TERMS)
    for column in ("final_lead_status", "lead_status", "lifecycle_stage", "latest_deal_stage", "deal_stage"):
        if column in df.columns:
            mask = mask | _text_series(df, column).str.lower().str.contains(pattern, regex=True, na=False)
    return mask


def _first_available_status(df: pd.DataFrame) -> pd.Series:
    status = pd.Series([""] * len(df), index=df.index)
    for column in STATUS_COLUMNS:
        values = _text_series(df, column)
        status = status.mask(status.eq("") & values.ne(""), values)
    temperature = _text_series(df, "lead_temperature")
    status = status.mask(status.eq("") & temperature.ne(""), temperature + " Lead")
    return status.mask(status.eq(""), "Missing Status")


def _formal_status_missing(df: pd.DataFrame) -> pd.Series:
    missing = pd.Series([True] * len(df), index=df.index)
    for column in STATUS_COLUMNS:
        missing = missing & ~_has_text_series(df, column)
    return missing


def _actual_revenue_by_contact(fact: pd.DataFrame) -> dict[str, float]:
    if fact.empty or "contact_id" not in fact.columns or "revenue_attributed" not in fact.columns:
        return {}
    work = fact.copy()
    work["contact_id"] = work["contact_id"].fillna("").astype(str)
    work["revenue_attributed"] = pd.to_numeric(work["revenue_attributed"], errors="coerce").fillna(0)
    return work[work["contact_id"].ne("")].groupby("contact_id")["revenue_attributed"].sum().to_dict()


def _deal_contact_sets(fact: pd.DataFrame) -> tuple[set[str], set[str]]:
    if fact.empty or "contact_id" not in fact.columns:
        return set(), set()
    contact_ids = fact["contact_id"].fillna("").astype(str)
    deal_ids = fact.get("deal_id", pd.Series("", index=fact.index)).fillna("").astype(str).str.strip()
    has_deal = set(fact.loc[contact_ids.ne("") & deal_ids.ne(""), "contact_id"].astype(str))
    won = set()
    if "is_won" in fact.columns:
        won = set(fact.loc[contact_ids.ne("") & _bool_series(fact, "is_won"), "contact_id"].astype(str))
    return has_deal, won


def prepare_pipeline_contacts(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    output = contacts.copy()
    if output.empty:
        return output
    required_columns = {
        "pipeline_status",
        "actual_revenue",
        "pipeline_potential_revenue",
        "is_enrolled_pipeline",
        "is_dead_pipeline",
        "formal_status_missing",
    }
    if required_columns.issubset(output.columns):
        return output
    output["contact_id"] = output.get("contact_id", pd.Series(index=output.index, dtype=str)).astype(str)
    enrolled = _enrolled_mask(output)
    dead = _dead_mask(output)
    output["pipeline_status"] = _first_available_status(output)
    output.loc[dead, "pipeline_status"] = "Closed Lost / Dead"
    output.loc[enrolled, "pipeline_status"] = "Enrolled / Closed Won"
    output["actual_revenue"] = output["contact_id"].map(_actual_revenue_by_contact(fact)).fillna(0)
    if "potential_revenue" in output.columns:
        output["pipeline_potential_revenue"] = _num_series(output, "potential_revenue").fillna(0)
    elif "open_pipeline_potential_revenue" in output.columns:
        output["pipeline_potential_revenue"] = _num_series(output, "open_pipeline_potential_revenue").fillna(0)
    else:
        output["pipeline_potential_revenue"] = 0
    output["is_enrolled_pipeline"] = enrolled
    output["is_dead_pipeline"] = dead
    output["formal_status_missing"] = _formal_status_missing(output)
    return output


def lead_status_summary(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_pipeline_contacts(contacts, fact)
    columns = [
        "status",
        "total leads",
        "paid leads",
        "hot leads",
        "stale leads",
        "dead leads",
        "enrolled",
        "actual revenue",
        "potential revenue",
    ]
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    stale = _num_series(prepared, "days_since_last_activity").fillna(9999) >= 14
    rows = []
    for status, group in prepared.groupby("pipeline_status", dropna=False):
        group_index = group.index
        rows.append(
            {
                "status": status,
                "total leads": int(_contact_ids(group).nunique()),
                "paid leads": int(_contact_ids(group[_bool_series(group, "paid_lead_flag")]).nunique()),
                "hot leads": int(_contact_ids(group[group.get("lead_temperature", "") == "Hot"]).nunique()),
                "stale leads": int(_contact_ids(group[stale.loc[group_index]]).nunique()),
                "dead leads": int(_contact_ids(group[group["is_dead_pipeline"]]).nunique()),
                "enrolled": int(_contact_ids(group[group["is_enrolled_pipeline"]]).nunique()),
                "actual revenue": float(group["actual_revenue"].sum()),
                "potential revenue": float(group["pipeline_potential_revenue"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("total leads", ascending=False)


def _recommended_bottleneck_action(status: str, stuck_30: int, stale_14: int) -> str:
    lowered = status.lower()
    if "missing" in lowered:
        return "Assign a meaningful lead status."
    if any(term in lowered for term in DEAD_TERMS):
        return "Confirm dead/lost reason and archive or nurture."
    if stuck_30:
        return "Manager review: clear next step or close out."
    if stale_14:
        return "Schedule follow-up and update next activity."
    return "Monitor normal movement."


def stage_bottlenecks(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_pipeline_contacts(contacts, fact)
    columns = [
        "stage/status",
        "leads stuck 7+ days",
        "leads stuck 14+ days",
        "leads stuck 30+ days",
        "average age",
        "recommended action",
    ]
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    inactive = _num_series(prepared, "days_since_last_activity").fillna(9999)
    age = _num_series(prepared, "lead_age_days").fillna(0)
    open_mask = ~prepared["is_enrolled_pipeline"] & ~prepared["is_dead_pipeline"]
    rows = []
    for status, group in prepared.groupby("pipeline_status", dropna=False):
        idx = group.index
        stuck_7 = int(_contact_ids(group[open_mask.loc[idx] & (inactive.loc[idx] >= 7)]).nunique())
        stuck_14 = int(_contact_ids(group[open_mask.loc[idx] & (inactive.loc[idx] >= 14)]).nunique())
        stuck_30 = int(_contact_ids(group[open_mask.loc[idx] & (inactive.loc[idx] >= 30)]).nunique())
        rows.append(
            {
                "stage/status": status,
                "leads stuck 7+ days": stuck_7,
                "leads stuck 14+ days": stuck_14,
                "leads stuck 30+ days": stuck_30,
                "average age": float(age.loc[idx].mean()) if len(idx) else 0.0,
                "recommended action": _recommended_bottleneck_action(str(status), stuck_30, stuck_14),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["leads stuck 30+ days", "leads stuck 14+ days", "leads stuck 7+ days"],
        ascending=False,
    )


def enrollment_path(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_pipeline_contacts(contacts, fact)
    columns = ["milestone", "leads", "paid leads", "potential revenue"]
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    deal_contact_ids, won_contact_ids = _deal_contact_sets(fact)
    contacted = _has_text_series(prepared, "last_activity_date") | (_num_series(prepared, "num_notes").fillna(0) > 0)
    has_application_or_deal = prepared["contact_id"].isin(deal_contact_ids) | _bool_series(prepared, "has_open_deal") | _bool_series(prepared, "has_won_deal") | _bool_series(prepared, "has_lost_deal")
    enrolled = prepared["is_enrolled_pipeline"] | prepared["contact_id"].isin(won_contact_ids)
    closed_lost_dead = prepared["is_dead_pipeline"]
    milestones = [
        ("Lead created", pd.Series([True] * len(prepared), index=prepared.index)),
        ("Contacted", contacted),
        ("Application / deal created", has_application_or_deal),
        ("Enrolled / closed won", enrolled),
        ("Closed lost / dead", closed_lost_dead),
    ]
    rows = []
    for milestone, mask in milestones:
        group = prepared[mask.fillna(False)]
        rows.append(
            {
                "milestone": milestone,
                "leads": int(_contact_ids(group).nunique()),
                "paid leads": int(_contact_ids(group[_bool_series(group, "paid_lead_flag")]).nunique()),
                "potential revenue": float(group["pipeline_potential_revenue"].sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def cleanup_needed(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_pipeline_contacts(contacts, fact)
    columns = ["cleanup issue", "leads", "paid leads", "potential revenue", "recommended action"]
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    status_missing = prepared["formal_status_missing"]
    status_text = prepared["pipeline_status"].fillna("").astype(str).str.lower()
    invalid_status = status_text.str.contains("invalid|junk|duplicate|test|spam", regex=True, na=False)
    age = _num_series(prepared, "lead_age_days").fillna(0)
    inactive = _num_series(prepared, "days_since_last_activity").fillna(9999)
    open_mask = ~prepared["is_enrolled_pipeline"] & ~prepared["is_dead_pipeline"]
    old_open = open_mask & (age >= 30)
    owner_missing = prepared.get("salesman_name", pd.Series("", index=prepared.index)).fillna("").astype(str).str.strip().isin(["", "Unassigned"])
    inactive_owner_open = open_mask & owner_missing
    no_next_activity = open_mask & ~_has_text_series(prepared, "next_activity_date")
    issues = [
        ("missing status", status_missing, "Populate lead/final/enrollment status."),
        ("invalid status", invalid_status, "Clean invalid/test/duplicate statuses."),
        ("old open leads", old_open, "Review open leads older than 30 days."),
        ("inactive owner with open leads", inactive_owner_open, "Assign an active owner."),
        ("no next activity", no_next_activity | (open_mask & (inactive >= 14)), "Set next activity or close the lead."),
    ]
    rows = []
    for issue, mask, action in issues:
        group = prepared[mask.fillna(False)]
        rows.append(
            {
                "cleanup issue": issue,
                "leads": int(_contact_ids(group).nunique()),
                "paid leads": int(_contact_ids(group[_bool_series(group, "paid_lead_flag")]).nunique()),
                "potential revenue": float(group["pipeline_potential_revenue"].sum()),
                "recommended action": action,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("leads", ascending=False)


def render_pipeline_health_page(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Pipeline Health")
    _section_intro("Pipeline Health")

    prepared = prepare_pipeline_contacts(contacts, fact)

    summary = lead_status_summary(prepared, fact)
    st.subheader("Lead Status Summary")
    _section_intro("Lead Status Summary")
    if summary.empty:
        st.warning("No lead status data is available.")
    else:
        _data_table(summary)
        chart = summary.head(15)
        st.plotly_chart(
            px.bar(chart, x="status", y="total leads", title="Leads by meaningful status"),
            width="stretch",
        )

    bottlenecks = stage_bottlenecks(prepared, fact)
    st.subheader("Stage Bottlenecks")
    _section_intro("Stage Bottlenecks")
    if bottlenecks.empty:
        st.warning("No bottleneck data is available.")
    else:
        _data_table(bottlenecks)
        chart = bottlenecks[bottlenecks["leads stuck 14+ days"] > 0].head(15)
        if not chart.empty:
            st.plotly_chart(
                px.bar(chart, x="stage/status", y="leads stuck 14+ days", title="Leads stuck 14+ days"),
                width="stretch",
            )

    path = enrollment_path(prepared, fact)
    st.subheader("Enrollment Path")
    _section_intro("Enrollment Path")
    if path.empty:
        st.warning("Enrollment path data is unavailable.")
    else:
        _data_table(path)
        st.plotly_chart(px.bar(path, x="milestone", y="leads", title="Enrollment path counts"), width="stretch")

    cleanup = cleanup_needed(prepared, fact)
    st.subheader("Cleanup Needed")
    _section_intro("Cleanup Needed")
    if cleanup.empty:
        st.warning("Cleanup data is unavailable.")
    else:
        _data_table(cleanup)
        st.plotly_chart(px.bar(cleanup, x="cleanup issue", y="leads", title="Records needing cleanup"), width="stretch")
