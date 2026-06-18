from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio
import streamlit as st

from modules.budget_loader import BudgetData, load_budget
from modules.charts import (
    CALMU_COLORS,
    bar_chart,
    configure_plotly_theme,
    donut_chart,
    enrollment_progress,
    funnel_chart,
    line_chart,
)
from modules.data_loader import UploadedLeadData, load_uploaded_lead_data, redact_pii
from modules.enrollment_tracker import EnrollmentTrackerData, load_enrollment_tracker
from modules.hubspot_client import HubSpotFetchResult, fetch_hubspot_contacts, get_access_token
from modules.metrics import (
    compare_pivot_totals,
    enrollment_by,
    enrollment_summary,
    filter_date_range,
    funnel_counts,
    lead_performance_by,
    roundup_value,
    source_performance,
    summarize_leads,
    weekly_counts,
)
from modules.qa_checks import run_qa_checks


APP_DIR = Path(__file__).resolve().parent
pio.templates.default = "plotly_white"


def apply_brand_theme() -> None:
    configure_plotly_theme()
    st.markdown(
        f"""
        <style>
        :root {{
            --calmu-blue: {CALMU_COLORS["blue"]};
            --calmu-lime: {CALMU_COLORS["lime"]};
            --calmu-navy: {CALMU_COLORS["navy"]};
            --calmu-royal: {CALMU_COLORS["royal"]};
            --calmu-sky: {CALMU_COLORS["sky"]};
            --calmu-green: {CALMU_COLORS["green"]};
            --calmu-mist: {CALMU_COLORS["mist"]};
            --calmu-slate: {CALMU_COLORS["slate"]};
        }}
        .stApp {{
            background: linear-gradient(180deg, rgba(171, 204, 227, 0.22), rgba(255,255,255,0) 280px), #FFFFFF;
            color: var(--calmu-navy);
        }}
        .block-container {{
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }}
        [data-testid="stSidebar"] {{
            background: var(--calmu-navy);
        }}
        [data-testid="stSidebar"] * {{
            color: #FFFFFF;
        }}
        [data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid #DFE8EE;
            border-top: 4px solid var(--calmu-blue);
            border-radius: 8px;
            padding: 13px 15px 11px;
            min-height: 112px;
            box-shadow: 0 12px 24px rgba(30, 41, 68, 0.06);
        }}
        div[data-testid="stMetricValue"] {{
            color: var(--calmu-navy);
            font-weight: 800;
        }}
        .calmu-masthead {{
            background: linear-gradient(135deg, var(--calmu-navy), var(--calmu-green));
            border-bottom: 6px solid var(--calmu-lime);
            border-radius: 8px;
            color: #FFFFFF;
            padding: 24px 28px;
            margin-bottom: 20px;
        }}
        .calmu-masthead h1 {{
            color: #FFFFFF;
            font-size: 2.35rem;
            line-height: 1.05;
            margin: 0;
            letter-spacing: 0;
        }}
        .calmu-kicker {{
            color: var(--calmu-lime);
            font-weight: 800;
            font-size: 0.78rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .calmu-masthead p {{
            color: rgba(255,255,255,0.84);
            margin: 10px 0 0;
            max-width: 860px;
        }}
        h1, h2, h3 {{
            color: var(--calmu-navy);
            letter-spacing: 0;
        }}
        .dataframe tbody tr th {{
            display: none;
        }}
        .dataframe thead tr th:first-child {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_static_inputs() -> tuple[UploadedLeadData, EnrollmentTrackerData, BudgetData]:
    return load_uploaded_lead_data(), load_enrollment_tracker(), load_budget()


@st.cache_data(ttl=900, show_spinner="Fetching read-only HubSpot contacts...")
def load_live_hubspot(refresh_key: int) -> HubSpotFetchResult:
    _ = refresh_key
    return fetch_hubspot_contacts()


def fmt_int(value: Any) -> str:
    return f"{int(round(float(value or 0))):,}"


def fmt_float(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "0.0"


def fmt_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "0.0%"


def fmt_money(value: Any) -> str:
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def metric_grid(items: list[tuple[str, str]], columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[start : start + columns]):
            col.metric(item[0], item[1])


def nonblank_options(*frames: pd.DataFrame, column: str) -> list[str]:
    values: set[str] = set()
    for frame in frames:
        if not frame.empty and column in frame.columns:
            values.update(
                frame[column]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .tolist()
            )
    return sorted(values)


def filter_values(df: pd.DataFrame, column: str, values: list[str]) -> pd.DataFrame:
    if df.empty or not values or column not in df.columns:
        return df
    return df[df[column].fillna("").astype(str).isin(values)].copy()


def build_filters(leads: pd.DataFrame, paid_leads: pd.DataFrame, enrollments: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Filters")
    all_dates: list[pd.Timestamp] = []
    for frame, column in [(leads, "create_date"), (paid_leads, "create_date"), (enrollments, "enrolled_date")]:
        if not frame.empty and column in frame.columns:
            parsed_dates = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            all_dates.extend(parsed_dates.dt.date.tolist())
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        selected_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = selected_range if isinstance(selected_range, tuple) and len(selected_range) == 2 else (None, None)
    else:
        start_date, end_date = None, None

    term = st.sidebar.multiselect("Term", nonblank_options(enrollments, column="term_label"))
    source = st.sidebar.multiselect("Source", nonblank_options(leads, paid_leads, enrollments, column="normalized_source"))
    lead_type = st.sidebar.multiselect("Paid/organic", ["Paid", "Organic", "Unknown"])
    udr = st.sidebar.multiselect("UDR / Contact owner", nonblank_options(leads, enrollments, column="contact_owner") + nonblank_options(enrollments, column="udr"))
    program = st.sidebar.multiselect("Program / Degree", nonblank_options(leads, column="degree") + nonblank_options(enrollments, column="program"))
    modality = st.sidebar.multiselect("Modality", nonblank_options(enrollments, column="modality"))
    student_type = st.sidebar.multiselect("Student type", nonblank_options(leads, paid_leads, enrollments, column="student_type"))
    payment = st.sidebar.multiselect("Payment / funding", nonblank_options(enrollments, column="payment_funding"))
    campus = st.sidebar.multiselect("Campus location", nonblank_options(leads, paid_leads, column="campus_location"))
    lead_status = st.sidebar.multiselect("Lead status", nonblank_options(leads, paid_leads, column="lead_status"))
    lifecycle = st.sidebar.multiselect("Lifecycle stage", nonblank_options(leads, paid_leads, column="lifecycle_stage"))
    return {
        "start_date": start_date,
        "end_date": end_date,
        "term": term,
        "source": source,
        "lead_type": lead_type,
        "udr": sorted(set(udr)),
        "program": sorted(set(program)),
        "modality": modality,
        "student_type": student_type,
        "payment": payment,
        "campus": campus,
        "lead_status": lead_status,
        "lifecycle": lifecycle,
    }


def apply_lead_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = filter_date_range(df, "create_date", filters["start_date"], filters["end_date"])
    out = filter_values(out, "normalized_source", filters["source"])
    out = filter_values(out, "lead_type", filters["lead_type"])
    out = filter_values(out, "contact_owner", filters["udr"])
    out = filter_values(out, "degree", filters["program"])
    out = filter_values(out, "student_type", filters["student_type"])
    out = filter_values(out, "campus_location", filters["campus"])
    out = filter_values(out, "lead_status", filters["lead_status"])
    out = filter_values(out, "lifecycle_stage", filters["lifecycle"])
    return out


def apply_enrollment_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    out = filter_date_range(df, "enrolled_date", filters["start_date"], filters["end_date"])
    out = filter_values(out, "term_label", filters["term"])
    out = filter_values(out, "normalized_source", filters["source"])
    out = filter_values(out, "lead_type", filters["lead_type"])
    out = filter_values(out, "udr", filters["udr"])
    out = filter_values(out, "program", filters["program"])
    out = filter_values(out, "modality", filters["modality"])
    out = filter_values(out, "student_type", filters["student_type"])
    out = filter_values(out, "payment_funding", filters["payment"])
    return out


def first_metric(summary: pd.DataFrame, metric: str) -> float | None:
    return roundup_value(summary, metric)


def required_weekly_pace(enrollments: pd.DataFrame, remaining: float) -> float:
    if remaining <= 0:
        return 0.0
    future_terms = pd.to_datetime(enrollments.get("term"), errors="coerce").dropna()
    if future_terms.empty:
        return 0.0
    target = future_terms.max().date()
    days = max((target - date.today()).days, 1)
    return float(remaining) / max(days / 7, 1)


def pivot_reconciliation(uploaded: UploadedLeadData) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if "PLS" in uploaded.paid_pivots:
        frames.append(compare_pivot_totals(uploaded.paid_leads, uploaded.paid_pivots["PLS"], "paid_lead_list", "Paid PLS"))
    if "PLC" in uploaded.paid_pivots:
        frames.append(compare_pivot_totals(uploaded.paid_leads, uploaded.paid_pivots["PLC"], "paid_lead_list", "Paid PLC"))
    if "L2C" in uploaded.udr_pivots:
        frames.append(compare_pivot_totals(uploaded.udr_leads, uploaded.udr_pivots["L2C"], "contact_owner", "UDR L2C"))
    if "L2A" in uploaded.udr_pivots:
        frames.append(compare_pivot_totals(uploaded.udr_leads, uploaded.udr_pivots["L2A"], "contact_owner", "UDR L2A"))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def page_executive(
    leads: pd.DataFrame,
    paid_leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    budget: BudgetData,
    tracker: EnrollmentTrackerData,
    data_mode: str,
) -> None:
    lead_summary = summarize_leads(leads)
    paid_summary = summarize_leads(paid_leads)
    goal = first_metric(tracker.roundup_summary, "Budget") or first_metric(budget.summary, "Budget") or 0
    starts = first_metric(tracker.roundup_summary, "Starts") or first_metric(budget.summary, "Starts") or 0
    enroll_summary = enrollment_summary(enrollments, goal=goal, starts=starts)
    remaining = enroll_summary["remaining_enrollments"]
    pace = required_weekly_pace(enrollments, remaining)
    total_budget = goal
    budget_remaining = max(float(total_budget or 0) - enroll_summary["actual_enrollments"], 0)

    st.subheader("Confirmed Data")
    metric_grid(
        [
            ("Total HubSpot leads", fmt_int(lead_summary.total_leads)),
            ("Paid leads", fmt_int(paid_summary.total_leads or lead_summary.paid_leads)),
            ("Organic leads", fmt_int(lead_summary.organic_leads or paid_summary.organic_leads)),
            ("Applicants", fmt_int(lead_summary.applicants)),
            ("CRM enrolled", fmt_int(lead_summary.crm_enrolled)),
            ("Actual enrollments", fmt_int(enroll_summary["actual_enrollments"])),
            ("Enrollment goal", fmt_int(enroll_summary["enrollment_goal"])),
            ("Starts", fmt_int(enroll_summary["starts"])),
            ("Projected / actual revenue", fmt_money(enroll_summary["revenue"])),
            ("Bad leads", fmt_int(lead_summary.bad_leads)),
            ("Total budget", fmt_int(total_budget)),
            ("Budget remaining", fmt_int(budget_remaining)),
        ]
    )

    st.subheader("Calculated Interpretation")
    metric_grid(
        [
            ("Contacted / progressed leads", fmt_int(lead_summary.contacted_leads)),
            ("Lead-to-contact %", fmt_percent(lead_summary.lead_to_contact_rate)),
            ("Lead-to-applicant %", fmt_percent(lead_summary.lead_to_applicant_rate)),
            ("Lead-to-CRM-enrolled %", fmt_percent(lead_summary.lead_to_crm_enrolled_rate)),
            ("Percent of goal", fmt_percent(enroll_summary["percent_of_goal"])),
            ("Remaining enrollments needed", fmt_int(remaining)),
            ("Weekly pace required", fmt_float(pace, 1)),
            ("Start %", fmt_percent(enroll_summary["start_rate"])),
            ("Revenue per enrollment", fmt_money(enroll_summary["revenue_per_enrollment"])),
            ("Average days to enroll", fmt_float(enroll_summary["average_days_to_enroll"], 1)),
            ("Bad lead rate", fmt_percent(lead_summary.bad_lead_rate)),
            ("Cost / budget per lead", fmt_float(total_budget / lead_summary.total_leads if lead_summary.total_leads else 0, 2)),
            ("Cost / budget per applicant", fmt_float(total_budget / lead_summary.applicants if lead_summary.applicants else 0, 2)),
            (
                "Cost / budget per enrollment",
                fmt_float(total_budget / enroll_summary["actual_enrollments"] if enroll_summary["actual_enrollments"] else 0, 2),
            ),
        ]
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(enrollment_progress(enroll_summary["actual_enrollments"], goal), use_container_width=True)
    with col2:
        st.plotly_chart(funnel_chart(funnel_counts(leads, enroll_summary["actual_enrollments"])), use_container_width=True)

    st.caption(data_mode)


def page_enrollment_tracker(enrollments: pd.DataFrame, tracker: EnrollmentTrackerData) -> None:
    st.subheader("Enrollment Tracker")
    summary = enrollment_summary(enrollments, goal=first_metric(tracker.roundup_summary, "Budget"), starts=first_metric(tracker.roundup_summary, "Starts"))
    metric_grid(
        [
            ("Total actual enrollments", fmt_int(summary["actual_enrollments"])),
            ("Starts", fmt_int(summary["starts"])),
            ("Start %", fmt_percent(summary["start_rate"])),
            ("Revenue", fmt_money(summary["revenue"])),
            ("Revenue per enrollment", fmt_money(summary["revenue_per_enrollment"])),
            ("Average days to enroll", fmt_float(summary["average_days_to_enroll"], 1)),
        ],
        columns=3,
    )
    st.plotly_chart(line_chart(weekly_counts(enrollments, "enrolled_date", "enrollments"), "week", "enrollments", "Weekly Enrollment Pace"), use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(bar_chart(enrollment_by(enrollments, "udr"), "udr", "actual_enrollments", "Enrollments by UDR", horizontal=True), use_container_width=True)
        st.plotly_chart(donut_chart(enrollment_by(enrollments, "student_type"), "student_type", "actual_enrollments", "Student Type Mix"), use_container_width=True)
    with cols[1]:
        st.plotly_chart(bar_chart(enrollment_by(enrollments, "normalized_source"), "normalized_source", "actual_enrollments", "Enrollments by Source", horizontal=True), use_container_width=True)
        st.plotly_chart(donut_chart(enrollment_by(enrollments, "payment_funding"), "payment_funding", "actual_enrollments", "Payment / Funding Mix"), use_container_width=True)
    st.dataframe(redact_pii(enrollment_by(enrollments, "program")), use_container_width=True, hide_index=True)


def page_source_performance(leads: pd.DataFrame, paid_leads: pd.DataFrame, enrollments: pd.DataFrame, budget: BudgetData) -> None:
    st.subheader("Source Performance")
    perf = source_performance(leads, enrollments, budget.allocations)
    paid_perf = lead_performance_by(paid_leads, "normalized_source")
    tabs = st.tabs(["All Sources", "Paid / Organic Lists"])
    with tabs[0]:
        st.plotly_chart(bar_chart(perf, "normalized_source", "leads", "Lead Volume by Source", horizontal=True), use_container_width=True)
        st.plotly_chart(bar_chart(perf, "normalized_source", "actual_enrollments", "Actual Enrollments by Source", horizontal=True), use_container_width=True)
        st.dataframe(perf, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.plotly_chart(bar_chart(paid_perf, "normalized_source", "leads", "Paid and Organic Lead List Performance", horizontal=True), use_container_width=True)
        st.dataframe(paid_perf, use_container_width=True, hide_index=True)


def page_udr_performance(leads: pd.DataFrame, enrollments: pd.DataFrame) -> None:
    st.subheader("UDR Performance")
    perf = lead_performance_by(leads, "contact_owner")
    enroll_perf = enrollment_by(enrollments, "udr")
    if not perf.empty and not enroll_perf.empty:
        perf = perf.merge(enroll_perf.rename(columns={"udr": "contact_owner"}), on="contact_owner", how="outer")
    st.plotly_chart(bar_chart(perf, "contact_owner", "leads", "UDR Lead Volume", horizontal=True), use_container_width=True)
    st.plotly_chart(bar_chart(perf, "contact_owner", "l2c", "UDR Lead-to-Contact %", horizontal=True), use_container_width=True)
    st.dataframe(perf, use_container_width=True, hide_index=True)


def page_program_mix(enrollments: pd.DataFrame) -> None:
    st.subheader("Program Mix")
    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(bar_chart(enrollment_by(enrollments, "program"), "program", "actual_enrollments", "Program Mix", horizontal=True), use_container_width=True)
        st.plotly_chart(donut_chart(enrollment_by(enrollments, "modality"), "modality", "actual_enrollments", "Modality Mix"), use_container_width=True)
    with cols[1]:
        st.plotly_chart(donut_chart(enrollment_by(enrollments, "new_roll"), "new_roll", "actual_enrollments", "New vs Roll / Repeat"), use_container_width=True)
        st.plotly_chart(donut_chart(enrollment_by(enrollments, "payment_funding"), "payment_funding", "actual_enrollments", "Payment / Funding Mix"), use_container_width=True)


def page_budget_performance(leads: pd.DataFrame, enrollments: pd.DataFrame, budget: BudgetData) -> None:
    st.subheader("Budget Performance")
    source_perf = source_performance(leads, enrollments, budget.allocations)
    total_budget = float(pd.to_numeric(budget.summary.loc[budget.summary["metric"].str.lower().eq("budget"), "numeric_value"], errors="coerce").fillna(0).sum()) if not budget.summary.empty else 0
    actual = len(enrollments)
    metric_grid(
        [
            ("Total budget", fmt_int(total_budget)),
            ("Actual enrollments", fmt_int(actual)),
            ("Budget variance", fmt_percent(actual / total_budget if total_budget else 0)),
            ("Cost / budget per lead", fmt_float(total_budget / len(leads) if len(leads) else 0, 2)),
            ("Cost / budget per applicant", fmt_float(total_budget / summarize_leads(leads).applicants if summarize_leads(leads).applicants else 0, 2)),
            ("Cost / budget per enrollment", fmt_float(total_budget / actual if actual else 0, 2)),
        ],
        columns=3,
    )
    st.plotly_chart(bar_chart(budget.allocations, "budget_name", "planned_budget", "Budget by UDR / Allocation", horizontal=True), use_container_width=True)
    st.plotly_chart(bar_chart(source_perf, "normalized_source", "cost_per_enrollment", "Cost / Budget per Actual Enrollment by Source", horizontal=True), use_container_width=True)
    st.dataframe(source_perf, use_container_width=True, hide_index=True)


def page_funnel(leads: pd.DataFrame, enrollments: pd.DataFrame) -> None:
    st.subheader("Lead Status & Lifecycle Funnel")
    summary = summarize_leads(leads)
    metric_grid(
        [
            ("Total leads", fmt_int(summary.total_leads)),
            ("Contacted / progressed", fmt_int(summary.contacted_leads)),
            ("Applicants", fmt_int(summary.applicants)),
            ("CRM enrolled", fmt_int(summary.crm_enrolled)),
            ("Bad leads", fmt_int(summary.bad_leads)),
            ("Uncontacted leads", fmt_int(summary.uncontacted_leads)),
        ],
        columns=3,
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(funnel_chart(funnel_counts(leads, len(enrollments))), use_container_width=True)
    with col2:
        lifecycle = lead_performance_by(leads, "lifecycle_stage")
        st.plotly_chart(bar_chart(lifecycle, "lifecycle_stage", "leads", "Lifecycle Stage Volume", horizontal=True), use_container_width=True)
    status = lead_performance_by(leads, "lead_status")
    st.plotly_chart(bar_chart(status, "lead_status", "leads", "Lead Status Volume", horizontal=True), use_container_width=True)


def page_trends(leads: pd.DataFrame, enrollments: pd.DataFrame) -> None:
    st.subheader("Trends")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(line_chart(weekly_counts(leads, "create_date", "leads"), "week", "leads", "Weekly Lead Trend"), use_container_width=True)
    with col2:
        st.plotly_chart(line_chart(weekly_counts(enrollments, "enrolled_date", "enrollments"), "week", "enrollments", "Weekly Enrollment Trend"), use_container_width=True)


def page_qa(qa: pd.DataFrame, pivot_diffs: pd.DataFrame) -> None:
    st.subheader("QA Checks")
    st.dataframe(qa, use_container_width=True, hide_index=True)
    if not pivot_diffs.empty:
        st.subheader("Pivot Reconciliation")
        st.dataframe(pivot_diffs, use_container_width=True, hide_index=True)


def page_notes(uploaded: UploadedLeadData, tracker: EnrollmentTrackerData, budget: BudgetData, hubspot: HubSpotFetchResult, data_mode: str) -> None:
    st.subheader("Assumptions / Data Notes")
    notes = [
        {"area": "Data mode", "note": data_mode},
        {"area": "HubSpot", "note": "Read-only contact fetch. No record edits, emails, deletes, or writes are implemented."},
        {"area": "Ownership", "note": "Contact Owner is used as UDR. Deal Owner is not used unless a future deal dataset is added as a fallback."},
        {"area": "Budget", "note": "The uploaded budget workbook contains enrollment goal/allocation fields, not a confirmed media spend ledger."},
        {"area": "Privacy", "note": "Executive pages suppress names, emails, phone numbers, and tracker notes."},
        {"area": "Brand", "note": "Theme uses CalMU guideline colors: #2938D5, #3D59D9, #1E2944, #EDFF81, #ABCCE3, #CD1141, #8F0028, #1A5347, #ABBEB3, #657874, #C4CDD3."},
    ]
    notes.extend({"area": "Uploaded file", "note": note} for note in uploaded.load_notes)
    notes.extend({"area": "Enrollment tracker", "note": note} for note in tracker.notes)
    notes.extend({"area": "Budget workbook", "note": note} for note in budget.notes)
    st.dataframe(pd.DataFrame(notes), use_container_width=True, hide_index=True)

    st.subheader("Weekly Summary Source")
    email_rows = [
        {"field": "Subject", "value": uploaded.email_context.subject},
        {"field": "Sent at", "value": uploaded.email_context.sent_at},
        {"field": "Embedded images", "value": str(uploaded.email_context.image_count)},
    ]
    st.dataframe(pd.DataFrame(email_rows), use_container_width=True, hide_index=True)
    if uploaded.email_context.data_lines:
        st.dataframe(pd.DataFrame({"extracted_line": uploaded.email_context.data_lines}), use_container_width=True, hide_index=True)
    else:
        st.info("No usable weekly summary data was found in parsed email text or practical OCR output.")

    if hubspot.token_present:
        st.subheader("HubSpot Fields")
        st.dataframe(
            pd.DataFrame(
                {
                    "used_properties": pd.Series(hubspot.used_properties),
                    "missing_configured_properties": pd.Series(hubspot.missing_properties[: len(hubspot.used_properties) or None]),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def raw_access_allowed() -> bool:
    expected = os.getenv("RAW_AUDIT_PASSWORD", "")
    try:
        expected = st.secrets.get("RAW_AUDIT_PASSWORD", expected)
    except Exception:
        pass
    expected = str(expected or "").strip()
    if expected:
        entered = st.text_input("Raw audit password", type="password")
        return entered == expected
    return st.checkbox("Show protected raw audit data for this local session")


def page_raw_audit(uploaded: UploadedLeadData, leads: pd.DataFrame, enrollments: pd.DataFrame) -> None:
    st.subheader("Raw Audit Data")
    protected = raw_access_allowed()
    frames = {
        "Analysis leads": leads,
        "Uploaded paid leads": uploaded.paid_leads,
        "Uploaded UDR leads": uploaded.udr_leads,
        "Enrollment tracker": enrollments,
    }
    for label, frame in frames.items():
        st.markdown(f"#### {label}")
        display = frame if protected else redact_pii(frame)
        st.dataframe(display.head(2000), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="CalMU Enrollment Performance", layout="wide", page_icon="CMU")
    apply_brand_theme()

    uploaded, tracker, budget = load_static_inputs()

    if "hubspot_refresh_key" not in st.session_state:
        st.session_state["hubspot_refresh_key"] = 0

    st.sidebar.title("CalMU")
    refresh_clicked = st.sidebar.button("Refresh data", use_container_width=True)
    if refresh_clicked:
        st.session_state["hubspot_refresh_key"] += 1

    token_present = get_access_token() is not None
    hubspot = load_live_hubspot(st.session_state["hubspot_refresh_key"]) if token_present else HubSpotFetchResult(
        contacts=pd.DataFrame(),
        properties=pd.DataFrame(),
        owners=pd.DataFrame(),
        used_properties=[],
        missing_properties=[],
        fetched_at=None,
        error=None,
        token_present=False,
    )

    if not hubspot.contacts.empty:
        analysis_leads = hubspot.contacts.copy()
        data_mode = "Live HubSpot contacts are active. Uploaded files remain baseline/reference sources."
    else:
        analysis_leads = uploaded.udr_leads.copy()
        if hubspot.error:
            data_mode = f"HubSpot fetch failed; static uploaded baseline is active. Error: {hubspot.error}"
            st.warning(data_mode)
        elif not token_present:
            data_mode = "No HubSpot token found; static uploaded baseline is active."
            st.info(data_mode)
        else:
            data_mode = "HubSpot returned no contacts; static uploaded baseline is active."

    last_refresh = hubspot.fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if hubspot.fetched_at else "Not refreshed"
    st.sidebar.caption(f"Last HubSpot refresh: {last_refresh}")

    filters = build_filters(analysis_leads, uploaded.paid_leads, tracker.enrollments)
    filtered_leads = apply_lead_filters(analysis_leads, filters)
    filtered_paid = apply_lead_filters(uploaded.paid_leads, filters)
    filtered_enrollments = apply_enrollment_filters(tracker.enrollments, filters)

    pivot_diffs = pivot_reconciliation(uploaded)
    hubspot_state = {
        "token_present": token_present,
        "error": hubspot.error,
        "fetched_at": hubspot.fetched_at,
    }
    qa = run_qa_checks(filtered_leads, filtered_paid, uploaded.udr_leads, filtered_enrollments, budget.allocations, pivot_diffs, hubspot_state)

    st.markdown(
        """
        <div class="calmu-masthead">
            <div class="calmu-kicker">California Miramar University</div>
            <h1>Enrollment, Lead, UDR, Source, and Budget Performance</h1>
            <p>Confirmed uploaded tracker/workbook data, live HubSpot contacts when available, and clearly separated calculated interpretations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pages = [
        "Executive Overview",
        "Enrollment Tracker",
        "Source Performance",
        "UDR Performance",
        "Program Mix",
        "Budget Performance",
        "Lead Status & Lifecycle Funnel",
        "Trends",
        "QA Checks",
        "Assumptions / Data Notes",
        "Raw Audit Data",
    ]
    page = st.sidebar.radio("Dashboard section", pages)

    if page == "Executive Overview":
        page_executive(filtered_leads, filtered_paid, filtered_enrollments, budget, tracker, data_mode)
    elif page == "Enrollment Tracker":
        page_enrollment_tracker(filtered_enrollments, tracker)
    elif page == "Source Performance":
        page_source_performance(filtered_leads, filtered_paid, filtered_enrollments, budget)
    elif page == "UDR Performance":
        page_udr_performance(filtered_leads, filtered_enrollments)
    elif page == "Program Mix":
        page_program_mix(filtered_enrollments)
    elif page == "Budget Performance":
        page_budget_performance(filtered_leads, filtered_enrollments, budget)
    elif page == "Lead Status & Lifecycle Funnel":
        page_funnel(filtered_leads, filtered_enrollments)
    elif page == "Trends":
        page_trends(filtered_leads, filtered_enrollments)
    elif page == "QA Checks":
        page_qa(qa, pivot_diffs)
    elif page == "Assumptions / Data Notes":
        page_notes(uploaded, tracker, budget, hubspot, data_mode)
    else:
        page_raw_audit(uploaded, filtered_leads, filtered_enrollments)


if __name__ == "__main__":
    main()
