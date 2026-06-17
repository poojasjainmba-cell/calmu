from __future__ import annotations

from datetime import date, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.budget_loader import BudgetData, load_budget
from modules.charts import CALMU_COLORS, bar_chart, configure_plotly_theme, enrollment_progress, line_chart
from modules.data_loader import UploadedLeadData, load_uploaded_lead_data, redact_pii
from modules.enrollment_ops import (
    BENCHMARKS,
    DISPLAY_LABELS,
    activity_summary_by_udr,
    apply_global_filters,
    canonicalize_udr_columns,
    canonicalize_udr_goals,
    display_frame,
    display_label,
    enrollment_group,
    enrollment_metrics,
    expected_goal_pct,
    funnel_metrics_by,
    goals_by_udr,
    has_activity_fields,
    no_recent_activity_mask,
    normalize_enrollments,
    normalize_leads,
    qa_summary,
    safe_divide,
    total_goal,
    udr_scorecard,
    vendor_performance,
)
from modules.enrollment_tracker import EnrollmentTrackerData, load_enrollment_tracker
from modules.hubspot_client import HubSpotFetchResult, fetch_hubspot_contacts, get_access_token
from modules.metrics import weekly_counts


def apply_brand_theme() -> None:
    configure_plotly_theme()
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
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
        html, body, [class*="css"] {{
            font-family: "Inter", "Proxima Nova", Arial, sans-serif;
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
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {{
            color: var(--calmu-navy);
            font-size: clamp(1.45rem, 2vw, 2.15rem);
            font-weight: 800;
            line-height: 1.05;
            white-space: normal;
            overflow-wrap: anywhere;
        }}
        .calmu-masthead {{
            background: linear-gradient(135deg, var(--calmu-navy), var(--calmu-green));
            border-bottom: 6px solid var(--calmu-lime);
            border-radius: 8px;
            color: #FFFFFF;
            padding: 24px 28px;
            margin-bottom: 18px;
        }}
        .calmu-masthead h1 {{
            color: #FFFFFF;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.02;
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
            max-width: 920px;
        }}
        .filter-panel {{
            background: #FFFFFF;
            border: 1px solid #DFE8EE;
            border-radius: 8px;
            padding: 14px 16px 6px;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(30, 41, 68, 0.05);
        }}
        h1, h2, h3 {{
            color: var(--calmu-navy);
            letter-spacing: 0;
        }}
        .stButton > button, .stDownloadButton > button {{
            background: var(--calmu-lime);
            color: var(--calmu-navy);
            border: 1px solid rgba(30, 41, 68, 0.16);
            border-radius: 8px;
            font-weight: 800;
        }}
        div[data-testid="stPlotlyChart"] {{
            background: #FFFFFF;
            border: 1px solid #E1E9EF;
            border-radius: 8px;
            padding: 8px;
            box-shadow: 0 12px 30px rgba(30, 41, 68, 0.05);
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid #E1E9EF;
            border-radius: 8px;
            overflow: hidden;
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


@st.cache_data(show_spinner=False)
def normalize_cached(leads: pd.DataFrame, enrollments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return normalize_leads(leads), normalize_enrollments(enrollments)


@st.cache_data(show_spinner=False)
def cached_metrics(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    goal: float,
    start: date | None,
    end: date | None,
) -> dict[str, float]:
    return enrollment_metrics(leads, enrollments, goal, start, end)


@st.cache_data(show_spinner=False)
def cached_udr_scorecard(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    goals: dict[str, float],
    expected_pct: float,
    no_recent_days: int,
) -> pd.DataFrame:
    return udr_scorecard(leads, enrollments, goals, expected_pct, BENCHMARKS, no_recent_days)


@st.cache_data(show_spinner=False)
def cached_vendor_performance(leads: pd.DataFrame, enrollments: pd.DataFrame, no_recent_days: int) -> pd.DataFrame:
    return vendor_performance(leads, enrollments, no_recent_days)


def fmt_int(value: Any) -> str:
    try:
        return f"{int(round(float(value or 0))):,}"
    except Exception:
        return "0"


def fmt_percent(value: Any) -> str:
    try:
        return f"{float(value or 0) * 100:.1f}%"
    except Exception:
        return "0.0%"


def fmt_number(value: Any, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "Not Available"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "Not Available"


def metric_grid(items: list[tuple[str, str]], columns: int = 4) -> None:
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value) in zip(cols, items[start : start + columns]):
            col.metric(label, value)


def _option_values(*frames: pd.DataFrame, column: str) -> list[str]:
    values: set[str] = set()
    for frame in frames:
        if not frame.empty and column in frame.columns:
            values.update(value for value in frame[column].dropna().astype(str).str.strip() if value)
    return sorted(values)


def _budget_udr_names(*frames: pd.DataFrame) -> list[str]:
    names: set[str] = set()
    for frame in frames:
        if frame.empty:
            continue
        for column in ["budget_name", "source", "assigned_udr", "enrollment_udr"]:
            if column in frame.columns:
                names.update(value for value in frame[column].dropna().astype(str).str.strip() if value)
    return sorted(names)


def _date_bounds(leads: pd.DataFrame, enrollments: pd.DataFrame) -> tuple[date | None, date | None]:
    dates: list[date] = []
    for frame, column in [(leads, "create_date"), (enrollments, "enrolled_date")]:
        if not frame.empty and column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
            dates.extend(parsed.dt.date.tolist())
    return (min(dates), max(dates)) if dates else (None, None)


def build_filter_bar(leads: pd.DataFrame, enrollments: pd.DataFrame) -> dict[str, Any]:
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    min_date, max_date = _date_bounds(leads, enrollments)
    row1 = st.columns([1.2, 1.1, 1, 1, 1])
    term = row1[0].multiselect("Term", _option_values(enrollments, column="term"))
    if min_date and max_date:
        selected_range = row1[1].date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = selected_range if isinstance(selected_range, tuple) and len(selected_range) == 2 else (None, None)
        date_range_default = start_date == min_date and end_date == max_date
    else:
        start_date, end_date = None, None
        date_range_default = True
        row1[1].caption("Date Range: Not Available")
    udr = row1[2].multiselect("UDR", sorted(set(_option_values(leads, column="assigned_udr") + _option_values(enrollments, column="enrollment_udr"))))
    program = row1[3].multiselect("Program", _option_values(leads, enrollments, column="program"))
    degree = row1[4].multiselect("Degree", _option_values(leads, enrollments, column="degree"))

    row2 = st.columns([1, 1, 1, 1])
    vendor = row2[0].multiselect("Vendor", _option_values(leads, enrollments, column="vendor"))
    modality = row2[1].multiselect("Modality", _option_values(leads, enrollments, column="modality"))
    student_type = row2[2].multiselect("Student Type", _option_values(leads, enrollments, column="student_type"))
    lead_type_choice = row2[3].selectbox("Lead Type", ["All", "Paid", "Organic"])
    no_recent_days = st.number_input("No Recent Activity Threshold (days)", min_value=1, max_value=90, value=7, step=1)
    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "term": term,
        "start_date": start_date,
        "end_date": end_date,
        "date_range_default": date_range_default,
        "udr": udr,
        "program": program,
        "degree": degree,
        "vendor": vendor,
        "modality": modality,
        "student_type": student_type,
        "lead_type": [] if lead_type_choice == "All" else [lead_type_choice],
        "no_recent_days": int(no_recent_days),
    }


def _format_display_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column.endswith("_rate") or column in {"pct_goal"}:
            out[column] = out[column].map(lambda value: fmt_percent(value) if pd.notna(value) else "Not Available")
        elif column in {
            "goal",
            "actual_enrollments",
            "leads",
            "leads_assigned",
            "applicants",
            "enrolled",
            "no_recent_activity_count",
            "duplicate_count",
            "do_not_contact_count",
            "unqualified_count",
            "lead_status_dead_lead_count",
        }:
            out[column] = out[column].map(lambda value: fmt_int(value) if pd.notna(value) else "Not Available")
        elif column in {"calls", "talk_time", "activities", "avg_talk_time_per_lead", "avg_activities_per_lead", "avg_days_to_enroll"}:
            out[column] = out[column].map(lambda value: fmt_number(value) if pd.notna(value) else "Not Available")
    return display_frame(out).replace({pd.NA: "Not Available"})


def show_table(df: pd.DataFrame, height: int = 360) -> None:
    if df.empty:
        st.info("Not Available")
        return
    st.dataframe(_format_display_values(df), use_container_width=True, hide_index=True, height=height)


def _add_avg_days(perf: pd.DataFrame, enrollments: pd.DataFrame, column: str) -> pd.DataFrame:
    if enrollments.empty or "days_to_enroll" not in enrollments.columns or column not in enrollments.columns:
        return perf
    avg = (
        enrollments.groupby(enrollments[column].fillna("").astype(str).replace("", "Not Available"))["days_to_enroll"]
        .mean()
        .reset_index(name="avg_days_to_enroll")
    )
    return perf.merge(avg, on=column, how="left")


def _rate_chart(df: pd.DataFrame, x: str, y: str, title: str) -> Any:
    if df.empty or x not in df.columns or y not in df.columns:
        return bar_chart(pd.DataFrame(), x, y, title)
    work = df.copy()
    work[y] = pd.to_numeric(work[y], errors="coerce").fillna(0)
    fig = px.bar(work.sort_values(y, ascending=False).head(15), x=x, y=y, title=title)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(template="plotly_white", height=380, title_font_color=CALMU_COLORS["navy"], margin={"l": 24, "r": 24, "t": 54, "b": 80})
    return fig


def page_executive(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    goal: float,
    filters: dict[str, Any],
) -> None:
    weekly_start = None if filters.get("date_range_default") else filters.get("start_date")
    weekly_end = None if filters.get("date_range_default") else filters.get("end_date")
    metrics = cached_metrics(leads, enrollments, goal, weekly_start, weekly_end)
    st.subheader("Executive Overview")
    metric_grid(
        [
            ("Total Enrollments", fmt_int(metrics["total_enrollments"])),
            ("Enrollment Goal", fmt_int(metrics["enrollment_goal"])),
            ("% to Goal", fmt_percent(metrics["pct_goal"])),
            ("Remaining to Goal", fmt_int(metrics["remaining_to_goal"])),
            ("Weekly Enrollments", fmt_int(metrics["weekly_enrollments"])),
            ("Leads", fmt_int(metrics["leads"])),
            ("Applicants", fmt_int(metrics["applicants"])),
            ("Enrolled", fmt_int(metrics["enrolled"])),
            ("Lead-to-Applicant %", fmt_percent(metrics["lead_to_applicant_rate"])),
            ("Applicant-to-Enrolled %", fmt_percent(metrics["applicant_to_enrolled_rate"])),
            ("Lead-to-Enrolled %", fmt_percent(metrics["lead_to_enrolled_rate"])),
        ],
        columns=4,
    )

    progress = enrollment_progress(metrics["total_enrollments"], metrics["enrollment_goal"])
    progress.update_layout(title="Enrollment Goal Progress")
    col1, col2 = st.columns(2)
    col1.plotly_chart(progress, use_container_width=True, key="exec_goal_progress")
    col2.plotly_chart(bar_chart(enrollment_group(enrollments, "enrollment_udr"), "enrollment_udr", "actual_enrollments", "Enrollments by UDR", horizontal=True), use_container_width=True, key="exec_enrollments_udr")

    col3, col4 = st.columns(2)
    col3.plotly_chart(bar_chart(enrollment_group(enrollments, "program"), "program", "actual_enrollments", "Enrollments by Program", horizontal=True), use_container_width=True, key="exec_enrollments_program")
    col4.plotly_chart(bar_chart(enrollment_group(enrollments, "degree"), "degree", "actual_enrollments", "Enrollments by Degree"), use_container_width=True, key="exec_enrollments_degree")

    col5, col6 = st.columns(2)
    col5.plotly_chart(bar_chart(enrollment_group(enrollments, "modality"), "modality", "actual_enrollments", "Enrollments by Modality"), use_container_width=True, key="exec_enrollments_modality")
    col6.plotly_chart(bar_chart(enrollment_group(enrollments, "vendor"), "vendor", "actual_enrollments", "Enrollments by Vendor", horizontal=True), use_container_width=True, key="exec_enrollments_vendor")


def page_udr_performance(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    goals: dict[str, float],
    expected_pct: float,
    no_recent_days: int,
) -> None:
    st.subheader("UDR Performance")
    scorecard = cached_udr_scorecard(leads, enrollments, goals, expected_pct, no_recent_days)
    st.markdown("#### UDR Performance Scorecard")
    st.caption(
        "Includes % Goal, Leads Assigned, Applicants, conversion rates, No Recent Activity %, "
        "status rates, optional activity fields, and a formula-based Performance Category."
    )
    columns = [
        "udr",
        "actual_enrollments",
        "goal",
        "pct_goal",
        "leads_assigned",
        "applicants",
        "lead_to_applicant_rate",
        "lead_to_enrolled_rate",
        "calls",
        "talk_time",
        "avg_talk_time_per_lead",
        "activities",
        "avg_activities_per_lead",
        "no_recent_activity_count",
        "no_recent_activity_rate",
        "duplicate_count",
        "duplicate_rate",
        "do_not_contact_count",
        "do_not_contact_rate",
        "unqualified_count",
        "unqualified_rate",
    ]
    if "lead_status_dead_lead_count" in scorecard.columns:
        columns.extend(["lead_status_dead_lead_count", "lead_status_dead_lead_rate"])
    columns.append("performance_category")
    show_table(scorecard[[column for column in columns if column in scorecard.columns]], height=430)

    if scorecard.empty:
        return
    selected_udr = st.selectbox("UDR drilldown", scorecard["udr"].tolist())
    udr_leads = leads[leads["assigned_udr"].eq(selected_udr)].copy()
    udr_enrollments = enrollments[enrollments["enrollment_udr"].eq(selected_udr)].copy()

    mix_tabs = st.tabs(["Program Mix", "Degree Mix", "Vendor Mix", "Modality Mix", "Status Breakdown", "Trends"])
    with mix_tabs[0]:
        st.plotly_chart(bar_chart(enrollment_group(udr_enrollments, "program"), "program", "actual_enrollments", "Program mix", horizontal=True), use_container_width=True, key="udr_program_mix")
    with mix_tabs[1]:
        st.plotly_chart(bar_chart(enrollment_group(udr_enrollments, "degree"), "degree", "actual_enrollments", "Degree mix"), use_container_width=True, key="udr_degree_mix")
    with mix_tabs[2]:
        st.plotly_chart(bar_chart(enrollment_group(udr_enrollments, "vendor"), "vendor", "actual_enrollments", "Vendor mix", horizontal=True), use_container_width=True, key="udr_vendor_mix")
    with mix_tabs[3]:
        st.plotly_chart(bar_chart(enrollment_group(udr_enrollments, "modality"), "modality", "actual_enrollments", "Modality mix"), use_container_width=True, key="udr_modality_mix")
    with mix_tabs[4]:
        left, right = st.columns(2)
        left.plotly_chart(bar_chart(funnel_metrics_by(udr_leads, pd.DataFrame(), "lead_status"), "lead_status", "leads", "Lead Status breakdown", horizontal=True), use_container_width=True, key="udr_lead_status_breakdown")
        right.plotly_chart(bar_chart(funnel_metrics_by(udr_leads, pd.DataFrame(), "lifecycle_stage"), "lifecycle_stage", "leads", "Lifecycle Stage breakdown", horizontal=True), use_container_width=True, key="udr_lifecycle_breakdown")
    with mix_tabs[5]:
        left, right = st.columns(2)
        if "activities" in udr_leads.columns and "create_date" in udr_leads.columns:
            activity = udr_leads.dropna(subset=["create_date"]).copy()
            activity["week"] = pd.to_datetime(activity["create_date"], errors="coerce", utc=True).dt.tz_convert(None).dt.to_period("W").dt.start_time
            trend = activity.groupby("week")["activities"].sum().reset_index()
            left.plotly_chart(line_chart(trend, "week", "activities", "Activity trend"), use_container_width=True, key="udr_activity_trend")
        else:
            left.info("Activity trend is Not Available because detailed activity volume fields are missing.")
        right.plotly_chart(line_chart(weekly_counts(udr_enrollments, "enrolled_date", "enrollments"), "week", "enrollments", "Enrollment trend"), use_container_width=True, key="udr_enrollment_trend")


def page_program_degree(leads: pd.DataFrame, enrollments: pd.DataFrame) -> None:
    st.subheader("Program / Degree")
    program_perf = _add_avg_days(funnel_metrics_by(leads, enrollments, "program"), enrollments, "program")
    degree_perf = funnel_metrics_by(leads, enrollments, "degree")

    col1, col2 = st.columns(2)
    col1.plotly_chart(bar_chart(program_perf, "program", "enrolled", "Enrollments by Program", horizontal=True), use_container_width=True, key="program_enrollments")
    col2.plotly_chart(bar_chart(degree_perf, "degree", "enrolled", "Enrollments by Degree"), use_container_width=True, key="degree_enrollments")

    col3, col4 = st.columns(2)
    col3.plotly_chart(bar_chart(program_perf, "program", "applicants", "Applicants by Program", horizontal=True), use_container_width=True, key="program_applicants")
    col4.plotly_chart(_rate_chart(program_perf, "program", "lead_to_applicant_rate", "Lead-to-Applicant % by Program"), use_container_width=True, key="program_lta")

    col5, col6 = st.columns(2)
    col5.plotly_chart(_rate_chart(program_perf, "program", "lead_to_enrolled_rate", "Lead-to-Enrolled % by Program"), use_container_width=True, key="program_lte")
    if "avg_days_to_enroll" in program_perf.columns:
        col6.plotly_chart(bar_chart(program_perf, "program", "avg_days_to_enroll", "Average Days to Enroll by Program", horizontal=True), use_container_width=True, key="program_avg_days")
    else:
        col6.info("Average Days to Enroll by Program is Not Available.")

    with st.expander("Program scorecard", expanded=False):
        show_table(program_perf, height=380)


def page_vendor(leads: pd.DataFrame, enrollments: pd.DataFrame, no_recent_days: int) -> None:
    st.subheader("Vendor Performance")
    perf = cached_vendor_performance(leads, enrollments, no_recent_days)
    col1, col2 = st.columns(2)
    col1.plotly_chart(bar_chart(perf, "vendor", "leads", "Leads by Vendor", horizontal=True), use_container_width=True, key="vendor_leads")
    col2.plotly_chart(bar_chart(perf, "vendor", "applicants", "Applicants by Vendor", horizontal=True), use_container_width=True, key="vendor_applicants")

    col3, col4 = st.columns(2)
    col3.plotly_chart(bar_chart(perf, "vendor", "enrolled", "Enrollments by Vendor", horizontal=True), use_container_width=True, key="vendor_enrollments")
    col4.plotly_chart(_rate_chart(perf, "vendor", "lead_to_enrolled_rate", "Lead-to-Enrolled % by Vendor"), use_container_width=True, key="vendor_lte")

    with st.expander("Vendor Performance Scorecard", expanded=True):
        st.caption(
            "Uses hard counts and rates only: Leads, Applicants, Enrollments, conversion rates, "
            "Duplicate %, Do Not Contact %, Unqualified %, and No Recent Activity %."
        )
        show_table(perf, height=430)


def page_activity(leads: pd.DataFrame, enrollments: pd.DataFrame, scorecard: pd.DataFrame, no_recent_days: int) -> None:
    st.subheader("Activity & Follow-Up")
    if not has_activity_fields(leads):
        st.info("Activity & Follow-Up is Not Available because activity fields are missing.")
        return
    activity = activity_summary_by_udr(leads, no_recent_days)
    col1, col2 = st.columns(2)
    if "calls" in activity.columns and activity["calls"].notna().any():
        col1.plotly_chart(bar_chart(activity, "udr", "calls", "Calls by UDR", horizontal=True), use_container_width=True, key="activity_calls")
    else:
        col1.info("Calls by UDR is Not Available.")
    if "talk_time" in activity.columns and activity["talk_time"].notna().any():
        col2.plotly_chart(bar_chart(activity, "udr", "talk_time", "Talk Time by UDR", horizontal=True), use_container_width=True, key="activity_talk_time")
    else:
        col2.info("Talk Time by UDR is Not Available.")

    col3, col4 = st.columns(2)
    if "avg_activities_per_lead" in activity.columns and activity["avg_activities_per_lead"].notna().any():
        col3.plotly_chart(bar_chart(activity, "udr", "avg_activities_per_lead", "Activities per Lead", horizontal=True), use_container_width=True, key="activity_avg_activities")
    else:
        col3.info("Activities per Lead is Not Available.")
    col4.plotly_chart(bar_chart(activity, "udr", "no_recent_activity_count", "No Recent Activity Count", horizontal=True), use_container_width=True, key="activity_no_recent")

    show_table(activity, height=360)

    applicants_no_recent = leads[
        leads["lifecycle_stage"].str.lower().eq("applicant") & no_recent_activity_mask(leads, no_recent_days)
    ].copy()
    with st.expander("Applicants with No Recent Activity", expanded=False):
        columns = [column for column in ["record_id", "email", "assigned_udr", "program", "vendor", "last_activity_date", "lead_status", "lifecycle_stage"] if column in applicants_no_recent.columns]
        display = redact_pii(applicants_no_recent[columns].head(500)) if columns else pd.DataFrame()
        st.dataframe(display, use_container_width=True, hide_index=True, height=320)

    with st.expander("Formula-based activity flags", expanded=False):
        if scorecard.empty:
            st.info("Not Available")
        else:
            flags = scorecard[
                scorecard["performance_category"].isin(
                    ["Low Activity", "Low Talk Time", "Low Conversion", "High No-Recent-Activity Rate"]
                )
            ].copy()
            show_table(flags, height=320)


def page_data_qa(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    uploaded: UploadedLeadData,
    last_refresh: str,
) -> None:
    st.subheader("Data QA")
    hubspot_enrolled = int(leads["lifecycle_stage"].str.lower().eq("enrolled").sum()) if "lifecycle_stage" in leads.columns else 0
    qa = qa_summary(leads, enrollments, hubspot_enrolled, last_refresh)
    st.markdown("#### Data QA Checks")
    st.caption(
        "Checks include Missing UDR, Missing Program, Missing Degree, Missing Vendor, Missing Modality, "
        "Missing Lead Status, Missing Lifecycle Stage, Missing Last Activity Date, Duplicate emails, "
        "Duplicate Record IDs, Enrollment tracker count vs HubSpot enrolled count, and Last refresh timestamp."
    )
    show_table(qa, height=360)

    with st.expander("Raw data samples", expanded=False):
        st.caption("Raw samples are limited and PII is redacted where possible.")
        st.markdown("#### Leads")
        st.dataframe(redact_pii(leads.head(1000)), use_container_width=True, hide_index=True, height=300)
        st.markdown("#### Enrollments")
        st.dataframe(redact_pii(enrollments.head(1000)), use_container_width=True, hide_index=True, height=300)
        if uploaded.load_notes:
            st.markdown("#### Load notes")
            st.dataframe(pd.DataFrame({"note": uploaded.load_notes}), use_container_width=True, hide_index=True, height=180)


def render_masthead(data_mode: str) -> None:
    st.markdown(
        f"""
        <div class="calmu-masthead">
            <div class="calmu-kicker">California Miramar University</div>
            <h1>Enrollment Performance Dashboard</h1>
            <p>{data_mode}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="CalMU Enrollment Performance", layout="wide", page_icon="CMU")
    apply_brand_theme()

    uploaded, tracker, budget = load_static_inputs()

    if "hubspot_refresh_key" not in st.session_state:
        st.session_state["hubspot_refresh_key"] = 0
    if "use_live_hubspot" not in st.session_state:
        st.session_state["use_live_hubspot"] = False

    st.sidebar.title("CalMU")
    page = st.sidebar.radio(
        "Page",
        [
            "Executive Overview",
            "UDR Performance",
            "Program / Degree",
            "Vendor Performance",
            "Activity & Follow-Up",
            "Data QA",
        ],
    )
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.session_state["use_live_hubspot"] = True
        st.session_state["hubspot_refresh_key"] += 1

    token_present = get_access_token() is not None
    should_fetch_hubspot = token_present and bool(st.session_state.get("use_live_hubspot"))
    hubspot = load_live_hubspot(st.session_state["hubspot_refresh_key"]) if should_fetch_hubspot else HubSpotFetchResult(
        contacts=pd.DataFrame(),
        properties=pd.DataFrame(),
        owners=pd.DataFrame(),
        used_properties=[],
        missing_properties=[],
        fetched_at=None,
        error=None,
        token_present=token_present,
    )

    if not hubspot.contacts.empty:
        raw_leads = hubspot.contacts.copy()
        data_mode = "Live HubSpot contacts are active. Enrollment totals come from the enrollment tracker."
    else:
        raw_leads = uploaded.udr_leads.copy()
        if hubspot.error:
            data_mode = f"HubSpot fetch failed; static uploaded baseline is active. Error: {hubspot.error}"
            st.warning(data_mode)
        elif not token_present:
            data_mode = "No HubSpot token found; static uploaded baseline is active."
            st.info(data_mode)
        elif not should_fetch_hubspot:
            data_mode = "Static uploaded baseline is active. Click Refresh data to load live HubSpot contacts."
        else:
            data_mode = "HubSpot returned no contacts; static uploaded baseline is active."

    last_refresh = hubspot.fetched_at.astimezone(timezone.utc).strftime("%b %d, %H:%M UTC") if hubspot.fetched_at else "Not refreshed"
    st.sidebar.caption(f"Last HubSpot refresh: {last_refresh}")

    leads, enrollments = normalize_cached(raw_leads, tracker.enrollments)
    known_udrs = _budget_udr_names(
        budget.allocations,
        budget.term_allocations,
        tracker.roundup_allocations,
        leads,
        enrollments,
    )
    leads, enrollments = canonicalize_udr_columns(leads, enrollments, known_udrs)
    render_masthead(data_mode)
    filters = build_filter_bar(leads, enrollments)
    filtered_leads, filtered_enrollments = apply_global_filters(leads, enrollments, filters)

    selected_terms = filters.get("term") or []
    if selected_terms:
        raw_udr_goals = (
            goals_by_udr(budget.term_allocations, selected_terms)
            or goals_by_udr(tracker.roundup_allocations, selected_terms)
        )
    else:
        raw_udr_goals = (
            goals_by_udr(budget.allocations)
            or goals_by_udr(tracker.roundup_allocations)
        )
    known_udrs = _budget_udr_names(budget.allocations, budget.term_allocations, tracker.roundup_allocations, filtered_leads, filtered_enrollments)
    udr_goals = canonicalize_udr_goals(raw_udr_goals, known_udrs)
    selected_udrs = set(filters.get("udr") or [])
    if selected_udrs:
        udr_goals = {udr: goal for udr, goal in udr_goals.items() if udr in selected_udrs}
    enrollment_goal = (
        sum(udr_goals.values())
        if (selected_terms or selected_udrs) and udr_goals
        else (total_goal(budget.summary) or total_goal(tracker.roundup_summary))
    )
    expected_pct = expected_goal_pct(filters.get("start_date"), filters.get("end_date"))
    scorecard = cached_udr_scorecard(
        filtered_leads,
        filtered_enrollments,
        udr_goals,
        expected_pct,
        filters["no_recent_days"],
    )

    if page == "Executive Overview":
        page_executive(filtered_leads, filtered_enrollments, enrollment_goal, filters)
    elif page == "UDR Performance":
        page_udr_performance(filtered_leads, filtered_enrollments, udr_goals, expected_pct, filters["no_recent_days"])
    elif page == "Program / Degree":
        page_program_degree(filtered_leads, filtered_enrollments)
    elif page == "Vendor Performance":
        page_vendor(filtered_leads, filtered_enrollments, filters["no_recent_days"])
    elif page == "Activity & Follow-Up":
        page_activity(filtered_leads, filtered_enrollments, scorecard, filters["no_recent_days"])
    else:
        page_data_qa(filtered_leads, filtered_enrollments, uploaded, last_refresh)

    with st.sidebar.expander("Metric formulas", expanded=False):
        st.write("Lead-to-Applicant % = Applicants / Leads")
        st.write("Lead-to-Enrolled % = Enrollments / Leads")
        st.write("Applicant-to-Enrolled % = Enrollments / Applicants")
        st.write("No Recent Activity % = No Recent Activity Count / Leads Assigned")
        st.write("UDR categories use the configured benchmark formulas only.")
        st.json(BENCHMARKS)


if __name__ == "__main__":
    main()
