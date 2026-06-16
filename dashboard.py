from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    FIELD_INVENTORY_PATH,
    FIELD_MAPPING_PATH,
    REMOVED_FIELDS_PATH,
    friendly_missing_token_message,
    load_json_safe,
    token_is_set,
)
from cohort_analysis import render_cohort_analysis_page
from definitions import definitions_frame, metric_definition, section_help_text
from metrics import (
    calculate_data_quality_metrics,
    calculate_executive_metrics,
    calculate_paid_lead_metrics,
    calculate_salesman_metrics,
    calculate_stuck_lead_metrics,
    monthly_leads,
    monthly_revenue,
)
from pipeline_health import render_pipeline_health_page
from storage import load_cached_data
from student_journey import render_student_journey_page
from tuition_loader import TUITION_SOURCE_NOTE, TUITION_SOURCE_URL, tuition_config_frame


APP_DIR = Path(__file__).resolve().parent
REVENUE_HELP_TEXT = (
    "Total program revenue estimates the full tuition value of the student. Annualized and "
    "6/12/24-month revenue spread that value over the normal program duration, so a 4-year "
    "bachelor's program is not treated the same as a 6-month certificate."
)
REVENUE_COLUMNS = [
    "total_program_revenue",
    "six_month_revenue",
    "twelve_month_revenue",
    "twenty_four_month_revenue",
    "annualized_program_revenue",
]
POTENTIAL_REVENUE_COLUMNS = [
    "potential_program_revenue",
    "potential_six_month_revenue",
    "potential_twelve_month_revenue",
    "potential_twenty_four_month_revenue",
    "potential_annualized_program_revenue",
]
PIPELINE_REVENUE_COLUMNS = [
    "potential_revenue",
    "enrolled_revenue",
    "open_pipeline_potential_revenue",
]
REVENUE_LABELS = {
    "total_program_revenue": "Total estimated program revenue",
    "six_month_revenue": "Estimated revenue next 6 months",
    "twelve_month_revenue": "Estimated revenue next 12 months",
    "twenty_four_month_revenue": "Estimated revenue next 24 months",
    "annualized_program_revenue": "Annualized revenue",
}
POTENTIAL_REVENUE_LABELS = {
    "potential_program_revenue": "Potential program revenue",
    "potential_six_month_revenue": "Potential revenue next 6 months",
    "potential_twelve_month_revenue": "Potential revenue next 12 months",
    "potential_twenty_four_month_revenue": "Potential revenue next 24 months",
    "potential_annualized_program_revenue": "Potential annualized revenue",
    "potential_revenue": "Potential revenue",
    "enrolled_revenue": "Enrolled revenue",
    "open_pipeline_potential_revenue": "Open pipeline potential revenue",
}


def money(value: Any) -> str:
    return f"${float(value or 0):,.0f}"


def pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.1f}%"


def days(value: Any) -> str:
    return f"{float(value or 0):.0f}"


def clean_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return []
    return sorted(value for value in df[column].dropna().astype(str).unique() if value.strip())


def has_values(df: pd.DataFrame, column: str) -> bool:
    return not df.empty and column in df.columns and df[column].dropna().astype(str).str.strip().ne("").any()


def bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[column].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def safe_series(df: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    if df.empty:
        return
    st.download_button(
        label=label,
        data=df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


def section_intro(name: str) -> None:
    help_text = section_help_text(name)
    if help_text:
        st.caption(help_text)


def metric_card(container: Any, label: str, value: Any, delta: Any = None) -> None:
    container.metric(label, value, delta, help=metric_definition(label) or None)


def table_column_config(df: pd.DataFrame) -> dict[str, Any]:
    config = {}
    for column in df.columns:
        help_text = metric_definition(column)
        if help_text:
            config[column] = st.column_config.Column(help=help_text)
    return config


def data_table(df: pd.DataFrame, container: Any = st) -> None:
    container.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=table_column_config(df),
    )


def revenue_metric_grid(metrics: dict[str, Any]) -> None:
    labels = [
        ("Total estimated program revenue", money(metrics.get("total_program_revenue", 0))),
        ("Next 6 months", money(metrics.get("six_month_revenue", 0))),
        ("Next 12 months", money(metrics.get("twelve_month_revenue", 0))),
        ("Next 24 months", money(metrics.get("twenty_four_month_revenue", 0))),
        ("Annualized revenue", money(metrics.get("annualized_program_revenue", 0))),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, labels):
        metric_card(col, label, value)
    extra = st.columns(3)
    metric_card(extra[0], "Potential revenue", money(metrics.get("potential_revenue", 0)))
    metric_card(extra[1], "Enrolled revenue", money(metrics.get("enrolled_revenue", 0)))
    metric_card(extra[2], "Open pipeline potential", money(metrics.get("open_pipeline_potential_revenue", 0)))


def potential_revenue_metric_grid(metrics: dict[str, Any]) -> None:
    labels = [
        ("Potential program revenue", money(metrics.get("potential_program_revenue", 0))),
        ("Potential next 6 months", money(metrics.get("potential_six_month_revenue", 0))),
        ("Potential next 12 months", money(metrics.get("potential_twelve_month_revenue", 0))),
        ("Potential next 24 months", money(metrics.get("potential_twenty_four_month_revenue", 0))),
        ("Potential annualized", money(metrics.get("potential_annualized_program_revenue", 0))),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, labels):
        metric_card(col, label, value)


def revenue_group_table(fact: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if fact.empty or group_column not in fact.columns:
        return pd.DataFrame()
    rows = []
    for group_value, group in fact.groupby(group_column, dropna=False):
        row = {
            group_column: group_value if str(group_value).strip() else "Unknown",
            "leads": group["contact_id"].dropna().astype(str).nunique() if "contact_id" in group else 0,
            "won_deals": group[
                bool_mask(group, "deal_countable") & bool_mask(group, "is_won")
            ]["deal_id"].dropna().astype(str).nunique()
            if "deal_id" in group
            else 0,
        }
        for column in [*REVENUE_COLUMNS, *POTENTIAL_REVENUE_COLUMNS, *PIPELINE_REVENUE_COLUMNS]:
            row[column] = pd.to_numeric(safe_series(group, column, 0), errors="coerce").fillna(0).sum()
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["twelve_month_revenue", "annualized_program_revenue"], ascending=False)


def run_sync() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "sync.py"],
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial_output = "\n".join(
            part.decode("utf-8", errors="replace") if isinstance(part, bytes) else str(part or "")
            for part in (exc.stdout, exc.stderr)
            if part
        ).strip()
        message = (
            "HubSpot refresh timed out before the dashboard cache finished building. "
            "Detailed activity history is skipped by default; if this still happens, retry once or run sync.py locally."
        )
        if partial_output:
            message = f"{message}\n\nLast sync output:\n{partial_output[-4000:]}"
        return False, message
    message = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    return result.returncode == 0, message


def show_header(
    metadata: dict[str, Any],
    contacts_df: pd.DataFrame,
    fact_df: pd.DataFrame,
) -> None:
    st.title("CalMU HubSpot Sales Performance")
    top = st.columns([1, 1, 1, 1])
    if top[0].button("Refresh HubSpot Data", width="stretch"):
        if not token_is_set():
            st.error(friendly_missing_token_message())
        else:
            with st.spinner("Refreshing HubSpot data..."):
                ok, message = run_sync()
            if ok:
                st.success("HubSpot data refreshed.")
                st.rerun()
            else:
                st.error(message or "HubSpot refresh failed.")

    last_refresh = metadata.get("last_refresh_time")
    metric_card(top[1], "Last refresh", last_refresh.strftime("%Y-%m-%d %H:%M UTC") if last_refresh else "Never")
    status = metadata.get("status") or {}
    metric_card(top[2], "HubSpot status", status.get("status", "No sync yet"))
    mapping = load_json_safe(FIELD_MAPPING_PATH, default={}) or {}
    mapped = sum(1 for section in ("contact", "deal") for value in (mapping.get(section) or {}).values() if value)
    missing = sum(1 for section in ("contact", "deal") for value in (mapping.get(section) or {}).values() if not value)
    metric_card(top[3], "Field mapping", f"{mapped} mapped", f"{missing} missing")

    if metadata.get("is_stale") and metadata.get("has_cache"):
        st.warning("Cached HubSpot data is older than 24 hours.")
    if not metadata.get("has_cache"):
        st.error("No cached dashboard data found. Run `python sync.py` after setting HUBSPOT_ACCESS_TOKEN.")
    mapped_deal_revenue = (mapping.get("deal") or {}).get("revenue")
    mapped_program_tuition = (mapping.get("contact") or {}).get("program_total_tuition")
    has_program_tuition_values = (
        "program_total_tuition" in contacts_df.columns
        and (pd.to_numeric(contacts_df["program_total_tuition"], errors="coerce").fillna(0) > 0).any()
    )
    if metadata.get("has_cache") and not mapped_deal_revenue and not mapped_program_tuition:
        st.warning(
            "Revenue and tuition fields are missing or empty in HubSpot. Revenue KPIs are unavailable and display as $0."
        )
    elif metadata.get("has_cache") and mapped_program_tuition and not has_program_tuition_values:
        st.warning(
            "A program tuition field is mapped, but the current cache has no fetched tuition values. Refresh HubSpot Data to populate duration-based revenue."
        )
    elif metadata.get("has_cache") and mapped_program_tuition and not mapped_deal_revenue:
        st.info(
            f"Program revenue is using the contact tuition field `{mapped_program_tuition}` because deal revenue is missing or empty."
        )
    st.info(TUITION_SOURCE_NOTE)
    st.info(REVENUE_HELP_TEXT)


def apply_filters(
    contacts_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    fact_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filters")
    contacts = contacts_df.copy()
    deals = deals_df.copy()
    fact = fact_df.copy()

    if has_values(contacts, "contact_created_at"):
        parsed_dates = pd.to_datetime(contacts["contact_created_at"], errors="coerce", utc=True)
        missing_created_dates = int(parsed_dates.isna().sum())
        dates = parsed_dates.dropna()
        if not dates.empty:
            min_date = dates.min().date()
            max_date = dates.max().date()
            selected = st.sidebar.date_input("Date range", value=(min_date, max_date))
            if missing_created_dates:
                st.sidebar.warning(
                    f"{missing_created_dates:,} contacts have missing created dates and are excluded by the date range."
                )
            if isinstance(selected, tuple) and len(selected) == 2:
                start, end = selected
                created = parsed_dates.dt.date
                contacts = contacts[(created >= start) & (created <= end)]
                if "contact_created_at" in fact.columns:
                    fact_dates = pd.to_datetime(fact["contact_created_at"], errors="coerce", utc=True).dt.date
                    fact = fact[((fact_dates >= start) & (fact_dates <= end)) | safe_series(fact, "contact_id").isna()]

    salesman = st.sidebar.multiselect("Salesman / Contact Owner", clean_options(contacts, "salesman_name"))
    if salesman:
        contacts = contacts[contacts["salesman_name"].astype(str).isin(salesman)]
        if "salesman_name" in fact.columns:
            fact = fact[fact["salesman_name"].astype(str).isin(salesman)]

    source_groups = st.sidebar.multiselect("Source group", clean_options(contacts, "source_group"))
    if source_groups:
        contacts = contacts[contacts["source_group"].astype(str).isin(source_groups)]
        if "source_group" in fact.columns:
            fact = fact[fact["source_group"].astype(str).isin(source_groups)]

    paid_filter = st.sidebar.selectbox("Paid vs organic", ["All", "Paid", "Organic"])
    if paid_filter != "All" and "paid_lead_flag" in contacts.columns:
        paid_mask = contacts["paid_lead_flag"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
        contacts = contacts[paid_mask] if paid_filter == "Paid" else contacts[~paid_mask]
        if "paid_lead_flag" in fact.columns:
            fact_paid = fact["paid_lead_flag"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
            fact = fact[fact_paid] if paid_filter == "Paid" else fact[~fact_paid]

    temperatures = st.sidebar.multiselect("Lead temperature", clean_options(contacts, "lead_temperature"))
    if temperatures:
        contacts = contacts[contacts["lead_temperature"].astype(str).isin(temperatures)]
        if "lead_temperature" in fact.columns:
            fact = fact[fact["lead_temperature"].astype(str).isin(temperatures)]

    for label, column in (
        ("Lifecycle stage", "lifecycle_stage"),
        ("Deal stage", "deal_stage"),
        ("Program", "program"),
    ):
        options = clean_options(contacts if column != "deal_stage" else fact, column)
        if options:
            selected = st.sidebar.multiselect(label, options)
            if selected:
                if column in contacts.columns:
                    contacts = contacts[contacts[column].astype(str).isin(selected)]
                if column in fact.columns:
                    fact = fact[fact[column].astype(str).isin(selected)]
                if column in deals.columns:
                    deals = deals[deals[column].astype(str).isin(selected)]

    contact_ids = set(contacts.get("contact_id", pd.Series(dtype=str)).dropna().astype(str))
    if contact_ids and "contact_id" in fact.columns:
        fact = fact[fact["contact_id"].isna() | fact["contact_id"].astype(str).isin(contact_ids)]
    return contacts, deals, fact


def metric_grid(metrics: dict[str, Any]) -> None:
    labels = [
        ("Total leads", metrics["total_leads"], None),
        ("Paid leads", metrics["paid_leads"], None),
        ("Organic leads", metrics["organic_leads"], None),
        ("Hot leads", metrics["hot_leads"], None),
        ("Dead leads", metrics["dead_leads"], None),
        ("Won deals", metrics["won_deals"], None),
        ("Total estimated program revenue", money(metrics["total_program_revenue"]), None),
        ("Annualized revenue", money(metrics["annualized_program_revenue"]), None),
        ("12-month revenue", money(metrics["twelve_month_revenue"]), None),
        ("Close rate", pct(metrics["close_rate"]), None),
        ("Paid close rate", pct(metrics["paid_close_rate"]), None),
        ("Avg days to close", days(metrics["average_days_to_close"]), None),
    ]
    for start in range(0, len(labels), 4):
        cols = st.columns(4)
        for col, (label, value, delta) in zip(cols, labels[start : start + 4]):
            metric_card(col, label, value, delta)


def page_executive(contacts: pd.DataFrame, deals: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Executive Overview")
    section_intro("Executive Overview")
    metrics = calculate_executive_metrics(contacts, fact)
    metric_grid(metrics)
    revenue_metric_grid(metrics)
    potential_revenue_metric_grid(metrics)

    left, right = st.columns(2)
    leads_month = monthly_leads(contacts)
    if not leads_month.empty:
        left.plotly_chart(px.line(leads_month, x="month", y="leads", markers=True, title="Leads by month"), width="stretch")
    else:
        left.warning("Lead created date is missing or empty in HubSpot.")

    revenue_month = monthly_revenue(fact)
    if not revenue_month.empty:
        right.plotly_chart(px.bar(revenue_month, x="month", y="revenue", title="Total estimated program revenue by month"), width="stretch")
    else:
        right.warning("Close date or revenue is missing or empty in HubSpot.")

    left, right = st.columns(2)
    if has_values(contacts, "source_group"):
        source = contacts.groupby("source_group")["contact_id"].nunique().reset_index(name="leads")
        left.plotly_chart(px.bar(source, x="source_group", y="leads", title="Leads by source group"), width="stretch")
    else:
        left.warning("Source fields are missing or empty in HubSpot.")

    if has_values(fact, "source_group") and "total_program_revenue" in fact.columns:
        revenue_source = fact.groupby("source_group")["total_program_revenue"].sum().reset_index()
        right.plotly_chart(px.bar(revenue_source, x="source_group", y="total_program_revenue", title="Total estimated program revenue by source group"), width="stretch")
    else:
        right.warning("Revenue by source is unavailable.")

    if has_values(contacts, "lead_temperature"):
        mix = contacts.groupby("lead_temperature")["contact_id"].nunique().reset_index(name="leads")
        st.plotly_chart(px.pie(mix, names="lead_temperature", values="leads", title="Hot vs dead lead mix"), width="stretch")

    degree_table = revenue_group_table(fact, "program_degree_level")
    if not degree_table.empty:
        st.subheader("Revenue by Degree Level")
        section_intro("Revenue by Degree Level")
        data_table(degree_table)
        download_button("Download degree-level revenue", degree_table, "degree_level_revenue.csv")


def page_daily_action_center(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Daily Action Center")
    section_intro("Daily Action Center")
    if contacts.empty:
        st.warning("No contacts match the current filters.")
        return

    work = contacts.copy()
    for column in (
        "contact_id",
        "student_name",
        "email",
        "phone",
        "salesman_name",
        "lead_temperature",
        "source_group",
        "paid_vendor",
        "program",
        "lead_status",
        "final_lead_status",
        "enrollment_status",
        "next_activity_date",
        "last_activity_date",
    ):
        if column not in work.columns:
            work[column] = pd.NA
    paid = bool_mask(work, "paid_lead_flag")
    hot = work["lead_temperature"].fillna("").astype(str).eq("Hot")
    reviveable = bool_mask(work, "reviveable_flag")
    open_deal = bool_mask(work, "has_open_deal")
    inactive = pd.to_numeric(safe_series(work, "days_since_last_activity"), errors="coerce").fillna(9999)
    no_next = safe_series(work, "next_activity_date").isna() | safe_series(work, "next_activity_date").astype(str).str.strip().isin(["", "NaT", "nan", "<NA>"])
    unassigned = work["salesman_name"].fillna("").astype(str).str.strip().isin(["", "Unassigned"])
    stale_paid = paid & ((inactive >= 7) | no_next | unassigned)
    old_open = open_deal & (inactive >= 14)

    work["action_reason"] = ""
    work.loc[hot, "action_reason"] = "Hot lead"
    work.loc[stale_paid, "action_reason"] = work.loc[stale_paid, "action_reason"].where(
        work.loc[stale_paid, "action_reason"].ne(""),
        "Paid lead leakage",
    )
    work.loc[old_open, "action_reason"] = work.loc[old_open, "action_reason"].where(
        work.loc[old_open, "action_reason"].ne(""),
        "Open deal needs follow-up",
    )
    work.loc[reviveable, "action_reason"] = work.loc[reviveable, "action_reason"].where(
        work.loc[reviveable, "action_reason"].ne(""),
        "Reviveable dead lead",
    )

    work["action_priority"] = 0
    work.loc[reviveable, "action_priority"] = 20
    work.loc[old_open, "action_priority"] = 40
    work.loc[stale_paid, "action_priority"] = 70
    work.loc[hot, "action_priority"] = 100
    work["recommended_action"] = "Review and update status."
    work.loc[hot, "recommended_action"] = "Call today and set the next enrollment step."
    work.loc[stale_paid, "recommended_action"] = "Assign owner, log outreach, and set next activity."
    work.loc[old_open, "recommended_action"] = "Confirm next step on the open deal or close it out."
    work.loc[reviveable, "recommended_action"] = "Try one final targeted follow-up, then archive if no response."

    action_queue = work[work["action_priority"] > 0].copy()
    cols = st.columns(5)
    metric_card(cols[0], "Hot leads", int(hot.sum()))
    metric_card(cols[1], "Paid no activity", int(stale_paid.sum()))
    metric_card(cols[2], "No activity 14 days", int((inactive >= 14).sum()))
    metric_card(cols[3], "No next activity", int(no_next.sum()))
    metric_card(cols[4], "Reviveable Dead Leads", int(reviveable.sum()))

    if action_queue.empty:
        st.success("No urgent action items match the current filters.")
        return

    action_columns = [
        "action_priority",
        "action_reason",
        "recommended_action",
        "contact_id",
        "student_name",
        "email",
        "phone",
        "salesman_name",
        "lead_temperature",
        "source_group",
        "paid_vendor",
        "program",
        "lead_status",
        "final_lead_status",
        "enrollment_status",
        "days_since_last_activity",
        "next_activity_date",
        "last_activity_date",
        "open_pipeline_potential_revenue",
        "potential_revenue",
    ]
    available = [column for column in action_columns if column in action_queue.columns]
    for revenue_column in ("open_pipeline_potential_revenue", "potential_revenue"):
        if revenue_column not in action_queue.columns:
            action_queue[revenue_column] = 0
    action_queue = action_queue.sort_values(
        ["action_priority", "open_pipeline_potential_revenue", "potential_revenue"],
        ascending=False,
    )
    data_table(action_queue[available].head(1000))
    download_button("Download daily action queue", action_queue[available], "daily_action_center.csv")


def page_degree_revenue(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Degree Revenue")
    section_intro("Degree Revenue")
    if fact.empty:
        st.warning("Revenue fact data is unavailable.")
        return

    degree_table = revenue_group_table(fact, "program_degree_level")
    if degree_table.empty:
        st.warning("Degree level revenue is unavailable because program or degree fields are missing or empty.")
    else:
        st.subheader("Revenue by Degree Level")
        section_intro("Revenue by Degree Level")
        data_table(degree_table)
        st.plotly_chart(
            px.bar(
                degree_table.head(20),
                x="program_degree_level",
                y=["total_program_revenue", "twelve_month_revenue", "annualized_program_revenue"],
                barmode="group",
                title="Total, 12-month, and annualized revenue by degree level",
            ),
            width="stretch",
        )
        download_button("Download degree revenue", degree_table, "degree_revenue.csv")

    program_column = "program" if "program" in fact.columns else "degree_program"
    program_table = revenue_group_table(fact, program_column)
    if program_table.empty:
        st.warning("Program-level revenue is unavailable because program fields are missing or empty.")
    else:
        st.subheader("Revenue by Program")
        section_intro("Revenue by Program")
        program_table = program_table.sort_values(["twelve_month_revenue", "annualized_program_revenue"], ascending=False)
        data_table(program_table.head(100))
        download_button("Download program revenue", program_table, "program_revenue.csv")


def page_paid(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Paid Leads Performance")
    section_intro("Paid Leads Performance")
    paid = calculate_paid_lead_metrics(contacts, fact)
    cols = st.columns(3)
    metric_card(cols[0], "Paid leads", paid["paid_leads"])
    metric_card(cols[1], "Paid close rate", pct(paid["paid_close_rate"]))
    metric_card(cols[2], "Revenue per paid lead", money(paid["revenue_per_paid_lead"]))
    revenue_metric_grid(paid)
    potential_revenue_metric_grid(paid)
    metric_card(st, "Avg paid days to close", days(paid["average_paid_days_to_close"]))

    left, right = st.columns(2)
    by_source = paid["paid_leads_by_source"]
    if not by_source.empty:
        left.plotly_chart(px.bar(by_source, x="source_group", y="paid_leads", title="Paid leads by source"), width="stretch")
    else:
        left.warning("Paid source fields are missing or empty in HubSpot.")

    by_campaign = paid["paid_leads_by_campaign"]
    if not by_campaign.empty:
        right.plotly_chart(px.bar(by_campaign, x="utm_campaign", y="paid_leads", title="Paid leads by campaign"), width="stretch")
    else:
        right.warning("Campaign field is missing or empty in HubSpot.")
    vendor = paid["paid_vendor_performance"]
    st.subheader("Vendor Performance")
    section_intro("Vendor Performance")
    vendor_search = st.text_input("Search vendor", placeholder="Atra")
    if not vendor.empty:
        if vendor_search.strip():
            vendor = vendor[
                vendor["vendor"].fillna("").astype(str).str.contains(vendor_search.strip(), case=False, regex=False)
            ]
        if not paid.get("vendor_cost_data_available"):
            st.info("Cost data is missing. Add vendor spend manually to config/vendor_costs.csv to calculate CPL and ROI.")
        data_table(vendor)
        download_button("Download paid vendor performance", vendor, "paid_vendor_performance.csv")
    else:
        st.warning("Paid vendor performance is unavailable.")
        if not paid.get("vendor_cost_data_available"):
            st.info("Cost data is missing. Add vendor spend manually to config/vendor_costs.csv to calculate CPL and ROI.")
    download_button("Download paid leads", contacts[bool_mask(contacts, "paid_lead_flag")], "paid_leads.csv")


def page_salesmen(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Salesmen / Contact Owner Performance")
    section_intro("Salesmen / Contact Owner Performance")
    table = calculate_salesman_metrics(contacts, fact)
    if table.empty:
        st.warning("No salesman performance data is available.")
    else:
        st.caption(
            "Sorted by action load, then open pipeline potential revenue, then paid lead leakage. "
            "Actual revenue uses HubSpot deal amount when available; estimated revenue uses CalMU tuition by degree level."
        )
        def chart_source(column: str, limit: int = 20) -> pd.DataFrame:
            values = pd.to_numeric(safe_series(table, column, 0), errors="coerce").fillna(0)
            source = table.assign(_chart_value=values)
            source = source[source["_chart_value"] > 0].sort_values("_chart_value", ascending=False).head(limit)
            return source.drop(columns=["_chart_value"])

        left, right = st.columns(2)
        potential_chart_table = chart_source("open_pipeline_potential_revenue")
        if not potential_chart_table.empty:
            left.plotly_chart(
                px.bar(
                    potential_chart_table,
                    x="salesman_name",
                    y="open_pipeline_potential_revenue",
                    title="Potential revenue by salesman",
                    labels={
                        "salesman_name": "Salesman",
                        "open_pipeline_potential_revenue": "Open pipeline potential revenue",
                    },
                ),
                width="stretch",
            )
        else:
            left.warning("Open pipeline potential revenue is unavailable for the current filters.")

        comparison_columns = ["actual_enrolled_revenue", "estimated_enrolled_revenue"]
        comparison_values = sum(
            pd.to_numeric(safe_series(table, column, 0), errors="coerce").fillna(0)
            for column in comparison_columns
        )
        comparison_table = (
            table.assign(_chart_value=comparison_values)
            .query("_chart_value > 0")
            .sort_values("_chart_value", ascending=False)
            .head(20)
            .drop(columns=["_chart_value"])
        )
        comparison = comparison_table.melt(
            id_vars="salesman_name",
            value_vars=[column for column in comparison_columns if column in comparison_table.columns],
            var_name="revenue_type",
            value_name="enrolled_revenue_value",
        )
        if not comparison.empty and pd.to_numeric(comparison["enrolled_revenue_value"], errors="coerce").fillna(0).sum() > 0:
            right.plotly_chart(
                px.bar(
                    comparison,
                    x="salesman_name",
                    y="enrolled_revenue_value",
                    color="revenue_type",
                    barmode="group",
                    title="Actual vs estimated revenue by salesman",
                    labels={
                        "salesman_name": "Salesman",
                        "enrolled_revenue_value": "Revenue",
                        "revenue_type": "Revenue type",
                    },
                ),
                width="stretch",
            )
        else:
            right.warning("Actual and estimated enrolled revenue are unavailable for the current filters.")

        left, right = st.columns(2)
        hot_chart_table = chart_source("hot_lead_potential_revenue")
        if not hot_chart_table.empty:
            left.plotly_chart(
                px.bar(
                    hot_chart_table,
                    x="salesman_name",
                    y="hot_lead_potential_revenue",
                    title="Hot lead potential revenue by salesman",
                    labels={"salesman_name": "Salesman", "hot_lead_potential_revenue": "Hot lead potential revenue"},
                ),
                width="stretch",
            )
        else:
            left.warning("Hot lead potential revenue is unavailable for the current filters.")

        paid_chart_table = chart_source("paid_lead_potential_revenue")
        if not paid_chart_table.empty:
            right.plotly_chart(
                px.bar(
                    paid_chart_table,
                    x="salesman_name",
                    y="paid_lead_potential_revenue",
                    title="Paid lead potential revenue by salesman",
                    labels={"salesman_name": "Salesman", "paid_lead_potential_revenue": "Paid lead potential revenue"},
                ),
                width="stretch",
            )
        else:
            right.warning("Paid lead potential revenue is unavailable for the current filters.")

        data_table(table)
        download_button("Download salesman performance", table, "salesman_performance.csv")

    drill_columns = [
        "contact_id",
        "contact_created_at",
        "salesman_name",
        "source_group",
        "paid_vendor",
        "paid_lead_flag",
        "program",
        "degree_level",
        "degree_program",
        "intended_program",
        "student_type",
        "enrollment_status",
        "start_term",
        "cohort",
        "campus",
        "modality",
        "program_total_tuition",
        "lifecycle_stage",
        "lead_status",
        "deal_stage",
        "revenue",
        "program_revenue_source",
        "program_duration_years",
        "program_duration_months",
        "total_program_revenue",
        "potential_program_revenue",
        "potential_revenue",
        "enrolled_revenue",
        "open_pipeline_potential_revenue",
        "revenue_confidence",
        "annualized_program_revenue",
        "six_month_revenue",
        "twelve_month_revenue",
        "twenty_four_month_revenue",
        "potential_twelve_month_revenue",
        "revenue_realization_note",
        "days_to_close",
        "lead_temperature",
        "dead_reason",
        "attribution_type",
    ]
    available = [column for column in drill_columns if column in fact.columns]
    drilldown = fact[available].copy() if available else pd.DataFrame()
    st.subheader("Lead Drilldown")
    section_intro("Lead Drilldown")
    data_table(drilldown)
    download_button("Download lead drilldown", drilldown, "lead_drilldown.csv")


def page_student_journey(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Per-Student Journey")
    section_intro("Student Journey")
    search = st.text_input("Find student", placeholder="Search contact id, email, owner, program, or source")
    journey_columns = [
        "contact_id",
        "contact_created_at",
        "email",
        "phone",
        "salesman_name",
        "source_group",
        "paid_vendor",
        "paid_lead_flag",
        "program",
        "degree_level",
        "degree_program",
        "intended_program",
        "student_type",
        "enrollment_status",
        "start_term",
        "cohort",
        "campus",
        "modality",
        "program_degree_level",
        "lifecycle_stage",
        "lead_status",
        "deal_id",
        "deal_created_at",
        "deal_stage",
        "close_date",
        "is_won",
        "is_lost",
        "lead_temperature",
        "days_since_last_activity",
        "days_to_close",
        "total_program_revenue",
        "potential_program_revenue",
        "potential_revenue",
        "enrolled_revenue",
        "open_pipeline_potential_revenue",
        "twelve_month_revenue",
        "potential_twelve_month_revenue",
        "revenue_confidence",
        "revenue_realization_note",
    ]
    available = [column for column in journey_columns if column in fact.columns]
    journey = fact[available].copy() if available else pd.DataFrame()
    if search and not journey.empty:
        haystack = journey.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        journey = journey[haystack.str.contains(search.lower(), na=False)]
    if not journey.empty:
        data_table(journey.head(500))
        download_button("Download student journey", journey, "student_journey.csv")
    else:
        st.warning("No student journey records match the current filters.")

    st.subheader("Journey Stage Revenue")
    section_intro("Journey Stage Revenue")
    lifecycle = revenue_group_table(fact, "lifecycle_stage")
    deal_stage = revenue_group_table(fact, "deal_stage")
    left, right = st.columns(2)
    if not lifecycle.empty:
        data_table(lifecycle, left)
        download_button("Download lifecycle journey revenue", lifecycle, "student_journey_lifecycle.csv")
    else:
        left.warning("Lifecycle-stage journey revenue is unavailable.")
    if not deal_stage.empty:
        data_table(deal_stage, right)
        download_button("Download deal-stage journey revenue", deal_stage, "student_journey_deal_stage.csv")
    else:
        right.warning("Deal-stage journey revenue is unavailable.")


def page_definitions() -> None:
    st.subheader("Definitions")
    section_intro("Definitions")
    st.subheader("Revenue Timing Definitions")
    revenue_timing_terms = [
        "Total Program Revenue",
        "Annualized Revenue",
        "Monthly Program Revenue",
        "6-Month Revenue",
        "12-Month Revenue",
        "24-Month Revenue",
        "Revenue Timing",
    ]
    for term in revenue_timing_terms:
        st.markdown(f"**{term}:** {metric_definition(term)}")
    st.subheader("All Metric Definitions")
    st.table(definitions_frame())
    st.subheader("Tuition Configuration")
    section_intro("Tuition Configuration")
    st.caption(f"{TUITION_SOURCE_NOTE} Source: {TUITION_SOURCE_URL}")
    data_table(tuition_config_frame())


def page_hot_dead(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Hot vs Dead Leads")
    section_intro("Hot vs Dead Leads")
    left, right = st.columns(2)
    if has_values(contacts, "lead_temperature"):
        hot = contacts[contacts["lead_temperature"] == "Hot"].groupby("salesman_name")["contact_id"].nunique().reset_index(name="hot_leads")
        dead = contacts[contacts["lead_temperature"] == "Dead"].groupby("salesman_name")["contact_id"].nunique().reset_index(name="dead_leads")
        left.plotly_chart(px.bar(hot, x="salesman_name", y="hot_leads", title="Hot leads by salesman"), width="stretch")
        right.plotly_chart(px.bar(dead, x="salesman_name", y="dead_leads", title="Dead leads by salesman"), width="stretch")

        by_source = contacts.groupby(["source_group", "lead_temperature"])["contact_id"].nunique().reset_index(name="leads")
        st.plotly_chart(px.bar(by_source, x="source_group", y="leads", color="lead_temperature", barmode="group", title="Hot vs dead by source"), width="stretch")
    else:
        st.warning("Lead temperature is unavailable.")

    stuck = calculate_stuck_lead_metrics(contacts)
    cols = st.columns(5)
    metric_card(cols[0], "No activity 7 days", stuck["leads_no_activity_7_days"])
    metric_card(cols[1], "No activity 14 days", stuck["leads_no_activity_14_days"])
    metric_card(cols[2], "No activity 30 days", stuck["leads_no_activity_30_days"])
    metric_card(cols[3], "Paid no activity", stuck["paid_leads_with_no_activity"])
    metric_card(cols[4], "Paid 30+ no open deal", stuck["paid_leads_older_30_days_no_open_deal"])

    reviveable = contacts[bool_mask(contacts, "reviveable_flag")]
    archive = contacts[bool_mask(contacts, "archive_remove_flag")]
    paid_no_activity = contacts[
        bool_mask(contacts, "paid_lead_flag")
        & (
            pd.to_numeric(safe_series(contacts, "days_since_last_activity"), errors="coerce").fillna(9999)
            >= 7
        )
    ]
    st.subheader("Reviveable Dead Leads")
    section_intro("Reviveable Dead Leads")
    data_table(reviveable)
    download_button("Download reviveable dead leads", reviveable, "reviveable_dead_leads.csv")
    st.subheader("Archive / Remove Leads")
    section_intro("Archive / Remove Leads")
    data_table(archive)
    download_button("Download archive remove leads", archive, "archive_remove_leads.csv")
    st.subheader("Paid Leads With No Activity")
    section_intro("Paid Leads With No Activity")
    data_table(paid_no_activity)
    download_button("Download paid leads with no activity", paid_no_activity, "paid_leads_no_activity.csv")


def page_quality(contacts: pd.DataFrame, deals: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Data Quality and Field Mapping")
    section_intro("Data Quality and Field Mapping")
    quality = calculate_data_quality_metrics(contacts, deals, fact)
    quality_items = list(quality.items())
    for start in range(0, len(quality_items), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, quality_items[start : start + 4]):
            metric_card(col, label.replace("_", " ").title(), value)

    mapping = load_json_safe(FIELD_MAPPING_PATH, default={}) or {}
    rows = []
    for object_type in ("contact", "deal"):
        for logical, field in (mapping.get(object_type) or {}).items():
            rows.append({"object_type": object_type, "dashboard_field": logical, "hubspot_field": field})
    mapping_df = pd.DataFrame(rows)
    st.subheader("Fields Being Used")
    section_intro("Fields Being Used")
    data_table(mapping_df)
    download_button("Download field mapping", mapping_df, "dashboard_field_mapping.csv")

    if REMOVED_FIELDS_PATH.exists():
        removed = pd.read_csv(REMOVED_FIELDS_PATH)
    else:
        removed = pd.DataFrame(columns=["object_type", "dashboard_field", "reason", "preferred_candidates"])
    st.subheader("Fields Removed")
    section_intro("Fields Removed")
    data_table(removed)
    download_button("Download removed fields", removed, "dashboard_removed_fields.csv")

    if FIELD_INVENTORY_PATH.exists():
        inventory = pd.read_csv(FIELD_INVENTORY_PATH)
        usable = inventory["usable"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
        low = inventory[usable & (pd.to_numeric(inventory["percent_filled"], errors="coerce") < 20)]
        missing = inventory[~usable]
        st.subheader("Low Population Fields")
        section_intro("Low Population Fields")
        data_table(low)
        st.subheader("Missing Fields")
        section_intro("Missing Fields")
        data_table(missing)
    else:
        st.warning("Field inventory has not been created yet.")

    fallback = fact[safe_series(fact, "attribution_type", "") == "fallback_to_deal_owner"]
    unclear = fact[safe_series(fact, "unclear_revenue_attribution", False).astype(str).str.lower().isin(["true", "1", "yes"])]
    st.subheader("Deal Owner Fallback Records")
    section_intro("Deal Owner Fallback Records")
    data_table(fallback)
    st.subheader("Unclear Revenue Attribution")
    section_intro("Unclear Revenue Attribution")
    data_table(unclear)


def main() -> None:
    st.set_page_config(page_title="CalMU HubSpot Sales Dashboard", layout="wide")
    tables, metadata = load_cached_data()
    contacts = tables["contacts_clean"]
    deals = tables["deals_clean"]
    fact = tables["lead_deal_fact"]
    activity_events = tables.get("activity_events", pd.DataFrame())

    show_header(metadata, contacts, fact)
    contacts, deals, fact = apply_filters(contacts, deals, fact)

    page = st.sidebar.radio(
        "Page",
        [
            "Executive Overview",
            "Daily Action Center",
            "Degree Revenue",
            "Paid Lead Vendor Performance",
            "Salesmen / Contact Owner Performance",
            "Student Journey",
            "Cohort Analysis",
            "Pipeline Health",
            "Hot vs Dead Leads",
            "Data Quality and Field Mapping",
            "Definitions",
        ],
    )

    if page == "Executive Overview":
        page_executive(contacts, deals, fact)
    elif page == "Daily Action Center":
        page_daily_action_center(contacts, fact)
    elif page == "Degree Revenue":
        page_degree_revenue(contacts, fact)
    elif page == "Paid Lead Vendor Performance":
        page_paid(contacts, fact)
    elif page == "Salesmen / Contact Owner Performance":
        page_salesmen(contacts, fact)
    elif page == "Student Journey":
        render_student_journey_page(contacts, fact, activity_events)
    elif page == "Cohort Analysis":
        render_cohort_analysis_page(contacts, fact)
    elif page == "Pipeline Health":
        render_pipeline_health_page(contacts, fact)
    elif page == "Hot vs Dead Leads":
        page_hot_dead(contacts, fact)
    elif page == "Data Quality and Field Mapping":
        page_quality(contacts, deals, fact)
    else:
        page_definitions()


if __name__ == "__main__":
    main()
