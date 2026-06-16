from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from definitions import metric_definition, section_help_text


COHORT_SUMMARY_COLUMNS = [
    "cohort",
    "cohort source",
    "leads",
    "paid leads",
    "paid %",
    "enrolled students",
    "enrollment rate",
    "actual revenue",
    "estimated revenue",
    "potential revenue",
    "avg days to first contact",
    "avg days to enroll",
    "avg touches to enroll",
    "top vendor",
    "top program",
    "top salesman",
]


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


def _data_table(df: pd.DataFrame, container: Any = st) -> None:
    container.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=_table_column_config(df),
    )


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


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    text = df[column].fillna("").astype(str).str.strip()
    return text.mask(text.str.lower().isin({"nan", "nat", "<na>", "none"}), "")


def _num_series(df: pd.DataFrame, column: str, default: float = 0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[column].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _has_text(df: pd.DataFrame, column: str) -> pd.Series:
    return _text_series(df, column).ne("")


def _date_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[column], errors="coerce", utc=True)


def _parse_term_start(value: Any) -> pd.Timestamp | pd.NaT:
    text = _clean_text(value)
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(pd.Series([text]), errors="coerce", utc=True).iloc[0]
    if pd.notna(parsed):
        return parsed
    match = re.search(r"(20\d{2})", text)
    if not match:
        return pd.NaT
    year = int(match.group(1))
    lowered = text.lower()
    month = 1
    if "spring" in lowered:
        month = 1
    elif "summer" in lowered:
        month = 5
    elif "fall" in lowered or "autumn" in lowered:
        month = 8
    elif "winter" in lowered:
        month = 12
    return pd.Timestamp(year=year, month=month, day=1, tz="UTC")


def _parse_term_start_series(values: pd.Series) -> pd.Series:
    unique_values = pd.Series(values.dropna().unique())
    parsed_lookup = {value: _parse_term_start(value) for value in unique_values}
    return values.map(parsed_lookup)


def _month_start_dates(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce", utc=True)
    return pd.to_datetime({"year": parsed.dt.year, "month": parsed.dt.month, "day": 1}, errors="coerce", utc=True)


def _month_label(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce", utc=True)
    year = parsed.dt.year.astype("Int64").astype(str)
    month = parsed.dt.month.astype("Int64").astype(str).str.zfill(2)
    label = year + "-" + month
    return label.where(parsed.notna(), "Unknown")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _top_value(df: pd.DataFrame, column: str) -> str:
    values = _text_series(df, column).replace("", "Unknown")
    if values.empty:
        return "Unknown"
    counts = values.value_counts()
    return str(counts.index[0]) if not counts.empty else "Unknown"


def _top_table(df: pd.DataFrame, column: str, label: str, limit: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[label, "leads", "paid leads", "enrolled students"])
    work = df.copy()
    work[label] = _text_series(work, column).replace("", "Unknown")
    return (
        work.groupby(label, dropna=False)
        .agg(
            leads=("contact_id", "nunique"),
            **{
                "paid leads": ("paid_lead_flag", lambda value: int(value.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"]).sum())),
                "enrolled students": ("cohort_is_enrolled", "sum"),
            },
        )
        .reset_index()
        .sort_values("leads", ascending=False)
        .head(limit)
    )


def _mix_table(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[label, "leads", "share"])
    work = df.copy()
    work[label] = _text_series(work, column).replace("", "Unknown")
    rows = work.groupby(label, dropna=False)["contact_id"].nunique().reset_index(name="leads")
    total = float(rows["leads"].sum())
    rows["share"] = rows["leads"].map(lambda value: _safe_div(value, total))
    return rows.sort_values("leads", ascending=False)


def _contact_revenue_from_fact(fact: pd.DataFrame, column: str) -> dict[str, float]:
    if fact.empty or "contact_id" not in fact.columns or column not in fact.columns:
        return {}
    work = fact.copy()
    work["contact_id"] = work["contact_id"].fillna("").astype(str)
    work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
    return work[work["contact_id"].ne("")].groupby("contact_id")[column].sum().to_dict()


def _contact_first_won_date(fact: pd.DataFrame) -> dict[str, pd.Timestamp]:
    if fact.empty or "contact_id" not in fact.columns:
        return {}
    work = fact.copy()
    won = _bool_series(work, "is_won") | _bool_series(work, "has_won_deal")
    date_columns = [column for column in ("closed_won_date", "close_date", "enrollment_date") if column in work.columns]
    if not date_columns:
        return {}
    dates = None
    for column in date_columns:
        parsed = _date_series(work, column)
        dates = parsed if dates is None else dates.combine_first(parsed)
    work = work[won].assign(_enrollment_date=dates[won])
    work = work.dropna(subset=["_enrollment_date"])
    if work.empty:
        return {}
    return work.groupby(work["contact_id"].fillna("").astype(str))["_enrollment_date"].min().to_dict()


def _days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return ((end - start).dt.total_seconds() / 86400).where(start.notna() & end.notna())


def _first_contact_days(df: pd.DataFrame) -> pd.Series:
    created = _date_series(df, "contact_created_at")
    response_ms = _num_series(df, "time_to_first_engagement", pd.NA)
    from_ms = response_ms / 86400000
    first_activity = None
    for column in ("first_engagement_date", "first_outreach_date", "last_sales_activity_timestamp", "last_activity_date"):
        parsed = _date_series(df, column)
        first_activity = parsed if first_activity is None else first_activity.combine_first(parsed)
    from_date = _days_between(created, first_activity) if first_activity is not None else pd.Series([pd.NA] * len(df), index=df.index)
    output = from_ms.where(response_ms.notna(), from_date)
    return output.where(output >= 0)


def _cohort_fields(df: pd.DataFrame, won_dates: dict[str, pd.Timestamp]) -> pd.DataFrame:
    output = df.copy()
    created = _date_series(output, "contact_created_at")
    official = _text_series(output, "cohort")
    start_term = _text_series(output, "start_term")
    enrollment_dates = pd.Series(output["contact_id"].fillna("").astype(str).map(won_dates), index=output.index)
    enrollment_dates = pd.to_datetime(enrollment_dates, errors="coerce", utc=True)

    cohort = official.where(official.ne(""))
    source = pd.Series(["official cohort"] * len(output), index=output.index).where(official.ne(""))
    start_date = _parse_term_start_series(official).where(official.ne(""))

    use_start = cohort.isna() & start_term.ne("")
    cohort = cohort.where(~use_start, start_term)
    source = source.where(~use_start, "start_term")
    start_date = start_date.where(~use_start, _parse_term_start_series(start_term))

    use_enrollment = cohort.isna() & enrollment_dates.notna()
    cohort = cohort.where(~use_enrollment, _month_label(enrollment_dates))
    source = source.where(~use_enrollment, "enrollment date")
    start_date = start_date.where(~use_enrollment, _month_start_dates(enrollment_dates))

    use_created = cohort.isna()
    cohort = cohort.where(~use_created, _month_label(created))
    source = source.where(~use_created, "contact created month")
    start_date = start_date.where(~use_created, _month_start_dates(created))

    output["cohort"] = cohort.fillna("Unknown").replace("", "Unknown")
    output["cohort_source"] = source.fillna("contact created month")
    output["cohort_start_date"] = pd.to_datetime(start_date, errors="coerce", utc=True)
    return output


def prepare_cohort_contacts(contacts: pd.DataFrame, fact: pd.DataFrame) -> pd.DataFrame:
    if contacts.empty:
        return contacts.copy()
    output = contacts.copy()
    output["contact_id"] = output.get("contact_id", pd.Series(index=output.index, dtype=str)).fillna("").astype(str)
    output = output[output["contact_id"].ne("")]
    output = output.drop_duplicates(subset=["contact_id"], keep="first").copy()
    won_dates = _contact_first_won_date(fact)
    output = _cohort_fields(output, won_dates)

    actual_revenue = _contact_revenue_from_fact(fact, "revenue_attributed")
    output["cohort_actual_revenue"] = output["contact_id"].map(actual_revenue).fillna(0)
    output["cohort_estimated_revenue"] = _num_series(output, "enrolled_revenue").fillna(0)
    output["cohort_potential_revenue"] = _num_series(output, "potential_revenue").fillna(0)
    output["cohort_open_potential_revenue"] = _num_series(output, "open_pipeline_potential_revenue").fillna(0)
    output["cohort_is_paid"] = _bool_series(output, "paid_lead_flag")
    output["cohort_is_enrolled"] = (
        _bool_series(output, "has_won_deal")
        | (_num_series(output, "enrolled_revenue").fillna(0) > 0)
        | _text_series(output, "enrollment_status").str.lower().str.contains("enroll|active|signed|registered|accepted", regex=True, na=False)
    )
    output["cohort_days_to_first_contact"] = _first_contact_days(output)
    enrollment_dates = pd.Series(output["contact_id"].map(won_dates), index=output.index)
    enrollment_dates = pd.to_datetime(enrollment_dates, errors="coerce", utc=True)
    output["cohort_enrollment_date"] = enrollment_dates
    output["cohort_days_to_enroll"] = _days_between(_date_series(output, "contact_created_at"), enrollment_dates)
    output["cohort_sales_touches_to_enroll"] = _num_series(output, "num_notes").where(output["cohort_is_enrolled"])
    output["cohort_paid_leakage"] = output["cohort_is_paid"] & ~output["cohort_is_enrolled"] & (
        _text_series(output, "salesman_name").isin(["", "Unassigned"])
        | ~_has_text(output, "next_activity_date")
        | (_num_series(output, "days_since_last_activity").fillna(9999) >= 14)
        | ~_bool_series(output, "has_open_deal")
    )
    return output


def cohort_summary(prepared: pd.DataFrame) -> pd.DataFrame:
    columns = [*COHORT_SUMMARY_COLUMNS]
    if prepared.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for cohort, group in prepared.groupby("cohort", dropna=False):
        leads = group["contact_id"].nunique()
        paid_leads = int(group["cohort_is_paid"].sum())
        enrolled = int(group["cohort_is_enrolled"].sum())
        rows.append(
            {
                "cohort": cohort,
                "cohort source": _top_value(group, "cohort_source"),
                "leads": int(leads),
                "paid leads": paid_leads,
                "paid %": _safe_div(paid_leads, leads),
                "enrolled students": enrolled,
                "enrollment rate": _safe_div(enrolled, leads),
                "actual revenue": float(group["cohort_actual_revenue"].sum()),
                "estimated revenue": float(group["cohort_estimated_revenue"].sum()),
                "potential revenue": float(group["cohort_potential_revenue"].sum()),
                "avg days to first contact": float(group["cohort_days_to_first_contact"].mean()) if group["cohort_days_to_first_contact"].notna().any() else 0.0,
                "avg days to enroll": float(group["cohort_days_to_enroll"].mean()) if group["cohort_days_to_enroll"].notna().any() else 0.0,
                "avg touches to enroll": float(group["cohort_sales_touches_to_enroll"].mean()) if group["cohort_sales_touches_to_enroll"].notna().any() else 0.0,
                "top vendor": _top_value(group, "paid_vendor"),
                "top program": _top_value(group, "program"),
                "top salesman": _top_value(group, "salesman_name"),
            }
        )
    output = pd.DataFrame(rows, columns=columns)
    sort_date = prepared.groupby("cohort")["cohort_start_date"].min()
    output["_sort_date"] = output["cohort"].map(sort_date)
    return output.sort_values(["_sort_date", "cohort"], ascending=[False, False]).drop(columns=["_sort_date"])


def cohort_calculated_fields(prepared: pd.DataFrame) -> pd.DataFrame:
    summary = cohort_summary(prepared).rename(
        columns={
            "cohort source": "cohort_source",
            "leads": "cohort_size",
            "paid leads": "cohort_paid_leads",
            "paid %": "cohort_paid_share",
            "enrolled students": "cohort_enrolled_students",
            "enrollment rate": "cohort_enrollment_rate",
            "actual revenue": "cohort_actual_revenue",
            "estimated revenue": "cohort_estimated_revenue",
            "potential revenue": "cohort_potential_revenue",
            "avg days to first contact": "cohort_avg_days_to_first_contact",
            "avg days to enroll": "cohort_avg_days_to_enroll",
            "avg touches to enroll": "cohort_avg_sales_touches_to_enroll",
            "top vendor": "cohort_top_vendor",
            "top program": "cohort_top_program",
            "top salesman": "cohort_top_salesman",
        }
    )
    if summary.empty:
        return summary
    start_dates = prepared.groupby("cohort")["cohort_start_date"].min()
    summary["cohort_start_date"] = summary["cohort"].map(start_dates)
    hot = prepared["lead_temperature"].fillna("").astype(str).eq("Hot")
    dead = prepared["lead_temperature"].fillna("").astype(str).eq("Dead")
    shares = []
    for cohort, group in prepared.groupby("cohort", dropna=False):
        leads = group["contact_id"].nunique()
        shares.append(
            {
                "cohort": cohort,
                "cohort_hot_lead_share": _safe_div(int(hot.loc[group.index].sum()), leads),
                "cohort_dead_lead_share": _safe_div(int(dead.loc[group.index].sum()), leads),
            }
        )
    return summary.merge(pd.DataFrame(shares), on="cohort", how="left")


def _apply_page_filters(prepared: pd.DataFrame) -> pd.DataFrame:
    filtered = prepared.copy()

    def options(column: str) -> list[str]:
        if column not in filtered.columns:
            return []
        return sorted(value for value in _text_series(filtered, column).unique() if value)

    with st.expander("Cohort filters", expanded=True):
        first, second, third = st.columns(3)
        selected_cohorts = first.multiselect("Cohort", options("cohort"))
        selected_programs = second.multiselect("Program", options("program"))
        selected_vendors = third.multiselect("Vendor", options("paid_vendor"))
        fourth, fifth, sixth = st.columns(3)
        selected_salesmen = fourth.multiselect("Salesman", options("salesman_name"))
        paid_filter = fifth.selectbox("Paid vs organic", ["All", "Paid", "Organic"])
        selected_campus = sixth.multiselect("Campus", options("campus"))
        selected_modality = st.multiselect("Modality", options("modality"))

    for column, selected in (
        ("cohort", selected_cohorts),
        ("program", selected_programs),
        ("paid_vendor", selected_vendors),
        ("salesman_name", selected_salesmen),
        ("campus", selected_campus),
        ("modality", selected_modality),
    ):
        if selected and column in filtered.columns:
            filtered = filtered[_text_series(filtered, column).isin(selected)]
    if paid_filter == "Paid":
        filtered = filtered[filtered["cohort_is_paid"]]
    elif paid_filter == "Organic":
        filtered = filtered[~filtered["cohort_is_paid"]]
    return filtered


def _selected_cohort_frame(prepared: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if prepared.empty:
        return prepared
    choices = summary["cohort"].astype(str).tolist() if not summary.empty else sorted(prepared["cohort"].astype(str).unique())
    selected = st.selectbox("Selected cohort", choices)
    return prepared[prepared["cohort"].astype(str).eq(str(selected))].copy()


def _render_characteristics(group: pd.DataFrame) -> None:
    st.subheader("Cohort Characteristics")
    _section_intro("Cohort Characteristics")
    if group.empty:
        st.warning("No records match the selected cohort filters.")
        return

    response = group["cohort_days_to_first_contact"].dropna()
    touches = group.loc[group["cohort_is_enrolled"], "cohort_sales_touches_to_enroll"].dropna()
    left, right = st.columns(2)
    left.metric("Average response time", f"{response.mean():.1f} days" if not response.empty else "N/A", help=metric_definition("Average response time") or None)
    right.metric("Average touches before enrollment", f"{touches.mean():.1f}" if not touches.empty else "N/A", help=metric_definition("Average Sales Touches Before Enrollment") or None)

    top_specs = [
        ("Top lead sources", "source_group", "source"),
        ("Top vendors", "paid_vendor", "vendor"),
        ("Top campaigns", "utm_campaign", "campaign"),
        ("Top programs", "program", "program"),
        ("Degree mix", "program_degree_level", "degree level"),
        ("Salesman mix", "salesman_name", "salesman"),
    ]
    for start in range(0, len(top_specs), 2):
        cols = st.columns(2)
        for col, (title, column, label) in zip(cols, top_specs[start : start + 2]):
            col.markdown(f"**{title}**")
            _data_table(_top_table(group, column, label), col)

    left, right = st.columns(2)
    temperature = _mix_table(group, "lead_temperature", "lead temperature")
    paid_mix = pd.DataFrame(
        [
            {"paid vs organic": "Paid", "leads": int(group["cohort_is_paid"].sum())},
            {"paid vs organic": "Organic", "leads": int((~group["cohort_is_paid"]).sum())},
        ]
    )
    paid_mix["share"] = paid_mix["leads"].map(lambda value: _safe_div(value, paid_mix["leads"].sum()))
    left.markdown("**Hot/warm/cold/dead mix**")
    _data_table(temperature, left)
    right.markdown("**Paid vs organic mix**")
    _data_table(paid_mix, right)


def _render_enrollment_inputs(group: pd.DataFrame) -> None:
    st.subheader("What It Took To Enroll")
    _section_intro("What It Took To Enroll")
    enrolled = group[group["cohort_is_enrolled"]].copy()
    if enrolled.empty:
        st.warning("No enrolled students are available in this cohort.")
        return
    rows = [
        {"metric": "median days to first contact", "value": enrolled["cohort_days_to_first_contact"].median()},
        {"metric": "median number of touches", "value": enrolled["cohort_sales_touches_to_enroll"].median()},
        {"metric": "median days to enrollment", "value": enrolled["cohort_days_to_enroll"].median()},
        {"metric": "most common source", "value": _top_value(enrolled, "source_group")},
        {"metric": "most common vendor", "value": _top_value(enrolled, "paid_vendor")},
        {"metric": "most common program", "value": _top_value(enrolled, "program")},
        {"metric": "most common salesman path", "value": _top_value(enrolled, "salesman_name")},
    ]
    output = pd.DataFrame(rows)
    output["value"] = output["value"].map(lambda value: f"{value:.1f}" if isinstance(value, (float, int)) and pd.notna(value) else value)
    _data_table(output)

    status_rows = []
    for column, label in (
        ("lead_status", "lead status"),
        ("final_lead_status", "final lead status"),
        ("lifecycle_stage", "lifecycle stage"),
        ("enrollment_status", "enrollment status"),
    ):
        table = _mix_table(enrolled, column, label).head(10)
        if not table.empty:
            table.insert(0, "status field", label)
            table = table.rename(columns={label: "status"})
            status_rows.append(table)
    st.markdown("**Common lead statuses before enrollment**")
    if status_rows:
        _data_table(pd.concat(status_rows, ignore_index=True))
    else:
        st.warning("Lead status fields are unavailable for enrolled students in this cohort.")


def _render_comparison(prepared: pd.DataFrame) -> None:
    st.subheader("Cohort Comparison")
    _section_intro("Cohort Comparison")
    if prepared.empty:
        st.warning("No cohorts are available for comparison.")
        return
    summary = cohort_summary(prepared)
    summary["revenue per lead"] = summary.apply(lambda row: _safe_div(row["estimated revenue"] or row["actual revenue"], row["leads"]), axis=1)
    leakage = (
        prepared.groupby("cohort")["cohort_paid_leakage"].sum().reset_index(name="paid lead leakage")
    )
    dead = (
        prepared.assign(_dead=prepared["lead_temperature"].fillna("").astype(str).eq("Dead"))
        .groupby("cohort")
        .agg(dead_leads=("_dead", "sum"), leads=("contact_id", "nunique"))
        .reset_index()
    )
    dead["dead lead rate"] = dead.apply(lambda row: _safe_div(row["dead_leads"], row["leads"]), axis=1)
    comparison = summary.merge(leakage, on="cohort", how="left").merge(dead[["cohort", "dead lead rate"]], on="cohort", how="left")
    comparison = comparison[
        [
            "cohort",
            "enrollment rate",
            "revenue per lead",
            "avg days to enroll",
            "avg touches to enroll",
            "paid lead leakage",
            "dead lead rate",
        ]
    ]
    comparison_choices = comparison["cohort"].astype(str).tolist()
    selected = st.multiselect("Compare cohorts", comparison_choices, default=comparison_choices[: min(6, len(comparison_choices))])
    if selected:
        comparison = comparison[comparison["cohort"].astype(str).isin(selected)]
    _data_table(comparison)
    if not comparison.empty:
        left, right = st.columns(2)
        left.plotly_chart(px.bar(comparison, x="cohort", y="enrollment rate", title="Enrollment rate by cohort"), width="stretch")
        right.plotly_chart(px.bar(comparison, x="cohort", y="revenue per lead", title="Revenue per lead by cohort"), width="stretch")


def render_cohort_analysis_page(contacts: pd.DataFrame, fact: pd.DataFrame) -> None:
    st.subheader("Cohort Analysis")
    _section_intro("Cohort Analysis")
    prepared = prepare_cohort_contacts(contacts, fact)
    if prepared.empty:
        st.warning("Cohort analysis requires contacts with contact IDs.")
        return

    filtered = _apply_page_filters(prepared)
    summary = cohort_summary(filtered)

    st.subheader("Cohort Summary Table")
    _section_intro("Cohort Summary Table")
    if summary.empty:
        st.warning("No cohorts match the selected filters.")
        return
    _data_table(summary)
    st.download_button(
        "Download cohort summary",
        data=cohort_calculated_fields(filtered).to_csv(index=False),
        file_name="cohort_analysis_summary.csv",
        mime="text/csv",
        width="stretch",
    )

    selected_group = _selected_cohort_frame(filtered, summary)
    _render_characteristics(selected_group)
    _render_enrollment_inputs(selected_group)
    _render_comparison(filtered)
