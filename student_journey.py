from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from definitions import metric_definition, section_help_text


PROFILE_FIELDS = [
    ("student_name", "Name"),
    ("email", "Email"),
    ("phone", "Phone"),
    ("salesman_name", "Contact owner / salesman"),
    ("lifecycle_stage", "Lifecycle stage"),
    ("lead_status", "Lead status"),
    ("final_lead_status", "Final lead status"),
    ("source_group", "Source group"),
    ("paid_vendor", "Vendor"),
    ("utm_campaign", "Campaign"),
    ("paid_lead_flag", "Paid lead flag"),
    ("program", "Program"),
    ("degree_level", "Degree level"),
    ("cohort", "Cohort"),
    ("campus", "Campus"),
    ("modality", "Modality"),
    ("estimated_total_tuition", "Estimated tuition"),
    ("actual_revenue", "Actual revenue"),
    ("potential_revenue", "Potential revenue"),
]

JOURNEY_COLUMNS = [
    "event_date",
    "event_type",
    "owner",
    "description",
    "outcome",
    "days_since_previous_event",
]


def _section_intro(name: str) -> None:
    help_text = section_help_text(name)
    if help_text:
        st.caption(help_text)


def _metric_card(container: Any, label: str, value: Any) -> None:
    container.metric(label, value, help=metric_definition(label) or None)


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
    return str(value).strip()


def _as_datetime(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _money(value: Any) -> str:
    return f"${float(value or 0):,.0f}"


def _days_between(start: Any, end: Any) -> int | None:
    start_date = _as_datetime(start)
    end_date = _as_datetime(end)
    if start_date is None or end_date is None:
        return None
    days = int((end_date - start_date).days)
    return days if days >= 0 else None


def _days_since(value: Any) -> int | None:
    event_date = _as_datetime(value)
    if event_date is None:
        return None
    return int((datetime.now(timezone.utc) - event_date.to_pydatetime()).days)


def _first_date(values: list[Any]) -> pd.Timestamp | None:
    dates = [_as_datetime(value) for value in values]
    dates = [date for date in dates if date is not None]
    return min(dates) if dates else None


def _first_response_date(contact: pd.Series, detailed_events: pd.DataFrame) -> pd.Timestamp | None:
    detailed_date = None
    if not detailed_events.empty and "event_date" in detailed_events.columns:
        detailed_date = pd.to_datetime(detailed_events["event_date"], errors="coerce", utc=True).dropna().min()
    if detailed_date is not None and not pd.isna(detailed_date):
        return detailed_date

    direct_date = _first_date(
        [
            contact.get("first_outreach_date"),
            contact.get("first_engagement_date"),
            contact.get("last_sales_activity_timestamp"),
            contact.get("last_activity_date"),
        ]
    )
    if direct_date is not None:
        return direct_date

    created = _as_datetime(contact.get("contact_created_at"))
    response_ms = pd.to_numeric(pd.Series([contact.get("time_to_first_engagement")]), errors="coerce").iloc[0]
    if created is not None and pd.notna(response_ms) and response_ms >= 0:
        return created + pd.to_timedelta(float(response_ms), unit="ms")
    return None


def search_students(contacts: pd.DataFrame, query: str) -> pd.DataFrame:
    if contacts.empty:
        return contacts
    query = query.strip().lower()
    if not query:
        return pd.DataFrame()
    search_columns = ["student_name", "email", "phone", "contact_id"]
    mask = pd.Series([False] * len(contacts), index=contacts.index)
    for column in search_columns:
        if column in contacts.columns:
            mask = mask | contacts[column].fillna("").astype(str).str.lower().str.contains(query, regex=False)
    return contacts[mask].copy()


def _contact_events(contact: pd.Series) -> pd.DataFrame:
    owner = _clean_text(contact.get("salesman_name")) or "Unassigned"
    rows = []

    def add_event(event_date: Any, event_type: str, description: str, outcome: str = "") -> None:
        parsed = _as_datetime(event_date)
        if parsed is not None:
            rows.append(
                {
                    "event_date": parsed,
                    "event_type": event_type,
                    "owner": owner,
                    "description": description,
                    "outcome": outcome,
                }
            )

    add_event(contact.get("contact_created_at"), "Contact created", "HubSpot contact record created")
    add_event(contact.get("owner_assigned_date"), "Owner assigned", "Contact owner assigned")
    add_event(contact.get("first_outreach_date"), "First outreach", "First outreach date from HubSpot summary")
    add_event(
        contact.get("first_engagement_date"),
        "First engagement",
        _clean_text(contact.get("first_engagement_type")) or "First engagement date from HubSpot summary",
        _clean_text(contact.get("first_engagement_description")),
    )
    add_event(contact.get("application_date"), "Application", "Application date")
    add_event(contact.get("enrollment_date"), "Enrollment", "Enrollment date")
    add_event(contact.get("last_sales_activity_timestamp"), "Last sales activity", "Last sales activity timestamp")
    add_event(contact.get("last_activity_date"), "Last activity", "Last activity date")
    add_event(contact.get("next_activity_date"), "Next activity", "Next scheduled activity")
    return pd.DataFrame(rows)


def _deal_events(contact_id: str, fact: pd.DataFrame) -> pd.DataFrame:
    if fact.empty or "contact_id" not in fact.columns:
        return pd.DataFrame()
    related = fact[fact["contact_id"].fillna("").astype(str) == contact_id].copy()
    if related.empty:
        return pd.DataFrame()
    rows = []
    for row in related.to_dict("records"):
        owner = _clean_text(row.get("salesman_name")) or "Unassigned"
        deal_id = _clean_text(row.get("deal_id"))
        stage = _clean_text(row.get("deal_stage"))

        def add_event(event_date: Any, event_type: str, description: str, outcome: str = "") -> None:
            parsed = _as_datetime(event_date)
            if parsed is not None:
                rows.append(
                    {
                        "event_date": parsed,
                        "event_type": event_type,
                        "owner": owner,
                        "description": description,
                        "outcome": outcome,
                    }
                )

        add_event(row.get("deal_created_at"), "Deal created", f"Deal {deal_id}".strip(), stage)
        add_event(row.get("close_date"), "Deal close date", f"Deal {deal_id}".strip(), stage)
        if bool(row.get("is_won")):
            add_event(row.get("closed_won_date") or row.get("close_date"), "Closed won", f"Deal {deal_id}".strip(), stage)
        elif bool(row.get("is_lost")):
            add_event(row.get("close_date"), "Closed lost", f"Deal {deal_id}".strip(), stage)
    return pd.DataFrame(rows)


def _activity_events(contact_id: str, activity_events: pd.DataFrame) -> pd.DataFrame:
    if activity_events.empty or "contact_id" not in activity_events.columns:
        return pd.DataFrame()
    related = activity_events[activity_events["contact_id"].fillna("").astype(str) == contact_id].copy()
    if related.empty:
        return pd.DataFrame()
    for column in JOURNEY_COLUMNS:
        if column not in related.columns:
            related[column] = pd.NA
    return related[JOURNEY_COLUMNS[:-1]].copy()


def build_journey_events(
    contact: pd.Series,
    fact: pd.DataFrame,
    activity_events: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    contact_id = _clean_text(contact.get("contact_id"))
    detailed = _activity_events(contact_id, activity_events)
    frames = [_contact_events(contact), detailed, _deal_events(contact_id, fact)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=JOURNEY_COLUMNS), detailed.empty

    events = pd.concat(frames, ignore_index=True, sort=False)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce", utc=True)
    events = events.dropna(subset=["event_date"]).sort_values(["event_date", "event_type"])
    events["days_since_previous_event"] = events["event_date"].diff().dt.days
    events["days_since_previous_event"] = events["days_since_previous_event"].fillna(0).clip(lower=0).astype(int)
    for column in JOURNEY_COLUMNS:
        if column not in events.columns:
            events[column] = pd.NA
    return events[JOURNEY_COLUMNS], detailed.empty


def _actual_revenue(contact_id: str, fact: pd.DataFrame) -> float:
    if fact.empty or "contact_id" not in fact.columns:
        return 0.0
    related = fact[fact["contact_id"].fillna("").astype(str) == contact_id]
    if related.empty or "revenue_attributed" not in related.columns:
        return 0.0
    return float(pd.to_numeric(related["revenue_attributed"], errors="coerce").fillna(0).sum())


def _first_deal_date(contact_id: str, fact: pd.DataFrame) -> pd.Timestamp | None:
    if fact.empty or "contact_id" not in fact.columns or "deal_created_at" not in fact.columns:
        return None
    related = fact[fact["contact_id"].fillna("").astype(str) == contact_id]
    dates = pd.to_datetime(related["deal_created_at"], errors="coerce", utc=True).dropna()
    return dates.min() if not dates.empty else None


def _enrollment_date(contact: pd.Series, events: pd.DataFrame) -> pd.Timestamp | None:
    direct = _first_date([contact.get("enrollment_date"), contact.get("start_term")])
    if direct is not None:
        return direct
    if events.empty:
        return None
    enrollment = events[events["event_type"].fillna("").astype(str).str.contains("Enrollment", case=False, na=False)]
    if enrollment.empty:
        return None
    return pd.to_datetime(enrollment["event_date"], errors="coerce", utc=True).dropna().min()


def journey_metrics(contact: pd.Series, fact: pd.DataFrame, events: pd.DataFrame, detailed_events_missing: bool) -> dict[str, Any]:
    contact_id = _clean_text(contact.get("contact_id"))
    created = _as_datetime(contact.get("contact_created_at"))
    detailed = events[events["event_type"].isin(["Call", "Email", "Meeting", "Note", "Task"])]
    first_contact = _first_response_date(contact, detailed)
    first_deal = _first_deal_date(contact_id, fact)
    enrollment = _enrollment_date(contact, events)
    last_activity = None
    if not events.empty:
        activity_like = events[~events["event_type"].isin(["Contact created", "Deal created"])]
        if not activity_like.empty:
            last_activity = pd.to_datetime(activity_like["event_date"], errors="coerce", utc=True).dropna().max()
    if last_activity is None:
        last_activity = _first_date([contact.get("last_activity_date"), contact.get("last_sales_activity_timestamp")])

    touch_count = int(len(detailed)) if not detailed_events_missing else int(float(contact.get("num_notes") or 0))
    return {
        "days_from_lead_creation_to_first_contact": _days_between(created, first_contact),
        "number_of_sales_touches": touch_count,
        "number_of_calls": int((detailed["event_type"] == "Call").sum()) if not detailed.empty else 0,
        "number_of_emails": int((detailed["event_type"] == "Email").sum()) if not detailed.empty else 0,
        "number_of_meetings": int((detailed["event_type"] == "Meeting").sum()) if not detailed.empty else 0,
        "days_from_lead_creation_to_deal_creation": _days_between(created, first_deal),
        "days_from_lead_creation_to_enrollment": _days_between(created, enrollment),
        "days_from_deal_creation_to_enrollment": _days_between(first_deal, enrollment),
        "days_since_last_activity": _days_since(last_activity),
        "current_lead_temperature": _clean_text(contact.get("lead_temperature")) or "Unknown",
        "recommended_next_action": recommended_next_action(contact, touch_count, _days_since(last_activity)),
    }


def recommended_next_action(contact: pd.Series, touch_count: int, days_since_last_activity: int | None) -> str:
    temperature = _clean_text(contact.get("lead_temperature")).lower()
    has_open_deal = str(contact.get("has_open_deal")).lower() in {"true", "1", "yes"}
    if temperature == "hot" and (days_since_last_activity is None or days_since_last_activity >= 1):
        return "Call today and schedule the next enrollment step."
    if has_open_deal and (days_since_last_activity is None or days_since_last_activity >= 3):
        return "Follow up on the open deal and confirm next action."
    if touch_count == 0:
        return "Make first contact attempt and log the outcome."
    if days_since_last_activity is not None and days_since_last_activity >= 7:
        return "Re-engage with a call and email sequence."
    return "Continue normal follow-up cadence."


def _profile_frame(contact: pd.Series, fact: pd.DataFrame) -> pd.DataFrame:
    contact = contact.copy()
    contact_id = _clean_text(contact.get("contact_id"))
    contact["actual_revenue"] = _actual_revenue(contact_id, fact)
    rows = []
    for column, label in PROFILE_FIELDS:
        value = contact.get(column)
        if column in {"estimated_total_tuition", "actual_revenue", "potential_revenue"}:
            value = _money(value)
        rows.append({"Field": label, "Value": _clean_text(value)})
    return pd.DataFrame(rows)


def render_student_journey_page(
    contacts: pd.DataFrame,
    fact: pd.DataFrame,
    activity_events: pd.DataFrame,
) -> None:
    st.subheader("Student Journey")
    _section_intro("Student Journey")
    query = st.text_input("Search student", placeholder="Name, email, phone, or HubSpot contact ID")
    matches = search_students(contacts, query)
    if not query.strip():
        st.info("Search for a student by name, email, phone, or HubSpot contact ID.")
        return
    if matches.empty:
        st.warning("No student matched that search.")
        return

    matches = matches.sort_values("contact_created_at", na_position="last").head(50)
    options = {
        f"{_clean_text(row.get('student_name')) or _clean_text(row.get('email')) or row.get('contact_id')} | {row.get('contact_id')}": idx
        for idx, row in matches.iterrows()
    }
    selected_label = st.selectbox("Matching students", list(options.keys()))
    contact = contacts.loc[options[selected_label]]
    contact_id = _clean_text(contact.get("contact_id"))

    st.subheader("Student Profile")
    _section_intro("Student Profile")
    _data_table(_profile_frame(contact, fact))

    events, detailed_events_missing = build_journey_events(contact, fact, activity_events)
    if detailed_events_missing:
        st.info("Detailed activity history is not available. Showing summary activity fields from HubSpot.")

    metrics = journey_metrics(contact, fact, events, detailed_events_missing)
    st.subheader("Calculated Metrics")
    _section_intro("Calculated Metrics")
    labels = [
        ("Days to first contact", metrics["days_from_lead_creation_to_first_contact"]),
        ("Sales touches", metrics["number_of_sales_touches"]),
        ("Calls", metrics["number_of_calls"]),
        ("Emails", metrics["number_of_emails"]),
        ("Meetings", metrics["number_of_meetings"]),
        ("Days to deal", metrics["days_from_lead_creation_to_deal_creation"]),
        ("Days to enrollment", metrics["days_from_lead_creation_to_enrollment"]),
        ("Deal to enrollment", metrics["days_from_deal_creation_to_enrollment"]),
        ("Days since last activity", metrics["days_since_last_activity"]),
        ("Lead temperature", metrics["current_lead_temperature"]),
    ]
    for start in range(0, len(labels), 5):
        cols = st.columns(5)
        for col, (label, value) in zip(cols, labels[start : start + 5]):
            _metric_card(col, label, "N/A" if value is None else value)
    st.info(metrics["recommended_next_action"])

    st.subheader("Journey Events")
    _section_intro("Journey Events")
    if events.empty:
        st.warning("No journey events are available for this student.")
    else:
        output = events.copy()
        output["event_date"] = output["event_date"].dt.strftime("%Y-%m-%d %H:%M")
        _data_table(output)
        st.download_button(
            "Download journey events",
            data=output.to_csv(index=False),
            file_name=f"student_journey_{contact_id}.csv",
            mime="text/csv",
            width="stretch",
        )
