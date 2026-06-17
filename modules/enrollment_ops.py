from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from .term_utils import normalize_term_token


BENCHMARKS = {
    "min_activities_per_lead": 1.0,
    "min_talk_time_per_lead": 60,
    "min_lead_to_applicant_rate": 0.10,
    "max_no_recent_activity_rate": 0.25,
    "max_duplicate_rate": 0.20,
    "max_unqualified_rate": 0.20,
}

COLUMN_ALIASES = {
    "Record ID": "record_id",
    "record_id": "record_id",
    "hs_object_id": "record_id",
    "id": "record_id",
    "Email": "email",
    "email": "email",
    "Contact owner": "assigned_udr",
    "Contact Owner": "assigned_udr",
    "contact_owner": "assigned_udr",
    "assigned_udr": "assigned_udr",
    "UDR": "enrollment_udr",
    "udr": "enrollment_udr",
    "Enrollment Tracker UDR": "enrollment_udr",
    "enrollment_udr": "enrollment_udr",
    "Lifecycle Stage": "lifecycle_stage",
    "lifecycle_stage": "lifecycle_stage",
    "Lead Status": "lead_status",
    "lead_status": "lead_status",
    "Create Date": "create_date",
    "create_date": "create_date",
    "createdate": "create_date",
    "Last Activity Date": "last_activity_date",
    "last_activity_date": "last_activity_date",
    "hs_last_sales_activity_timestamp": "last_activity_date",
    "notes_last_updated": "last_activity_date",
    "Paid Lead List": "paid_lead_list",
    "paid_lead_list": "paid_lead_list",
    "Organic Lead List": "organic_lead_list",
    "organic_lead_list": "organic_lead_list",
    "Program": "program",
    "program": "program",
    "Degree Program": "program",
    "Academic Program": "program",
    "degree_program": "program",
    "Degree": "degree",
    "degree": "degree",
    "Modality": "modality",
    "modality": "modality",
    "Delivery Type": "modality",
    "Campus / Format": "modality",
    "Campus Location": "campus_location",
    "campus_location": "campus_location",
    "Source": "vendor",
    "source": "vendor",
    "Vendor": "vendor",
    "vendor": "vendor",
    "normalized_source": "vendor",
    "Student Type": "student_type",
    "student_type": "student_type",
    "Term": "term",
    "term": "term",
    "term_label": "term",
    "Enrolled Date": "enrolled_date",
    "enrolled_date": "enrolled_date",
    "Days to enroll": "days_to_enroll",
    "days_to_enroll": "days_to_enroll",
    "lead_type": "lead_type",
    "Lead Type": "lead_type",
    "calls": "calls",
    "call_count": "calls",
    "num_calls": "calls",
    "Talk Time": "talk_time",
    "talk_time": "talk_time",
    "total_talk_time": "talk_time",
    "hs_call_duration": "talk_time",
    "Activities": "activities",
    "activities": "activities",
    "number_of_sales_activities": "activities",
    "num_notes": "activities",
}

DISPLAY_LABELS = {
    "actual_enrollments": "Enrollments",
    "applicant_to_enrolled_rate": "Applicant-to-Enrolled %",
    "applicants": "Applicants",
    "assigned_udr": "Assigned UDR",
    "avg_activities_per_lead": "Avg Activities per Lead",
    "avg_days_to_enroll": "Average Days to Enroll",
    "avg_talk_time_per_lead": "Avg Talk Time per Lead",
    "calls": "Calls",
    "degree": "Degree",
    "do_not_contact_count": "Do Not Contact Count",
    "do_not_contact_rate": "Do Not Contact %",
    "duplicate_count": "Duplicate Lead Count",
    "duplicate_rate": "Duplicate %",
    "enrolled": "Enrolled",
    "enrollment_goal": "Enrollment Goal",
    "enrollment_udr": "Enrollment UDR",
    "goal": "Goal",
    "lead_status": "Lead Status",
    "lead_status_dead_lead_count": "Lead Status: Dead Lead Count",
    "lead_status_dead_lead_rate": "Lead Status: Dead Lead %",
    "lead_to_applicant_rate": "Lead-to-Applicant %",
    "lead_to_enrolled_rate": "Lead-to-Enrolled %",
    "leads": "Leads",
    "leads_assigned": "Leads Assigned",
    "lifecycle_stage": "Lifecycle Stage",
    "modality": "Modality",
    "no_recent_activity_count": "No Recent Activity Count",
    "no_recent_activity_rate": "No Recent Activity %",
    "pct_goal": "% Goal",
    "performance_category": "Performance Category",
    "program": "Program",
    "remaining_to_goal": "Remaining to Goal",
    "student_type": "Student Type",
    "talk_time": "Talk Time",
    "total_enrollments": "Total Enrollments",
    "udr": "UDR",
    "unqualified_count": "Unqualified Count",
    "unqualified_rate": "Unqualified %",
    "vendor": "Vendor",
    "weekly_enrollments": "Weekly Enrollments",
}

STATUS_DUPLICATE = "duplicate lead"
STATUS_DO_NOT_CONTACT = "do not contact"
STATUS_UNQUALIFIED = "app submitted - unqualified"
STATUS_DEAD_LEAD = "dead lead"


def display_label(column: str) -> str:
    return DISPLAY_LABELS.get(column, column.replace("_", " ").title())


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={column: display_label(column) for column in df.columns})


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return " ".join(str(value).replace("\xa0", " ").strip().split())


def infer_degree(program: Any) -> str:
    text = clean_text(program).lower()
    if not text:
        return ""
    if "certificate" in text or text.startswith("cert") or "-cy" in text and "as" not in text:
        return "Certificate"
    if "associate" in text or text.startswith("as") or "asba" in text or "asbt" in text:
        return "Associate"
    if "bachelor" in text or "bsba" in text or "bsbt" in text:
        return "Bachelor"
    if "master" in text or "mba" in text or "mscis" in text or "msai" in text:
        return "Master"
    if "doctoral" in text or "doctorate" in text or "dba" in text:
        return "Doctoral"
    return "Unknown"


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    data: dict[str, pd.Series] = {}
    for column in dict.fromkeys(df.columns):
        subset = df.loc[:, df.columns == column]
        if subset.shape[1] == 1:
            data[column] = subset.iloc[:, 0]
        else:
            data[column] = subset.replace("", pd.NA).bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(data, index=df.index)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {column: COLUMN_ALIASES[column] for column in out.columns if column in COLUMN_ALIASES}
    out = out.rename(columns=rename)
    return _coalesce_duplicate_columns(out)


def _ensure(out: pd.DataFrame, column: str, default: Any = "") -> None:
    if column not in out.columns:
        out[column] = default


def _text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].map(clean_text)


def _clean_names(values: list[Any] | set[Any] | tuple[Any, ...]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = clean_text(value)
        key = name.lower()
        if name and key not in seen:
            names.append(name)
            seen.add(key)
    return names


def _build_udr_aliases(known_names: list[Any] | set[Any] | tuple[Any, ...]) -> dict[str, str]:
    names = _clean_names(known_names)
    full_names = [name for name in names if len(name.split()) > 1]
    short_names = [name for name in names if len(name.split()) == 1]
    candidates = full_names + [name for name in short_names if name.lower() not in {full.split()[0].lower() for full in full_names}]

    alias_groups: dict[str, set[str]] = {}
    for name in candidates:
        parts = name.split()
        aliases = {name.lower()}
        if len(parts) > 1:
            aliases.add(parts[0].lower())
            aliases.add(" ".join(parts[:2]).lower())
        for alias in aliases:
            alias_groups.setdefault(alias, set()).add(name)
    return {alias: next(iter(matches)) for alias, matches in alias_groups.items() if len(matches) == 1}


def canonicalize_udr_name(value: Any, known_names: list[Any] | set[Any] | tuple[Any, ...]) -> str:
    name = clean_text(value)
    if not name:
        return ""
    aliases = _build_udr_aliases(known_names)
    candidates = _clean_names(known_names)
    return _canonicalize_udr_name_with_aliases(name, aliases, candidates)


def _canonicalize_udr_name_with_aliases(name: Any, aliases: dict[str, str], candidates: list[str]) -> str:
    name = clean_text(name)
    if not name:
        return ""
    key = name.lower()
    if key in aliases:
        return aliases[key]
    matches = [candidate for candidate in candidates if candidate.lower().startswith(f"{key} ")]
    return matches[0] if len(matches) == 1 else name


def canonicalize_udr_columns(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    known_names: list[Any] | set[Any] | tuple[Any, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_names = list(known_names)
    if not leads.empty and "assigned_udr" in leads.columns:
        all_names.extend(_text_column(leads, "assigned_udr").tolist())
    if not enrollments.empty and "enrollment_udr" in enrollments.columns:
        all_names.extend(_text_column(enrollments, "enrollment_udr").tolist())
    aliases = _build_udr_aliases(all_names)
    candidates = _clean_names(all_names)

    out_leads = leads.copy()
    out_enrollments = enrollments.copy()
    if "assigned_udr" in out_leads.columns:
        out_leads["assigned_udr"] = _text_column(out_leads, "assigned_udr").map(lambda value: _canonicalize_udr_name_with_aliases(value, aliases, candidates))
    if "enrollment_udr" in out_enrollments.columns:
        out_enrollments["enrollment_udr"] = _text_column(out_enrollments, "enrollment_udr").map(lambda value: _canonicalize_udr_name_with_aliases(value, aliases, candidates))
    return out_leads, out_enrollments


def canonicalize_udr_goals(
    goals: dict[str, float],
    known_names: list[Any] | set[Any] | tuple[Any, ...],
) -> dict[str, float]:
    all_names = list(known_names) + list(goals.keys())
    aliases = _build_udr_aliases(all_names)
    candidates = _clean_names(all_names)
    out: dict[str, float] = {}
    for name, value in goals.items():
        canonical = _canonicalize_udr_name_with_aliases(name, aliases, candidates)
        if not canonical:
            continue
        out[canonical] = out.get(canonical, 0.0) + float(value or 0)
    return out


def _date_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NaT] * len(df), index=df.index)
    return pd.to_datetime(df[column], errors="coerce", utc=True)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def normalize_leads(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    for column in [
        "record_id",
        "email",
        "assigned_udr",
        "program",
        "degree",
        "vendor",
        "modality",
        "campus_location",
        "student_type",
        "lead_type",
        "lead_status",
        "lifecycle_stage",
        "term",
    ]:
        _ensure(out, column)
    if "program" not in out.columns or not _text_column(out, "program").ne("").any():
        out["program"] = _text_column(out, "degree")
    if not _text_column(out, "program").ne("").any() and "campus_location" in out.columns:
        out["program"] = ""
    out["degree"] = _text_column(out, "program").map(infer_degree)
    if not _text_column(out, "modality").ne("").any() and "campus_location" in out.columns:
        out["modality"] = _text_column(out, "campus_location")
    for column in [
        "record_id",
        "email",
        "assigned_udr",
        "program",
        "degree",
        "vendor",
        "modality",
        "student_type",
        "lead_type",
        "lead_status",
        "lifecycle_stage",
        "term",
    ]:
        out[column] = _text_column(out, column)
    out["email"] = out["email"].str.lower()
    out["create_date"] = _date_column(out, "create_date")
    out["last_activity_date"] = _date_column(out, "last_activity_date")
    for column in ["calls", "talk_time", "activities"]:
        if column in out.columns:
            out[column] = _numeric_column(out, column)
    return dedupe_leads(out)


def normalize_enrollments(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)
    for column in [
        "student",
        "enrollment_udr",
        "program",
        "degree",
        "vendor",
        "modality",
        "student_type",
        "lead_type",
        "term",
        "workbook_term",
        "days_to_enroll",
    ]:
        _ensure(out, column)
    out["program"] = _text_column(out, "program")
    out["degree"] = _text_column(out, "program").map(infer_degree)
    for column in ["student", "enrollment_udr", "vendor", "modality", "student_type", "lead_type", "workbook_term"]:
        out[column] = _text_column(out, column)
    term_values = _text_column(out, "term")
    workbook_terms = _text_column(out, "workbook_term")
    term_dates = pd.to_datetime(term_values, errors="coerce")
    out["term"] = term_values.mask(workbook_terms.ne("") & term_dates.notna(), workbook_terms)
    out["enrolled_date"] = _date_column(out, "enrolled_date")
    out["days_to_enroll"] = _numeric_column(out, "days_to_enroll")
    return out[_text_column(out, "student").ne("")].copy()


def _lead_identity(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    record = _text_column(df, "record_id")
    email = _text_column(df, "email").str.lower()
    fallback = pd.Series([f"row-{idx}" for idx in df.index], index=df.index)
    return record.mask(record.eq(""), email).mask(lambda series: series.eq(""), fallback)


def dedupe_leads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["_lead_identity"] = _lead_identity(out)
    sort_columns = [column for column in ["last_activity_date", "create_date"] if column in out.columns]
    if sort_columns:
        out = out.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")
    return out.drop_duplicates("_lead_identity", keep="first").drop(columns=["_lead_identity"])


def filter_by_date(df: pd.DataFrame, column: str, start: date | None, end: date | None) -> pd.DataFrame:
    if df.empty or column not in df.columns or (start is None and end is None):
        return df.copy()
    dates = pd.to_datetime(df[column], errors="coerce", utc=True)
    mask = pd.Series([True] * len(df), index=df.index)
    if start:
        mask &= dates.dt.date >= start
    if end:
        mask &= dates.dt.date <= end
    return df[mask.fillna(False)].copy()


def apply_global_filters(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    filters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_leads = filter_by_date(leads, "create_date", filters.get("start_date"), filters.get("end_date"))
    out_enrollments = filter_by_date(enrollments, "enrolled_date", filters.get("start_date"), filters.get("end_date"))

    pairs = [
        ("term", "term", "term"),
        ("udr", "assigned_udr", "enrollment_udr"),
        ("program", "program", "program"),
        ("degree", "degree", "degree"),
        ("vendor", "vendor", "vendor"),
        ("modality", "modality", "modality"),
        ("student_type", "student_type", "student_type"),
        ("lead_type", "lead_type", "lead_type"),
    ]
    for key, lead_col, enrollment_col in pairs:
        values = filters.get(key) or []
        if not values:
            continue
        if lead_col in out_leads.columns and _text_column(out_leads, lead_col).ne("").any():
            out_leads = out_leads[_text_column(out_leads, lead_col).isin(values)].copy()
        if enrollment_col in out_enrollments.columns and _text_column(out_enrollments, enrollment_col).ne("").any():
            out_enrollments = out_enrollments[_text_column(out_enrollments, enrollment_col).isin(values)].copy()
    return out_leads, out_enrollments


def has_activity_fields(leads: pd.DataFrame) -> bool:
    return any(column in leads.columns for column in ["last_activity_date", "calls", "talk_time", "activities"])


def exact_status_mask(df: pd.DataFrame, status: str) -> pd.Series:
    if df.empty or "lead_status" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return _text_column(df, "lead_status").str.lower().eq(status)


def no_recent_activity_mask(leads: pd.DataFrame, threshold_days: int = 7) -> pd.Series:
    if leads.empty:
        return pd.Series(dtype=bool)
    if "last_activity_date" not in leads.columns:
        return pd.Series([True] * len(leads), index=leads.index)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(threshold_days or 7))
    dates = pd.to_datetime(leads["last_activity_date"], errors="coerce", utc=True)
    return dates.isna() | (dates < cutoff)


def total_goal(summary: pd.DataFrame) -> float:
    if summary.empty or "metric" not in summary.columns:
        return 0.0
    match = summary[summary["metric"].fillna("").astype(str).str.lower().isin(["budget", "goal", "enrollment goal"])]
    if match.empty:
        return 0.0
    return float(pd.to_numeric(match.iloc[0].get("numeric_value"), errors="coerce") or 0)


def goals_by_udr(allocations: pd.DataFrame, selected_terms: list[str] | None = None) -> dict[str, float]:
    if allocations.empty:
        return {}
    work = normalize_columns(allocations)
    name_col = "vendor" if "vendor" in work.columns else ("budget_name" if "budget_name" in work.columns else "source")
    if name_col not in work.columns:
        return {}
    names = _text_column(work, name_col)
    work = work[~names.str.lower().isin({"total", "starts", "start%"})].copy()
    if work.empty:
        return {}
    value_col = "planned_budget"
    if selected_terms:
        if "term" not in work.columns or "term_budget" not in work.columns:
            return {}
        term_values = {normalize_term_token(term) for term in selected_terms if normalize_term_token(term)}
        term_mask = pd.Series([False] * len(work), index=work.index)
        for column in ["term", "term_label", "raw_term"]:
            if column in work.columns:
                term_mask |= _text_column(work, column).map(normalize_term_token).isin(term_values)
        if "term_metric" in work.columns:
            term_mask &= _text_column(work, "term_metric").str.lower().eq("goal")
        term_work = work[term_mask].copy()
        if term_work.empty:
            return {}
        work = term_work
        value_col = "term_budget"
    elif "term" in work.columns:
        total_mask = _text_column(work, "term").str.lower().eq("total")
        if total_mask.any():
            work = work[total_mask].copy()
        else:
            work = work.drop_duplicates(name_col)
    if value_col not in work.columns:
        return {}
    values = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
    grouped = values.groupby(_text_column(work, name_col)).sum()
    return {key: float(value) for key, value in grouped.to_dict().items() if key}


def expected_goal_pct(start: date | None, end: date | None) -> float:
    if not start or not end or end <= start:
        return 1.0
    today = date.today()
    if today <= start:
        return 0.0
    if today >= end:
        return 1.0
    return safe_divide((today - start).days + 1, (end - start).days + 1)


def enrollment_metrics(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    enrollment_goal: float,
    start: date | None,
    end: date | None,
) -> dict[str, float]:
    total_leads = int(len(leads))
    applicants = int(_text_column(leads, "lifecycle_stage").str.lower().eq("applicant").sum())
    enrolled = int(len(enrollments))
    today = date.today()
    if start or end:
        weekly = enrolled
    else:
        week_start = today - timedelta(days=today.weekday())
        weekly_dates = pd.to_datetime(enrollments.get("enrolled_date", pd.Series(dtype=object)), errors="coerce", utc=True)
        weekly = int(((weekly_dates.dt.date >= week_start) & (weekly_dates.dt.date <= today)).sum())
    return {
        "total_enrollments": float(enrolled),
        "enrollment_goal": float(enrollment_goal or 0),
        "pct_goal": safe_divide(enrolled, enrollment_goal),
        "remaining_to_goal": max(float(enrollment_goal or 0) - enrolled, 0),
        "weekly_enrollments": float(weekly),
        "leads": float(total_leads),
        "applicants": float(applicants),
        "enrolled": float(enrolled),
        "lead_to_applicant_rate": safe_divide(applicants, total_leads),
        "applicant_to_enrolled_rate": safe_divide(enrolled, applicants),
        "lead_to_enrolled_rate": safe_divide(enrolled, total_leads),
    }


def _status_counts(group: pd.DataFrame) -> dict[str, int]:
    return {
        "duplicate_count": int(exact_status_mask(group, STATUS_DUPLICATE).sum()),
        "do_not_contact_count": int(exact_status_mask(group, STATUS_DO_NOT_CONTACT).sum()),
        "unqualified_count": int(exact_status_mask(group, STATUS_UNQUALIFIED).sum()),
        "lead_status_dead_lead_count": int(exact_status_mask(group, STATUS_DEAD_LEAD).sum()),
    }


def assign_udr_category(row: pd.Series | dict[str, Any], expected_goal: float, benchmarks: dict[str, float]) -> str:
    def value(key: str) -> Any:
        return row.get(key) if isinstance(row, dict) else row.get(key)

    if float(value("leads_assigned") or 0) == 0 or float(value("goal") or 0) == 0:
        return "Insufficient Data"
    if float(value("pct_goal") or 0) >= 1:
        return "Above Goal"
    if float(value("pct_goal") or 0) >= expected_goal:
        return "On Pace"

    activities = value("avg_activities_per_lead")
    if activities is not None and pd.notna(activities):
        if float(activities) < benchmarks["min_activities_per_lead"]:
            return "Low Activity"

    talk_time = value("avg_talk_time_per_lead")
    if talk_time is not None and pd.notna(talk_time):
        if float(talk_time) < benchmarks["min_talk_time_per_lead"]:
            return "Low Talk Time"

    if float(value("lead_to_applicant_rate") or 0) < benchmarks["min_lead_to_applicant_rate"]:
        return "Low Conversion"
    if float(value("no_recent_activity_rate") or 0) > benchmarks["max_no_recent_activity_rate"]:
        return "High No-Recent-Activity Rate"
    if float(value("duplicate_rate") or 0) > benchmarks["max_duplicate_rate"]:
        return "High Duplicate Rate"
    if float(value("unqualified_rate") or 0) > benchmarks["max_unqualified_rate"]:
        return "High Unqualified Rate"
    return "Below Goal"


def udr_scorecard(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    goals: dict[str, float],
    expected_goal: float,
    benchmarks: dict[str, float] | None = None,
    no_recent_days: int = 7,
) -> pd.DataFrame:
    benchmarks = benchmarks or BENCHMARKS
    udrs = sorted(
        {
            *_text_column(leads, "assigned_udr").dropna().loc[lambda s: s.ne("")].unique().tolist(),
            *_text_column(enrollments, "enrollment_udr").dropna().loc[lambda s: s.ne("")].unique().tolist(),
            *[key for key in goals if key],
        }
    )
    rows: list[dict[str, Any]] = []
    dead_exists = exact_status_mask(leads, STATUS_DEAD_LEAD).any()
    for udr in udrs:
        lead_group = leads[_text_column(leads, "assigned_udr").eq(udr)].copy()
        enrollment_group = enrollments[_text_column(enrollments, "enrollment_udr").eq(udr)].copy()
        leads_assigned = int(len(lead_group))
        applicants = int(_text_column(lead_group, "lifecycle_stage").str.lower().eq("applicant").sum())
        enroll_count = int(len(enrollment_group))
        goal = float(goals.get(udr, 0))
        status_counts = _status_counts(lead_group)
        no_recent_count = int(no_recent_activity_mask(lead_group, no_recent_days).sum()) if leads_assigned else 0
        row: dict[str, Any] = {
            "udr": udr,
            "actual_enrollments": enroll_count,
            "goal": goal,
            "pct_goal": safe_divide(enroll_count, goal),
            "leads_assigned": leads_assigned,
            "applicants": applicants,
            "lead_to_applicant_rate": safe_divide(applicants, leads_assigned),
            "lead_to_enrolled_rate": safe_divide(enroll_count, leads_assigned),
            "no_recent_activity_count": no_recent_count,
            "no_recent_activity_rate": safe_divide(no_recent_count, leads_assigned),
            "duplicate_count": status_counts["duplicate_count"],
            "duplicate_rate": safe_divide(status_counts["duplicate_count"], leads_assigned),
            "do_not_contact_count": status_counts["do_not_contact_count"],
            "do_not_contact_rate": safe_divide(status_counts["do_not_contact_count"], leads_assigned),
            "unqualified_count": status_counts["unqualified_count"],
            "unqualified_rate": safe_divide(status_counts["unqualified_count"], leads_assigned),
        }
        for source_col, total_col, avg_col in [
            ("calls", "calls", None),
            ("talk_time", "talk_time", "avg_talk_time_per_lead"),
            ("activities", "activities", "avg_activities_per_lead"),
        ]:
            if source_col in lead_group.columns:
                total = float(pd.to_numeric(lead_group[source_col], errors="coerce").fillna(0).sum())
                row[total_col] = total
                if avg_col:
                    row[avg_col] = safe_divide(total, leads_assigned)
            else:
                row[total_col] = pd.NA
                if avg_col:
                    row[avg_col] = pd.NA
        if dead_exists:
            row["lead_status_dead_lead_count"] = status_counts["lead_status_dead_lead_count"]
            row["lead_status_dead_lead_rate"] = safe_divide(status_counts["lead_status_dead_lead_count"], leads_assigned)
        row["performance_category"] = assign_udr_category(row, expected_goal, benchmarks)
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["pct_goal", "actual_enrollments", "leads_assigned"], ascending=[False, False, False])
    return out


def enrollment_group(enrollments: pd.DataFrame, column: str) -> pd.DataFrame:
    if enrollments.empty or column not in enrollments.columns:
        return pd.DataFrame(columns=[column, "actual_enrollments"])
    out = enrollments.groupby(_text_column(enrollments, column), dropna=False).size().reset_index(name="actual_enrollments")
    out = out.rename(columns={"index": column})
    out[column] = out[column].replace("", "Not Available")
    return out.sort_values("actual_enrollments", ascending=False)


def funnel_metrics_by(leads: pd.DataFrame, enrollments: pd.DataFrame, column: str) -> pd.DataFrame:
    lead_counts = pd.DataFrame(columns=[column, "leads", "applicants"])
    if not leads.empty and column in leads.columns:
        grouped = leads.groupby(_text_column(leads, column), dropna=False)
        lead_counts = grouped.agg(leads=("record_id", "size")).reset_index()
        lead_counts["applicants"] = grouped.apply(
            lambda group: int(_text_column(group, "lifecycle_stage").str.lower().eq("applicant").sum()),
            include_groups=False,
        ).values
    enroll_counts = enrollment_group(enrollments, column).rename(columns={"actual_enrollments": "enrolled"})
    frames = []
    if not lead_counts.empty:
        frames.append(lead_counts.set_index(column))
    if not enroll_counts.empty:
        frames.append(enroll_counts.set_index(column))
    if not frames:
        return pd.DataFrame(columns=[column, "leads", "applicants", "enrolled"])
    out = pd.concat(frames, axis=1).reset_index().rename(columns={"index": column})
    for metric in ["leads", "applicants", "enrolled"]:
        if metric not in out.columns:
            out[metric] = 0
        out[metric] = pd.to_numeric(out[metric], errors="coerce").fillna(0)
    out[column] = out[column].replace("", "Not Available")
    out["lead_to_applicant_rate"] = out.apply(lambda row: safe_divide(row["applicants"], row["leads"]), axis=1)
    out["lead_to_enrolled_rate"] = out.apply(lambda row: safe_divide(row["enrolled"], row["leads"]), axis=1)
    return out.sort_values(["enrolled", "leads"], ascending=False)


def vendor_performance(leads: pd.DataFrame, enrollments: pd.DataFrame, no_recent_days: int = 7) -> pd.DataFrame:
    base = funnel_metrics_by(leads, enrollments, "vendor")
    if base.empty:
        return base
    rows = []
    for _, row in base.iterrows():
        vendor = row["vendor"]
        group = leads[_text_column(leads, "vendor").replace("", "Not Available").eq(vendor)].copy()
        leads_count = float(row["leads"] or 0)
        status_counts = _status_counts(group)
        no_recent_count = int(no_recent_activity_mask(group, no_recent_days).sum()) if len(group) else 0
        extra = {
            "duplicate_rate": safe_divide(status_counts["duplicate_count"], leads_count),
            "do_not_contact_rate": safe_divide(status_counts["do_not_contact_count"], leads_count),
            "unqualified_rate": safe_divide(status_counts["unqualified_count"], leads_count),
            "no_recent_activity_rate": safe_divide(no_recent_count, leads_count),
        }
        rows.append({**row.to_dict(), **extra})
    return pd.DataFrame(rows).sort_values(["enrolled", "leads"], ascending=False)


def activity_summary_by_udr(leads: pd.DataFrame, no_recent_days: int = 7) -> pd.DataFrame:
    if leads.empty or "assigned_udr" not in leads.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for udr, group in leads.groupby(_text_column(leads, "assigned_udr"), dropna=False):
        label = udr if udr else "Not Available"
        lead_count = int(len(group))
        no_recent = int(no_recent_activity_mask(group, no_recent_days).sum()) if lead_count else 0
        row: dict[str, Any] = {
            "udr": label,
            "leads_assigned": lead_count,
            "no_recent_activity_count": no_recent,
            "no_recent_activity_rate": safe_divide(no_recent, lead_count),
        }
        for source_col, total_col, avg_col in [
            ("calls", "calls", None),
            ("talk_time", "talk_time", "avg_talk_time_per_lead"),
            ("activities", "activities", "avg_activities_per_lead"),
        ]:
            if source_col in group.columns:
                total = float(pd.to_numeric(group[source_col], errors="coerce").fillna(0).sum())
                row[total_col] = total
                if avg_col:
                    row[avg_col] = safe_divide(total, lead_count)
            else:
                row[total_col] = pd.NA
                if avg_col:
                    row[avg_col] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows).sort_values("no_recent_activity_rate", ascending=False)


def qa_summary(
    leads: pd.DataFrame,
    enrollments: pd.DataFrame,
    hubspot_enrolled_count: int,
    last_refresh: str,
) -> pd.DataFrame:
    def missing(frame: pd.DataFrame, column: str) -> int:
        if frame.empty or column not in frame.columns:
            return 0
        return int(_text_column(frame, column).eq("").sum())

    duplicate_emails = 0
    if "email" in leads.columns:
        emails = _text_column(leads, "email").str.lower()
        duplicate_emails = int(emails[emails.ne("")].duplicated().sum())
    duplicate_ids = 0
    if "record_id" in leads.columns:
        ids = _text_column(leads, "record_id")
        duplicate_ids = int(ids[ids.ne("")].duplicated().sum())
    missing_last_activity = len(leads)
    if not leads.empty and "last_activity_date" in leads.columns:
        missing_last_activity = int(pd.to_datetime(leads["last_activity_date"], errors="coerce").isna().sum())
    elif leads.empty:
        missing_last_activity = 0
    rows = [
        {"check": "Missing UDR", "count": missing(leads, "assigned_udr")},
        {"check": "Missing Program", "count": missing(enrollments, "program") + missing(leads, "program")},
        {"check": "Missing Degree", "count": missing(enrollments, "degree") + missing(leads, "degree")},
        {"check": "Missing Vendor", "count": missing(enrollments, "vendor") + missing(leads, "vendor")},
        {"check": "Missing Modality", "count": missing(enrollments, "modality") + missing(leads, "modality")},
        {"check": "Missing Lead Status", "count": missing(leads, "lead_status")},
        {"check": "Missing Lifecycle Stage", "count": missing(leads, "lifecycle_stage")},
        {"check": "Missing Last Activity Date", "count": missing_last_activity},
        {"check": "Duplicate emails", "count": duplicate_emails},
        {"check": "Duplicate Record IDs", "count": duplicate_ids},
        {
            "check": "Enrollment tracker count vs HubSpot enrolled count",
            "count": abs(int(len(enrollments)) - int(hubspot_enrolled_count)),
            "details": f"Tracker={len(enrollments):,}; HubSpot enrolled={hubspot_enrolled_count:,}.",
        },
        {"check": "Last refresh timestamp", "count": 0, "details": last_refresh},
    ]
    out = pd.DataFrame(rows)
    out["status"] = out["count"].apply(lambda value: "Pass" if int(value or 0) == 0 else "Review")
    return out[["check", "status", "count", "details"]] if "details" in out.columns else out
