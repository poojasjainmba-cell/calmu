from __future__ import annotations

from typing import Any

import pandas as pd

from config import VENDOR_COSTS_PATH


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


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    series = df[column]
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def _series(df: pd.DataFrame, column: str, default: Any = pd.NA) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def _unique_count(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].dropna().astype(str).nunique())


def _sum_unique_contacts(df: pd.DataFrame, column: str) -> float:
    if df.empty or column not in df.columns:
        return 0.0
    work = df.copy()
    if "contact_id" in work.columns:
        work = work.drop_duplicates(subset=["contact_id"])
    return _sum_money(work, column)


def _estimated_enrolled_revenue_and_students(
    contacts_df: pd.DataFrame,
    fact_df: pd.DataFrame,
) -> tuple[float, int]:
    contact_values: list[pd.Series] = []
    no_contact_revenue = 0.0

    enrolled_revenue = pd.to_numeric(_series(contacts_df, "enrolled_revenue", 0), errors="coerce").fillna(0)
    enrolled_contacts = contacts_df[enrolled_revenue > 0].copy()
    if not enrolled_contacts.empty and "contact_id" in enrolled_contacts.columns:
        enrolled_contacts["_estimated_enrolled_revenue"] = pd.to_numeric(
            enrolled_contacts["enrolled_revenue"],
            errors="coerce",
        ).fillna(0)
        contact_values.append(
            enrolled_contacts.groupby(enrolled_contacts["contact_id"].astype(str))["_estimated_enrolled_revenue"].max()
        )
    else:
        no_contact_revenue += float(enrolled_revenue[enrolled_revenue > 0].sum())

    won_fact = _countable_won_fact(fact_df)
    if not won_fact.empty:
        won_fact = won_fact.copy()
        won_fact["_estimated_enrolled_revenue"] = pd.to_numeric(
            _series(won_fact, "total_program_revenue", 0),
            errors="coerce",
        ).fillna(0)
        if "contact_id" in won_fact.columns:
            contact_id = won_fact["contact_id"].fillna("").astype(str).str.strip()
            with_contact = won_fact[contact_id.ne("")]
            without_contact = won_fact[contact_id.eq("")]
            if not with_contact.empty:
                contact_values.append(
                    with_contact.groupby(with_contact["contact_id"].astype(str))["_estimated_enrolled_revenue"].max()
                )
            if not without_contact.empty:
                dedupe_columns = ["deal_id"] if "deal_id" in without_contact.columns else None
                no_contact_revenue += float(
                    without_contact.drop_duplicates(subset=dedupe_columns)["_estimated_enrolled_revenue"].sum()
                )
        else:
            dedupe_columns = ["deal_id"] if "deal_id" in won_fact.columns else None
            no_contact_revenue += float(won_fact.drop_duplicates(subset=dedupe_columns)["_estimated_enrolled_revenue"].sum())

    if not contact_values:
        return no_contact_revenue, int(no_contact_revenue > 0)

    combined = pd.concat(contact_values).groupby(level=0).max()
    return float(combined.sum() + no_contact_revenue), int(combined.size + int(no_contact_revenue > 0))


def _countable_won_fact(fact_df: pd.DataFrame) -> pd.DataFrame:
    if fact_df.empty:
        return fact_df.copy()
    mask = _bool_series(fact_df, "deal_countable") & _bool_series(fact_df, "is_won")
    return fact_df[mask].copy()


def _sum_money(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    source_columns = [column]
    if column == "total_program_revenue":
        source_columns.extend(["program_revenue_input", "revenue_attributed"])
    for source_column in source_columns:
        if source_column in df.columns:
            return float(pd.to_numeric(df[source_column], errors="coerce").fillna(0).sum())
    return 0.0


def _revenue_totals(df: pd.DataFrame) -> dict[str, float]:
    totals = {
        column: _sum_money(df, column)
        for column in [*REVENUE_COLUMNS, *POTENTIAL_REVENUE_COLUMNS, *PIPELINE_REVENUE_COLUMNS]
    }
    totals["revenue"] = totals["total_program_revenue"]
    return totals


def _potential_totals(df: pd.DataFrame) -> dict[str, float]:
    return {column: _sum_money(df, column) for column in [*POTENTIAL_REVENUE_COLUMNS, *PIPELINE_REVENUE_COLUMNS]}


def _vendor_series(df: pd.DataFrame) -> pd.Series:
    if "vendor" in df.columns:
        vendor = df["vendor"].fillna("").astype(str).str.strip()
        if vendor.ne("").any():
            return vendor.replace("", "Unknown")
    if "paid_vendor" in df.columns:
        vendor = df["paid_vendor"].fillna("").astype(str).str.strip()
        if vendor.ne("").any():
            return vendor.replace("", "Unknown")
    utm = _series(df, "utm_source", "").fillna("").astype(str).str.strip()
    source = _series(df, "source_group", "Unknown").fillna("Unknown").astype(str).str.strip()
    vendor = utm.where(utm.ne(""), source)
    return vendor.replace("", "Unknown")


def _load_vendor_costs() -> pd.DataFrame:
    if not VENDOR_COSTS_PATH.exists():
        return pd.DataFrame(columns=["vendor", "month", "spend", "notes"])
    costs = pd.read_csv(VENDOR_COSTS_PATH)
    for column in ("vendor", "month", "spend", "notes"):
        if column not in costs.columns:
            costs[column] = pd.NA
    costs["vendor"] = costs["vendor"].fillna("").astype(str).str.strip()
    costs["spend"] = pd.to_numeric(costs["spend"], errors="coerce").fillna(0)
    return costs[costs["vendor"].ne("") & (costs["spend"] > 0)]


def _contacted_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series([False] * len(df), index=df.index)
    for column in ("last_activity_date", "last_sales_activity_timestamp", "first_outreach_date", "first_engagement_date"):
        if column in df.columns:
            mask = mask | df[column].notna() & df[column].astype(str).str.strip().ne("")
    if "num_notes" in df.columns:
        mask = mask | (pd.to_numeric(df["num_notes"], errors="coerce").fillna(0) > 0)
    return mask


def _paid_vendor_performance(paid_contacts: pd.DataFrame, fact_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "vendor",
        "source_group",
        "campaign",
        "leads",
        "paid_leads",
        "contacted_leads",
        "uncontacted_leads",
        "hot_leads",
        "dead_leads",
        "deals_created",
        "enrolled_students",
        "actual_revenue",
        "estimated_enrolled_revenue",
        "open_potential_revenue",
        "close_rate",
        "enrollment_rate",
        "average_response_time",
        "average_days_to_enroll",
        "average_sales_touches_before_enrollment",
        "cost_per_lead",
        "revenue_per_lead",
        "estimated_roi",
        "leakage_score",
        "vendor_confidence",
        "vendor_source_field",
    ]
    if paid_contacts.empty:
        return pd.DataFrame(columns=columns)

    contacts = paid_contacts.copy()
    contacts["vendor"] = _vendor_series(contacts)
    contacts["source_group"] = _series(contacts, "source_group", "Unknown").fillna("Unknown").astype(str).replace("", "Unknown")
    contacts["campaign"] = _series(contacts, "utm_campaign", "Unknown").fillna("Unknown").astype(str).replace("", "Unknown")
    contacts["vendor_confidence"] = _series(contacts, "vendor_confidence", "Low").fillna("Low").astype(str).replace("", "Low")
    contacts["vendor_source_field"] = _series(contacts, "vendor_source_field", "").fillna("").astype(str)

    fact = fact_df.copy()
    if not fact.empty:
        fact["vendor"] = _vendor_series(fact)
        fact["source_group"] = _series(fact, "source_group", "Unknown").fillna("Unknown").astype(str).replace("", "Unknown")
        fact["campaign"] = _series(fact, "utm_campaign", "Unknown").fillna("Unknown").astype(str).replace("", "Unknown")

    costs = _load_vendor_costs()
    spend_by_vendor = costs.groupby("vendor")["spend"].sum().to_dict() if not costs.empty else {}
    paid_leads_by_vendor = contacts.groupby("vendor")["contact_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []

    for (vendor, source_group, campaign), group in contacts.groupby(["vendor", "source_group", "campaign"], dropna=False):
        contact_ids = set(group.get("contact_id", pd.Series(dtype=str)).dropna().astype(str))
        related_fact = (
            fact[fact.get("contact_id", pd.Series(dtype=str)).astype(str).isin(contact_ids)].copy()
            if contact_ids and not fact.empty and "contact_id" in fact.columns
            else pd.DataFrame()
        )
        won_fact = _countable_won_fact(related_fact)
        paid_leads = _unique_count(group, "contact_id")
        contacted_leads = _unique_count(group[_contacted_mask(group)], "contact_id")
        hot_leads = int((_series(group, "lead_temperature", "") == "Hot").sum())
        dead_leads = int((_series(group, "lead_temperature", "") == "Dead").sum())
        deals_created = _unique_count(related_fact, "deal_id")
        estimated_enrolled_revenue, enrolled_students = _estimated_enrolled_revenue_and_students(group, related_fact)
        actual_revenue = _sum_money(won_fact, "revenue_attributed")
        open_potential_revenue = _sum_unique_contacts(group, "open_pipeline_potential_revenue")

        response_hours = (
            pd.to_numeric(_series(group, "time_to_first_engagement", pd.NA), errors="coerce").dropna() / 3600000
        )
        days_to_enroll = pd.to_numeric(won_fact.get("days_to_close", pd.Series(dtype=float)), errors="coerce").dropna()
        enrolled_contacts = group[pd.to_numeric(_series(group, "enrolled_revenue", 0), errors="coerce").fillna(0) > 0]
        touches = pd.to_numeric(_series(enrolled_contacts, "num_notes", pd.NA), errors="coerce").dropna()
        uncontacted_leads = max(paid_leads - contacted_leads, 0)
        leakage_score = min(_safe_div(uncontacted_leads + dead_leads, paid_leads) * 100, 100.0)

        vendor_spend = float(spend_by_vendor.get(str(vendor), 0))
        vendor_paid_leads = int(paid_leads_by_vendor.get(str(vendor), 0))
        allocated_spend = vendor_spend * _safe_div(paid_leads, vendor_paid_leads)
        revenue_for_roi = estimated_enrolled_revenue or actual_revenue
        confidence_mix = group["vendor_confidence"].value_counts()
        source_fields = sorted(field for field in group["vendor_source_field"].dropna().astype(str).unique() if field)

        rows.append(
            {
                "vendor": vendor,
                "source_group": source_group,
                "campaign": campaign,
                "leads": paid_leads,
                "paid_leads": paid_leads,
                "contacted_leads": contacted_leads,
                "uncontacted_leads": uncontacted_leads,
                "hot_leads": hot_leads,
                "dead_leads": dead_leads,
                "deals_created": deals_created,
                "enrolled_students": enrolled_students,
                "actual_revenue": actual_revenue,
                "estimated_enrolled_revenue": estimated_enrolled_revenue,
                "open_potential_revenue": open_potential_revenue,
                "close_rate": _safe_div(_unique_count(won_fact, "deal_id"), paid_leads),
                "enrollment_rate": _safe_div(enrolled_students, paid_leads),
                "average_response_time": float(response_hours.mean()) if not response_hours.empty else 0.0,
                "average_days_to_enroll": float(days_to_enroll.mean()) if not days_to_enroll.empty else 0.0,
                "average_sales_touches_before_enrollment": float(touches.mean()) if not touches.empty else 0.0,
                "cost_per_lead": _safe_div(allocated_spend, paid_leads),
                "revenue_per_lead": _safe_div(revenue_for_roi, paid_leads),
                "estimated_roi": _safe_div(revenue_for_roi - allocated_spend, allocated_spend),
                "leakage_score": leakage_score,
                "vendor_confidence": " / ".join(
                    f"{level}: {int(confidence_mix.get(level, 0))}" for level in ("High", "Medium", "Low")
                ),
                "vendor_source_field": ", ".join(source_fields) or "Unknown",
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["open_potential_revenue", "estimated_enrolled_revenue", "paid_leads"],
        ascending=False,
    )


def _group_revenue(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if df.empty or group_column not in df.columns:
        return pd.DataFrame(columns=[group_column, *REVENUE_COLUMNS, *POTENTIAL_REVENUE_COLUMNS, *PIPELINE_REVENUE_COLUMNS])
    grouped = df.groupby(group_column, dropna=False)
    rows = []
    for group_value, group in grouped:
        row = {group_column: group_value}
        row.update(_revenue_totals(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _group_money_columns(df: pd.DataFrame, group_column: str, columns: list[str]) -> pd.DataFrame:
    if df.empty or group_column not in df.columns:
        return pd.DataFrame(columns=[group_column, *columns])
    rows = []
    for group_value, group in df.groupby(group_column, dropna=False):
        row = {group_column: group_value}
        for column in columns:
            row[column] = _sum_money(group, column)
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_executive_metrics(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> dict[str, Any]:
    total_leads = _unique_count(contacts_df, "contact_id")
    paid_mask = _bool_series(contacts_df, "paid_lead_flag")
    paid_leads = _unique_count(contacts_df[paid_mask], "contact_id") if not contacts_df.empty else 0
    organic_leads = max(total_leads - paid_leads, 0)
    hot_leads = int((contacts_df.get("lead_temperature", pd.Series(dtype=str)) == "Hot").sum())
    dead_leads = int((contacts_df.get("lead_temperature", pd.Series(dtype=str)) == "Dead").sum())

    won_fact = _countable_won_fact(fact_df)
    won_deals = _unique_count(won_fact, "deal_id")
    close_rate = _safe_div(won_deals, total_leads)

    paid_won_fact = won_fact[_bool_series(won_fact, "paid_lead_flag")] if not won_fact.empty else won_fact
    paid_close_rate = _safe_div(_unique_count(paid_won_fact, "deal_id"), paid_leads)
    days_to_close = pd.to_numeric(won_fact.get("days_to_close", pd.Series(dtype=float)), errors="coerce").dropna()
    revenue_totals = _revenue_totals(won_fact)
    revenue_totals.update(_potential_totals(contacts_df))

    return {
        "total_leads": total_leads,
        "paid_leads": paid_leads,
        "organic_leads": organic_leads,
        "hot_leads": hot_leads,
        "dead_leads": dead_leads,
        "won_deals": won_deals,
        **revenue_totals,
        "close_rate": close_rate,
        "paid_close_rate": paid_close_rate,
        "average_days_to_close": float(days_to_close.mean()) if not days_to_close.empty else 0.0,
        "median_days_to_close": float(days_to_close.median()) if not days_to_close.empty else 0.0,
        "revenue_per_lead": _safe_div(revenue_totals["total_program_revenue"], total_leads),
        "revenue_per_paid_lead": _safe_div(revenue_totals["total_program_revenue"], paid_leads),
    }


def calculate_salesman_metrics(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> pd.DataFrame:
    names = sorted(
        set(contacts_df.get("salesman_name", pd.Series(dtype=str)).dropna().astype(str))
        | set(fact_df.get("salesman_name", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, Any]] = []
    won_fact = _countable_won_fact(fact_df)

    for name in names:
        contacts = contacts_df[contacts_df.get("salesman_name", pd.Series(dtype=str)).astype(str) == name]
        all_fact = fact_df[fact_df.get("salesman_name", pd.Series(dtype=str)).astype(str) == name]
        fact = won_fact[won_fact.get("salesman_name", pd.Series(dtype=str)).astype(str) == name]
        total_leads = _unique_count(contacts, "contact_id")
        paid_mask = _bool_series(contacts, "paid_lead_flag")
        paid_leads = _unique_count(contacts[paid_mask], "contact_id") if not contacts.empty else 0
        organic_leads = max(total_leads - paid_leads, 0)
        won_deals = _unique_count(fact, "deal_id")
        revenue_totals = _revenue_totals(fact)
        revenue_totals.update(_potential_totals(contacts))
        paid_fact = fact[_bool_series(fact, "paid_lead_flag")] if not fact.empty else fact
        days = pd.to_numeric(fact.get("days_to_close", pd.Series(dtype=float)), errors="coerce").dropna()
        actual_enrolled_revenue = _sum_money(fact, "revenue_attributed")
        estimated_enrolled_revenue, enrolled_students = _estimated_enrolled_revenue_and_students(contacts, all_fact)
        open_pipeline_potential_revenue = _sum_unique_contacts(contacts, "open_pipeline_potential_revenue")
        hot_contacts = contacts[contacts.get("lead_temperature", pd.Series(dtype=str)) == "Hot"]
        hot_lead_potential_revenue = _sum_unique_contacts(hot_contacts, "open_pipeline_potential_revenue")
        paid_contacts = contacts[paid_mask] if not contacts.empty else contacts
        paid_lead_potential_revenue = _sum_unique_contacts(paid_contacts, "open_pipeline_potential_revenue")
        paid_won_contact_ids = set(
            paid_fact.get("contact_id", pd.Series(dtype=str)).dropna().astype(str)
        )
        paid_contacts_without_wins = paid_contacts[
            ~paid_contacts.get("contact_id", pd.Series(dtype=str)).astype(str).isin(paid_won_contact_ids)
        ]
        stale_paid = pd.to_numeric(
            _series(paid_contacts_without_wins, "days_since_last_activity", 0),
            errors="coerce",
        ).fillna(9999)
        paid_lead_leakage = _unique_count(
            paid_contacts_without_wins[
                (stale_paid >= 7)
                | ~_bool_series(paid_contacts_without_wins, "has_open_deal")
            ],
            "contact_id",
        )
        action_load = int(
            (contacts.get("lead_temperature", pd.Series(dtype=str)) == "Hot").sum()
            + _bool_series(contacts, "has_open_deal").sum()
            + paid_lead_leakage
        )
        confidence_counts = (
            contacts.get("revenue_confidence", pd.Series(dtype=str))
            .fillna("Low")
            .astype(str)
            .replace("", "Low")
            .value_counts()
        )
        revenue_confidence_mix = " / ".join(
            f"{level}: {int(confidence_counts.get(level, 0))}" for level in ("High", "Medium", "Low")
        )

        row = {
            "salesman_name": name,
            "action_load": action_load,
            "paid_lead_leakage": paid_lead_leakage,
            "total_leads_assigned": total_leads,
            "paid_leads_assigned": paid_leads,
            "organic_leads_assigned": organic_leads,
            "hot_leads": int((contacts.get("lead_temperature", pd.Series(dtype=str)) == "Hot").sum()),
            "warm_leads": int((contacts.get("lead_temperature", pd.Series(dtype=str)) == "Warm").sum()),
            "cold_leads": int((contacts.get("lead_temperature", pd.Series(dtype=str)) == "Cold").sum()),
            "dead_leads": int((contacts.get("lead_temperature", pd.Series(dtype=str)) == "Dead").sum()),
            "open_leads": int(_bool_series(contacts, "has_open_deal").sum()),
            "won_deals": won_deals,
            "close_rate": _safe_div(won_deals, total_leads),
            "paid_close_rate": _safe_div(_unique_count(paid_fact, "deal_id"), paid_leads),
            "average_days_to_close": float(days.mean()) if not days.empty else 0.0,
            "median_days_to_close": float(days.median()) if not days.empty else 0.0,
            "revenue_per_lead": _safe_div(revenue_totals["total_program_revenue"], total_leads),
            "revenue_per_paid_lead": _safe_div(revenue_totals["total_program_revenue"], paid_leads),
            "actual_enrolled_revenue": actual_enrolled_revenue,
            "estimated_enrolled_revenue": estimated_enrolled_revenue,
            "open_pipeline_potential_revenue": open_pipeline_potential_revenue,
            "hot_lead_potential_revenue": hot_lead_potential_revenue,
            "paid_lead_potential_revenue": paid_lead_potential_revenue,
            "average_estimated_revenue_per_lead": _safe_div(
                _sum_unique_contacts(contacts, "potential_program_revenue"),
                total_leads,
            ),
            "average_estimated_revenue_per_enrolled_student": _safe_div(
                estimated_enrolled_revenue,
                enrolled_students,
            ),
            "revenue_confidence_mix": revenue_confidence_mix,
        }
        row.update(revenue_totals)
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(
            ["action_load", "open_pipeline_potential_revenue", "paid_lead_leakage"],
            ascending=False,
        )
        if rows
        else pd.DataFrame()
    )


def calculate_paid_lead_metrics(contacts_df: pd.DataFrame, fact_df: pd.DataFrame) -> dict[str, Any]:
    paid_contacts = contacts_df[_bool_series(contacts_df, "paid_lead_flag")] if not contacts_df.empty else contacts_df
    paid_fact = _countable_won_fact(fact_df)
    paid_fact = paid_fact[_bool_series(paid_fact, "paid_lead_flag")] if not paid_fact.empty else paid_fact
    paid_leads = _unique_count(paid_contacts, "contact_id")
    paid_won_deals = _unique_count(paid_fact, "deal_id")
    revenue_totals = _revenue_totals(paid_fact)
    revenue_totals.update(_potential_totals(paid_contacts))
    paid_days = pd.to_numeric(paid_fact.get("days_to_close", pd.Series(dtype=float)), errors="coerce").dropna()

    by_source = (
        paid_contacts.groupby("source_group", dropna=False)["contact_id"].nunique().reset_index(name="paid_leads")
        if not paid_contacts.empty and "source_group" in paid_contacts.columns
        else pd.DataFrame(columns=["source_group", "paid_leads"])
    )
    by_campaign = (
        paid_contacts[paid_contacts["utm_campaign"].notna() & (paid_contacts["utm_campaign"].astype(str).str.strip() != "")]
        .groupby("utm_campaign", dropna=False)["contact_id"]
        .nunique()
        .reset_index(name="paid_leads")
        .sort_values("paid_leads", ascending=False)
        if not paid_contacts.empty and "utm_campaign" in paid_contacts.columns
        else pd.DataFrame()
    )
    by_vendor = _paid_vendor_performance(paid_contacts, fact_df)

    return {
        "paid_leads": paid_leads,
        "paid_revenue": revenue_totals["total_program_revenue"],
        **revenue_totals,
        "paid_close_rate": _safe_div(paid_won_deals, paid_leads),
        "average_paid_days_to_close": float(paid_days.mean()) if not paid_days.empty else 0.0,
        "revenue_per_paid_lead": _safe_div(revenue_totals["total_program_revenue"], paid_leads),
        "paid_leads_by_source": by_source,
        "paid_leads_by_campaign": by_campaign,
        "paid_vendor_performance": by_vendor,
        "vendor_cost_data_available": not _load_vendor_costs().empty,
    }


def calculate_stuck_lead_metrics(contacts_df: pd.DataFrame) -> dict[str, int]:
    if contacts_df.empty:
        return {
            "leads_no_activity_7_days": 0,
            "leads_no_activity_14_days": 0,
            "leads_no_activity_30_days": 0,
            "paid_leads_with_no_activity": 0,
            "paid_leads_older_30_days_no_open_deal": 0,
        }

    age = pd.to_numeric(contacts_df.get("lead_age_days", pd.Series([0] * len(contacts_df))), errors="coerce")
    inactive = pd.to_numeric(
        contacts_df.get("days_since_last_activity", pd.Series([pd.NA] * len(contacts_df))), errors="coerce"
    )
    never = _bool_series(contacts_df, "never_contacted")
    paid = _bool_series(contacts_df, "paid_lead_flag")
    has_open = _bool_series(contacts_df, "has_open_deal")

    def no_activity(days: int) -> pd.Series:
        return (inactive >= days) | (never & (age >= days))

    return {
        "leads_no_activity_7_days": _unique_count(contacts_df[no_activity(7)], "contact_id"),
        "leads_no_activity_14_days": _unique_count(contacts_df[no_activity(14)], "contact_id"),
        "leads_no_activity_30_days": _unique_count(contacts_df[no_activity(30)], "contact_id"),
        "paid_leads_with_no_activity": _unique_count(contacts_df[paid & no_activity(7)], "contact_id"),
        "paid_leads_older_30_days_no_open_deal": _unique_count(contacts_df[paid & (age > 30) & ~has_open], "contact_id"),
    }


def calculate_data_quality_metrics(
    contacts_df: pd.DataFrame,
    deals_df: pd.DataFrame,
    fact_df: pd.DataFrame,
) -> dict[str, int]:
    owner_text = _series(contacts_df, "salesman_id", "").fillna("").astype(str).str.strip()
    missing_owner = contacts_df[owner_text == ""]
    missing_source = contacts_df[
        (_series(contacts_df, "source_group", "Unknown").fillna("Unknown") == "Unknown")
        | (_series(contacts_df, "source_text", "").fillna("").astype(str).str.strip() == "")
    ]
    contacts_without_deals = fact_df[_series(fact_df, "fact_record_type", "") == "contact_without_deal"]
    deals_without_contacts = fact_df[_series(fact_df, "fact_record_type", "") == "deal_without_contact"]
    fallback = fact_df[_series(fact_df, "attribution_type", "") == "fallback_to_deal_owner"]
    unclear = fact_df[_bool_series(fact_df, "unclear_revenue_attribution")]

    return {
        "records_missing_owner": _unique_count(missing_owner, "contact_id"),
        "records_missing_source": _unique_count(missing_source, "contact_id"),
        "records_missing_contact_created_date": int(_series(contacts_df, "contact_created_at").isna().sum()),
        "records_missing_close_date": int(_series(deals_df, "close_date").isna().sum()),
        "records_missing_revenue": int((pd.to_numeric(_series(deals_df, "revenue", 0), errors="coerce").fillna(0) <= 0).sum()),
        "won_records_missing_program_revenue": int(
            (
                pd.to_numeric(
                    _series(_countable_won_fact(fact_df), "total_program_revenue", 0),
                    errors="coerce",
                ).fillna(0)
                <= 0
            ).sum()
        ),
        "contacts_without_deals": _unique_count(contacts_without_deals, "contact_id"),
        "deals_without_contacts": _unique_count(deals_without_contacts, "deal_id"),
        "deal_owner_fallback_used": int(fallback.shape[0]),
        "unclear_revenue_attribution": _unique_count(unclear, "deal_id"),
    }


def monthly_leads(contacts_df: pd.DataFrame) -> pd.DataFrame:
    if contacts_df.empty or "contact_created_at" not in contacts_df.columns:
        return pd.DataFrame(columns=["month", "leads"])
    dates = pd.to_datetime(contacts_df["contact_created_at"], errors="coerce", utc=True)
    output = contacts_df[dates.notna()].assign(month=dates[dates.notna()].dt.strftime("%Y-%m"))
    if output.empty or "contact_id" not in output.columns:
        return pd.DataFrame(columns=["month", "leads"])
    return output.groupby("month")["contact_id"].nunique().reset_index(name="leads")


def monthly_revenue(fact_df: pd.DataFrame, revenue_column: str = "total_program_revenue") -> pd.DataFrame:
    won_fact = _countable_won_fact(fact_df)
    if won_fact.empty or "close_date" not in won_fact.columns:
        return pd.DataFrame(columns=["month", "revenue"])
    dates = pd.to_datetime(won_fact["close_date"], errors="coerce", utc=True)
    output = won_fact[dates.notna()].assign(month=dates[dates.notna()].dt.strftime("%Y-%m"))
    source_column = next(
        (
            candidate
            for candidate in (revenue_column, "program_revenue_input", "revenue_attributed")
            if candidate in output.columns
        ),
        None,
    )
    if output.empty or not source_column:
        return pd.DataFrame(columns=["month", "revenue"])
    return output.groupby("month")[source_column].sum().reset_index(name="revenue")
