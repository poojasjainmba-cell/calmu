from __future__ import annotations

from typing import Any

import pandas as pd

from tuition_loader import get_degree_tuition, load_degree_tuition_config


REVENUE_WINDOWS = {
    "six_month_revenue": 6,
    "twelve_month_revenue": 12,
    "twenty_four_month_revenue": 24,
}
POTENTIAL_REVENUE_WINDOWS = {
    "potential_six_month_revenue": 6,
    "potential_twelve_month_revenue": 12,
    "potential_twenty_four_month_revenue": 24,
}


def _positive_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)


def _window_revenue(monthly: pd.Series, total: pd.Series, months: int) -> pd.Series:
    return (monthly * months).clip(upper=total)


def add_program_tuition_fields(
    df: pd.DataFrame,
    program_column: str = "program",
    config: dict[str, dict[str, Any]] | None = None,
    degree_level_column: str = "degree_level",
) -> pd.DataFrame:
    output = df.copy()
    config = config or load_degree_tuition_config()
    if program_column not in output.columns:
        output[program_column] = pd.NA
    explicit_degree = (
        output[degree_level_column]
        if degree_level_column in output.columns
        else pd.Series([pd.NA] * len(output), index=output.index)
    )
    if "revenue_confidence" in output.columns:
        confidence = output["revenue_confidence"].fillna("").astype(str)
        explicit_degree = explicit_degree.where(confidence.eq("High"), pd.NA)

    rows = [
        get_degree_tuition(program, config, explicit_degree_level=degree)
        for program, degree in zip(output[program_column], explicit_degree)
    ]
    tuition_df = pd.DataFrame(rows, index=output.index)
    for column in (
        "degree_level",
        "program_degree_level",
        "tuition_per_credit",
        "credits_required",
        "estimated_total_tuition",
        "estimated_annual_tuition",
        "program_duration_years",
        "program_duration_months",
        "duration_note",
        "tuition_estimate_source",
        "revenue_confidence",
    ):
        output[column] = tuition_df[column]
    return output


def add_program_duration_fields(
    df: pd.DataFrame,
    program_column: str = "program",
    config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    return add_program_tuition_fields(df, program_column=program_column, config=config)


def add_program_revenue_fields(
    df: pd.DataFrame,
    total_revenue_column: str = "revenue_attributed",
    program_column: str = "program",
    config: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output = add_program_tuition_fields(df, program_column=program_column, config=config)
    total = (
        _positive_numeric(output[total_revenue_column])
        if total_revenue_column in output.columns
        else pd.Series([0.0] * len(output), index=output.index)
    )
    potential = _positive_numeric(output["estimated_total_tuition"])
    months = pd.to_numeric(output["program_duration_months"], errors="coerce")
    years = pd.to_numeric(output["program_duration_years"], errors="coerce")

    output["total_program_revenue"] = total
    output["annualized_program_revenue"] = (total / years).where(years > 0, 0).fillna(0)
    output["monthly_program_revenue"] = (total / months).where(months > 0, 0).fillna(0)
    for column, window_months in REVENUE_WINDOWS.items():
        output[column] = _window_revenue(output["monthly_program_revenue"], total, window_months)

    output["potential_program_revenue"] = potential
    output["potential_annualized_program_revenue"] = (potential / years).where(years > 0, 0).fillna(0)
    output["potential_monthly_program_revenue"] = (potential / months).where(months > 0, 0).fillna(0)
    for column, window_months in POTENTIAL_REVENUE_WINDOWS.items():
        output[column] = _window_revenue(output["potential_monthly_program_revenue"], potential, window_months)

    status_text = (
        output.get("enrollment_status", pd.Series([""] * len(output), index=output.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    deal_stage_text = (
        output.get("deal_stage", pd.Series([""] * len(output), index=output.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    is_won = (
        output.get("is_won", pd.Series([False] * len(output), index=output.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )
    is_lost = (
        output.get("is_lost", pd.Series([False] * len(output), index=output.index))
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    ) | deal_stage_text.str.contains("lost|denied|rejected", na=False)
    enrolled = status_text.str.contains("enrolled|ea signed|active|registered|accepted", na=False)
    enrolled_or_won = enrolled | is_won | deal_stage_text.str.contains("won", na=False)
    open_pipeline = (~enrolled_or_won) & (~is_lost)

    output["potential_revenue"] = potential.where(open_pipeline, 0)
    output["enrolled_revenue"] = potential.where(enrolled_or_won, 0)
    output["open_pipeline_potential_revenue"] = potential.where(open_pipeline, 0)

    def note(row: pd.Series) -> str:
        degree_level = row.get("program_degree_level")
        duration_months = row.get("program_duration_months")
        duration_note = str(row.get("duration_note") or "").strip()
        revenue_source = str(row.get("program_revenue_source") or "").strip()
        if not degree_level:
            return "Program duration could not be matched; update config/degree_tuition.json."
        if float(row.get("total_program_revenue") or 0) <= 0:
            return (
                f"{degree_level} potential tuition is based on CalMU published rates over "
                f"{duration_months:g} months; realized revenue is not counted until a deal is won/countable."
            )
        if revenue_source == "calmu_published_tuition_config":
            source_text = "published CalMU tuition"
        elif revenue_source == "contact_program_total_tuition":
            source_text = "HubSpot contact tuition"
        elif revenue_source == "deal_revenue":
            source_text = "HubSpot deal revenue"
        else:
            source_text = "available tuition/revenue"
        suffix = f"; {duration_note}." if duration_note else "."
        return f"{degree_level} revenue uses {source_text} over {duration_months:g} months{suffix}"

    output["revenue_realization_note"] = output.apply(note, axis=1)
    return output
