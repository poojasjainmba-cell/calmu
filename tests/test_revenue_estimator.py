from __future__ import annotations

import pandas as pd

from revenue_estimator import add_program_revenue_fields


def test_potential_revenue_uses_published_tuition_config() -> None:
    df = pd.DataFrame([{"program": "Bachelor of Science", "program_revenue_input": 0}])

    result = add_program_revenue_fields(df, total_revenue_column="program_revenue_input")

    assert result.loc[0, "potential_program_revenue"] == 70200
    assert result.loc[0, "potential_revenue"] == 70200
    assert result.loc[0, "open_pipeline_potential_revenue"] == 70200
    assert result.loc[0, "enrolled_revenue"] == 0
    assert result.loc[0, "potential_monthly_program_revenue"] == 1462.5
    assert result.loc[0, "potential_twelve_month_revenue"] == 17550


def test_realized_revenue_window_is_capped() -> None:
    df = pd.DataFrame([{"program": "Certificate in AI", "program_revenue_input": 10530}])

    result = add_program_revenue_fields(df, total_revenue_column="program_revenue_input")

    assert result.loc[0, "monthly_program_revenue"] == 1755
    assert result.loc[0, "twelve_month_revenue"] == 10530


def test_enrolled_revenue_uses_estimated_total_tuition() -> None:
    df = pd.DataFrame(
        [
            {
                "program": "MBA",
                "program_revenue_input": 0,
                "enrollment_status": "EA Signed",
            }
        ]
    )

    result = add_program_revenue_fields(df, total_revenue_column="program_revenue_input")

    assert result.loc[0, "degree_level"] == "Master"
    assert result.loc[0, "enrolled_revenue"] == 32799
    assert result.loc[0, "potential_revenue"] == 0
