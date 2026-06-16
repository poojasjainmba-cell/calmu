from __future__ import annotations

import pandas as pd

from degree_revenue import add_program_revenue_fields, get_program_duration


def test_default_degree_durations() -> None:
    assert get_program_duration("Certificate in Cybersecurity")["program_duration_months"] == 6
    assert get_program_duration("Associate of Science")["program_duration_months"] == 24
    assert get_program_duration("Bachelor of Science")["program_duration_months"] == 48
    assert get_program_duration("Master of Business Administration")["program_duration_months"] == 24


def test_monthly_revenue_and_twelve_month_cap() -> None:
    df = pd.DataFrame(
        [
            {
                "program": "Certificate in Artificial Intelligence",
                "revenue_attributed": 12000,
            },
            {
                "program": "Bachelor of Science in Business Administration",
                "revenue_attributed": 48000,
            },
        ]
    )

    result = add_program_revenue_fields(df)

    assert result.loc[0, "program_duration_months"] == 6
    assert result.loc[0, "monthly_program_revenue"] == 2000
    assert result.loc[0, "twelve_month_revenue"] == 12000
    assert result.loc[1, "program_duration_months"] == 48
    assert result.loc[1, "monthly_program_revenue"] == 1000
    assert result.loc[1, "twelve_month_revenue"] == 12000


def test_doctoral_duration_keeps_policy_note() -> None:
    duration = get_program_duration("Doctor of Business Administration")

    assert duration["program_duration_months"] == 48
    assert "CalMU policy" in duration["duration_note"]
