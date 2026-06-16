from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from lead_scoring import score_leads


def test_hot_warm_cold_dead_scoring_works() -> None:
    today = datetime(2026, 6, 10, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {
                "contact_created_at": "2026-06-08T00:00:00Z",
                "last_activity_date": "2026-06-09T00:00:00Z",
                "next_activity_date": "2026-06-11T00:00:00Z",
                "paid_lead_flag": True,
                "lifecycle_stage": "MQL",
                "lead_status": "Qualified",
                "has_open_deal": True,
                "email": "student@example.com",
                "phone": "",
            },
            {
                "contact_created_at": "2026-03-01T00:00:00Z",
                "last_activity_date": pd.NaT,
                "next_activity_date": pd.NaT,
                "paid_lead_flag": False,
                "lifecycle_stage": "",
                "lead_status": "",
                "has_open_deal": False,
                "email": "",
                "phone": "",
            },
        ]
    )

    scored = score_leads(df, today=today)

    assert scored.loc[0, "lead_temperature"] == "Hot"
    assert scored.loc[1, "lead_temperature"] == "Dead"
    assert scored.loc[1, "dead_reason"] in {"inactive_60_days_no_open_deal", "never_contacted_30_days"}


def test_scoring_handles_missing_contact_values_without_false_points() -> None:
    today = datetime(2026, 6, 10, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [
            {
                "contact_created_at": "2026-06-01T00:00:00Z",
                "last_activity_date": pd.NaT,
                "next_activity_date": pd.NaT,
                "paid_lead_flag": False,
                "lifecycle_stage": "",
                "lead_status": "",
                "has_open_deal": False,
                "email": pd.NA,
                "phone": pd.NA,
                "program": pd.NA,
            }
        ]
    )

    scored = score_leads(df, today=today)

    assert scored.loc[0, "lead_score"] == 0
    assert bool(scored.loc[0, "reviveable_flag"]) is False


def test_parse_datetime_series_preserves_existing_datetimes() -> None:
    raw = pd.Series(pd.to_datetime(["2024-04-14T13:45:39.372Z"], utc=True))

    parsed = score_leads(
        pd.DataFrame(
            {
                "contact_created_at": raw,
                "last_activity_date": raw,
                "next_activity_date": pd.NaT,
                "paid_lead_flag": False,
                "has_open_deal": False,
                "email": "",
                "phone": "",
                "lifecycle_stage": "",
                "lead_status": "",
            }
        ),
        today=datetime(2024, 4, 15, tzinfo=timezone.utc),
    )

    assert parsed.loc[0, "contact_created_at"].year == 2024
