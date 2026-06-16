from __future__ import annotations

import pandas as pd

from source_classification import add_source_classification, classify_source_group, is_paid_source


def test_paid_lead_classification_works() -> None:
    assert is_paid_source("Google Ads CPC campaign") is True
    assert classify_source_group("facebook paid social") == "Paid Social"
    assert classify_source_group("organic search") == "Organic Search"

    df = pd.DataFrame({"utm_source": ["google ads", "newsletter"], "utm_medium": ["cpc", "email"]})
    classified = add_source_classification(df, ["utm_source", "utm_medium"])

    assert bool(classified.loc[0, "paid_lead_flag"]) is True
    assert classified.loc[0, "source_group"] == "Paid Search"
    assert classified.loc[0, "vendor"] == "Google"
    assert classified.loc[0, "vendor_confidence"] == "High"
    assert classified.loc[0, "vendor_source_field"] == "utm_source"
    assert bool(classified.loc[1, "paid_lead_flag"]) is False
    assert classified.loc[1, "source_group"] == "Email"


def test_vendor_normalization_uses_custom_vendor_fields() -> None:
    df = pd.DataFrame(
        {
            "agency_partner": ["atra analytics", "Klient Boost"],
            "campaigns": ["spring", "fall"],
        }
    )

    classified = add_source_classification(df)

    assert classified["vendor"].tolist() == ["Atra", "KlientBoost"]
    assert classified["vendor_source_field"].tolist() == ["agency_partner", "agency_partner"]
