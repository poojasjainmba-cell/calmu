from __future__ import annotations

import pandas as pd

from refresh_tuition import parse_tuition_page
from tuition_loader import get_degree_tuition


def test_default_tuition_values() -> None:
    certificate = get_degree_tuition("Certificate in Cybersecurity")
    bachelor = get_degree_tuition("Bachelor of Science in Business Administration")
    master = get_degree_tuition("MBA")
    doctoral = get_degree_tuition("Doctor of Business Administration")

    assert certificate["estimated_total_tuition"] == 10530
    assert bachelor["credits_required"] == 120
    assert master["tuition_per_credit"] == 841
    assert doctoral["estimated_annual_tuition"] == 10608


def test_degree_level_inference_aliases() -> None:
    assert get_degree_tuition("BSBA")["degree_level"] == "Bachelor"
    assert get_degree_tuition("BSBT")["degree_level"] == "Bachelor"
    assert get_degree_tuition("MSCIS")["degree_level"] == "Master"
    assert get_degree_tuition("MSAI")["degree_level"] == "Master"
    assert get_degree_tuition("DBA")["degree_level"] == "Doctoral"


def test_explicit_degree_level_has_high_confidence() -> None:
    result = get_degree_tuition("Business Administration", explicit_degree_level="Bachelor")

    assert result["degree_level"] == "Bachelor"
    assert result["revenue_confidence"] == "High"


def test_parse_tuition_page_values() -> None:
    html = """
    Certificate Programs $585 18 $10,530 $10,530
    Associate Degree $585 60 $35,100 $14,040
    Bachelor's Degree $585 120 $70,200 $14,040
    Master's Degree Programs $841 39 $32,799 $15,138
    Doctoral Degree Programs $884 61 $53,924 $10,608
    """

    parsed = parse_tuition_page(html)

    assert parsed["Certificate"]["estimated_total_tuition"] == 10530
    assert parsed["Associate"]["credits_required"] == 60
    assert parsed["Bachelor"]["estimated_annual_tuition"] == 14040
    assert parsed["Master"]["tuition_per_credit"] == 841
    assert parsed["Doctoral"]["credits_required"] == 61


def test_unmatched_program_returns_missing_values() -> None:
    result = get_degree_tuition(pd.NA)

    assert result["program_degree_level"] is None
    assert result["tuition_estimate_source"] == "unmatched_program"
    assert result["revenue_confidence"] == "Low"
