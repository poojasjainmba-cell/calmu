from __future__ import annotations

from definitions import definitions_frame, metric_definition, section_help_text


def test_requested_plain_english_definitions_are_available() -> None:
    terms = set(definitions_frame()["Term"])

    for term in [
        "Hot Lead",
        "Warm Lead",
        "Cold Lead",
        "Dead Lead",
        "Reviveable Dead Lead",
        "Paid Lead Leakage",
        "Potential Revenue",
        "Estimated Enrolled Revenue",
        "Actual Revenue",
        "Sales Efficiency",
        "Action Load Score",
        "Vendor",
        "Cohort",
        "Total Program Revenue",
        "Annualized Revenue",
        "Monthly Program Revenue",
        "6-Month Revenue",
        "12-Month Revenue",
        "24-Month Revenue",
        "Revenue Timing",
    ]:
        assert term in terms
        assert metric_definition(term)


def test_metric_definition_matches_display_labels_and_snake_case() -> None:
    assert "open leads or opportunities" in metric_definition("Potential revenue")
    assert "open leads or opportunities" in metric_definition("potential_revenue")
    assert "urgent follow-up work" in metric_definition("action_load")
    assert "short and long programs fairly" in metric_definition("annualized_program_revenue")
    assert "degree or certificate" in metric_definition("total_program_revenue")
    assert "next 12 months" in metric_definition("twelve_month_revenue")
    assert "4-year bachelor's program" in metric_definition("Revenue Timing")


def test_section_help_uses_what_why_action_format() -> None:
    help_text = section_help_text("Pipeline Health")

    assert "What:" in help_text
    assert "Why:" in help_text
    assert "Action:" in help_text
