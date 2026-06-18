from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

import pandas as pd

from .budget_loader import BudgetData
from .data_loader import UploadedLeadData
from .enrollment_tracker import EnrollmentTrackerData


UDR_COLUMNS = {"contact_owner", "udr", "budget_name", "normalized_budget_name", "source", "row_label"}


def _mapping_from_value(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return {str(key): str(label) for key, label in parsed.items() if str(key).strip() and str(label).strip()}
    if isinstance(value, dict):
        return {str(key): str(label) for key, label in value.items() if str(key).strip() and str(label).strip()}
    try:
        return {str(key): str(label) for key, label in dict(value).items() if str(key).strip() and str(label).strip()}
    except Exception:
        return {}


def load_udr_label_map() -> dict[str, str]:
    mapping = _mapping_from_value(os.getenv("UDR_LABEL_MAP"))
    try:
        import streamlit as st

        mapping.update(_mapping_from_value(st.secrets.get("UDR_LABEL_MAP")))
        mapping.update(_mapping_from_value(st.secrets.get("udr_label_map")))
    except Exception:
        pass
    return mapping


def apply_label_map(df: pd.DataFrame, mapping: dict[str, str], columns: set[str] = UDR_COLUMNS) -> pd.DataFrame:
    if df.empty or not mapping:
        return df
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda value: mapping.get(str(value).strip(), value) if pd.notna(value) else value)
    return out


def apply_udr_label_overrides(
    uploaded: UploadedLeadData,
    tracker: EnrollmentTrackerData,
    budget: BudgetData,
) -> tuple[UploadedLeadData, EnrollmentTrackerData, BudgetData, bool]:
    mapping = load_udr_label_map()
    if not mapping:
        return uploaded, tracker, budget, False

    uploaded = replace(
        uploaded,
        paid_leads=apply_label_map(uploaded.paid_leads, mapping),
        udr_leads=apply_label_map(uploaded.udr_leads, mapping),
        paid_pivots={name: apply_label_map(frame, mapping) for name, frame in uploaded.paid_pivots.items()},
        udr_pivots={name: apply_label_map(frame, mapping) for name, frame in uploaded.udr_pivots.items()},
    )
    tracker = replace(
        tracker,
        enrollments=apply_label_map(tracker.enrollments, mapping),
        roundup_allocations=apply_label_map(tracker.roundup_allocations, mapping),
    )
    budget = replace(
        budget,
        allocations=apply_label_map(budget.allocations, mapping),
        term_allocations=apply_label_map(budget.term_allocations, mapping),
    )
    return uploaded, tracker, budget, True
