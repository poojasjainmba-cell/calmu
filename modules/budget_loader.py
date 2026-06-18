from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .source_mapping import clean_text, normalize_source_value
from .term_utils import parse_budget_term_header


BUDGET_FILE = Path("user_files/01-2026Budget.xlsx")
PUBLIC_BASELINE_DIR = Path("public_data/baseline")


@dataclass
class BudgetData:
    summary: pd.DataFrame
    allocations: pd.DataFrame
    term_allocations: pd.DataFrame
    raw_sheet_names: list[str]
    notes: list[str]


def _summary_rows(raw: pd.DataFrame, sheet: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        metric = clean_text(row.get(1, ""))
        if metric.lower() in {"budget", "enrolled", "starts", "start%"}:
            rows.append(
                {
                    "sheet": sheet,
                    "metric": metric,
                    "value": row.get(2),
                    "numeric_value": pd.to_numeric(row.get(2), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def _allocation_rows(raw: pd.DataFrame, sheet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    header_idx: int | None = None
    total_col: int | None = None
    for idx, row in raw.iterrows():
        values = [clean_text(value).lower() for value in row.tolist()]
        budget_cols = [i for i, value in enumerate(values) if value == "budget"]
        if budget_cols:
            header_idx = idx
            total_col = budget_cols[-1]
            break
    if header_idx is None or total_col is None or total_col == 0:
        return pd.DataFrame(), pd.DataFrame()

    name_col = total_col - 1
    headers = raw.iloc[header_idx].to_dict()
    term_meta = {
        column: parse_budget_term_header(value)
        for column, value in headers.items()
        if column > total_col and clean_text(value)
    }
    term_cols = [column for column, meta in term_meta.items() if meta and meta["term_metric"] in {"actual", "goal"}]
    allocation_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    special_labels = {
        "total": "Total",
        "starts": "Starts",
        "start%": "Start%",
    }
    for idx in range(header_idx + 1, len(raw)):
        label = clean_text(raw.iat[idx, name_col])
        label_lower = label.lower()
        total = pd.to_numeric(raw.iat[idx, total_col], errors="coerce")
        term_values = {column: pd.to_numeric(raw.iat[idx, column], errors="coerce") for column in term_cols}
        numeric_term_values = [float(value) for value in term_values.values() if pd.notna(value)]
        if label_lower in special_labels:
            for column in term_cols:
                term_value = term_values[column]
                if pd.notna(term_value):
                    meta = term_meta[column] or {}
                    metric = {
                        "total": meta.get("term_metric", "actual"),
                        "starts": "starts",
                        "start%": "start_rate",
                    }[label_lower]
                    term_rows.append(
                        {
                            "sheet": sheet,
                            "budget_dimension": "roundup_metric",
                            "budget_name": special_labels[label_lower],
                            "source": special_labels[label_lower],
                            "raw_term": meta.get("raw_term", clean_text(headers[column])),
                            "term": meta.get("term", clean_text(headers[column])),
                            "term_label": meta.get("term_label", clean_text(headers[column])),
                            "term_metric": metric,
                            "term_value": float(term_value),
                            "term_budget": float(term_value),
                            "planned_budget": float(total) if pd.notna(total) else 0.0,
                        }
                    )
            if label_lower == "start%":
                break
            continue
        if not label or (pd.isna(total) and not numeric_term_values):
            continue
        normalized = normalize_source_value(label)
        planned_budget = float(total) if pd.notna(total) else sum(
            float(value)
            for column, value in term_values.items()
            if pd.notna(value) and (term_meta[column] or {}).get("term_metric") == "goal"
        )
        if pd.notna(total):
            allocation_rows.append(
                {
                    "sheet": sheet,
                    "budget_dimension": "UDR allocation",
                    "budget_name": label,
                    "normalized_budget_name": normalized,
                    "source": normalized,
                    "planned_budget": float(total),
                    "budget_metric": "goal",
                }
            )
        for column in term_cols:
            term_value = term_values[column]
            if pd.notna(term_value):
                meta = term_meta[column] or {}
                term_rows.append(
                    {
                        "sheet": sheet,
                        "budget_dimension": "UDR allocation",
                        "budget_name": label,
                        "source": normalized,
                        "raw_term": meta.get("raw_term", clean_text(headers[column])),
                        "term": meta.get("term", clean_text(headers[column])),
                        "term_label": meta.get("term_label", clean_text(headers[column])),
                        "term_metric": meta.get("term_metric", "goal"),
                        "term_value": float(term_value),
                        "term_budget": float(term_value),
                        "planned_budget": planned_budget,
                    }
                )
    return pd.DataFrame(allocation_rows), pd.DataFrame(term_rows)


def load_budget(path: Path = BUDGET_FILE) -> BudgetData:
    if not path.exists():
        summary_path = PUBLIC_BASELINE_DIR / "budget_summary.csv"
        allocations_path = PUBLIC_BASELINE_DIR / "budget_allocations.csv"
        term_path = PUBLIC_BASELINE_DIR / "budget_term_allocations.csv"
        if summary_path.exists() or allocations_path.exists() or term_path.exists():
            summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
            allocations = pd.read_csv(allocations_path) if allocations_path.exists() else pd.DataFrame()
            term_allocations = pd.read_csv(term_path) if term_path.exists() else pd.DataFrame()
            return BudgetData(
                summary,
                allocations,
                term_allocations,
                ["sanitized_public_baseline"],
                [
                    f"Private budget workbook not found; using sanitized public baseline from {PUBLIC_BASELINE_DIR}.",
                    "Sanitized baseline excludes names, emails, phone numbers, raw Record IDs, and notes.",
                ],
            )
        return BudgetData(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [f"Missing budget workbook: {path}"])

    xl = pd.ExcelFile(path)
    summary_frames: list[pd.DataFrame] = []
    allocation_frames: list[pd.DataFrame] = []
    term_frames: list[pd.DataFrame] = []
    for sheet in xl.sheet_names:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        summary_frames.append(_summary_rows(raw, sheet))
        allocations, term_allocations = _allocation_rows(raw, sheet)
        allocation_frames.append(allocations)
        term_frames.append(term_allocations)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    allocations = pd.concat(allocation_frames, ignore_index=True) if allocation_frames else pd.DataFrame()
    term_allocations = pd.concat(term_frames, ignore_index=True) if term_frames else pd.DataFrame()
    notes = [
        f"Budget workbook sheets inspected: {', '.join(xl.sheet_names)}",
        "Budget workbook is treated as planned enrollment/UDR allocation because no media spend ledger fields were present.",
    ]
    return BudgetData(summary, allocations, term_allocations, xl.sheet_names, notes)
