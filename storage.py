from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config import LAST_REFRESH_PATH, PROCESSED_DIR, RAW_DIR, REFRESH_STATUS_PATH, load_json_safe


PROCESSED_TABLES = (
    "contacts_clean",
    "deals_clean",
    "lead_deal_fact",
    "student_journey_fact",
    "cohort_fact",
    "vendor_fact",
    "salesman_revenue_fact",
    "activity_events",
)
RAW_TABLES = ("contacts", "deals", "owners", "associations", "activities")


def _base_path(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def save_table(df: pd.DataFrame, directory: Path, name: str) -> Path:
    base = _base_path(directory, name)
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
        return parquet_path
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def load_table(directory: Path, name: str) -> pd.DataFrame:
    parquet_path = (directory / name).with_suffix(".parquet")
    csv_path = (directory / name).with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No cached table found for {name}")


def save_raw_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {name: str(save_table(df, RAW_DIR, name)) for name, df in tables.items()}


def save_processed_tables(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    return {name: str(save_table(df, PROCESSED_DIR, name)) for name, df in tables.items()}


def load_processed_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in PROCESSED_TABLES:
        try:
            tables[name] = load_table(PROCESSED_DIR, name)
        except FileNotFoundError:
            tables[name] = pd.DataFrame()
    return tables


def read_last_refresh_time() -> datetime | None:
    if not LAST_REFRESH_PATH.exists():
        return None
    text = LAST_REFRESH_PATH.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def cache_age_hours() -> float | None:
    refreshed_at = read_last_refresh_time()
    if not refreshed_at:
        return None
    if refreshed_at.tzinfo is None:
        refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - refreshed_at).total_seconds() / 3600


def cache_is_stale(max_age_hours: int = 24) -> bool:
    age = cache_age_hours()
    return age is None or age > max_age_hours


def load_refresh_status() -> dict[str, Any]:
    return load_json_safe(REFRESH_STATUS_PATH, default={}) or {}


def load_cached_data() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    tables = load_processed_tables()
    status = load_refresh_status()
    missing = [name for name, df in tables.items() if df.empty]
    metadata = {
        "last_refresh_time": read_last_refresh_time(),
        "cache_age_hours": cache_age_hours(),
        "is_stale": cache_is_stale(),
        "status": status,
        "missing_tables": missing,
        "has_cache": any(not df.empty for df in tables.values()),
    }
    return tables, metadata
