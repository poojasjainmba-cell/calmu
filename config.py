from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps check_env usable before install
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIT_DIR = DATA_DIR / "audit"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG_DIR = BASE_DIR / "config"

FIELD_INVENTORY_PATH = AUDIT_DIR / "hubspot_field_inventory.csv"
FIELD_MAPPING_PATH = AUDIT_DIR / "dashboard_field_mapping.json"
REMOVED_FIELDS_PATH = AUDIT_DIR / "dashboard_removed_fields.csv"
LAST_REFRESH_PATH = DATA_DIR / "last_refresh_time.txt"
REFRESH_STATUS_PATH = DATA_DIR / "refresh_status.json"
PAID_SOURCE_RULES_PATH = CONFIG_DIR / "paid_source_rules.json"
VENDOR_RULES_PATH = CONFIG_DIR / "vendor_rules.json"
VENDOR_COSTS_PATH = CONFIG_DIR / "vendor_costs.csv"
DEGREE_TUITION_PATH = CONFIG_DIR / "degree_tuition.json"
METRIC_DEFINITIONS_PATH = CONFIG_DIR / "metric_definitions.json"

MISSING_TOKEN_MESSAGE = (
    "HUBSPOT_ACCESS_TOKEN is missing. Create a HubSpot private app token, then "
    "set it in your shell or in a local .env file as HUBSPOT_ACCESS_TOKEN=your_token_here."
)

PLACEHOLDER_TOKENS = {
    "your_token_here",
    "your_hubspot_private_app_token",
    "your_actual_hubspot_private_app_token",
}


def load_environment() -> None:
    load_dotenv(BASE_DIR / ".env")


load_environment()
HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()


def ensure_directories() -> None:
    for path in (AUDIT_DIR, RAW_DIR, PROCESSED_DIR, CONFIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_access_token() -> str:
    load_environment()
    token = os.getenv("HUBSPOT_ACCESS_TOKEN", "").strip()
    if token.lower() in PLACEHOLDER_TOKENS:
        return ""
    return token


def token_is_set() -> bool:
    return bool(get_access_token())


def require_access_token() -> str:
    token = get_access_token()
    if not token:
        raise RuntimeError(MISSING_TOKEN_MESSAGE)
    return token


def load_json_safe(path: Path | str, default: Any | None = None) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        return default
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path | str, payload: Any) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def friendly_missing_token_message() -> str:
    return MISSING_TOKEN_MESSAGE
