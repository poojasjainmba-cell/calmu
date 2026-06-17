from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

from .data_loader import standardize_leads


BASE_URL = "https://api.hubapi.com"

DEFAULT_CONTACT_PROPERTIES = [
    "email",
    "firstname",
    "lastname",
    "phone",
    "hs_object_id",
    "lifecyclestage",
    "hs_lead_status",
    "hubspot_owner_id",
    "createdate",
    "lastmodifieddate",
    "notes_last_updated",
    "hs_last_sales_activity_timestamp",
    "hs_latest_source",
    "hs_latest_source_data_1",
    "hs_latest_source_data_2",
    "hs_analytics_source",
    "hs_analytics_source_data_1",
    "hs_analytics_source_data_2",
    "utm_source",
    "utm_campaign",
    "utm_medium",
    "campaign",
    "degree",
    "program",
    "student_type",
    "event_attended",
    "campus_location",
    "paid_lead_list",
    "organic_lead_list",
]


@dataclass
class HubSpotFetchResult:
    contacts: pd.DataFrame
    properties: pd.DataFrame
    owners: pd.DataFrame
    used_properties: list[str]
    missing_properties: list[str]
    fetched_at: datetime | None
    error: str | None
    token_present: bool


class HubSpotReadOnlyClient:
    def __init__(self, token: str, timeout: int = 30, max_retries: int = 3) -> None:
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{BASE_URL}{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"HubSpot request failed: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code in {401, 403}:
                raise RuntimeError("HubSpot denied access. Confirm the private app token and read-only scopes.")
            if response.status_code == 429:
                wait_seconds = int(response.headers.get("Retry-After", "0") or 0) or 2**attempt
                if attempt >= self.max_retries:
                    raise RuntimeError("HubSpot rate limit persisted after retries.")
                time.sleep(max(wait_seconds, 1))
                continue
            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"HubSpot temporary error {response.status_code}: {response.text[:300]}")
                time.sleep(2**attempt)
                continue
            if not response.ok:
                raise RuntimeError(f"HubSpot API error {response.status_code}: {response.text[:500]}")
            return response.json() if response.text else {}
        raise RuntimeError(f"HubSpot request failed: {last_error}")

    def properties(self, object_type: str = "contacts") -> pd.DataFrame:
        payload = self.request("GET", f"/crm/v3/properties/{object_type}")
        return pd.DataFrame(payload.get("results", []))

    def owners(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100, "archived": "false"}
            if after:
                params["after"] = after
            payload = self.request("GET", "/crm/v3/owners/", params=params)
            for owner in payload.get("results", []):
                first = owner.get("firstName") or ""
                last = owner.get("lastName") or ""
                rows.append(
                    {
                        "owner_id": str(owner.get("id") or ""),
                        "contact_owner": " ".join([first, last]).strip() or owner.get("email") or str(owner.get("id")),
                    }
                )
            after = ((payload.get("paging") or {}).get("next") or {}).get("after")
            if not after:
                break
        return pd.DataFrame(rows)

    def contacts(self, properties: Iterable[str], max_records: int = 10000) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        after: str | None = None
        props = list(dict.fromkeys(properties))
        while True:
            params: dict[str, Any] = {"limit": 100, "archived": "false"}
            if props:
                params["properties"] = props
            if after:
                params["after"] = after
            payload = self.request("GET", "/crm/v3/objects/contacts", params=params)
            for item in payload.get("results", []):
                row = {
                    "id": item.get("id"),
                    "createdAt": item.get("createdAt"),
                    "updatedAt": item.get("updatedAt"),
                }
                row.update(item.get("properties") or {})
                row.setdefault("hs_object_id", item.get("id"))
                rows.append(row)
                if max_records and len(rows) >= max_records:
                    return pd.DataFrame(rows)
            after = ((payload.get("paging") or {}).get("next") or {}).get("after")
            if not after:
                break
        return pd.DataFrame(rows)


def get_access_token() -> str | None:
    load_dotenv()
    token = os.getenv("HUBSPOT_ACCESS_TOKEN")
    try:
        import streamlit as st

        token = st.secrets.get("HUBSPOT_ACCESS_TOKEN", token)
    except Exception:
        pass
    token = str(token or "").strip()
    return token or None


def _secret_or_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    try:
        import streamlit as st

        value = st.secrets.get(name, value)
    except Exception:
        pass
    return str(value or "").strip()


def _configured_property_candidates() -> list[str]:
    extra = _secret_or_env("HUBSPOT_CONTACT_PROPERTIES")
    extra_props = [part.strip() for part in str(extra).split(",") if part.strip()]
    return list(dict.fromkeys([*DEFAULT_CONTACT_PROPERTIES, *extra_props]))


def _select_available_properties(schema: pd.DataFrame) -> tuple[list[str], list[str]]:
    available = set(schema.get("name", pd.Series(dtype=str)).dropna().astype(str))
    candidates = _configured_property_candidates()
    used = [prop for prop in candidates if prop in available]
    missing = [prop for prop in candidates if prop not in available]
    return used, missing


def fetch_hubspot_contacts(max_records: int | None = None) -> HubSpotFetchResult:
    token = get_access_token()
    if not token:
        return HubSpotFetchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], None, None, False)

    try:
        max_records = max_records if max_records is not None else int(_secret_or_env("HUBSPOT_MAX_CONTACTS", "10000") or 10000)
    except ValueError:
        max_records = 10000

    try:
        client = HubSpotReadOnlyClient(token)
        properties = client.properties("contacts")
        used, missing = _select_available_properties(properties)
        raw_contacts = client.contacts(used, max_records=max_records or 0)
        owners = pd.DataFrame()
        try:
            owners = client.owners()
        except Exception:
            owners = pd.DataFrame()
        if not owners.empty and "hubspot_owner_id" in raw_contacts.columns:
            raw_contacts["hubspot_owner_id"] = raw_contacts["hubspot_owner_id"].fillna("").astype(str)
            raw_contacts = raw_contacts.merge(owners, left_on="hubspot_owner_id", right_on="owner_id", how="left")
        contacts = standardize_leads(raw_contacts, "HubSpot live")
        if "contact_owner_y" in contacts.columns:
            contacts["contact_owner"] = contacts["contact_owner_y"].fillna(contacts.get("contact_owner_x", ""))
        return HubSpotFetchResult(
            contacts=contacts,
            properties=properties,
            owners=owners,
            used_properties=used,
            missing_properties=missing,
            fetched_at=datetime.now(timezone.utc),
            error=None,
            token_present=True,
        )
    except Exception as exc:
        return HubSpotFetchResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], None, str(exc), True)
