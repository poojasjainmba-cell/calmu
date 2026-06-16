from __future__ import annotations

import time
from typing import Any, Iterable

import pandas as pd
import requests

from config import require_access_token


class HubSpotAPIError(RuntimeError):
    """Raised when HubSpot returns an unrecoverable API error."""


class HubSpotPermissionError(HubSpotAPIError):
    """Raised when the token is missing scopes or is unauthorized."""


class HubSpotClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.hubapi.com",
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.token = token or require_access_token()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise HubSpotAPIError(f"HubSpot request failed: {exc}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code in (401, 403):
                raise HubSpotPermissionError(
                    "HubSpot denied access. Check that HUBSPOT_ACCESS_TOKEN is valid "
                    "and has CRM object, owner, property, and association read scopes."
                )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
                if attempt >= self.max_retries:
                    raise HubSpotAPIError("HubSpot rate limit persisted after retries.")
                time.sleep(max(wait_seconds, 1))
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self.max_retries:
                    raise HubSpotAPIError(
                        f"HubSpot temporary error {response.status_code}: {response.text[:300]}"
                    )
                time.sleep(2**attempt)
                continue

            if not response.ok:
                raise HubSpotAPIError(
                    f"HubSpot API error {response.status_code}: {response.text[:500]}"
                )

            if not response.text:
                return {}
            return response.json()

        raise HubSpotAPIError(f"HubSpot request failed: {last_error}")

    @staticmethod
    def _dedupe_properties(properties: Iterable[str] | None) -> list[str]:
        if not properties:
            return []
        seen: set[str] = set()
        output: list[str] = []
        for prop in properties:
            if prop and prop not in seen:
                output.append(prop)
                seen.add(prop)
        return output

    def get_properties(self, object_type: str) -> pd.DataFrame:
        payload = self._request("GET", f"/crm/v3/properties/{object_type}")
        rows = payload.get("results", [])
        return pd.DataFrame(rows)

    def fetch_objects(
        self,
        object_type: str,
        properties: Iterable[str] | None = None,
        limit: int = 100,
        max_records: int | None = None,
        archived: bool = False,
        progress_label: str | None = None,
    ) -> pd.DataFrame:
        props = self._dedupe_properties(properties)
        params: dict[str, Any] = {"limit": min(limit, 100), "archived": str(archived).lower()}
        if props:
            params["properties"] = props

        rows: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            if after:
                params["after"] = after
            payload = self._request("GET", f"/crm/v3/objects/{object_type}", params=params)
            for item in payload.get("results", []):
                row = {
                    "id": item.get("id"),
                    "createdAt": item.get("createdAt"),
                    "updatedAt": item.get("updatedAt"),
                    "archived": item.get("archived", False),
                }
                row.update(item.get("properties") or {})
                row.setdefault("hs_object_id", item.get("id"))
                rows.append(row)
                if progress_label and len(rows) % 1000 == 0:
                    print(f"Fetched {len(rows):,} {progress_label}...", flush=True)
                if max_records and len(rows) >= max_records:
                    return pd.DataFrame(rows)

            paging = payload.get("paging", {})
            after = (paging.get("next") or {}).get("after")
            if not after:
                break

        return pd.DataFrame(rows)

    def search_objects(
        self,
        object_type: str,
        properties: Iterable[str] | None = None,
        limit: int = 200,
        max_records: int | None = None,
        progress_label: str | None = None,
        sorts: list[dict[str, str]] | None = None,
    ) -> pd.DataFrame:
        props = self._dedupe_properties(properties)
        rows: list[dict[str, Any]] = []
        after: str | None = None

        while True:
            body: dict[str, Any] = {"limit": min(limit, 200)}
            if props:
                body["properties"] = props
            if sorts:
                body["sorts"] = sorts
            if after:
                body["after"] = after

            payload = self._request("POST", f"/crm/v3/objects/{object_type}/search", json=body)
            for item in payload.get("results", []):
                row = {
                    "id": item.get("id"),
                    "createdAt": item.get("createdAt"),
                    "updatedAt": item.get("updatedAt"),
                    "archived": item.get("archived", False),
                }
                row.update(item.get("properties") or {})
                row.setdefault("hs_object_id", item.get("id"))
                rows.append(row)
                if progress_label and len(rows) % 1000 == 0:
                    print(f"Fetched {len(rows):,} {progress_label}...", flush=True)
                if max_records and len(rows) >= max_records:
                    return pd.DataFrame(rows)

            paging = payload.get("paging", {})
            after = (paging.get("next") or {}).get("after")
            if not after:
                break

        return pd.DataFrame(rows)

    def fetch_contacts(
        self, properties: Iterable[str] | None = None, max_records: int | None = None
    ) -> pd.DataFrame:
        return self.fetch_objects("contacts", properties=properties, max_records=max_records)

    def fetch_deals(
        self, properties: Iterable[str] | None = None, max_records: int | None = None
    ) -> pd.DataFrame:
        return self.fetch_objects("deals", properties=properties, max_records=max_records)

    def fetch_activity_objects(
        self,
        activity_type: str,
        properties: Iterable[str] | None = None,
        max_records: int | None = None,
    ) -> pd.DataFrame:
        return self.fetch_objects(
            activity_type,
            properties=properties,
            max_records=max_records,
            progress_label=activity_type,
        )

    def fetch_owners(self, limit: int = 100, progress_label: str | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        params: dict[str, Any] = {"limit": min(limit, 100), "archived": "false"}
        after: str | None = None
        while True:
            if after:
                params["after"] = after
            payload = self._request("GET", "/crm/v3/owners/", params=params)
            for owner in payload.get("results", []):
                first = owner.get("firstName") or ""
                last = owner.get("lastName") or ""
                full_name = " ".join([first, last]).strip()
                rows.append(
                    {
                        "owner_id": str(owner.get("id") or ""),
                        "user_id": owner.get("userId"),
                        "email": owner.get("email"),
                        "first_name": owner.get("firstName"),
                        "last_name": owner.get("lastName"),
                        "salesman_name": full_name or owner.get("email") or owner.get("id"),
                        "archived": owner.get("archived", False),
                    }
                )
                if progress_label and len(rows) % 1000 == 0:
                    print(f"Fetched {len(rows):,} {progress_label}...", flush=True)
            paging = payload.get("paging", {})
            after = (paging.get("next") or {}).get("after")
            if not after:
                break
        return pd.DataFrame(rows)

    def fetch_contact_deal_associations(
        self,
        contact_ids: Iterable[Any],
        progress_label: str | None = None,
    ) -> pd.DataFrame:
        ids = [str(contact_id) for contact_id in contact_ids if pd.notna(contact_id) and str(contact_id)]
        if not ids:
            return pd.DataFrame(columns=["contact_id", "deal_id", "association_type"])

        rows: list[dict[str, Any]] = []
        for start in range(0, len(ids), 100):
            chunk = ids[start : start + 100]
            payload = self._request(
                "POST",
                "/crm/v4/associations/contacts/deals/batch/read",
                json={"inputs": [{"id": contact_id} for contact_id in chunk]},
            )
            for result in payload.get("results", []):
                from_id = str((result.get("from") or {}).get("id") or result.get("fromObjectId") or "")
                for target in result.get("to", []) or []:
                    deal_id = target.get("toObjectId") or target.get("id")
                    assoc_types = target.get("associationTypes") or []
                    assoc_type = (
                        assoc_types[0].get("label")
                        or assoc_types[0].get("typeId")
                        if assoc_types
                        else None
                    )
                    rows.append(
                        {
                            "contact_id": from_id,
                            "deal_id": str(deal_id or ""),
                            "association_type": assoc_type,
                        }
                    )
            if progress_label and (start + len(chunk)) % 1000 == 0:
                print(
                    f"Checked associations for {start + len(chunk):,}/{len(ids):,} {progress_label}; "
                    f"found {len(rows):,} links...",
                    flush=True,
                )
        return pd.DataFrame(rows).drop_duplicates()

    def fetch_activity_contact_associations(
        self,
        activity_type: str,
        activity_ids: Iterable[Any],
        progress_label: str | None = None,
    ) -> pd.DataFrame:
        ids = [str(activity_id) for activity_id in activity_ids if pd.notna(activity_id) and str(activity_id)]
        if not ids:
            return pd.DataFrame(columns=["activity_type", "activity_id", "contact_id", "association_type"])

        rows: list[dict[str, Any]] = []
        for start in range(0, len(ids), 100):
            chunk = ids[start : start + 100]
            payload = self._request(
                "POST",
                f"/crm/v4/associations/{activity_type}/contacts/batch/read",
                json={"inputs": [{"id": activity_id} for activity_id in chunk]},
            )
            for result in payload.get("results", []):
                from_id = str((result.get("from") or {}).get("id") or result.get("fromObjectId") or "")
                for target in result.get("to", []) or []:
                    contact_id = target.get("toObjectId") or target.get("id")
                    assoc_types = target.get("associationTypes") or []
                    assoc_type = (
                        assoc_types[0].get("label")
                        or assoc_types[0].get("typeId")
                        if assoc_types
                        else None
                    )
                    rows.append(
                        {
                            "activity_type": activity_type,
                            "activity_id": from_id,
                            "contact_id": str(contact_id or ""),
                            "association_type": assoc_type,
                        }
                    )
            if progress_label and (start + len(chunk)) % 1000 == 0:
                print(
                    f"Checked associations for {start + len(chunk):,}/{len(ids):,} {progress_label}; "
                    f"found {len(rows):,} links...",
                    flush=True,
                )
        return pd.DataFrame(rows).drop_duplicates()
