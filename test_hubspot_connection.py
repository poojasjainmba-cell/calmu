from __future__ import annotations

from config import friendly_missing_token_message, token_is_set
from hubspot_client import HubSpotAPIError, HubSpotClient


def main() -> int:
    if not token_is_set():
        print(friendly_missing_token_message())
        return 1

    checks: list[tuple[str, bool, str]] = []
    try:
        client = HubSpotClient()
        client.fetch_contacts(["hs_object_id"], max_records=1)
        checks.append(("contacts access", True, "ok"))
    except HubSpotAPIError as exc:
        checks.append(("contacts access", False, str(exc)))

    try:
        client.fetch_deals(["hs_object_id"], max_records=1)
        checks.append(("deals access", True, "ok"))
    except HubSpotAPIError as exc:
        checks.append(("deals access", False, str(exc)))

    try:
        client.fetch_owners()
        checks.append(("owners access", True, "ok"))
    except HubSpotAPIError as exc:
        checks.append(("owners access", False, str(exc)))

    try:
        client.get_properties("contacts")
        checks.append(("contact properties access", True, "ok"))
    except HubSpotAPIError as exc:
        checks.append(("contact properties access", False, str(exc)))

    try:
        client.get_properties("deals")
        checks.append(("deal properties access", True, "ok"))
    except HubSpotAPIError as exc:
        checks.append(("deal properties access", False, str(exc)))

    print("HubSpot connection check")
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"- {status}: {name} ({detail})")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
