from __future__ import annotations

from config import get_access_token


def main() -> int:
    token = get_access_token()
    if token:
        print("HUBSPOT_ACCESS_TOKEN is set")
        return 0
    print("HUBSPOT_ACCESS_TOKEN is missing")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
