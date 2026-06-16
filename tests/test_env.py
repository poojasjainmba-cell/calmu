from __future__ import annotations

import check_env
from config import get_access_token


def test_environment_token_check_does_not_print_token(monkeypatch, capsys) -> None:
    secret = "pat-na1-do-not-print-this"
    monkeypatch.setenv("HUBSPOT_ACCESS_TOKEN", secret)

    result = check_env.main()
    captured = capsys.readouterr()

    assert result == 0
    assert "HUBSPOT_ACCESS_TOKEN is set" in captured.out
    assert secret not in captured.out


def test_placeholder_token_is_treated_as_missing(monkeypatch, capsys) -> None:
    monkeypatch.setenv("HUBSPOT_ACCESS_TOKEN", "your_hubspot_private_app_token")

    result = check_env.main()
    captured = capsys.readouterr()

    assert result == 1
    assert get_access_token() == ""
    assert "HUBSPOT_ACCESS_TOKEN is missing" in captured.out
