from __future__ import annotations

import pandas as pd

from modules.label_overrides import apply_label_map, load_udr_label_map


def test_udr_label_map_can_load_from_json_env(monkeypatch) -> None:
    monkeypatch.setenv("UDR_LABEL_MAP", '{"UDR 01": "Owner One"}')

    assert load_udr_label_map()["UDR 01"] == "Owner One"


def test_apply_label_map_only_replaces_matching_udr_labels() -> None:
    frame = pd.DataFrame(
        [
            {"udr": "UDR 01", "normalized_source": "Oby"},
            {"udr": "UDR 02", "normalized_source": "Referral"},
        ]
    )

    mapped = apply_label_map(frame, {"UDR 01": "Owner One"})

    assert mapped["udr"].tolist() == ["Owner One", "UDR 02"]
    assert mapped["normalized_source"].tolist() == ["Oby", "Referral"]
