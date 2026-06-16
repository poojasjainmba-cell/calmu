from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from config import FIELD_MAPPING_PATH, load_json_safe


def load_dashboard_mapping(path: Path | str = FIELD_MAPPING_PATH) -> dict[str, Any]:
    return load_json_safe(path, default={}) or {}


def _mapping_section(mapping: dict[str, Any], object_type: str) -> dict[str, Any]:
    return mapping.get(object_type) or mapping.get(f"{object_type}s") or {}


def get_mapped_field(
    logical_name: str,
    object_type: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> str | None:
    mapping = mapping or load_dashboard_mapping()
    if object_type:
        value = _mapping_section(mapping, object_type).get(logical_name)
        return value or None

    for section in ("contact", "deal"):
        value = _mapping_section(mapping, section).get(logical_name)
        if value:
            return value
    return None


def field_available(
    logical_name: str,
    object_type: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> bool:
    return bool(get_mapped_field(logical_name, object_type=object_type, mapping=mapping))


def required_fields_available(
    logical_names: Iterable[str],
    object_type: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> bool:
    return all(field_available(name, object_type=object_type, mapping=mapping) for name in logical_names)


def missing_fields(
    logical_names: Iterable[str],
    object_type: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> list[str]:
    return [
        name
        for name in logical_names
        if not field_available(name, object_type=object_type, mapping=mapping)
    ]


def warn_if_missing(
    logical_names: Iterable[str],
    object_type: str | None = None,
    mapping: dict[str, Any] | None = None,
) -> list[str]:
    return missing_fields(logical_names, object_type=object_type, mapping=mapping)


def mapped_properties_for_object(object_type: str, mapping: dict[str, Any] | None = None) -> list[str]:
    mapping = mapping or load_dashboard_mapping()
    values = [
        value
        for value in _mapping_section(mapping, object_type).values()
        if isinstance(value, str) and value
    ]
    return sorted(set(values))
