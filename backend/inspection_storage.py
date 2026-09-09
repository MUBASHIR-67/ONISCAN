"""Small JSON-backed inspection history for the ONI-SCAN prototype."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "backend" / "data"
INSPECTIONS_PATH = DATA_DIR / "inspections.json"


class InspectionStorageError(RuntimeError):
    """Raised when inspection history cannot be read or written safely."""


def load_inspections() -> list[dict[str, Any]]:
    """Load persisted records, treating a missing file as an empty history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not INSPECTIONS_PATH.exists():
        return []
    try:
        payload = json.loads(INSPECTIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InspectionStorageError("Inspection history storage is missing or corrupted.") from exc
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise InspectionStorageError("Inspection history storage has an invalid structure.")
    return payload


def _write_inspections(records: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=DATA_DIR, delete=False, suffix=".tmp"
        ) as temporary_file:
            json.dump(records, temporary_file, indent=2, ensure_ascii=True)
            temporary_file.write("\n")
            temporary_path = temporary_file.name
        os.replace(temporary_path, INSPECTIONS_PATH)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError:
                pass
        raise InspectionStorageError("Inspection history could not be saved.") from exc


def append_inspection(record: dict[str, Any]) -> dict[str, Any]:
    records = load_inspections()
    records.insert(0, record)
    _write_inspections(records)
    return record


def find_inspection(inspection_id: str) -> dict[str, Any] | None:
    return next((record for record in load_inspections() if record.get("inspection_id") == inspection_id), None)


def update_inspection(inspection_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    records = load_inspections()
    for record in records:
        if record.get("inspection_id") == inspection_id:
            record.update(updates)
            _write_inspections(records)
            return record
    return None
