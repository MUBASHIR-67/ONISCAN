"""Traceable, verification-first grading for ONI-SCAN onion inspections.

This module deliberately does not equate YOLO classes to e-NAM procurement
parameters.  A qualified inspector supplies verified procurement observations
before the configured standard can yield a final range.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


STANDARD_PATH = Path(__file__).with_name("grading_standard.json")


def _load_standard() -> dict[str, Any]:
    with STANDARD_PATH.open(encoding="utf-8") as standard_file:
        standard = json.load(standard_file)
    if standard.get("range_names") != ["Range-I", "Range-II", "Range-III"]:
        raise ValueError("The grading standard must preserve Range-I, Range-II, and Range-III terminology.")
    return standard


AI_TO_PROCUREMENT_MAPPING: dict[str, dict[str, str]] = {
    "acceptable": {"procurement_observation": "unmapped", "status": "Not assessed by AI as an official e-NAM parameter"},
    "damaged": {"procurement_observation": "external_damage_candidate", "status": "Requires human classification; not an official defect category"},
    "bad-onion": {"procurement_observation": "external_defect_signal", "status": "Requires human classification; not an official defect category"},
    "visible_rot": {"procurement_observation": "external_visible_deterioration_signal", "status": "Requires human verification; RGB cannot establish internal rot"},
    "sprouted": {"procurement_observation": "sprouting_candidate", "status": "Requires human verification before use as sprouted percentage"},
}


def get_standard() -> dict[str, Any]:
    """Return a copy so callers cannot mutate the configured source standard."""
    return deepcopy(_load_standard())


def map_ai_observations(ai_observations: dict[str, Any] | None) -> dict[str, Any]:
    """Keep model observations traceable without promoting them to official findings."""
    observations = ai_observations or {}
    class_counts = observations.get("class_counts", {})
    mapped = []
    for class_name, count in class_counts.items():
        mapping = AI_TO_PROCUREMENT_MAPPING.get(
            str(class_name), {"procurement_observation": "unmapped", "status": "No official mapping configured"}
        )
        mapped.append({"ai_class": str(class_name), "count": count, **mapping})
    return {"raw": observations, "mapping": mapped}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_rule(value: float, bounds: dict[str, float]) -> tuple[bool, str]:
    if "max" in bounds and value > bounds["max"]:
        return False, f"value {value} exceeds configured maximum {bounds['max']}"
    if "max_exclusive" in bounds and value >= bounds["max_exclusive"]:
        return False, f"value {value} is not below configured maximum {bounds['max_exclusive']}"
    if "min" in bounds and value < bounds["min"]:
        return False, f"value {value} is below configured minimum {bounds['min']}"
    if "min_exclusive" in bounds and value <= bounds["min_exclusive"]:
        return False, f"value {value} is not above configured minimum {bounds['min_exclusive']}"
    return True, "within configured bound"


def evaluate_grading(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate verified observations against the configured e-NAM standard.

    No range is issued unless a human has marked the observations verified and
    selected an official range.  This avoids silently filling gaps or inferring
    assayer judgement from YOLO detections.
    """
    standard = get_standard()
    human = payload.get("human_verified_observations") or {}
    verification_status = str(payload.get("human_verification", {}).get("status", "pending")).lower()
    result: dict[str, Any] = {
        "inspection_id": payload.get("inspection_id"),
        "standard": {"id": standard["id"], "name": standard["name"], "source": standard["source"]},
        "ai_observations": map_ai_observations(payload.get("ai_observations")),
        "human_verified_observations": human,
        "human_verification": {"status": verification_status},
        "grading": {"status": "pending_verification", "range": None, "reasons": [], "checks": []},
    }
    grading = result["grading"]
    if verification_status != "verified":
        grading["reasons"].append("Human verification is required before procurement grading.")
        return result

    missing = [field for field in standard["required_verified_fields"] if not _is_number(human.get(field))]
    selected_range = human.get("selected_range")
    if missing or selected_range not in standard["range_names"]:
        grading["status"] = "not_enough_information"
        if missing:
            grading["reasons"].append("Missing verified observations: " + ", ".join(missing) + ".")
        if selected_range not in standard["range_names"]:
            grading["reasons"].append("A human-verified official Range-I, Range-II, or Range-III selection is required.")
        return result

    checks = []
    all_pass = True
    for field, rule in standard["rules"].items():
        if rule.get("optional") and not _is_number(human.get(field)):
            continue
        if not _is_number(human.get(field)):
            continue
        passed, detail = _check_rule(float(human[field]), rule["ranges"][selected_range])
        checks.append({"field": field, "label": rule["label"], "value": human[field], "passed": passed, "detail": detail})
        all_pass = all_pass and passed

    grading["checks"] = checks
    if not all_pass:
        grading["status"] = "not_enough_information"
        grading["reasons"].append("Verified observations do not satisfy every configured bound for the selected official range.")
        return result

    grading["status"] = "graded"
    grading["range"] = selected_range
    grading["reasons"].append(f"Human-verified observations satisfy the configured {selected_range} bounds.")
    return result
