from backend.grading_engine import evaluate_grading, get_standard


AI_OBSERVATIONS = {
    "class_counts": {"acceptable": 6, "damaged": 2, "visible_rot": 1, "sprouted": 1},
    "measurements": [{"diameter_mm": 58.2, "calibrated": True}],
}


def test_standard_loads_with_official_ranges():
    standard = get_standard()
    assert standard["range_names"] == ["Range-I", "Range-II", "Range-III"]
    assert standard["source"]["url"].startswith("https://enam.gov.in/")


def test_missing_verification_does_not_grade():
    result = evaluate_grading({"ai_observations": AI_OBSERVATIONS})
    assert result["grading"]["status"] == "pending_verification"
    assert result["grading"]["range"] is None
    assert result["grading"]["reasons"]


def test_verified_but_incomplete_observations_do_not_grade():
    result = evaluate_grading(
        {
            "ai_observations": AI_OBSERVATIONS,
            "human_verification": {"status": "verified"},
            "human_verified_observations": {"selected_range": "Range-I"},
        }
    )
    assert result["grading"]["status"] == "not_enough_information"
    assert result["grading"]["range"] is None


def test_complete_verified_observations_can_grade_configured_range():
    result = evaluate_grading(
        {
            "inspection_id": "ONI-TEST-0001",
            "ai_observations": AI_OBSERVATIONS,
            "human_verification": {"status": "verified"},
            "human_verified_observations": {
                "cut_pct": 2.0,
                "double_split_pct": 1.0,
                "sprouted_pct": 1.0,
                "rooting_pct": 1.0,
                "size_mm": 60.0,
                "selected_range": "Range-I",
            },
        }
    )
    assert result["grading"]["status"] == "graded"
    assert result["grading"]["range"] == "Range-I"
    assert result["grading"]["checks"]
