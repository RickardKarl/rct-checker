"""Tests for pipeline module."""

import pytest

from src.statistical_analysis.pipeline import run_test_pipeline

# Test data fixtures - add more here as needed
TWO_GROUPS = {
    "groups": [
        {"group_id": "treatment", "sample_size": 50},
        {"group_id": "control", "sample_size": 50},
    ],
    "rows": [
        {
            "variable": "age",
            "variable_type": "Continuous",
            "level": "",
            "values": [
                {
                    "group_id": "treatment",
                    "original": "45.2 (10.1)",
                    "mean": 45.2,
                    "sd": 10.1,
                },
                {
                    "group_id": "control",
                    "original": "44.8 (9.8)",
                    "mean": 44.8,
                    "sd": 9.8,
                },
            ],
        },
        {
            "variable": "smoker",
            "variable_type": "Categorical",
            "level": "Yes",
            "values": [
                {"group_id": "treatment", "original": "12 (24%)", "count": 12},
                {"group_id": "control", "original": "15 (30%)", "count": 15},
            ],
        },
    ],
}

THREE_GROUPS = {
    "groups": [
        {"group_id": "treatment_a", "sample_size": 40},
        {"group_id": "treatment_b", "sample_size": 40},
        {"group_id": "control", "sample_size": 40},
    ],
    "rows": [
        {
            "variable": "age",
            "variable_type": "Continuous",
            "level": "",
            "values": [
                {
                    "group_id": "treatment_a",
                    "original": "42.1 (8.5)",
                    "mean": 42.1,
                    "sd": 8.5,
                },
                {
                    "group_id": "treatment_b",
                    "original": "44.3 (9.2)",
                    "mean": 44.3,
                    "sd": 9.2,
                },
                {
                    "group_id": "control",
                    "original": "43.0 (8.8)",
                    "mean": 43.0,
                    "sd": 8.8,
                },
            ],
        },
        {
            "variable": "smoker",
            "variable_type": "Categorical",
            "level": "Yes",
            "values": [
                {"group_id": "treatment_a", "original": "8 (20%)", "count": 8},
                {"group_id": "treatment_b", "original": "10 (25%)", "count": 10},
                {"group_id": "control", "original": "12 (30%)", "count": 12},
            ],
        },
    ],
}


@pytest.mark.parametrize(
    "json_data",
    [
        pytest.param(TWO_GROUPS, id="two_groups"),
        pytest.param(THREE_GROUPS, id="three_groups"),
    ],
)
def test_run_test_pipeline_returns_combined_pvalue(json_data):
    """Test pipeline produces valid combined p-value."""
    result = run_test_pipeline(json_data)

    assert "fisher_method-combined" in result
    assert 0 <= result["fisher_method-combined"]["p_value"] <= 1
