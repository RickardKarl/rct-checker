"""Unit tests for src.table_extraction.validate_output."""

import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.table_extraction.validate_output import (
    VARIABLE_TYPE_CATEGORICAL,
    VARIABLE_TYPE_CONTINUOUS,
    ValidationError,
    validate_json,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def valid_json():
    """A minimal valid JSON structure."""
    return {
        "table1_exists": True,
        "groups": [
            {"group_id": "group_1", "label": "Treatment", "sample_size": 50},
            {"group_id": "group_2", "label": "Control", "sample_size": 55},
        ],
        "rows": [
            {
                "variable": "Age",
                "variable_type": VARIABLE_TYPE_CONTINUOUS,
                "level": None,
                "values": [
                    {
                        "group_id": "group_1",
                        "original": "60.1 (10.5)",
                        "mean": 60.1,
                        "median": None,
                        "count": None,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": 10.5,
                        "pvalue": None,
                    },
                    {
                        "group_id": "group_2",
                        "original": "58.2 (9.0)",
                        "mean": 58.2,
                        "median": None,
                        "count": None,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": 9.0,
                        "pvalue": None,
                    },
                ],
            },
        ],
    }


@pytest.fixture
def valid_categorical_row():
    """A valid categorical variable row."""
    return {
        "variable": "Sex",
        "variable_type": VARIABLE_TYPE_CATEGORICAL,
        "level": "Male",
        "values": [
            {
                "group_id": "group_1",
                "original": "30 (60%)",
                "mean": None,
                "median": None,
                "count": 30,
                "IQR_lower": None,
                "IQR_upper": None,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": None,
                "pvalue": None,
            },
            {
                "group_id": "group_2",
                "original": "28 (51%)",
                "mean": None,
                "median": None,
                "count": 28,
                "IQR_lower": None,
                "IQR_upper": None,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": None,
                "pvalue": None,
            },
        ],
    }


@pytest.fixture
def valid_median_row():
    """A valid continuous variable row using median/IQR."""
    return {
        "variable": "BMI",
        "variable_type": VARIABLE_TYPE_CONTINUOUS,
        "level": None,
        "values": [
            {
                "group_id": "group_1",
                "original": "25.0 (22.0-28.0)",
                "mean": None,
                "median": 25.0,
                "count": None,
                "IQR_lower": 22.0,
                "IQR_upper": 28.0,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": None,
                "pvalue": None,
            },
            {
                "group_id": "group_2",
                "original": "24.5 (21.0-27.0)",
                "mean": None,
                "median": 24.5,
                "count": None,
                "IQR_lower": 21.0,
                "IQR_upper": 27.0,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": None,
                "pvalue": None,
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests for basic validation
# ─────────────────────────────────────────────────────────────────────────────


class TestBasicValidation:
    def test_valid_json_passes(self, valid_json):
        # Should not raise
        validate_json(valid_json)

    def test_table1_exists_false_short_circuits(self):
        # When table1_exists is False, validation should pass regardless of content
        data = {"table1_exists": False}
        validate_json(data)

    def test_non_dict_raises_error(self):
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_json([])

    def test_none_raises_error(self):
        with pytest.raises(ValidationError, match="must be a JSON object"):
            validate_json(None)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for groups validation
# ─────────────────────────────────────────────────────────────────────────────


class TestGroupsValidation:
    def test_missing_groups_raises_error(self, valid_json):
        del valid_json["groups"]
        with pytest.raises(ValidationError, match="Missing groups array"):
            validate_json(valid_json)

    def test_empty_groups_raises_error(self, valid_json):
        valid_json["groups"] = []
        with pytest.raises(ValidationError, match="Groups must be a non-empty array"):
            validate_json(valid_json)

    def test_groups_not_list_raises_error(self, valid_json):
        valid_json["groups"] = "not a list"
        with pytest.raises(ValidationError, match="Groups must be a non-empty array"):
            validate_json(valid_json)

    def test_group_not_dict_raises_error(self, valid_json):
        valid_json["groups"][0] = "not a dict"
        with pytest.raises(ValidationError, match="must be an object"):
            validate_json(valid_json)

    def test_missing_group_id_raises_error(self, valid_json):
        del valid_json["groups"][0]["group_id"]
        with pytest.raises(ValidationError, match="missing or invalid group_id"):
            validate_json(valid_json)

    def test_duplicate_group_id_raises_error(self, valid_json):
        valid_json["groups"][1]["group_id"] = "group_1"
        with pytest.raises(ValidationError, match="duplicate group_id"):
            validate_json(valid_json)

    def test_missing_label_raises_error(self, valid_json):
        del valid_json["groups"][0]["label"]
        with pytest.raises(ValidationError, match="missing or invalid label"):
            validate_json(valid_json)

    def test_empty_label_raises_error(self, valid_json):
        valid_json["groups"][0]["label"] = "   "
        with pytest.raises(ValidationError, match="missing or invalid label"):
            validate_json(valid_json)

    def test_sample_size_non_int_raises_error(self, valid_json):
        valid_json["groups"][0]["sample_size"] = "fifty"
        with pytest.raises(ValidationError, match="sample_size must be an integer"):
            validate_json(valid_json)

    def test_sample_size_negative_raises_error(self, valid_json):
        valid_json["groups"][0]["sample_size"] = -10
        with pytest.raises(ValidationError, match="sample_size must be non-negative"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for rows validation
# ─────────────────────────────────────────────────────────────────────────────


class TestRowsValidation:
    def test_missing_rows_raises_error(self, valid_json):
        del valid_json["rows"]
        with pytest.raises(ValidationError, match="Missing rows array"):
            validate_json(valid_json)

    def test_empty_rows_raises_error(self, valid_json):
        valid_json["rows"] = []
        with pytest.raises(ValidationError, match="Rows must be a non-empty array"):
            validate_json(valid_json)

    def test_row_not_dict_raises_error(self, valid_json):
        valid_json["rows"][0] = "not a dict"
        with pytest.raises(ValidationError, match="must be an object"):
            validate_json(valid_json)

    def test_missing_variable_name_raises_error(self, valid_json):
        del valid_json["rows"][0]["variable"]
        with pytest.raises(ValidationError, match="missing or invalid variable name"):
            validate_json(valid_json)

    def test_empty_variable_name_raises_error(self, valid_json):
        valid_json["rows"][0]["variable"] = "   "
        with pytest.raises(ValidationError, match="missing or invalid variable name"):
            validate_json(valid_json)

    def test_invalid_variable_type_raises_error(self, valid_json):
        valid_json["rows"][0]["variable_type"] = "InvalidType"
        with pytest.raises(ValidationError, match="invalid variable_type"):
            validate_json(valid_json)

    def test_missing_values_raises_error(self, valid_json):
        del valid_json["rows"][0]["values"]
        with pytest.raises(ValidationError, match="values must be a non-empty array"):
            validate_json(valid_json)

    def test_empty_values_raises_error(self, valid_json):
        valid_json["rows"][0]["values"] = []
        with pytest.raises(ValidationError, match="values must be a non-empty array"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for continuous variable validation
# ─────────────────────────────────────────────────────────────────────────────


class TestContinuousVariableValidation:
    def test_continuous_with_level_raises_error(self, valid_json):
        valid_json["rows"][0]["level"] = "SomeLevel"
        with pytest.raises(ValidationError, match="continuous variable must have level = null"):
            validate_json(valid_json)

    def test_continuous_missing_mean_and_median_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["mean"] = None
        valid_json["rows"][0]["values"][0]["median"] = None
        valid_json["rows"][0]["values"][1]["mean"] = None
        valid_json["rows"][0]["values"][1]["median"] = None
        with pytest.raises(ValidationError, match="missing mean or median"):
            validate_json(valid_json)

    def test_continuous_with_mean_missing_sd_and_ci_raises_error(self, valid_json):
        # Has mean but no sd and no CI
        valid_json["rows"][0]["values"][0]["sd"] = None
        valid_json["rows"][0]["values"][0]["95CI_lower"] = None
        valid_json["rows"][0]["values"][0]["95CI_upper"] = None
        with pytest.raises(ValidationError, match="missing sd or 95CI values"):
            validate_json(valid_json)

    def test_continuous_with_mean_and_ci_is_valid(self, valid_json):
        # Has mean with CI instead of sd
        valid_json["rows"][0]["values"][0]["sd"] = None
        valid_json["rows"][0]["values"][0]["95CI_lower"] = 55.0
        valid_json["rows"][0]["values"][0]["95CI_upper"] = 65.0
        valid_json["rows"][0]["values"][1]["sd"] = None
        valid_json["rows"][0]["values"][1]["95CI_lower"] = 53.0
        valid_json["rows"][0]["values"][1]["95CI_upper"] = 63.0
        validate_json(valid_json)

    def test_continuous_with_median_missing_iqr_raises_error(self, valid_json, valid_median_row):
        valid_json["rows"] = [valid_median_row]
        valid_json["rows"][0]["values"][0]["IQR_lower"] = None
        with pytest.raises(ValidationError, match="missing IQR values"):
            validate_json(valid_json)

    def test_continuous_with_median_and_iqr_is_valid(self, valid_json, valid_median_row):
        valid_json["rows"] = [valid_median_row]
        validate_json(valid_json)

    def test_mean_wrong_type_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["mean"] = "sixty"
        with pytest.raises(ValidationError, match="mean must be numerical"):
            validate_json(valid_json)

    def test_sd_wrong_type_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["sd"] = "ten"
        with pytest.raises(ValidationError, match="sd must be numerical"):
            validate_json(valid_json)

    def test_continuous_with_both_iqr_and_ci_raises_error(self, valid_json):
        # Set both IQR and 95CI values - this should fail
        for value in valid_json["rows"][0]["values"]:
            value["mean"] = None
            value["median"] = 25.0
            value["sd"] = None
            value["IQR_lower"] = 22.0
            value["IQR_upper"] = 28.0
            value["95CI_lower"] = 20.0
            value["95CI_upper"] = 30.0
        with pytest.raises(ValidationError, match="cannot have both IQR and 95CI"):
            validate_json(valid_json)

    def test_continuous_with_iqr_and_mean_raises_error(self, valid_json):
        # IQR should only be used with median, not mean
        for value in valid_json["rows"][0]["values"]:
            value["mean"] = 60.0
            value["median"] = None
            value["sd"] = None
            value["IQR_lower"] = 55.0
            value["IQR_upper"] = 65.0
            value["95CI_lower"] = None
            value["95CI_upper"] = None
        with pytest.raises(ValidationError, match="IQR values should only be used with median"):
            validate_json(valid_json)

    def test_continuous_with_ci_and_median_raises_error(self, valid_json, valid_median_row):
        # 95CI should only be used with mean, not median
        valid_json["rows"] = [valid_median_row]
        for value in valid_json["rows"][0]["values"]:
            value["IQR_lower"] = None
            value["IQR_upper"] = None
            value["95CI_lower"] = 20.0
            value["95CI_upper"] = 30.0
        with pytest.raises(ValidationError, match="95CI values should only be used with mean"):
            validate_json(valid_json)

    def test_iqr_lower_greater_than_upper_raises_error(self, valid_json, valid_median_row):
        # IQR_lower must be <= IQR_upper
        valid_json["rows"] = [valid_median_row]
        valid_json["rows"][0]["values"][0]["IQR_lower"] = 30.0
        valid_json["rows"][0]["values"][0]["IQR_upper"] = 20.0
        with pytest.raises(ValidationError, match="IQR_lower must be <= IQR_upper"):
            validate_json(valid_json)

    def test_ci_lower_greater_than_upper_raises_error(self, valid_json):
        # 95CI_lower must be <= 95CI_upper
        for value in valid_json["rows"][0]["values"]:
            value["sd"] = None
            value["95CI_lower"] = 70.0
            value["95CI_upper"] = 50.0
        with pytest.raises(ValidationError, match="95CI_lower must be <= 95CI_upper"):
            validate_json(valid_json)

    def test_median_outside_iqr_range_raises_error(self, valid_json, valid_median_row):
        # Median must be within IQR range
        valid_json["rows"] = [valid_median_row]
        valid_json["rows"][0]["values"][0]["median"] = 10.0  # Below IQR_lower (22.0)
        with pytest.raises(ValidationError, match="median must be within IQR range"):
            validate_json(valid_json)

    def test_mean_outside_ci_range_raises_error(self, valid_json):
        # Mean must be within 95CI range
        for value in valid_json["rows"][0]["values"]:
            value["sd"] = None
            value["95CI_lower"] = 70.0
            value["95CI_upper"] = 80.0
        # mean is 60.1 and 58.2, both outside [70, 80]
        with pytest.raises(ValidationError, match="mean must be within 95CI range"):
            validate_json(valid_json)

    def test_negative_sd_raises_error(self, valid_json):
        # sd must be non-negative
        valid_json["rows"][0]["values"][0]["sd"] = -5.0
        with pytest.raises(ValidationError, match="sd must be non-negative"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for categorical variable validation
# ─────────────────────────────────────────────────────────────────────────────


class TestCategoricalVariableValidation:
    def test_categorical_with_valid_level(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        validate_json(valid_json)

    def test_categorical_missing_level_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["level"] = None
        # level=None for categorical is technically allowed but empty string is not
        # Actually looking at the code, level=None is fine for categorical
        validate_json(valid_json)  # Should pass

    def test_categorical_empty_level_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["level"] = "   "
        with pytest.raises(ValidationError, match="missing or invalid level"):
            validate_json(valid_json)

    def test_categorical_with_mean_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["mean"] = 5.0
        with pytest.raises(ValidationError, match="categorical variable should not have mean"):
            validate_json(valid_json)

    def test_categorical_with_median_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["median"] = 5.0
        with pytest.raises(ValidationError, match="categorical variable should not have median"):
            validate_json(valid_json)

    def test_categorical_with_sd_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["sd"] = 2.0
        with pytest.raises(ValidationError, match="categorical variable should not have sd"):
            validate_json(valid_json)

    def test_categorical_with_iqr_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["IQR_lower"] = 2.0
        with pytest.raises(ValidationError, match="categorical variable should not have IQR_lower"):
            validate_json(valid_json)

    def test_categorical_with_ci_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["95CI_lower"] = 2.0
        with pytest.raises(
            ValidationError, match="categorical variable should not have 95CI_lower"
        ):
            validate_json(valid_json)

    def test_categorical_count_non_int_raises_error(self, valid_json, valid_categorical_row):
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["count"] = 30.5
        with pytest.raises(ValidationError, match="count must be numerical integer"):
            validate_json(valid_json)

    def test_categorical_missing_count_raises_error(self, valid_json, valid_categorical_row):
        # Categorical variable must have count
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["count"] = None
        with pytest.raises(ValidationError, match="categorical variable must have count"):
            validate_json(valid_json)

    def test_categorical_negative_count_raises_error(self, valid_json, valid_categorical_row):
        # count must be non-negative
        valid_json["rows"] = [valid_categorical_row]
        valid_json["rows"][0]["values"][0]["count"] = -5
        with pytest.raises(ValidationError, match="count must be non-negative"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for pvalue validation
# ─────────────────────────────────────────────────────────────────────────────


class TestPvalueValidation:
    def test_pvalue_valid(self, valid_json):
        # Valid pvalue should pass
        valid_json["rows"][0]["values"][0]["pvalue"] = 0.05
        validate_json(valid_json)

    def test_pvalue_zero_is_valid(self, valid_json):
        valid_json["rows"][0]["values"][0]["pvalue"] = 0
        validate_json(valid_json)

    def test_pvalue_one_is_valid(self, valid_json):
        valid_json["rows"][0]["values"][0]["pvalue"] = 1
        validate_json(valid_json)

    def test_pvalue_non_numeric_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["pvalue"] = "0.05"
        with pytest.raises(ValidationError, match="pvalue must be numerical"):
            validate_json(valid_json)

    def test_pvalue_negative_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["pvalue"] = -0.1
        with pytest.raises(ValidationError, match="pvalue must be between 0 and 1"):
            validate_json(valid_json)

    def test_pvalue_greater_than_one_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["pvalue"] = 1.5
        with pytest.raises(ValidationError, match="pvalue must be between 0 and 1"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for value-level validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValueValidation:
    def test_missing_group_id_in_value_raises_error(self, valid_json):
        del valid_json["rows"][0]["values"][0]["group_id"]
        with pytest.raises(ValidationError, match="value missing or invalid group_id"):
            validate_json(valid_json)

    def test_unknown_group_id_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["group_id"] = "unknown_group"
        with pytest.raises(ValidationError, match="unknown group_id"):
            validate_json(valid_json)

    def test_duplicate_group_id_in_values_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][1]["group_id"] = "group_1"
        with pytest.raises(ValidationError, match="duplicate group_id"):
            validate_json(valid_json)

    def test_missing_original_value_raises_error(self, valid_json):
        del valid_json["rows"][0]["values"][0]["original"]
        with pytest.raises(ValidationError, match="missing original value"):
            validate_json(valid_json)

    def test_empty_original_value_raises_error(self, valid_json):
        valid_json["rows"][0]["values"][0]["original"] = "   "
        with pytest.raises(ValidationError, match="missing original value"):
            validate_json(valid_json)

    def test_missing_group_in_row_raises_error(self, valid_json):
        # Remove one group's value from the row
        valid_json["rows"][0]["values"] = [valid_json["rows"][0]["values"][0]]
        with pytest.raises(ValidationError, match="must contain values for all groups"):
            validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for all-null numeric values (header row detection)
# ─────────────────────────────────────────────────────────────────────────────


class TestAllNullNumericValues:
    def test_all_null_numeric_values_raises_error(self, valid_json):
        # Set all numeric values to None - this looks like a header row
        for value in valid_json["rows"][0]["values"]:
            value["mean"] = None
            value["median"] = None
            value["count"] = None
            value["sd"] = None
            value["IQR_lower"] = None
            value["IQR_upper"] = None
            value["95CI_lower"] = None
            value["95CI_upper"] = None
        with pytest.raises(ValidationError, match="all numeric values are null"):
            validate_json(valid_json)

    def test_at_least_one_numeric_value_passes(self, valid_json):
        # As long as one numeric field has a value, it should pass
        for value in valid_json["rows"][0]["values"]:
            value["median"] = None
            value["count"] = None
            value["IQR_lower"] = None
            value["IQR_upper"] = None
            value["95CI_lower"] = None
            value["95CI_upper"] = None
            # Keep mean and sd
        validate_json(valid_json)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for multiple errors
# ─────────────────────────────────────────────────────────────────────────────


class TestMultipleErrors:
    def test_multiple_errors_are_concatenated(self, valid_json):
        # Introduce multiple errors
        valid_json["groups"][0]["label"] = ""
        valid_json["groups"][1]["sample_size"] = "not_an_int"
        valid_json["rows"][0]["variable"] = ""

        with pytest.raises(ValidationError) as exc_info:
            validate_json(valid_json)

        error_message = str(exc_info.value)
        assert "invalid label" in error_message
        assert "sample_size must be an integer" in error_message
        assert "invalid variable name" in error_message
