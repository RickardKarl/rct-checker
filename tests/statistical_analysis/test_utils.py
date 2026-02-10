"""Unit tests for src.statistical_analysis.utils."""

import numpy as np
import pandas as pd
import pytest

from src.statistical_analysis.utils import (
    contingency_table_binary,
    process_categorical_variables,
    process_continuous_variables_mean,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def categorical_df():
    """DataFrame with categorical variables."""
    return pd.DataFrame(
        {
            "Variable": ["Sex (Male)", "Smoker (Yes)", "Diabetes (Yes)"],
            "Variable type": ["Categorical", "Categorical", "Categorical"],
            "group_1 (count)": [30, 15, 10],
            "group_2 (count)": [28, 20, 12],
        }
    )


@pytest.fixture
def continuous_df():
    """DataFrame with continuous variables."""
    return pd.DataFrame(
        {
            "Variable": ["Age", "BMI", "Blood Pressure"],
            "Variable type": ["Continuous", "Continuous", "Continuous"],
            "group_1 (mean)": [60.0, 25.0, 120.0],
            "group_1 (sd)": [10.0, 4.0, 15.0],
            "group_2 (mean)": [58.0, 26.0, 118.0],
            "group_2 (sd)": [9.0, 5.0, 14.0],
        }
    )


@pytest.fixture
def mixed_df():
    """DataFrame with both categorical and continuous variables."""
    return pd.DataFrame(
        {
            "Variable": ["Age", "Sex (Male)", "BMI"],
            "Variable type": ["Continuous", "Categorical", "Continuous"],
            "group_1 (mean)": [60.0, np.nan, 25.0],
            "group_1 (sd)": [10.0, np.nan, 4.0],
            "group_1 (count)": [np.nan, 30, np.nan],
            "group_2 (mean)": [58.0, np.nan, 26.0],
            "group_2 (sd)": [9.0, np.nan, 5.0],
            "group_2 (count)": [np.nan, 28, np.nan],
        }
    )


@pytest.fixture
def sample_size_two_groups():
    """Sample sizes for two groups."""
    return {"group_1": 50, "group_2": 55}


# ─────────────────────────────────────────────────────────────────────────────
# Tests for process_categorical_variables
# ─────────────────────────────────────────────────────────────────────────────


class TestProcessCategoricalVariables:
    def test_filters_categorical_only(self, mixed_df):
        result = process_categorical_variables(mixed_df, total_sample_size=105)
        assert len(result) == 1
        assert result.iloc[0]["Variable"] == "Sex (Male)"

    def test_computes_population_rate(self, categorical_df):
        total_sample_size = 105  # 50 + 55
        result = process_categorical_variables(categorical_df, total_sample_size)

        # Sex: (30 + 28) / 105 = 0.552...
        sex_row = result[result["Variable"] == "Sex (Male)"]
        expected_rate = (30 + 28) / 105
        assert abs(sex_row["population_rate"].iloc[0] - expected_rate) < 1e-10

    def test_filters_rows_with_missing_counts(self):
        df = pd.DataFrame(
            {
                "Variable": ["Sex (Male)", "Smoker (Yes)"],
                "Variable type": ["Categorical", "Categorical"],
                "group_1 (count)": [30, np.nan],
                "group_2 (count)": [28, 20],
            }
        )
        result = process_categorical_variables(df, total_sample_size=105)
        assert len(result) == 1
        assert result.iloc[0]["Variable"] == "Sex (Male)"

    def test_returns_none_for_none_input(self):
        result = process_categorical_variables(None, total_sample_size=100)
        assert result is None

    def test_returns_empty_for_empty_df(self):
        empty_df = pd.DataFrame()
        result = process_categorical_variables(empty_df, total_sample_size=100)
        assert result.empty

    def test_raises_for_zero_sample_size(self, categorical_df):
        with pytest.raises(ValueError, match="must be positive"):
            process_categorical_variables(categorical_df, total_sample_size=0)

    def test_raises_for_negative_sample_size(self, categorical_df):
        with pytest.raises(ValueError, match="must be positive"):
            process_categorical_variables(categorical_df, total_sample_size=-10)

    def test_returns_empty_when_no_categorical_vars(self, continuous_df):
        result = process_categorical_variables(continuous_df, total_sample_size=105)
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests for process_continuous_variables_mean
# ─────────────────────────────────────────────────────────────────────────────


class TestProcessContinuousVariables:
    def test_filters_continuous_only(self, mixed_df, sample_size_two_groups):
        result = process_continuous_variables_mean(mixed_df, sample_size_two_groups)
        assert len(result) == 2
        assert set(result["Variable"]) == {"Age", "BMI"}

    def test_computes_population_mean(self, continuous_df, sample_size_two_groups):
        result = process_continuous_variables_mean(continuous_df, sample_size_two_groups)

        # Age: (60 + 58) / 2 = 59
        age_row = result[result["Variable"] == "Age"]
        assert age_row["population_mean"].iloc[0] == 59.0

    def test_computes_sem(self, continuous_df, sample_size_two_groups):
        result = process_continuous_variables_mean(continuous_df, sample_size_two_groups)

        age_row = result[result["Variable"] == "Age"]
        # SEM for group_1: 10 / sqrt(50) = 1.414...
        expected_sem = 10.0 / np.sqrt(50)
        assert abs(age_row["group_1 (sem)"].iloc[0] - expected_sem) < 1e-10

    def test_computes_zscore(self, continuous_df, sample_size_two_groups):
        result = process_continuous_variables_mean(continuous_df, sample_size_two_groups)

        age_row = result[result["Variable"] == "Age"]
        # z = (mean - pop_mean) / sem
        # z_group1 = (60 - 59) / (10/sqrt(50)) = 1 / 1.414... = 0.707...
        sem = 10.0 / np.sqrt(50)
        expected_zscore = (60.0 - 59.0) / sem
        assert abs(age_row["group_1 (zscore)"].iloc[0] - expected_zscore) < 1e-10

    def test_computes_sd_from_ci_when_missing(self, sample_size_two_groups):
        """When SD is missing but CI is available, SD should be computed."""
        df = pd.DataFrame(
            {
                "Variable": ["Age"],
                "Variable type": ["Continuous"],
                "group_1 (mean)": [60.0],
                "group_1 (sd)": [np.nan],
                "group_1 (95CI_lower)": [57.0],
                "group_1 (95CI_upper)": [63.0],
                "group_2 (mean)": [58.0],
                "group_2 (sd)": [9.0],
            }
        )
        result = process_continuous_variables_mean(df, sample_size_two_groups)

        # CI width = 63 - 57 = 6
        # SD = (6 / (2 * 1.96)) * sqrt(50) = (6 / 3.92) * 7.07 = 10.82...
        ci_width = 6.0
        expected_sd = (ci_width / (2 * 1.96)) * np.sqrt(50)
        assert abs(result["group_1 (sd)"].iloc[0] - expected_sd) < 1e-6

    def test_raises_when_sd_and_ci_missing(self, sample_size_two_groups):
        df = pd.DataFrame(
            {
                "Variable": ["Age"],
                "Variable type": ["Continuous"],
                "group_1 (mean)": [60.0],
                "group_1 (sd)": [np.nan],
                "group_1 (95CI_lower)": [np.nan],
                "group_1 (95CI_upper)": [np.nan],
                "group_2 (mean)": [58.0],
                "group_2 (sd)": [9.0],
            }
        )
        with pytest.raises(ValueError, match="Cannot compute SD"):
            process_continuous_variables_mean(df, sample_size_two_groups)

    def test_raises_when_sem_is_zero(self, sample_size_two_groups):
        df = pd.DataFrame(
            {
                "Variable": ["Age"],
                "Variable type": ["Continuous"],
                "group_1 (mean)": [60.0],
                "group_1 (sd)": [0.0],  # Zero SD leads to zero SEM
                "group_2 (mean)": [58.0],
                "group_2 (sd)": [9.0],
            }
        )
        with pytest.raises(ValueError, match="SEM is zero"):
            process_continuous_variables_mean(df, sample_size_two_groups)

    def test_returns_none_for_none_input(self, sample_size_two_groups):
        result = process_continuous_variables_mean(None, sample_size_two_groups)
        assert result is None

    def test_returns_empty_for_empty_df(self, sample_size_two_groups):
        empty_df = pd.DataFrame()
        result = process_continuous_variables_mean(empty_df, sample_size_two_groups)
        assert result.empty

    def test_raises_for_empty_sample_size(self, continuous_df):
        with pytest.raises(ValueError, match="must not be empty"):
            process_continuous_variables_mean(continuous_df, {})

    def test_raises_for_zero_sample_size(self, continuous_df):
        with pytest.raises(ValueError, match="must be positive"):
            process_continuous_variables_mean(continuous_df, {"group_1": 50, "group_2": 0})

    def test_raises_for_negative_sample_size(self, continuous_df):
        with pytest.raises(ValueError, match="must be positive"):
            process_continuous_variables_mean(continuous_df, {"group_1": -10, "group_2": 55})

    def test_filters_rows_with_all_missing_means(self, sample_size_two_groups):
        df = pd.DataFrame(
            {
                "Variable": ["Age", "BMI"],
                "Variable type": ["Continuous", "Continuous"],
                "group_1 (mean)": [60.0, np.nan],
                "group_1 (sd)": [10.0, np.nan],
                "group_2 (mean)": [58.0, np.nan],
                "group_2 (sd)": [9.0, np.nan],
            }
        )
        result = process_continuous_variables_mean(df, sample_size_two_groups)
        assert len(result) == 1
        assert result.iloc[0]["Variable"] == "Age"

    def test_returns_empty_when_no_continuous_vars(self, sample_size_two_groups):
        """When all variables are categorical, return empty DataFrame."""
        df = pd.DataFrame(
            {
                "Variable": ["Sex (Male)", "Smoker (Yes)"],
                "Variable type": ["Categorical", "Categorical"],
                "group_1 (mean)": [np.nan, np.nan],
                "group_1 (sd)": [np.nan, np.nan],
                "group_2 (mean)": [np.nan, np.nan],
                "group_2 (sd)": [np.nan, np.nan],
            }
        )
        result = process_continuous_variables_mean(df, sample_size_two_groups)
        assert len(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests for contingency_table_binary
# ─────────────────────────────────────────────────────────────────────────────


class TestContingencyTableBinary:
    def test_two_groups_correct_shape(self):
        group_count = {"A": 30, "B": 25}
        group_total = {"A": 50, "B": 55}
        table = contingency_table_binary(group_count, group_total)

        assert table.shape == (2, 2)

    def test_two_groups_correct_values(self):
        group_count = {"A": 30, "B": 25}
        group_total = {"A": 50, "B": 55}
        table = contingency_table_binary(group_count, group_total)

        # Row 0 (A): [success=30, failure=50-30=20]
        # Row 1 (B): [success=25, failure=55-25=30]
        expected = np.array([[30, 20], [25, 30]])
        np.testing.assert_array_equal(table, expected)

    def test_three_groups(self):
        group_count = {"A": 10, "B": 15, "C": 20}
        group_total = {"A": 30, "B": 40, "C": 50}
        table = contingency_table_binary(group_count, group_total)

        assert table.shape == (3, 2)
        expected = np.array([[10, 20], [15, 25], [20, 30]])
        np.testing.assert_array_equal(table, expected)

    def test_preserves_group_order(self):
        # Use ordered dict-like behavior (Python 3.7+)
        group_count = {"X": 5, "Y": 10, "Z": 15}
        group_total = {"X": 20, "Y": 30, "Z": 40}
        table = contingency_table_binary(group_count, group_total)

        # First row should be X, second Y, third Z
        assert table[0, 0] == 5  # X success
        assert table[1, 0] == 10  # Y success
        assert table[2, 0] == 15  # Z success

    def test_zero_successes(self):
        group_count = {"A": 0, "B": 10}
        group_total = {"A": 50, "B": 55}
        table = contingency_table_binary(group_count, group_total)

        expected = np.array([[0, 50], [10, 45]])
        np.testing.assert_array_equal(table, expected)

    def test_all_successes(self):
        group_count = {"A": 50, "B": 55}
        group_total = {"A": 50, "B": 55}
        table = contingency_table_binary(group_count, group_total)

        expected = np.array([[50, 0], [55, 0]])
        np.testing.assert_array_equal(table, expected)

    def test_single_group(self):
        group_count = {"A": 30}
        group_total = {"A": 50}
        table = contingency_table_binary(group_count, group_total)

        assert table.shape == (1, 2)
        expected = np.array([[30, 20]])
        np.testing.assert_array_equal(table, expected)

    def test_returns_int_dtype(self):
        group_count = {"A": 30, "B": 25}
        group_total = {"A": 50, "B": 55}
        table = contingency_table_binary(group_count, group_total)

        assert table.dtype == np.int_

    def test_empty_groups(self):
        group_count = {}
        group_total = {}
        table = contingency_table_binary(group_count, group_total)

        assert table.shape == (0, 2)
