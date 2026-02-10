"""Unit tests for src.statistical_analysis.statistical_tests."""

import numpy as np
import pytest

from src.statistical_analysis.statistical_tests import chi_square_variance_test

# ─────────────────────────────────────────────────────────────────────────────
# Test configuration
# ─────────────────────────────────────────────────────────────────────────────

N_SIMULATIONS = 1000
SAMPLE_SIZE = 50
ALPHA = 0.05
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Tests for chi_square_variance_test - Type 1 error and power
# ─────────────────────────────────────────────────────────────────────────────


class TestChiSquareVarianceTestTypeIError:
    """Test that type 1 error rate is controlled at alpha level."""

    @pytest.mark.parametrize("true_sigma", [0.5, 1.0, 2.0])
    def test_type1_error_controlled_when_sigma_matches(self, true_sigma):
        """
        When sigma0 matches the true standard deviation, the type 1 error
        rate should be approximately alpha (0.05).

        We use a one-sided binomial test to check that the observed rejection
        rate is not significantly greater than alpha.
        """
        rng = np.random.default_rng(RANDOM_SEED)
        rejections = 0

        for _ in range(N_SIMULATIONS):
            # Generate z-scores from N(0, true_sigma^2)
            zscores = rng.normal(loc=0, scale=true_sigma, size=SAMPLE_SIZE)
            p_value, _ = chi_square_variance_test(zscores, sigma0=true_sigma)

            if p_value < ALPHA:
                rejections += 1

        observed_rate = rejections / N_SIMULATIONS

        # Allow some margin for sampling variability
        # Using a conservative upper bound: alpha + 2*SE where SE = sqrt(alpha*(1-alpha)/n)
        se = np.sqrt(ALPHA * (1 - ALPHA) / N_SIMULATIONS)
        upper_bound = ALPHA + 3 * se  # ~99.7% CI upper bound

        assert observed_rate <= upper_bound, (
            f"Type 1 error rate {observed_rate:.3f} exceeds expected upper bound "
            f"{upper_bound:.3f} for true_sigma={true_sigma}"
        )


class TestChiSquareVarianceTestPower:
    """Test that the test has adequate power to detect variance deviations."""

    @pytest.mark.parametrize(
        "true_sigma,hypothesized_sigma,min_power",
        [
            # When true variance is 2x hypothesized, expect high power
            (2.0, 1.0, 0.80),
            (1.0, 0.5, 0.80),
            # When true variance is 0.5x hypothesized, expect high power
            (0.5, 1.0, 0.80),
            (1.0, 2.0, 0.80),
            # Smaller effect size - lower power threshold
            (1.5, 1.0, 0.50),
            (1.0, 1.5, 0.50),
        ],
    )
    def test_power_when_sigma_mismatched(self, true_sigma, hypothesized_sigma, min_power):
        """
        When sigma0 does not match the true standard deviation, the test
        should reject with high probability (good power).
        """
        rng = np.random.default_rng(RANDOM_SEED)
        rejections = 0

        for _ in range(N_SIMULATIONS):
            # Generate z-scores from N(0, true_sigma^2)
            zscores = rng.normal(loc=0, scale=true_sigma, size=SAMPLE_SIZE)
            p_value, _ = chi_square_variance_test(zscores, sigma0=hypothesized_sigma)

            if p_value < ALPHA:
                rejections += 1

        observed_power = rejections / N_SIMULATIONS

        assert observed_power >= min_power, (
            f"Power {observed_power:.3f} is below minimum {min_power:.3f} "
            f"for true_sigma={true_sigma}, hypothesized_sigma={hypothesized_sigma}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests for chi_square_variance_test - Input validation
# ─────────────────────────────────────────────────────────────────────────────


class TestChiSquareVarianceTestInputValidation:
    """Test input validation for chi_square_variance_test."""

    def test_empty_zscores_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            chi_square_variance_test([])

    def test_none_zscores_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            chi_square_variance_test(None)

    def test_single_element_raises(self):
        with pytest.raises(ValueError, match="at least 2 elements"):
            chi_square_variance_test([1.0])

    def test_nan_in_zscores_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            chi_square_variance_test([1.0, np.nan, 2.0])

    def test_inf_in_zscores_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            chi_square_variance_test([1.0, np.inf, 2.0])

    def test_negative_inf_in_zscores_raises(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            chi_square_variance_test([1.0, -np.inf, 2.0])

    def test_zero_sigma0_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            chi_square_variance_test([1.0, 2.0, 3.0], sigma0=0)

    def test_negative_sigma0_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            chi_square_variance_test([1.0, 2.0, 3.0], sigma0=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for chi_square_variance_test - Output validation
# ─────────────────────────────────────────────────────────────────────────────


class TestChiSquareVarianceTestOutput:
    """Test output properties of chi_square_variance_test."""

    def test_returns_tuple_of_two(self):
        result = chi_square_variance_test([1.0, 2.0, 3.0])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pvalue_between_0_and_1(self):
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(100):
            zscores = rng.normal(size=30)
            p_value, _ = chi_square_variance_test(zscores)
            assert 0 <= p_value <= 1

    def test_test_statistic_non_negative(self):
        rng = np.random.default_rng(RANDOM_SEED)
        for _ in range(100):
            zscores = rng.normal(size=30)
            _, test_stat = chi_square_variance_test(zscores)
            assert test_stat >= 0
