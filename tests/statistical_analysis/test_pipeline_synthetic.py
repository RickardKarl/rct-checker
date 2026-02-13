"""Integration test: p-value uniformity under the null hypothesis.

Generates many synthetic datasets with proper randomization (all groups
drawn from the same distribution) and checks that the pipeline's combined
p-values do not reject H0 more often than expected (~5%).
"""

import pytest

from src.statistical_analysis.pipeline import run_test_pipeline
from tests.statistical_analysis.generate_synthetic_json import generate_synthetic_json

N_SIMULATIONS = 1_000
ALPHA = 0.05
MAX_FALSE_POSITIVE_RATE = ALPHA + 5e-2


@pytest.mark.integration_statistical_analysis
def test_pvalue_false_positive_rate_under_null():
    """Under proper randomization, at most ~5% of combined p-values should be < 0.05."""
    rejections = 0

    for seed in range(N_SIMULATIONS):
        data = generate_synthetic_json(seed=seed)
        result = run_test_pipeline(data, random_seed=seed)
        p_value = result["fisher_method-combined"]["p_value"]

        if p_value < ALPHA:
            rejections += 1

    false_positive_rate = rejections / N_SIMULATIONS

    assert false_positive_rate <= MAX_FALSE_POSITIVE_RATE, (
        f"False positive rate {false_positive_rate:.3f} ({rejections}/{N_SIMULATIONS}) "
        f"exceeds maximum allowed rate of {MAX_FALSE_POSITIVE_RATE}"
    )
