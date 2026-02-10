import logging

import numpy as np
from scipy.stats import MonteCarloMethod, combine_pvalues, fisher_exact

from src.statistical_analysis.statistical_tests import chi_square_variance_test
from src.statistical_analysis.utils import (
    contingency_table_binary,
    process_categorical_variables,
    process_continuous_variables_mean,
)
from src.table_extraction.utils import to_csv_wide

logger = logging.getLogger(__name__)


def run_test_pipeline(
    json_data,
    skip_continuous_var=False,
    skip_categorical_var=False,
    random_seed=None,
):
    """
    Run statistical analysis pipeline on extracted table data.

    Performs chi-squared variance tests on continuous variables and Fisher's
    exact tests on categorical variables, then combines p-values using
    Fisher's method.

    Parameters
    ----------
    json_data : dict
        Extracted table data containing "groups" key with group_id and
        sample_size for each group.
    skip_continuous_var : bool, optional
        If True, skip chi-squared variance test on continuous variables.
        Defaults to False.
    skip_categorical_var : bool, optional
        If True, skip Fisher's exact test on categorical variables.
        Defaults to False.
    random_seed : int, optional
        Seed for random number generator used in Monte Carlo Fisher's test
        (when more than 2 groups). Defaults to None (non-reproducible).

    Returns
    -------
    dict
        Dictionary containing test results with keys:
        - "cont_chi_squared_variance": chi-squared test results (if run)
        - "fisher_test-{idx}": Fisher's test results for each variable
        - "fisher_method-combined": combined p-value using Fisher's method

    Raises
    ------
    ValueError
        If total sample size is zero or no p-values are collected.
    """
    df = to_csv_wide(json_data, out_path=None)
    sample_size = {g["group_id"]: g["sample_size"] for g in json_data["groups"]}
    total_sample_size = np.sum([s for _, s in sample_size.items()])
    group_ids = list(sample_size.keys())

    if total_sample_size == 0:
        raise ValueError("Total sample size is zero. Cannot proceed with analysis.")

    ##############################
    # Run statistical tests
    ##############################

    pvals = []
    out = {}

    #########################################
    # Chi-squared variance test on z-scores
    #########################################

    if skip_continuous_var:
        logger.info("Skipping testing on continuous variables.")
    else:
        # return copy of df with rows of continous variable with zscore compute
        cont_df = process_continuous_variables_mean(df, sample_size)
        zscores = cont_df.filter(regex="(zscore)").values.flatten()

        logger.info(
            f"Found {len(cont_df)} continuous variables with mean, {len(zscores)} z-scores computed"
        )

        if len(cont_df) > 0 and len(zscores) > 0:
            p_value, test_stat = chi_square_variance_test(zscores)
            pvals.append(p_value)
            # Log results
            logger.debug(f"Chi-squared variance test: pvalue={p_value:.4f}")

            out["cont_chi_squared_variance"] = {
                "p_value": p_value,
                "test_statistic": test_stat,
                "zscores": zscores,
            }
        else:
            logger.warning("No continuous variables with mean found, skipping chi-squared test")

    ##############################
    # Count test
    ##############################

    if skip_categorical_var:
        logger.info("Skipping testing on categorical variables.")
    else:

        cat_df = process_categorical_variables(df, total_sample_size)

        if len(cat_df) == 0:
            logger.warning("No categorical variables found, skipping Fisher's exact test")
        else:
            logger.info(f"Found {len(cat_df)} categorical variables with valid counts")

            if len(group_ids) > 2:
                logger.debug(
                    "Identified more than two groups: Running Monte Carlo version of Fisher's test."
                )
                rng = np.random.default_rng(random_seed)
                method_exact_fisher = MonteCarloMethod(rng=rng)
            else:
                logger.debug("Identified two groups: Running exact version of Fisher's test.")
                method_exact_fisher = None

            for _, row in cat_df.iterrows():

                variable_name = row["Variable"]
                group_counts = {g_id: row[f"{g_id} (count)"] for g_id in group_ids}

                # Run Fisher's exact test
                contingency_table = contingency_table_binary(group_counts, sample_size)

                test_stat, p_value = fisher_exact(contingency_table, method=method_exact_fisher)
                pvals.append(p_value)

                out[f"fisher_test-{row.name}"] = {
                    "p_value": p_value,
                    "test_statistic": test_stat,
                    "contingency_table": contingency_table,
                    "variable_name": variable_name,
                    "test_statistic_is_odds_ratio": len(group_ids) == 2,
                }

                # Log results
                logger.debug(f"Fisher's exact test - row {row.name}, pvalue={p_value:.4f}")

    # Validate p-values
    if len(pvals) == 0:
        logger.error("No p-values collected. Cannot perform combined test.")
        raise ValueError("No p-values collected. Cannot perform combined test.")

    if not all(0 <= p <= 1 for p in pvals):
        logger.error(f"Invalid p-values detected (not in range [0,1]): {pvals}")
        raise ValueError("P-values must be between 0 and 1")

    test_stat, combined_p = combine_pvalues(pvals, method="fisher")

    out["fisher_method-combined"] = {
        "p_value": combined_p,
        "test_statistic": test_stat,
    }

    logger.info(
        f"Combined test (Fisher's method): statistic={test_stat:.4f}, pvalue={combined_p:.4f}"
    )

    return out
