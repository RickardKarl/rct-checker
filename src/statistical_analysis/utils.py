import logging

import numpy as np

logger = logging.getLogger(__name__)


def process_categorical_variables(df, total_sample_size):
    """
    Process categorical variables from a DataFrame and compute population rates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing variable data with a "Variable type" column.
    total_sample_size : int
        Total sample size across all groups.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only categorical variables with valid counts
        and a computed "population_rate" column.

    Raises
    ------
    ValueError
        If total_sample_size is not positive.
    """
    if df is None or df.empty:
        return df
    if total_sample_size <= 0:
        raise ValueError("total_sample_size must be positive")

    cat_df = df.copy()
    cat_df = cat_df[cat_df["Variable type"] == "Categorical"]

    # Get rows where all count columns are not NAN (need complete data for all groups)
    count_cols = cat_df.filter(regex="(count)").columns
    cat_df = cat_df[cat_df[count_cols].notna().all(axis=1)]

    # Compute population rate
    cat_df["population_rate"] = cat_df[count_cols].sum(axis=1) / total_sample_size

    return cat_df


def process_continuous_variables_mean(df, sample_size):
    """
    Process continuous mean variables from a DataFrame and compute z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing variable data with a "Variable type" column.
    sample_size : dict
        Dictionary mapping group IDs to their sample sizes.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only continuous variables with computed
        z-scores, SEMs, and population means.

    Raises
    ------
    ValueError
        If sample_size is empty, contains non-positive values, or if SD/CI
        data is missing for z-score computation.
    """
    if df is None or df.empty:
        return df
    if not sample_size:
        raise ValueError("sample_size must not be empty")
    if any(s <= 0 for s in sample_size.values()):
        raise ValueError("All sample sizes must be positive")

    group_ids = list(sample_size.keys())
    mean_df = df.copy()
    mean_df = mean_df[mean_df["Variable type"] == "Continuous"]

    # Get rows where mean is not NAN
    mean_cols = mean_df.filter(regex="(mean)").columns
    mean_df = mean_df[mean_df[mean_cols].notna().any(axis=1)]

    # Compute population mean
    mean_df["population_mean"] = mean_df[mean_cols].mean(axis=1)

    # Compute Z-scores
    for group in group_ids:
        mean_col = f"{group} (mean)"
        sd_col = f"{group} (sd)"
        sem_col = f"{group} (sem)"
        z_col = f"{group} (zscore)"

        # Check if SD is missing, compute from 95CI_lower and 95CI_upper. If these are not available, raise error.
        if mean_df[sd_col].isna().any():
            missing_sd_mask = mean_df[sd_col].isna()

            ci_lower_col = f"{group} (95CI_lower)"
            ci_upper_col = f"{group} (95CI_upper)"

            # check for the rows where sd is missing whether ci_upper_col and ci_lower_col are not missing
            if mean_df.loc[missing_sd_mask, [ci_lower_col, ci_upper_col]].isna().any().any():
                raise ValueError(
                    f"Cannot compute SD for {group}: both SD and confidence intervals are missing"
                )

            ci_width = (
                mean_df.loc[missing_sd_mask, ci_upper_col]
                - mean_df.loc[missing_sd_mask, ci_lower_col]
            )
            mean_df.loc[missing_sd_mask, sd_col] = (ci_width / (2 * 1.96)) * np.sqrt(
                sample_size[group]
            )

        sem = mean_df[sd_col] / np.sqrt(sample_size[group])
        if (sem == 0).any():
            raise ValueError(f"SEM is zero for group {group}, cannot compute z-scores")
        zscore = (mean_df[mean_col] - mean_df["population_mean"]) / sem

        mean_df[z_col] = zscore
        mean_df[sem_col] = sem

    return mean_df


def contingency_table_binary(group_count, group_total):
    """
    Build a k x 2 contingency table (Success / Failure) for k groups.

    Parameters
    ----------
    group_count : dict
        {group_name: success_count}
    group_total : dict
        {group_name: total_count}

    Returns
    -------
    table : np.ndarray
        Shape (k, 2): [success, failure]
    labels : list
        Group labels in row order

    Raises
    ------
    ValueError
        If success count exceeds total count for any group.
    """
    labels = list(group_count.keys())
    table = np.zeros((len(labels), 2), dtype=int)

    for i, g in enumerate(labels):
        success = group_count[g]
        total = group_total[g]
        if success > total:
            raise ValueError(
                f"Invalid data for group '{g}': success count ({success}) "
                f"exceeds total count ({total})"
            )
        table[i, 0] = success
        table[i, 1] = total - success

    return table
