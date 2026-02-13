import numpy as np

from src.table_extraction.validate_output import validate_json


def generate_synthetic_json(
    seed: int = 42,
    total_sample_size: int | None = None,
    num_groups: int | None = None,
    num_variables: int | None = None,
) -> dict:
    """
    Generate synthetic JSON data matching the schema expected by run_test_pipeline().

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    total_sample_size : int or None
        Total number of samples across all groups. If None, randomly chosen between 50–500.
    num_groups : int or None
        Number of groups. If None, randomly chosen between 2–4.
    num_variables : int or None
        Number of variables to generate. If None, randomly chosen between 3–10.

    Returns
    -------
    dict
        JSON-compatible dictionary validated against the expected schema.
    """
    rng = np.random.default_rng(seed)

    if total_sample_size is None:
        total_sample_size = int(rng.integers(50, 501))
    if num_groups is None:
        num_groups = int(rng.integers(2, 5))
    if num_variables is None:
        num_variables = int(rng.integers(3, 20))

    # Allocate samples to groups (each group gets at least 10)
    min_per_group = 10
    remaining = total_sample_size - min_per_group * num_groups
    if remaining < 0:
        raise ValueError(
            f"total_sample_size ({total_sample_size}) too small for "
            f"{num_groups} groups with minimum 10 each"
        )

    # Random split of remaining samples
    splits = rng.dirichlet(np.ones(num_groups)) * remaining
    group_sizes = np.floor(splits).astype(int) + min_per_group
    # Assign leftover to first group to match total exactly
    group_sizes[0] += total_sample_size - int(group_sizes.sum())
    group_sizes = group_sizes.tolist()

    # Build groups array
    groups = []
    for i in range(num_groups):
        groups.append(
            {
                "group_id": f"group_{i + 1}",
                "label": f"Group {i + 1}",
                "sample_size": group_sizes[i],
            }
        )

    # Generate rows
    rows = []
    for var_idx in range(num_variables):
        is_continuous = rng.random() < 0.5

        if is_continuous:
            row = _generate_continuous_row(rng, var_idx, groups)
        else:
            row = _generate_categorical_row(rng, var_idx, groups)

        rows.append(row)

    json_data = {
        "title": "Synthetic test data",
        "table1_exists": True,
        "groups": groups,
        "rows": rows,
    }

    validate_json(json_data)
    return json_data


def _generate_continuous_row(rng, var_idx, groups):
    """Generate a continuous variable row with mean/SD."""
    pop_mean = rng.uniform(10, 100)
    pop_sd = rng.uniform(1, 20)

    values = []
    for g in groups:
        n = g["sample_size"]
        samples = rng.normal(pop_mean, pop_sd, size=n)
        mean = round(float(np.mean(samples)), 1)
        sd = round(float(np.std(samples, ddof=1)), 1)
        # Ensure SD is not zero (would cause division by zero in z-score computation)
        if sd == 0.0:
            sd = 0.1

        values.append(
            {
                "group_id": g["group_id"],
                "original": f"{mean} ({sd})",
                "mean": mean,
                "median": None,
                "count": None,
                "IQR_lower": None,
                "IQR_upper": None,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": sd,
                "pvalue": None,
            }
        )

    return {
        "variable": f"variable{var_idx + 1}",
        "variable_type": "Continuous",
        "level": None,
        "values": values,
    }


def _generate_categorical_row(rng, var_idx, groups):
    """Generate a categorical variable row with counts."""
    base_rate = rng.uniform(0.1, 0.9)

    values = []
    for g in groups:
        n = g["sample_size"]
        count = int(rng.binomial(n, base_rate))

        pct = round(count / n * 100, 1)
        values.append(
            {
                "group_id": g["group_id"],
                "original": f"{count} ({pct}%)",
                "mean": None,
                "median": None,
                "count": count,
                "IQR_lower": None,
                "IQR_upper": None,
                "95CI_lower": None,
                "95CI_upper": None,
                "sd": None,
                "pvalue": None,
            }
        )

    return {
        "variable": f"variable{var_idx + 1}",
        "variable_type": "Categorical",
        "level": "Overall",
        "values": values,
    }
