# Constants for variable types
VARIABLE_TYPE_CONTINUOUS = "Continuous"
VARIABLE_TYPE_CATEGORICAL = "Categorical"


class ValidationError(Exception):
    pass


def validate_json(json_data):
    """
    Validate extracted JSON data from research paper tables.

    Expected schema:
    {
        "groups": [{"group_id": str, "label": str, "sample_size": int}, ...],
        "rows": [{
            "variable": str,
            "variable_type": "Continuous"|"Categorical",
            "level": str|null,
            "values": [{"group_id": str, "mean": float|null, "median": float|null,
                        "count": int|null, "IQR_lower": float|null, "IQR_upper": float|null,
                        "95CI_lower": float|null, "95CI_upper": float|null,
                        "sd": float|null, "original": str}, ...]
        }, ...]
    }

    Args:
        json_data: Dictionary containing extracted table data

    Returns:
        None on successful validation or if table1_exists is False

    Raises:
        ValidationError: If validation fails, with all error messages concatenated
    """
    errors = []

    # ─────────────────────────────
    # Top-level object
    # ─────────────────────────────
    if not isinstance(json_data, dict):
        raise ValidationError("Payload must be a JSON object.")

    table1_exists = json_data.get("table1_exists")
    groups_raw = json_data.get("groups")
    rows = json_data.get("rows")

    if not table1_exists:
        return None

    # ─────────────────────────────
    # Groups
    # ─────────────────────────────
    groups = set()

    if groups_raw is None:
        errors.append("Missing groups array.")
        groups_raw = []
    elif not isinstance(groups_raw, list) or not groups_raw:
        errors.append("Groups must be a non-empty array.")
        groups_raw = []

    seen_groups = set()
    for i, g in enumerate(groups_raw):
        if not isinstance(g, dict):
            errors.append(f"group_{i}: must be an object.")
            continue

        gid = g.get("group_id")
        label = g.get("label")
        sample_size = g.get("sample_size")

        if not isinstance(gid, str):
            errors.append(f"group_{i}: missing or invalid group_id.")
        elif gid in seen_groups:
            errors.append(f"group_{i}: duplicate group_id '{gid}'.")
        else:
            seen_groups.add(gid)

        if not isinstance(label, str) or not label.strip():
            errors.append(f"group_{i}: missing or invalid label.")

        if not isinstance(sample_size, int):
            errors.append(f"group_{i}: sample_size must be an integer.")
        elif sample_size < 0:
            errors.append(f"group_{i}: sample_size must be non-negative.")

    groups = seen_groups

    # ─────────────────────────────
    # Rows
    # ─────────────────────────────
    if rows is None:
        errors.append("Missing rows array.")
        rows = []
    elif not isinstance(rows, list) or not rows:
        errors.append("Rows must be a non-empty array.")
        rows = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {i}: must be an object.")
            continue

        variable = row.get("variable")
        var_type = row.get("variable_type")
        level = row.get("level")
        values = row.get("values")

        row_string = f"(variable={variable}, variable_type={var_type}, level={level})"

        if not isinstance(variable, str) or not variable.strip():
            errors.append(f"row {row_string}: missing or invalid variable name.")

        if var_type not in {VARIABLE_TYPE_CONTINUOUS, VARIABLE_TYPE_CATEGORICAL}:
            errors.append(f"row {row_string}: invalid variable_type '{var_type}'.")

        if level is not None:
            if var_type == VARIABLE_TYPE_CONTINUOUS:
                errors.append(f"row {row_string}: continuous variable must have level = null.")
            if var_type == VARIABLE_TYPE_CATEGORICAL and (
                not isinstance(level, str) or not level.strip()
            ):
                errors.append(f"row {row_string}: missing or invalid level.")

        if not isinstance(values, list) or not values:
            errors.append(f"row {row_string}: values must be a non-empty array.")
            continue

        # Check for rows where all numeric values are null across all groups
        # These are likely header rows (e.g., "Stroke subtype, n (%)") that shouldn't be extracted
        all_values_null = True
        for v in values:
            if isinstance(v, dict):
                has_numeric = (
                    v.get("mean") is not None
                    or v.get("median") is not None
                    or v.get("count") is not None
                    or v.get("sd") is not None
                    or v.get("IQR_lower") is not None
                    or v.get("IQR_upper") is not None
                    or v.get("95CI_lower") is not None
                    or v.get("95CI_upper") is not None
                )
                if has_numeric:
                    all_values_null = False
                    break

        if all_values_null:
            errors.append(
                f"row {row_string}: all numeric values are null for all groups. "
                "This may be a header row (e.g., 'Stroke subtype, n (%)') rather than a data row. "
                "Please remove this row or extract the actual data from its sub-rows."
            )
            continue

        # Row-level validation for continuous variables: check once per row (not per group)
        if var_type == VARIABLE_TYPE_CONTINUOUS and values:
            # Use first value to check row-level requirements (these should be consistent across groups)
            first_val = values[0] if isinstance(values[0], dict) else {}
            first_mean = first_val.get("mean")
            first_median = first_val.get("median")

            if first_mean is None and first_median is None:
                errors.append(f"row {row_string}: missing mean or median.")
            elif first_mean is not None:
                if first_val.get("sd") is None and (
                    first_val.get("95CI_lower") is None or first_val.get("95CI_upper") is None
                ):
                    errors.append(
                        f"row {row_string}: missing sd or 95CI values when mean is provided."
                    )
            elif first_median is not None:
                if first_val.get("IQR_lower") is None or first_val.get("IQR_upper") is None:
                    errors.append(f"row {row_string}: missing IQR values when median is provided.")

            # Check that IQR and 95CI are not provided at the same time
            has_iqr = (
                first_val.get("IQR_lower") is not None or first_val.get("IQR_upper") is not None
            )
            has_ci = (
                first_val.get("95CI_lower") is not None or first_val.get("95CI_upper") is not None
            )
            if has_iqr and has_ci:
                errors.append(
                    f"row {row_string}: cannot have both IQR and 95CI values. "
                    "Use IQR with median or 95CI with mean."
                )

            # Check that IQR is only used with median, not mean
            if has_iqr and first_mean is not None:
                errors.append(
                    f"row {row_string}: IQR values should only be used with median, not mean."
                )

            # Check that 95CI is only used with mean, not median
            if has_ci and first_median is not None:
                errors.append(
                    f"row {row_string}: 95CI values should only be used with mean, not median."
                )

        seen = set()
        for v in values:
            if not isinstance(v, dict):
                errors.append(f"row {row_string}: value entry must be an object.")
                continue

            gid = v.get("group_id")
            mean = v.get("mean")
            median = v.get("median")
            count = v.get("count")
            iqr_lower = v.get("IQR_lower")
            iqr_upper = v.get("IQR_upper")
            ci_lower = v.get("95CI_lower")
            ci_upper = v.get("95CI_upper")
            sd = v.get("sd")
            pvalue = v.get("pvalue")
            original = v.get("original")

            if not isinstance(gid, str):
                errors.append(f"row {row_string}: value missing or invalid group_id.")
            else:
                if gid not in groups:
                    errors.append(f"row {row_string}: unknown group_id '{gid}'.")
                if gid in seen:
                    errors.append(f"row {row_string}: duplicate group_id '{gid}'.")
                seen.add(gid)

            if not isinstance(original, str) or not original.strip():
                errors.append(f"row {row_string}: missing original value.")

            # ── Type-specific rules ──
            if var_type == VARIABLE_TYPE_CATEGORICAL:
                if mean is not None:
                    errors.append(f"row {row_string}: categorical variable should not have mean.")
                if median is not None:
                    errors.append(f"row {row_string}: categorical variable should not have median.")
                if iqr_lower is not None:
                    errors.append(
                        f"row {row_string}: categorical variable should not have IQR_lower."
                    )
                if iqr_upper is not None:
                    errors.append(
                        f"row {row_string}: categorical variable should not have IQR_upper."
                    )
                if ci_lower is not None:
                    errors.append(
                        f"row {row_string}: categorical variable should not have 95CI_lower."
                    )
                if ci_upper is not None:
                    errors.append(
                        f"row {row_string}: categorical variable should not have 95CI_upper."
                    )
                if sd is not None:
                    errors.append(f"row {row_string}: categorical variable should not have sd.")
                if count is None:
                    errors.append(f"row {row_string}: categorical variable must have count.")
                elif not isinstance(count, int):
                    errors.append(f"row {row_string}: count must be numerical integer.")
                elif count < 0:
                    errors.append(f"row {row_string}: count must be non-negative.")

            if var_type == VARIABLE_TYPE_CONTINUOUS:
                # Validate data types only (presence checks done at row level)
                if mean is not None and not isinstance(mean, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': mean must be numerical if provided."
                    )
                if median is not None and not isinstance(median, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': median must be numerical if provided."
                    )

                # Validate individual fields when present
                if iqr_lower is not None and not isinstance(iqr_lower, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': IQR_lower must be numerical float."
                    )
                if iqr_upper is not None and not isinstance(iqr_upper, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': IQR_upper must be numerical float."
                    )
                if ci_lower is not None and not isinstance(ci_lower, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': 95CI_lower must be numerical float."
                    )
                if ci_upper is not None and not isinstance(ci_upper, int | float):
                    errors.append(
                        f"row {row_string}, group '{gid}': 95CI_upper must be numerical float."
                    )
                if sd is not None and not isinstance(sd, int | float):
                    errors.append(f"row {row_string}, group '{gid}': sd must be numerical float.")

                # Validate sd is non-negative
                if sd is not None and isinstance(sd, int | float) and sd < 0:
                    errors.append(f"row {row_string}, group '{gid}': sd must be non-negative.")

                # Validate IQR range (lower <= upper)
                if (
                    iqr_lower is not None
                    and iqr_upper is not None
                    and isinstance(iqr_lower, int | float)
                    and isinstance(iqr_upper, int | float)
                    and iqr_lower > iqr_upper
                ):
                    errors.append(
                        f"row {row_string}, group '{gid}': IQR_lower must be <= IQR_upper."
                    )

                # Validate 95CI range (lower <= upper)
                if (
                    ci_lower is not None
                    and ci_upper is not None
                    and isinstance(ci_lower, int | float)
                    and isinstance(ci_upper, int | float)
                    and ci_lower > ci_upper
                ):
                    errors.append(
                        f"row {row_string}, group '{gid}': 95CI_lower must be <= 95CI_upper."
                    )

                # Validate median is within IQR range
                if (
                    median is not None
                    and iqr_lower is not None
                    and iqr_upper is not None
                    and isinstance(median, int | float)
                    and isinstance(iqr_lower, int | float)
                    and isinstance(iqr_upper, int | float)
                    and iqr_lower <= iqr_upper
                ):
                    if median < iqr_lower or median > iqr_upper:
                        errors.append(
                            f"row {row_string}, group '{gid}': median must be within IQR range."
                        )

                # Validate mean is within 95CI range
                if (
                    mean is not None
                    and ci_lower is not None
                    and ci_upper is not None
                    and isinstance(mean, int | float)
                    and isinstance(ci_lower, int | float)
                    and isinstance(ci_upper, int | float)
                    and ci_lower <= ci_upper
                ):
                    if mean < ci_lower or mean > ci_upper:
                        errors.append(
                            f"row {row_string}, group '{gid}': mean must be within 95CI range."
                        )

            # ── pvalue validation (applies to both variable types) ──
            if pvalue is not None:
                if not isinstance(pvalue, int | float):
                    errors.append(f"row {row_string}, group '{gid}': pvalue must be numerical.")
                elif pvalue < 0 or pvalue > 1:
                    errors.append(
                        f"row {row_string}, group '{gid}': pvalue must be between 0 and 1."
                    )

        if groups and seen != groups:
            errors.append(f"row {row_string}: must contain values for all groups.")

    if errors:
        raise ValidationError("\n".join(errors))
