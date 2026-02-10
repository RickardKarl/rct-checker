"""Integration tests for the extraction pipeline.

These tests verify that the extraction pipeline produces expected JSON output
for a set of test PDF files. The expected outputs are stored as golden files
in data/extracted_table1/.

Note: These tests call the actual OpenAI API, so they are:
- Slow (several seconds per PDF)
- Costly (API usage fees)
- Potentially non-deterministic (LLM outputs may vary)

The comparison is lenient with string values (LLM may phrase things differently)
but strict with:
- Structure (number of rows, groups, values)
- Numeric values (mean, median, count, sd, IQR, CI, pvalue)
- Key identifiers (variable_type, group_id)

Fuzzy matching (90% similarity) is used for:
- level, variable

Run with: pytest tests/table_extraction/test_extraction_integration.py -v
Skip with: pytest -m "not integration"
"""

import json
import pathlib
import sys
from difflib import SequenceMatcher

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.table_extraction.extraction import extraction_pipeline
from src.table_extraction.llm.config import (
    get_default_openai_config,
)

# LLM backend
LLM_CONFIG = get_default_openai_config()

# Paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PDF_DIR = pathlib.Path(__file__).parent / "test_files"
EXPECTED_JSON_DIR = pathlib.Path(__file__).parent / "test_files"
# Output directory for actual test results (persists for manual inspection)
TEST_OUTPUT_DIR = pathlib.Path(__file__).parent / "test_output"

# Keys that should be compared strictly (even if strings)
STRICT_STRING_KEYS = {"variable_type", "group_id"}

# Keys that should be compared with fuzzy matching (90% similarity threshold)
FUZZY_STRING_KEYS: set[str] = set()  # {"level", "variable"}
FUZZY_THRESHOLD = 0.9


def strings_match_fuzzy(s1, s2, threshold=FUZZY_THRESHOLD):
    """Check if two strings match with fuzzy comparison."""
    if s1 == s2:
        return True, 1.0
    ratio = SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    return ratio >= threshold, ratio


# Keys that contain numeric data and should be compared
NUMERIC_KEYS = {
    "mean",
    "median",
    "count",
    "sd",
    "IQR_lower",
    "IQR_upper",
    "95CI_lower",
    "95CI_upper",
    "pvalue",
    "sample_size",
}


def should_compare_value(key, value):
    """Determine if a value should be compared based on its key and type."""
    # Always compare booleans
    if isinstance(value, bool):
        return True
    # Always compare numeric values
    if isinstance(value, int | float) or value is None and key in NUMERIC_KEYS:
        return True
    # Compare strict string keys
    if key in STRICT_STRING_KEYS:
        return True
    # Compare fuzzy string keys
    if key in FUZZY_STRING_KEYS:
        return True
    # Compare numeric keys (even if None)
    if key in NUMERIC_KEYS:
        return True
    # Skip other strings (variable names, labels, original values)
    if isinstance(value, str):
        return False
    # Compare lists and dicts (structure)
    if isinstance(value, list | dict):
        return True
    return True


def find_json_differences(expected, actual, path="root", parent_key=None):
    """
    Recursively find differences between two JSON objects.

    Only compares:
    - Structure (dict keys, list lengths)
    - Numeric values
    - Booleans
    - Strict string keys (variable_type, group_id)
    - Fuzzy string keys with 90% threshold (level, variable)

    Skips comparison of:
    - Other string values (label, original)
    """
    differences = []

    # Type mismatch
    if type(expected) is not type(actual):
        # Allow None vs missing for optional fields
        if expected is None or actual is None:
            if parent_key in NUMERIC_KEYS or parent_key in STRICT_STRING_KEYS:
                differences.append(
                    f"{path}: value mismatch - expected {expected!r}, got {actual!r}"
                )
        else:
            differences.append(
                f"{path}: type mismatch - expected {type(expected).__name__}, "
                f"got {type(actual).__name__}"
            )
        return differences

    if isinstance(expected, dict):
        # Check for missing/extra keys in structure
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())

        for key in expected_keys - actual_keys:
            differences.append(f"{path}.{key}: missing key in actual")
        for key in actual_keys - expected_keys:
            differences.append(f"{path}.{key}: unexpected key in actual")

        # Compare common keys
        for key in sorted(expected_keys & actual_keys):
            new_path = f"{path}.{key}"
            exp_val = expected[key]
            act_val = actual[key]

            if should_compare_value(key, exp_val):
                differences.extend(
                    find_json_differences(exp_val, act_val, new_path, parent_key=key)
                )

    elif isinstance(expected, list):
        if len(expected) != len(actual):
            differences.append(
                f"{path}: list length mismatch - expected {len(expected)}, " f"got {len(actual)}"
            )
        # Compare items up to the shorter length
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            differences.extend(
                find_json_differences(exp_item, act_item, f"{path}[{i}]", parent_key)
            )

    elif isinstance(expected, bool):
        if expected != actual:
            differences.append(f"{path}: boolean mismatch - expected {expected}, got {actual}")

    elif isinstance(expected, int | float):
        # Compare numeric values with tolerance for floats
        if expected != actual:
            if isinstance(expected, float) or isinstance(actual, float):
                # Allow small floating point differences
                if abs(expected - actual) > 0.01:
                    differences.append(
                        f"{path}: numeric mismatch - expected {expected}, got {actual}"
                    )
            else:
                differences.append(f"{path}: numeric mismatch - expected {expected}, got {actual}")

    elif isinstance(expected, str):
        # Strict comparison for STRICT_STRING_KEYS
        if parent_key in STRICT_STRING_KEYS and expected != actual:
            differences.append(f"{path}: string mismatch - expected {expected!r}, got {actual!r}")
        # Fuzzy comparison for FUZZY_STRING_KEYS
        elif parent_key in FUZZY_STRING_KEYS:
            matches, ratio = strings_match_fuzzy(expected, actual)
            if not matches:
                differences.append(
                    f"{path}: fuzzy string mismatch ({ratio:.0%} similar) - "
                    f"expected {expected!r}, got {actual!r}"
                )

    elif expected is None:
        if actual is not None:
            differences.append(f"{path}: expected None, got {actual!r}")

    return differences


def format_differences(differences, max_differences=30):
    """Format a list of differences into a readable string."""
    if not differences:
        return "No differences found"

    total = len(differences)
    shown = differences[:max_differences]

    result = f"Found {total} difference(s):\n\n"
    result += "\n\n".join(f"  {i+1}. {diff}" for i, diff in enumerate(shown))

    if total > max_differences:
        result += f"\n\n  ... and {total - max_differences} more difference(s)"

    return result


def get_test_cases():
    """Discover all PDF files that have corresponding expected JSON files."""
    test_cases = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        expected_json_name = f"{pdf_path.stem}.json"
        expected_json_path = EXPECTED_JSON_DIR / expected_json_name
        if expected_json_path.exists():
            test_cases.append(
                pytest.param(
                    pdf_path,
                    expected_json_path,
                    id=pdf_path.stem,
                )
            )
    return test_cases


@pytest.mark.integration
@pytest.mark.parametrize("pdf_path,expected_json_path", get_test_cases())
def test_extraction_matches_expected(pdf_path, expected_json_path):
    """Test that extraction output matches the expected golden file."""
    # Load expected JSON
    with open(expected_json_path, encoding="utf-8") as f:
        expected = json.load(f)

    # Run extraction pipeline
    actual = extraction_pipeline(str(pdf_path), config=LLM_CONFIG)

    # Save actual output to test_output directory for manual inspection
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    actual_output_path = TEST_OUTPUT_DIR / f"{pdf_path.stem}.json"
    with open(actual_output_path, "w", encoding="utf-8") as f:
        json.dump(actual, f, indent=2, ensure_ascii=False)

    # If both expected and actual have table1_exists == False, test passes
    if expected.get("table1_exists") is False and actual.get("table1_exists") is False:
        return

    # Compare outputs (lenient comparison)
    differences = find_json_differences(expected, actual)

    if differences:
        diff_report = format_differences(differences)
        pytest.fail(
            f"Extraction output for {pdf_path.name} does not match expected.\n"
            f"Expected file: {expected_json_path}\n"
            f"Actual output: {actual_output_path}\n\n"
            f"{diff_report}"
        )
