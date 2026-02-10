"""Statistical analysis module for detecting anomalies in research data."""

from src.statistical_analysis.pipeline import run_test_pipeline
from src.statistical_analysis.report import (
    ReportCollector,
    generate_markdown_report,
)
from src.statistical_analysis.statistical_tests import chi_square_variance_test
from src.statistical_analysis.utils import (
    contingency_table_binary,
    process_categorical_variables,
    process_continuous_variables_mean,
)

__all__ = [
    # Main pipeline
    "run_test_pipeline",
    # Report generation
    "ReportCollector",
    "generate_markdown_report",
    # Statistical tests
    "chi_square_variance_test",
    # Utilities
    "contingency_table_binary",
    "process_categorical_variables",
    "process_continuous_variables_mean",
]
