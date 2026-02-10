"""Tests for report module."""

from pathlib import Path

from src.statistical_analysis.report import ReportCollector, generate_markdown_report


def test_generate_markdown_report_creates_valid_report(tmp_path):
    """Test that generate_markdown_report creates a valid Markdown file."""
    collector = ReportCollector()
    collector.add_result(
        paper_id="test-001",
        source="test.pdf",
        title="Test Paper",
        test_output={"fisher_method-combined": {"p_value": 0.03}},
    )

    report_path = tmp_path / "report.md"
    result = generate_markdown_report(collector, str(report_path))

    assert Path(result).exists()
    content = report_path.read_text()
    assert "# Data Analysis Report" in content
    assert "test-001" in content
