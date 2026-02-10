"""
Report generation for data analysis results.

This module provides functionality to collect analysis results from multiple
papers and generate consolidated Markdown reports.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PaperResult:
    """Results from analyzing a single paper."""

    paper_id: str
    source: str
    title: str = None
    combined_p_value: float = None
    chi_squared_p_value: float = None
    chi_squared_n_zscores: int = 0
    fisher_tests_count: int = 0
    fisher_p_values: list = field(default_factory=list)
    plot_path: str = None
    error: str = None


class ReportCollector:
    """Collects analysis results from multiple papers for report generation."""

    def __init__(self):
        self.results: list[PaperResult] = []

    def add_result(
        self,
        paper_id: str,
        source: str,
        title: str = None,
        test_output: dict = None,
        plot_path: str = None,
        error: str = None,
    ):
        """
        Add analysis result for a paper.

        Parameters
        ----------
        paper_id : str
            Unique identifier for the paper.
        source : str
            Source of the paper (e.g., PDF filename or URL).
        title : str, optional
            Title of the paper.
        test_output : dict, optional
            Output from run_test_pipeline().
        plot_path : str, optional
            Path to saved plot image.
        error : str, optional
            Error message if analysis failed.
        """
        result = PaperResult(paper_id=paper_id, source=source, title=title, error=error)

        if test_output and not error:
            # Extract combined p-value
            if "fisher_method-combined" in test_output:
                result.combined_p_value = test_output["fisher_method-combined"]["p_value"]

            # Extract chi-squared results
            if "cont_chi_squared_variance" in test_output:
                chi_sq = test_output["cont_chi_squared_variance"]
                result.chi_squared_p_value = chi_sq["p_value"]
                result.chi_squared_n_zscores = len(chi_sq["zscores"])

            # Extract Fisher's test results
            fisher_keys = [k for k in test_output.keys() if k.startswith("fisher_test")]
            result.fisher_tests_count = len(fisher_keys)
            result.fisher_p_values = [test_output[k]["p_value"] for k in fisher_keys]

            result.plot_path = plot_path

        self.results.append(result)

    def get_summary_stats(self) -> dict:
        """
        Calculate summary statistics across all papers.

        Returns
        -------
        dict
            Summary statistics including counts and flagged papers.
        """
        successful = [r for r in self.results if r.error is None]
        flagged = [
            r for r in successful if r.combined_p_value is not None and r.combined_p_value < 0.05
        ]

        return {
            "total_papers": len(self.results),
            "successful_analyses": len(successful),
            "failed_analyses": len(self.results) - len(successful),
            "flagged_papers": len(flagged),
            "flagged_rate": len(flagged) / len(successful) if successful else 0,
        }


def generate_markdown_report(collector: ReportCollector, output_path: str) -> str:
    """
    Generate a Markdown report from collected analysis results.

    Parameters
    ----------
    collector : ReportCollector
        Collector containing analysis results.
    output_path : str
        Path to save the Markdown report.

    Returns
    -------
    str
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = collector.get_summary_stats()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []

    # Header
    lines.append("# Data Analysis Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Papers analyzed:** {stats['total_papers']}")
    lines.append(f"- **Successful analyses:** {stats['successful_analyses']}")
    lines.append(f"- **Failed analyses:** {stats['failed_analyses']}")
    lines.append(
        f"- **Papers flagged (p < 0.05):** {stats['flagged_papers']} "
        f"({stats['flagged_rate']:.1%})"
    )
    lines.append("")

    # Results table
    lines.append("## Results Overview")
    lines.append("")
    lines.append("| ID | Title | Combined p-value | Chi-sq p-value | Fisher tests | Status |")
    lines.append("|:---|:------|:-----------------|:---------------|:-------------|:-------|")

    for r in collector.results:
        if r.error:
            status = "Error"
            combined_p = "-"
            chi_sq_p = "-"
            fisher_n = "-"
        else:
            combined_p = f"{r.combined_p_value:.4f}" if r.combined_p_value else "-"
            chi_sq_p = f"{r.chi_squared_p_value:.4f}" if r.chi_squared_p_value else "-"
            fisher_n = str(r.fisher_tests_count) if r.fisher_tests_count else "-"
            status = "Flagged" if r.combined_p_value and r.combined_p_value < 0.05 else "OK"

        # Use title if available, otherwise truncate source
        title_display = r.title or r.source
        if len(title_display) > 40:
            title_display = title_display[:37] + "..."

        lines.append(
            f"| {r.paper_id} | {title_display} | {combined_p} | {chi_sq_p} | {fisher_n} | {status} |"
        )

    lines.append("")

    # Detailed results section
    lines.append("## Detailed Results")
    lines.append("")

    for r in collector.results:
        lines.append(f"### Paper {r.paper_id}")
        lines.append("")
        if r.title:
            lines.append(f"**Title:** {r.title}")
            lines.append("")
        lines.append(f"**Source:** {r.source}")
        lines.append("")

        if r.error:
            lines.append(f"**Error:** {r.error}")
            lines.append("")
            continue

        if r.combined_p_value is not None:
            lines.append(f"**Combined p-value:** {r.combined_p_value:.4f}")
            lines.append("")

        # Chi-squared results
        if r.chi_squared_p_value is not None:
            lines.append("**Continuous Variables (Chi-squared variance test):**")
            lines.append(f"- p-value: {r.chi_squared_p_value:.4f}")
            lines.append(f"- Number of z-scores: {r.chi_squared_n_zscores}")
            lines.append("")

        # Fisher's test results
        if r.fisher_tests_count > 0:
            lines.append("**Categorical Variables (Fisher's exact test):**")
            lines.append(f"- Variables tested: {r.fisher_tests_count}")
            if r.fisher_p_values:
                min_p = min(r.fisher_p_values)
                max_p = max(r.fisher_p_values)
                lines.append(f"- p-value range: [{min_p:.4f}, {max_p:.4f}]")
            lines.append("")

        # Plot
        if r.plot_path:
            # Use relative path from report location
            plot_rel_path = Path(r.plot_path).name
            lines.append(
                f'<img src="figures/{plot_rel_path}" alt="Analysis plots for paper {r.paper_id}" height="200">'
            )
            lines.append("")

    # Write report
    report_content = "\n".join(lines)
    output_path.write_text(report_content)

    logger.info(f"Report generated: {output_path}")
    return str(output_path)
