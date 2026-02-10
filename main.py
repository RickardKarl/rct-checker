import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.database import (
    add_extraction,
    get_all_extractions,
    get_extraction_by_id,
    get_extraction_by_source,
    init_db,
)
from src.database.models import ExtractionStatus
from src.statistical_analysis.pipeline import run_test_pipeline
from src.statistical_analysis.plotting import plot_test_output
from src.statistical_analysis.report import ReportCollector, generate_markdown_report
from src.table_extraction.extraction import extraction_pipeline
from src.table_extraction.llm import (
    get_default_huggingface_config,
    get_default_openai_config,
)


def configure_logging(log_level: str):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    return logging.getLogger(__name__)


def get_llm_config(backend: str):
    """Get LLM configuration for the specified backend."""
    if backend == "openai":
        return get_default_openai_config()
    elif backend == "huggingface":
        return get_default_huggingface_config()
    else:
        raise SystemExit(f"Unknown LLM backend: {backend}")


def _label_from_source(pdf_source: str) -> str:
    """Derive a display label from a PDF source path or URL."""
    if pdf_source.startswith(("http://", "https://")):
        return pdf_source.split("/")[-1] or pdf_source
    return pathlib.Path(pdf_source).name


def _resolve_pdf_sources(pdf_arg: str) -> list[str]:
    """Resolve --pdf argument into a list of PDF source strings.

    Accepts a single PDF file path, a directory of PDFs, or a URL.
    """
    # URL
    if pdf_arg.startswith(("http://", "https://")):
        return [pdf_arg]

    path = pathlib.Path(pdf_arg)

    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise SystemExit(f"Not a PDF file: {path}")
        return [str(path.resolve())]

    if path.is_dir():
        pdfs = sorted(path.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDF files found in directory: {path}")
        return [str(p.resolve()) for p in pdfs]

    raise SystemExit(f"Path does not exist: {path}")


def cmd_extract(args):
    """Run the table extraction pipeline on PDFs."""
    logger = configure_logging(args.log_level)
    init_db()

    llm_config = get_llm_config(args.llm_backend)
    sources = _resolve_pdf_sources(args.pdf)

    logger.info(f"Processing {len(sources)} PDF(s)")

    for pdf_source in sources:
        label = _label_from_source(pdf_source)
        logger.info(f"Extracting: {label}")

        existing = get_extraction_by_source(pdf_source, llm_config.model)
        if existing and existing.status == ExtractionStatus.SUCCESS and not args.force:
            logger.info("  Already extracted (use --force to re-extract)")
            continue

        try:
            json_output = extraction_pipeline(pdf_source, config=llm_config)
            add_extraction(
                pdf_source=pdf_source,
                model=llm_config.model,
                status=ExtractionStatus.SUCCESS,
                table1_json=json_output,
            )
            logger.info("  Success")
        except Exception as e:
            add_extraction(
                pdf_source=pdf_source,
                model=llm_config.model,
                status=ExtractionStatus.FAILED,
                error_msg=str(e),
            )
            logger.error(f"  Failed: {e}")


def cmd_analyze(args):
    """Run the data analysis pipeline."""
    logger = configure_logging(args.log_level)

    # Setup report collector if report is requested
    report_collector = ReportCollector() if args.report else None
    report_plots_enabled = args.report and args.report_plots

    # Setup report output paths
    if args.report:
        if args.report is True:
            # Default path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = pathlib.Path(f"reports/analysis_report_{timestamp}.md")
        else:
            report_path = pathlib.Path(args.report)

        # Create figures directory if report plots are enabled
        if report_plots_enabled:
            figures_dir = report_path.parent / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

    # Build list of items to analyze: [(item_id, source, title, data_dict), ...]
    items = []

    if args.json:
        # File-based analysis (debug mode)
        json_path = pathlib.Path(args.json)

        if json_path.is_file():
            if json_path.suffix != ".json":
                logger.error(f"File must be a JSON file: {json_path}")
                return
            json_files = [json_path]
        elif json_path.is_dir():
            json_files = sorted(json_path.glob("*.json"))
        else:
            logger.error(f"Path does not exist: {json_path}")
            return

        if not json_files:
            logger.warning(f"No JSON files found in {json_path}")
            return

        for idx, json_file in enumerate(json_files):
            with open(json_file) as f:
                data = json.load(f)
                title = data.get("title")
                items.append((str(idx + 1), json_file.name, title, data))
    else:
        # Database mode (default)
        init_db()

        if args.id:
            extraction = get_extraction_by_id(args.id)
            if not extraction:
                logger.error(f"No extraction found with ID: {args.id}")
                return
            if extraction.status != ExtractionStatus.SUCCESS:
                logger.error(f"Extraction {args.id} failed, no data to analyze")
                return
            extractions = [extraction]
        else:
            extractions = get_all_extractions(status=ExtractionStatus.SUCCESS)
            if not extractions:
                logger.warning("No successful extractions found in database")
                return

        for ext in extractions:
            title = _label_from_source(ext.pdf_source)
            items.append((str(ext.id), ext.pdf_source, title, ext.table1_json))

    logger.info(f"Found {len(items)} item(s) to analyze")

    # Determine if interactive plotting should be enabled
    plot_enabled = args.plot and len(items) == 1
    if args.plot and len(items) > 1:
        logger.warning(
            "Interactive plotting is only supported for a single item. Plotting will be disabled."
        )

    for paper_id, source, title, data in items:
        logger.info(f"Processing: {source}")
        output = None
        error_msg = None
        plot_path = None

        try:
            output = run_test_pipeline(
                data,
                skip_continuous_var=args.skip_cont,
                skip_categorical_var=args.skip_cat,
            )
            logger.debug(f"Successfully processed {source}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing {source}: {error_msg}")

        # Handle plotting
        if output and not error_msg:
            try:
                if report_plots_enabled:
                    plot_path = str(figures_dir / f"paper_{paper_id}_plots.png")
                    plot_test_output(output, save_path=plot_path)
                elif plot_enabled:
                    plot_test_output(output)
            except Exception as e:
                logger.error(f"Plotting failed: {str(e)}")

        # Collect results for report
        if report_collector:
            report_collector.add_result(
                paper_id=paper_id,
                source=source,
                title=title,
                test_output=output,
                plot_path=plot_path,
                error=error_msg,
            )

    # Generate report if requested
    if report_collector:
        generate_markdown_report(report_collector, str(report_path))
        logger.info(f"Report generated: {report_path}")


def cmd_list(args):
    """List all extractions in the database."""
    init_db()

    # Get extractions with optional status filter
    status_filter = None
    if args.status:
        status_filter = ExtractionStatus(args.status)

    extractions = get_all_extractions(status=status_filter)

    if not extractions:
        print("No extractions found in database.")
        return

    # Print header
    print(f"\n{'ID':<6} {'Status':<10} {'Model':<15} {'Extracted At':<20} {'Source'}")
    print("-" * 100)

    for ext in extractions:
        source = ext.pdf_source
        if len(source) > 55:
            source = "..." + source[-52:]
        extracted_at = ext.extracted_at.strftime("%Y-%m-%d %H:%M") if ext.extracted_at else "N/A"
        print(f"{ext.id:<6} {ext.status.value:<10} {ext.model:<15} {extracted_at:<20} {source}")

    print(f"\nTotal: {len(extractions)} extraction(s)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="RCT Checker - Scientific paper analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract Table 1 data from PDFs")
    extract_parser.add_argument(
        "--pdf",
        required=True,
        help="PDF file path, directory of PDFs, or URL",
    )
    extract_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if PDF was already processed",
    )
    extract_parser.add_argument(
        "--llm-backend",
        choices=["openai", "huggingface"],
        default="openai",
        help="LLM backend for extraction (default: openai)",
    )
    extract_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run statistical analysis on extracted data"
    )
    analyze_parser.add_argument(
        "--id",
        type=int,
        help="Analyze a specific extraction by database ID",
    )
    analyze_parser.add_argument(
        "--json",
        help="Path to JSON file or directory (debug mode). If not provided, uses database.",
    )
    analyze_parser.add_argument(
        "--skip-cont",
        action="store_true",
        help="Skip analysing continuous variables",
    )
    analyze_parser.add_argument(
        "--skip-cat",
        action="store_true",
        help="Skip analysing categorical variables",
    )
    analyze_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate z-score distribution plots (only works with a single extraction)",
    )
    analyze_parser.add_argument(
        "--report",
        nargs="?",
        const=True,
        default=False,
        metavar="PATH",
        help="Generate a Markdown report. Optionally specify output path (default: reports/analysis_report_<timestamp>.md)",
    )
    analyze_parser.add_argument(
        "--report-plots",
        action="store_true",
        help="Include plots in the report (requires --report)",
    )
    analyze_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )

    analyze_parser.set_defaults(func=cmd_analyze)

    # List command
    list_parser = subparsers.add_parser("list", help="List all extractions in the database")
    list_parser.add_argument(
        "--status",
        choices=["success", "failed"],
        help="Filter by extraction status",
    )
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
