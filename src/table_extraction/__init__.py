"""Table extraction module for extracting structured data from PDF tables."""

from src.table_extraction.config.schema import Group, PaperTable1, Row, ValueEntry
from src.table_extraction.extraction import PDFQualityError, extraction_pipeline
from src.table_extraction.utils import (
    PDFQualityResult,
    extract_table_text,
    is_url,
    to_csv_wide,
    validate_pdf_quality,
)
from src.table_extraction.validate_output import validate_json

__all__ = [
    # Main extraction pipeline
    "extraction_pipeline",
    "PDFQualityError",
    # Schema models
    "PaperTable1",
    "Group",
    "Row",
    "ValueEntry",
    # Utilities
    "extract_table_text",
    "is_url",
    "to_csv_wide",
    # PDF quality validation
    "PDFQualityResult",
    "validate_pdf_quality",
    # Output validation
    "validate_json",
]
