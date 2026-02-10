import logging
import re
import tempfile
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import fitz
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# Minimum thresholds for PDF quality checks
MIN_EXTRACTABLE_CHARS = 100
MIN_PAGES = 1
TABLE_PATTERNS = [
    r"\bTable\s*1\b",
    r"\bTABLE\s*1\b",
    r"\bBaseline\s+[Cc]haracteristics\b",
    r"\bDemographic\s+[Cc]haracteristics\b",
    r"\bPatient\s+[Cc]haracteristics\b",
]


@dataclass
class PDFQualityResult:
    """Result of PDF quality validation."""

    is_valid: bool
    page_count: int
    char_count: int
    has_table_indicators: bool
    errors: list[str]
    warnings: list[str]


def validate_pdf_quality(pdf_path: str) -> PDFQualityResult:
    """
    Validate PDF quality before attempting LLM extraction.

    Checks if the PDF:
    - Can be opened and read
    - Contains extractable text (not just scanned images)
    - Has indicators of Table 1 or baseline characteristics

    Parameters
    ----------
    pdf_path : str
        Path to PDF file or URL.

    Returns
    -------
    PDFQualityResult
        Validation result with details about the PDF quality.

    Examples
    --------
    >>> result = validate_pdf_quality("paper.pdf")
    >>> if not result.is_valid:
    ...     print(f"PDF validation failed: {result.errors}")
    >>> if result.warnings:
    ...     print(f"Warnings: {result.warnings}")
    """
    errors = []
    warnings = []

    # Handle URL input
    if is_url(pdf_path):
        try:
            pdf_content = download_pdf(pdf_path)
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                with fitz.open(tmp_file.name) as doc:
                    return _validate_pdf_document(doc, errors, warnings)
        except RuntimeError as e:
            errors.append(f"Failed to download PDF: {e}")
            return PDFQualityResult(
                is_valid=False,
                page_count=0,
                char_count=0,
                has_table_indicators=False,
                errors=errors,
                warnings=warnings,
            )
    else:
        try:
            with fitz.open(pdf_path) as doc:
                return _validate_pdf_document(doc, errors, warnings)
        except Exception as e:
            errors.append(f"Failed to open PDF: {e}")
            return PDFQualityResult(
                is_valid=False,
                page_count=0,
                char_count=0,
                has_table_indicators=False,
                errors=errors,
                warnings=warnings,
            )


def _validate_pdf_document(
    doc: fitz.Document, errors: list[str], warnings: list[str]
) -> PDFQualityResult:
    """Validate an opened PDF document."""
    page_count = len(doc)
    if page_count < MIN_PAGES:
        errors.append(f"PDF has {page_count} pages, minimum required is {MIN_PAGES}")

    # Extract all text and count characters
    all_text = ""
    for page in doc:
        all_text += page.get_text("text") or ""

    char_count = len(all_text.strip())
    if char_count < MIN_EXTRACTABLE_CHARS:
        errors.append(
            f"PDF contains only {char_count} extractable characters "
            f"(minimum: {MIN_EXTRACTABLE_CHARS}). "
            "This may be a scanned PDF without OCR."
        )

    # Check for table indicators
    has_table_indicators = False
    for pattern in TABLE_PATTERNS:
        if re.search(pattern, all_text, re.IGNORECASE):
            has_table_indicators = True
            break

    if not has_table_indicators:
        warnings.append(
            "No Table 1 or baseline characteristics indicators found. "
            "Extraction may not find relevant data."
        )

    is_valid = len(errors) == 0

    return PDFQualityResult(
        is_valid=is_valid,
        page_count=page_count,
        char_count=char_count,
        has_table_indicators=has_table_indicators,
        errors=errors,
        warnings=warnings,
    )


def is_url(path: str) -> bool:
    """Check if a string is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except ValueError:
        return False


def download_pdf(url: str, timeout: int = 30) -> bytes:
    """
    Download a PDF from a URL.

    Parameters
    ----------
    url : str
        URL to download from.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    bytes
        PDF content as bytes.

    Raises
    ------
    RuntimeError
        If download fails or content is not a PDF.
    """
    logger.info(f"Downloading PDF from {url}")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download PDF from {url}: {e}") from e

    content_type = response.headers.get("Content-Type", "")
    if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
        logger.warning(
            f"Content-Type '{content_type}' may not be a PDF, attempting to process anyway"
        )

    return response.content


def _extract_pages_from_doc(doc: fitz.Document) -> list[dict[str, Any]]:
    """Extract text from all pages of a PDF document."""
    return [{"page": i, "text": page.get_text("text") or ""} for i, page in enumerate(doc)]


def extract_table_text(pdf_path: str, expand_pages: int = 1) -> str:
    """
    Extract text focused on Table 1 using page-based chunking.

    Parameters
    ----------
    pdf_path : str
        Path to PDF file or URL to a PDF.
    expand_pages : int
        Number of pages to include before and after detected Table 1 pages.

    Returns
    -------
    str
        Extracted text.
    """
    # Handle URL input by downloading to a temporary file
    if is_url(pdf_path):
        pdf_content = download_pdf(pdf_path)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
            tmp_file.write(pdf_content)
            tmp_file.flush()
            with fitz.open(tmp_file.name) as doc:
                pages_text = _extract_pages_from_doc(doc)
    else:
        with fitz.open(pdf_path) as doc:
            pages_text = _extract_pages_from_doc(doc)

    # Identify pages likely containing Table 1
    table_pages = set()
    for page in pages_text:
        for pattern in TABLE_PATTERNS:
            if re.search(pattern, page["text"], re.IGNORECASE):
                table_pages.add(page["page"])
                break

    # Fallback: return full text if Table 1 not found
    if not table_pages:
        logger.warning("Did not find 'Table 1' in PDF; returning full text as fallback.")
        return "\n\n".join(p["text"] for p in pages_text)

    selected_pages = {0}
    for p in table_pages:
        for offset in range(-expand_pages, expand_pages + 1):
            idx = p + offset
            if 0 <= idx < len(pages_text):
                selected_pages.add(idx)

    selected_pages = sorted(selected_pages)

    return "\n\n".join(pages_text[p]["text"] for p in selected_pages)


def _to_numeric(value: Any, as_int: bool = False) -> float | int | None:
    """Convert a value to numeric type, returning None if value is None."""
    if value is None:
        return None
    if as_int:
        return int(value)
    return float(value)


def to_csv_wide(json_data: dict[str, Any], out_path: str | None = None) -> pd.DataFrame | None:
    """
    Convert extracted JSON data to wide-format CSV.

    Parameters
    ----------
    json_data : dict
        Extracted Table 1 data.
    out_path : str, optional
        Path to write CSV. If None, returns DataFrame.

    Returns
    -------
    pd.DataFrame or None
        DataFrame if out_path is None, otherwise writes to file.
    """
    rows = []

    for row in json_data["rows"]:
        level = row["level"]
        level_str = f" ({level})" if row["variable_type"] == "Categorical" else ""
        record: dict[str, Any] = {
            "Variable": row["variable"] + level_str,
            "Variable type": row["variable_type"],
        }

        for v in row["values"]:
            gid = v["group_id"]
            record[f"{gid} (original)"] = v["original"]
            record[f"{gid} (mean)"] = _to_numeric(v.get("mean"))
            record[f"{gid} (median)"] = _to_numeric(v.get("median"))
            record[f"{gid} (count)"] = _to_numeric(v.get("count"), as_int=True)
            record[f"{gid} (IQR_lower)"] = _to_numeric(v.get("IQR_lower"))
            record[f"{gid} (IQR_upper)"] = _to_numeric(v.get("IQR_upper"))
            record[f"{gid} (95CI_lower)"] = _to_numeric(v.get("95CI_lower"))
            record[f"{gid} (95CI_upper)"] = _to_numeric(v.get("95CI_upper"))
            record[f"{gid} (sd)"] = _to_numeric(v.get("sd"))
            record[f"{gid} (pvalue)"] = _to_numeric(v.get("pvalue"))

        rows.append(record)

    df = pd.DataFrame(rows)
    if out_path:
        df.to_csv(out_path, index=False)
        return None
    return df
