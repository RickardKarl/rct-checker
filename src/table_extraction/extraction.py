"""Table extraction pipeline using LLM backends."""

import logging
from pathlib import Path
from typing import Any

from src.table_extraction.config.schema import PaperTable1
from src.table_extraction.llm import (
    ExtractionRequest,
    LLMConfig,
    Message,
    StructuredOutputConfig,
    create_backend,
)
from src.table_extraction.utils import extract_table_text, validate_pdf_quality
from src.table_extraction.validate_output import validate_json

logger = logging.getLogger(__name__)


class PDFQualityError(Exception):
    """Raised when PDF fails quality validation."""

    pass


# Module directory for resolving relative paths
_MODULE_DIR = Path(__file__).parent
_PROMPTS_DIR = _MODULE_DIR / "config" / "prompts"

# Lazy-loaded module state
_extraction_prompt: str | None = None
_repair_prompt_template: str | None = None
_schema: dict | None = None


def _load_prompts() -> tuple[str, str]:
    """Load extraction and repair prompts from files."""
    global _extraction_prompt, _repair_prompt_template
    if _extraction_prompt is None:
        with open(_PROMPTS_DIR / "extraction_prompt.txt") as f:
            _extraction_prompt = f.read()
    if _repair_prompt_template is None:
        with open(_PROMPTS_DIR / "repair_prompt.txt") as f:
            _repair_prompt_template = f.read()
    return _extraction_prompt, _repair_prompt_template


def _get_schema() -> dict:
    """Get the Pydantic JSON schema (lazy initialization)."""
    global _schema
    if _schema is None:
        _schema = PaperTable1.model_json_schema()
    return _schema


def extraction_pipeline(
    pdf_path: str,
    config: LLMConfig | None = None,
) -> dict[str, Any]:
    """
    PDF -> text -> LLM -> validation/repair -> JSON data.

    Parameters
    ----------
    pdf_path : str
        Path to PDF file or URL to a PDF.
    config : LLMConfig, optional
        Configuration for creating a backend.

    Returns
    -------
    dict
        Validated JSON data containing extracted Table 1 information.

    Raises
    ------
    PDFQualityError
        If PDF fails quality validation (unreadable, no extractable text).
    RuntimeError
        If extraction fails after max attempts or LLM returns incomplete response.
    """
    # Validate PDF quality before expensive LLM calls
    quality = validate_pdf_quality(pdf_path)
    if not quality.is_valid:
        raise PDFQualityError(f"PDF failed quality validation: {'; '.join(quality.errors)}")
    for warning in quality.warnings:
        logger.warning(f"PDF quality warning: {warning}")

    extraction_prompt, repair_prompt_template = _load_prompts()
    schema = _get_schema()

    # Resolve config defaults
    if config is None:
        from src.table_extraction.llm import get_default_openai_config

        config = get_default_openai_config()

    backend = create_backend(config)

    logger.info(f"Starting Table 1 extraction for {pdf_path} using {backend.name}/{backend.model}")

    # Extract text from PDF
    text = extract_table_text(pdf_path)
    prompt = extraction_prompt + "\n\n" + text

    # Build initial extraction request
    request = ExtractionRequest(
        messages=[
            Message(role="system", content="You output JSON only."),
            Message(role="user", content=prompt),
        ],
        output_config=StructuredOutputConfig(
            schema=schema,
            schema_name="table1_extraction",
            strict=True,
        ),
        max_tokens=config.max_output_tokens,
    )

    # Use the backend's extract_with_repair method
    result = backend.extract_with_repair(
        initial_request=request,
        repair_prompt_template=repair_prompt_template,
        validate_fn=validate_json,
        max_attempts=config.max_attempts,
    )

    logger.info("Extraction completed successfully")
    return result
