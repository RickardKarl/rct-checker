from datetime import datetime, timezone

from src.database.models import Base, Extraction, ExtractionStatus
from src.database.session import SessionLocal, engine


def init_db() -> None:
    """Initialize the database, creating tables if they don't exist."""
    Base.metadata.create_all(engine)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction operations
# ─────────────────────────────────────────────────────────────────────────────


def get_extraction_by_id(extraction_id: int) -> Extraction | None:
    """Look up an extraction by its database ID."""
    with SessionLocal() as session:
        return session.query(Extraction).filter(Extraction.id == extraction_id).first()


def get_extraction_by_source(pdf_source: str, model: str) -> Extraction | None:
    """
    Look up an extraction by PDF source path/URL and model.

    Args:
        pdf_source: File path or URL to PDF
        model: Model name used for extraction

    Returns:
        Extraction record if found, None otherwise
    """
    with SessionLocal() as session:
        return (
            session.query(Extraction)
            .filter(Extraction.pdf_source == pdf_source, Extraction.model == model)
            .first()
        )


def get_all_extractions(
    status: ExtractionStatus | None = None,
    model: str | None = None,
) -> list[Extraction]:
    """
    Get all extractions, optionally filtered by status and/or model.

    Args:
        status: Filter by extraction status (SUCCESS or FAILED)
        model: Filter by model name

    Returns:
        List of Extraction records
    """
    with SessionLocal() as session:
        query = session.query(Extraction)
        if status is not None:
            query = query.filter(Extraction.status == status)
        if model is not None:
            query = query.filter(Extraction.model == model)
        return query.order_by(Extraction.extracted_at.desc()).all()


def add_extraction(
    pdf_source: str,
    model: str,
    status: ExtractionStatus,
    table1_json: dict | None = None,
    error_msg: str | None = None,
) -> Extraction:
    """
    Add or update an extraction record in the database.

    Args:
        pdf_source: File path or URL to PDF
        model: Model name used for extraction
        status: ExtractionStatus.SUCCESS or ExtractionStatus.FAILED
        table1_json: Extracted JSON data (if successful)
        error_msg: Error message (if failed)

    Returns:
        The created or updated Extraction record
    """
    with SessionLocal() as session:
        existing = (
            session.query(Extraction)
            .filter(Extraction.pdf_source == pdf_source, Extraction.model == model)
            .first()
        )

        if existing:
            existing.status = status
            existing.extracted_at = datetime.now(timezone.utc)
            existing.table1_json = table1_json
            existing.error_msg = error_msg
            session.commit()
            session.refresh(existing)
            return existing
        else:
            extraction = Extraction(
                pdf_source=pdf_source,
                model=model,
                status=status,
                extracted_at=datetime.now(timezone.utc),
                table1_json=table1_json,
                error_msg=error_msg,
            )
            session.add(extraction)
            session.commit()
            session.refresh(extraction)
            return extraction
