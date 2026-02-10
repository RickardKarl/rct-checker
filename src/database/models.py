import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Enum, Integer, String, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


def _utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class ExtractionStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"


class Extraction(Base):
    __tablename__ = "extractions"
    __table_args__ = (UniqueConstraint("pdf_source", "model", name="uq_pdf_source_model"),)

    id = Column(Integer, primary_key=True)
    pdf_source = Column(Text, index=True, nullable=False)
    model = Column(String, nullable=False)  # Model used for extraction
    status = Column(Enum(ExtractionStatus), nullable=False)
    extracted_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    table1_json = Column(JSON)  # Extracted JSON data (null if failed)
    error_msg = Column(Text)  # Error message if failed
