"""Unit tests for database operations."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, Extraction, ExtractionStatus


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine)
    return TestSession


@pytest.fixture
def session(test_db):
    """Provide a database session for testing."""
    session = test_db()
    yield session
    session.close()


class TestExtractionModel:
    """Tests for the Extraction model."""

    def test_create_extraction_with_pdf_source(self, session):
        extraction = Extraction(
            pdf_source="/path/to/file.pdf",
            model="gpt-4o",
            status=ExtractionStatus.SUCCESS,
            table1_json={"table1_exists": True, "groups": []},
        )
        session.add(extraction)
        session.commit()

        result = session.query(Extraction).first()
        assert result.pdf_source == "/path/to/file.pdf"
        assert result.status == ExtractionStatus.SUCCESS

    def test_create_failed_extraction(self, session):
        extraction = Extraction(
            pdf_source="/path/to/file.pdf",
            model="gpt-4o",
            status=ExtractionStatus.FAILED,
            error_msg="Extraction failed: no table found",
        )
        session.add(extraction)
        session.commit()

        result = session.query(Extraction).first()
        assert result.status == ExtractionStatus.FAILED
        assert "no table found" in result.error_msg

    def test_filter_by_status(self, session):
        session.add(
            Extraction(
                pdf_source="a.pdf",
                model="gpt-4o",
                status=ExtractionStatus.SUCCESS,
            )
        )
        session.add(
            Extraction(
                pdf_source="b.pdf",
                model="gpt-4o",
                status=ExtractionStatus.FAILED,
            )
        )
        session.commit()

        successes = (
            session.query(Extraction).filter(Extraction.status == ExtractionStatus.SUCCESS).all()
        )
        assert len(successes) == 1
        assert successes[0].pdf_source == "a.pdf"


class TestExtractionStatus:
    """Tests for ExtractionStatus enum."""

    def test_success_value(self):
        assert ExtractionStatus.SUCCESS.value == "success"

    def test_failed_value(self):
        assert ExtractionStatus.FAILED.value == "failed"

    def test_create_from_value(self):
        assert ExtractionStatus("success") == ExtractionStatus.SUCCESS
        assert ExtractionStatus("failed") == ExtractionStatus.FAILED
