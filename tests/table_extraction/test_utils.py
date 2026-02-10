"""Unit tests for src.table_extraction.utils."""

import pathlib
import sys

import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from src.table_extraction.utils import (
    _to_numeric,
    extract_table_text,
    is_url,
    to_csv_wide,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a sample PDF with Table 1 on page 2."""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Intro page without table")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Table 1\nBaseline characteristics\nGroup A 10\nGroup B 12")

    page3 = doc.new_page()
    page3.insert_text((72, 72), "Follow-up page")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def pdf_without_table1(tmp_path):
    """Create a PDF without Table 1."""
    pdf_path = tmp_path / "no_table.pdf"
    doc = fitz.open()

    page1 = doc.new_page()
    page1.insert_text((72, 72), "Introduction")

    page2 = doc.new_page()
    page2.insert_text((72, 72), "Methods section")

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sample_json():
    """Sample JSON data matching the actual extraction schema."""
    return {
        "table1_exists": True,
        "groups": [
            {"group_id": "group_1", "label": "Treatment", "sample_size": 50},
            {"group_id": "group_2", "label": "Control", "sample_size": 55},
        ],
        "rows": [
            {
                "variable": "Age",
                "variable_type": "Continuous",
                "level": None,
                "values": [
                    {
                        "group_id": "group_1",
                        "original": "60.1 (10.5)",
                        "mean": 60.1,
                        "median": None,
                        "count": None,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": 10.5,
                        "pvalue": None,
                    },
                    {
                        "group_id": "group_2",
                        "original": "58.2 (9.0)",
                        "mean": 58.2,
                        "median": None,
                        "count": None,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": 9.0,
                        "pvalue": 0.05,
                    },
                ],
            },
            {
                "variable": "Sex",
                "variable_type": "Categorical",
                "level": "Male",
                "values": [
                    {
                        "group_id": "group_1",
                        "original": "30 (60%)",
                        "mean": None,
                        "median": None,
                        "count": 30,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": None,
                        "pvalue": None,
                    },
                    {
                        "group_id": "group_2",
                        "original": "28 (51%)",
                        "mean": None,
                        "median": None,
                        "count": 28,
                        "IQR_lower": None,
                        "IQR_upper": None,
                        "95CI_lower": None,
                        "95CI_upper": None,
                        "sd": None,
                        "pvalue": 0.12,
                    },
                ],
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests for is_url
# ─────────────────────────────────────────────────────────────────────────────


class TestIsUrl:
    def test_http_url(self):
        assert is_url("http://example.com/paper.pdf") is True

    def test_https_url(self):
        assert is_url("https://example.com/paper.pdf") is True

    def test_local_path(self):
        assert is_url("/path/to/file.pdf") is False

    def test_relative_path(self):
        assert is_url("data/file.pdf") is False

    def test_windows_path(self):
        assert is_url("C:\\Users\\file.pdf") is False

    def test_ftp_url(self):
        # ftp is not supported
        assert is_url("ftp://example.com/file.pdf") is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests for _to_numeric
# ─────────────────────────────────────────────────────────────────────────────


class TestToNumeric:
    def test_none_returns_none(self):
        assert _to_numeric(None) is None

    def test_float_value(self):
        assert _to_numeric(3.14) == 3.14

    def test_int_value(self):
        assert _to_numeric(42) == 42.0

    def test_as_int_converts_to_int(self):
        assert _to_numeric(3.7, as_int=True) == 3
        assert isinstance(_to_numeric(3.7, as_int=True), int)

    def test_none_with_as_int(self):
        assert _to_numeric(None, as_int=True) is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests for extract_table_text
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTableText:
    def test_extracts_table1_with_surrounding_pages(self, sample_pdf):
        text = extract_table_text(str(sample_pdf), expand_pages=1)
        assert "Table 1" in text
        assert "Baseline characteristics" in text
        # With expand_pages=1, should include pages before and after
        assert "Intro page without table" in text
        assert "Follow-up page" in text

    def test_expand_pages_zero_only_table_page(self, sample_pdf):
        text = extract_table_text(str(sample_pdf), expand_pages=0)
        assert "Table 1" in text
        # First page is always included
        assert "Intro page without table" in text
        # But page 3 should not be included
        assert "Follow-up page" not in text

    def test_fallback_when_no_table1(self, pdf_without_table1):
        text = extract_table_text(str(pdf_without_table1))
        # Should return all text as fallback
        assert "Introduction" in text
        assert "Methods section" in text


# ─────────────────────────────────────────────────────────────────────────────
# Tests for to_csv_wide
# ─────────────────────────────────────────────────────────────────────────────


class TestToCsvWide:
    def test_writes_expected_columns(self, tmp_path, sample_json):
        out_path = tmp_path / "wide.csv"
        to_csv_wide(sample_json, str(out_path))

        df = pd.read_csv(out_path)
        # Check key columns exist
        assert "Variable" in df.columns
        assert "Variable type" in df.columns
        assert "group_1 (original)" in df.columns
        assert "group_1 (mean)" in df.columns
        assert "group_1 (sd)" in df.columns
        assert "group_2 (original)" in df.columns
        assert "group_2 (pvalue)" in df.columns

    def test_continuous_variable_values(self, tmp_path, sample_json):
        out_path = tmp_path / "wide.csv"
        to_csv_wide(sample_json, str(out_path))

        df = pd.read_csv(out_path)
        age_row = df[df["Variable"] == "Age"]
        assert len(age_row) == 1
        assert age_row.iloc[0]["group_1 (mean)"] == 60.1
        assert age_row.iloc[0]["group_1 (sd)"] == 10.5
        assert age_row.iloc[0]["group_2 (mean)"] == 58.2

    def test_categorical_variable_values(self, tmp_path, sample_json):
        out_path = tmp_path / "wide.csv"
        to_csv_wide(sample_json, str(out_path))

        df = pd.read_csv(out_path)
        # Categorical variables include level in name
        sex_row = df[df["Variable"] == "Sex (Male)"]
        assert len(sex_row) == 1
        assert sex_row.iloc[0]["group_1 (count)"] == 30
        assert sex_row.iloc[0]["group_2 (count)"] == 28

    def test_returns_dataframe_when_no_path(self, sample_json):
        df = to_csv_wide(sample_json, out_path=None)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
