"""Unit tests for the LLM module."""

import pathlib
import sys
from unittest.mock import Mock, patch

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.table_extraction.llm import (
    BackendType,
    ExtractionRequest,
    ExtractionResponse,
    LLMConfig,
    Message,
    OpenAIBackend,
    StructuredOutputConfig,
    create_backend,
)


class TestDataClasses:
    """Tests for LLM data classes."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_structured_output_config_defaults(self):
        config = StructuredOutputConfig(schema={"type": "object"})
        assert config.schema == {"type": "object"}
        assert config.schema_name == "extraction"
        assert config.strict is True

    def test_extraction_request_defaults(self):
        request = ExtractionRequest(
            messages=[Message(role="user", content="test")],
            output_config=StructuredOutputConfig(schema={}),
        )
        assert request.max_tokens == 10000
        assert request.previous_context is None

    def test_extraction_response_defaults(self):
        response = ExtractionResponse(json_data={"key": "value"}, raw_text=None)
        assert response.json_data == {"key": "value"}
        assert response.is_complete is True
        assert response.error is None


class TestCreateBackend:
    """Tests for the backend factory."""

    def test_create_openai_backend(self):
        config = LLMConfig(
            backend=BackendType.OPENAI,
            model="gpt-4",
            max_output_tokens=10000,
            max_attempts=5,
            options={},
        )
        backend = create_backend(config)
        assert isinstance(backend, OpenAIBackend)
        assert backend.name == "openai"
        assert backend.model == "gpt-4"

    def test_create_huggingface_backend(self):
        pytest.importorskip("torch", reason="torch not installed")
        from src.table_extraction.llm import HuggingFaceBackend

        config = LLMConfig(
            backend=BackendType.HUGGINGFACE,
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_output_tokens=10000,
            max_attempts=5,
            options={"device": "cpu", "load_in_4bit": False},
        )
        backend = create_backend(config)
        assert isinstance(backend, HuggingFaceBackend)
        assert backend.name == "huggingface"
        assert backend.model == "meta-llama/Llama-3.1-8B-Instruct"

    def test_invalid_backend_raises(self):
        config = LLMConfig(
            backend="invalid",  # type: ignore
            model="test",
            max_output_tokens=10000,
            max_attempts=5,
            options={},
        )
        with pytest.raises(ValueError):
            create_backend(config)


class TestOpenAIBackend:
    """Tests for OpenAIBackend."""

    def test_properties(self):
        backend = OpenAIBackend(model="gpt-5-mini")
        assert backend.name == "openai"
        assert backend.model == "gpt-5-mini"

    def test_missing_api_key_raises(self):
        backend = OpenAIBackend()
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                backend._get_client()

    @patch("src.table_extraction.llm.openai_backend.OpenAI")
    def test_extract_success(self, mock_openai_class):
        # Setup mock response
        mock_response = Mock()
        mock_response.status = "completed"
        mock_response.id = "resp_123"
        mock_response.error = None
        mock_response.output = [
            Mock(
                type="message",
                content=[Mock(type="output_text", text='{"result": "success"}')],
            )
        ]

        mock_client = Mock()
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        backend = OpenAIBackend(api_key="test-key")
        request = ExtractionRequest(
            messages=[Message(role="user", content="test")],
            output_config=StructuredOutputConfig(schema={}),
        )

        response = backend.extract(request)

        assert response.is_complete is True
        assert response.json_data == {"result": "success"}
        assert response.context == "resp_123"

    @patch("src.table_extraction.llm.openai_backend.OpenAI")
    def test_extract_incomplete_response(self, mock_openai_class):
        mock_response = Mock()
        mock_response.status = "incomplete"
        mock_response.id = "resp_123"
        mock_response.error = None
        mock_response.incomplete_details = Mock(reason="max_tokens")

        mock_client = Mock()
        mock_client.responses.parse.return_value = mock_response
        mock_openai_class.return_value = mock_client

        backend = OpenAIBackend(api_key="test-key")
        request = ExtractionRequest(
            messages=[Message(role="user", content="test")],
            output_config=StructuredOutputConfig(schema={}),
        )

        response = backend.extract(request)

        assert response.is_complete is False
        assert response.error == "max_tokens"


class TestHuggingFaceBackend:
    """Tests for HuggingFaceBackend."""

    def test_properties(self):
        pytest.importorskip("torch", reason="torch not installed")
        from src.table_extraction.llm import HuggingFaceBackend

        backend = HuggingFaceBackend(model_id="test-model")
        assert backend.name == "huggingface"
        assert backend.model == "test-model"

    def test_lazy_loading(self):
        pytest.importorskip("torch", reason="torch not installed")
        from src.table_extraction.llm import HuggingFaceBackend

        backend = HuggingFaceBackend(model_id="test-model")
        # Model should not be loaded until needed
        assert backend._model is None
        assert backend._tokenizer is None
