"""OpenAI backend using the responses.parse API."""

import json
import logging
import os

from openai import OpenAI

from .base import (
    ExtractionRequest,
    ExtractionResponse,
    LLMBackend,
)

logger = logging.getLogger(__name__)


class OpenAIBackend(LLMBackend):
    """OpenAI backend using the responses.parse API for structured output."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
        api_key: str | None = None,
    ):
        """
        Initialize the OpenAI backend.

        Parameters
        ----------
        model : str
            The model to use (default: gpt-5-mini).
        api_key : str, optional
            OpenAI API key. If not provided, will use the OPENAI_API_KEY
            environment variable.
        """
        self._model = model
        self._api_key = api_key
        self._client: OpenAI | None = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> OpenAI:
        """Get or create the OpenAI client (lazy initialization)."""
        if self._client is not None:
            return self._client

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
            )

        self._client = OpenAI(api_key=api_key)
        return self._client

    def _extract_json_from_response(self, response) -> tuple[dict | None, str | None]:
        """Extract JSON data and raw text from the OpenAI response."""
        for item in response.output:
            if item.type == "message":
                for block in item.content:
                    if block.type == "output_text":
                        raw_text = block.text
                        try:
                            return json.loads(block.text), raw_text
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON: {e}")
                            return None, raw_text
        return None, None

    def _parse_response(self, response) -> ExtractionResponse:
        """Parse OpenAI response and handle all post-call error cases."""
        if response.error:
            logger.error(f"LLM error: {response.error}")

        if response.status != "completed":
            return ExtractionResponse(
                json_data=None,
                raw_text=None,
                context=response.id,
                error=getattr(response.incomplete_details, "reason", "unknown"),
                is_complete=False,
                model=self._model,
            )

        json_data, raw_text = self._extract_json_from_response(response)

        if json_data is None:
            error = (
                "Model returned invalid JSON (likely truncated)."
                if raw_text
                else "No JSON found in response."
            )
            return ExtractionResponse(
                json_data=None,
                raw_text=raw_text,
                context=response.id,
                error=error,
                is_complete=raw_text is None,
                model=self._model,
            )

        return ExtractionResponse(
            json_data=json_data,
            raw_text=raw_text,
            context=response.id,
            model=self._model,
            is_complete=True,
        )

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Perform structured extraction using OpenAI's responses.parse API.

        Parameters
        ----------
        request : ExtractionRequest
            The extraction request.

        Returns
        -------
        ExtractionResponse
            The extraction result.
        """
        client = self._get_client()
        input_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        previous_response_id = (
            request.previous_context if isinstance(request.previous_context, str) else None
        )

        try:
            api_kwargs = {
                "model": self._model,
                "previous_response_id": previous_response_id,
                "max_output_tokens": request.max_tokens,
                "input": input_messages,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": request.output_config.schema_name,
                        "schema": request.output_config.schema,
                        "strict": request.output_config.strict,
                    },
                },
            }
            # Only add reasoning parameter for models that support it (gpt-5 series)
            if self._model.startswith("gpt-5"):
                api_kwargs["reasoning"] = {"effort": "minimal"}

            response = client.responses.parse(**api_kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return ExtractionResponse(
                json_data=None,
                raw_text=None,
                error=str(e),
                is_complete=False,
                model=self._model,
            )

        return self._parse_response(response)
