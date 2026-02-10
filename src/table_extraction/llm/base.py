"""Abstract base class and data types for LLM backends."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.table_extraction.validate_output import ValidationError

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message in a conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class StructuredOutputConfig:
    """Configuration for structured JSON output."""

    schema: dict[str, Any]
    schema_name: str = "extraction"
    strict: bool = True


@dataclass
class ExtractionRequest:
    """Request for structured extraction from an LLM."""

    messages: list[Message]
    output_config: StructuredOutputConfig
    max_tokens: int = 10000
    previous_context: Any = None  # Backend-specific (e.g., OpenAI response ID)


@dataclass
class ExtractionResponse:
    """Response from an LLM extraction call."""

    json_data: dict[str, Any] | None
    raw_text: str | None
    context: Any = None  # For chaining requests
    model: str = ""
    error: str | None = None
    is_complete: bool = True


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'openai', 'huggingface')."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model identifier being used."""
        pass

    @abstractmethod
    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Perform structured extraction.

        Parameters
        ----------
        request : ExtractionRequest
            The extraction request with messages, schema, and optional context.

        Returns
        -------
        ExtractionResponse
            The extraction result with JSON data (if successful) and context
            for potential follow-up requests.
        """
        pass

    def extract_with_repair(
        self,
        initial_request: ExtractionRequest,
        repair_prompt_template: str,
        validate_fn: Callable[[dict[str, Any]], None],
        max_attempts: int = 5,
    ) -> dict[str, Any]:
        """
        Extract with automatic repair loop.

        This is a template method that handles the repair loop logic.
        Subclasses can override for backend-specific optimizations.

        Parameters
        ----------
        initial_request : ExtractionRequest
            The initial extraction request.
        repair_prompt_template : str
            Template with {ERROR_MESSAGES} placeholder.
        validate_fn : callable
            Function that raises ValidationError if JSON is invalid.
        max_attempts : int
            Maximum extraction attempts.

        Returns
        -------
        dict
            Validated JSON data.

        Raises
        ------
        RuntimeError
            If extraction fails after max_attempts.
        """
        response: ExtractionResponse | None = None
        error_message: str | None = None

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                logger.info(f"Attempt {attempt}/{max_attempts}: Initial extraction")
                request = initial_request
            else:
                logger.info(f"Attempt {attempt}/{max_attempts}: Repair extraction")
                logger.debug(f"Previous error: {error_message}")
                # Build repair request
                repair_content = repair_prompt_template.replace(
                    "{ERROR_MESSAGES}", error_message or ""
                )
                request = ExtractionRequest(
                    messages=[
                        initial_request.messages[0],  # Keep system message
                        Message(role="user", content=repair_content),
                    ],
                    output_config=initial_request.output_config,
                    max_tokens=initial_request.max_tokens,
                    previous_context=response.context if response else None,
                )

            response = self.extract(request)

            if not response.is_complete:
                logger.error(f"Incomplete response from LLM: {response.error}")
                raise RuntimeError(f"Incomplete response: {response.error}")

            if response.json_data is None:
                error_message = response.error or "No JSON returned"
                logger.warning(f"No JSON data returned: {error_message}")
                continue

            try:
                validate_fn(response.json_data)
                logger.info(f"Validation passed on attempt {attempt}")
                return response.json_data
            except ValidationError as e:
                error_message = str(e)
                logger.warning(f"Validation failed: {error_message}\nModel output = {response}")

        logger.error(f"Failed to produce valid output after {max_attempts} attempts")
        raise RuntimeError(f"Failed to produce valid output after {max_attempts} attempts.")
