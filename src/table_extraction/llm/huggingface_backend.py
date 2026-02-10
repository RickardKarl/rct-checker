"""HuggingFace backend using outlines for structured JSON generation."""

import json
import logging
from dataclasses import dataclass
from typing import Any

import torch

torch._dynamo.config.disable = True  # Runs faster without
import outlines  # noqa: E402
from outlines.types import JsonSchema  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from .base import (  # noqa: E402
    ExtractionRequest,
    ExtractionResponse,
    LLMBackend,
    Message,
)


@dataclass
class HuggingFaceContext:
    """Context for chaining HuggingFace requests, storing conversation history."""

    messages: list[Message]  # Full conversation including assistant response


logger = logging.getLogger(__name__)


class HuggingFaceBackend(LLMBackend):
    """
    Local LLM backend using HuggingFace transformers + outlines for structured output.

    Outlines uses grammar-constrained generation to guarantee valid JSON output
    matching the provided schema - no retries needed for JSON validity.

    Requirements:
        pip install outlines transformers torch
        Optional for quantization: pip install bitsandbytes
    """

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        load_in_4bit: bool = False,
    ):
        """
        Initialize the HuggingFace backend.

        Parameters
        ----------
        model_id : str
            HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        device : str, optional
            Device to load model on ("cuda", "cpu", or None for auto).
        load_in_4bit : bool
            Whether to use 4-bit quantization for lower memory usage.
        """
        self._model_id = model_id
        self._device = device
        self._load_in_4bit = load_in_4bit
        self._model: Any = None  # Lazy load
        self._tokenizer: Any = None

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model(self) -> str:
        return self._model_id

    def _load_model(self) -> None:
        if self._model is not None:
            return

        logger.info(f"Loading model {self._model_id}...")

        # Load tokenizer first
        tokenizer = self._load_tokenizer()

        # Load the HuggingFace model
        model_kwargs = {}
        if self._device:
            model_kwargs["device_map"] = self._device
        if self._load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        hf_model = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kwargs)

        # Compile model for faster inference
        # hf_model = torch.compile(hf_model)

        # Wrap with outlines Transformers
        self._model = outlines.Transformers(hf_model, tokenizer)

        logger.info(f"Model {self._model_id} loaded successfully")

    def _load_tokenizer(self) -> Any:
        """Load the tokenizer for chat template formatting."""
        if self._tokenizer is not None:
            return self._tokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        return self._tokenizer

    def _build_prompt(self, messages: list[Message]) -> str:
        """
        Convert messages to a prompt string using the model's chat template.

        Parameters
        ----------
        messages : list[Message]
            The messages to format.

        Returns
        -------
        str
            The formatted prompt string.
        """
        tokenizer = self._load_tokenizer()

        # Format as chat messages for the tokenizer
        chat = [{"role": m.role, "content": m.content} for m in messages]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def extract(self, request: ExtractionRequest) -> ExtractionResponse:
        """
        Perform structured extraction using outlines constrained generation.

        Parameters
        ----------
        request : ExtractionRequest
            The extraction request.

        Returns
        -------
        ExtractionResponse
            The extraction result.
        """
        self._load_model()

        # Build conversation including previous context if available
        if request.previous_context is not None and isinstance(
            request.previous_context, HuggingFaceContext
        ):
            # Include previous conversation history, then append new user message
            messages = request.previous_context.messages + [request.messages[-1]]
        else:
            messages = request.messages

        # Build prompt from messages using chat template
        prompt = self._build_prompt(messages)

        # Create JSON generator with schema constraint
        generator = outlines.Generator(
            self._model,
            output_type=JsonSchema(request.output_config.schema),
        )

        try:
            # Generate - outlines guarantees valid JSON matching schema
            logger.info("Generating structured output...")
            result = generator(prompt, max_new_tokens=request.max_tokens)

            logger.debug(f"Result (type {type(result)}) = {result}")

            if isinstance(result, str):
                json_result = json.loads(result)
            else:
                raise ValueError(f"Output from LLM is not a string, but instead {type(result)}")

            # Store conversation history for potential chaining
            context = HuggingFaceContext(
                messages=messages + [Message(role="assistant", content=result)]
            )

            return ExtractionResponse(
                json_data=json_result,
                raw_text=result,
                context=context,
                model=self._model_id,
                is_complete=True,
            )
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ExtractionResponse(
                json_data=None,
                raw_text=None,
                error=str(e),
                is_complete=False,
                model=self._model_id,
            )
