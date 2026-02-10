"""Configuration and factory for LLM backends."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .base import LLMBackend

# Default model names (can be overridden via environment variables)
DEFAULT_OPENAI_MODEL = os.getenv("RCT_CHECKER_OPENAI_MODEL", "gpt-5-mini")
DEFAULT_HUGGINGFACE_MODEL = os.getenv("RCT_CHECKER_HUGGINGFACE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


class BackendType(Enum):
    """Supported LLM backend types."""

    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""

    backend: BackendType
    model: str
    max_output_tokens: int
    max_attempts: int
    # Backend-specific options
    # OpenAI: api_key
    # HuggingFace: device, load_in_4bit
    options: dict[str, Any]


def create_backend(config: LLMConfig) -> LLMBackend:
    """
    Factory function to create an LLM backend from configuration.

    Parameters
    ----------
    config : LLMConfig
        The backend configuration.

    Returns
    -------
    LLMBackend
        Configured backend instance.

    Raises
    ------
    ValueError
        If the backend type is not supported.
    NotImplementedError
        If the backend type is not yet implemented.
    """
    if config.backend == BackendType.OPENAI:
        from .openai_backend import OpenAIBackend

        return OpenAIBackend(
            model=config.model,
            api_key=config.options.get("api_key"),
        )

    elif config.backend == BackendType.HUGGINGFACE:
        from .huggingface_backend import HuggingFaceBackend

        return HuggingFaceBackend(
            model_id=config.model,
            device=config.options.get("device"),
            load_in_4bit=config.options.get("load_in_4bit", False),
        )

    else:
        raise ValueError(f"Unknown backend type: {config.backend}")


def get_default_openai_config() -> LLMConfig:
    """
    Get the default OpenAI configuration.

    The model can be overridden via the RCT_CHECKER_OPENAI_MODEL environment variable.
    """
    return LLMConfig(
        backend=BackendType.OPENAI,
        model=DEFAULT_OPENAI_MODEL,
        max_output_tokens=20000,
        max_attempts=5,
        options={},
    )


def get_default_huggingface_config() -> LLMConfig:
    """
    Get the default HuggingFace configuration.

    The model can be overridden via the RCT_CHECKER_HUGGINGFACE_MODEL environment variable.
    """
    return LLMConfig(
        backend=BackendType.HUGGINGFACE,
        model=DEFAULT_HUGGINGFACE_MODEL,
        max_output_tokens=5000,
        max_attempts=5,
        options={
            "device": None,
            "load_in_4bit": False,  # Requires bitsandbytes + CUDA
        },
    )
