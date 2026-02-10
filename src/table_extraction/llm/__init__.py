"""LLM backends for table extraction."""

from .base import (
    ExtractionRequest,
    ExtractionResponse,
    LLMBackend,
    Message,
    StructuredOutputConfig,
)
from .config import (
    BackendType,
    LLMConfig,
    create_backend,
    get_default_huggingface_config,
    get_default_openai_config,
)

__all__ = [
    # Data types
    "ExtractionRequest",
    "ExtractionResponse",
    "LLMBackend",
    "Message",
    "StructuredOutputConfig",
    # Configuration
    "BackendType",
    "LLMConfig",
    "create_backend",
    "get_default_huggingface_config",
    "get_default_openai_config",
    # Backends (lazy-loaded)
    "HuggingFaceBackend",
    "OpenAIBackend",
]


def __getattr__(name: str):
    """Lazy load backend classes to avoid importing heavy dependencies."""
    if name == "HuggingFaceBackend":
        from .huggingface_backend import HuggingFaceBackend

        return HuggingFaceBackend
    if name == "OpenAIBackend":
        from .openai_backend import OpenAIBackend

        return OpenAIBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
