"""
providers package
=================
LLM provider abstraction layer for the Investment Committee system.

Public API
----------
from providers import get_provider, LLMProvider, LLMResponse, ProviderError

Quick start
-----------
>>> from providers import get_provider
>>> provider = get_provider("gemini")                         # reads GEMINI_API_KEY
>>> response = await provider.complete("Analyse NVDA", max_tokens=512)
>>> print(response.text, response.tokens_used, response.latency_ms)
"""

from .base import LLMProvider, LLMResponse, ProviderError
from .factory import get_provider
from .gemini import GeminiProvider

__all__ = [
    # Core abstractions
    "LLMProvider",
    "LLMResponse",
    "ProviderError",
    # Factory
    "get_provider",
    # Concrete providers
    "GeminiProvider",
]
