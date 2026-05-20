"""
providers/base.py
=================
Abstract contract for every LLM provider in the investment-committee system.

Public surface
--------------
LLMResponse     – dataclass returned by complete()
ProviderError   – custom exception raised on any provider failure
LLMProvider     – ABC with complete() and stream() that all providers implement
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Structured response returned by :meth:`LLMProvider.complete`."""

    text: str
    tokens_used: int
    model: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """Raised when an LLM provider call fails for any reason.

    Attributes
    ----------
    provider:
        Name of the provider that raised the error (e.g. ``"gemini"``).
    original:
        The underlying exception, if any.
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        original: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.original = original

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProviderError(provider={self.provider!r}, "
            f"message={str(self)!r}, original={self.original!r})"
        )


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base class that every concrete LLM provider must implement.

    Sub-classes must override :meth:`complete` and :meth:`stream`.
    They should also set :attr:`provider_name` to a short identifier used
    in log lines and :class:`ProviderError` payloads.
    """

    #: Short identifier for this provider, e.g. ``"gemini"`` or ``"anthropic"``.
    provider_name: str = "base"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        system: Optional[str] = None,
        temperature: float = 0.7,
        response_schema: Optional[Any] = None,
    ) -> LLMResponse:
        """Send a single prompt and await a complete response.

        Parameters
        ----------
        prompt:
            The user-turn message to send to the model.
        max_tokens:
            Hard ceiling on the number of tokens the model may generate.
        system:
            Optional system-level instruction prepended to the conversation.
        temperature:
            Sampling temperature (0 = deterministic, 1 = creative).
        response_schema:
            Optional Pydantic model class to force JSON output structure.

        Returns
        -------
        LLMResponse

        Raises
        ------
        ProviderError
            On any API failure, authentication error, or unexpected response
            shape.
        """

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token as an async generator.

        Parameters
        ----------
        prompt:
            The user-turn message to send to the model.
        max_tokens:
            Hard ceiling on the number of tokens the model may generate.
        system:
            Optional system-level instruction prepended to the conversation.

        Yields
        ------
        str
            Successive text chunks as they arrive from the provider.

        Raises
        ------
        ProviderError
            On any API failure before or during streaming.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _truncate_for_log(self, text: str, max_chars: int = 120) -> str:
        """Return a safe, single-line truncated version of *text* for logging."""
        single = text.replace("\n", " ").replace("\r", "")
        if len(single) <= max_chars:
            return single
        return single[:max_chars] + "…"

    def _log_call(
        self,
        *,
        model: str,
        tokens: int,
        latency_ms: float,
        prompt: str,
        call_type: str = "complete",
    ) -> None:
        """Emit a structured INFO log line for every provider call."""
        logger.info(
            "[%s] %s | model=%s tokens=%d latency=%.1fms | prompt=%r",
            self.provider_name,
            call_type,
            model,
            tokens,
            latency_ms,
            self._truncate_for_log(prompt),
        )
