"""
providers/anthropic.py
======================
Concrete LLM provider backed by Anthropic's Claude SDK.

Environment variables
---------------------
ANTHROPIC_API_KEY   – required; your Anthropic API key.
ANTHROPIC_MODEL     – optional; defaults to claude-3-5-haiku-20241022.

Usage
-----
>>> from providers.anthropic import ClaudeProvider
>>> provider = ClaudeProvider()
>>> response = await provider.complete("Analyse battery tech", max_tokens=1024)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Optional, Type

import anthropic

from .base import LLMProvider, LLMResponse, ProviderError
from config import ANTHROPIC_API_KEY as _ENV_API_KEY, ANTHROPIC_MODEL as _ENV_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = _ENV_MODEL or "claude-3-5-haiku-20241022"


class ClaudeProvider(LLMProvider):
    """LLM provider backed by Anthropic's Claude SDK.

    Parameters
    ----------
    model:
        Claude model name.  Defaults to ``claude-3-5-haiku-20241022``.
    api_key:
        If omitted the constructor reads ``ANTHROPIC_API_KEY`` from the
        environment and raises :class:`ProviderError` when it is absent.
    synthesis_model:
        Optional higher-quality model to use when ``use_synthesis_model=True``
        is passed to complete(). Useful for the final synthesis step.
    """

    provider_name = "anthropic"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        synthesis_model: Optional[str] = None,
    ) -> None:
        from config import ANTHROPIC_SYNTHESIS_MODEL
        resolved_key = api_key or _ENV_API_KEY
        if not resolved_key:
            raise ProviderError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file.",
                provider=self.provider_name,
            )

        self.model_name = model
        self.synthesis_model_name = synthesis_model or ANTHROPIC_SYNTHESIS_MODEL or model
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)

        logger.debug(
            "[claude] Initialised ClaudeProvider with model=%s synthesis_model=%s",
            model, self.synthesis_model_name,
        )

    # ------------------------------------------------------------------
    # complete()
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        max_tokens: int,
        system: Optional[str] = None,
        temperature: float = 0.7,
        response_schema: Optional[Any] = None,
        use_synthesis_model: bool = False,
    ) -> LLMResponse:
        """Send *prompt* and return a fully-buffered :class:`LLMResponse`.

        If ``response_schema`` is provided, the prompt is augmented to
        request JSON output matching that schema (Claude does not have
        native JSON mode equivalent to Gemini, so we inject schema
        description into the prompt and parse output).
        """
        model = self.synthesis_model_name if use_synthesis_model else self.model_name

        # Build messages
        user_content = prompt
        if response_schema and hasattr(response_schema, "model_json_schema"):
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            user_content = (
                f"{prompt}\n\n"
                f"IMPORTANT: Respond ONLY with a valid JSON object matching this schema "
                f"(no markdown fences, no preamble, no explanation outside JSON):\n"
                f"{schema_json}"
            )

        messages = [{"role": "user", "content": user_content}]

        # Build kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        last_exc: Optional[Exception] = None

        for attempt in range(3):  # up to 2 retries on transient errors
            try:
                response = await self._client.messages.create(**kwargs)
                last_exc = None
                break
            except anthropic.RateLimitError as exc:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s back-off on rate limits
                logger.warning(
                    "[claude] rate limit attempt %d — waiting %ds: %s",
                    attempt + 1, wait, exc,
                )
                if attempt < 2:
                    await asyncio.sleep(wait)
                    last_exc = exc
                    continue
                raise ProviderError(
                    f"Claude rate limited after {attempt + 1} attempts: {exc}",
                    provider=self.provider_name,
                    original=exc,
                ) from exc
            except (anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                    last_exc = exc
                    logger.warning("[claude] transient error attempt %d: %s — retrying", attempt + 1, exc)
                    continue
                raise ProviderError(
                    f"Claude connection failed after retries: {exc}",
                    provider=self.provider_name,
                    original=exc,
                ) from exc
            except anthropic.APIError as exc:
                raise ProviderError(
                    f"Claude API error: {exc}",
                    provider=self.provider_name,
                    original=exc,
                ) from exc

        if last_exc:
            raise ProviderError(
                f"Claude complete() failed after retries: {last_exc}",
                provider=self.provider_name,
                original=last_exc,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Extract text
        try:
            text = response.content[0].text
        except (IndexError, AttributeError) as exc:
            raise ProviderError(
                f"Claude response was empty or blocked: {exc}",
                provider=self.provider_name,
                original=exc,
            ) from exc

        # Extract token counts
        tokens_used = 0
        if response.usage:
            tokens_used = (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)

        result = LLMResponse(
            text=text,
            tokens_used=tokens_used,
            model=model,
            latency_ms=round(latency_ms, 2),
        )

        self._log_call(
            model=model,
            tokens=tokens_used,
            latency_ms=latency_ms,
            prompt=prompt,
            call_type="complete",
        )

        return result

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        max_tokens: int,
        system: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks from a streaming Claude response."""
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        tokens_used = 0

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text_chunk in stream.text_stream:
                    if text_chunk:
                        yield text_chunk

                # Capture final usage from the final message
                final_message = await stream.get_final_message()
                if final_message.usage:
                    tokens_used = (
                        (final_message.usage.input_tokens or 0)
                        + (final_message.usage.output_tokens or 0)
                    )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Claude stream() failed: {exc}",
                provider=self.provider_name,
                original=exc,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        self._log_call(
            model=self.model_name,
            tokens=tokens_used,
            latency_ms=latency_ms,
            prompt=prompt,
            call_type="stream",
        )
