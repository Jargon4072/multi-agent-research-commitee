"""
providers/gemini.py
===================
Concrete LLM provider backed by Google's **new** Generative AI SDK
(``google-genai``).

Environment variables
---------------------
GEMINI_API_KEY   – required; your Google AI Studio / Vertex API key.

Usage
-----
>>> from providers.gemini import GeminiProvider
>>> provider = GeminiProvider()                         # uses default model
>>> provider = GeminiProvider(model="gemini-2.0-flash")
>>> response = await provider.complete("Analyse NVDA", max_tokens=1024)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import AsyncIterator, Optional

from google import genai
from google.genai import types as genai_types

from .base import LLMProvider, LLMResponse, ProviderError
from config import GEMINI_API_KEY as _ENV_API_KEY, GEMINI_MODEL as _ENV_MODEL

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = _ENV_MODEL or "gemini-2.0-flash"


class GeminiProvider(LLMProvider):
    """LLM provider backed by ``google-genai`` (the current Gemini SDK).

    Parameters
    ----------
    model:
        Gemini model name.  Defaults to ``"gemini-2.0-flash"``.
    api_key:
        If omitted the constructor reads ``GEMINI_API_KEY`` from the
        environment and raises :class:`ProviderError` when it is absent.
    """

    provider_name = "gemini"

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        # Priority: explicit arg > env var from config.py (.env) > os.environ
        resolved_key = api_key or _ENV_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ProviderError(
                "GEMINI_API_KEY is not set. Add it to your .env file.",
                provider=self.provider_name,
            )

        self.model_name = model
        # The new SDK uses a client object instead of module-level configure()
        self._client = genai.Client(api_key=resolved_key)

        logger.debug(
            "[gemini] Initialised GeminiProvider with model=%s", model
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
    ) -> LLMResponse:
        """Send *prompt* and return a fully-buffered :class:`LLMResponse`."""
        # If response_schema is a Pydantic model, convert to dict and strip
        # additionalProperties which Gemini explicitly forbids.
        final_schema = response_schema
        if response_schema and hasattr(response_schema, "model_json_schema"):
            final_schema = response_schema.model_json_schema()
            def _strip(obj):
                if isinstance(obj, dict):
                    obj.pop("additionalProperties", None)
                    for v in obj.values(): _strip(v)
                elif isinstance(obj, list):
                    for v in obj: _strip(v)
            _strip(final_schema)

        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system,  # None is a no-op
            response_schema=final_schema,
            response_mime_type="application/json" if response_schema else None,
        )


        t0 = time.perf_counter()
        last_exc: Optional[Exception] = None
        for attempt in range(2):          # 1 retry on transient connection errors
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                last_exc = None
                break
            except Exception as exc:
                # Re-raise immediately on non-transient errors
                exc_str = str(exc)
                is_transient = any(
                    kw in exc_str.lower()
                    for kw in ("dns", "connect", "timeout", "network", "temporary")
                )
                if not is_transient or attempt == 1:
                    raise ProviderError(
                        f"Gemini complete() failed: {exc}",
                        provider=self.provider_name,
                        original=exc,
                    ) from exc
                last_exc = exc
                logger.warning(
                    "[gemini] transient error attempt %d: %s — retrying", attempt + 1, exc
                )
                await asyncio.sleep(2 ** attempt)   # 1s, 2s back-off

        if last_exc:
            raise ProviderError(
                f"Gemini complete() failed after retries: {last_exc}",
                provider=self.provider_name,
                original=last_exc,
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # ── extract text ──────────────────────────────────────────────
        try:
            text = response.text
        except (ValueError, AttributeError) as exc:
            raise ProviderError(
                f"Gemini response was blocked or empty: {exc}",
                provider=self.provider_name,
                original=exc,
            ) from exc

        # ── extract token counts ───────────────────────────────────────
        tokens_used = 0
        if response.usage_metadata:
            tokens_used = response.usage_metadata.total_token_count or 0

        result = LLMResponse(
            text=text,
            tokens_used=tokens_used,
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
        )

        self._log_call(
            model=self.model_name,
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
        """Yield text chunks from a streaming Gemini response."""
        config = genai_types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system,
        )

        t0 = time.perf_counter()
        tokens_used = 0

        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config,
            ):
                # Extract text — some chunks carry only usage metadata
                chunk_text: str = ""
                try:
                    chunk_text = chunk.text or ""
                except (ValueError, AttributeError):
                    pass

                if chunk_text:
                    yield chunk_text

                # Capture token count from the last chunk that carries it
                if chunk.usage_metadata:
                    tokens_used = chunk.usage_metadata.total_token_count or tokens_used

        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"Gemini stream() failed: {exc}",
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
